#!/usr/bin/env bash
# vllm-bench-vram.sh — benchmark a live vLLM server and track its ACTUAL VRAM
# consumption (the VLLM::EngineCore process GPU memory, as shown by nvitop).
#
# DESIGN (avoids measurement side-effects):
#   * The `vllm bench serve` CLIENT runs on the mac-studio (default: a throwaway
#     vllm/vllm-openai container there), hitting the vLLM endpoint on the
#     dgx-spark over the network. It does NOT run inside the vLLM serving
#     container, so its own memory can't perturb the measurement.
#   * The VRAM signal is the VLLM::EngineCore process's GPU memory, read via
#     nvitop (NVML) on the dgx-spark — the same number nvitop's UI shows. This
#     is the authoritative "what vLLM consumes from VRAM" figure (on GB10 the KV
#     pool lives in HBM and is invisible to /proc/meminfo, cgroup, docker stats).
#   * nvitop sampling runs on the dgx-spark over SSH in parallel with the bench.
#
# HELPER SCRIPTS (in this directory — no embedded Python):
#   vram-reader.py        — runs ON THE DGX-SPARK; prints a process's GPU memory (GiB)
#   bench-cpu-wrapper.py  — runs in the bench-client container (mac-studio); forces
#                           vLLM's CPU platform so the no-GPU client can parse args
#   merge-vram.py         — runs on the mac-studio; merges the VRAM trace into the bench JSON
#
# NOTE: do NOT force `--platform linux/amd64` on the mac-studio client container —
# the vllm image is multi-arch with a real linux/arm64 build and the Mac Docker
# daemon is aarch64, so the default runs natively (amd64 = Rosetta/QEMU, slower
# for no benefit).
#
# REQUIREMENTS:
#   * Run from the mac-studio (which can reach the vLLM endpoint on the dgx-spark
#     over the LAN).
#   * A `vllm bench serve` CLIENT. Three modes (pick via --client):
#       - mac-container (DEFAULT, cleanest): throwaway vllm/vllm-openai container
#         ON THE MAC-STUDIO. Needs Docker + the image pulled here (linux/arm64,
#         native on Apple Silicon).
#       - local: runs `vllm bench` directly on the mac-studio (needs vllm CLI).
#       - dgx-container: throwaway container ON THE DGX-SPARK (fallback if no
#         Mac Docker). Does not persist the bench JSON (container is --rm).
#   * SSH access to the dgx-spark (SSH_HOST, default dgx-spark) with nvitop
#     installed. The repo is synced to the dgx-spark via Unison (no scp needed).
#
# USAGE (from the mac-studio):
#   bash vllm-bench-vram.sh                          # defaults (mac-container client)
#   bash vllm-bench-vram.sh --client local           # if vllm CLI is on the mac-studio
#   bash vllm-bench-vram.sh --client dgx-container   # fallback: client on the dgx-spark
#   bash vllm-bench-vram.sh --input-len 32000 --num-prompts 20
#   bash vllm-bench-vram.sh --ramp-start 2 --ramp-end 16 --max-concurrency 4
#   bash vllm-bench-vram.sh --ssh-host dgx-spark --proc-name "VLLM::EngineCore"
#
# Args are forwarded to `vllm bench serve` EXCEPT the client/sampling ones
# (listed below). See `vllm bench serve --help=all` for the full set.

set -uo pipefail

# ── Paths to the helper scripts (same dir as this script) ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRAM_READER="$SCRIPT_DIR/vram-reader.py"
CPU_WRAPPER="$SCRIPT_DIR/bench-cpu-wrapper.py"
MERGE_VRAM="$SCRIPT_DIR/merge-vram.py"

# ── Defaults ─────────────────────────────────────────────────────────────────
SERVER="http://$DGX_HOST:8888 (DGX Spark IP — see project .env / docs/infra-context.md)"     # vLLM base URL (host:port), reachable from client
SSH_HOST="dgx-spark"                    # SSH alias for the dgx-spark (nvitop sampling)
PROC_NAME="VLLM::EngineCore"            # process to track (nvitop name)
REMOTE_READER=""                        # dgx-spark path to vram-reader.py (default: synced repo path)
CLIENT="mac-container"                  # mac-container | local | dgx-container
BENCH_IMAGE="vllm/vllm-openai:v0.27.1"  # image for the throwaway bench-client container
MODEL=""                                # empty -> vllm bench auto-detects from server
API_ENDPOINT="/v1/chat/completions"     # vLLM API path (openai-chat backend)
OUT=""                                  # empty -> ./bench-vram-<ts>.json
SAMPLE_INTERVAL=3                       # seconds between nvitop VRAM samples
RAMP_STRATEGY="linear"                  # linear | exponential
RAMP_START=1                            # starting RPS
RAMP_END=8                              # ending RPS
NUM_PROMPTS=20
INPUT_LEN=8000                          # random input tokens per request
OUTPUT_LEN=128                          # random output tokens per request
MAX_CONCURRENCY=2                       # cap at server max_num_seqs
EXTRA_ARGS=()                           # passthrough to vllm bench serve

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)           SERVER="$2"; shift 2 ;;
    --ssh-host)         SSH_HOST="$2"; shift 2 ;;
    --proc-name)        PROC_NAME="$2"; shift 2 ;;
    --remote-reader)    REMOTE_READER="$2"; shift 2 ;;
    --client)           CLIENT="$2"; shift 2 ;;
    --bench-image)      BENCH_IMAGE="$2"; shift 2 ;;
    --model)            MODEL="$2"; shift 2 ;;
    --endpoint)         API_ENDPOINT="$2"; shift 2 ;;
    --out)              OUT="$2"; shift 2 ;;
    --sample-interval)  SAMPLE_INTERVAL="$2"; shift 2 ;;
    --ramp-strategy)    RAMP_STRATEGY="$2"; shift 2 ;;
    --ramp-start)       RAMP_START="$2"; shift 2 ;;
    --ramp-end)         RAMP_END="$2"; shift 2 ;;
    --num-prompts)      NUM_PROMPTS="$2"; shift 2 ;;
    --input-len)        INPUT_LEN="$2"; shift 2 ;;
    --output-len)       OUTPUT_LEN="$2"; shift 2 ;;
    --max-concurrency)  MAX_CONCURRENCY="$2"; shift 2 ;;
    --help|-h)          grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── Sanity: helper scripts must exist ────────────────────────────────────────
for f in "$VRAM_READER" "$CPU_WRAPPER" "$MERGE_VRAM"; do
  [[ -f "$f" ]] || { echo "ERROR: helper script missing: $f"; exit 2; }
done

TS=$(date -u +%Y%m%d-%H%M%S)
OUT="${OUT:-bench-vram-${TS}.json}"
VRAM_TRACE="/tmp/vram-trace-${TS}.jsonl"
: > "$VRAM_TRACE"

# vllm bench builds URL as http://<host>:<port><api_endpoint>; pass bare host+port.
HOST="${SERVER#http://}"; HOST="${HOST#https://}"; HOST="${HOST%%:*}"
PORT="${SERVER##*:}"
[[ "$PORT" == "$HOST" ]] && PORT=8888

echo "=== vllm-bench-vram ==="
echo "Server    : $SERVER  (api: $API_ENDPOINT)"
echo "Client    : $CLIENT"
echo "VRAM proc : $PROC_NAME  (via nvitop on $SSH_HOST)"
echo "Ramp      : $RAMP_STRATEGY $RAMP_START -> $RAMP_END RPS"
echo "Prompts   : $NUM_PROMPTS  (input≈$INPUT_LEN, output≈$OUTPUT_LEN tokens)"
echo "MaxConc   : $MAX_CONCURRENCY"
echo "Out       : $OUT"
echo "----------------------------------------"

# ── VRAM reader path on the dgx-spark ───────────────────────────────────────
# The repo is synced to the dgx-spark via Unison (see root AGENTS.md "Sync &
# Deployment"), so vram-reader.py is already there — no scp needed. Default to
# the synced repo path; override with --remote-reader if the layout differs.
REMOTE_READER="${REMOTE_READER:-/home/saumen/workspace/github/saumen/vllm/vllm-bench/vram-reader.py}"

# ── VRAM sampler (nvitop on the dgx-spark, over SSH, background) ────────────
# Polls the PROC_NAME process's GPU memory every SAMPLE_INTERVAL seconds by
# running vram-reader.py on the dgx-spark (synced via Unison).
vram_sampler() {
  local last=""
  while true; do
    local gib
    gib=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" \
      "python3 $REMOTE_READER \"$PROC_NAME\"" 2>/dev/null | tr -d '[:space:]')
    if [[ -n "$gib" && "$gib" != "$last" ]]; then
      printf '{"t": %s, "vram_gib": %s}\n' "$(date +%s)" "$gib" >> "$VRAM_TRACE"
      last="$gib"
    fi
    sleep "$SAMPLE_INTERVAL"
  done
}
vram_sampler &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

# ── Build the vllm bench serve args ─────────────────────────────────────────
# --model only added if non-empty (else vllm bench auto-detects from the server).
# $1 = result-filename path (differs per client mode: local OUT vs in-container).
build_bench_args() {
  local result_path="$1"
  BENCH_ARGS=(serve --backend openai-chat
    --host "$HOST" --port "$PORT"
    --endpoint "$API_ENDPOINT"
    --dataset-name random
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY"
    --ramp-up-strategy "$RAMP_STRATEGY"
    --ramp-up-start-rps "$RAMP_START" --ramp-up-end-rps "$RAMP_END"
    --save-result --result-filename "$result_path")
  [[ -n "$MODEL" ]] && BENCH_ARGS+=(--model "$MODEL")
  BENCH_ARGS+=("${EXTRA_ARGS[@]}")
}

# ── Run the benchmark (client mode) ─────────────────────────────────────────
# The bench client must be SEPARATE from the vLLM serving container so its own
# memory can't perturb the VRAM measurement. Three modes:
case "$CLIENT" in
  mac-container)
    # Throwaway vllm container ON THE MAC-STUDIO. --network host so it can
    # reach the dgx-spark endpoint; no GPU (client only). The image's entrypoint
    # is `vllm`, which crashes on a no-GPU linux container (device inference); we
    # bypass it with bench-cpu-wrapper.py (mounted) which forces the CPU
    # platform before invoking the CLI.
    if ! command -v docker >/dev/null 2>&1; then
      echo "ERROR: docker not found on the mac-studio (needed for mac-container client)."; exit 3
    fi
    if ! docker image inspect "$BENCH_IMAGE" >/dev/null 2>&1; then
      echo "Pulling $BENCH_IMAGE (~20 GB, one-time)..."
      docker pull "$BENCH_IMAGE" || { echo "ERROR: pull failed"; exit 3; }
    fi
    # Mount a host dir so the --save-result JSON (written inside the container)
    # lands on the mac-studio.
    OUTDIR_MAC=$(mktemp -d /tmp/bench-out.XXXXXX)
    OUT_IN_CONTAINER="/tmp/bench-out/result.json"
    build_bench_args "$OUT_IN_CONTAINER"
    echo "Running bench client in mac-studio container ($BENCH_IMAGE) -> $SERVER"
    docker run --rm --network host --entrypoint python3 \
      -v "$CPU_WRAPPER":/w.py -v "$OUTDIR_MAC":/tmp/bench-out "$BENCH_IMAGE" \
      /w.py bench "${BENCH_ARGS[@]}"
    BENCH_RC=$?
    # Move the result JSON from the mounted dir to the requested OUT path.
    if [[ -f "$OUTDIR_MAC/result.json" ]]; then
      mv "$OUTDIR_MAC/result.json" "$OUT"
    fi
    rm -rf "$OUTDIR_MAC"
    ;;
  local)
    # vllm CLI directly on the mac-studio.
    if ! command -v vllm >/dev/null 2>&1; then
      echo "ERROR: 'vllm' CLI not found on the mac-studio. Install it (pip install vllm)"; exit 3
    fi
    build_bench_args "$OUT"
    echo "Running bench client locally -> $SERVER"
    vllm bench "${BENCH_ARGS[@]}"
    BENCH_RC=$?
    ;;
  dgx-container)
    # Throwaway vllm container ON THE DGX-SPARK (fallback if no Mac Docker). The
    # wrapper is synced to the dgx-spark via Unison at the same relative repo path.
    build_bench_args "$OUT"
    echo "Running bench client on dgx-spark in throwaway container ($BENCH_IMAGE)"
    # dgx-spark-side path to the synced wrapper (same relative path under its home).
    DGX_WRAPPER="/home/saumen/workspace/github/saumen/vllm/vllm-bench/bench-cpu-wrapper.py"
    ssh "$SSH_HOST" "docker run --rm --network host --entrypoint python3 \
      -v $DGX_WRAPPER:/w.py $BENCH_IMAGE /w.py bench ${BENCH_ARGS[*]}"
    BENCH_RC=$?
    echo "NOTE: dgx-container mode does not persist the bench JSON (container is --rm)."
    OUT="/dev/null"
    ;;
  *)
    echo "ERROR: unknown --client '$CLIENT' (use mac-container|local|dgx-container)"; exit 2 ;;
esac

kill $SAMPLER_PID 2>/dev/null
wait $SAMPLER_PID 2>/dev/null

# ── Merge VRAM summary into the bench result (via merge-vram.py) ────────────
python3 "$MERGE_VRAM" "$OUT" "$VRAM_TRACE"

echo "----------------------------------------"
echo "Bench result: $OUT"
echo "VRAM trace  : $VRAM_TRACE"
exit $BENCH_RC
