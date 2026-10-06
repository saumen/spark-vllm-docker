---
name: dgx-spark-vllm-launch
description: Use when launching vLLM on DGX Spark with NVFP4 scripts.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [vllm, dgx-spark, nvfp4, flashinfer, cudagraph, speculative-decoding]
    related_skills: [serving-llms-vllm]
---

# DGX Spark vLLM Launch Patterns

## Overview

This skill covers launch patterns for vLLM on DGX Spark (arm64 + sm_121a) with NVFP4
models, specifically the `start-laguna-s-2.1.sh`, `start-unsloth-qwen3.6-nvfp4.sh`,
and `start-unsloth-qwen3.8-27b-nvfp4.sh` scripts. The Qwen3.8 script is a **common
launcher** — all vLLM params (context window, max seqs, YaRN enablement, speculative
tokens) are env-var driven with defaults, and a `Makefile` defines four per-preset
KV-pinned targets (`qwen38-1m-1seq`, `qwen38-500k-2seq`, `qwen38-262k-2seq`,
`qwen38-262k-4seq`). A `kv-cache-budget.py` calculator budgets against the explicit
`--kv-cache-memory` byte pin (NOT the buggy `--gpu-memory-utilization` sizing — see
§GPU Memory Budgeting). These scripts wrap
`vllm serve` with hardware-specific flags, speculative decoding, and NVFP4 JIT
compilation settings.

Key differences from generic vLLM deployment:
- Uses `--compilation-config` with cudagraph modes (see CUDAGraphMode pitfalls)
- Uses `--speculative-config` with DFlash or MTP methods
- Sets `CUTE_DSL_ARCH=sm_121a` for FP4 kernel JIT
- Mounts cache volumes from host to container

## When to Use

- Launching vLLM on DGX Spark with Laguna S 2.1, Unsloth Qwen3.6, or Qwen3.8 NVFP4 models
- Debugging startup warnings about CUDAGraphMode incompatibility
- Tuning speculative decoding or compilation config for NVFP4
- Modifying launch scripts that reference `OVERRIDE_GEN_CONFIG` or `CUDAGRAPH_MODE` variables
- Creating memory/context variants of existing launchers (e.g. 1M→262K context, 4→2 seqs, different KV caps) — use the env-var + Makefile pattern (see below), not separate script copies
- **Sizing or debugging KV/VRAM on GB10 unified memory** — the vLLM startup profiler is buggy here (negative "peak activation"), so `--gpu-memory-utilization` OVER-ALLOCATES the KV pool. Pin `--kv-cache-memory` (bytes) instead. See §GPU Memory Budgeting and `references/vllm-memory-budgeting.md`.
- **Measuring whether vLLM VRAM stays flat or rises under load** — on GB10 the GPU KV pool is invisible to `/proc/meminfo`, cgroup `memory.current`, `docker stats`, and `nvidia-smi`. The **authoritative signal is the `VLLM::EngineCore` process GPU memory via nvitop/NVML** (e.g. 77.26 GiB — the same number nvitop's UI shows); `GPU KV cache usage: %` is the secondary variable-part-only signal (noisier — fluctuates with prefix-cache eviction). Use `vllm bench serve` (preferred — throughput/TTFT/TPOT/spec-decode metrics + VRAM in one; see `vllm-bench-vram.sh`, which runs the client in a SEPARATE container so it can't perturb the measurement) or the lighter `vllm-vram-load-test.py`. **Run the Mac bench-client container natively (`linux/arm64`) — do NOT add `--platform linux/amd64`** (it's multi-arch with a real arm64 build; amd64 = Rosetta/QEMU, slower for no benefit). **Interpretation:** a "FLAT" verdict from the light Python ramp is *inconclusive* (under-loaded). To see the real climb, run the heavier `vllm bench serve` path with enough `--num-prompts` that requests stay running at `max_num_seqs` — measured 2026-08-16: sustained 24×64k load drove KV usage 20.5%→29.8% (≈11.3 GiB of the 38 GiB pool) and it held. VRAM is always bounded by `base + pin` (≈62 + 38 ≈ 100 GiB ceiling), so it cannot OOM from KV growth alone while the pin is set. See `references/vllm-memory-budgeting.md` §Measuring on the box.

**Don't use for:** Generic vLLM deployment without DGX Spark, non-NVFP4 models,
or non-Docker environments. Use `serving-llms-vllm` instead.

## Variable Extraction Pattern

When adding configurable launch parameters, extract them into named variables with
inline comments referencing the source of truth (e.g., a social media post, model
card, or docs URL). This makes future updates straightforward and self-documenting.

```bash
# Temperature set to 1.0 per https://x.com/poolsideai/status/2083225324645503399
OVERRIDE_GEN_CONFIG='{"temperature":1.0,"top_p":0.95}'

# CUDAGraphMode FULL is not supported with FlashInferBackend (requires AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
# FULL_AND_PIECEWISE is also unsupported with spec-decode. Hardcode PIECEWISE to skip runtime trial/fallback.
CUDAGRAPH_MODE='{"cudagraph_mode":"PIECEWISE"}'
```

Then reference the variables in the `docker run` command:

```bash
docker run ... \
  --override-generation-config "$OVERRIDE_GEN_CONFIG" \
  --compilation-config "$CUDAGRAPH_MODE" \
  ...
```

**Verification:** After editing, run `bash -n <script>` to confirm syntax, then
grep for the old hardcoded value to ensure it was fully replaced.

## Env-Var + Makefile Pattern (Common Launcher)

When a launcher needs multiple memory/context profiles (e.g. "1M YaRN, 1 seq,
~100 GiB" vs "262K native, 2 seqs, ~82 GiB"), consolidate into a **single common
script** with env-var-driven params + a **Makefile** with one target per preset.
Do NOT create separate script copies — they drift and duplicate 95% of the code.

**Script side:** extract each varying param into `${VAR:-default}` with validation:

```bash
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-1000000}"
YARN_ENABLED="${YARN_ENABLED:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.575}"
SERVICE="${SERVICE:-vllm-unsloth-qwen38-27b-nvfp4}"
```

Gate conditional flags behind the env vars:
- `--hf-overrides` (YaRN rope params) only when `YARN_ENABLED=true`
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` in the env heredoc only when `YARN_ENABLED=true`
- Add cross-check warnings (e.g. YaRN enabled but context ≤ 262144)

**Makefile side:** one target per preset, each setting the env vars inline.
**All presets share the SAME `SERVICE` container name** (a single `SERVICE :=`
variable at the top) so the pre-flight `docker stop` in the script kills any
existing instance — this prevents two vLLM engines from co-running on the
128 GB unified-memory DGX Spark (which would OOM the disk). Do NOT give each
preset a distinct container name.

```make
SCRIPT := unsloth-qwen3.8-27b-nvfp4/start-unsloth-qwen3.8-27b-nvfp4.sh
SERVICE := vllm-unsloth-qwen38-27b-nvfp4

qwen38-1m-1seq:
	SERVICE=$(SERVICE) \
	MAX_NUM_SEQS=1 \
	CONTEXT_WINDOW=1000000 \
	YARN_ENABLED=true \
	KV_CACHE_MEMORY=40802750464 \
	bash $(SCRIPT)
```

**Note:** presets that pin `KV_CACHE_MEMORY` do NOT also set
`GPU_MEMORY_UTILIZATION` — the byte pin overrides util for KV sizing (vLLM logs
"This does not respect the gpu_memory_utilization config"), so passing a util in a
pinned preset is dead weight that misleads readers. The launcher keeps its own util
default (0.575) only as a fallback for direct invocation with no pin.

**Verification:** `make -n <target>` prints the resolved command without executing
it — use this to confirm env var values flow through correctly (and that the
`SCRIPT :=` path resolves to the model subfolder). Also verify the
script's defaults match the "default" Makefile target (the one with no overrides).

**Model-specific folder:** keep the common launcher + `kv-cache-budget.py` in a
per-model subfolder (e.g. `unsloth-qwen3.8-27b-nvfp4/`) since they encode
model-specific architecture constants; keep the `Makefile` at the repo root so
presets are reachable via `make <target>` from anywhere. Point the Makefile's
`SCRIPT :=` at the subfolder path:

```make
SCRIPT := unsloth-qwen3.8-27b-nvfp4/start-unsloth-qwen3.8-27b-nvfp4.sh
SERVICE := vllm-unsloth-qwen38-27b-nvfp4
```

**AGENTS.md update:** when consolidating, update the container flags table to show
`${VAR}` references (not hardcoded values), add an env-var table with defaults,
add a Makefile preset table, and update the commands reference to use `make`
targets instead of `bash script.sh`. Note in the container-names instruction that
all presets share one name.

**KV budget calculator:** keep the budget math in a single Python script
(`kv-cache-budget.py`) rather than only in shell comments — comments drift and
can't be re-run. Derive `bytes_per_token` from the model architecture, not a
round estimate: `2 (K+V) × num_kv_heads × head_dim × bytes_per_elem ×
num_kv_cache_layers`. For Qwen3.8-27B (GQA 4 KV heads, head_dim 256, fp8, 19
KV-cache layers = 16 full-attention + 3 padding) that is **38,912 B/token**,
not ~32. The script should expose (a) preset-verification mode that cross-checks
every Makefile KV pin against the safe steady-state cap, and (b) a
`--context N --seqs N --kv N` mode that budgets against a KV pool size (GiB) and
prints the recommended `KV_CACHE_MEMORY=` (bytes). **Important
(measured 2026-08-15): the fixed footprint does NOT scale with `max_num_seqs`**
— vLLM pre-allocates activation buffers sized for `max_num_batched_tokens`,
not per-sequence. **The Qwen3.8 launcher pins `--kv-cache-memory` (bytes, via a
`KV_CACHE_MEMORY` env var), NOT `--gpu-memory-utilization`** (since 2026-08-16):
on GB10 unified memory + MTP spec-decode the vLLM startup profiler reports a
NEGATIVE "peak activation" (~-28 GiB) and an inflated "consumed memory" (~60
GiB), so util-based sizing OVER-ALLOCATES the KV pool (unpinned 0.85 runs hit a
72 GiB KV pool → ~134 GiB → OOM). The byte pin overrides the buggy profiler. See
§GPU Memory Budgeting for the corrected model.

## CUDAGraphMode Pitfalls

See `references/cudagraph-mode.md` for the full detail on CUDAGraphMode
incompatibility with FlashInferBackend and speculative decoding, including
the exact startup warnings and how to fix them.

## GPU Memory Budgeting

See `references/vllm-memory-budgeting.md` for the corrected VRAM model on GB10:
the vLLM profiler bug (negative "peak activation") that makes
`--gpu-memory-utilization` over-allocate the KV pool, why to pin
`--kv-cache-memory` (bytes) instead, how to measure the true KV-independent base
directly (NOT by subtracting the KV pool from the startup figure), and the
steady-state = base + KV pool model.

### Unified-Memory Coexistence Constraint (DGX Spark GB10)

The DGX Spark has **no separate VRAM** — GPU and CPU share 128 GB LPDDR5x
(121.69 GiB total, ~115.25 GiB usable after OS/driver overhead).

**Measuring vLLM GPU memory (verified working on GB10):**
- **The `VLLM::EngineCore` process GPU memory via nvitop/NVML is the authoritative
  signal** for what vLLM consumes from VRAM (the same number nvitop's UI shows,
  e.g. 77.26 GiB):
  ```bash
  ssh dgx-spark "python3 -c 'import nvitop
  for g in nvitop.Device.all():
    for pid,p in g.processes().items():
      if p.name()==\"VLLM::EngineCore\": print(f\"{p.gpu_memory()/1073741824:.2f} GiB\")'"
  ```
  (nvitop is NVML-backed and works on GB10 even though `nvidia-smi`'s top-level
  "Memory-Usage" column says "Not Supported".)
- **`GPU KV cache usage: %` is the secondary (variable-part-only) signal** — the
  fraction of the pinned KV pool that is resident:
  `docker logs --tail 40 <container> 2>&1 | grep -oE 'GPU KV cache usage: [0-9.]+%' | tail -1`
  (or `curl -s localhost:8888/metrics | grep kv_cache_usage`). It fluctuates with
  prefix-cache eviction (decays toward 0% between stages) — expected noise, not a
  leak. Use it to attribute how much of a VRAM rise is KV vs. other.
- **⚠️ Do NOT use `/proc/meminfo` (`MemTotal − MemAvailable`), cgroup
  `memory.current`, `docker stats`, or process RSS to measure vLLM VRAM on GB10.**
  The KV pool lives in GPU HBM, which host RAM accounting does NOT see. A load
  test confirmed `/proc/meminfo` "used" stayed flat at ~86 GiB and cgroup
  `memory.current` at ~15.8 GiB while the KV pool was exercised. `nvidia-smi
  --query-compute-apps` showed only Xorg/gnome-shell, not the vLLM engine — not
  reliable here; prefer nvitop.

**⚠️ The vLLM startup profiler is BUGGY on GB10 (measured 2026-08-16, vLLM
v0.27.1, MTP spec-decode).** The `gpu_worker.py` startup log reports a
**NEGATIVE "peak activation"** (~-28 GiB) and an **inflated "consumed memory
(weights + non-torch)"** (~60 GiB). Both are wrong. Because the profiler thinks
non-KV cost is ~31.5 GiB when the real KV-independent base is ~62 GiB resident,
sizing the KV pool via `--gpu-memory-utilization` **OVER-ALLOCATES** it:
unpinned 0.85 runs hit a **72 GiB KV pool → ~134 GiB total → OOM**. Do NOT trust
the "peak activation" or "consumed memory" figures from that log line, and do NOT
size KV from `--gpu-memory-utilization` on this box.

**Corrected VRAM model (v2, measured 2026-08-16):**
```
steady-state = BASE (KV-independent, resident) + KV pool
```
- **BASE ≈ 62 GiB** (KV-independent: weights + CUDA graphs + runtime), and it is
  **resident at startup**. Measure it **directly** as vLLM's "GPU-MEM used" at
  startup with a KV pin: the KV pool is `cudaMalloc`'d but NOT resident until
  prompts fault it in, so the startup figure ≈ BASE. (Measured 500k-2seq, 38 GiB
  pin: vLLM actual **62.18 GiB at startup** ⇒ BASE ≈ 62 GiB.)
- **KV pool** — set explicitly by the `--kv-cache-memory` byte pin; becomes
  resident as prompts write into it (the "<TBD> GiB after a prompt" ≈ the pool
  faulting in). Check steady-state AFTER a prompt, not just at startup.

**⚠️ Do NOT derive BASE by subtracting the KV pool from the startup figure.** The
first (wrong) model did `BASE = startup_footprint − KV_pool` (61.98 − 38.14 = 23.84
GiB). That is bogus: the KV pool is not resident at startup, so the startup figure
does not contain its bytes. It produced a phantom 23.84 GiB base + phantom 15 GiB
"runtime" → ~77 GiB steady-state. The true base is ~62 GiB → ~100 GiB steady-state.

**Pin `--kv-cache-memory` (bytes), not `--gpu-memory-utilization`.** The byte pin
makes vLLM skip the buggy profiler and use exactly your pool size. Each preset pins
its own `seqs × ctx × bytes_per_token` + 1 GiB margin (bytes/token = 38,912). Keep
steady-state < ~110 GiB. The launcher exposes this as a `KV_CACHE_MEMORY` env var
(bytes); presets do NOT also pass `GPU_MEMORY_UTILIZATION` (the pin overrides it).

| Preset (2026-08-16 v2) | ctx | seqs | KV pin | steady-state | sharing |
|---|---|---|---|---|---|
| qwen38-1m-1seq | 1,000,000 | 1 | 38 GiB | ~100 GiB | no |
| qwen38-500k-2seq | 500,000 | 2 | 38 GiB | ~100 GiB | no |
| qwen38-262k-2seq | 262,144 | 2 | 20 GiB | ~82 GiB | yes (~40 GiB free) |
| qwen38-262k-4seq | 262,144 | 4 | 39 GiB | ~101 GiB | no |

**Size each pin to its own `seqs × ctx × bytes_per_token`, not one blanket value.**
A single 38 GiB pin across all presets is a "one safe number" shortcut. 262k-2seq
only needs ~19 GiB (2×262K), so it uses a 20 GiB pin — which is also what makes it
the coexistence preset (~40 GiB free for a second instance). Verify with
`kv-cache-budget.py --context N --seqs N --kv G`.

**Max safe util on GB10 is 0.85** (0.95 OOM'd 2026-08-15) — but util is now
only a fallback; the byte pin is the real control.

**Diagnosis command:**
```bash
ssh dgx-spark "python3 -c 'import nvitop
for g in nvitop.Device.all():
  for pid,p in g.processes().items():
    if p.name()==\"VLLM::EngineCore\": print(f\"vLLM VRAM: {p.gpu_memory()/1073741824:.2f} GiB\")'"
```
Read the `VLLM::EngineCore` GPU memory (the authoritative VRAM signal — see the
measuring note above; do NOT use `free -h`/`MemAvailable`/cgroup for vLLM VRAM).
If steady-state (after a prompt) VRAM > ~110 GiB, the KV pool is too big — lower
the `KV_CACHE_MEMORY` pin.

## Common Pitfalls

1. **Hardcoding values inline** instead of using named variables — makes future
   updates error-prone and removes the ability to add inline rationale.

2. **Using `cudagraph_mode: FULL`** with FlashInferBackend — triggers runtime
   fallback warnings and wastes startup time. See references.

3. **Changing one script without checking the other** — `start-laguna-s-2.1.sh`
   and `start-unsloth-qwen3.6-nvfp4.sh` share similar patterns but are NOT
   identical. Verify changes per-script.

   To find what actually changed between two launchers (e.g. when the user asks
   "did I miss anything that's different/better now?"), diff the `docker run`
   flag sets programmatically rather than eyeballing. Extract each script's
   `docker run ... docker logs` block, parse `--flag value` pairs, and print a
   side-by-side table marking rows where the two differ. This surfaces real
   diffs that prose comparison misses — e.g. a spec-decode method change
   (DFlash external draft model → built-in MTP), an added
   `--compilation-config`, a `gpu-memory-utilization` bump, or dropped
   backends (`--moe-backend`/`--attention-backend`/`--language-model-only`).
   Then split results into "worth mentioning" (user-facing wins) vs "config
   noise" (removed-because-no-longer-needed, not improvements) before
   reporting. Note: `hermes_tools.read_file` returns a dict keyed by
   `content_returned` (not `content`) — read the script with plain `open()` in
   `execute_code` instead.

4. **Leaving dead code in launch scripts** — helper functions like `ok()`
   and `info()` are sometimes carried over from earlier versions but never
   invoked. These produce no output but inflate the script and confuse
   readers. When cleaning up, verify with `grep -n "^ok()"` /
   `grep -n "^info()"` that they are truly unused, and that `die()` is
   still called in the profile validation `case` block. Remove the dead
   functions but keep `die()` — it is the only one actually used.

5. **Forgetting to update AGENTS.md** — launch script changes that affect
   documented behavior must be reflected in the AGENTS.md container flags
   table.

6. **Hardcoding `num_speculative_tokens` in the `--speculative-config` JSON** —
   extract it to a named variable (`SPECULATIVE_TOKENS=7`) and interpolate it
   into the JSON so the acceptance-rate rationale can live in a comment. When
   bumping it (e.g. 3→7), record WHY: the user's signal was "the 3rd accepted
   token is already above 85% acceptance rate, so try higher to see if the MTP
   head sustains it." Note `MAX_NUM_BATCHED_TOKENS` must stay ≥
   `max_num_seqs × (1 + num_speculative_tokens)` per step.

7. **Using `grep -P` on macOS** — BSD grep (macOS) does not support `-P`
   (PCRE). Verification scripts that run on the Mac host will fail with
   `grep: invalid option -- P`. Use `grep -oE` (POSIX ERE) or `awk`/`sed`
   instead. Example: extract a value with
   `grep -oE 'CONTEXT_WINDOW=[0-9]+' script.sh | cut -d= -f2` rather than
   `grep -oP 'CONTEXT_WINDOW=\K[0-9]+' script.sh`.

8. **Trusting the vLLM startup profiler on GB10 (negative "peak activation").**
   The `gpu_worker.py` log reports a NEGATIVE peak activation (~-28 GiB) and an
   inflated "consumed memory" (~60 GiB) on GB10 unified memory + MTP spec-decode.
   Sizing KV via `--gpu-memory-utilization` then OVER-ALLOCATES the pool (72 GiB
   → ~134 GiB → OOM). Pin `--kv-cache-memory` (bytes) instead. **Measure the true
   KV-independent BASE directly** as vLLM's "GPU-MEM used" at startup with a pin
   (~62 GiB — the KV pool is allocated but not resident at startup). Do NOT derive
   BASE by subtracting the KV pool from the startup figure (that produced the
   phantom 23.84 GiB base). Check steady-state AFTER a real prompt (the KV pool
   faults in, up to the pin size). See §GPU Memory Budgeting.

9. **macOS has no `timeout` binary.** Ad-hoc verification scripts that wrap a
   (possibly blocking) command in `timeout 20 ...` fail with rc=127
   ("command not found") on the Mac host. Use a background job + kill guard
   instead: `cmd & pid=$!; ( sleep 20; kill "$pid" 2>/dev/null ) & killer=$!;
   wait "$pid"; kill "$killer" 2>/dev/null`. (GNU `timeout` exists on Linux/DSH
   but not stock macOS.)

10. **Testing launcher flag-wiring without launching a real container.** To
    verify that an env var actually reaches the `docker run` argv (e.g. that
    `KV_CACHE_MEMORY=...` becomes `--kv-cache-memory <bytes>`, or that an empty
    value adds NO flag), put a `docker` **stub** on `PATH` for every launcher
    invocation and have it record its argv:
    ```bash
    STUB=$(mktemp -d); trap 'rm -rf "$STUB"' EXIT
    cat > "$STUB/docker" <<'EOF'
    #!/usr/bin/env bash
    case "${1:-}" in
      logs|stop|rm|inspect) exit 0 ;;          # never tail/launch
      run) [[ -n "${DOCKER_ARGV_OUT:-}" ]] && printf '%s\n' "$@" > "$DOCKER_ARGV_OUT"; exit 0 ;;
      *) exit 0 ;;
    esac
    EOF
    chmod +x "$STUB/docker"; export PATH="$STUB:$PATH"; export DOCKER_ARGV_OUT
    DOCKER_ARGV_OUT="$STUB/argv" KV_CACHE_MEMORY=40802750464 ... bash "$LAUNCHER" >/dev/null 2>&1 &
    pid=$!; ( sleep 20; kill "$pid" 2>/dev/null ) & k=$!; wait "$pid"; kill "$k" 2>/dev/null
    grep -A1 -- '--kv-cache-memory' "$STUB/argv" | tail -1   # expect the byte value
    ```
    Gotchas that cost real time: (a) the launcher's trailing `docker logs -f`
    hangs a naive stub — the stub MUST exit 0 on `logs`; (b) the pre-flight
    `docker stop` hits the stub too, so the stub must not crash under `set -u`
    (guard every var with `${VAR:-}`); (c) `DOCKER_ARGV_OUT` must be **exported**
    so the stub (a child process) can read it; (d) bound the run with the
    background-kill guard from pitfall 9, not `timeout`. This gives real
    behavioral evidence (the exact argv vLLM receives) without displacing a
    running engine.

## Verification Checklist

- [ ] `bash -n <script>` passes (syntax check)
- [ ] New variable values are correct (grep to confirm)
- [ ] Old hardcoded values are fully removed (grep to confirm)
- [ ] AGENTS.md container flags table updated if behavior changed
- [ ] Inline comments reference source of truth for each variable
- [ ] Helper functions (`ok()`, `info()`) removed if unused; `die()` retained
- [ ] `scripts/verify-launch-script.sh` passes all checks
- [ ] For Makefile presets: `make -n <target>` prints the correct env var values
- [ ] For Makefile presets: script defaults match the "default" Makefile target
- [ ] For env-var-driven params: validation rejects bad values (non-integer, unknown YARN_ENABLED)

## References

- `references/cudagraph-mode.md` — CUDAGraphMode incompatibility with FlashInferBackend + spec-decode
- `references/vllm-memory-budgeting.md` — corrected GB10 VRAM model: the vLLM profiler bug (negative "peak activation"), why to pin `--kv-cache-memory` (bytes) instead of `--gpu-memory-utilization`, deriving the true KV-independent base, the steady-state = base + KV pool model, **how to actually measure vLLM VRAM on GB10 (the `VLLM::EngineCore` process GPU memory via nvitop/NVML — NOT `/proc/meminfo`/cgroup/RSS; `GPU KV cache usage: %` is the secondary variable-part-only signal), the reusable `vllm-vram-load-test.py` load test, and the preferred `vllm bench serve` approach (`vllm-bench-vram.sh`, client in a separate container) with its gotchas**
- `references/qwen38-27b-kv-budget.md` — Qwen3.8-27B worked example: architecture-derived bytes/token (38,912), the KV-pinned Makefile presets, and the shared-container-name decisions
- `scripts/verify-launch-script.sh` — automated checks for syntax, dead code, and die() presence
- `scripts/vllm-vram-load-test.py` — reusable GB10 VRAM load test: parses vLLM's `GPU KV cache usage: %` log line (the only reliable VRAM signal on GB10), drives a small synthetic-prompt ramp (concurrency capped at `max_num_seqs`), and reports a FLAT/RISING/DECLINING verdict + peak resident-KV GiB. Stdlib-only, runs on the box.
- `scripts/vllm-bench-vram.sh` — preferred GB10 VRAM load test: wraps `vllm bench serve` (throughput/TTFT/TPOT/spec-decode metrics) with the client run in a **SEPARATE container** (default `--client mac-container`: a throwaway `vllm/vllm-openai` container on the Mac hitting the DGX endpoint over the LAN, so it can't perturb the measurement) + a background sampler that reads the **`VLLM::EngineCore` process GPU memory via nvitop/NVML** (over SSH) and merges the peak into the result JSON under `vram_usage`. Handles the `vllm bench` gotchas (bare host/port, `--endpoint /v1/chat/completions`, the Mac "Failed to infer device type" fix via the `device_type`→`cpu` wrapper, recovering the in-container `--save-result` JSON via a volume mount, native `linux/arm64` — do NOT add `--platform amd64`). Bash + docker + python3 stdlib; **no embedded Python** — the logic lives in the three helper scripts below.
- `scripts/vram-reader.py` — (runs on the box over SSH) prints a named process's GPU memory (GiB) via nvitop/NVML; the authoritative GB10 VRAM signal. Graceful (empty output, exit 0) when nvitop is absent or the process isn't found.
- `scripts/bench-cpu-wrapper.py` — (runs in the bench-client container) forces vLLM's CPU platform (`device_type`→`cpu`) so a no-GPU linux container can run `vllm bench serve`; bypasses the image's `vllm` entrypoint.
- `scripts/merge-vram.py` — (runs on the client) merges a `{t, vram_gib}` JSONL trace into the bench result JSON under `vram_usage` (first/peak/last/rise/flat, flat = rise < 2 GiB).
