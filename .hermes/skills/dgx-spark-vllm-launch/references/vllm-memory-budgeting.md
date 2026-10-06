# vLLM GPU Memory Budgeting (DGX Spark GB10) — CORRECTED 2026-08-16 (v2)

How to size the KV pool on GB10 unified memory so a warmed-up vLLM engine stays
safe (no 99%-RAM + swap, no OOM under load).

> **Supersedes the 2026-08-15 util-based guidance AND the first 2026-08-16
> "BASE 23.84 + RUNTIME 15" model.** Both were wrong on this box. See
> §Why the old models were wrong. The v2 model is grounded in a direct
> measurement: vLLM actual **62.18 GiB at startup** with a 38 GiB KV pin.

## The vLLM startup profiler is BUGGY on GB10 (measured 2026-08-16, vLLM v0.27.1)

On GB10 unified memory + MTP speculative decoding, the `gpu_worker.py` startup
log reports:

```
Free memory on device (115.94/121.69 GiB) on startup. Desired GPU memory
utilization is (0.575, 69.97 GiB). Actual usage is 59.97 GiB for consumed
memory (weights + non-torch), -28.14 GiB for peak activation, and 0.18 GiB
for CUDAGraph memory. ... Current kv cache memory in use is 38.14 GiB.
```

Two figures are wrong:
- **"peak activation" is NEGATIVE (~-28 GiB).** Physically impossible — a
  profiling artifact (negative CUDA-graph/activation delta on large unified
  memory; same family as vLLM #44740 / #35983).
- **"consumed memory (weights + non-torch)" is inflated (~60 GiB)** by the same
  bug.

Because the profiler thinks non-KV cost is ~31.5 GiB when the real KV-independent
base is ~62 GiB resident, sizing the KV pool via `--gpu-memory-utilization`
**OVER-ALLOCATES** it:
- `KV = (util × 121.69) − 31.5` (buggy) instead of a sane budget
- At util 0.85: `103.44 − 31.5 = 72 GiB` KV pool → **~134 GiB total → OOM.**

**Do NOT trust the "peak activation" or "consumed memory" from that log line,
and do NOT size KV from `--gpu-memory-utilization` on this box.**

## Pin `--kv-cache-memory` (bytes), not `--gpu-memory-utilization`

The byte pin makes vLLM **skip the buggy profiler** and use exactly your pool
size (vLLM `gpu_worker.py`: when `kv_cache_memory_bytes` is set, it logs
"skipped memory profiling" and reserves exactly that many bytes). Confirmed in
the startup log:

```
Initial free memory 115.07 GiB, reserved 38.0 GiB memory for KV Cache as
specified by kv_cache_memory_bytes config and skipped memory profiling. This
does not respect the gpu_memory_utilization config.
```

This is the reliable control on GB10.

The Qwen3.8 launcher exposes this as a `KV_CACHE_MEMORY` env var (bytes):
```bash
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
# ...
KV_CACHE_FLAG=()
if [ -n "$KV_CACHE_MEMORY" ]; then
  KV_CACHE_FLAG=(--kv-cache-memory "$KV_CACHE_MEMORY")
fi
docker run ... "${KV_CACHE_FLAG[@]}" ...   # empty array adds nothing when pin unset
```

**When a preset sets `KV_CACHE_MEMORY`, do NOT also set `GPU_MEMORY_UTILIZATION`**
in that preset — the pin overrides it for KV sizing, so the util value is dead
weight that misleads readers. (The launcher may keep its own `GPU_MEMORY_UTILIZATION`
default as a fallback for *direct* invocation with no pin; that is harmless. But
presets that pin KV should not pass a util.)

## Corrected VRAM model (v2, measured 2026-08-16)

```
steady-state = BASE (KV-independent, resident) + KV pool
```

- **BASE ≈ 62 GiB** — KV-independent (weights + CUDA graphs + runtime), and it is
  **resident at startup**. Measure it **directly** as vLLM's "GPU-MEM used" at
  startup with a KV pin: the KV pool is `cudaMalloc`'d but **NOT resident** until
  prompts fault it in, so the startup figure ≈ BASE.
  - Measured: 500k-2seq (38 GiB pin) → vLLM actual **62.18 GiB at startup** ⇒ BASE ≈ 62 GiB.
- **KV pool** — the explicit `--kv-cache-memory` byte pin. It becomes resident as
  prompts write into it; at full utilization it equals the pin size.

**The "<TBD> GiB after a prompt" you observe ≈ the KV pool becoming resident**
(up to the pin size, depending on how much context is actually used). Check
steady-state AFTER a real prompt, not just at startup.

### ⚠️ Do NOT derive BASE by subtracting the KV pool from the startup figure

The first (wrong) model did `BASE = startup_footprint − KV_pool`
(61.98 − 38.14 = 23.84 GiB). That is **bogus**: at startup the KV pool is
allocated but not resident, so the startup figure does NOT contain the KV pool's
bytes. Subtracting it produced a phantom 23.84 GiB base and a phantom "+15 GiB
runtime." The true base is ~62 GiB. **Measure BASE directly (startup figure with a
pin), don't derive it by subtraction.**

## Sizing the pin

Each pin = full-context KV need (`seqs × ctx × bytes_per_token`) + ~1 GiB margin.
`bytes_per_token = 38,912` for Qwen3.8-27B (2 × 4 KV heads × 256 head_dim × 1 B
fp8 × 19 KV layers). Keep steady-state under a safe cap (use **~110 GiB** — with
BASE 62, a 38–39 GiB pin lands at ~100–101 GiB, ~20 GiB headroom; above ~110 the
box approaches the 99%-RAM + swap regime).

| KV pin | steady-state (62 + pin) | headroom (of 121.69) | status |
|--------|-------------------------|----------------------|--------|
| 20 GiB | ~82 GiB | ~40 GiB | SAFE (sharing) |
| 26 GiB | ~88 GiB | ~34 GiB | SAFE |
| 38 GiB | ~100 GiB | ~22 GiB | SAFE (default) |
| 39 GiB | ~101 GiB | ~21 GiB | SAFE |
| 72 GiB (unpinned 0.85) | ~134 GiB | negative | **UNSAFE** (OOM) |

## The 4 Qwen3.8 presets (KV-pinned, per-preset, 2026-08-16 v2)

Each preset pins its OWN `seqs × ctx × bytes_per_token` + 1 GiB margin. No
`GPU_MEMORY_UTILIZATION` is passed (the pin overrides it).

| Preset | ctx | seqs | YaRN | KV pin | steady-state | sharing |
|--------|-----|------|------|--------|--------------|---------|
| qwen38-1m-1seq | 1,000,000 | 1 | yes | 38 GiB (40802750464) | ~100 GiB | no |
| qwen38-500k-2seq | 500,000 | 2 | yes | 38 GiB (40802750464) | ~100 GiB | no |
| qwen38-262k-2seq | 262,144 | 2 | no | 20 GiB (21474836480) | ~82 GiB | yes (~40 GiB free) |
| qwen38-262k-4seq | 262,144 | 4 | no | 39 GiB (41875931136) | ~101 GiB | no |

All share `SERVICE=vllm-unsloth-qwen38-27b-nvfp4` (pre-flight `docker stop`
prevents two engines co-running). **Size each pin to its own `seqs × ctx ×
bytes_per_token`, not one blanket value** — a single 38 GiB pin across all presets
is a "one safe number" shortcut. 262k-2seq only needs ~19 GiB (2×262K), so it uses
a 20 GiB pin, which is also what makes it the coexistence preset (~40 GiB free for
a second instance).

## Keep the math in a calculator

The repo's `kv-cache-budget.py` is the single source of truth. It budgets against
an explicit KV pool size (the `KV_CACHE_MEMORY` pin), NOT the buggy util sizing:
- **Preset mode** (default): verifies all 4 presets against the 110 GiB safe
  steady-state cap. Exits 1 on violation.
- **Custom mode**: `--context N --seqs N --kv N [--yarn]` — budgets against a KV
  pool size (GiB) and prints `KV_CACHE_MEMORY=` (bytes) for the launcher.
- **JSON mode**: `--json`.

Constants (v2): `BASE_GIB = 62.0`, `SAFE_STEADY_STATE_CAP_GIB = 110.0`,
`BYTES_PER_TOKEN = 38912`. There is **no `RUNTIME_GIB`** — steady-state is simply
`BASE_GIB + kv_gib`.

## Measuring on the box (verified working on GB10)

**⚠️ The GPU KV pool is NOT visible in host RAM accounting on GB10.** Verified by
a load test (2026-08-16): with a 38 GiB KV pin, driving 16k→256k context × 2
concurrent requests left `/proc/meminfo` "used" **flat at ~86 GiB** and cgroup
`memory.current` at **15.8 GiB** / `docker stats` at **5.3 GiB** — none of which
track the KV pool. The KV cache is allocated in **GPU HBM** via CUDA, which the
Linux cgroup/`/proc/meminfo` accounting does NOT see.

### RIGHT signal: the `VLLM::EngineCore` process GPU memory (via nvitop / NVML)

This is the authoritative "what vLLM consumes from VRAM" number — the same value
nvitop's UI shows (e.g. **77.26 GiB**). It is the process's actual HBM allocation
(constant base + resident KV). Read it with nvitop's Python API (NVML-backed,
works on GB10 even though `nvidia-smi`'s top-level "Memory-Usage" column says
"Not Supported"):

```python
import nvitop
for gpu in nvitop.Device.all():
    for pid, p in gpu.processes().items():
        if p.name() == "VLLM::EngineCore":
            print(p.gpu_memory() / 1073741824)   # GiB
```

One-liner (over SSH from the Mac):
```bash
ssh dgx-spark "python3 -c 'import nvitop,sys
for g in nvitop.Device.all():
  for pid,p in g.processes().items():
    if p.name()==\"VLLM::EngineCore\": print(f\"{p.gpu_memory()/1073741824:.2f}\"); break'"
```

**WRONG signals for "what vLLM consumes from VRAM" (do not use):**
- `/proc/meminfo` (`MemTotal − MemAvailable`) — SYSTEM RAM (OS, page cache,
  Xorg, gnome-shell). Stays ~flat regardless of KV load.
- cgroup `memory.current` / `docker stats` / process RSS — the container's
  CPU-side RAM (weights in host RAM, page cache). Does NOT include the GPU KV
  pool (~15.8 GiB cgroup vs ~77 GiB real VRAM).
- `nvidia-smi --query-compute-apps=pid,used_memory` — on this box it showed only
  Xorg/gnome-shell, not the vLLM engine. NOT reliable here; use nvitop/NVML.

### Secondary signal: `GPU KV cache usage: %` (the variable part only)

vLLM's log line `GPU KV cache usage: X%` is the **fraction of the pinned KV pool
that is resident** — one component of total VRAM (the variable one), NOT the total.
Useful for isolating KV growth, but it **fluctuates with prefix-cache eviction**
(decays toward 0% between stages), so it is noisier than the nvitop process
memory. Read it with:
```bash
docker logs --tail 40 <container> 2>&1 | grep -oE 'GPU KV cache usage: [0-9.]+%' | tail -1
```
`curl -s localhost:8888/metrics | grep -E 'cache_config_info|kv_cache_usage'` is
an alternative. **Prefer the nvitop process memory for the headline "vLLM VRAM"
number; use KV-usage % to attribute how much of the rise is KV vs. other.**

Sample steady-state AFTER a real prompt (3 samples, 3 s apart) to confirm it is
flat, not still climbing.

### Load test to confirm flat-vs-rising (reusable)

The repo has `vllm-vram-load-test.py` (stdlib-only, runs on the box). It parses
the `GPU KV cache usage: %` log line, reads `max_num_seqs` + the KV pool size
from the vLLM startup log, fires a small ramp of synthetic-prompt chat
completions (concurrency auto-capped at `max_num_seqs`), samples the log every
few seconds, and reports a FLAT/RISING/DECLINING verdict + peak resident-KV GiB.
Key params: `--ramp 8k,16k,32k,64k` (small — enough to see the trend, no need to
fill the pool; the user explicitly prefers NOT ramping to full context because
256k×2 stages queue and take ~9 min), `--concurrency 2`, `--max-tokens 128`,
`--container vllm-unsloth-qwen38-27b-nvfp4`, `--out /tmp/vram-load.json`.
Synthetic prompt sizing: ~2.8 chars/token for repetitive filler on this model
(the old 4.0 default UNDER-estimated tokens, so a "256k" prompt became ~499k
actual tokens and hit the 500k model limit — a `--max_model_len` HTTP 400).

### `vllm bench serve` (preferred — richer metrics + VRAM in one)

> **Note on the bundled script:** the skill's `scripts/vllm-bench-vram.sh` is a
> known-good reference implementation. The repo's **`vllm-bench/` folder** is the
> **current** version — it samples the `VLLM::EngineCore` process GPU memory via
> **nvitop** (not the KV-usage % log line) and runs the bench client in a
> **separate container** (`--client mac-container` default). The repo layout
> (2026-08-16) is `vllm-bench/vllm-bench-vram.sh` (the bash driver) plus **extracted
> helper `.py` files** (no embedded Python in the bash): `vram-reader.py` (runs on
> the box over SSH — prints a process's GPU memory via nvitop),
> `bench-cpu-wrapper.py` (runs in the client container — forces vLLM's CPU
> platform, the "Failed to infer device type" fix), and `merge-vram.py` (runs on
> the client — merges the VRAM trace into the bench JSON under `vram_usage`).
> `vllm-vram-load-test.py` (the lighter KV-usage-% ramp) is also in `vllm-bench/`.
> If the skill's `scripts/` copy differs, prefer the repo `vllm-bench/` copy and
> update the skill script to match. The design + gotchas below describe the
> current version.

`vllm bench serve` is the standard vLLM benchmarking tool and is **preferred over
the custom Python load test** when you want real serving metrics (request/output/
total throughput, TTFT/TPOT/ITL percentiles, speculative-decoding acceptance rate)
*plus* the VRAM signal. It does NOT report VRAM natively, so pair it with a
background sampler that reads the `VLLM::EngineCore` process GPU memory via
nvitop (over SSH) and merges the peak into the result JSON. The repo has
`vllm-bench-vram.sh` which does exactly this.

**⚠️ Run the bench client SEPARATE from the vLLM serving container.** Running
`vllm bench` via `docker exec` *inside* the vLLM container adds the client's own
Python/aiohttp memory to the container footprint and perturbs the measurement.
`vllm-bench-vram.sh` supports three `--client` modes:
- **`mac-container` (default, cleanest):** a throwaway `vllm/vllm-openai` container
  **on the mac-studio** (linux/arm64, native on Apple Silicon; ~22 GB, pulled once)
  with `--network host`, hitting the dgx-spark endpoint over the LAN. Fully separate
  from the dgx-spark vLLM container.
- **`local`:** the `vllm` CLI directly on the mac-studio (needs it installed).
- **`dgx-container`:** a throwaway container **on the dgx-spark** (not the vLLM one)
  — fallback if no Mac Docker. Does not persist the bench JSON (container is `--rm`).
  (Renamed from the earlier `box-container` — "box" was ambiguous; use explicit
  `mac-studio` / `dgx-spark` names.)

**No `scp` — the repo is synced to the dgx-spark via Unison.** `vram-reader.py`
runs on the dgx-spark from the Unison-synced repo path (default
`/home/saumen/workspace/github/saumen/vllm/vllm-bench/vram-reader.py`); override
with `--remote-reader` if the layout differs. The `dgx-container` mode likewise
mounts `bench-cpu-wrapper.py` from the synced path. Do NOT `scp` helpers to
`/tmp` — the sync makes it redundant (and the user flagged it).

**Mac-container gotchas (each cost real time):**
0. **Do NOT force `--platform linux/amd64` on the Mac bench-client container.** The
   `vllm/vllm-openai` image is multi-arch with a real `linux/arm64` build, and the
   Mac (Apple Silicon) Docker daemon is `aarch64` — so the default (no `--platform`)
   runs **natively**. Forcing amd64 means Rosetta 2 / QEMU emulation: slower for no
   benefit, since the bench client is just an HTTP load generator and doesn't need to
   match the server's arch. Verify with `docker image inspect <img> --format
   '{{.Os}}/{{.Architecture}}'` (expect `linux/arm64`) and `docker info | grep
   Architecture` (expect `aarch64`). Only use `--platform` if the image lacked an
   arm64 build (it doesn't here).
1. **vLLM's arg-parser crashes on the Mac with "Failed to infer device type."**
   The container is linux (not darwin), so vLLM's CPU-platform plugin
   (`sys.platform.startswith("darwin")`) doesn't auto-activate and
   `current_platform.device_type` resolves to `UnspecifiedPlatform` (''). The
   bench client never touches a real device, but `vllm bench`'s arg-parser
   construction instantiates a default `VllmConfig` which calls it. **Fix:**
   bypass the image entrypoint with `--entrypoint python3` and a tiny wrapper that
   patches `type(_current_platform).device_type = property(lambda self: "cpu")`
   before calling `vllm.entrypoints.cli.main.main()`. (`VLLM_TARGET_DEVICE=cpu`
   and `--device cpu` alone do NOT work — the failure happens during parser
   construction, before CLI args are read.)
2. **`--save-result` writes the JSON inside the (removed) container.** Mount a
   host dir (`-v $OUTDIR:/tmp/bench-out`), point `--result-filename` at
   `/tmp/bench-out/result.json`, then `mv` it out after the run. (The
   `--result-filename` and its value are SEPARATE argv elements, so string
   substitution over the args array won't work — rebuild the array.)
3. **`--backend openai-chat` requires `--endpoint /v1/chat/completions`** (default
   is `/v1/completions`, which fails URL validation).
4. **`--host`/`--port` must be BARE (no scheme).** vllm bench builds the URL as
   `http://<host>:<port><endpoint>`, so `--host http://...` produces
   `http://http:8888` (DNS failure). Strip the scheme.
5. **Ramp-up request rates:** `--ramp-up-strategy {linear,exponential}` +
   `--ramp-up-start-rps` + `--ramp-up-end-rps`. Use a modest ramp (e.g. 1→8 RPS)
   so requests don't queue (queued requests take much longer and skew TTFT).
6. **`--max-concurrency` should match the server's `max_num_seqs`** (read it from
   the vLLM startup log). Firing more concurrent requests than `max_num_seqs`
   just queues them — the VRAM peak is understated and wall-time balloons.
7. **`--dataset-name random` + `--random-input-len`/`--random-output-len`** for
   synthetic prompts (no external dataset download). `--num-prompts` controls
   total requests.

**Example (from the Mac):**
```bash
bash vllm-bench-vram.sh \
  --client mac-container \
  --server http://$DGX_HOST:8888 (DGX Spark IP — see project .env / docs/infra-context.md) \
  --ssh-host dgx-spark --proc-name "VLLM::EngineCore" \
  --num-prompts 20 --input-len 16000 --output-len 128 \
  --ramp-start 1 --ramp-end 8 --max-concurrency 2 \
  --out /tmp/bench-vram.json
```
Output: a full `vllm bench serve` report (throughput, TTFT/TPOT/ITL, spec-decode
acceptance) + a merged `vram_usage` block (first/peak/last GiB of
`VLLM::EngineCore` GPU memory, rise, flat verdict) in the result JSON.

### Measured result (2026-08-16, 500k-2seq, 38 GiB pin): VRAM RISES under sustained load

The definitive answer to "does vLLM VRAM stay flat or rise?": **it rises, bounded
by the pin.** Two complementary measurements:
- **nvitop `VLLM::EngineCore` GPU memory** (the headline VRAM number): ~**77 GiB**
  at light load, climbing as more KV becomes resident. This is the constant base
  (~62 GiB) + resident KV.
- **`GPU KV cache usage: %`** (the variable part): a heavy `vllm bench serve` run
  (24 × 64k-token prompts, ramp 1→4 RPS, concurrency 2 = `max_num_seqs`) drove KV
  usage from **20.5% → 29.8% peak** and it *held* at ~29.7% at the end (did NOT
  decay) — 29.8% of the 38 GiB pool ≈ **11.3 GiB resident KV** at peak.

**Why the light ramp looks "flat" but the heavy bench looks "rising":** the
`vllm-vram-load-test.py` default ramp (8k→64k, a few requests) only moves KV usage
a few % (e.g. 5.3%→7.3%) and it *decays between stages* (prefix-cache eviction),
so its verdict is often "FLAT" — meaning the load was too light to move the
needle, NOT that VRAM is constant. To see the real climb you need **sustained
concurrent load at `max_num_seqs`** (the `vllm bench serve` path with enough
`--num-prompts` that requests stay queued/running). KV-usage % is the fraction of
the *pinned* pool that is resident, so it can never exceed 100% — the ceiling is
the pin, and total VRAM = base + (KV-usage % × pin).

**Interpretation rule:** a "FLAT" verdict from the light Python ramp is
inconclusive (under-loaded). Trust the `vllm bench serve` numbers (heavier,
sustained) for the real flat-vs-rising answer. Either way, VRAM is bounded by
`base + pin` — it cannot OOM from KV growth alone as long as the pin is set.
**KV-usage % fluctuating (rising then decaying between stages) is EXPECTED noise
from prefix-cache eviction — not a memory leak. Watch the nvitop process memory
for the headline trend.**

## Why the old models were wrong

**2026-08-15 model** (use `--gpu-memory-utilization`, "VRAM is flat under load,"
fixed footprint ~43.2 GiB): all four claims failed. The profiler bug means util
sizing over-allocates KV; VRAM is NOT flat (a 38 GiB pool grows after prompts);
the "43.2 GiB fixed" figure mixed in KV and used the buggy log.

**First 2026-08-16 model** (`steady-state = BASE 23.84 + KV + RUNTIME 15` → ~77
GiB): the BASE was derived by *subtracting* the KV pool from the startup figure
(61.98 − 38.14 = 23.84). But the KV pool is **not resident at startup**, so that
subtraction was bogus — it produced a phantom 23.84 GiB base and a phantom +15 GiB
"runtime." The direct measurement (vLLM actual 62.18 GiB at startup with a 38 GiB
pin) shows the true base is ~62 GiB and steady-state is ~100 GiB, not ~77.

## Pitfalls

1. **`--gpu-memory-utilization` over-allocates KV on GB10** (the profiler bug).
   Pin `--kv-cache-memory` (bytes) instead. If you must use util, treat the
   resulting KV pool as a lower bound on what you'll actually get and verify
   steady-state after a prompt.

2. **`--kv-cache-memory` is NOT `--kv-cache-memory-bytes`.** Different vLLM
   versions / scripts use different flag names. Check the running container's
   actual flags with `docker inspect <name> --format '{{.Config.Cmd}}'`.
   (Laguna uses `--kv-cache-memory-bytes` via `KV_CACHE_BYTES_MAP`; Qwen3.8 uses
   `--kv-cache-memory` via the `KV_CACHE_MEMORY` env var.)

3. **The startup footprint is NOT the ceiling.** The KV pool becomes resident
   after a real prompt (up to the pin size). Always check steady-state after a
   prompt, not just at startup.

4. **Don't derive BASE by subtracting the KV pool from the startup figure.** The
   KV pool is allocated but not resident at startup, so the subtraction is bogus
   (it produced the phantom 23.84 GiB base). Measure BASE directly as the startup
   figure with a pin (~62 GiB).

5. **Docs drift.** When changing these values, update the AGENTS.md container
   flags table AND the README.md per-model table in the same pass — both carry
   the memory rows and the README row is easy to miss.

6. **High `--gpu-memory-utilization` OOMs even when the math fits.** On GB10,
   0.95 OOM'd (2026-08-15). Keep a margin on unified-memory GPUs. (Now moot for
   Qwen3.8 since the byte pin is the control, but still true if you fall back to
   util.)

7. **Don't pass `GPU_MEMORY_UTILIZATION` in a preset that pins KV.** The pin
   overrides it for KV sizing, so the util value is dead weight that misleads
   readers. Presets that set `KV_CACHE_MEMORY` should not also set
   `GPU_MEMORY_UTILIZATION`.

8. **Don't measure "vLLM VRAM" with `/proc/meminfo`, cgroup `memory.current`,
   `docker stats`, or process RSS on GB10.** None of them see the GPU KV pool
   (it lives in HBM, not host RAM). A load test confirmed `/proc/meminfo` "used"
   stays flat at ~86 GiB and cgroup `memory.current` at ~15.8 GiB while the KV
   pool is being exercised. The authoritative signal is the **`VLLM::EngineCore`
   process GPU memory via nvitop/NVML** (e.g. 77.26 GiB); `GPU KV cache usage: %`
   is the secondary (variable-part-only, noisier) signal. If you see a "flat"
   host-RAM number under load and conclude "VRAM is flat," you measured the wrong
   thing. See §Measuring on the box.

## Verification

- `bash -n <script>` for syntax.
- `docker inspect <container> --format '{{.Config.Cmd}}'` to confirm the running
  flags include `--kv-cache-memory <bytes>` (catches stale containers).
- `curl -s localhost:8888/metrics | grep cache_config_info` for the resolved KV
  config (`kv_cache_memory_bytes`, `kv_cache_size_tokens`).
- **For actual VRAM under load, read the `VLLM::EngineCore` process GPU memory
  via nvitop/NVML** (the authoritative number, e.g. 77.26 GiB) — or run
  `scripts/vllm-bench-vram.sh` (preferred) / `scripts/vllm-vram-load-test.py`.
  Do NOT use `free -h`/`/proc/meminfo`/cgroup/`nvidia-smi` for vLLM VRAM — none
  of them see GPU HBM on GB10. `GPU KV cache usage: %` is the secondary
  variable-part-only signal. (See §Measuring on the box and pitfall 8.)
- After a real prompt, re-sample to confirm steady-state is flat and under cap.
- The vLLM startup log should read "reserved N GiB memory for KV Cache as
  specified by kv_cache_memory_bytes config and skipped memory profiling. This
  does not respect the gpu_memory_utilization config." — that line confirms the
  pin is active and util is ignored.
