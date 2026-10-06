# Qwen3.8-27B-NVFP4 KV Budget Worked Example (DGX Spark GB10) — CORRECTED 2026-08-16 (v2)

Concrete numbers from the `unsloth-qwen3.8-27b-nvfp4/start-unsloth-qwen3.8-27b-nvfp4.sh`
common launcher + root `Makefile` (4 presets) + `unsloth-qwen3.8-27b-nvfp4/kv-cache-budget.py`.
The launcher and calculator live in the model-specific subfolder; the Makefile stays at
the repo root so presets run via `make <target>` from anywhere. Use as a reference when
sizing or debugging this model's memory.

> **Supersedes the 2026-08-15 util-based layout AND the first 2026-08-16 "BASE 23.84 +
> RUNTIME 15" model.** The v2 model is grounded in a direct measurement: vLLM actual
> **62.18 GiB at startup** with a 38 GiB KV pin. See `references/vllm-memory-budgeting.md`
> for the full corrected model and why the old derivations were wrong.

## Architecture-derived constants

- **bytes_per_token = 38,912** = 2 (K+V) × 4 KV heads × 256 head_dim × 1 B (fp8) × 19 KV-cache layers
- 19 KV-cache layers = 16 full-attention (GQA 24 q-heads / 4 kv-heads) + 3 padding; the 48
  linear-attention (Gated DeltaNet) layers keep no KV cache.
- **BASE (KV-independent, resident at startup) ≈ 62 GiB** — measured directly as vLLM's
  "GPU-MEM used" at startup with a KV pin (500k-2seq, 38 GiB pin: **62.18 GiB**). The KV
  pool is `cudaMalloc`'d but NOT resident until prompts fault it in, so the startup figure
  ≈ BASE. Do NOT derive BASE by subtracting the KV pool from the startup figure (that
  produced the phantom 23.84 GiB base in the first model).
- **steady-state = BASE + KV pool** (38 GiB pin → ~100 GiB). The KV pool becomes resident
  as prompts write into it; the "<TBD> GiB after a prompt" ≈ the pool faulting in.
- **GPU**: 121.69 GiB total, ~115 GiB initial free (unified memory, no separate VRAM).
  Safe steady-state cap: **~110 GiB** (with BASE 62, a 38–39 GiB pin lands at ~100–101 GiB,
  ~20 GiB headroom).

## The 4 presets (KV-pinned, per-preset, since 2026-08-16 v2)

The launcher pins `--kv-cache-memory` (bytes) via the `KV_CACHE_MEMORY` env var, overriding
the buggy profiler. **Each preset pins its own `seqs × ctx × bytes_per_token` + 1 GiB
margin, and does NOT pass `GPU_MEMORY_UTILIZATION`** (the pin overrides it for KV sizing).

| Preset | ctx | seqs | YaRN | KV pin | steady-state | KV tokens | max ctx/seq | sharing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen38-1m-1seq | 1,000,000 | 1 | yes | 38 GiB (40802750464) | ~100 GiB | 1,048,576 | 1,048,576 | no |
| qwen38-500k-2seq | 500,000 | 2 | yes | 38 GiB (40802750464) | ~100 GiB | 1,048,576 | 524,288 | no |
| qwen38-262k-2seq | 262,144 | 2 | no | 20 GiB (21474836480) | ~82 GiB | 551,882 | 275,941 | yes (~40 GiB free) |
| qwen38-262k-4seq | 262,144 | 4 | no | 39 GiB (41875931136) | ~101 GiB | 1,076,170 | 269,042 | no |

**Why per-preset pins (not one blanket value):** size each pin to its own
`seqs × ctx × bytes_per_token`. 1m-1seq 1×1M ≈ 36.2 GiB → 38 GiB; 500k-2seq 2×500K ≈
36.2 GiB → 38 GiB; 262k-4seq 4×262K ≈ 38.0 GiB → 39 GiB; but **262k-2seq 2×262K ≈ 19 GiB
→ 20 GiB** (a blanket 38 GiB there would carry ~19 GiB of unneeded KV). The 20 GiB pin is
also what makes 262k-2seq the coexistence preset: ~82 GiB steady-state leaves ~40 GiB free
for a second vLLM instance. `max ctx/seq` is informational: it assumes all N sequences run
at full context simultaneously, which prefix caching + staggered requests avoid.

**The unpinned 0.85 configs are UNSAFE**: without the byte pin, the buggy profiler sizes
the KV pool to ~72 GiB → ~134 GiB total → OOM. The pin is what makes these presets safe.

## Key decisions captured

- **Shared container name**: all presets use `SERVICE=vllm-unsloth-qwen38-27b-nvfp4`.
  The pre-flight `docker stop` prevents two engines co-running (would OOM the 128 GB
  unified memory). Do NOT split per-preset names.
- **`--kv-cache-memory` (bytes) over `--gpu-memory-utilization`** (since 2026-08-16):
  the vLLM profiler is buggy on GB10 (negative "peak activation"), so util sizing
  over-allocates KV. The byte pin skips the profiler and uses exactly your pool size.
  The launcher exposes this as the `KV_CACHE_MEMORY` env var (bytes); an empty value
  falls back to `--gpu-memory-utilization`.
- **Presets do NOT pass `GPU_MEMORY_UTILIZATION`**: when `KV_CACHE_MEMORY` is set, vLLM
  ignores util for KV sizing (confirmed in the startup log: "This does not respect the
  gpu_memory_utilization config"). Passing it in a preset is dead weight that misleads
  readers. The launcher keeps its own util default (0.575) only as a fallback for direct
  invocation with no pin.
- **YaRN off for 262k presets**: native context is 262144, so remove `--hf-overrides`
  (YaRN rope params) AND `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Gated by `YARN_ENABLED=false`.
- **num_speculative_tokens = 3** (MTP): the launcher comment previously claimed 7 but
  the value is 3 (verified in startup logs). `MAX_NUM_BATCHED_TOKENS` (32768) must stay
  ≥ `max_num_seqs × (1 + num_speculative_tokens)` per step.

## Reproduce

```bash
# from the repo root (Makefile is at root; launcher+calculator in the model subfolder)
make qwen38-1m-1seq qwen38-500k-2seq qwen38-262k-2seq qwen38-262k-4seq   # launch presets
python3 unsloth-qwen3.8-27b-nvfp4/kv-cache-budget.py                          # verify all 4 presets vs 110 GiB cap
python3 unsloth-qwen3.8-27b-nvfp4/kv-cache-budget.py --context 500000 --seqs 2 --kv 38   # custom → KV_CACHE_MEMORY= (bytes)
python3 unsloth-qwen3.8-27b-nvfp4/kv-cache-budget.py --json                   # machine-readable
# override the pin per-run:
KV_CACHE_MEMORY=27917284240 make qwen38-262k-4seq   # 26 GiB pin, more headroom
```
