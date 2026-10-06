# Citing session figures about the DGX Spark

Session-specific measured numbers (e.g. "11.3 GiB resident KV at peak", "62 GiB BASE",
"~100 GiB steady-state") are **preset- and container-specific** — they describe one KV-pinned
engine at one context, not the box. GB10 has **no discrete VRAM**: GPU and CPU share 128 GB
LPDDR5x (121.69 GiB total, ~115 GiB usable).

When reasoning about the box in a new session:
- Do NOT reuse a remembered session number as a spec. Re-measure the live engine:
  - `curl -s http://<host>:8888/v1/models` → the running model's `max_model_len` (e.g. 262144).
  - nvitop/NVML `VLLM::EngineCore` GPU memory → actual VRAM (see §Measuring in
    `vllm-memory-budgeting.md`).
- When quoting a measured figure, label it as a session/preset measurement, not a box property.
  e.g. "11.3 GiB was the resident KV at peak in the 500k-2seq 38 GiB-pinned run" — not
  "the GPU has 11.3 GiB".

This prevents the recurring error of restating a stale session figure as hardware truth.
