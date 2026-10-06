#!/usr/bin/env python3
"""vram-reader.py — print the GPU memory (GiB) of a named process, via nvitop.

Used by vllm-bench-vram.sh's VRAM sampler: it runs ON THE DGX BOX (over SSH) and
prints a single float — the process's GPU memory in GiB — so the shell can log a
timestamped sample. This is the authoritative "what vLLM consumes from VRAM"
signal on GB10 (the VLLM::EngineCore process's HBM allocation, same number
nvitop's UI shows).

Why nvitop and not the obvious metrics: on GB10 (DGX Spark) the KV pool lives in
GPU HBM, which is invisible to /proc/meminfo (system RAM), cgroup memory.current /
docker stats (container CPU-side RAM), and nvidia-smi (reports "Not Supported"
for GB10 memory). nvitop uses NVML and reports per-process GPU memory correctly.

Usage:
    python3 vram-reader.py "VLLM::EngineCore"        # -> e.g. 77.26
    python3 vram-reader.py                           # -> default VLLM::EngineCore

Prints an empty line (and exits 0) if the process isn't found, so the sampler
can skip that sample without erroring.
"""

import sys

DEFAULT_PROC = "VLLM::EngineCore"


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROC
    try:
        import nvitop
    except ImportError:
        # nvitop not installed on this host — signal "no reading" cleanly.
        print("")
        return
    for gpu in nvitop.Device.all():
        for _pid, proc in gpu.processes().items():
            try:
                name = proc.name()
            except Exception:
                continue
            if name == target:
                gib = proc.gpu_memory() / 1073741824
                print(f"{gib:.2f}")
                return
    # Not found (or no GPU processes) — empty output, exit 0.
    print("")


if __name__ == "__main__":
    main()
