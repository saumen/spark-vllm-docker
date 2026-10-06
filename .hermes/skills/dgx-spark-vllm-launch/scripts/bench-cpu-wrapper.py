#!/usr/bin/env python3
"""bench-cpu-wrapper.py — run `vllm bench serve` from a CPU-only (no-GPU) host.

The vllm/vllm-openai Docker image's entrypoint is `vllm`, and constructing its
arg parser instantiates a default VllmConfig, which calls
`current_platform.device_type`. On a Mac the container is linux (not darwin), so
vLLM's CPU-platform plugin doesn't auto-activate and device_type resolves to
'' (UnspecifiedPlatform) -> "Failed to infer device type" crash.

The `vllm bench serve` CLIENT never touches a real device (it's just an HTTP
load generator), so we patch device_type to 'cpu' before invoking the CLI. This
wrapper is mounted into the throwaway bench-client container and run as the
entrypoint (bypassing the image's `vllm` entrypoint):

    docker run --rm --network host --entrypoint python3 \
      -v bench-cpu-wrapper.py:/w.py <image> /w.py bench serve <args...>

NOTE: do NOT force `--platform linux/amd64` on the Mac — the image is multi-arch
with a real linux/arm64 build and the Mac Docker daemon is aarch64, so the
default runs natively. amd64 = Rosetta/QEMU, slower for no benefit.

Usage (inside the container):
    python3 bench-cpu-wrapper.py bench serve --host ... --port ... ...
"""

import os
import sys


def _force_cpu_platform() -> None:
    """Patch vLLM's current platform so device_type reports 'cpu'."""
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    import vllm.platforms as _p

    cur = _p._current_platform
    # Preferred: override the class-level device_type property.
    try:
        type(cur).device_type = property(lambda self: "cpu")
        return
    except Exception:
        pass
    # Fallback: set the instance attribute directly.
    try:
        cur.device_type = "cpu"
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not patch device_type: {e}", file=sys.stderr)


def main() -> None:
    _force_cpu_platform()
    from vllm.entrypoints.cli.main import main as vllm_main

    # The image's entrypoint normally prepends "vllm"; we bypass it, so add it.
    sys.argv = ["vllm"] + sys.argv[1:]
    vllm_main()


if __name__ == "__main__":
    main()
