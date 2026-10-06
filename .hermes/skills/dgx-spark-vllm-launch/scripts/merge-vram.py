#!/usr/bin/env python3
"""merge-vram.py — merge a VRAM sample trace into the vllm bench result JSON.

Reads a JSONL trace of {t, vram_gib} samples (produced by vllm-bench-vram.sh's
sampler, which polls vram-reader.py on the box), computes a summary
(first/peak/last/rise/flat), and injects it into the bench result JSON under the
"vram_usage" key. Also prints a human-readable summary to stdout.

Usage:
    python3 merge-vram.py <bench-result.json> <vram-trace.jsonl>

The bench result JSON is the file written by `vllm bench serve --save-result`.
If it doesn't exist (e.g. box-container mode), the summary is still printed.
"""

import json
import os
import sys

# A VRAM rise under this many GiB is considered "flat" (within noise / the
# constant base). Above it, VRAM is genuinely climbing (more resident KV).
FLAT_THRESHOLD_GIB = 2.0


def load_trace(trace_path: str):
    """Load {t, vram_gib} samples from a JSONL file. Returns a list of floats."""
    vals = []
    if not os.path.exists(trace_path):
        return vals
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                vals.append(float(rec["vram_gib"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return vals


def summarize(vals):
    """Build the vram_usage summary dict from a list of GiB readings."""
    if not vals:
        return {"vram_samples": 0, "note": "no VRAM samples captured"}
    return {
        "vram_samples": len(vals),
        "vram_first_gib": vals[0],
        "vram_peak_gib": max(vals),
        "vram_last_gib": vals[-1],
        "vram_rise_gib": round(max(vals) - vals[0], 2),
        "vram_flat": abs(max(vals) - vals[0]) < FLAT_THRESHOLD_GIB,
    }


def merge(bench_path: str, summary: dict) -> None:
    """Inject the summary into the bench result JSON under 'vram_usage'."""
    if not os.path.exists(bench_path) or bench_path == "/dev/null":
        return
    try:
        with open(bench_path) as f:
            bench = json.load(f)
        bench["vram_usage"] = summary
        with open(bench_path, "w") as f:
            json.dump(bench, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not merge into bench JSON: {e})", file=sys.stderr)


def print_summary(summary: dict) -> None:
    print("=== VRAM (VLLM::EngineCore GPU memory) summary ===")
    print(f"  samples: {summary.get('vram_samples', 0)}")
    if summary.get("vram_samples"):
        print(f"  first: {summary['vram_first_gib']} GiB   "
              f"peak: {summary['vram_peak_gib']} GiB   "
              f"last: {summary['vram_last_gib']} GiB")
        verdict = "FLAT" if summary["vram_flat"] else "RISING"
        print(f"  rise (peak-first): {summary['vram_rise_gib']} GiB   "
              f"verdict: {verdict}")


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <bench-result.json> <vram-trace.jsonl>",
              file=sys.stderr)
        sys.exit(2)
    bench_path, trace_path = sys.argv[1], sys.argv[2]
    summary = summarize(load_trace(trace_path))
    merge(bench_path, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
