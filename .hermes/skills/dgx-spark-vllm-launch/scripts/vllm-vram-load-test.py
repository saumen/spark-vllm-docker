#!/usr/bin/env python3
"""
vLLM VRAM load test — measures what vLLM actually consumes from GPU VRAM under
load, by parsing vLLM's own "GPU KV cache usage: %" log line (the authoritative
signal on GB10, where the KV pool lives in GPU HBM and is invisible to
/proc/meminfo and cgroup memory.current).

WHY THIS METRIC (and not the others):
  - /proc/meminfo (MemTotal - MemAvailable): measures SYSTEM RAM (OS, page
    cache, Xorg). Stays ~flat regardless of KV load. WRONG signal.
  - cgroup memory.current / docker stats: measures the container's CPU-side
    RAM (weights in host RAM, page cache). Does NOT include the GPU KV pool.
    WRONG signal for VRAM.
  - vLLM "GPU KV cache usage: %": the fraction of the pinned KV pool that is
    actually resident. This is the part of VRAM that grows under load and is
    directly measurable. RIGHT signal.

  Note: KV-cache usage is "one part only" of total VRAM (weights + CUDA graphs
  + activations are other parts), but it is the part that varies with load and
  is the thing we want to watch for flat-vs-rising behavior.

SELF-CONTAINED: Python 3 stdlib only. Runs on the box (DGX Spark).

USAGE:
    python3 vllm-vram-load-test.py                          # defaults
    python3 vllm-vram-load-test.py --ramp 8k,16k,32k,64k    # smaller ramp
    python3 vllm-vram-load-test.py --concurrency 2 --max-tokens 128
    python3 vllm-vram-load-test.py --container vllm-unsloth-qwen38-27b-nvfp4 \
        --endpoint http://localhost:8888 --out /tmp/vram-load.json

WHAT IT DOES:
  1. Reads the server's max_num_seqs and KV pool size from the vLLM startup log.
  2. Captures a baseline "GPU KV cache usage" (idle).
  3. For each context size in --ramp, fires min(--concurrency, max_num_seqs)
     parallel chat completions with synthetic prompts of ~that many tokens,
     and samples the vLLM log every --interval seconds for the KV-usage %.
  4. Reports whether VRAM (KV usage) stayed FLAT or ROSE, the peak %, and the
     implied resident KV GiB (peak % x pool size).
  5. Writes a JSON trace (--out) for later analysis.

TOKEN ESTIMATE: synthetic prompt = repeated filler; ~2.8 chars/token (measured
on this model for repetitive text). Adjust --chars-per-token for tighter control.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


KV_USAGE_RE = re.compile(r"GPU KV cache usage:\s*([\d.]+)%")


# -- vLLM log access ----------------------------------------------------------

def docker_logs_tail(container, lines=200):
    """Return the last `lines` of the container's log (stdout+stderr)."""
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout + out.stderr
    except Exception as e:  # noqa: BLE001
        return f"[docker logs error: {e}]"


def latest_kv_usage_pct(container):
    """Parse the most recent 'GPU KV cache usage: X%' from the log. None if absent."""
    log = docker_logs_tail(container, lines=40)
    matches = KV_USAGE_RE.findall(log)
    if not matches:
        return None
    return float(matches[-1])


def read_server_config(container):
    """Extract max_num_seqs and kv_cache_memory_bytes from the vLLM startup log."""
    log = docker_logs_tail(container, lines=4000)
    cfg = {"max_num_seqs": None, "kv_cache_memory_bytes": None,
           "gpu_kv_cache_tokens": None}
    m = re.search(r"'max_num_seqs':\s*(\d+)", log)
    if m:
        cfg["max_num_seqs"] = int(m.group(1))
    m = re.search(r"'kv_cache_memory_bytes':\s*(\d+)", log)
    if m:
        cfg["kv_cache_memory_bytes"] = int(m.group(1))
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
    if m:
        cfg["gpu_kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    return cfg


# -- Synthetic prompt ---------------------------------------------------------

def make_prompt(target_tokens, chars_per_token=2.8):
    """Filler prompt of ~target_tokens tokens (repetitive, non-compressible)."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
             "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
             "victor", "whiskey", "xray", "yankee", "zulu"]
    target_chars = int(target_tokens * chars_per_token)
    parts, i, total = [], 0, 0
    while total < target_chars:
        w = words[i % len(words)] + str(i // len(words))
        parts.append(w)
        total += len(w) + 1
        i += 1
    return " ".join(parts)[:target_chars]


# -- vLLM chat completion -----------------------------------------------------

def chat_completion(endpoint, model, prompt, max_tokens, timeout=600):
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return False, time.time() - t0, f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, f"{type(e).__name__}: {e}"


# -- KV-usage sampler (polls the vLLM log) ------------------------------------

class KvSampler:
    """Polls 'GPU KV cache usage' from the vLLM log every `interval` seconds."""

    def __init__(self, container, interval):
        self.container = container
        self.interval = interval
        self.samples = []  # (t_epoch, kv_pct)
        self._stop = False

    def _loop(self):
        import threading
        while not self._stop:
            pct = latest_kv_usage_pct(self.container)
            if pct is not None:
                self.samples.append((time.time(), pct))
            time.sleep(self.interval)

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=self.interval + 2)

    def peak(self):
        return max((s[1] for s in self.samples), default=None)


# -- Ramp stage ---------------------------------------------------------------

def run_stage(endpoint, model, container, ctx_tokens, concurrency, max_tokens,
              interval, chars_per_token, hold, timeout):
    prompt = make_prompt(ctx_tokens, chars_per_token)
    approx_tokens = int(len(prompt) / chars_per_token)

    sampler = KvSampler(container, interval)
    sampler.start()
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(chat_completion, endpoint, model, prompt,
                          max_tokens, timeout) for _ in range(concurrency)]
        for f in cf.as_completed(futs):
            results.append(f.result())
    wall = time.time() - t0
    time.sleep(hold)  # let the log flush + catch late KV-usage updates
    sampler.stop()

    ok = sum(1 for r in results if r[0])
    return {
        "ctx_tokens": ctx_tokens,
        "approx_prompt_tokens": approx_tokens,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ok": ok, "failed": concurrency - ok,
        "wall_s": round(wall, 1),
        "peak_kv_pct": sampler.peak(),
        "errors": [r[2] for r in results if not r[0]][:3],
        "trace": sampler.samples,
    }


# -- Main ---------------------------------------------------------------------

def parse_ramp(s):
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        mult = 1000 if part.endswith("k") else (1_000_000 if part.endswith("m") else 1)
        if part.endswith(("k", "m")):
            part = part[:-1]
        out.append(int(float(part) * mult))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="vLLM VRAM load test (GPU KV cache usage under load).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--endpoint", default="http://localhost:8888",
                    help="vLLM OpenAI base URL (default: %(default)s)")
    ap.add_argument("--container", default="vllm-unsloth-qwen38-27b-nvfp4",
                    help="Docker container name to read logs from (default: %(default)s)")
    ap.add_argument("--model", default=None,
                    help="model id (default: auto-detect from /v1/models)")
    ap.add_argument("--ramp", default="8k,16k,32k,64k",
                    help="comma-separated context sizes (default: %(default)s; "
                         "small -- enough to see flat-vs-rising, no need to fill the pool)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel requests per stage (default: %(default)s; "
                         "auto-capped at server max_num_seqs)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="max completion tokens per request (default: %(default)s)")
    ap.add_argument("--chars-per-token", type=float, default=2.8,
                    help="chars/token for prompt sizing (default: %(default)s)")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="log sample interval seconds (default: %(default)s)")
    ap.add_argument("--hold", type=float, default=6.0,
                    help="seconds to keep sampling after each stage (default: %(default)s)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-request timeout seconds (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="JSON trace path (default: ./vram-load-<ts>.json)")
    args = ap.parse_args()

    ramp = parse_ramp(args.ramp)
    if not ramp:
        print("ERROR: empty --ramp", file=sys.stderr); sys.exit(2)

    # Server config
    cfg = read_server_config(args.container)
    max_seqs = cfg["max_num_seqs"]
    pool_bytes = cfg["kv_cache_memory_bytes"]
    pool_tokens = cfg["gpu_kv_cache_tokens"]
    pool_gib = (pool_bytes / 1024 ** 3) if pool_bytes else None
    conc = args.concurrency
    if max_seqs and conc > max_seqs:
        print(f"NOTE: --concurrency {conc} > server max_num_seqs {max_seqs}; "
              f"capping to {max_seqs} (extra requests would queue).")
        conc = max_seqs

    # Model id
    model = args.model
    if model is None:
        try:
            with urllib.request.urlopen(args.endpoint.rstrip("/") + "/v1/models",
                                        timeout=10) as r:
                model = json.loads(r.read())["data"][0]["id"]
        except Exception as e:
            print(f"ERROR: could not auto-detect model: {e}", file=sys.stderr); sys.exit(2)

    print(f"Endpoint   : {args.endpoint}")
    print(f"Container  : {args.container}")
    print(f"Model      : {model}")
    print(f"max_num_seqs: {max_seqs}   concurrency: {conc}")
    print(f"KV pool    : {pool_gib:.1f} GiB ({pool_tokens:,} tokens)" if pool_gib else "KV pool    : (unknown)")
    print(f"Ramp       : {ramp}   max_tokens={args.max_tokens}")
    print("-" * 64)

    # Baseline (idle) KV usage
    base_pct = latest_kv_usage_pct(args.container)
    print(f"Baseline KV usage (idle): {base_pct}%")
    print("-" * 64)

    stages = []
    for ctx in ramp:
        print(f"Stage ctx~{ctx:,} x {conc} ...", end=" ", flush=True)
        st = run_stage(args.endpoint, model, args.container, ctx, conc,
                       args.max_tokens, args.interval, args.chars_per_token,
                       args.hold, args.timeout)
        stages.append(st)
        if st["peak_kv_pct"] is None:
            print(f"peak=N/A ok={st['ok']}/{st['concurrency']} {st['wall_s']}s")
        else:
            print(f"peak KV={st['peak_kv_pct']:.1f}% "
                  f"ok={st['ok']}/{st['concurrency']} {st['wall_s']}s")
        for e in st["errors"]:
            print(f"    ERROR: {e}")

    # -- Analysis: flat vs rising --
    print()
    print("=" * 64)
    print("ANALYSIS (does vLLM VRAM / KV usage stay flat or rise?)")
    print("=" * 64)
    peaks = [(s["ctx_tokens"], s["peak_kv_pct"]) for s in stages
             if s["peak_kv_pct"] is not None]
    if not peaks:
        print("No KV-usage samples captured -- could not analyze.")
        sys.exit(1)

    for ctx, pct in peaks:
        gib = (pct / 100.0 * pool_gib) if pool_gib else None
        print(f"  ctx~{ctx:>7,}  KV usage {pct:5.1f}%"
              + (f"  ~ {gib:.2f} GiB resident" if gib is not None else ""))

    # Trend: compare first vs last peak
    first, last = peaks[0][1], peaks[-1][1]
    delta = last - first
    print()
    if abs(delta) < 5.0:
        verdict = "FLAT (KV usage does not materially rise with context)"
    elif delta > 0:
        verdict = f"RISING (+{delta:.1f}% from first to last stage)"
    else:
        verdict = f"DECLINING ({delta:.1f}% -- likely prefix-cache eviction between stages)"
    print(f"  Verdict: {verdict}")
    print(f"  Baseline (idle): {base_pct}%   Peak (last stage): {last}%")
    if pool_gib:
        print(f"  Peak resident KV: {last/100*pool_gib:.2f} GiB of {pool_gib:.1f} GiB pool")
    print()
    print("  NOTE: KV usage is ONE part of total VRAM (weights + CUDA graphs +")
    print("  activations are others, and are constant). This shows the VARIABLE")
    print("  part. If KV usage is flat, total VRAM is flat (only constant parts).")

    # -- JSON trace --
    out_path = args.out or f"vram-load-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint, "container": args.container, "model": model,
        "max_num_seqs": max_seqs, "concurrency": conc,
        "kv_pool_gib": pool_gib, "kv_pool_tokens": pool_tokens,
        "baseline_kv_pct": base_pct,
        "stages": [{k: v for k, v in s.items() if k != "trace"} | {"trace": s["trace"]}
                   for s in stages],
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"JSON trace: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
