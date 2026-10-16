#!/usr/bin/env python3
"""Measure TTFT against an OpenAI-compatible completions server (vLLM or SGLang).

Standard library only. Three experiments, each appending rows to one CSV:

  cold    unique prompts of increasing length, one at a time. No prefix can hit,
          so TTFT = overhead + n / prefill_tps. Fit gives the simulator's two constants.
  shared  one long shared system prompt + short unique question, one at a time.
          Run against a server with prefix caching on, then off.
  load    same shared-prefix workload with N concurrent clients; reports throughput.
  hitsweep  cache-hit TTFT at several cached lengths with a fixed short tail.
  paired  alternates a cache hit and a matched-length cache miss in one time window, so
          drift (thermal, clocks, GPU sharing with Windows) cannot bias the comparison.

Every request uses max_tokens=1, so TTFT is essentially prefill time.
"""
import argparse
import csv
import http.client
import json
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import date
from urllib.parse import urlparse

WORDS = (
    "router cache prefix replica token block queue prefill decode latency throughput "
    "session product system prompt memory eviction hash layer attention batch stream "
    "request worker scheduler policy balance hotspot shard tenant context window "
    "kernel tensor buffer socket thread signal metric sample window cluster node"
).split()


def make_text(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def stream_ttft(host, port, model, prompt, timeout=300):
    """Return (ttft_seconds, prompt_tokens, cached_tokens or None)."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    ttft = None
    usage = {}
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if ttft is None and obj.get("choices") and obj["choices"][0].get("text") is not None:
            ttft = time.perf_counter() - t0
        if obj.get("usage"):
            usage = obj["usage"]
    conn.close()
    if ttft is None:
        ttft = time.perf_counter() - t0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return ttft, usage.get("prompt_tokens"), cached


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True).strip()
        return out
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True, help="e.g. vllm-0.x.y or sglang-0.x.y")
    ap.add_argument("--prefix-caching", choices=["on", "off"], required=True)
    ap.add_argument("--experiment", choices=["cold", "shared", "load", "paired", "hitsweep"], required=True)
    ap.add_argument("--hit-words", default="400,800,1600,2300,3400,4600", help="hitsweep: shared prefix sizes in words")
    ap.add_argument("--out", required=True, help="CSV file to append to")
    ap.add_argument("--n", type=int, default=30, help="requests per point (shared/load: total requests)")
    ap.add_argument("--lengths", default="256,512,1024,2048,3000,4096,6000", help="cold sweep prompt lengths (approx tokens)")
    ap.add_argument("--sys-words", type=int, default=2300, help="shared system prompt size in words (~3000 tokens)")
    ap.add_argument("--clients", type=int, default=8, help="concurrency for load experiment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    u = urlparse(args.url)
    host, port = u.hostname, u.port or 80
    rng = random.Random(args.seed)
    rows = []
    meta = {
        "date": date.today().isoformat(), "gpu": gpu_info(), "engine": args.engine,
        "model": args.model, "prefix_caching": args.prefix_caching,
        "experiment": args.experiment, "host_os": platform.platform(),
    }

    # Warm up kernels / CUDA graphs so the first measured point isn't an outlier.
    for _ in range(args.warmup):
        stream_ttft(host, port, args.model, make_text(rng, 200))

    if args.experiment == "cold":
        # ~1.3 tokens per word for this vocabulary; actual count comes back in usage.
        # Jobs are shuffled so thermal/clock drift is spread across lengths, not confounded with them.
        jobs = [(int(t), i) for t in args.lengths.split(",") for i in range(args.n)]
        rng.shuffle(jobs)
        for target, i in jobs:
            prompt = f"u{rng.getrandbits(48):x} " + make_text(rng, int(target / 1.3))
            ttft, ptoks, cached = stream_ttft(host, port, args.model, prompt)
            rows.append({**meta, "point": target, "i": i, "kind": "miss", "prompt_tokens": ptoks,
                         "cached_tokens": cached or 0, "ttft_s": round(ttft, 5)})
    elif args.experiment == "hitsweep":
        # Cache-hit TTFT vs cached length, with a fixed short unique tail. Distinguishes a cost
        # proportional to cached tokens from a flat per-request hit cost. Order is shuffled.
        words = [int(x) for x in args.hit_words.split(",")]
        systems = {w: f"SYSTEM{w}: " + make_text(rng, w) + "\n" for w in words}
        for w in words:
            stream_ttft(host, port, args.model, systems[w] + "prime")
        jobs = [(w, i) for w in words for i in range(args.n)]
        rng.shuffle(jobs)
        for w, i in jobs:
            q = f"Q{rng.getrandbits(32):x}: " + make_text(rng, 20)
            ttft, ptoks, cached = stream_ttft(host, port, args.model, systems[w] + q)
            rows.append({**meta, "point": w, "i": i, "kind": "hit", "prompt_tokens": ptoks,
                         "cached_tokens": cached or 0, "ttft_s": round(ttft, 5)})
    elif args.experiment == "paired":
        # Hit and miss at matched length, alternating in random order inside one time window.
        # A unique prompt on a caching server is a cache miss, so no second server is needed
        # and drift between runs cannot bias the hit/miss comparison.
        system = "SYSTEM: " + make_text(rng, args.sys_words) + "\n"
        stream_ttft(host, port, args.model, system + "prime")
        for i in range(args.n):
            hit = system + f"Q{rng.getrandbits(32):x}: " + make_text(rng, 20)
            miss = f"u{rng.getrandbits(48):x} " + make_text(rng, args.sys_words + 20)
            order = [("hit", hit), ("miss", miss)]
            rng.shuffle(order)
            for kind, prompt in order:
                ttft, ptoks, cached = stream_ttft(host, port, args.model, prompt)
                rows.append({**meta, "point": 0, "i": i, "kind": kind, "prompt_tokens": ptoks,
                             "cached_tokens": cached or 0, "ttft_s": round(ttft, 5)})
    else:
        system = "SYSTEM: " + make_text(rng, args.sys_words) + "\n"
        questions = [f"Q{rng.getrandbits(32):x}: " + make_text(rng, 20) for _ in range(args.n)]
        # Prime the prefix once (not recorded) so every recorded request is a steady-state hit.
        stream_ttft(host, port, args.model, system + "prime")

        if args.experiment == "shared":
            for i, q in enumerate(questions):
                ttft, ptoks, cached = stream_ttft(host, port, args.model, system + q)
                rows.append({**meta, "point": 0, "i": i, "kind": "hit" if args.prefix_caching == "on" else "miss",
                             "prompt_tokens": ptoks, "cached_tokens": cached or 0, "ttft_s": round(ttft, 5)})
        else:  # load
            lock = threading.Lock()
            queue = list(enumerate(questions))
            t_start = time.perf_counter()

            def worker():
                while True:
                    with lock:
                        if not queue:
                            return
                        i, q = queue.pop()
                    ttft, ptoks, cached = stream_ttft(host, port, args.model, system + q)
                    with lock:
                        rows.append({**meta, "point": args.clients, "i": i,
                                     "kind": "hit" if args.prefix_caching == "on" else "miss",
                                     "prompt_tokens": ptoks, "cached_tokens": cached or 0, "ttft_s": round(ttft, 5)})

            threads = [threading.Thread(target=worker) for _ in range(args.clients)]
            for t in threads: t.start()
            for t in threads: t.join()
            wall = time.perf_counter() - t_start
            for r in rows:
                r["throughput_rps"] = round(len(rows) / wall, 3)

    fields = list(rows[0].keys())
    for r in rows:
        for k in fields:
            r.setdefault(k, "")
    try:
        with open(args.out, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    except FileExistsError:
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)

    # Console summary.
    groups = {}
    for r in rows:
        groups.setdefault((r["point"], r["kind"]), []).append(r)
    print(f"{args.engine} cache={args.prefix_caching} {args.experiment}")
    for (point, kind), rs in sorted(groups.items()):
        t = [r["ttft_s"] for r in rs]
        pt = statistics.mean(r["prompt_tokens"] or 0 for r in rs)
        ct = statistics.mean(r["cached_tokens"] for r in rs)
        print(f"  point={point:>5} {kind:>4} n={len(rs):>3} prompt_tok={pt:7.0f} cached_tok={ct:7.0f} "
              f"ttft p50={pct(t,50)*1000:7.1f}ms p99={pct(t,99)*1000:7.1f}ms")


if __name__ == "__main__":
    sys.exit(main())
