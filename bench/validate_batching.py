#!/usr/bin/env python3
"""Score two single-replica prefill queue models against a measured open-loop run.

    python bench/validate_batching.py bench/results/<model>__<engine>__openloop.csv --profile rtx4070-qwen1.5b-vllm

Inputs are the *measured* arrival times, prompt lengths and cached-token counts, so this tests the
queue/prefill model only, not cache state. Models:

  fifo     the simulator's current model: one request at a time, work = prefill_seconds(...).
  batched  stepwise: each step takes up to `budget` uncached tokens from the head of the FIFO queue
           (a request can be split across steps, several can share one step). Step time =
           step_overhead + step_tokens/prefill_tps + cached_tokens_of_newly_started/cached_tps.
           The profile's fixed `overhead` is split: fraction f blocks the GPU once per step, the
           rest is added to each request's TTFT without blocking (tokenisation, HTTP, sampling).

Standard library only.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import cache_router_sim as sim  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def fifo(reqs, a):
    free, out = 0.0, []
    for arr, unc, cac in reqs:
        free = max(arr, free) + sim.prefill_seconds(unc, cac, a)
        out.append(free - arr)
    return out


def batched(reqs, a, budget, f):
    step_oh, req_oh = a.overhead * f, a.overhead * (1 - f)
    n = len(reqs)
    rem = [r[1] for r in reqs]
    started = [False] * n
    ttft = [0.0] * n
    queue, nxt, t = [], 0, 0.0
    while nxt < n or queue:
        while nxt < n and reqs[nxt][0] <= t:
            queue.append(nxt); nxt += 1
        if not queue:
            t = reqs[nxt][0]
            continue
        tokens, cached_cost, finishing = 0, 0.0, []
        for i in queue:
            take = min(rem[i], budget - tokens)
            if not started[i]:
                started[i] = True
                if a.cached_tps:
                    cached_cost += reqs[i][2] / a.cached_tps
            rem[i] -= take
            tokens += take
            if rem[i] == 0:
                finishing.append(i)
            if tokens >= budget:
                break
        t += step_oh + tokens / a.prefill_tps + cached_cost
        for i in finishing:
            ttft[i] = t + req_oh - reqs[i][0]
            queue.remove(i)
    return ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--profile", default="rtx4070-qwen1.5b-vllm", choices=sorted(sim.PROFILES))
    ap.add_argument("--budgets", default="2048,8192")
    ap.add_argument("--splits", default="0,0.5,1", help="fraction f of fixed overhead that blocks the GPU per step")
    args = ap.parse_args()

    base = sim.PROFILES[args.profile]
    rows = list(csv.DictReader(open(args.csv, newline="")))
    by_rate = {}
    for r in rows:
        by_rate.setdefault(float(r["point"]), []).append(r)
    data = []
    for rate in sorted(by_rate):
        rs = sorted(by_rate[rate], key=lambda r: float(r["arrival_s"]))
        reqs = [(float(r["arrival_s"]), int(r["prompt_tokens"]) - int(r["cached_tokens"]), int(r["cached_tokens"])) for r in rs]
        data.append((rate, reqs, [float(r["ttft_s"]) for r in rs]))

    def profile(k):
        """All service times multiplied by k (k=1 is the sequential-request profile as fitted)."""
        return type("A", (), {"overhead": base["overhead"] * k, "prefill_tps": base["prefill_tps"] / k,
                              "cached_tps": base["cached_tps"] / k})()

    variants = [("fifo", lambda reqs, a: fifo(reqs, a))]
    for b in [int(x) for x in args.budgets.split(",")]:
        for f in [float(x) for x in args.splits.split(",")]:
            variants.append((f"batch B={b} f={f:g}", lambda reqs, a, b=b, f=f: batched(reqs, a, b, f)))

    def cell_errors(fn, k):
        a, out = profile(k), []
        for rate, reqs, meas in data:
            p = fn(reqs, a)
            for q in (0.5, 0.99):
                m = pct(meas, q)
                out.append((pct(p, q) - m) / m)
        return out

    ks = [round(0.8 + 0.05 * i, 2) for i in range(0, 41)]  # 0.8 .. 2.8
    print("Each variant at its own best slowdown k (all service times x k; k=1 = sequential profile as fitted).")
    print("Score = median |relative error| over rate x {p50,p99} cells.\n")
    best = {}
    for name, fn in variants:
        scored = [(statistics.median(abs(e) for e in cell_errors(fn, k)), k) for k in ks]
        best[name] = min(scored)
        err, k = best[name]
        at1 = statistics.median(abs(e) for e in cell_errors(fn, 1.0))
        print(f"  {name:<20} best k={k:<5} median |err|={100 * err:5.1f}%   (at k=1: {100 * at1:5.1f}%)")

    winner = min(best, key=lambda n: best[n][0])
    for name in ("fifo", winner) if winner != "fifo" else ("fifo",):
        k = best[name][1]
        fn = dict(variants)[name]
        errs = cell_errors(fn, k)
        a = profile(k)
        print(f"\n{name} at k={k}:")
        print(f"  {'rate':>5} {'stat':>4} {'measured':>9} {'model':>9} {'err':>6}")
        i = 0
        for rate, reqs, meas in data:
            p = fn(reqs, a)
            for stat, q in (("p50", 0.5), ("p99", 0.99)):
                print(f"  {rate:>5g} {stat:>4} {pct(meas, q) * 1000:7.0f}ms {pct(p, q) * 1000:7.0f}ms {100 * errs[i]:+5.0f}%")
                i += 1


if __name__ == "__main__":
    main()
