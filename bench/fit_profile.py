#!/usr/bin/env python3
"""Fit the simulator's cost model to one engine's benchmark CSV and score it on held-out runs.

    python bench/fit_profile.py bench/results/<model>__<engine>.csv

Model (sim/cache_router_sim.py: prefill_seconds):
    ttft = overhead + uncached_tokens / prefill_tps + cached_tokens / cached_tps

Fit: weighted least squares minimising *relative* error over per-point medians of the `cold`
(cache-miss length sweep) and `hitsweep` (cache-hit length sweep) experiments. Relative error
matters: an unweighted fit is dominated by the long prompts and gives a ~0 ms overhead that is
badly wrong for short ones. The `paired` and `shared` experiments are NOT used in the fit; they
are scored against it as held-out checks. Standard library only.
"""
import csv
import statistics
import sys

med = statistics.median


def load(path):
    return list(csv.DictReader(open(path, newline="")))


def group(rows, exp, kind=None):
    g = {}
    for r in rows:
        if r["experiment"] == exp and (kind is None or r["kind"] == kind):
            g.setdefault(int(r["point"]), []).append(r)
    return g


def summarize(g):
    """-> [(uncached_tokens, cached_tokens, median_ttft_s, n)] one row per sweep point."""
    out = []
    for rs in g.values():
        n = statistics.mean(int(r["prompt_tokens"]) for r in rs)
        c = statistics.mean(int(r["cached_tokens"]) for r in rs)
        out.append((n - c, c, med(float(r["ttft_s"]) for r in rs), len(rs)))
    return out


def solve(A, b):
    """Gauss-Jordan elimination for the small normal-equation system."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def main():
    rows = load(sys.argv[1])
    train = summarize(group(rows, "cold")) + summarize(group(rows, "hitsweep"))
    # minimise sum(((a + u*b + c*d) - y) / y)^2  ->  rows [1/y, u/y, c/y] against target 1
    X = [[1 / y, u / y, c / y] for u, c, y, _ in train]
    AtA = [[sum(r[i] * r[j] for r in X) for j in range(3)] for i in range(3)]
    Atb = [sum(r[i] for r in X) for i in range(3)]
    a, b, d = solve(AtA, Atb)
    pred = lambda u, c: a + u * b + c * d

    print(f'{{"prefill_tps": {1 / b:,.0f}, "overhead": {a:.4f}, "cached_tps": {1 / d:,.0f}}}')
    print(f"  ({a * 1000:.1f} ms fixed, {d * 1e6:.2f} us per cached token)\n")
    print("training points (per-point median TTFT):")
    errs = []
    for u, c, y, n in sorted(train, key=lambda t: (t[1], t[0])):
        e = (pred(u, c) - y) / y
        errs.append(abs(e))
        print(f"  uncached={u:6.0f} cached={c:6.0f} measured={y * 1000:6.1f}ms pred={pred(u, c) * 1000:6.1f}ms err={100 * e:+5.1f}%")
    print(f"  median |err| = {100 * med(errs):.1f}%   max = {100 * max(errs):.1f}%\n")

    print("held-out (not used in fit):")
    for exp, kind in (("shared", "hit"), ("paired", "miss"), ("paired", "hit")):
        for u, c, y, n in summarize(group(rows, exp, kind)):
            print(f"  {exp:6} {kind:4} n={n:3} uncached={u:6.0f} cached={c:6.0f} "
                  f"measured p50={y * 1000:6.1f}ms pred={pred(u, c) * 1000:6.1f}ms err={100 * (pred(u, c) - y) / y:+5.1f}%")


if __name__ == "__main__":
    main()
