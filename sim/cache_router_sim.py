#!/usr/bin/env python3
"""
Cache-aware routing simulator for LLM serving.

Compares six router strategies across N replicas, each holding a prefix (KV)
cache of fixed capacity with LRU eviction. Reports cache hit rate,
time-to-first-token (TTFT) and load imbalance.

Model (deliberately simple; see README for limits):
- Prompts are split into blocks of BLOCK tokens. Each full block gets a
  positional id: system-prompt blocks are keyed by (product, index) so every
  session of a product shares them; blocks past the system prompt are keyed
  by (session, index) so they're private to that session. A block matches
  only if every earlier block in the prompt also matches (vLLM-style chained
  identity). The partial last block is never cached.
- Each replica runs prefill one request at a time, FIFO. Prefill time grows with
  the number of uncached tokens. Decode is assumed to overlap through batching
  and is not modelled. TTFT = queueing delay + prefill time.
- Workload: several products (system prompts) with Zipf popularity. Each user
  session is a multi-turn chat whose history grows on top of its system prompt.

This engine is a deterministic port of the JS engine in index.html: same
PRNG (mulberry32), same hash function (for the consistent-hash ring), same
block-identity formula, same draw order. Given the same config and seed, the
two engines produce numerically identical results (see tools/parity_check.*).

Usage:
    python cache_router_sim.py
    python cache_router_sim.py --replicas 8 --rates 4,8,12 --csv results.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass

BLOCK = 16  # tokens per KV block
SESS_BASE = 1_000_000_000


# --------------------------------------------------------------------------- RNG (mulberry32, ported from index.html's `rng`)

def _imul32(a: int, b: int) -> int:
    """Emulates JS Math.imul: 32-bit wraparound multiply. Sign-agnostic: the
    low 32 bits of a product are the same whether operands are read as signed
    or unsigned, so plain unsigned arithmetic reproduces the same bit pattern."""
    return (a * b) & 0xFFFFFFFF


def make_rng(seed: int):
    """Direct port of index.html's `rng(seed)` (mulberry32). Bit-identical
    output to the JS version for the same seed and call count."""
    state = [seed & 0xFFFFFFFF]

    def r() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        a = state[0]
        t = _imul32(a ^ (a >> 15), (1 | a) & 0xFFFFFFFF)
        old_t = t
        t = (t + _imul32(t ^ (t >> 7), (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t = t ^ old_t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return r


def _h32(a: int, b: int) -> int:
    """Direct port of index.html's `h32` (used only for the consistent-hash
    ring, never for block identity)."""
    x = (_imul32((a & 0xFFFFFFFF) ^ 0x9E3779B9, 0x85EBCA6B) ^ (b & 0xFFFFFFFF)) & 0xFFFFFFFF
    x ^= x >> 13
    x &= 0xFFFFFFFF
    x = _imul32(x, 0xC2B2AE35)
    x ^= x >> 16
    return x & 0xFFFFFFFF


def _js_expo(r, rate: float) -> float:
    import math
    return -math.log(1 - r()) / rate


def _js_randint(r, a: int, b: int) -> int:
    return a + int(r() * (b - a + 1))


def _bh(sys_full: int, product: int, session: int, i: int) -> int:
    """Block identity: a plain positional id, not a content hash. System-prompt
    blocks are shared across every session of the product; later blocks are
    private to the session."""
    if i < sys_full:
        return product * 100_000 + i + 1
    return SESS_BASE + session * 10_000 + i


# --------------------------------------------------------------------------- workload

@dataclass
class Request:
    t: float          # arrival time (s)
    session: int
    product: int
    tokens: int       # full prompt length
    sys_full: int     # number of full system-prompt blocks
    n: int            # number of full blocks in the prompt


def generate_workload(args) -> list[Request]:
    r = make_rng(args.seed)
    products = args.products

    sys_len = [_js_randint(r, args.sys_min, args.sys_max) for _ in range(products)]

    w = []
    ws = 0.0
    for k in range(products):
        v = 1 / pow(k + 1, args.zipf)
        w.append(v)
        ws += v

    def pick() -> int:
        x = r() * ws
        for k in range(products):
            x -= w[k]
            if x <= 0:
                return k
        return products - 1

    session_rate = args.rate / ((1 + args.max_turns) / 2)
    requests: list[Request] = []
    t, sid = 0.0, 0
    while True:
        t += _js_expo(r, session_rate)
        if t > args.duration:
            break
        p = pick()
        history = sys_len[p]
        turn_t = t
        turns = _js_randint(r, 1, args.max_turns)
        for _ in range(turns):
            tok = history + _js_randint(r, 30, 300)
            if turn_t <= args.duration:
                requests.append(Request(turn_t, sid, p, tok, sys_len[p] // BLOCK, tok // BLOCK))
            history = tok + _js_randint(r, 100, 500)
            turn_t += _js_expo(r, 1 / args.think_time)
        sid += 1
    requests.sort(key=lambda req: req.t)
    return requests


# --------------------------------------------------------------------------- replicas

class Replica:
    def __init__(self, capacity_blocks: int):
        self.cache: dict[int, None] = {}  # insertion order = LRU order; front = evicted first
        self.capacity = capacity_blocks
        self.free_at = 0.0       # when the prefill queue drains
        self.busy_time = 0.0     # total prefill work done (s)

    def match(self, req: Request) -> int:
        """Number of leading blocks already cached."""
        n = 0
        for i in range(req.n):
            if _bh(req.sys_full, req.product, req.session, i) not in self.cache:
                break
            n += 1
        return n

    def admit(self, req: Request) -> None:
        # Touch deepest blocks first so the shared beginning ends up most recent:
        # leaves are evicted before the prefixes they hang off.
        for i in range(req.n - 1, -1, -1):
            h = _bh(req.sys_full, req.product, req.session, i)
            if h in self.cache:
                del self.cache[h]
            self.cache[h] = None
        while len(self.cache) > self.capacity:
            self.cache.pop(next(iter(self.cache)))

    def backlog(self, t: float) -> float:
        return max(0.0, self.free_at - t)


def prefill_seconds(tokens: int, args) -> float:
    return args.overhead + tokens / args.prefill_tps


def _least_loaded(t: float, reps: list[Replica]) -> int:
    best = 0
    for r in range(1, len(reps)):
        if reps[r].backlog(t) < reps[best].backlog(t):
            best = r
    return best


# --------------------------------------------------------------------------- routers

class Ring:
    """Consistent-hash ring with virtual nodes."""

    def __init__(self, n: int, vnodes: int = 100):
        pts = sorted((_h32(r * 131 + 7, v * 977 + 3), r) for r in range(n) for v in range(vnodes))
        self.keys = [k for k, _ in pts]
        self.owners = [r for _, r in pts]

    def get(self, key: int) -> int:
        i = bisect.bisect(self.keys, _h32(key, 0x51ED)) % len(self.keys)
        return self.owners[i]


class RoundRobin:
    name = "round-robin"

    def __init__(self, n, args):
        self.i, self.n = 0, n

    def route(self, req, t, reps):
        self.i = (self.i + 1) % self.n
        return self.i


class LeastLoaded:
    name = "least-loaded"

    def __init__(self, n, args):
        pass

    def route(self, req, t, reps):
        return _least_loaded(t, reps)


class Sticky:
    name = "sticky-session"

    def __init__(self, n, args):
        self.ring = Ring(n)

    def route(self, req, t, reps):
        return self.ring.get(req.session)


class PrefixHash:
    name = "prefix-hash"

    def __init__(self, n, args):
        self.ring = Ring(n)

    def route(self, req, t, reps):
        key = _bh(req.sys_full, req.product, req.session, 0)
        return self.ring.get(key)


class ApproxTree:
    """Router remembers which replica it sent each block to (never sees evictions)."""
    name = "approx-prefix-tree"

    def __init__(self, n, args):
        self.sent: dict[int, set] = {}
        self.threshold = args.match_threshold
        self.slack = args.balance_slack

    def route(self, req, t, reps):
        n = len(reps)
        alive, best_len = set(range(n)), {r: 0 for r in range(n)}
        for depth in range(1, req.n + 1):
            h = _bh(req.sys_full, req.product, req.session, depth - 1)
            alive &= self.sent.get(h, set())
            if not alive:
                break
            for r in alive:
                best_len[r] = depth
        least = _least_loaded(t, reps)
        best = 0
        for r in range(1, n):
            if best_len[r] > best_len[best] or (best_len[r] == best_len[best] and reps[r].backlog(t) < reps[best].backlog(t)):
                best = r
        ratio = best_len[best] / max(1, req.n)
        overloaded = reps[best].backlog(t) > reps[least].backlog(t) + self.slack
        choice = best if ratio >= self.threshold and not overloaded else least
        for i in range(req.n):
            h = _bh(req.sys_full, req.product, req.session, i)
            self.sent.setdefault(h, set()).add(choice)
        return choice


class KVAware:
    """Router sees each replica's real cache and scores estimated TTFT."""
    name = "kv-aware-cost"

    def __init__(self, n, args):
        pass

    def route(self, req, t, reps):
        args = self.args
        best, best_cost = 0, float("inf")
        for r in range(len(reps)):
            uncached = req.tokens - reps[r].match(req) * BLOCK
            cost = prefill_seconds(uncached, args) + reps[r].backlog(t)
            if cost < best_cost:
                best_cost, best = cost, r
        return best


STRATEGIES = [RoundRobin, LeastLoaded, Sticky, PrefixHash, ApproxTree, KVAware]


# --------------------------------------------------------------------------- simulation

def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def simulate(strategy_cls, requests: list[Request], args) -> dict:
    reps = [Replica(args.cache_tokens // BLOCK) for _ in range(args.replicas)]
    router = strategy_cls(args.replicas, args)
    router.args = args
    ttfts, cached_total, prompt_total = [], 0, 0
    for req in requests:
        r = reps[router.route(req, req.t, reps)]
        cached = r.match(req) * BLOCK
        work = prefill_seconds(req.tokens - cached, args)
        start = max(req.t, r.free_at)
        r.free_at = start + work
        r.busy_time += work
        r.admit(req)
        ttfts.append(r.free_at - req.t)
        cached_total += cached
        prompt_total += req.tokens
    busy = [r.busy_time for r in reps]
    mean_busy = sum(busy) / len(busy)
    return {
        "strategy": strategy_cls.name,
        "rate": args.rate,
        "hit_rate": cached_total / prompt_total,
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p99": percentile(ttfts, 0.99),
        "imbalance": max(busy) / mean_busy if mean_busy else 0.0,
        "utilization": mean_busy / args.duration,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--cache-tokens", type=int, default=60_000, help="KV cache capacity per replica (tokens)")
    ap.add_argument("--rates", default="6,14,20", help="comma-separated request rates to test (req/s, whole fleet)")
    ap.add_argument("--duration", type=float, default=900, help="simulated seconds")
    ap.add_argument("--products", type=int, default=6, help="distinct system prompts")
    ap.add_argument("--zipf", type=float, default=1.2, help="popularity skew; higher = hotter top prompt")
    ap.add_argument("--sys-min", type=int, default=1_000)
    ap.add_argument("--sys-max", type=int, default=6_000)
    ap.add_argument("--max-turns", type=int, default=9)
    ap.add_argument("--think-time", type=float, default=20, help="mean seconds between turns")
    ap.add_argument("--prefill-tps", type=float, default=8_000, help="prefill tokens/s per replica")
    ap.add_argument("--overhead", type=float, default=0.02, help="fixed seconds per request")
    ap.add_argument("--match-threshold", type=float, default=0.5, help="approx tree: min prefix match ratio")
    ap.add_argument("--balance-slack", type=float, default=1.0, help="approx tree: max extra backlog (s)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", help="write results to this CSV file")
    args = ap.parse_args()

    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        args.rate = rate
        requests = generate_workload(args)
        print(f"\nrate {rate:g} req/s · {len(requests)} requests · {args.replicas} replicas · "
              f"{args.cache_tokens:,} cache tokens each")
        print(f"{'strategy':<20}{'hit rate':>9}{'TTFT p50':>10}{'TTFT p99':>10}{'imbalance':>11}{'util':>7}")
        for cls in STRATEGIES:
            row = simulate(cls, requests, args)
            rows.append(row)
            print(f"{row['strategy']:<20}{row['hit_rate']:>9.1%}{row['ttft_p50']:>9.2f}s"
                  f"{row['ttft_p99']:>9.2f}s{row['imbalance']:>10.2f}x{row['utilization']:>7.0%}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
