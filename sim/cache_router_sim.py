#!/usr/bin/env python3
"""
Cache-aware routing simulator for LLM serving.

Compares six router strategies across N replicas, each holding a prefix (KV)
cache of fixed capacity with LRU eviction. Reports cache hit rate,
time-to-first-token (TTFT) and load imbalance.

Model (deliberately simple; see README for limits):
- Prompts are split into blocks of BLOCK tokens. Each full block is identified
  by a chained hash (previous block's hash + its own content), as in vLLM, so a
  block only matches when its whole history matches. The partial last block is
  never cached.
- Each replica runs prefill one request at a time, FIFO. Prefill time grows with
  the number of uncached tokens. Decode is assumed to overlap through batching
  and is not modelled. TTFT = queueing delay + prefill time.
- Workload: several products (system prompts) with Zipf popularity. Each user
  session is a multi-turn chat whose history grows on top of its system prompt.

Usage:
    python cache_router_sim.py
    python cache_router_sim.py --replicas 8 --rates 4,8,12 --csv results.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass

BLOCK = 16  # tokens per KV block


def stable_hash(*parts) -> int:
    """Deterministic 64-bit hash (Python's built-in hash is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "big")


# --------------------------------------------------------------------------- workload

@dataclass
class Request:
    t: float          # arrival time (s)
    session: int
    product: int
    tokens: int       # full prompt length
    hashes: list      # chained hashes of the prompt's full blocks


def generate_workload(args, rng: random.Random) -> list[Request]:
    # System prompts: one per product, Zipf popularity (product 0 is the hottest).
    sys_len = [rng.randint(args.sys_min, args.sys_max) for _ in range(args.products)]
    sys_chain = []
    for p, n_tok in enumerate(sys_len):
        chain, prev = [], 0
        for i in range(n_tok // BLOCK):
            prev = stable_hash("sys", p, i, prev)
            chain.append(prev)
        sys_chain.append(chain)
    weights = [1 / (k + 1) ** args.zipf for k in range(args.products)]

    # Sessions arrive as a Poisson process; each runs several turns.
    mean_turns = (1 + args.max_turns) / 2
    session_rate = args.rate / mean_turns
    requests: list[Request] = []
    t, sid = 0.0, 0
    while True:
        t += rng.expovariate(session_rate)
        if t > args.duration:
            break
        p = rng.choices(range(args.products), weights)[0]
        chain = list(sys_chain[p])      # this session's own hash chain, extended lazily
        history = sys_len[p]            # tokens already in the conversation
        turn_t = t
        for _ in range(rng.randint(1, args.max_turns)):
            user = rng.randint(30, 300)
            prompt_tokens = history + user
            full_blocks = prompt_tokens // BLOCK
            while len(chain) < full_blocks:  # blocks past the system prompt are session-specific
                chain.append(stable_hash("sess", sid, len(chain), chain[-1] if chain else 0))
            if turn_t <= args.duration:
                requests.append(Request(turn_t, sid, p, prompt_tokens, chain[:full_blocks]))
            history = prompt_tokens + rng.randint(100, 500)   # model's answer joins the history
            turn_t += rng.expovariate(1 / args.think_time)
        sid += 1
    requests.sort(key=lambda r: r.t)
    return requests


# --------------------------------------------------------------------------- replicas

class Replica:
    def __init__(self, capacity_blocks: int):
        self.cache: OrderedDict[int, None] = OrderedDict()  # LRU: front = evicted first
        self.capacity = capacity_blocks
        self.free_at = 0.0       # when the prefill queue drains
        self.busy_time = 0.0     # total prefill work done (s)

    def match(self, hashes: list) -> int:
        """Number of leading blocks already cached."""
        n = 0
        for h in hashes:
            if h not in self.cache:
                break
            n += 1
        return n

    def admit(self, hashes: list) -> None:
        # Touch deepest blocks first so the shared beginning ends up most recent:
        # leaves are evicted before the prefixes they hang off.
        for h in reversed(hashes):
            if h in self.cache:
                self.cache.move_to_end(h)
            else:
                self.cache[h] = None
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def backlog(self, t: float) -> float:
        return max(0.0, self.free_at - t)


def prefill_seconds(tokens: int, args) -> float:
    return args.overhead + tokens / args.prefill_tps


# --------------------------------------------------------------------------- routers

class Ring:
    """Consistent-hash ring with virtual nodes."""

    def __init__(self, n: int, vnodes: int = 100):
        pts = sorted((stable_hash("node", r, v), r) for r in range(n) for v in range(vnodes))
        self.keys = [k for k, _ in pts]
        self.owners = [r for _, r in pts]

    def get(self, key) -> int:
        i = bisect.bisect(self.keys, stable_hash("key", key)) % len(self.keys)
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
        return min(range(len(reps)), key=lambda r: reps[r].backlog(t))


class Sticky:
    name = "sticky-session"

    def __init__(self, n, args):
        self.ring = Ring(n)

    def route(self, req, t, reps):
        return self.ring.get(("session", req.session))


class PrefixHash:
    name = "prefix-hash"

    def __init__(self, n, args):
        self.ring = Ring(n)

    def route(self, req, t, reps):
        key = req.hashes[0] if req.hashes else ("product", req.product)
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
        for depth, h in enumerate(req.hashes, start=1):
            alive &= self.sent.get(h, set())
            if not alive:
                break
            for r in alive:
                best_len[r] = depth
        least = min(range(n), key=lambda r: reps[r].backlog(t))
        best = max(range(n), key=lambda r: (best_len[r], -reps[r].backlog(t)))
        ratio = best_len[best] / max(1, len(req.hashes))
        overloaded = reps[best].backlog(t) > reps[least].backlog(t) + self.slack
        choice = best if ratio >= self.threshold and not overloaded else least
        for h in req.hashes:
            self.sent.setdefault(h, set()).add(choice)
        return choice


class KVAware:
    """Router sees each replica's real cache and scores estimated TTFT."""
    name = "kv-aware-cost"

    def __init__(self, n, args):
        self.load_weight = args.kv_load_weight

    def route(self, req, t, reps):
        def cost(r):
            uncached = req.tokens - reps[r].match(req.hashes) * BLOCK
            return prefill_seconds(uncached, self.args) + self.load_weight * reps[r].backlog(t)
        return min(range(len(reps)), key=cost)


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
        cached = r.match(req.hashes) * BLOCK
        work = prefill_seconds(req.tokens - cached, args)
        start = max(req.t, r.free_at)
        r.free_at = start + work
        r.busy_time += work
        r.admit(req.hashes)
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
    ap.add_argument("--kv-load-weight", type=float, default=1.0, help="kv-aware: weight on queue backlog")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", help="write results to this CSV file")
    args = ap.parse_args()

    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        args.rate = rate
        requests = generate_workload(args, random.Random(args.seed))
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
