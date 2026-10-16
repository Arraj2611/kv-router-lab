#!/usr/bin/env python3
"""
Compares the Python engine against the JS engine using the shared config in
tools/parity_config.json. Two stages:

1. Golden sequence: the first 1000 raw RNG draws and the first 200 generated
   requests must match bit-for-bit (integer fields exactly, `t` within a
   tight float tolerance -- see note below). This isolates "did the RNG/
   workload port go wrong" from "do the aggregate metrics agree", and reports
   the first diverging index so a real divergence is easy to find.
2. Metrics: hit rate, TTFT p50/p99 and load imbalance must match within 1e-6
   relative, for every strategy at every configured rate. Strategy ranking
   (by p99 and by hit rate) is also checked, as a second, coarser signal.

Run `node tools/parity_check.js` first -- it writes golden_js.json and
parity_js.json from the JS engine for this script to compare against.

Note on `t`: workload generation calls Math.log()/math.log() once per
inter-arrival or think-time draw. The RNG's raw integer output is bit-exact
across languages (verified by stage 1), but log() itself can differ by a
ULP between V8 and CPython's libm. That shows up only in `t` (a continuous
arrival time), never in which product/turn-count/token-count gets picked
(those come straight from the exact RNG output), so it can't flip any
routing decision. `t` is compared with a tolerance tighter than the 1e-6
used for aggregate metrics, not exact equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
import cache_router_sim as sim  # noqa: E402

TOOLS = ROOT / "tools"
T_TOL = 1e-9       # arrival-time tolerance (log() ULP noise), relative
METRIC_TOL = 1e-6  # aggregate-metric tolerance, relative

JS_KEY = {"rr": "round_robin", "least": "least_loaded", "sticky": "sticky",
          "prefix": "prefix_hash", "approx": "approx_tree", "kv": "kv_aware"}
PY_KEY = {"round-robin": "round_robin", "least-loaded": "least_loaded", "sticky-session": "sticky",
          "prefix-hash": "prefix_hash", "approx-prefix-tree": "approx_tree", "kv-aware-cost": "kv_aware"}


def load_config() -> dict:
    return json.loads((TOOLS / "parity_config.json").read_text())


def make_args(cfg: dict, rate: float, cpu_tokens: int = 0, cache_tokens: int | None = None,
              duration: float | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        cpu_tokens=cpu_tokens, reload_tps=cfg["reloadTps"],
        replicas=cfg["replicas"], cache_tokens=cache_tokens or cfg["cacheTokens"], duration=duration or cfg["duration"],
        products=cfg["products"], zipf=cfg["zipf"], sys_min=cfg["sysMin"], sys_max=cfg["sysMax"],
        max_turns=cfg["maxTurns"], think_time=cfg["think"], prefill_tps=cfg["tps"], overhead=cfg["overhead"], cached_tps=cfg["cachedTps"],
        match_threshold=cfg["thresh"], balance_slack=cfg["slack"], seed=cfg["seed"], rate=rate,
    )


def rel_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def check_golden(cfg: dict) -> bool:
    golden_path = TOOLS / "golden_js.json"
    if not golden_path.exists():
        print(f"missing {golden_path} -- run `node tools/parity_check.js` first", file=sys.stderr)
        return False
    golden = json.loads(golden_path.read_text())

    r = sim.make_rng(cfg["seed"])
    py_rvals = [r() for _ in range(1000)]
    for i, (a, b) in enumerate(zip(py_rvals, golden["rVals"])):
        if a != b:
            print(f"GOLDEN MISMATCH: r() diverges at draw {i}: py={a!r} js={b!r}", file=sys.stderr)
            return False
    print(f"golden RNG: {len(py_rvals)} draws match exactly.")

    args = make_args(cfg, cfg["rates"][0])
    py_reqs = sim.generate_workload(args)[:200]
    js_reqs = golden["requests"]
    if len(py_reqs) != len(js_reqs):
        print(f"GOLDEN MISMATCH: request count differs: py={len(py_reqs)} js={len(js_reqs)}", file=sys.stderr)
        return False
    for i, (pr, jr) in enumerate(zip(py_reqs, js_reqs)):
        for field, py_val, js_val in [
            ("sid", pr.session, jr["sid"]), ("p", pr.product, jr["p"]),
            ("tokens", pr.tokens, jr["tokens"]), ("sysFull", pr.sys_full, jr["sysFull"]), ("n", pr.n, jr["n"]),
        ]:
            if py_val != js_val:
                print(f"GOLDEN MISMATCH: request {i} field '{field}' diverges: py={py_val!r} js={js_val!r}", file=sys.stderr)
                return False
        if not rel_close(pr.t, jr["t"], T_TOL):
            print(f"GOLDEN MISMATCH: request {i} field 't' diverges beyond tolerance: py={pr.t!r} js={jr['t']!r}", file=sys.stderr)
            return False
    print(f"golden requests: first {len(py_reqs)} match (fields exact, t within {T_TOL:g} relative).")

    # the example trace, parsed by the Python parser, against what the JS parser produced
    trace_reqs, info = sim.parse_trace((ROOT / cfg["traceFile"]).read_text(encoding="utf-8"))
    js_trace = golden["traceRequests"]
    if len(trace_reqs) != len(js_trace):
        print(f"GOLDEN MISMATCH: trace request count differs: py={len(trace_reqs)} js={len(js_trace)}", file=sys.stderr)
        return False
    for i, (pr, jr) in enumerate(zip(trace_reqs, js_trace)):
        for field, py_val, js_val in [
            ("sid", pr.session, jr["sid"]), ("p", pr.product, jr["p"]), ("tokens", pr.tokens, jr["tokens"]),
            ("sysFull", pr.sys_full, jr["sysFull"]), ("n", pr.n, jr["n"]), ("t", pr.t, jr["t"]),
        ]:
            if py_val != js_val:
                print(f"GOLDEN MISMATCH: trace request {i} field '{field}' diverges: py={py_val!r} js={js_val!r}", file=sys.stderr)
                return False
    jinfo = golden["traceInfo"]
    for key_py, key_js in (("sessions", "sessions"), ("prefixes", "prefixes"), ("duration", "duration"),
                           ("fallback_prefixes", "fallbackPrefixes")):
        if info[key_py] != jinfo[key_js]:
            print(f"GOLDEN MISMATCH: trace info '{key_py}': py={info[key_py]!r} js={jinfo[key_js]!r}", file=sys.stderr)
            return False
    print(f"golden trace: all {len(trace_reqs)} parsed requests match exactly ({info['sessions']} sessions, "
          f"{info['prefixes']} prefixes).")
    return True


def run_python(cfg: dict) -> list[dict]:
    rows = []
    trace_reqs, info = sim.parse_trace((ROOT / cfg["traceFile"]).read_text(encoding="utf-8"))
    for cpu in cfg["cpuTokensList"]:
        for rate in cfg["rates"]:
            args = make_args(cfg, rate, cpu)
            requests = sim.generate_workload(args)
            for cls in sim.STRATEGIES:
                rows.append({**sim.simulate(cls, requests, args), "cpu": cpu, "wl": "syn"})
        trate = len(trace_reqs) / info["duration"]
        targs = make_args(cfg, trate, cpu, cache_tokens=cfg["traceCacheTokens"], duration=info["duration"])
        for cls in sim.STRATEGIES:
            rows.append({**sim.simulate(cls, trace_reqs, targs), "cpu": cpu, "wl": "trace"})
    return rows


def ranking(rows: list[dict], key_map: dict, metric: str, reverse: bool) -> list[str]:
    ranked = sorted(rows, key=lambda row: row[metric], reverse=reverse)
    return [key_map[row.get("strategy") or row.get("id")] for row in ranked]


def main() -> int:
    cfg = load_config()

    if not check_golden(cfg):
        return 1

    js_path = TOOLS / "parity_js.json"
    if not js_path.exists():
        print(f"missing {js_path} -- run `node tools/parity_check.js` first", file=sys.stderr)
        return 2
    js_rows = json.loads(js_path.read_text())
    py_rows = run_python(cfg)

    problems = []

    # exact-ish metric comparison, matched by (rate, canonical strategy key)
    py_by_key = {(row["wl"], row["rate"], row["cpu"], PY_KEY[row["strategy"]]): row for row in py_rows}
    js_by_key = {(row["wl"], row["rate"], row["cpu"], JS_KEY[row["id"]]): row for row in js_rows}
    if len(js_by_key) != len(py_by_key):
        problems.append(f"row count differs: js={len(js_by_key)} py={len(py_by_key)}")
    print(f"\n{'wl':<7}{'rate':<8}{'cpu':<7}{'strategy':<14}{'metric':<14}{'js':>14}{'py':>14}  status")
    for (wl, rate, cpu, key), js_row in sorted(js_by_key.items()):
        py_row = py_by_key.get((wl, rate, cpu, key))
        if py_row is None:
            problems.append(f"{wl} rate={rate} cpu={cpu} {key}: no matching Python row")
            continue
        metrics = [("hit_rate", "hit_rate"), ("ttft_p50", "ttft_p50"), ("ttft_p99", "ttft_p99"), ("imbalance", "imbalance")]
        if cpu:  # the CPU-tier metric only exists (and is only compared) when the tier is on
            metrics.append(("cpu_hit_share", "cpu_hit_share"))
        for metric, py_field in metrics:
            a, b = js_row[metric], py_row[py_field]
            ok = rel_close(a, b, METRIC_TOL)
            print(f"{wl:<7}{rate:<8.2f}{cpu:<7}{key:<14}{metric:<14}{a:>14.6f}{b:>14.6f}  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"{wl} rate={rate} cpu={cpu} {key} {metric}: js={a!r} py={b!r} (exceeds {METRIC_TOL:g} relative)")

    # ranking, as a coarser secondary signal, per (workload, rate, cpu) group
    groups = sorted({(row["wl"], row["rate"], row["cpu"]) for row in js_rows})
    for wl, rate, cpu in groups:
        js_rate = [row for row in js_rows if (row["wl"], row["rate"], row["cpu"]) == (wl, rate, cpu)]
        py_rate = [row for row in py_rows if (row["wl"], row["rate"], row["cpu"]) == (wl, rate, cpu)]
        for js_field, py_field, reverse, label in [
            ("ttft_p99", "ttft_p99", False, "p99 TTFT"),
            ("hit_rate", "hit_rate", True, "hit rate"),
        ]:
            js_rank = ranking(js_rate, JS_KEY, js_field, reverse)
            py_rank = ranking(py_rate, PY_KEY, py_field, reverse)
            if js_rank != py_rank:
                problems.append(f"{wl} rate={rate} cpu={cpu} {label} ranking: js={js_rank} py={py_rank}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\nengines match within tolerance on every metric and ranking at all tested rates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
