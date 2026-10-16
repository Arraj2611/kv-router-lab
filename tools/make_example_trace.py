#!/usr/bin/env python3
"""Generate docs/example_trace.jsonl: a small SYNTHETIC trace in the simulator's trace format.

It is not real traffic. It exists so the format, the parser and the parity check have something to
run on. Three shared prompts, Poisson session arrivals, 1-5 turns per session with growing prompts.
Standard library only; deterministic (seed 42).

    python tools/make_example_trace.py > docs/example_trace.jsonl
"""
import json
import random
import sys

rng = random.Random(42)
# Prefix lengths deliberately not multiples of the 16-token block, so the parsers' rounding is exercised.
PREFIXES = [("assistant-a", 1213, 0.5), ("assistant-b", 2405, 0.3), ("assistant-c", 3598, 0.2)]
DURATION = 40.0       # seconds of session arrivals
SESSION_RATE = 3.0    # new sessions per second

rows = []
t, sid = 0.0, 0
while True:
    t += rng.expovariate(SESSION_RATE)
    if t > DURATION:
        break
    name, plen, _ = rng.choices(PREFIXES, weights=[p[2] for p in PREFIXES])[0]
    history, turn_t = plen, t
    for _ in range(rng.randint(1, 5)):
        tokens = history + rng.randint(30, 250)
        rows.append({"t": round(turn_t, 3), "session": f"s{sid:03d}", "tokens": tokens,
                     "prefix_id": name, "prefix_tokens": plen})
        history = tokens + rng.randint(80, 400)
        turn_t += rng.expovariate(1 / 8.0)
    sid += 1

rows.sort(key=lambda r: r["t"])
for r in rows:
    sys.stdout.write(json.dumps(r, separators=(",", ":")) + "\n")
