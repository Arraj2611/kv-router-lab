#!/usr/bin/env python3
"""Checks that the Python and JS trace parsers agree on bad input and on edge cases.

Every case in tools/trace_cases.json must either be rejected by both parsers with the same
message, or parsed by both into identical requests. Needs `node` on PATH. Exits non-zero on any
disagreement. Complements tools/parity_check.py, which compares the two engines on the example trace.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
import cache_router_sim as sim  # noqa: E402


def main() -> int:
    cases = json.loads((ROOT / "tools" / "trace_cases.json").read_text(encoding="utf-8"))["cases"]
    js = json.loads(subprocess.run(["node", str(ROOT / "tools" / "trace_check.js")], check=True,
                                   capture_output=True, text=True, encoding="utf-8").stdout)
    bad = 0
    for case, j in zip(cases, js):
        try:
            reqs, info = sim.parse_trace(case["text"])
            py = {"ok": True,
                  "reqs": [[r.t, r.session, r.product, r.tokens, r.sys_full, r.n] for r in reqs],
                  "info": [info["sessions"], info["prefixes"], info["duration"], info["fallback_prefixes"]]}
        except ValueError as e:
            py = {"ok": False, "error": str(e)}
        same = py == {k: v for k, v in j.items() if k != "name"}
        bad += not same
        detail = py.get("error") if not py["ok"] else f"{len(py['reqs'])} request(s)"
        print(f"{'OK      ' if same else 'MISMATCH'} {case['name']:<58} {detail}")
        if not same:
            print(f"    py={py}\n    js={ {k: v for k, v in j.items() if k != 'name'} }", file=sys.stderr)
    print(f"\n{len(cases) - bad}/{len(cases)} cases agree.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
