# CLAUDE.md — KV Router Lab

Read this first in every session. The task list lives in `docs/ROADMAP.md`.

## What this project is

A **simulator** of cache-aware routing for LLM serving. It compares six strategies for spreading requests across GPU replicas that each hold a prefix (KV) cache, and reports cache hit rate, time to first token (TTFT) and load imbalance.

It is a portfolio project. The owner (Rajeev) is an AI engineer aiming at systems roles at AI labs, so **measured, honest evidence beats polish**. A claim in the README is only as good as the number behind it.

## Hard rules

1. **Always call it a simulation.** It does not run a model or use a GPU. Never write README, page or post copy that implies otherwise. Numbers calibrated from real hardware must say which hardware, model, engine and version.
2. **Two engines, one model.** The simulator exists twice:
   - Python: `sim/cache_router_sim.py`
   - JavaScript: inside `index.html`, in `<script type="text/plain" id="engine-src">`
   Any change to workload generation, block hashing, eviction, strategies or metrics must be made in **both**, and the parity check (ROADMAP task 2) must pass before committing.
3. **No build step.** `index.html` is one self-contained file that must work when opened directly from disk (`file://`) and on GitHub Pages. No npm, no bundler, no framework. The only external request allowed is Google Fonts, and the page must still work if it fails.
4. **Python CLI has no third-party dependencies** (standard library only, Python 3.9+). Analysis or plotting extras go in a separate `tools/` folder with their own `requirements.txt`.
5. **Don't change default results silently.** If a change moves the numbers, regenerate `sim/results.csv` and the README results table in the same commit, and say why in the commit message.

## Layout

```
index.html               the interactive lab (UI + JS engine), also the GitHub Pages site
sim/cache_router_sim.py  Python simulator and CLI
sim/results.csv          output of the default CLI run
docs/                    demo.gif, ROADMAP.md, any write-ups
README.md                public face of the project
```

## Commands

```bash
python sim/cache_router_sim.py                         # default run: 4 replicas, 6/14/20 req/s
python sim/cache_router_sim.py --csv sim/results.csv   # regenerate committed results
python sim/cache_router_sim.py --help
python -m http.server 8000                             # then open http://localhost:8000
```

## Model semantics (keep both engines identical)

- Prompts split into 16-token blocks. Block identity is chained: a block matches only if all earlier blocks match (vLLM-style). The partial last block is never cached.
- System-prompt blocks are shared across all sessions of a product; later blocks belong to one session.
- Each replica: FIFO prefill, one request at a time; prefill time = `overhead + uncached_tokens / prefill_tps`. Decode is not modelled. TTFT = queue wait + prefill.
- LRU eviction; on admit, blocks are touched deepest-first so shared prefixes are evicted last.
- Workload: Poisson session arrivals, Zipf product popularity, 1..max_turns turns, exponential think time; each turn's answer joins the history.
- Default seed 7. Both engines use their own RNG, so exact numbers differ between Python and JS; **strategy ranking** must match.

## The page (`index.html`)

- Colors are CSS tokens on `:root` with light and dark variants; canvas code reads them via `readTokens()`. Never hard-code a color in drawing code.
- Product colors are a validated colorblind-safe palette (`--p0`..`--p5`). Don't reorder or replace them.
- Respect `prefers-reduced-motion` (no autoplay, no flying dots).
- Must work at 400px width with no horizontal scroll.
- The six-strategy comparison runs in a Web Worker built from a Blob, with a main-thread fallback. Keep both paths working.

## Owner's hardware

Laptop with an NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM. Fits a 1B model in fp16 comfortably, a 3B model tightly, 7–8B only 4-bit. vLLM and SGLang need Linux; on Windows use WSL2 with the NVIDIA driver. Ask which OS before giving setup steps.

## How to work with the owner

- For anything beyond a small fix, state the plan in a few lines and wait for a go-ahead.
- Small, focused commits with clear messages.
- When a result looks surprising, say so and check it before writing it into the README.
- Point out weak claims or shaky methodology directly; don't smooth them over.
