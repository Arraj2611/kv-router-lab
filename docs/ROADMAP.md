# Roadmap

Work top to bottom unless the owner says otherwise. Each task lists what "done" means. Tick the box in the same commit that finishes it.

## Phase 1 — Ship it publicly

### 1. Publish the repo and the live demo
- [x] Public GitHub repo `kv-router-lab` with this folder pushed to `main`
- [x] GitHub Pages enabled from `main`, root; the live URL loads the lab
- [x] README's "Live demo" line points at the real Pages URL
- [x] Repo description and topics set: `llm`, `inference`, `kv-cache`, `vllm`, `sglang`, `simulation`

**Done when:** the Pages URL works in a private browser window on desktop and phone.

### 2. Engine parity check
The Python and JS engines must agree on strategy ranking.
- [x] `tools/parity_check.js` (Node, no dependencies): extracts the engine from `index.html`, runs all six strategies at 6 and 20 req/s with default settings, writes JSON
- [x] `tools/parity_check.py`: runs the Python CLI with the same settings, compares rankings by p99 TTFT and by hit rate, exits non-zero on a mismatch
- [x] GitHub Action runs it on every push and pull request

**Done when:** CI is green and a deliberately broken strategy in one engine makes it fail.

### 3. Demo GIF
- [ ] `docs/demo.gif`, 10–15 s, under 5 MB: Prefix hash at 18 req/s until GPU 1's queue goes red, then switch to KV-aware cost
- [ ] Referenced at the top of the README

### 4. Small page polish
- [ ] Shareable deep link to a strategy via a bare hash token (e.g. `#prefix`, `#kv`); only plain tokens, no `key=value`
- [ ] "Copy settings" button that copies the matching CLI command

## Phase 2 — Make it real (highest value for the portfolio)

### 5. Calibrate against real serving on the RTX 4070
Goal: replace the guessed `--prefill-tps` and cache-hit savings with measured numbers.
- [ ] `bench/` folder with a script that serves a 1B instruct model (e.g. Llama-3.2-1B-Instruct) with SGLang, and separately with vLLM
- [ ] Workload: one long shared system prompt (~3,000 tokens) + many short questions; run with prefix caching on and off
- [ ] Record: prefill tokens/s, TTFT p50/p99, throughput, cache hit rate; save raw results as CSV with GPU, driver, engine version, model and date
- [ ] Add `--profile rtx4070-1b` to the simulator that loads the measured constants
- [ ] README section "Calibration": method, numbers, and what the simulator still gets wrong

**Done when:** the simulator's predicted TTFT for the single-replica, cache-on/cache-off cases is within a stated error of the measured TTFT, and the README states that error.

### 6. Batching-aware prefill
The current model serves prefill one request at a time, which inflates queues near saturation.
- [ ] Model chunked or batched prefill with a token budget per step
- [ ] Compare the old and new model against the Phase 2 measurements and keep whichever is closer

## Phase 3 — Extensions

### 7. CPU offload tier
- [ ] Evicted blocks spill to a CPU tier with its own capacity; reloading costs `tokens / reload_tps` (cheaper than recompute)
- [ ] New metric: share of hits served from CPU
- [ ] Shown in the page as a second, thinner bar under each GPU's cache

### 8. Real trace replay
- [ ] Accept a JSONL trace (timestamp, session id, prompt token count, shared-prefix id)
- [ ] Document the format; ship a small anonymized example

### 9. Replica failure
- [ ] Kill a replica at time T, bring it back empty at T+Δ; show how fast each strategy's hit rate recovers

## Phase 4 — Write-up

### 10. Blog post / LinkedIn
- [ ] Short write-up in `docs/WRITEUP.md`: problem, six strategies, results, calibration, limits
- [ ] LinkedIn post with the GIF attached natively and the Pages link; says plainly that it is a simulation, calibrated on an RTX 4070 if Phase 2 is done
