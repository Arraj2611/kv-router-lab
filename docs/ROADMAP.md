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
- [x] `docs/demo.gif`, 10–15 s, under 5 MB: Prefix hash at 18 req/s until GPU 1's queue goes red, then switch to KV-aware cost
- [x] Referenced at the top of the README

### 4. Small page polish
- [x] Shareable deep link to a strategy via a bare hash token (e.g. `#prefix`, `#kv`); only plain tokens, no `key=value`
- [x] "Copy settings" button that copies the matching CLI command

## Phase 2 — Make it real (highest value for the portfolio)

### 5. Calibrate against real serving on the RTX 4070
Goal: replace the guessed `--prefill-tps` and cache-hit savings with measured numbers.
- [x] `bench/` folder with a script that serves a 1B instruct model (e.g. Llama-3.2-1B-Instruct) with SGLang, and separately with vLLM — ran Qwen2.5-1.5B-Instruct (Llama repo returned 403 for the HF token); rerun on Llama is `bench/run_experiments.sh <engine> on meta-llama/Llama-3.2-1B-Instruct`
- [x] Workload: one long shared system prompt (~3,000 tokens) + many short questions; run with prefix caching on and off — shared prefix is ~2.4k tokens, plus cold and hit length sweeps up to 4.7k; "off" is measured as unique prompts on the cache-on server (a miss), which removes run-to-run drift (the separate cache-off server run is kept as `run1_blocked_superseded`)
- [x] Record: prefill tokens/s, TTFT p50/p99, throughput, cache hit rate; save raw results as CSV with GPU, driver, engine version, model and date — throughput under load was too noisy to report
- [x] Add `--profile rtx4070-1b` to the simulator that loads the measured constants — named `rtx4070-qwen1.5b-vllm` / `-sglang`; also added a `cached_tps` cost term to both engines
- [x] README section "Calibration": method, numbers, and what the simulator still gets wrong

**Done when:** the simulator's predicted TTFT for the single-replica, cache-on/cache-off cases is within a stated error of the measured TTFT, and the README states that error. *(Met for sequential requests: ~6% on a hit, 4–8% median on sweeps. Under alternating hit/miss the error is 12–46% because TTFT on this laptop GPU is bimodal; stated in the README.)*

### 6. Batching-aware prefill
The current model serves prefill one request at a time, which inflates queues near saturation.
- [x] Model chunked or batched prefill with a token budget per step — prototyped in `bench/validate_batching.py` (not in the engines)
- [x] Compare the old and new model against the Phase 2 measurements and keep whichever is closer — no measurable difference (see README Limits), so FIFO is kept and the engines are unchanged. Measured with open-loop runs (`bench_client.py --experiment openloop`), two seeds.

## Phase 3 — Extensions

### 7. CPU offload tier
- [x] Evicted blocks spill to a CPU tier with its own capacity; reloading costs `tokens / reload_tps` (cheaper than recompute) — both engines, `--cpu-tokens` / `--reload-tps`; reload speed is an unmeasured default (40k tok/s)
- [x] New metric: share of hits served from CPU — `cpu_hit_share`; parity-checked with the tier off and on
- [x] Shown in the page as a second, thinner bar under each GPU's cache — plus a "CPU offload" slider and the share in the hits card

### 8. Real trace replay
- [x] Accept a JSONL trace (timestamp, session id, prompt token count, shared-prefix id) — plus optional `prefix_tokens`; CLI `--trace` and a "Load trace…" button on the page; both parsers cross-checked (`tools/trace_check.py`) and the example trace is in the parity check. The page flow was exercised in a browser: loading the example trace, a rejected bad file (error shown, previous replay kept) and "Use synthetic".
- [x] Document the format; ship a small anonymized example — `docs/TRACE_FORMAT.md`; the example is synthetic, not anonymized real traffic (no real traces available)

### 9. Replica failure
- [x] Kill a replica at time T, bring it back empty at T+Δ; show how fast each strategy's hit rate recovers — both engines (`--fail-at/--fail-for/--fail-replica`), parity-checked (workload `fail`), CLI columns for hit rate before / worst window / recovery time. Graceful-drain simplification; not wired into the page UI (the JS engine supports it via `cfg.failFor`).

## Phase 4 — Write-up

### 10. Blog post / LinkedIn
- [ ] Short write-up in `docs/WRITEUP.md`: problem, six strategies, results, calibration, limits
- [ ] LinkedIn post with the GIF attached natively and the Pages link; says plainly that it is a simulation, calibrated on an RTX 4070 if Phase 2 is done
