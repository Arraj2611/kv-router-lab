# KV Router Lab

An interactive simulator of **cache-aware routing for LLM serving**: how a router should spread requests across GPU replicas so that each replica's prefix (KV) cache actually gets reused, without overloading any one of them.

**▶ Live demo:** https://Arraj2611.github.io/kv-router-lab/

![Prefix-hash routing overloading one GPU, then KV-aware routing spreading the same traffic](docs/demo.gif)

> This is a **simulation**. It models queues, prefill time and KV-cache eviction with simple arithmetic and runs on your CPU. It does not use a GPU or run a real model.

## Why it matters

LLM serving engines like vLLM and SGLang reuse the KV cache of a shared prompt prefix (system prompt, tools, chat history) instead of recomputing it. That cache lives on one GPU, so with several replicas the **router decides whether caching pays off**:

- Ignore the cache (round-robin) and every replica recomputes the same prefixes.
- Pin each prefix to one replica (prefix hashing) and the most popular prompt overloads its replica.
- Score replicas on *prefill saved vs. queue wait* and you get both high reuse and even load.

The lab lets you watch all three happen.

## Run it

**In the browser (no install):** open `index.html` in any modern browser, or use the live demo link above.

**Python CLI (no dependencies, Python 3.9+):**

```bash
python sim/cache_router_sim.py                    # 4 replicas, 6 / 14 / 20 req/s
python sim/cache_router_sim.py --replicas 8 --rates 10,30 --csv results.csv
python sim/cache_router_sim.py --help             # every knob
```

The browser engine is a JavaScript port of the Python simulator and produces the same strategy ranking.

## Strategies

| Strategy | How it picks a replica |
| --- | --- |
| Round-robin | Next replica in turn |
| Least-loaded | Shortest prefill queue |
| Sticky session | Consistent hash of the user's session |
| Prefix hash | Consistent hash of the first prompt block (≈ the system prompt) |
| Approximate prefix tree | Remembers where it sent each prefix; follows the longest match unless that replica is overloaded (similar to SGLang's router). Never sees evictions. |
| KV-aware cost | Reads each replica's real cache; picks the lowest estimated time to first token = uncached prefill + queue wait (similar to NVIDIA Dynamo / llm-d) |

## Results (Python CLI, seed 7, 4 replicas, 60k-token cache each, 20 req/s)

| Strategy | Cache hits | TTFT p50 | TTFT p99 | Load imbalance |
| --- | --- | --- | --- | --- |
| Round-robin | 63.1% | 1.26 s | 16.99 s | 1.01× |
| Least-loaded | 63.0% | 0.76 s | 11.86 s | 1.04× |
| Sticky session | 69.3% | 0.64 s | 10.84 s | 1.11× |
| Prefix hash | 70.8% | 432.5 s | 1414.3 s | 3.54× |
| Approximate prefix tree | 69.2% | 0.41 s | 2.55 s | 1.10× |
| **KV-aware cost** | **70.9%** | **0.23 s** | **1.06 s** | 1.11× |

Full numbers for three load levels are in [`sim/results.csv`](sim/results.csv).

## How the model works

- Prompts are split into 16-token blocks with chained identities, as in vLLM: a block counts as cached only if every block before it is cached.
- Each replica runs prefill one request at a time and evicts least-recently-used blocks, leaves before shared prefixes.
- Workload: six system prompts with Zipf popularity (one hot product) and multi-turn chats that grow every turn.
- Time to first token = queueing delay + prefill time. Prefill time = `overhead + uncached_tokens / prefill_tps + cached_tokens / cached_tps`. The last term is the cost of attending over an already-cached prefix; it defaults to 0 (cached tokens free), which is what the table above uses. Decoding is not modelled.
- Optional CPU offload tier (`--cpu-tokens N`, or the "CPU offload" slider on the page): blocks evicted from the GPU spill into a CPU tier with its own capacity and LRU. A request matches the leading blocks present in either tier; blocks found on the CPU are copied back at `--reload-tps` (default 40,000 tokens/s, an unmeasured guess chosen to be several times faster than recompute) and re-enter the GPU cache. `cpu_hit_share` reports what fraction of cached tokens came from the CPU tier. Off by default, so the results table above is unchanged. Example at 20 req/s with the default cost model: adding 60k tokens of CPU tier per replica raised kv-aware cost's hit rate from 70.9% to 78.8% (9% of hits came from the CPU) and cut p99 TTFT from 1.06 s to 0.63 s; the table of strategies at other sizes is one command away.

- Replica failure: `--fail-at T --fail-for D --fail-replica N` kills one replica at time T (both cache tiers lost, unroutable) and brings it back empty after D seconds. Requests already admitted still complete (no drops or retries), and strategies that ignore queue depth (round-robin, sticky, prefix-hash) fail over to the least-loaded live replica. The CLI then adds three columns: fleet hit rate in the 120 s before, the worst 5 s window from the failure on, and seconds after recovery until a 5 s window reaches 95% of the "before" rate (`never` if it doesn't within the run). Example, seed 7, 6 req/s, 4 replicas, GPU 2 down from 300 s to 420 s:

  | strategy | hit before | worst 5 s window | recovery |
  |---|---|---|---|
  | round-robin | 66.8% | 48.6% | 15 s |
  | least-loaded | 65.8% | 51.0% | 25 s |
  | sticky-session | 81.5% | 61.2% | 15 s |
  | prefix-hash | 76.5% | 60.3% | 5 s |
  | approx-prefix-tree | 76.5% | 41.8% | 5 s |
  | kv-aware-cost | 79.6% | 65.6% | 0 s |

  This is one seed and one failure event, and a 5 s window holds only about 30 requests at 6 req/s, so differences of one or two windows in the recovery column are noise; the visible pattern is that kv-aware cost dips least and recovers first, and approx-prefix-tree dips most (it keeps remembering prefixes on the replica that came back empty). At 14 req/s every strategy recovers inside the first window. On the page, the **Fail GPU 1** toggle runs this scenario live (the failed GPU is dimmed and labelled DOWN with a countdown, the latency chart shades the outage) and adds the three columns to the comparison table; "Copy settings" gives the matching CLI command. Not modelled: in-flight requests failing, client retries, a health-check delay before routing stops, or warm-up slower than cache refill.
- Trace replay: `python sim/cache_router_sim.py --trace docs/example_trace.jsonl` replays a JSONL request trace (arrival time, session, prompt length, shared-prefix id) instead of the synthetic workload; the page has a matching "Load trace…" button. Format and caveats: [`docs/TRACE_FORMAT.md`](docs/TRACE_FORMAT.md). The shipped example is synthetic.

## Calibration

The default constants (`--prefill-tps 8000 --overhead 0.02`) were guesses. I measured the real thing once, on one machine, and shipped the result as opt-in profiles, one per model and engine: `--profile rtx4070-llama3.2-1b-vllm`, `rtx4070-llama3.2-1b-sglang`, `rtx4070-qwen1.5b-vllm`, `rtx4070-qwen1.5b-sglang`. The simulator is still a simulation; these profiles only set its per-replica cost constants.

**What was measured.** RTX 4070 Laptop GPU (8 GB), driver 617.14, WSL2 Ubuntu 22.04, two models in fp16, `meta-llama/Llama-3.2-1B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, each on vLLM 0.31.0 and SGLang 0.5.22, 8192 context, all on 2026-10-16. Qwen was run first (the Llama repo was not accessible yet) and Llama a few hours later, so they are separate sessions on the same machine. Requests used `max_tokens=1`, so TTFT is prefill time. One request at a time unless stated.
- *Cold sweep:* unique prompts of 212 to 4,734 tokens, run in shuffled order (a cache miss).
- *Hit sweep:* a cached prefix of 400 to 4,720 tokens plus a ~30-token unique tail (a cache hit).
- *Held out, not used to fit:* a 2.4k-token shared-prefix run, and a paired run that alternates a hit with a matched-length miss.

Scripts and raw CSVs are in [`bench/`](bench/); `python bench/fit_profile.py <csv>` reproduces the constants.

| | Llama-3.2-1B vLLM | Llama-3.2-1B SGLang | Qwen2.5-1.5B vLLM | Qwen2.5-1.5B SGLang |
|---|---|---|---|---|
| Fixed cost per request | 16.5 ms | 17.4 ms | 17.7 ms | 19.7 ms |
| Prefill speed | 16,949 tok/s | 16,998 tok/s | 11,585 tok/s | 11,090 tok/s |
| Cost of a cached token | 3.5 µs | 3.6 µs | 4.0 µs | 3.4 µs |
| Fit error on the sweeps (median / worst point) | 5.2% / 10.6% | 5.0% / 9.6% | 4.2% / 10.5% | 8.2% / 16.9% |
| 2.4k-token cache hit, model vs measured p50 | -2% | +1% | -6% | -6% |
| 2.4k-token miss in the paired run, model vs measured p50 | -7% (157 vs 169 ms) | -1% (158 vs 160 ms) | -12% (224 vs 255 ms) | -46% (235 vs 436 ms) |
| 2.4k-token hit in the paired run, model vs measured p50 | -9% | -3% | -26% | -38% |

**Stated error.** For one request at a time, the calibrated model's median error on the length sweeps is 4–8% (worst single point 10–17%), and a 2.4k-token cache hit is within 6% of the model in all four setups. The alternating hit/miss case separates the two sessions. On Llama the model is within 9% (vLLM) and 3% (SGLang). On Qwen it under-predicts by 12–46% on a miss and 26–38% on a hit, because TTFT in that session was bimodal: the same 2.4k-token prompt took about 230 ms or about 450 ms, and the slow mode showed up more in some runs than others (a repeat of the SGLang paired run gave a 328 ms miss instead of 436 ms). Llama, run a few hours later on the same machine, showed no such slow mode in its p50s (p99s still spiked to 200–450 ms). The driver reported a software power cap during a third of the sampled intervals in the Qwen session, which fits a power-limited laptop GPU, but I did not isolate the cause, and each model is a single session, so I can't say whether the difference comes from the model or from the machine's state at the time. Treat the constants as medians with up to roughly ±30% run-to-run spread; the Llama profiles are the better-behaved ones.

**What it changed.** Without a cached-token cost the model said caching speeds up a 2.4k-token prompt about 26x. Measured in the paired runs: about 6x on Qwen/vLLM (255 ms to 41 ms) and about 5.7x on Llama/vLLM (169 ms to 30 ms). With the third term the model gives about 7x (Qwen) and 5.8x (Llama), which is the case the constants were fit on. Under all four measured profiles, kv-aware cost is still best by p99 TTFT at 6, 14 and 20 req/s, and prefix-hash is still worst at 14 and 20 req/s. At 6 req/s prefix-hash is worst under the Qwen profiles but approx-prefix-tree is worst under the Llama ones; the strategies in between change order.

**What it does not cover.** Two small models on one GPU, two engines at default settings plus `--enable-cache-report` on SGLang; the open-loop queueing runs and the batching test used Qwen/vLLM only. Concurrent-load numbers varied too much between runs to report as constants (vLLM p50 at 8 clients ranged from 70 to 215 ms; at 11 req/s open-loop the hit p50 was 5.8 s in one run and 0.4 s in another), so queue lengths near saturation are only validated to roughly a factor of two (see Limits). The vLLM runs used the PyTorch sampler instead of FlashInfer's because the CUDA toolkit was not installed at the time; at `max_tokens=1` that should not affect prefill, but it is a difference from a default install.

## Limits

- **FIFO prefill, no batching.** I prototyped a stepwise token-budget batching model (2,048 or 8,192 tokens per step, with the fixed cost either per request or shared per step) in [`bench/validate_batching.py`](bench/validate_batching.py) and scored it against two open-loop vLLM runs on one replica (70% hit / 30% miss traffic, 3–11 req/s, different seeds). It was not closer than FIFO: median error 14–16% vs 14.6% in run 1 and 19–22% vs 21% in run 2, which is inside the noise, so the engines keep FIFO. The larger problem is that the effective service speed differed by about 1.5x between those two runs on the same machine (best-fit slowdown 1.5 vs 1.0), and near saturation a few percent of service time swings queueing delay several-fold. Trust the ranking of strategies more than the absolute seconds, and treat queueing numbers near saturation as good to a factor of about two at best.
- Cost constants are measured on one laptop GPU and one small model (see Calibration). The built-in defaults are still guesses.
- The CPU offload tier's reload speed is an assumption, not a measurement, and it does not model PCIe contention with other transfers. Replica failure is a simple graceful-drain model (see above); on the page it is a fixed scenario (GPU 1 fails at 40% of the run for 20% of it, at most 120 s), not a free-form control; the approximate router never forgets what it routed.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Next up: batching-aware prefill, checked against the measurements above.

## License

MIT
