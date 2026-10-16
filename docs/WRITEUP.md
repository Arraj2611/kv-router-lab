# Routing LLM requests for cache reuse: a simulator, six strategies, and what I measured

**KV Router Lab is a simulation.** It does not run a model or use a GPU. Its cost constants were measured once, on one laptop GPU, and everything else is arithmetic. Code and live demo: https://Arraj2611.github.io/kv-router-lab/

## The problem

Serving engines such as vLLM and SGLang reuse the KV cache of a shared prompt prefix (system prompt, tools, earlier chat turns) instead of recomputing it. That cache lives on one GPU. With several replicas, the router decides whether caching pays off:

- Ignore the cache (round-robin) and every replica recomputes the same prefixes.
- Pin each prefix to one replica (prefix hash) and the most popular prompt overloads its replica.
- Score replicas on prefill saved against queue wait, and you can get both reuse and even load.

## What the simulator models

Prompts are split into 16-token blocks with chained identity (a block matches only if all earlier blocks match, as in vLLM). Each replica prefills one request at a time and evicts least-recently-used blocks, leaves before shared prefixes. The workload is six system prompts with Zipf popularity and multi-turn chats whose history grows each turn. Time to first token (TTFT) is queue wait plus prefill; decoding is not modelled. A JavaScript engine (the web page) and a Python engine (CLI) are bit-for-bit twins, checked in CI.

## Six strategies, one result

At 20 req/s on 4 replicas with 60k-token caches (seed 7, default cost model):

| Strategy | Cache hits | TTFT p50 | TTFT p99 |
| --- | --- | --- | --- |
| Round-robin | 63.1% | 1.26 s | 16.99 s |
| Least-loaded | 63.0% | 0.76 s | 11.86 s |
| Sticky session | 69.3% | 0.64 s | 10.84 s |
| Prefix hash | 70.8% | 432.5 s | 1414.3 s |
| Approximate prefix tree | 69.2% | 0.41 s | 2.55 s |
| **KV-aware cost** | **70.9%** | **0.23 s** | **1.06 s** |

Prefix hash has nearly the best hit rate and by far the worst latency: the hot prompt's replica never drains. The absolute seconds near saturation should not be trusted (see limits); the ranking is the claim. Under the measured cost constants below, KV-aware cost is still best and prefix hash still worst by p99 TTFT at 6, 14 and 20 req/s.

## Calibration against real engines

I replaced the guessed prefill speed with measurements: vLLM 0.31.0 and SGLang 0.5.22 on an RTX 4070 Laptop GPU (8 GB, WSL2), serving Qwen2.5-1.5B-Instruct in fp16 (the Llama repo I wanted was not accessible). Both engines gave about the same constants: roughly 18-20 ms fixed cost per request, about 11-11.6k tokens/s prefill, and about 3.4-4 microseconds per cached token for attending over a cached prefix.

That last term mattered. With cached tokens free, the model predicted caching would speed up a 2.4k-token prompt about 26x; vLLM measured about 6x (255 ms to 41 ms). The simulator now has a per-cached-token cost.

How accurate is it? For one request at a time, the median error on length sweeps is 4-8% and a 2.4k-token cache hit is about 6% low. When hits and misses alternate, it under-predicts by 12-46%, because TTFT on this laptop GPU is bimodal (the same prompt takes about 230 ms or about 450 ms) and I did not isolate why. Full method and numbers: the README's Calibration section and `bench/`.

## Things I tested that did not pan out

- **Batching.** I prototyped a token-budget batched-prefill model and scored it against two open-loop vLLM runs. It was no closer than the FIFO model. Worse, the effective service speed differed about 1.5x between the two runs on the same machine, and near saturation a few percent of service time swings queueing delay several-fold. So the engines keep FIFO and queueing numbers near saturation are good to roughly a factor of two at best. My first reading of the data ("sustained load makes the GPU 1.5x slower") came from one run and did not survive a repeat.
- **My own first benchmark design** confounded run order with the thing being measured, and a negative test showed my first example trace hid a rounding difference between the two parsers. Both are fixed; both are why the checks in this repo try to fail on purpose.

## Extensions

- **CPU offload tier:** GPU evictions spill to a CPU tier; reloading costs tokens divided by a reload speed (40k tokens/s, an unmeasured guess). At 20 req/s, 60k tokens of CPU tier per replica raised KV-aware cost's hit rate from 70.9% to 78.8% and cut p99 TTFT from 1.06 s to 0.63 s in the default model.
- **Trace replay:** a JSONL format (arrival time, session, prompt length, shared-prefix id) for replaying your own traffic; the shipped example is synthetic.
- **Replica failure:** kill one replica and bring it back empty. KV-aware cost dips least and recovers first in the example run, but that is one seed and one event.

## Limits

- It is a simulation with one measured configuration (one laptop GPU, one small model, two engines at default settings). The built-in default constants are still guesses.
- FIFO prefill, no batching, no decoding.
- The reload speed for CPU offload is an assumption; replica failure is a graceful-drain model (admitted requests complete) and is not in the web UI.
- The approximate-prefix-tree router never forgets what it routed, which exaggerates its weakness after evictions or failures.

## Reproduce

`python sim/cache_router_sim.py` runs the default comparison. `python tools/parity_check.py` (after `node tools/parity_check.js`) checks the two engines agree. `bench/` has the scripts, raw CSVs and `fit_profile.py` for the calibration.
