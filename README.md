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
| Round-robin | 69.9% | 2.38 s | 16.66 s | 1.03× |
| Least-loaded | 70.0% | 1.15 s | 9.09 s | 1.05× |
| Sticky session | 75.2% | 0.55 s | 12.07 s | 1.08× |
| Prefix hash | 75.3% | 172 s | 594 s | 2.21× |
| Approximate prefix tree | 74.9% | 0.40 s | 3.33 s | 1.08× |
| **KV-aware cost** | **76.2%** | **0.22 s** | **1.63 s** | 1.07× |

Full numbers for three load levels are in [`sim/results.csv`](sim/results.csv).

## How the model works

- Prompts are split into 16-token blocks with chained identities, as in vLLM: a block counts as cached only if every block before it is cached.
- Each replica runs prefill one request at a time and evicts least-recently-used blocks, leaves before shared prefixes.
- Workload: six system prompts with Zipf popularity (one hot product) and multi-turn chats that grow every turn.
- Time to first token = queueing delay + prefill of the uncached tokens. Decoding is not modelled.

## Limits

- **No batching.** Queues grow faster than on a real engine near saturation. Trust the ranking of strategies more than the absolute seconds.
- One constant prefill speed (`--prefill-tps`), not measured from real hardware.
- No cache offload tier, no replica failures; the approximate router never forgets what it routed.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Next up: calibrating the simulator against real SGLang / vLLM measurements on an RTX 4070.

## License

MIT
