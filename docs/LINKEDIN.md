# LinkedIn post (draft, not posted)

Attach `docs/demo.gif` natively (upload the GIF to the post; do not link it). Put the Pages link in the first comment or the end of the post.

---

I built a simulator to answer one question about LLM serving: when you have several GPU replicas that each keep a prefix (KV) cache, how should the router pick one?

It compares six strategies. The result that surprised me: prefix hashing gets almost the best cache hit rate (70.8% vs 70.9% for the best) and by far the worst latency, because the most popular system prompt pins one replica until its queue never drains. A router that scores each replica on "prefill saved vs queue wait" gets the reuse and the balance.

To be clear: this is a simulation. It does not run a model. I did measure its cost constants on my own RTX 4070 laptop GPU with vLLM and SGLang (Qwen2.5-1.5B): about 11k tokens/s prefill, and caching a 2.4k-token prompt cut TTFT from 255 ms to 41 ms. Those measurements fixed a real flaw in my first model, which treated cached tokens as free and predicted a 26x speedup instead of ~6x.

What it still gets wrong, stated in the repo: it under-predicts latency by 12-46% when cache hits and misses alternate (my laptop GPU's timing is bimodal and I did not find out why), it has no batching (I tested a batched model; it was no closer to the data), and queueing numbers near saturation are good to about a factor of two.

Interactive demo, code, raw benchmark CSVs and the write-up are in the repo. The Python and JS engines are checked against each other in CI.

#LLM #inference #vLLM #SGLang #KVcache

Demo: https://Arraj2611.github.io/kv-router-lab/
Repo: https://github.com/Arraj2611/kv-router-lab
