# Waggle RLM-style Benchmark Results

> **Warning:** This benchmark follows the benchmark families used in the RLM paper,
> but uses deterministic synthetic memory tasks mapped to Waggle's graph/transcript
> environment. It should **not** be compared numerically to the RLM paper until the
> exact public datasets and matching model setup are run.

| Benchmark family | Scale | Method | Score | F1 | Ev. Coverage | Tokens returned | Latency (ms) |
|---|---:|---|---:|---:|---:|---:|---:|
| ContextReset | 128 | no_memory | 0.000 | 0.000 | 0.000 | 0 | 0 |
| ContextReset | 128 | raw_context | 0.000 | 0.000 | 0.250 | 1413 | 3 |
| ContextReset | 128 | query_graph | 0.875 | 1.000 | 1.000 | 96 | 1 |
| ContextReset | 128 | prime_context | 0.000 | 0.000 | 0.000 | 32 | 3 |
| ContextReset | 128 | bm25_topk | 0.000 | 0.000 | 0.250 | 1413 | 2 |
| ContextReset | 128 | rmca_full | 1.000 | 1.000 | 1.000 | 315 | 13 |

## Token efficiency: build_context vs baselines

| Benchmark family | Scale | Method | Tokens returned | Score |
|---|---:|---|---:|---:|
| ContextReset | 128 | no_memory | 0 | 0.000 |
| ContextReset | 128 | prime_context | 32 | 0.000 |
| ContextReset | 128 | query_graph | 96 | 0.875 |
| ContextReset | 128 | rmca_full | 315 | 1.000 |
| ContextReset | 128 | raw_context | 1413 | 0.000 |
| ContextReset | 128 | bm25_topk | 1413 | 0.000 |
