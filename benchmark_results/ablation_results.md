# RMCA Ablation Study Results

> **Warning:** Results use deterministic synthetic data. Do not compare numerically to the RLM paper.

| Benchmark family | Scale | Variant | Score | Delta vs full | Annotation | Tokens |
|---|---:|---|---:|---:|---|---:|
| pairwise_hidden_edge | 128 | rmca_full | 1.0000 | 0.0000 |  | 421 |
| pairwise_hidden_edge | 128 | rmca_no_graph_expansion | 1.0000 | 0.0000 |  | 421 |
| pairwise_hidden_edge | 128 | rmca_no_conflict_resolution | 1.0000 | 0.0000 |  | 350 |
| pairwise_hidden_edge | 128 | rmca_no_decomposition | 0.0000 | -1.0000 | decompose responsible for -1.000 on pairwise_hidden_edge | 143 |

