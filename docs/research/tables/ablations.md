# Ablation Study Results

_From `benchmark_results/ablation_results.csv`_

| Family | Scale | Variant | Score | Δ vs full | Interpretation |
|---|---|---|---|---|---|
| pairwise_hidden_edge | 128 | rmca_full | 1.000 | +0.0000 |  |
| pairwise_hidden_edge | 128 | rmca_no_graph_expansion | 1.000 | +0.0000 |  |
| pairwise_hidden_edge | 128 | rmca_no_conflict_resolution | 1.000 | +0.0000 |  |
| pairwise_hidden_edge | 128 | rmca_no_decomposition | 0.000 | -1.0000 | decompose responsible for -1.000 on pairwise_hidden_edge |
