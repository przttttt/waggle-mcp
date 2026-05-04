# Groq Answer-Level Evaluation

_From `benchmark_results/groq_answer_level_results.csv` (or `answer_level_results.csv`)_

| Scale | Method | Mean F1 ± Std | Hall. Rate | Insuff. Rate | Tokens |
|---|---|---|---|---|---|
| 128 | bm25_topk | 0.105 ± 0.182 | 0.667 | 0.000 | 1411 |
| 128 | query_graph | 0.118 ± 0.204 | 0.333 | 0.000 | 108 |
| 128 | raw_context | 0.058 ± 0.100 | 0.667 | 0.000 | 1405 |
| 128 | rmca_full | 0.348 ± 0.325 | 0.333 | 0.000 | 381 |
