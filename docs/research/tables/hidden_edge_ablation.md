# Table: Pairwise Hidden-Edge Ablation

> **Synthetic data caveat:** Deterministic synthetic tasks. Do not compare to RLM paper numerically.
>
> **Benchmark note:** `pairwise_hidden_edge` uses node contents without conflict vocabulary.
> Conflict is represented only by typed `contradicts` edges. Results use 3 seeds (42, 43, 44);
> all seeds produced identical scores (fully deterministic benchmark).

## Ablation Results (mean across seeds 42, 43, 44)

| Variant | Scale | Score | Δ vs full | Tokens | Claim supported? |
|---|---:|---:|---:|---:|---|
| `rmca_full` | 128 | 1.000 | — | 421 | Baseline |
| `rmca_full` | 512 | 1.000 | — | 464 | Baseline |
| `rmca_no_graph_expansion` | 128 | 1.000 | 0.000 | 421 | ❌ Not isolated |
| `rmca_no_graph_expansion` | 512 | 1.000 | 0.000 | 464 | ❌ Not isolated |
| `rmca_no_conflict_resolution` | 128 | 1.000 | 0.000 | 350 | ❌ Not isolated |
| `rmca_no_conflict_resolution` | 512 | 1.000 | 0.000 | 393 | ❌ Not isolated |
| `rmca_no_decomposition` | 128 | **0.000** | **-1.000** | 143 | ✅ Confirmed |
| `rmca_no_decomposition` | 512 | **0.000** | **-1.000** | 145 | ✅ Confirmed |

## Claim Summary

| Claim | Supported? | Evidence | Caveat |
|---|---|---|---|
| Decomposition is load-bearing | ✅ Yes | Score 1.0→0.0 at both scales, all 3 seeds | Synthetic data; deterministic embedding |
| Graph expansion is load-bearing | ❌ Not isolated | No delta at either scale | Deterministic embedding retrieves conflict nodes without edge traversal |
| Conflict resolution is load-bearing | ❌ Not isolated | No delta at either scale | Scorer checks label presence, not explicit conflict annotation |

## Why Graph Expansion Cannot Be Isolated Here

The `_DeterministicEmbedding` model hashes token characters into a 16-dim vector.
Choice labels ("Cloud database") and constraint labels ("Local deployment required")
are semantically distinctive enough that direct retrieval surfaces them without
needing to traverse the `contradicts` edge.

To isolate graph expansion, a future benchmark needs:
1. A real embedding model (e.g., `all-MiniLM-L6-v2`)
2. Semantically similar node labels requiring edge traversal to distinguish
3. A scorer requiring explicit conflict annotation in the context pack
