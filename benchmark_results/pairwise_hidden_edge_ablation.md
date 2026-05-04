# Pairwise Hidden-Edge Ablation Results

> **Synthetic data caveat:** These results use deterministic synthetic Waggle memory tasks.
> They should not be compared numerically to the RLM paper until the exact public datasets
> and matching model setup are run.

## Benchmark Design

`pairwise_hidden_edge` is a harder variant of the pairwise conflict task designed to isolate
graph traversal benefits. The key difference from the standard `pairwise` benchmark:

- **Node contents do NOT contain conflict words** ("conflict", "contradict", "violates").
- **Conflict is represented only by typed `contradicts` edges** between choice nodes and
  constraint nodes.
- A single semantic query retrieves choices and constraints by label similarity, but cannot
  discover the conflict relationship without traversing the edge.

The question is: *"Which implementation choices are incompatible with the active deployment
requirements?"* — phrased neutrally, with no conflict vocabulary in the question or nodes.

## Results

Ablation across 4 variants × 2 scales × 3 seeds (seeds 42, 43, 44). All seeds produced
identical scores (deterministic benchmark).

| Variant | Scale=128 Score | Scale=512 Score | Δ vs full | Interpretation |
|---|---:|---:|---:|---|
| `rmca_full` | 1.000 | 1.000 | — | Baseline |
| `rmca_no_graph_expansion` | 1.000 | 1.000 | 0.000 | **No delta** |
| `rmca_no_conflict_resolution` | 1.000 | 1.000 | 0.000 | **No delta** |
| `rmca_no_decomposition` | **0.000** | **0.000** | **-1.000** | **Load-bearing** |

## Claim Assessment

### Decomposition is load-bearing ✅ Confirmed

Disabling decomposition drops score from 1.0 to 0.0 at both scales across all 3 seeds.
This replicates the finding from the standard pairwise benchmark. Decomposition generates
targeted subqueries ("which choices conflict with constraints") that retrieve the relevant
nodes. Without it, a single generic query fails to surface the conflict structure.

### Graph expansion is load-bearing ❌ Not isolated by this benchmark

`rmca_no_graph_expansion` scores identically to `rmca_full` (1.000) at both scales.

**Why this does not prove graph expansion is irrelevant:** The `pairwise_hidden_edge`
benchmark uses a deterministic embedding model (`_DeterministicEmbedding`) that hashes
token characters into a 16-dimensional vector. This model retrieves nodes by label
similarity — and the choice/constraint labels ("Cloud database", "Local deployment
required") are semantically distinctive enough that direct retrieval already surfaces
the conflict nodes without needing edge traversal.

**What would be needed to isolate graph expansion:** A benchmark where:
1. Choice and constraint nodes have semantically similar labels (so direct retrieval
   cannot distinguish them), AND
2. The conflict relationship is only discoverable by following the `contradicts` edge
   from a retrieved node to a node that would NOT be retrieved by semantic similarity alone.

This requires a real embedding model (not the deterministic hash-based one) and carefully
constructed node labels with controlled semantic overlap.

### Conflict resolution is load-bearing ❌ Not isolated by this benchmark

`rmca_no_conflict_resolution` scores 1.000 at both scales. The scoring function checks
whether conflict pair labels appear in the context pack — and they do, because the nodes
are retrieved. Explicit conflict resolution (marking superseded nodes, recording conflict
entries) does not change whether the labels appear in the pack.

**What would be needed:** A scoring function that checks whether the context pack
explicitly marks the conflict relationship (e.g., "Possible conflict: X contradicts Y"),
not just whether both node labels are present. With such a scorer, `rmca_no_conflict_resolution`
would likely score lower than `rmca_full`.

## Root Cause: Benchmark Limitation

The `pairwise_hidden_edge` benchmark successfully isolates decomposition as load-bearing,
but fails to isolate graph expansion and conflict resolution because:

1. The deterministic embedding model retrieves conflict nodes by label similarity without
   needing edge traversal.
2. The scoring function (pairwise F1 over label presence) does not require explicit
   conflict annotation in the context pack.

These are benchmark design limitations, not RMCA limitations. The benchmark needs to be
redesigned with:
- A real embedding model (e.g., `all-MiniLM-L6-v2`)
- Semantically similar node labels that require edge traversal to distinguish
- A scorer that requires explicit conflict annotation

## Summary

| Claim | Status | Evidence |
|---|---|---|
| Decomposition is load-bearing | ✅ Confirmed | Score 1.0→0.0 across all seeds/scales |
| Graph expansion is load-bearing | ❌ Not isolated | No delta; benchmark design limitation |
| Conflict resolution is load-bearing | ❌ Not isolated | No delta; scorer limitation |

The honest conclusion: **decomposition is the only RMCA component that has been causally
isolated as load-bearing on synthetic pairwise tasks.** Graph expansion and conflict
resolution may be load-bearing in real-world settings, but the current synthetic benchmark
cannot demonstrate this.
