# Benchmark Methodology

This document outlines how Waggle is evaluated against its internal fixtures and external benchmarks.

## Evaluation Philosophy

Waggle is designed as a structured memory system, not a raw transcript store. Our evaluation methodology reflects this:

1. **Ingestion via Tools**: We ingest data through `observe_conversation` rather than bulk-loading pre-processed nodes. This ensures we measure the performance of the full extraction-to-retrieval pipeline.
2. **Retrieval Recall (R@k)**: We primarily report retrieval recall (R@k) rather than end-to-end QA accuracy. This isolates the performance of the memory engine from the reasoning capabilities of the specific LLM used for answering.
3. **Structured Context**: Waggle returns typed subgraphs (decisions, facts, relationships) rather than raw text chunks. While this may result in lower verbatim recall on some benchmarks, it provides richer relational context at a significantly lower token cost.

## External Benchmarks

| Benchmark | Dataset Size | Primary Metric | Command | Held-out? |
|-----------|--------------|----------------|---------|-----------|
| **LongMemEval** | 500 questions | R@5, Exact@5 | `scripts/benchmark_longmemeval.py` | Optional (--held-out) |
| **LoCoMo** | 1,986 questions | R@10, R@5 | `scripts/benchmark_locomo.py` | No |
| **ConvoMem** | 250 items | Avg Recall | `scripts/benchmark_convomem.py` | No |
| **MemBench** | 8,500 items | R@5 | `scripts/benchmark_membench.py` | No |

### Held-out Evaluation

For **LongMemEval**, we support a held-out methodology using the `--held-out` flag. This randomly splits the 500-question set into 50 "dev" questions (for parameter tuning) and 450 "test" questions (for final reporting). 

## Comparison Disclaimer

**Important**: Do not compare Waggle's retrieval recall (R@k) directly against another project's end-to-end QA accuracy. These are fundamentally different metrics. Waggle captures whether the relevant memory *subgraph* was retrieved, not whether an LLM synthesized a correct final answer from that subgraph.

For detailed reproduction instructions, see:
- [benchmarks/longmemeval/README.md](../benchmarks/longmemeval/README.md)
- [benchmarks/locomo/README.md](../benchmarks/locomo/README.md)
- [benchmarks/convomem/README.md](../benchmarks/convomem/README.md)
- [benchmarks/membench/README.md](../benchmarks/membench/README.md)
