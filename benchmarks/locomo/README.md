# LoCoMo Benchmark

LoCoMo (Long Conversational Memory) evaluates long-term memory across multi-session dialogues.

## Download Instructions
Download the dataset from the official [snap-research/locomo](https://snap-research.github.io/locomo) page or repo.

## Reproduction Commands
```bash
.venv/bin/python scripts/benchmark_locomo.py path/to/locomo.json --mode graph --output benchmarks/locomo/results_graph_raw.json
.venv/bin/python scripts/benchmark_locomo.py path/to/locomo.json --mode fusion --output benchmarks/locomo/results_graph_hybrid.json
```

## Methodology
Waggle ingests the conversational history through `observe_conversation` to build a typed knowledge graph. Retrieval is performed using `query_graph` in both graph-native and hybrid (fusion) modes.

