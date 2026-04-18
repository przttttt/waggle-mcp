# ConvoMem Benchmark

ConvoMem evaluates conversational memory across six categories: User Facts, Assistant Facts, Changing Facts, Abstention, Preferences, and Implicit Connections.

## Download Instructions
Download the evidence files from the [Salesforce/ConvoMem](https://github.com/SalesforceAIResearch/ConvoMem) repository.

## Reproduction Commands
```bash
.venv/bin/python scripts/benchmark_convomem.py path/to/convomem_test.json --mode graph --output benchmarks/convomem/results_graph_raw.json
```

