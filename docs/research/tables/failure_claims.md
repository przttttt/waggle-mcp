# Failure Claims Summary

_Hardcoded claims table — update as evidence accumulates._

| Claim | Supported? | Evidence | Caveat |
|---|---|---|---|
| RMCA decomposition improves pairwise | ✅ Yes | ablation: no_decomp drops 1.0→0.0 | Synthetic data only |
| RMCA structured context improves LLM answerability | ⚠️ Partial | Groq F1=0.64 vs 0.00 at scale=128 | Single scale, single model |
| RMCA reduces injected tokens vs raw_context | ✅ Yes | S-NIAH: 14% of raw tokens | Synthetic data only |
| Graph expansion is load-bearing | ❌ Not yet | No delta at scale=128 | Need pairwise_hidden_edge |
| Conflict resolution is load-bearing | ❌ Not yet | No delta at scale=128 | Need pairwise_hidden_edge |
| RMCA solves ContextReset | ❌ Not yet | Score 0.0 in current setup | Scope/query bug being fixed |
| Results generalize to real traces | ❌ Not yet | Synthetic data only | Need real dataset runs |
