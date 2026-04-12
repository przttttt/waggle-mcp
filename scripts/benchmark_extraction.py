"""
Micro-benchmark for waggle-mcp extraction + retrieval accuracy.
Runs entirely locally. No Ollama required (uses regex pipeline).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile, json, time
from waggle.graph import MemoryGraph
from waggle.embeddings import EmbeddingModel

# ── 1. Extraction test cases ──────────────────────────────────────────────
# Each entry: (user_msg, assistant_msg, expected_node_types)
EXTRACTION_CASES = [
    (
        "Let's use PostgreSQL. MySQL replication has been painful.",
        "Sure, I'll update the stack to use PostgreSQL.",
        {"decision"},  # must contain at least one decision node
    ),
    (
        "I prefer dark mode everywhere.",
        "Noted, I'll keep that in mind.",
        {"preference"},
    ),
    (
        "The JWT tokens expire after 15 minutes.",
        "Got it. I'll document that in the auth spec.",
        {"fact"},
    ),
    (
        "Should we use GraphQL instead of REST?",
        "Both have tradeoffs. GraphQL gives you flexible queries but adds complexity.",
        {"question"},
    ),
    (
        "FastAPI is the right call here — async support is critical.",
        "Agreed. FastAPI it is.",
        {"decision"},
    ),
    (
        "The team is most familiar with Python.",
        "Understood. Let's keep the stack Python-first.",
        {"preference", "fact"},
    ),
    (
        "We decided to use Redis for caching session tokens.",
        "Good call. Redis is fast and TTL support makes it ideal for sessions.",
        {"decision"},
    ),
    (
        "What was the reason we dropped SQLite?",
        "We moved to Neo4j because SQLite doesn't support concurrent writes well.",
        {"question", "decision"},
    ),
]

# ── 2. Retrieval test cases ────────────────────────────────────────────────
# Store these facts, then check each query retrieves the right label.
RETRIEVAL_CASES = [
    {
        "label": "PostgreSQL decision",
        "content": "Chose PostgreSQL over MySQL due to painful MySQL replication",
        "node_type": "decision",
        "query": "what database did we choose",
        "expected_label_contains": "PostgreSQL",
    },
    {
        "label": "JWT token expiry",
        "content": "JWT tokens expire after 15 minutes",
        "node_type": "fact",
        "query": "how long do auth tokens last",
        "expected_label_contains": "JWT",
    },
    {
        "label": "dark mode preference",
        "content": "User strongly prefers dark mode on all interfaces",
        "node_type": "preference",
        "query": "user interface theme preference",
        "expected_label_contains": "dark mode",
    },
    {
        "label": "FastAPI backend choice",
        "content": "Chose FastAPI over Flask because async support is critical",
        "node_type": "decision",
        "query": "backend framework decision",
        "expected_label_contains": "FastAPI",
    },
    {
        "label": "Redis session caching",
        "content": "Redis is used for caching session tokens, with TTL support",
        "node_type": "decision",
        "query": "how are sessions stored",
        "expected_label_contains": "Redis",
    },
]

# ── 3. Deduplication test cases ────────────────────────────────────────────
# Each pair should merge into 1 node.
DEDUP_CASES = [
    (
        {"label": "Use Postgres", "content": "We decided to use PostgreSQL"},
        {"label": "PostgreSQL chosen", "content": "The team decided on PostgreSQL for the database"},
    ),
    (
        {"label": "Dark mode preferred", "content": "User prefers dark mode"},
        {"label": "Dark mode", "content": "User likes dark mode for the UI"},
    ),
    (
        {"label": "FastAPI choice", "content": "FastAPI was chosen as the backend framework"},
        {"label": "FastAPI selected", "content": "We chose FastAPI because of async support"},
    ),
]


def run_extraction_benchmark(graph):
    from waggle.intelligence import extract_conversation_candidates
    hits = 0
    for user_msg, assistant_msg, expected_types in EXTRACTION_CASES:
        candidates = extract_conversation_candidates(
            user_message=user_msg, assistant_response=assistant_msg
        )
        found_types = {str(c["node_type"].value) for c in candidates}
        if found_types & expected_types:
            hits += 1
    accuracy = hits / len(EXTRACTION_CASES)
    print(f"  Extraction accuracy:    {hits}/{len(EXTRACTION_CASES)} = {accuracy:.0%}")
    return accuracy


def run_retrieval_benchmark(graph):
    # Pre-store nodes
    from waggle.models import NodeType
    for case in RETRIEVAL_CASES:
        graph.add_node(
            label=case["label"],
            content=case["content"],
            node_type=NodeType(case["node_type"]),
            tags=["benchmark"],
        )

    hits = 0
    for case in RETRIEVAL_CASES:
        result = graph.query(query=case["query"], max_nodes=5)
        top_labels = [n.label.lower() for n in result.nodes]
        if any(case["expected_label_contains"].lower() in lbl for lbl in top_labels):
            hits += 1
    accuracy = hits / len(RETRIEVAL_CASES)
    print(f"  Retrieval accuracy:     {hits}/{len(RETRIEVAL_CASES)} = {accuracy:.0%}")
    return accuracy


def run_dedup_benchmark(graph):
    from waggle.models import NodeType
    hits = 0
    for pair in DEDUP_CASES:
        a, b = pair
        r1 = graph.add_node(label=a["label"], content=a["content"], node_type=NodeType.DECISION)
        r2 = graph.add_node(label=b["label"], content=b["content"], node_type=NodeType.DECISION)
        # A hit = the second add reused the first node (not created new)
        if not r2.created:
            hits += 1
    accuracy = hits / len(DEDUP_CASES)
    print(f"  Deduplication accuracy: {hits}/{len(DEDUP_CASES)} = {accuracy:.0%}")
    return accuracy


def main():
    print("=" * 55)
    print("waggle-mcp micro-benchmark")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.db")
        model = EmbeddingModel()
        graph = MemoryGraph(db_path=db_path, embedding_model=model)

        print("\n[1] Fact Extraction (regex pipeline)")
        ex = run_extraction_benchmark(graph)

        print("\n[2] Semantic Retrieval (local all-MiniLM-L6-v2)")
        ret = run_retrieval_benchmark(graph)

        print("\n[3] Deduplication (cosine similarity)")
        ded = run_dedup_benchmark(graph)

        print("\n" + "=" * 55)
        print(f"  Fact extraction      {ex:.0%}")
        print(f"  Retrieval relevance  {ret:.0%}")
        print(f"  Deduplication        {ded:.0%}")
        print("=" * 55)

        results = {
            "extraction": round(ex, 3),
            "retrieval": round(ret, 3),
            "deduplication": round(ded, 3),
        }
        out = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
