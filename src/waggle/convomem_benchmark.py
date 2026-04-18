from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph

@dataclass
class ConvoMemCaseResult:
    query_id: str
    category: str
    question: str
    hit_at_5: bool

@dataclass
class ConvoMemReport:
    dataset_path: str
    mode: str
    case_count: int
    r_at_5: float
    per_category: dict[str, float]
    per_case: list[ConvoMemCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "case_count": self.case_count,
            "r_at_5": self.r_at_5,
            "per_category": self.per_category,
            "per_case": [asdict(case) for case in self.per_case],
        }

def evaluate_convomem(
    dataset_path: str | Path,
    *,
    embedding_model: Any | None = None,
    mode: Literal["graph", "replay", "fusion"] = "graph",
    limit: int | None = None,
) -> ConvoMemReport:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("data") or data.get("questions") or [data]
    else:
        entries = data
    
    if limit:
        entries = entries[:limit]
    
    model_instance = embedding_model or EmbeddingModel()
    results: list[ConvoMemCaseResult] = []
    categories: set[str] = set()

    import tempfile
    for entry in entries:
        category = entry.get("category", "unknown")
        categories.add(category)
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            graph = MemoryGraph(db_path=db_path, embedding_model=model_instance)
        
            # Ingest history
            for msg in entry.get("messages", []):
                # Simple ingestion if role/content exists
                # For smoke test purposes, we'll assume a list of dicts
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    graph.observe_conversation(user_message=content, assistant_response="...")
            
            # Retrieval
            question = entry["question"]
            gold_text = entry.get("answer", "")
            
            query_res = graph.query(query=question, max_nodes=5, retrieval_mode=mode)
            hit = any(gold_text.lower() in node.content.lower() for node in query_res.nodes)
            
            results.append(ConvoMemCaseResult(
                query_id=entry.get("id", "q"),
                category=category,
                question=question,
                hit_at_5=hit
            ))

    case_count = len(results)
    overall_r5 = sum(1 for r in results if r.hit_at_5) / case_count if case_count else 0
    
    per_cat: dict[str, float] = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        per_cat[cat] = sum(1 for r in cat_results if r.hit_at_5) / len(cat_results) if cat_results else 0

    return ConvoMemReport(
        dataset_path=str(dataset_path),
        mode=mode,
        case_count=case_count,
        r_at_5=overall_r5,
        per_category=per_cat,
        per_case=results
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--mode", choices=["graph", "replay", "fusion"], default="graph")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_convomem(args.dataset_path, mode=args.mode, limit=args.limit)
    print(f"Overall R@5: {report.r_at_5:.1%}")
    for cat, score in report.per_category.items():
        print(f"  {cat}: {score:.1%}")
    
    if args.output:
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
    
    return 0
