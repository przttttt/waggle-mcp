from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import hashlib

import numpy as np

from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph

@dataclass
class MemBenchCaseResult:
    query_id: str
    category: str
    question: str
    hit_at_5: bool

@dataclass
class MemBenchReport:
    dataset_path: str
    mode: str
    case_count: int
    r_at_5: float
    cache_status: str
    cache_path: str
    per_case: list[MemBenchCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "case_count": self.case_count,
            "r_at_5": self.r_at_5,
            "cache_status": self.cache_status,
            "cache_path": self.cache_path,
            "per_case": [asdict(case) for case in self.per_case],
        }

def evaluate_membench(
    dataset_path: str | Path,
    *,
    embedding_model: Any | None = None,
    mode: Literal["graph", "replay", "fusion"] = "graph",
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> MemBenchReport:
    entries = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if limit:
        entries = entries[:limit]
    
    model_instance = embedding_model or EmbeddingModel()
    
    # Caching check
    cache_status = "cold"
    cache_path = ""
    
    results: list[MemBenchCaseResult] = []

    import tempfile
    for entry in entries:
        category = entry.get("category", "unknown")
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            graph = MemoryGraph(db_path=db_path, embedding_model=model_instance)
        
            # Ingest
            history = entry.get("history", [])
            for turn in history:
                graph.observe_conversation(user_message=str(turn.get("user", "")), assistant_response=str(turn.get("assistant", "")))
            
            # Retrieval
            question = entry["question"]
            gold_id = entry.get("gold_id")
            
            query_res = graph.query(query=question, max_nodes=5, retrieval_mode=mode)
            hit = any(node.id == gold_id for node in query_res.nodes)
            
            results.append(MemBenchCaseResult(
                query_id=entry.get("id", "q"),
                category=category,
                question=question,
                hit_at_5=hit
            ))

    case_count = len(results)
    overall_r5 = sum(1 for r in results if r.hit_at_5) / case_count if case_count else 0

    return MemBenchReport(
        dataset_path=str(dataset_path),
        mode=mode,
        case_count=case_count,
        r_at_5=overall_r5,
        cache_status=cache_status,
        cache_path=cache_path,
        per_case=results
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--mode", choices=["graph", "replay", "fusion"], default="graph")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_membench(args.dataset_path, mode=args.mode, limit=args.limit, cache_dir=args.cache_dir)
    print(f"Overall R@5: {report.r_at_5:.1%}")
    
    if args.output:
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
    
    return 0
