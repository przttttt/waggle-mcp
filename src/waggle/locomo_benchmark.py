from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph, NodeType
from waggle.models import Node

@dataclass
class LoCoMoCaseResult:
    query_id: str
    question: str
    correct_session_ids: list[str]
    retrieved_session_ids: list[str]
    hit_at_5: bool
    hit_at_10: bool

@dataclass
class LoCoMoReport:
    dataset_path: str
    mode: str
    case_count: int
    cache_status: str
    cache_path: str
    r_at_5: float
    r_at_10: float
    per_case: list[LoCoMoCaseResult]
    split_type: str = "full"
    split_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "case_count": self.case_count,
            "cache_status": self.cache_status,
            "cache_path": self.cache_path,
            "r_at_5": self.r_at_5,
            "r_at_10": self.r_at_10,
            "per_case": [asdict(case) for case in self.per_case],
        }

def _load_locomo_entries(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def evaluate_locomo(
    dataset_path: str | Path,
    *,
    embedding_model: Any | None = None,
    mode: Literal["graph", "replay", "fusion"] = "graph",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
) -> LoCoMoReport:
    entries = _load_locomo_entries(dataset_path)
    if limit:
        entries = entries[:limit]

    model_instance = embedding_model or EmbeddingModel()
    results: list[LoCoMoCaseResult] = []

    import tempfile
    for dialogue in entries:
        dia_id = dialogue.get("dia_id", "default")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            graph = MemoryGraph(db_path=db_path, embedding_model=model_instance, tenant_id=dia_id)
            
            for session in dialogue.get("sessions", []):
                session_id = session.get("id") or session.get("session_id")
                # Simplified ingestion for the benchmark:
                turns = session.get("turns", [])
                for i in range(0, len(turns), 2):
                    u = turns[i]
                    a = turns[i+1] if i+1 < len(turns) else {"role": "assistant", "content": "..."}
                    graph.observe_conversation(
                        user_message=u.get("content", ""),
                        assistant_response=a.get("content", ""),
                        session_id=str(session_id)
                    )

            for qa in dialogue.get("qa", []):
                question = qa["question"]
                gold_ids = qa.get("correct_session_ids") or qa.get("answer_session_ids", [])
                
                query_res = graph.query(
                    query=question,
                    max_nodes=10,
                    retrieval_mode=mode
                )
                
                retrieved_session_ids = []
                for node in query_res.nodes:
                    if node.session_id and node.session_id not in retrieved_session_ids:
                        retrieved_session_ids.append(node.session_id)
                
                hit_at_5 = any(sid in retrieved_session_ids[:5] for sid in gold_ids)
                hit_at_10 = any(sid in retrieved_session_ids[:10] for sid in gold_ids)
                
                results.append(
                    LoCoMoCaseResult(
                        query_id=qa.get("id") or qa.get("q_id", "q"),
                        question=question,
                        correct_session_ids=gold_ids,
                        retrieved_session_ids=retrieved_session_ids,
                        hit_at_5=hit_at_5,
                        hit_at_10=hit_at_10
                    )
                )

    case_count = len(results)
    r5 = sum(1 for r in results if r.hit_at_5) / case_count if case_count else 0
    r10 = sum(1 for r in results if r.hit_at_10) / case_count if case_count else 0

    return LoCoMoReport(
        dataset_path=str(dataset_path),
        mode=mode,
        case_count=case_count,
        cache_status="cold",
        cache_path="",
        r_at_5=r5,
        r_at_10=r10,
        per_case=results
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--mode", choices=["graph", "replay", "fusion"], default="graph")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_locomo(args.dataset_path, mode=args.mode, limit=args.limit)
    print(f"R@5: {report.r_at_5:.1%}")
    print(f"R@10: {report.r_at_10:.1%}")
    
    if args.output:
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
    
    return 0
