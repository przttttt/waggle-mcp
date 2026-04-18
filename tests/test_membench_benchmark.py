import json
from pathlib import Path
from waggle.membench_benchmark import evaluate_membench

def test_membench_adapter_runs_on_synthetic_data(tmp_path):
    dataset_path = tmp_path / "membench_test.json"
    dataset_path.write_text(json.dumps([
        {
            "id": "q1",
            "category": "factual",
            "question": "What database is used?",
            "gold_id": "node_1"
        }
    ]))
    
    report = evaluate_membench(dataset_path, limit=1)
    assert report.case_count == 1
    assert isinstance(report.r_at_5, float)
