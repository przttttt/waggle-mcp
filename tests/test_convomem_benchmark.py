import json
from pathlib import Path
from waggle.convomem_benchmark import evaluate_convomem

def test_convomem_adapter_runs_on_synthetic_data(tmp_path):
    dataset_path = tmp_path / "convomem_test.json"
    dataset_path.write_text(json.dumps([
        {
            "id": "q1",
            "category": "Preferences",
            "question": "What is the user's favorite database?",
            "answer": "PostgreSQL",
            "messages": [
                {"role": "user", "content": "I love PostgreSQL."}
            ]
        }
    ]))
    
    report = evaluate_convomem(dataset_path, limit=1)
    assert report.case_count == 1
    assert "Preferences" in report.per_category
