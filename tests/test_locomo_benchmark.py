import json
from pathlib import Path
from waggle.locomo_benchmark import evaluate_locomo

def test_locomo_adapter_runs_on_synthetic_data(tmp_path):
    dataset_path = tmp_path / "locomo_test.json"
    dataset_path.write_text(json.dumps([
        {
            "dia_id": "test_1",
            "sessions": [
                {
                    "id": "session_1",
                    "date_time": "2024-01-01T12:00:00Z",
                    "turns": [
                        {"role": "user", "content": "The user likes PostgreSQL."},
                        {"role": "assistant", "content": "Noted."}
                    ]
                }
            ],
            "qa": [
                {
                    "id": "q1",
                    "question": "What database does the user like?",
                    "correct_session_ids": ["session_1"]
                }
            ]
        }
    ]))
    
    report = evaluate_locomo(dataset_path, limit=1)
    assert report.case_count == 1
    # Note: Accuracy depends on embeddings/extraction, but we just verify it runs
    assert isinstance(report.r_at_5, float)
    assert isinstance(report.r_at_10, float)
