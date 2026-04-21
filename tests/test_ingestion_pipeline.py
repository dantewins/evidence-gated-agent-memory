import json
import tempfile

from memory_inference.ingestion.locomo_loader import load_locomo_samples
from memory_inference.ingestion.longmemeval_loader import load_longmemeval_records


def test_locomo_loader_parses_sessions_questions_and_event_summary() -> None:
    fixture = [
        {
            "sample_id": "sample_001",
            "conversation": {
                "session_1": [
                    {"dia_id": 0, "speaker": "Alice", "text": "I got a new job at Google."},
                ],
                "session_1_date_time": "2024-01-10",
            },
            "event_summary": {
                "Alice": ["Got a job at Google"],
            },
            "qa": [
                {
                    "question": "Where does Alice work now?",
                    "answer": "Google",
                    "category": 1,
                }
            ],
        }
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(fixture, tmp)
    tmp.close()

    samples = load_locomo_samples(tmp.name)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_001"
    assert samples[0].sessions[0].label == "session_1"
    assert samples[0].sessions[0].turns[0].speaker == "Alice"
    assert samples[0].event_summary["Alice"] == ["Got a job at Google"]
    assert samples[0].questions[0].question == "Where does Alice work now?"


def test_longmemeval_loader_normalizes_flat_and_nested_sessions() -> None:
    fixture = [
        {
            "question_id": "q_flat",
            "question_type": "knowledge-update",
            "question": "What city does the user live in now?",
            "answer": "Boston",
            "haystack_sessions": [
                {"role": "user", "content": "I moved to Boston.", "has_answer": True},
            ],
            "haystack_session_ids": ["sess_1"],
            "haystack_dates": ["2024-01-20"],
        },
        {
            "question_id": "q_nested",
            "question_type": "single-session-assistant",
            "question": "What did the assistant suggest?",
            "answer": "Try yoga",
            "haystack_sessions": [[
                {"role": "assistant", "content": "Try yoga.", "has_answer": True},
            ]],
            "haystack_session_ids": ["sess_2"],
            "haystack_dates": ["2024-01-21"],
        },
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(fixture, tmp)
    tmp.close()

    records = load_longmemeval_records(tmp.name)

    assert len(records) == 2
    assert records[0].sessions[0].label == "sess_1"
    assert records[0].sessions[0].turns[0].speaker == "user"
    assert records[1].sessions[0].label == "sess_2"
    assert records[1].sessions[0].turns[0].speaker == "assistant"
