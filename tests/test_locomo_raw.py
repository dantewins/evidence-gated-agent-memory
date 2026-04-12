"""Tests for raw LoCoMo adapter with inline fixtures."""
import json
import tempfile

from memory_inference.benchmarks.locomo_raw import (
    load_raw_locomo,
    preprocess_raw_locomo,
)
from memory_inference.consolidation.revision_types import QueryMode

FIXTURE = [
    {
        "sample_id": "sample_001",
        "conversation": {
            "session_1": [
                {"dia_id": 0, "speaker": "Alice", "text": "I got a new job at Google."},
                {"dia_id": 1, "speaker": "Bob", "text": "Congrats! What role?"},
            ],
            "session_1_date_time": "2024-01-10",
            "session_2": [
                {"dia_id": 0, "speaker": "Alice", "text": "I actually switched to Meta."},
            ],
            "session_2_date_time": "2024-02-15",
        },
        "event_summary": {
            "Alice": ["Got a job at Google", "Switched to Meta"],
            "Bob": [],
        },
        "qa": [
            {
                "question": "Where does Alice work now?",
                "answer": "Meta",
                "category": 1,
                "evidence": ["1-0"],
            },
            {
                "question": "Where did Alice work before Meta?",
                "answer": "Google",
                "category": 2,
                "evidence": ["0-0"],
            },
        ],
    },
    {
        "sample_id": "sample_002",
        "conversation": {
            "session_1": [
                {"dia_id": 0, "speaker": "Caroline", "text": "I moved from Sweden four years ago."},
            ],
            "session_1_date_time": "2024-03-01",
        },
        "event_summary": {
            "Caroline": ["Moved from Sweden four years ago"],
        },
        "qa": [
            {
                "question": "Where did Caroline move from?",
                "answer": "Sweden",
                "category": 1,
                "evidence": ["D1:1"],
            },
            {
                "question": "Would Caroline be likely to own a spaceship?",
                "category": 5,
                "evidence": [],
            },
        ],
    },
]


def _write_fixture():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(FIXTURE, tmp)
    tmp.close()
    return tmp.name


class TestLoadRawLoCoMo:
    def test_basic_loading(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        # 2 scored QA pairs in sample_001 + 1 scored QA in sample_002
        assert len(batches) == 3

    def test_event_summary_entries(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        events = [u for u in batches[0].updates if u.provenance == "locomo_event_summary"]
        assert len(events) == 2
        assert events[0].entity == "Alice"
        employers = [u for u in batches[0].updates if u.attribute == "employer"]
        assert {u.value for u in employers} >= {"Google", "Meta"}

    def test_dialogue_entries(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        dialogues = [u for u in batches[0].updates if u.provenance == "locomo_dialogue"]
        assert len(dialogues) == 3  # 2 from session_1 + 1 from session_2

    def test_query_mapping(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        q0 = batches[0].queries[0]
        assert q0.question == "Where does Alice work now?"
        assert q0.answer == "Meta"
        assert q0.query_mode == QueryMode.CURRENT_STATE
        assert q0.entity == "Alice"
        assert q0.attribute == "employer"

    def test_temporal_category(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        q1 = batches[1].queries[0]
        assert q1.query_mode == QueryMode.HISTORY

    def test_dialogue_entries_include_session_date(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        dialogues = [u for u in batches[0].updates if u.provenance == "locomo_dialogue"]
        assert dialogues[0].metadata["source_date"] == "2024-01-10"
        assert dialogues[0].scope == "session_1"

    def test_limit(self):
        path = _write_fixture()
        batches = load_raw_locomo(path, limit=1)
        # limit=1 limits samples, not batches, but 1 sample -> 2 QA -> 2 batches
        assert len(batches) == 2

    def test_missing_answer_adversarial_question_is_skipped(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        questions = [batch.queries[0].question for batch in batches]
        assert "Would Caroline be likely to own a spaceship?" not in questions
        assert "Where did Caroline move from?" in questions
        followup = next(batch for batch in batches if batch.queries[0].question == "Where did Caroline move from?")
        assert followup.queries[0].attribute == "origin"


class TestPreprocessRawLoCoMo:
    def test_integrity_stats(self):
        path = _write_fixture()
        dataset = preprocess_raw_locomo(path)
        assert dataset.source_dataset == "locomo"
        assert dataset.total_queries == 3
        assert dataset.dropped_records == 0
        assert dataset.total_updates > 0
