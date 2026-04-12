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
        # 2 QA pairs -> 2 batches
        assert len(batches) == 2

    def test_event_summary_entries(self):
        path = _write_fixture()
        batches = load_raw_locomo(path)
        # Events from Alice (2) + dialogue turns (3) = 5 updates per batch
        events = [u for u in batches[0].updates if u.provenance == "locomo_event_summary"]
        assert len(events) == 2
        assert events[0].entity == "Alice"

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

    def test_limit(self):
        path = _write_fixture()
        batches = load_raw_locomo(path, limit=1)
        # limit=1 limits samples, not batches, but 1 sample -> 2 QA -> 2 batches
        assert len(batches) == 2


class TestPreprocessRawLoCoMo:
    def test_integrity_stats(self):
        path = _write_fixture()
        dataset = preprocess_raw_locomo(path)
        assert dataset.source_dataset == "locomo"
        assert dataset.total_queries == 2
        assert dataset.dropped_records == 0
        assert dataset.total_updates > 0
