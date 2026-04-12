"""Tests for raw LongMemEval adapter with inline fixtures."""
import json
import tempfile
from pathlib import Path

from memory_inference.benchmarks.longmemeval_raw import (
    load_raw_longmemeval,
    preprocess_raw_longmemeval,
)
from memory_inference.consolidation.revision_types import QueryMode

FIXTURE = [
    {
        "question_id": "q_001",
        "question_type": "knowledge-update",
        "question": "What city does the user live in now?",
        "answer": "Boston",
        "question_date": "2024-01-20",
        "haystack_sessions": [
            {"role": "user", "content": "I just moved to New York.", "has_answer": False},
            {"role": "assistant", "content": "That sounds exciting!", "has_answer": False},
            {"role": "user", "content": "Actually I moved to Boston instead.", "has_answer": True},
        ],
        "answer_session_ids": ["sess_2"],
        "multi_attributes": [],
    },
    {
        "question_id": "q_002",
        "question_type": "temporal-reasoning",
        "question": "Where did the user live before Boston?",
        "answer": "New York",
        "haystack_sessions": [
            {"role": "user", "content": "I live in New York."},
            {"role": "user", "content": "I moved to Boston."},
        ],
        "answer_session_ids": ["sess_1"],
    },
    {
        "question_id": "q_003_abs",
        "question_type": "single-session-assistant",
        "question": "What did the assistant suggest?",
        "answer": "Try yoga",
        "haystack_dates": ["2024-01-21"],
        "haystack_session_ids": ["sess_3"],
        "haystack_sessions": [[
            {"role": "user", "content": "How do I relax more?"},
            {"role": "assistant", "content": "Try yoga.", "has_answer": True},
        ]],
        "answer_session_ids": ["sess_3"],
    },
]


def _write_fixture():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(FIXTURE, tmp)
    tmp.close()
    return tmp.name


class TestLoadRawLongMemEval:
    def test_basic_loading(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        assert len(batches) == 3

    def test_updates_from_sessions(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        assert len(batches[0].updates) == 3  # 3 dialogue turns
        assert batches[0].updates[0].entity == "user"
        assert batches[0].updates[0].attribute == "dialogue"
        assert batches[0].updates[0].importance != batches[0].updates[1].importance

    def test_query_mapping(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        q = batches[0].queries[0]
        assert q.question == "What city does the user live in now?"
        assert q.answer == "Boston"
        assert q.query_mode == QueryMode.CURRENT_STATE

    def test_temporal_query_mode(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        q = batches[1].queries[0]
        assert q.query_mode == QueryMode.HISTORY

    def test_assistant_question_targets_assistant_entity(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        q = batches[2].queries[0]
        assert q.entity == "assistant"

    def test_abstention_suffix_sets_supports_abstention(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path)
        q = batches[2].queries[0]
        assert q.supports_abstention is True

    def test_limit(self):
        path = _write_fixture()
        batches = load_raw_longmemeval(path, limit=1)
        assert len(batches) == 1


class TestPreprocessRawLongMemEval:
    def test_integrity_stats(self):
        path = _write_fixture()
        dataset = preprocess_raw_longmemeval(path)
        assert dataset.total_sessions == 3
        assert dataset.total_queries == 3
        assert dataset.total_updates == 7  # 3 + 2 + 2 turns
        assert dataset.dropped_records == 0
        assert dataset.source_dataset == "longmemeval"

    def test_empty_turns_skipped(self):
        fixture = [{"question_id": "q_x", "question": "q", "answer": "a",
                     "haystack_sessions": [{"role": "u", "content": "  "}]}]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        batches = load_raw_longmemeval(tmp.name)
        assert len(batches[0].updates) == 0

    def test_nested_sessions_preserve_source_dates(self):
        fixture = [{
            "question_id": "q_nested",
            "question": "When did the user move?",
            "answer": "2024-01-20",
            "haystack_dates": ["2024-01-20"],
            "haystack_session_ids": ["sess_1"],
            "haystack_sessions": [[
                {"role": "user", "content": "I moved to Boston yesterday."},
                {"role": "assistant", "content": "Noted."},
            ]],
        }]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        batches = load_raw_longmemeval(tmp.name)
        assert batches[0].updates[0].metadata["source_date"] == "2024-01-20"
        assert batches[0].updates[0].metadata["session_label"] == "sess_1"
        assert batches[0].updates[0].scope == "sess_1"
