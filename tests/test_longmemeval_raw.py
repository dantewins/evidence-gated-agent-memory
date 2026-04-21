"""Tests for raw LongMemEval adapter with inline fixtures."""
import json
import tempfile
from pathlib import Path

from memory_inference.domain.enums import QueryMode
from memory_inference.datasets.preprocessing import load_raw_longmemeval_dataset

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
        records = load_raw_longmemeval_dataset(path).records
        assert len(records) == 3

    def test_updates_from_sessions(self):
        path = _write_fixture()
        records = load_raw_longmemeval_dataset(path).records
        dialogue_updates = [u for u in records[0].context.updates if u.attribute == "dialogue"]
        structured_updates = [u for u in records[0].context.updates if u.attribute == "home_city"]
        assert len(dialogue_updates) == 3
        assert dialogue_updates[0].entity == "user"
        assert structured_updates
        assert {u.value for u in structured_updates} >= {"New York", "Boston"}
        assert dialogue_updates[0].importance != dialogue_updates[1].importance

    def test_query_mapping(self):
        path = _write_fixture()
        case = load_raw_longmemeval_dataset(path).records[0].cases[0]
        q = case.runtime_query
        assert q.question == "What city does the user live in now?"
        assert case.eval_target.gold_answer == "Boston"
        assert q.query_mode == QueryMode.CURRENT_STATE
        assert q.attribute == "home_city"

    def test_temporal_query_mode(self):
        path = _write_fixture()
        q = load_raw_longmemeval_dataset(path).records[1].cases[0].runtime_query
        assert q.query_mode == QueryMode.HISTORY
        assert q.timestamp == 2

    def test_assistant_question_targets_assistant_entity(self):
        path = _write_fixture()
        q = load_raw_longmemeval_dataset(path).records[2].cases[0].runtime_query
        assert q.entity == "assistant"

    def test_abstention_suffix_sets_supports_abstention(self):
        path = _write_fixture()
        q = load_raw_longmemeval_dataset(path).records[2].cases[0].runtime_query
        assert q.supports_abstention is True

    def test_limit(self):
        path = _write_fixture()
        records = load_raw_longmemeval_dataset(path, limit=1).records
        assert len(records) == 1


class TestPreprocessRawLongMemEval:
    def test_integrity_stats(self):
        path = _write_fixture()
        dataset = load_raw_longmemeval_dataset(path)
        assert dataset.total_contexts == 3
        assert dataset.total_queries == 3
        assert dataset.total_updates > 7  # dialogue turns plus extracted structured facts
        assert dataset.dropped_records == 0
        assert dataset.source_dataset == "longmemeval"
        assert dataset.records[0].cases[0].runtime_query.attribute == "home_city"

    def test_empty_turns_skipped(self):
        fixture = [{"question_id": "q_x", "question": "q", "answer": "a",
                     "haystack_sessions": [{"role": "u", "content": "  "}]}]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        records = load_raw_longmemeval_dataset(tmp.name).records
        assert len(records[0].context.updates) == 0

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
        records = load_raw_longmemeval_dataset(tmp.name).records
        assert records[0].context.updates[0].source_date == "2024-01-20"
        assert records[0].context.updates[0].session_label == "sess_1"
        assert records[0].context.updates[0].scope == "sess_1"

    def test_query_falls_back_to_dialogue_for_when_questions(self):
        fixture = [{
            "question_id": "q_when",
            "question": "When did the user move?",
            "answer": "2024-01-20",
            "haystack_dates": ["2024-01-20"],
            "haystack_session_ids": ["sess_1"],
            "haystack_sessions": [[
                {"role": "user", "content": "I moved to Boston yesterday."},
            ]],
        }]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        records = load_raw_longmemeval_dataset(tmp.name).records
        assert records[0].cases[0].runtime_query.attribute == "dialogue"

    def test_structured_facts_include_support_metadata_and_event_scope(self):
        fixture = [{
            "question_id": "q_playlist",
            "question": "What is the name of the playlist I created on Spotify?",
            "answer": "Summer Vibes",
            "haystack_dates": ["2024-01-20"],
            "haystack_session_ids": ["sess_1"],
            "haystack_sessions": [[
                {"role": "user", "content": "I created a playlist on Spotify called Summer Vibes."},
            ]],
        }]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        records = load_raw_longmemeval_dataset(tmp.name).records
        facts = [u for u in records[0].context.updates if u.attribute == "created_name"]

        assert facts
        assert facts[0].source_kind == "structured_fact"
        assert facts[0].source_attribute == "dialogue"
        assert facts[0].support_text == "I created a playlist on Spotify called Summer Vibes."
        assert facts[0].scope.startswith("sess_1:turn_0:fact_")
