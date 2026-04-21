"""Tests for raw LoCoMo adapter with inline fixtures."""
import json
import tempfile

from memory_inference.domain.enums import QueryMode
from memory_inference.datasets.preprocessing import load_raw_locomo_dataset

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
        records = load_raw_locomo_dataset(path).records
        assert len(records) == 2
        assert len(records[0].cases) == 2
        assert len(records[1].cases) == 1

    def test_event_summary_entries(self):
        path = _write_fixture()
        records = load_raw_locomo_dataset(path).records
        events = [u for u in records[0].context.updates if u.provenance == "locomo_event_summary"]
        assert len(events) == 2
        assert events[0].entity == "Alice"
        employers = [u for u in records[0].context.updates if u.attribute == "employer"]
        assert {u.value for u in employers} >= {"Google", "Meta"}

    def test_dialogue_entries(self):
        path = _write_fixture()
        records = load_raw_locomo_dataset(path).records
        dialogues = [u for u in records[0].context.updates if u.provenance == "locomo_dialogue"]
        assert len(dialogues) == 3  # 2 from session_1 + 1 from session_2

    def test_query_mapping(self):
        path = _write_fixture()
        case = load_raw_locomo_dataset(path).records[0].cases[0]
        q0 = case.runtime_query
        assert q0.question == "Where does Alice work now?"
        assert case.eval_target.gold_answer == "Meta"
        assert q0.query_mode == QueryMode.CURRENT_STATE
        assert q0.entity == "Alice"
        assert q0.attribute == "employer"

    def test_temporal_category(self):
        path = _write_fixture()
        q1 = load_raw_locomo_dataset(path).records[0].cases[1].runtime_query
        assert q1.query_mode == QueryMode.HISTORY

    def test_dialogue_entries_include_session_date(self):
        path = _write_fixture()
        records = load_raw_locomo_dataset(path).records
        dialogues = [u for u in records[0].context.updates if u.provenance == "locomo_dialogue"]
        assert dialogues[0].source_date == "2024-01-10"
        assert dialogues[0].scope == "session_1"

    def test_limit(self):
        path = _write_fixture()
        records = load_raw_locomo_dataset(path, limit=1).records
        assert len(records) == 1
        assert len(records[0].cases) == 2

    def test_missing_answer_adversarial_question_is_skipped(self):
        path = _write_fixture()
        records = load_raw_locomo_dataset(path).records
        questions = [case.runtime_query.question for record in records for case in record.cases]
        assert "Would Caroline be likely to own a spaceship?" not in questions
        assert "Where did Caroline move from?" in questions
        followup = next(
            case.runtime_query
            for record in records
            for case in record.cases
            if case.runtime_query.question == "Where did Caroline move from?"
        )
        assert followup.attribute == "origin"

    def test_non_temporal_unstructured_locomo_questions_default_to_dialogue(self):
        fixture = [
            {
                "sample_id": "sample_event_fallback",
                "conversation": {
                    "session_1": [
                        {"dia_id": 0, "speaker": "Caroline", "text": "I visited the science museum with friends."},
                    ],
                    "session_1_date_time": "2024-03-01",
                },
                "event_summary": {
                    "Caroline": ["Visited the science museum with friends"],
                },
                "qa": [
                    {
                        "question": "What did Caroline do with friends?",
                        "answer": "Visited the science museum",
                        "category": 1,
                    },
                ],
            }
        ]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(fixture, tmp)
        tmp.close()
        records = load_raw_locomo_dataset(tmp.name).records

        assert records[0].cases[0].runtime_query.attribute == "dialogue"


class TestPreprocessRawLoCoMo:
    def test_integrity_stats(self):
        path = _write_fixture()
        dataset = load_raw_locomo_dataset(path)
        assert dataset.source_dataset == "locomo"
        assert dataset.total_contexts == 2
        assert dataset.total_queries == 3
        assert dataset.dropped_records == 0
        assert dataset.total_updates > 0
        assert len(dataset.records[0].cases) == 2
