from memory_inference.datasets.compiler import (
    compile_locomo_samples,
    compile_longmemeval_records,
)
from memory_inference.datasets.normalized_io import load_normalized, serialize_normalized
from memory_inference.domain.benchmark import (
    RawConversationSession,
    RawConversationTurn,
    RawLoCoMoQuestion,
    RawLoCoMoSample,
    RawLongMemEvalRecord,
)
from memory_inference.domain.enums import QueryMode


def test_compile_locomo_samples_preserves_query_semantics() -> None:
    sample = RawLoCoMoSample(
        sample_id="sample_001",
        sessions=[
            RawConversationSession(
                label="session_1",
                date="2024-01-10",
                turns=[
                    RawConversationTurn(
                        speaker="Alice",
                        text="I got a new job at Google.",
                        turn_id="0",
                    )
                ],
            )
        ],
        event_summary={"Alice": ["Got a job at Google"]},
        questions=[
            RawLoCoMoQuestion(
                question="Where does Alice work now?",
                answer="Google",
                category="1",
            )
        ],
    )

    dataset = compile_locomo_samples([sample])

    assert dataset.total_queries == 1
    assert dataset.total_contexts == 1
    record = dataset.records[0]
    assert record.context.context_id == "sample_001"
    assert record.cases[0].runtime_query.attribute == "employer"
    assert record.cases[0].runtime_query.query_mode == QueryMode.CURRENT_STATE
    assert any(update.attribute == "employer" for update in record.context.updates)


def test_compile_longmemeval_records_preserves_temporal_mode() -> None:
    record = RawLongMemEvalRecord(
        question_id="q_001",
        question_type="temporal-reasoning",
        question="Where did the user live before Boston?",
        answer="New York",
        sessions=[
            RawConversationSession(
                label="sess_1",
                date="2024-01-20",
                turns=[
                    RawConversationTurn(speaker="user", text="I live in New York."),
                    RawConversationTurn(speaker="user", text="I moved to Boston."),
                ],
            )
        ],
    )

    dataset = compile_longmemeval_records([record])

    assert dataset.total_queries == 1
    assert dataset.total_contexts == 1
    compiled = dataset.records[0]
    assert compiled.cases[0].runtime_query.query_mode == QueryMode.HISTORY
    assert compiled.cases[0].runtime_query.attribute == "home_city"


def test_compile_locomo_samples_uses_shared_context_with_multiple_cases() -> None:
    sample = RawLoCoMoSample(
        sample_id="sample_shared",
        sessions=[
            RawConversationSession(
                label="session_1",
                date="2024-01-10",
                turns=[RawConversationTurn(speaker="Alice", text="I work at Google.", turn_id="0")],
            )
        ],
        event_summary={"Alice": ["Started a job at Google"]},
        questions=[
            RawLoCoMoQuestion(question="Where does Alice work now?", answer="Google", category="1"),
            RawLoCoMoQuestion(question="Where did Alice work before now?", answer="Google", category="2"),
        ],
    )

    dataset = compile_locomo_samples([sample])

    assert dataset.total_contexts == 1
    assert dataset.total_queries == 2
    assert len(dataset.records) == 1
    assert len(dataset.records[0].cases) == 2
    assert dataset.records[0].cases[0].context_id == dataset.records[0].context.context_id


def test_compiled_dataset_round_trips_via_normalized_io(tmp_path) -> None:
    record = RawLongMemEvalRecord(
        question_id="q_roundtrip",
        question_type="knowledge-update",
        question="What city does the user live in now?",
        answer="Boston",
        sessions=[
            RawConversationSession(
                label="sess_1",
                date="2024-01-20",
                turns=[RawConversationTurn(speaker="user", text="I moved to Boston.")],
            )
        ],
    )

    dataset = compile_longmemeval_records([record])
    output = tmp_path / "compiled.json"
    serialize_normalized(dataset, output)
    restored = load_normalized(output)

    assert restored.schema_version == dataset.schema_version
    assert restored.total_contexts == 1
    assert restored.total_queries == 1
    assert restored.records[0].context.context_id == "q_roundtrip"
    assert restored.records[0].cases[0].runtime_query.attribute == "home_city"
