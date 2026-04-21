from memory_inference.memory.policies import AppendOnlyMemoryPolicy
from memory_inference.memory.policies import ExactMatchMemoryPolicy
from memory_inference.memory.policies import StrongRetrievalMemoryPolicy
from memory_inference.datasets.normalized_io import NormalizedRecord
from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.orchestration.experiment import evaluate_structured_policy_full
from memory_inference.evaluation.targets import EvalTarget
from tests.factories import make_query, make_record


def test_exact_match_policy_preserves_multiple_scopes() -> None:
    policy = ExactMatchMemoryPolicy()
    policy.ingest(
        [
            make_record(
                entry_id="boston",
                entity="user",
                attribute="favorite_spot",
                value="Cafe Vittoria",
                timestamp=0,
                session_id="s",
                scope="boston",
            ),
            make_record(
                entry_id="miami",
                entity="user",
                attribute="favorite_spot",
                value="Joe's Stone Crab",
                timestamp=1,
                session_id="s",
                scope="miami",
            ),
        ]
    )

    retrieved = policy.retrieve("user", "favorite_spot")

    assert {entry.scope for entry in retrieved.entries} == {"boston", "miami"}


def test_strong_retrieval_prioritizes_exact_entity_and_attribute() -> None:
    policy = StrongRetrievalMemoryPolicy()
    policy.ingest(
        [
            make_record(
                entry_id="target",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=1,
                session_id="s",
            ),
            make_record(
                entry_id="distractor",
                entity="friend",
                attribute="home_city",
                value="Boston",
                timestamp=2,
                session_id="s",
            ),
            make_record(
                entry_id="wrong-attr",
                entity="user",
                attribute="employer",
                value="Google",
                timestamp=3,
                session_id="s",
            ),
        ]
    )
    query = make_query(
        query_id="q1",
        entity="user",
        attribute="home_city",
        question="Where does the user live now?",
        answer="Boston",
        timestamp=4,
        session_id="s",
    )

    top = policy.retrieve_for_query(query, top_k=1).entries[0]

    assert top.entity == query.entity
    assert top.attribute == query.attribute


def test_structured_evaluation_resets_policy_state_per_batch() -> None:
    records = [
        _normalized_record(
            context_id="batch-1",
            updates=[
                make_record(
                    entry_id="u1",
                    entity="user",
                    attribute="home_city",
                    value="Boston",
                    timestamp=0,
                    session_id="batch-1",
                )
            ],
            queries=[
                make_query(
                    query_id="q1",
                    entity="user",
                    attribute="home_city",
                    question="Where do I live?",
                    timestamp=1,
                    session_id="batch-1",
                )
            ],
            answers=["Boston"],
        ),
        _normalized_record(
            context_id="batch-2",
            updates=[
                make_record(
                    entry_id="u2",
                    entity="user",
                    attribute="home_city",
                    value="Seattle",
                    timestamp=0,
                    session_id="batch-2",
                )
            ],
            queries=[
                make_query(
                    query_id="q2",
                    entity="user",
                    attribute="home_city",
                    question="Where do I live?",
                    timestamp=1,
                    session_id="batch-2",
                )
            ],
            answers=["Seattle"],
        ),
    ]

    result = evaluate_structured_policy_full(
        AppendOnlyMemoryPolicy,
        DeterministicValidityReader(),
        records,
    )

    assert result.metrics.accuracy == 1.0
    assert [case.correct for case in result.evaluated_cases] == [True, True]


def test_structured_evaluation_does_not_double_ingest_repeated_full_context_batches() -> None:
    updates = [
        make_record(
            entry_id="u1",
            entity="user",
            attribute="dialogue",
            value="I live in Boston.",
            timestamp=0,
            session_id="sample-1",
        ),
        make_record(
            entry_id="u2",
            entity="user",
            attribute="dialogue",
            value="I graduated with Business Administration.",
            timestamp=1,
            session_id="sample-1",
        ),
    ]
    records = [
        _normalized_record(
            context_id="sample-1-q1",
            updates=list(updates),
            queries=[
                make_query(
                    query_id="q1",
                    entity="user",
                    attribute="dialogue",
                    question="Where do I live?",
                    timestamp=2,
                    session_id="sample-1",
                )
            ],
            answers=["Boston"],
        ),
        _normalized_record(
            context_id="sample-1-q2",
            updates=list(updates),
            queries=[
                make_query(
                    query_id="q2",
                    entity="user",
                    attribute="dialogue",
                    question="What degree did I graduate with?",
                    timestamp=2,
                    session_id="sample-1",
                )
            ],
            answers=["Business Administration"],
        ),
    ]

    result = evaluate_structured_policy_full(
        AppendOnlyMemoryPolicy,
        DeterministicValidityReader(),
        records,
    )

    assert len(result.evaluated_cases) == 2
    assert len(result.evaluated_cases[0].retrieval_bundle.records) == len(updates)
    assert len(result.evaluated_cases[1].retrieval_bundle.records) == len(updates)


def _normalized_record(context_id: str, updates, queries, answers) -> NormalizedRecord:
    context = ExperimentContext(
        context_id=context_id,
        session_id=context_id,
        updates=list(updates),
    )
    cases = [
        ExperimentCase(
            case_id=query.query_id,
            context_id=context_id,
            runtime_query=query,
            eval_target=EvalTarget(query_id=query.query_id, gold_answer=answer),
        )
        for query, answer in zip(queries, answers)
    ]
    return NormalizedRecord(
        schema_version="test",
        source_dataset="test",
        source_split="test",
        source_record_id=context_id,
        context=context,
        cases=cases,
    )
