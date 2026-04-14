from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.exact_match import ExactMatchMemoryPolicy
from memory_inference.consolidation.strong_retrieval import StrongRetrievalMemoryPolicy
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.run_experiment import evaluate_structured_policy_full
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query


def test_exact_match_policy_preserves_multiple_scopes() -> None:
    policy = ExactMatchMemoryPolicy()
    policy.ingest(
        [
            MemoryEntry(
                entry_id="boston",
                entity="user",
                attribute="favorite_spot",
                value="Cafe Vittoria",
                timestamp=0,
                session_id="s",
                scope="boston",
            ),
            MemoryEntry(
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
            MemoryEntry(
                entry_id="target",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=1,
                session_id="s",
            ),
            MemoryEntry(
                entry_id="distractor",
                entity="friend",
                attribute="home_city",
                value="Boston",
                timestamp=2,
                session_id="s",
            ),
            MemoryEntry(
                entry_id="wrong-attr",
                entity="user",
                attribute="employer",
                value="Google",
                timestamp=3,
                session_id="s",
            ),
        ]
    )
    query = Query(
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
    batches = [
        BenchmarkBatch(
            session_id="batch-1",
            updates=[
                MemoryEntry(
                    entry_id="u1",
                    entity="user",
                    attribute="dialogue",
                    value="I live in Boston.",
                    timestamp=0,
                    session_id="batch-1",
                )
            ],
            queries=[
                Query(
                    query_id="q1",
                    entity="user",
                    attribute="dialogue",
                    question="Where do I live?",
                    answer="Boston",
                    timestamp=1,
                    session_id="batch-1",
                )
            ],
        ),
        BenchmarkBatch(
            session_id="batch-2",
            updates=[
                MemoryEntry(
                    entry_id="u2",
                    entity="user",
                    attribute="dialogue",
                    value="I live in Seattle.",
                    timestamp=0,
                    session_id="batch-2",
                )
            ],
            queries=[
                Query(
                    query_id="q2",
                    entity="user",
                    attribute="dialogue",
                    question="Where do I live?",
                    answer="Seattle",
                    timestamp=1,
                    session_id="batch-2",
                )
            ],
        ),
    ]

    result = evaluate_structured_policy_full(
        AppendOnlyMemoryPolicy,
        DeterministicValidityReader(),
        batches,
    )

    assert result.metrics.accuracy == 1.0
    assert [example.correct for example in result.examples] == [True, True]


def test_structured_evaluation_does_not_double_ingest_repeated_full_context_batches() -> None:
    updates = [
        MemoryEntry(
            entry_id="u1",
            entity="user",
            attribute="dialogue",
            value="I live in Boston.",
            timestamp=0,
            session_id="sample-1",
        ),
        MemoryEntry(
            entry_id="u2",
            entity="user",
            attribute="dialogue",
            value="I graduated with Business Administration.",
            timestamp=1,
            session_id="sample-1",
        ),
    ]
    batches = [
        BenchmarkBatch(
            session_id="sample-1-q1",
            updates=list(updates),
            queries=[
                Query(
                    query_id="q1",
                    entity="user",
                    attribute="dialogue",
                    question="Where do I live?",
                    answer="Boston",
                    timestamp=2,
                    session_id="sample-1",
                )
            ],
        ),
        BenchmarkBatch(
            session_id="sample-1-q2",
            updates=list(updates),
            queries=[
                Query(
                    query_id="q2",
                    entity="user",
                    attribute="dialogue",
                    question="What degree did I graduate with?",
                    answer="Business Administration",
                    timestamp=2,
                    session_id="sample-1",
                )
            ],
        ),
    ]

    result = evaluate_structured_policy_full(
        AppendOnlyMemoryPolicy,
        DeterministicValidityReader(),
        batches,
    )

    assert len(result.examples) == 2
    assert len(result.examples[0].retrieved) == len(updates)
    assert len(result.examples[1].retrieved) == len(updates)
