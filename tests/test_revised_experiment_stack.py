import json

from memory_inference.memory.policies import AppendOnlyMemoryPolicy
from memory_inference.memory.policies import ExactMatchMemoryPolicy
from memory_inference.memory.policies import StrongRetrievalMemoryPolicy
from memory_inference.datasets.normalized_io import NormalizedDataset, NormalizedRecord
from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.orchestration.experiment import (
    evaluate_structured_policy_full,
    run_dataset_experiment,
)
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


def test_structured_evaluation_batches_reader_across_contexts_and_reports_progress() -> None:
    class BatchTrackingReasoner(DeterministicValidityReader):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def answer_many_with_traces(self, queries, contexts):
            self.batch_sizes.append(len(queries))
            return [
                DeterministicValidityReader.answer_with_trace(self, query, context)
                for query, context in zip(queries, contexts)
            ]

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
    reasoner = BatchTrackingReasoner()
    events = []

    result = evaluate_structured_policy_full(
        AppendOnlyMemoryPolicy,
        reasoner,
        records,
        reader_flush_size=8,
        progress_callback=events.append,
        benchmark_name="test",
    )

    assert result.metrics.accuracy == 1.0
    assert reasoner.batch_sizes == [2]
    assert [event.phase for event in events].count("case_prepared") == 2
    assert [event.phase for event in events].count("case_finished") == 2


def test_dataset_experiment_streams_cases_and_reports_policy_finished(tmp_path) -> None:
    records = [
        _normalized_record(
            context_id="ctx",
            updates=[
                make_record(
                    entry_id="u1",
                    entity="user",
                    attribute="home_city",
                    value="Boston",
                    timestamp=0,
                    session_id="ctx",
                )
            ],
            queries=[
                make_query(
                    query_id="q1",
                    entity="user",
                    attribute="home_city",
                    question="Where do I live?",
                    timestamp=1,
                    session_id="ctx",
                )
            ],
            answers=["Boston"],
        )
    ]
    dataset = NormalizedDataset(records=records, total_contexts=1, total_cases=1)
    cases_output = tmp_path / "cases.jsonl"
    events = []

    result = run_dataset_experiment(
        benchmark_name="test",
        dataset=dataset,
        reasoner=DeterministicValidityReader(),
        policy_factories=[AppendOnlyMemoryPolicy],
        cases_output=str(cases_output),
        progress_callback=events.append,
    )

    rows = [json.loads(line) for line in cases_output.read_text().splitlines()]
    assert result.metrics[0].accuracy == 1.0
    assert rows[0]["policy_name"] == "append_only"
    assert rows[0]["case_id"] == "q1"
    assert [event.phase for event in events][-1] == "policy_finished"


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
