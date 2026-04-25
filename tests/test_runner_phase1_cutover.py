from memory_inference.memory.policies import AppendOnlyMemoryPolicy
from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.scoring import evaluate_executed_cases
from memory_inference.evaluation.targets import EvalTarget
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.orchestration.runner import ContextCaseRunner
import pytest


def test_runner_executes_new_context_case_contract() -> None:
    context = ExperimentContext(
        context_id="ctx-1",
        session_id="ctx-1",
        updates=[
            MemoryRecord(
                record_id="e1",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=0,
                session_id="ctx-1",
            )
        ],
    )
    case = ExperimentCase(
        case_id="q1",
        context_id="ctx-1",
        runtime_query=RuntimeQuery(
            query_id="q1",
            context_id="ctx-1",
            entity="user",
            attribute="home_city",
            question="Where do I live?",
            timestamp=1,
            session_id="ctx-1",
        ),
        eval_target=EvalTarget(query_id="q1", gold_answer="Boston"),
    )

    runner = ContextCaseRunner(policy=AppendOnlyMemoryPolicy(), reasoner=DeterministicValidityReader())
    executed = runner.run_cases_for_context(context, [case])
    results = evaluate_executed_cases(executed)

    assert len(results) == 1
    assert results[0].prediction == "Boston"
    assert results[0].correct is True
    assert results[0].case.eval_target.gold_answer == "Boston"


def test_runner_output_is_gold_free_until_scoring() -> None:
    context = ExperimentContext(
        context_id="ctx-1",
        session_id="ctx-1",
        updates=[
            MemoryRecord(
                record_id="e1",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=0,
                session_id="ctx-1",
            )
        ],
    )
    case = ExperimentCase(
        case_id="q1",
        context_id="ctx-1",
        runtime_query=RuntimeQuery(
            query_id="q1",
            context_id="ctx-1",
            entity="user",
            attribute="home_city",
            question="Where do I live?",
            timestamp=1,
            session_id="ctx-1",
        ),
        eval_target=EvalTarget(query_id="q1", gold_answer="Boston"),
    )

    runner = ContextCaseRunner(policy=AppendOnlyMemoryPolicy(), reasoner=DeterministicValidityReader())
    executed = runner.run_cases_for_context(context, [case])

    assert len(executed) == 1
    assert executed[0].prediction == "Boston"
    assert not hasattr(executed[0], "correct")


def test_runner_fails_fast_if_reused_across_different_contexts() -> None:
    runner = ContextCaseRunner(policy=AppendOnlyMemoryPolicy(), reasoner=DeterministicValidityReader())
    runner.prepare_context(
        ExperimentContext(
            context_id="ctx-1",
            session_id="ctx-1",
            updates=[
                MemoryRecord(
                    record_id="e1",
                    entity="user",
                    attribute="home_city",
                    value="Boston",
                    timestamp=0,
                    session_id="ctx-1",
                )
            ],
        )
    )

    with pytest.raises(ValueError):
        runner.prepare_context(
            ExperimentContext(
                context_id="ctx-2",
                session_id="ctx-2",
                updates=[],
            )
        )


def test_runner_uses_batched_reasoner_path_when_available() -> None:
    class BatchTrackingReasoner(DeterministicValidityReader):
        def __init__(self) -> None:
            super().__init__()
            self.batch_calls = 0
            self.single_calls = 0

        def answer_with_trace(self, query, context):
            self.single_calls += 1
            return super().answer_with_trace(query, context)

        def answer_many_with_traces(self, queries, contexts):
            self.batch_calls += 1
            return [
                DeterministicValidityReader.answer_with_trace(self, query, context)
                for query, context in zip(queries, contexts)
            ]

    context = ExperimentContext(
        context_id="ctx-1",
        session_id="ctx-1",
        updates=[
            MemoryRecord(
                record_id="e1",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=0,
                session_id="ctx-1",
            )
        ],
    )
    cases = [
        ExperimentCase(
            case_id="q1",
            context_id="ctx-1",
            runtime_query=RuntimeQuery(
                query_id="q1",
                context_id="ctx-1",
                entity="user",
                attribute="home_city",
                question="Where do I live?",
                timestamp=1,
                session_id="ctx-1",
            ),
            eval_target=EvalTarget(query_id="q1", gold_answer="Boston"),
        ),
        ExperimentCase(
            case_id="q2",
            context_id="ctx-1",
            runtime_query=RuntimeQuery(
                query_id="q2",
                context_id="ctx-1",
                entity="user",
                attribute="home_city",
                question="What city am I in now?",
                timestamp=1,
                session_id="ctx-1",
            ),
            eval_target=EvalTarget(query_id="q2", gold_answer="Boston"),
        ),
    ]

    reasoner = BatchTrackingReasoner()
    runner = ContextCaseRunner(policy=AppendOnlyMemoryPolicy(), reasoner=reasoner)
    executed = runner.run_cases_for_context(context, cases)

    assert len(executed) == 2
    assert reasoner.batch_calls == 1
    assert reasoner.single_calls == 0
