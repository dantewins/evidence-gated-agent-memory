from memory_inference.domain.benchmark import ExperimentCase
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.domain.results import EvaluatedCase, ReaderTrace
from memory_inference.evaluation.targets import EvalTarget
from memory_inference.evaluation.metrics import compute_metrics
from tests.factories import make_query, make_record


def test_metrics_report_prompt_tokens_and_retrieved_memory_tokens() -> None:
    query = make_query(
        query_id="q",
        entity="user",
        attribute="home_city",
        question="Where does the user live now?",
        answer="Boston",
        timestamp=1,
        session_id="s",
    )
    retrieved = [
        make_record(
            entry_id="m1",
            entity="user",
            attribute="home_city",
            value="Boston",
            timestamp=0,
            session_id="s",
        )
    ]
    eval_target = EvalTarget(query_id=query.query_id, gold_answer="Boston")
    evaluated = EvaluatedCase(
        case=ExperimentCase(
            case_id=query.query_id,
            context_id=query.context_id,
            runtime_query=query,
            eval_target=eval_target,
        ),
        retrieval_bundle=RetrievalBundle(records=retrieved),
        reader_trace=ReaderTrace(
            answer="Boston",
            prompt_tokens=99,
            completion_tokens=3,
        ),
        prediction="Boston",
        correct=True,
        policy_name="mem0",
    )

    metrics = compute_metrics("mem0", [evaluated])

    assert metrics.avg_prompt_tokens == 99.0
    assert metrics.avg_retrieved_context_tokens == 5.0
    assert metrics.avg_context_tokens == 99.0
    assert metrics.amortized_end_to_end_tokens == 102.0
