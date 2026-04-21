from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.domain.results import EvaluatedCase, ReaderTrace
from memory_inference.evaluation.targets import EvalTarget
from tests.factories import make_query, make_record


def test_runtime_query_keeps_gold_answer_out_of_runtime_contract() -> None:
    query = make_query(query_id="q1", entity="user", attribute="city")

    assert not hasattr(query, "answer")


def test_experiment_case_separates_runtime_query_from_eval_target() -> None:
    query = make_query(query_id="q1", context_id="ctx1", entity="user", attribute="city")
    case = ExperimentCase(
        case_id="case-1",
        context_id="ctx1",
        runtime_query=query,
        eval_target=EvalTarget(query_id="q1", gold_answer="Boston"),
    )

    assert case.runtime_query.question == "?"
    assert case.eval_target.gold_answer == "Boston"


def test_evaluated_case_uses_canonical_records_and_bundles() -> None:
    record = make_record(entry_id="m1", entity="user", attribute="city", value="Boston")
    query = make_query(query_id="q1", context_id="ctx1", entity="user", attribute="city")
    context = ExperimentContext(context_id="ctx1", session_id="s1", updates=[record])
    case = ExperimentCase(
        case_id="case-1",
        context_id=context.context_id,
        runtime_query=query,
        eval_target=EvalTarget(query_id="q1", gold_answer="Boston"),
    )
    evaluated = EvaluatedCase(
        case=case,
        retrieval_bundle=RetrievalBundle(records=[record]),
        reader_trace=ReaderTrace(answer="Boston"),
        prediction="Boston",
        correct=True,
        policy_name="append_only",
    )

    assert evaluated.retrieval_bundle.records[0].record_id == "m1"
    assert evaluated.correct is True
