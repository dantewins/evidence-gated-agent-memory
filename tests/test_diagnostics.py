import json

from memory_inference.domain.benchmark import ExperimentCase
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.domain.results import EvaluatedCase, ReaderTrace
from memory_inference.evaluation.diagnostics import (
    evaluated_case_to_diagnostic_row,
    write_diagnostic_jsonl,
)
from memory_inference.evaluation.targets import EvalTarget
from tests.factories import make_query, make_record


def test_diagnostic_row_captures_case_level_tradeoff_signals(tmp_path) -> None:
    query = make_query(
        query_id="q",
        entity="user",
        attribute="home_city",
        question="Where does the user live now?",
        answer="Boston",
        timestamp=2,
        session_id="s",
    )
    evaluated = EvaluatedCase(
        case=ExperimentCase(
            case_id="case-1",
            context_id=query.context_id,
            runtime_query=query,
            eval_target=EvalTarget(
                query_id=query.query_id,
                gold_answer="Boston",
                benchmark_name="test",
                benchmark_category="knowledge-update",
            ),
        ),
        retrieval_bundle=RetrievalBundle(
            records=[
                make_record(
                    entry_id="old",
                    entity="user",
                    attribute="home_city",
                    value="Seattle",
                    timestamp=1,
                    session_id="s",
                    metadata={"memory_kind": "state"},
                ),
                make_record(
                    entry_id="new",
                    entity="user",
                    attribute="home_city",
                    value="Boston",
                    timestamp=2,
                    session_id="s",
                    metadata={"memory_kind": "state"},
                ),
            ],
            debug={
                "retrieval_mode": "diagnostic",
                "validity_removed": "1",
                "support_compacted": "4",
                "temporal_pruned": "2",
                "conflict_values": "3",
                "decision_source": "ledger",
            },
        ),
        reader_trace=ReaderTrace(answer="Boston", prompt_tokens=32, latency_ms=4.0),
        prediction="Boston",
        correct=True,
        policy_name="policy",
    )

    row = evaluated_case_to_diagnostic_row(evaluated, benchmark="locomo")

    assert row["benchmark"] == "locomo"
    assert row["policy_name"] == "policy"
    assert row["retrieval_hit"] is True
    assert row["stale_state_exposure"] is True
    assert row["prompt_tokens"] == 32
    assert row["retrieval_mode"] == "diagnostic"
    assert row["validity_removed"] == 1
    assert row["support_compacted"] == 4
    assert row["temporal_pruned"] == 2
    assert row["conflict_values"] == 3
    assert row["decision_source"] == "ledger"

    output = tmp_path / "cases.jsonl"
    write_diagnostic_jsonl(output, [row])

    loaded = json.loads(output.read_text().strip())
    assert loaded["case_id"] == "case-1"
    assert loaded["retrieved_records"][0]["value"] == "Seattle"
