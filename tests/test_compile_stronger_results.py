import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compile_stronger_results.py"
    spec = importlib.util.spec_from_file_location("compile_stronger_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compile_stronger_results_compares_all_non_baseline_policies(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        _row("mem0", "c1", correct=False, prompt_tokens=100, gold_mismatch=True),
        _row(
            "odv2_support_compact",
            "c1",
            correct=False,
            prompt_tokens=80,
            retrieval_mode="odv2_mem0_selective_compact",
            support_compacted=1,
            gold_mismatch=True,
        ),
        _row(
            "odv2_mem0_selective",
            "c1",
            correct=True,
            prompt_tokens=70,
            retrieval_mode="odv2_mem0_selective_guard",
            validity_removed=1,
        ),
    ]
    cases_path.write_text("\n".join(json.dumps(row) for row in rows))

    summaries, audit_rows = _load_module().compile_paths([cases_path])

    all_paired = {
        (row["target_policy"], row["scope"]): row
        for row in summaries
    }
    assert all_paired[("odv2_support_compact", "all paired cases")]["delta_accuracy"] == "0.000"
    assert all_paired[("odv2_mem0_selective", "all paired cases")]["delta_accuracy"] == "1.000"
    assert all_paired[("odv2_mem0_selective", "all paired cases")]["wins"] == 1
    assert all_paired[("odv2_mem0_selective", "all paired cases")]["verdict"] == "accuracy_plus_efficiency"
    assert all_paired[("odv2_mem0_selective", "Predeclared validity-sensitive union")]["n"] == 1
    assert all_paired[("odv2_mem0_selective", "Current-state same-key evidence retrieved")]["n"] == 1
    assert audit_rows
    assert audit_rows[0]["baseline_gold_mismatch_exposure"] is True
    assert audit_rows[0]["predeclared_validity_sensitive"] is True


def test_compile_stronger_results_writes_csv_and_audit(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row("mem0", "c1", correct=True, prompt_tokens=100),
                _row(
                    "odv2_mem0_selective",
                    "c1",
                    correct=True,
                    prompt_tokens=50,
                    retrieval_mode="odv2_mem0_selective_compact",
                    support_compacted=1,
                ),
            ]
        )
    )

    buffer = StringIO()
    with redirect_stdout(buffer):
        exit_code = _load_module().main(
            [
                "--audit-output",
                str(audit_path),
                str(cases_path),
            ]
        )

    assert exit_code == 0
    csv_output = buffer.getvalue()
    assert "target_policy,scope,n" in csv_output
    assert "delta_gold_mismatch_exposure" in csv_output
    assert "odv2_mem0_selective,ODV2 intervened,1" in csv_output
    assert audit_path.read_text().strip()


def _row(
    policy: str,
    case_id: str,
    *,
    correct: bool,
    prompt_tokens: int,
    gold_mismatch: bool = False,
    retrieval_mode: str = "mem0_active_dense",
    validity_removed: int = 0,
    support_compacted: int = 0,
) -> dict[str, object]:
    return {
        "benchmark": "longmemeval",
        "category": "multi-session",
        "policy_name": policy,
        "case_id": case_id,
        "query_mode": "CURRENT_STATE",
        "correct": correct,
        "stale_state_exposure": gold_mismatch,
        "prompt_tokens": prompt_tokens,
        "retrieval_mode": retrieval_mode,
        "validity_removed": validity_removed,
        "support_compacted": support_compacted,
        "entity": "user",
        "attribute": "home_city",
        "question": "Where does the user live now?",
        "gold_answer": "Boston",
        "prediction": "Boston" if correct else "Seattle",
        "retrieved_records": [
            _record("home_city", "Seattle"),
            _record("home_city", "Boston"),
        ],
    }


def _record(attribute: str, value: str) -> dict[str, str]:
    return {
        "entity": "user",
        "attribute": attribute,
        "value": value,
        "memory_kind": "state",
        "source_kind": "structured_fact",
    }
