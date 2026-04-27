import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compile_boss_results.py"
    spec = importlib.util.spec_from_file_location("compile_boss_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compile_boss_results_reports_usable_efficiency_claim(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        _row("mem0", "c1", correct=True, prompt_tokens=100),
        _row(
            "odv2_mem0_selective",
            "c1",
            correct=True,
            prompt_tokens=80,
            retrieval_mode="odv2_mem0_selective_compact",
            support_compacted=1,
        ),
        _row(
            "mem0",
            "c2",
            correct=False,
            stale=True,
            prompt_tokens=200,
            records=[
                _record("home_city", "Seattle"),
                _record("home_city", "Boston"),
            ],
        ),
        _row("odv2_mem0_selective", "c2", correct=False, stale=True, prompt_tokens=180),
    ]
    cases_path.write_text("\n".join(json.dumps(row) for row in rows))

    report = _load_module().render_report([cases_path])

    assert "Status: usable efficiency result against Mem0." in report
    assert "| longmemeval:knowledge-update | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -20.00 | 0 | 0 | usable |" in report
    assert "| ODV2 intervened | 1 | 0.000 | 0.000 | -20.00 | 0 | 0 | usable |" in report
    assert "| Mem0 same-key conflict | 1 | 0.000 | 0.000 | -20.00 | 0 | 0 | usable |" in report


def test_compile_boss_results_exports_csv(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        _row("mem0", "c1", correct=True, prompt_tokens=100),
        _row(
            "odv2_mem0_selective",
            "c1",
            correct=True,
            prompt_tokens=80,
            retrieval_mode="odv2_mem0_selective_compact",
            support_compacted=1,
        ),
    ]
    cases_path.write_text("\n".join(json.dumps(row) for row in rows))

    buffer = StringIO()
    with redirect_stdout(buffer):
        _load_module().write_csv_report([cases_path])
    csv_output = buffer.getvalue()

    assert "result,slice,n,mem0_accuracy,odv2_accuracy,delta_accuracy" in csv_output
    assert "longmemeval:knowledge-update,all paired cases,1,1.000,1.000,0.000" in csv_output
    assert "longmemeval:knowledge-update,ODV2 intervened,1,1.000,1.000,0.000" in csv_output


def _row(
    policy: str,
    case_id: str,
    *,
    correct: bool,
    prompt_tokens: int,
    stale: bool = False,
    retrieval_mode: str = "mem0_active_dense",
    support_compacted: int = 0,
    records=None,
) -> dict[str, object]:
    return {
        "benchmark": "longmemeval",
        "category": "knowledge-update",
        "policy_name": policy,
        "case_id": case_id,
        "correct": correct,
        "exact_match": correct,
        "stale_state_exposure": stale,
        "retrieval_hit": correct,
        "prompt_tokens": prompt_tokens,
        "retrieval_mode": retrieval_mode,
        "support_compacted": support_compacted,
        "retrieved_records": records or [_record("home_city", "Boston")],
        "entity": "user",
        "attribute": "home_city",
    }


def _record(attribute: str, value: str) -> dict[str, str]:
    return {
        "entity": "user",
        "attribute": attribute,
        "value": value,
        "memory_kind": "state",
        "source_kind": "structured_fact",
    }
