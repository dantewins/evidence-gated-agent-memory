import json
import importlib

from memory_inference.cli import main as cli_main
from memory_inference.evaluation.metrics import ExperimentMetrics
from memory_inference.orchestration.experiment import DatasetExperimentResult


def _raw_longmemeval_payload() -> list[dict]:
    return [
        {
            "question_id": "q_001",
            "question_type": "knowledge-update",
            "question": "What city does the user live in now?",
            "answer": "Boston",
            "haystack_sessions": [{"role": "user", "content": "I moved to Boston."}],
            "haystack_session_ids": ["sess_1"],
            "haystack_dates": ["2024-01-20"],
        }
    ]


def test_cli_delegates_benchmark_execution_to_orchestration(tmp_path, monkeypatch) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps(_raw_longmemeval_payload()))
    calls: dict[str, object] = {}

    def fake_run_dataset_experiment(**kwargs):
        calls.update(kwargs)
        return DatasetExperimentResult(
            benchmark="longmemeval",
            metrics=[
                ExperimentMetrics(
                    policy_name="append_only",
                    total_queries=1,
                    correct_queries=1,
                    accuracy=1.0,
                    exact_match_accuracy=1.0,
                    span_match_accuracy=1.0,
                    abstention_accuracy=0.0,
                    proactive_interference_rate=0.0,
                    stale_state_exposure_rate=0.0,
                    retrieval_hit_rate=1.0,
                    avg_retrieved_items=1.0,
                    avg_retrieved_chars=10.0,
                    avg_retrieved_context_tokens=2.0,
                    avg_context_tokens=2.0,
                    avg_prompt_tokens=2.0,
                    avg_completion_tokens=1.0,
                    avg_snapshot_size=1.0,
                    maintenance_tokens=0,
                    maintenance_latency_ms=0.0,
                    amortized_end_to_end_tokens=3.0,
                    avg_query_latency_ms=1.0,
                    cache_hit_rate=0.0,
                )
            ],
            manifest=None,
        )

    cli_module = importlib.import_module("memory_inference.cli.main")
    monkeypatch.setattr(cli_module, "run_dataset_experiment", fake_run_dataset_experiment)

    cli_main(
        [
            "longmemeval",
            "--input",
            str(source),
            "--input-format",
            "raw",
            "--policy",
            "append_only",
        ]
    )

    assert calls["benchmark_name"] == "longmemeval"
    assert calls["manifest_output"] == ""
    assert calls["cases_output"] == ""
    assert len(calls["policy_factories"]) == 1
