"""Tests for reporting module."""
from memory_inference.metrics import ExperimentMetrics
from memory_inference.reporting import (
    latex_ablation_table,
    latex_main_table,
    markdown_summary,
    pgfplots_cost_accuracy,
)


def _metrics(name="test", accuracy=0.7, state_em=0.8, tokens=100.0):
    return ExperimentMetrics(
        policy_name=name,
        total_queries=100,
        correct_queries=int(accuracy * 100),
        accuracy=accuracy,
        abstention_accuracy=1.0,
        proactive_interference_rate=0.0,
        avg_retrieved_items=1.0,
        avg_retrieved_chars=100.0,
        avg_context_tokens=5.0,
        avg_completion_tokens=0.0,
        avg_snapshot_size=5.0,
        maintenance_tokens=0,
        maintenance_latency_ms=0.0,
        amortized_end_to_end_tokens=tokens,
        avg_query_latency_ms=0.0,
        cache_hit_rate=0.0,
        current_state_exact_match=state_em,
        supersession_precision=0.75,
        supersession_recall=1.0,
        conflict_detection_f1=1.0,
        scope_split_accuracy=1.0,
    )


class TestLatexMainTable:
    def test_contains_booktabs(self):
        table = latex_main_table([_metrics()])
        assert r"\toprule" in table
        assert r"\bottomrule" in table

    def test_bold_best_accuracy(self):
        m1 = _metrics("a", accuracy=0.7)
        m2 = _metrics("b", accuracy=0.9)
        table = latex_main_table([m1, m2])
        assert r"\textbf{0.900}" in table

    def test_with_confidence_intervals(self):
        from memory_inference.statistics import ConfidenceInterval
        m = _metrics("test", accuracy=0.75)
        cis = {"test": ConfidenceInterval(mean=0.75, lower=0.70, upper=0.80, alpha=0.05)}
        table = latex_main_table([m], cis=cis)
        assert "0.700" in table
        assert "0.800" in table


class TestLatexAblationTable:
    def test_structure(self):
        table = latex_ablation_table([_metrics("offline_delta_v2")])
        assert "ablation" in table.lower()
        assert r"\toprule" in table


class TestPgfplotsCostAccuracy:
    def test_data_format(self):
        data = pgfplots_cost_accuracy([_metrics("append_only", tokens=150.0)])
        lines = data.strip().split("\n")
        assert lines[0] == "policy tokens accuracy"
        assert "append-only" in lines[1]


class TestMarkdownSummary:
    def test_table_format(self):
        md = markdown_summary([_metrics("test")])
        assert "| test |" in md
        assert "Accuracy" in md
