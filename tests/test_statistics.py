"""Tests for statistical infrastructure."""
from memory_inference.statistics import (
    ConfidenceInterval,
    bootstrap_ci,
    paired_bootstrap_test,
    scenario_family_breakdown,
)
from memory_inference.types import InferenceExample, MemoryEntry, Query


def _example(query_id: str, correct: bool) -> InferenceExample:
    q = Query(
        query_id=query_id, entity="u", attribute="a",
        question="q", answer="a", timestamp=1, session_id="s",
    )
    return InferenceExample(
        query=q, retrieved=[], prediction="a" if correct else "wrong",
        correct=correct, policy_name="test",
    )


class TestBootstrapCI:
    def test_all_ones(self):
        ci = bootstrap_ci([1.0] * 100)
        assert ci.mean == 1.0
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_all_zeros(self):
        ci = bootstrap_ci([0.0] * 100)
        assert ci.mean == 0.0

    def test_mixed(self):
        vals = [1.0] * 70 + [0.0] * 30
        ci = bootstrap_ci(vals, seed=42)
        assert 0.60 < ci.lower < ci.mean < ci.upper < 0.85
        assert abs(ci.mean - 0.70) < 0.01

    def test_empty(self):
        ci = bootstrap_ci([])
        assert ci.mean == 0.0

    def test_bool_input(self):
        ci = bootstrap_ci([True, True, False, True])
        assert 0.5 <= ci.mean <= 1.0


class TestPairedBootstrapTest:
    def test_identical_sequences(self):
        a = [True] * 50 + [False] * 50
        p = paired_bootstrap_test(a, a, seed=42)
        assert p == 1.0  # no difference

    def test_clearly_different(self):
        # 95 correct vs 5 correct on the same 100 queries: huge paired gap
        a = [True] * 95 + [False] * 5
        b = [True] * 5 + [False] * 95
        p = paired_bootstrap_test(a, b, seed=42)
        assert p < 0.05

    def test_empty(self):
        p = paired_bootstrap_test([], [])
        assert p == 1.0


class TestScenarioFamilyBreakdown:
    def test_groups_by_prefix(self):
        examples = [
            _example("s1-q-user_00-city", True),
            _example("s1-q-user_01-city", False),
            _example("s2-q-user_00-city", True),
        ]
        bd = scenario_family_breakdown(examples)
        assert "S1" in bd
        assert "S2" in bd
        assert len(bd["S1"]) == 2
        assert bd["S1"] == [True, False]
        assert bd["S2"] == [True]
