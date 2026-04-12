"""Tests for failure analysis module."""
from memory_inference.consolidation.revision_types import MemoryStatus
from memory_inference.failure_analysis import (
    CONFLICT_LEAK,
    MISSING_ENTRY,
    NOISE_LEAK,
    REASONER_ERROR,
    STALE_RETRIEVAL,
    bucket_failures,
    classify_failure,
    export_failures_csv,
    failure_summary,
)
from memory_inference.types import InferenceExample, MemoryEntry, Query


def _entry(value="v", ts=1, confidence=1.0, status=MemoryStatus.ACTIVE):
    return MemoryEntry(
        entry_id="e1", entity="user", attribute="city",
        value=value, timestamp=ts, session_id="s1",
        confidence=confidence, status=status,
    )


def _example(correct, retrieved, prediction="wrong", query_id="s1-q-user_00"):
    q = Query(
        query_id=query_id, entity="user", attribute="city",
        question="q", answer="gold", timestamp=10, session_id="s1",
    )
    return InferenceExample(
        query=q, retrieved=retrieved, prediction=prediction,
        correct=correct, policy_name="test",
    )


class TestClassifyFailure:
    def test_missing_entry(self):
        ex = _example(False, [])
        assert classify_failure(ex) == MISSING_ENTRY

    def test_noise_leak(self):
        noisy = _entry("wrong", confidence=0.1)
        ex = _example(False, [noisy], prediction="wrong")
        assert classify_failure(ex) == NOISE_LEAK

    def test_conflict_leak(self):
        conflicted = _entry("v1", status=MemoryStatus.CONFLICTED)
        ex = _example(False, [conflicted])
        assert classify_failure(ex) == CONFLICT_LEAK

    def test_stale_retrieval(self):
        stale = _entry("old_value", ts=1)
        ex = _example(False, [stale])
        assert classify_failure(ex) == STALE_RETRIEVAL

    def test_reasoner_error(self):
        correct_entry = _entry("gold", ts=5)
        ex = _example(False, [correct_entry], prediction="wrong")
        assert classify_failure(ex) == REASONER_ERROR


class TestBucketFailures:
    def test_only_failures_included(self):
        correct = _example(True, [_entry("gold")], prediction="gold")
        wrong = _example(False, [])
        failures = bucket_failures([correct, wrong])
        assert len(failures) == 1
        assert failures[0].failure_mode == MISSING_ENTRY

    def test_scenario_family_extracted(self):
        ex = _example(False, [], query_id="s3-q-user_05-city")
        failures = bucket_failures([ex])
        assert failures[0].scenario_family == "S3"


class TestFailureSummary:
    def test_counts(self):
        failures = bucket_failures([
            _example(False, [], query_id="s1-a"),
            _example(False, [], query_id="s2-b"),
            _example(False, [_entry("old")], query_id="s3-c"),
        ])
        summary = failure_summary(failures)
        assert summary[MISSING_ENTRY] == 2
        assert summary[STALE_RETRIEVAL] == 1


class TestExportCSV:
    def test_csv_output(self):
        failures = bucket_failures([_example(False, [], query_id="s1-a")])
        csv_str = export_failures_csv(failures)
        assert "scenario_family" in csv_str
        assert "S1" in csv_str
