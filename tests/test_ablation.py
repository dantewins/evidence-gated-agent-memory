"""Tests for ablation consolidators."""
from memory_inference.consolidation.ablation import (
    NoArchiveConsolidator,
    NoConflictConsolidator,
    NoRevertConsolidator,
    NoScopeConsolidator,
)
from memory_inference.consolidation.revision_types import RevisionOp
from memory_inference.types import MemoryEntry


def _entry(value="v", ts=1, confidence=1.0, scope="default"):
    return MemoryEntry(
        entry_id="e1", entity="user", attribute="city",
        value=value, timestamp=ts, session_id="s1",
        confidence=confidence, scope=scope,
    )


class TestNoRevertConsolidator:
    def test_revert_becomes_revise(self):
        c = NoRevertConsolidator()
        existing = _entry("old", ts=1)
        new = _entry("old", ts=3)  # same value as prior, would be REVERT
        op = c.classify_revision(new, existing, prior_values={"old"})
        # Without revert detection, new value != existing value AND value in prior -> still REVISE
        # Actually new.value == existing.value here, so it's REINFORCE
        # Let me use a 3-step scenario: old -> new_val -> old (revert)
        existing2 = _entry("new_val", ts=2)
        new2 = _entry("old", ts=3)
        op2 = c.classify_revision(new2, existing2, prior_values={"old"})
        assert op2 == RevisionOp.REVISE  # not REVERT

    def test_non_revert_ops_unchanged(self):
        c = NoRevertConsolidator()
        assert c.classify_revision(_entry(), None) == RevisionOp.ADD
        existing = _entry("a", ts=1)
        same = _entry("a", ts=2)
        assert c.classify_revision(same, existing) == RevisionOp.REINFORCE


class TestNoConflictConsolidator:
    def test_conflict_becomes_revise(self):
        c = NoConflictConsolidator()
        existing = _entry("a", ts=1)
        conflict = _entry("b", ts=1)  # same timestamp, different value
        op = c.classify_revision(conflict, existing)
        assert op == RevisionOp.REVISE  # not CONFLICT_UNRESOLVED

    def test_non_conflict_ops_unchanged(self):
        c = NoConflictConsolidator()
        assert c.classify_revision(_entry(), None) == RevisionOp.ADD


class TestNoScopeConsolidator:
    def test_scope_split_becomes_revise(self):
        c = NoScopeConsolidator()
        existing = _entry("a", scope="boston")
        new = _entry("b", scope="miami")
        op = c.classify_revision(new, existing)
        assert op == RevisionOp.REVISE  # not SPLIT_SCOPE

    def test_non_scope_ops_unchanged(self):
        c = NoScopeConsolidator()
        assert c.classify_revision(_entry(), None) == RevisionOp.ADD


class TestNoArchiveConsolidator:
    def test_low_confidence_becomes_add(self):
        c = NoArchiveConsolidator()
        low = _entry(confidence=0.1)
        op = c.classify_revision(low, None)
        assert op == RevisionOp.ADD  # not LOW_CONFIDENCE

    def test_normal_confidence_unchanged(self):
        c = NoArchiveConsolidator()
        normal = _entry(confidence=0.9)
        assert c.classify_revision(normal, None) == RevisionOp.ADD
