"""Tests for classify_revision() on BaseConsolidator / MockConsolidator."""
from memory_inference.domain.enums import RevisionOp
from memory_inference.domain.memory import MemoryRecord
from memory_inference.llm.mock_consolidator import MockConsolidator
from tests.factories import make_record


def _e(entry_id: str, value: str, ts: int, confidence: float = 1.0,
       scope: str = "default") -> MemoryRecord:
    return make_record(
        entry_id=entry_id, entity="u", attribute="a",
        value=value, timestamp=ts, session_id="s",
        confidence=confidence, scope=scope,
    )


def test_add_when_no_existing_entry() -> None:
    c = MockConsolidator()
    op = c.classify_revision(_e("e2", "new", ts=2), existing=None)
    assert op == RevisionOp.ADD


def test_reinforce_when_same_value() -> None:
    c = MockConsolidator()
    existing = _e("e1", "v", ts=1)
    new = _e("e2", "v", ts=2)
    assert c.classify_revision(new, existing) == RevisionOp.REINFORCE


def test_revise_when_newer_timestamp_different_value() -> None:
    c = MockConsolidator()
    existing = _e("e1", "old", ts=1)
    new = _e("e2", "new", ts=3)
    assert c.classify_revision(new, existing) == RevisionOp.REVISE


def test_revert_when_value_matches_earlier_superseded_entry() -> None:
    """v1 -> v2 -> v1 pattern: returning to a prior value = REVERT."""
    c = MockConsolidator()
    # existing is v2, new is v1 (later timestamp)
    existing = _e("e2", "v2", ts=2)
    new = _e("e3", "v1", ts=3)
    # Pass prior_values to indicate v1 was seen before
    op = c.classify_revision(new, existing, prior_values={"v1"})
    assert op == RevisionOp.REVERT


def test_conflict_unresolved_on_equal_timestamp() -> None:
    c = MockConsolidator()
    existing = _e("e1", "alpha", ts=5)
    new = _e("e2", "beta", ts=5)
    assert c.classify_revision(new, existing) == RevisionOp.CONFLICT_UNRESOLVED


def test_split_scope_when_scopes_differ() -> None:
    c = MockConsolidator()
    existing = _e("e1", "North End", ts=1, scope="boston")
    new = _e("e2", "Wynwood", ts=2, scope="miami")
    assert c.classify_revision(new, existing) == RevisionOp.SPLIT_SCOPE


def test_low_confidence_when_below_threshold() -> None:
    c = MockConsolidator(low_confidence_threshold=0.4)
    existing = _e("e1", "old", ts=1)
    new = _e("e2", "noisy", ts=2, confidence=0.3)
    assert c.classify_revision(new, existing) == RevisionOp.LOW_CONFIDENCE


def test_classify_revision_increments_total_calls() -> None:
    c = MockConsolidator()
    before = c.total_calls
    c.classify_revision(_e("e2", "v", ts=2), _e("e1", "v", ts=1))
    assert c.total_calls == before + 1
