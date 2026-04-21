"""Tests for OfflineDeltaConsolidationPolicyV2 state machine and retrieval modes."""
import dataclasses

from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies import ODV2Policy, offline_delta_v2_policy
from tests.factories import make_query, make_record


def _e(entry_id: str, value: str, ts: int, confidence: float = 1.0,
       scope: str = "default") -> MemoryRecord:
    return make_record(
        entry_id=entry_id, entity="u", attribute="a",
        value=value, timestamp=ts, session_id="s",
        confidence=confidence, scope=scope,
    )


def _policy() -> ODV2Policy:
    return offline_delta_v2_policy(consolidator=MockConsolidator())


# ------------------------------------------------------------------ #
# Core state transitions                                               #
# ------------------------------------------------------------------ #

def test_add_new_entry_becomes_active() -> None:
    p = _policy()
    p.ingest([_e("e1", "v1", ts=1)])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    assert len(result.entries) == 1
    assert result.entries[0].value == "v1"


def test_revise_moves_old_to_archive() -> None:
    p = _policy()
    p.ingest([_e("e1", "old", ts=1)])
    p.maybe_consolidate()
    p.ingest([_e("e2", "new", ts=2)])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    assert result.entries[0].value == "new"
    assert any(e.value == "old" for e in p.archive.get(("u", "a"), []))


def test_reinforce_keeps_current_state_active() -> None:
    p = _policy()
    p.ingest([_e("e1", "v", ts=1), _e("e2", "v", ts=2)])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    assert result.entries[0].status in (MemoryStatus.ACTIVE, MemoryStatus.REINFORCED)


def test_conflict_goes_to_conflict_table() -> None:
    p = _policy()
    p.ingest([_e("e1", "alpha", ts=5), _e("e2", "beta", ts=5)])
    p.maybe_consolidate()
    conflicts = p.conflict_table.get(("u", "a"), [])
    assert len(conflicts) == 2


def test_revert_marks_entry_as_reverted() -> None:
    p = _policy()
    p.ingest([_e("e1", "v1", ts=1), _e("e2", "v2", ts=2), _e("e3", "v1", ts=3)])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    assert result.entries[0].value == "v1"


def test_split_scope_preserves_both_scopes() -> None:
    p = _policy()
    p.ingest([
        _e("e1", "North End", ts=1, scope="boston"),
        _e("e2", "Wynwood", ts=2, scope="miami"),
    ])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    scopes = {e.scope for e in result.entries}
    assert "boston" in scopes
    assert "miami" in scopes


def test_low_confidence_does_not_pollute_current_state() -> None:
    p = offline_delta_v2_policy(
        consolidator=MockConsolidator(low_confidence_threshold=0.5)
    )
    p.ingest([_e("e1", "good", ts=1, confidence=1.0)])
    p.maybe_consolidate()
    p.ingest([_e("e2", "noisy", ts=2, confidence=0.3)])
    p.maybe_consolidate()
    result = p.retrieve("u", "a")
    assert result.entries[0].value == "good"


# ------------------------------------------------------------------ #
# Retrieval modes                                                      #
# ------------------------------------------------------------------ #

def test_current_state_retrieval_returns_active_only() -> None:
    p = _policy()
    p.ingest([_e("e1", "old", ts=1), _e("e2", "new", ts=2)])
    p.maybe_consolidate()
    q = make_query(query_id="q", entity="u", attribute="a", question="?", answer="new",
              timestamp=3, session_id="s", query_mode=QueryMode.CURRENT_STATE)
    result = p.retrieve_by_mode(q)
    assert all(e.status in (MemoryStatus.ACTIVE, MemoryStatus.REINFORCED) for e in result.entries)


def test_history_retrieval_returns_all_entries() -> None:
    p = _policy()
    p.ingest([_e("e1", "old", ts=1), _e("e2", "new", ts=2)])
    p.maybe_consolidate()
    q = make_query(query_id="q", entity="u", attribute="a", question="?", answer="new",
              timestamp=3, session_id="s", query_mode=QueryMode.HISTORY)
    result = p.retrieve_by_mode(q)
    assert len(result.entries) >= 2


def test_conflict_aware_retrieval_surfaces_conflict_table() -> None:
    p = _policy()
    p.ingest([_e("e1", "alpha", ts=5), _e("e2", "beta", ts=5)])
    p.maybe_consolidate()
    q = make_query(query_id="q", entity="u", attribute="a", question="?", answer="UNKNOWN",
              timestamp=6, session_id="s", query_mode=QueryMode.CONFLICT_AWARE,
              supports_abstention=True)
    result = p.retrieve_by_mode(q)
    assert result.debug.get("conflict_count", "0") != "0"


def test_state_with_provenance_returns_active_and_superseded() -> None:
    p = _policy()
    p.ingest([_e("e1", "old", ts=1), _e("e2", "new", ts=2)])
    p.maybe_consolidate()
    q = make_query(query_id="q", entity="u", attribute="a", question="?", answer="new",
              timestamp=3, session_id="s", query_mode=QueryMode.STATE_WITH_PROVENANCE)
    result = p.retrieve_by_mode(q)
    statuses = {e.status for e in result.entries}
    assert MemoryStatus.ACTIVE in statuses or MemoryStatus.REINFORCED in statuses
    assert MemoryStatus.SUPERSEDED in statuses or MemoryStatus.ARCHIVED in statuses
