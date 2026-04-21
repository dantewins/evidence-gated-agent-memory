"""Tests for StateOracle over canonical memory records."""

from memory_inference.domain.enums import MemoryStatus
from memory_inference.evaluation.state_oracle import StateOracle
from memory_inference.domain.memory import MemoryRecord
from tests.factories import make_record


def _e(entry_id: str, entity: str, attribute: str, value: str, timestamp: int,
       status: MemoryStatus = MemoryStatus.ACTIVE,
       scope: str = "default",
       supersedes_id: str | None = None) -> MemoryRecord:
    return make_record(
        entry_id=entry_id, entity=entity, attribute=attribute,
        value=value, timestamp=timestamp, session_id="s",
        status=status, scope=scope, supersedes_id=supersedes_id,
    )


def test_active_value_returns_active_entry() -> None:
    e = _e("e1", "u", "a", "v1", 1)
    oracle = StateOracle([e])
    result = oracle.active_value("u", "a")
    assert result is not None
    assert result.value == "v1"


def test_active_value_returns_none_when_only_superseded() -> None:
    e = _e("e1", "u", "a", "v1", 1, status=MemoryStatus.SUPERSEDED)
    oracle = StateOracle([e])
    assert oracle.active_value("u", "a") is None


def test_active_value_prefers_most_recent_active_entry() -> None:
    e1 = _e("e1", "u", "a", "old", 1)
    e2 = _e("e2", "u", "a", "new", 2)
    oracle = StateOracle([e1, e2])
    assert oracle.active_value("u", "a").value == "new"


def test_superseded_chain_returns_superseded_entries() -> None:
    active = _e("e2", "u", "a", "new", 2, supersedes_id="e1")
    old = _e("e1", "u", "a", "old", 1, status=MemoryStatus.SUPERSEDED)
    oracle = StateOracle([active, old])
    chain = oracle.superseded_chain("u", "a")
    assert any(e.entry_id == "e1" for e in chain)


def test_unresolved_conflicts_returns_conflicted_entries() -> None:
    c1 = _e("c1", "u", "a", "alpha", 5, status=MemoryStatus.CONFLICTED)
    c2 = _e("c2", "u", "a", "beta", 5, status=MemoryStatus.CONFLICTED)
    oracle = StateOracle([c1, c2])
    conflicts = oracle.unresolved_conflicts("u", "a")
    assert len(conflicts) == 2


def test_scope_splits_groups_by_scope() -> None:
    e_boston = _e("e1", "u", "fav_spot", "North End", 1, scope="boston")
    e_miami = _e("e2", "u", "fav_spot", "Wynwood", 2, scope="miami")
    oracle = StateOracle([e_boston, e_miami])
    splits = oracle.scope_splits("u", "fav_spot")
    assert len(splits) == 2
    assert "boston" in splits
    assert "miami" in splits


def test_scope_splits_only_includes_active() -> None:
    active = _e("e1", "u", "fav_spot", "North End", 1, scope="boston")
    archived = _e("e2", "u", "fav_spot", "Old", 1, scope="boston", status=MemoryStatus.ARCHIVED)
    oracle = StateOracle([active, archived])
    splits = oracle.scope_splits("u", "fav_spot")
    assert len(splits["boston"]) == 1


def test_current_state_match_detects_correct_active_value() -> None:
    e = _e("e1", "u", "a", "v_gold", 1)
    oracle = StateOracle([e])
    assert oracle.current_state_match("u", "a", gold_value="v_gold") is True


def test_current_state_match_returns_false_for_wrong_value() -> None:
    e = _e("e1", "u", "a", "v_actual", 1)
    oracle = StateOracle([e])
    assert oracle.current_state_match("u", "a", gold_value="v_gold") is False
