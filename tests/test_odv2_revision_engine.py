from memory_inference.domain.enums import MemoryStatus
from memory_inference.domain.memory import MemoryRecord
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.revision import ODV2RevisionEngine
from memory_inference.memory.stores import ArchiveStore, ConflictStore, ScopedCurrentStateStore
from tests.factories import make_record


def _e(entry_id: str, value: str, ts: int, confidence: float = 1.0, scope: str = "default") -> MemoryRecord:
    return make_record(
        entry_id=entry_id,
        entity="u",
        attribute="a",
        value=value,
        timestamp=ts,
        session_id="s",
        confidence=confidence,
        scope=scope,
    )


def test_odv2_revision_engine_revise_moves_old_value_to_archive() -> None:
    engine = ODV2RevisionEngine(consolidator=MockConsolidator())
    state_store = ScopedCurrentStateStore()
    archive_store = ArchiveStore()
    conflict_store = ConflictStore()
    prior_values = {("u", "a"): {"old", "new"}}

    state_store.put(_e("old", "old", 1))
    engine.process_entry(
        _e("new", "new", 2),
        state_store=state_store,
        archive_store=archive_store,
        conflict_store=conflict_store,
        prior_values=prior_values,  # type: ignore[arg-type]
    )

    assert [entry.value for entry in state_store.by_query(entity="u", attribute="a", entity_matches=lambda l, r: l == r)] == ["new"]
    assert archive_store.entries[("u", "a")][0].status == MemoryStatus.SUPERSEDED
    assert archive_store.entries[("u", "a")][0].value == "old"


def test_odv2_revision_engine_conflict_moves_entries_out_of_current_state() -> None:
    engine = ODV2RevisionEngine(consolidator=MockConsolidator())
    state_store = ScopedCurrentStateStore()
    archive_store = ArchiveStore()
    conflict_store = ConflictStore()
    prior_values = {("u", "a"): {"alpha", "beta"}}

    state_store.put(_e("old", "alpha", 5))
    engine.process_entry(
        _e("new", "beta", 5),
        state_store=state_store,
        archive_store=archive_store,
        conflict_store=conflict_store,
        prior_values=prior_values,  # type: ignore[arg-type]
    )

    assert not state_store.by_query(entity="u", attribute="a", entity_matches=lambda l, r: l == r)
    assert {entry.value for entry in conflict_store.entries[("u", "a")]} == {"alpha", "beta"}


def test_odv2_revision_engine_split_scope_keeps_parallel_active_entries() -> None:
    engine = ODV2RevisionEngine(consolidator=MockConsolidator())
    state_store = ScopedCurrentStateStore()
    archive_store = ArchiveStore()
    conflict_store = ConflictStore()
    prior_values = {("u", "a"): {"Boston", "Miami"}}

    state_store.put(_e("boston", "Boston", 1, scope="boston"))
    engine.process_entry(
        _e("miami", "Miami", 2, scope="miami"),
        state_store=state_store,
        archive_store=archive_store,
        conflict_store=conflict_store,
        prior_values=prior_values,  # type: ignore[arg-type]
    )

    scopes = {entry.scope for entry in state_store.by_query(entity="u", attribute="a", entity_matches=lambda l, r: l == r)}
    assert scopes == {"boston", "miami"}
