from memory_inference.domain.enums import MemoryStatus
from memory_inference.memory.retrieval import DenseRanker
from memory_inference.memory.revision import Mem0RevisionEngine
from memory_inference.memory.stores import ArchiveStore, ConflictStore, CurrentStateStore
from tests.factories import make_record


class FakeDenseEncoder:
    def encode_query(self, text: str) -> tuple[float, ...]:
        return (1.0 if "state" in text.lower() else 0.0, float(len(text)))

    def encode_passage(self, text: str) -> tuple[float, ...]:
        return self.encode_query(text)

    def encode_passages(self, texts) -> list[tuple[float, ...]]:
        return [self.encode_query(text) for text in texts]

    def similarity(self, left, right) -> float:
        return sum(a * b for a, b in zip(left, right))


def test_mem0_revision_engine_updates_and_archives_superseded_state() -> None:
    engine = Mem0RevisionEngine()
    state_store = CurrentStateStore()
    archive_store = ArchiveStore()
    conflict_store = ConflictStore()
    ranker = DenseRanker(encoder=FakeDenseEncoder())

    old = engine.prepare_entry(
        make_record(
            entry_id="old",
            entity="user",
            attribute="employer",
            value="Google",
            timestamp=1,
            session_id="s",
            metadata={"memory_kind": "state"},
        )
    )
    new = engine.prepare_entry(
        make_record(
            entry_id="new",
            entity="user",
            attribute="employer",
            value="Meta",
            timestamp=2,
            session_id="s",
            metadata={"memory_kind": "state"},
        )
    )

    engine.apply(old, state_store=state_store, ranker=ranker, archive_store=archive_store, conflict_store=conflict_store)
    engine.apply(new, state_store=state_store, ranker=ranker, archive_store=archive_store, conflict_store=conflict_store)

    active = state_store.same_key("user", "employer")
    archived = archive_store.entries[("user", "employer")]

    assert [entry.value for entry in active] == ["Meta"]
    assert archived[0].value == "Google"
    assert archived[0].status == MemoryStatus.SUPERSEDED


def test_mem0_revision_engine_records_out_of_order_conflicts() -> None:
    engine = Mem0RevisionEngine()
    state_store = CurrentStateStore()
    archive_store = ArchiveStore()
    conflict_store = ConflictStore()
    ranker = DenseRanker(encoder=FakeDenseEncoder())

    newer = engine.prepare_entry(
        make_record(
            entry_id="newer",
            entity="user",
            attribute="home_city",
            value="Seattle",
            timestamp=5,
            session_id="s",
            metadata={"memory_kind": "state"},
        )
    )
    older = engine.prepare_entry(
        make_record(
            entry_id="older",
            entity="user",
            attribute="home_city",
            value="Boston",
            timestamp=4,
            session_id="s",
            metadata={"memory_kind": "state"},
        )
    )

    engine.apply(newer, state_store=state_store, ranker=ranker, archive_store=archive_store, conflict_store=conflict_store)
    engine.apply(older, state_store=state_store, ranker=ranker, archive_store=archive_store, conflict_store=conflict_store)

    assert {entry.value for entry in conflict_store.entries[("user", "home_city")]} == {"Seattle", "Boston"}
