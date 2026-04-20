from memory_inference.consolidation.mem0_variants import (
    Mem0AllFeaturesPolicy,
    Mem0ArchiveConflictPolicy,
    Mem0HistoryAwarePolicy,
    Mem0SupportLinksPolicy,
)
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.types import MemoryEntry, Query


class FakeDenseEncoder:
    def encode_query(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passage(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passages(self, texts) -> list[tuple[float, ...]]:
        return [self._encode(text) for text in texts]

    def similarity(self, left, right) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _encode(self, text: str) -> tuple[float, ...]:
        lower = text.lower()
        return (
            1.0 if "google" in lower else 0.0,
            1.0 if "meta" in lower else 0.0,
            1.0 if "employer" in lower or "work" in lower or "job" in lower else 0.0,
            1.0 if "boston" in lower else 0.0,
            1.0 if "seattle" in lower else 0.0,
        )


def test_mem0_support_links_policy_has_explicit_name() -> None:
    policy = Mem0SupportLinksPolicy(encoder=FakeDenseEncoder())
    assert policy.name == "mem0_support_links"


def test_mem0_archive_conflict_archives_superseded_state() -> None:
    policy = Mem0ArchiveConflictPolicy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            MemoryEntry(
                entry_id="old",
                entity="user",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            MemoryEntry(
                entry_id="new",
                entity="user",
                attribute="employer",
                value="Meta",
                timestamp=2,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
        ]
    )

    archived_values = [entry.value for entry in policy.archive[("user", "employer")]]
    active_values = [entry.value for entry in policy.active_store.values() if entry.attribute == "employer"]

    assert "Google" in archived_values
    assert "Meta" in active_values


def test_mem0_history_aware_prefers_older_semantic_match_for_history_query() -> None:
    policy = Mem0HistoryAwarePolicy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            MemoryEntry(
                entry_id="old",
                entity="Alice",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            MemoryEntry(
                entry_id="new",
                entity="Alice",
                attribute="employer",
                value="Meta",
                timestamp=2,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
        ]
    )

    query = Query(
        query_id="history-q",
        entity="Alice",
        attribute="employer",
        question="What was Alice's previous employer?",
        answer="Google",
        timestamp=3,
        session_id="s",
        query_mode=QueryMode.HISTORY,
    )
    result = policy.retrieve_for_query(query, top_k=1)

    assert result.debug["retrieval_mode"] == "mem0_history_dense"
    assert result.entries[0].value == "Google"


def test_mem0_all_features_surfaces_conflicts_for_conflict_aware_queries() -> None:
    policy = Mem0AllFeaturesPolicy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            MemoryEntry(
                entry_id="boston",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            MemoryEntry(
                entry_id="seattle",
                entity="user",
                attribute="home_city",
                value="Seattle",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
        ]
    )

    query = Query(
        query_id="conflict-q",
        entity="user",
        attribute="home_city",
        question="What city does the user live in?",
        answer="ABSTAIN",
        timestamp=2,
        session_id="s",
        query_mode=QueryMode.CONFLICT_AWARE,
        supports_abstention=True,
    )
    result = policy.retrieve_for_query(query, top_k=4)
    returned_values = {entry.value for entry in result.entries}

    assert result.debug["retrieval_mode"] == "mem0_state_augmented"
    assert {"Boston", "Seattle"}.issubset(returned_values)
