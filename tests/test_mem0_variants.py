from memory_inference.domain.enums import QueryMode
from memory_inference.memory.policies import (
    mem0_all_features_policy,
    mem0_archive_conflict_policy,
    mem0_history_aware_policy,
)
from tests.factories import make_query, make_record


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


def test_mem0_archive_conflict_archives_superseded_state() -> None:
    policy = mem0_archive_conflict_policy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            make_record(
                entry_id="old",
                entity="user",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            make_record(
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

    current_state_query = make_query(
        query_id="current-q",
        entity="user",
        attribute="employer",
        question="Where does the user work now?",
        answer="Meta",
        timestamp=3,
        session_id="s",
        query_mode=QueryMode.CURRENT_STATE,
    )
    result = policy.retrieve_for_query(current_state_query, top_k=2)

    assert result.debug["retrieval_mode"] == "mem0_state_augmented"
    assert {entry.value for entry in result.entries} == {"Google", "Meta"}
    assert len({entry.entry_id for entry in result.entries}) == 2


def test_mem0_history_aware_prefers_older_semantic_match_for_history_query() -> None:
    policy = mem0_history_aware_policy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            make_record(
                entry_id="old",
                entity="Alice",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            make_record(
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

    query = make_query(
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
    policy = mem0_all_features_policy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            make_record(
                entry_id="boston",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            make_record(
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

    query = make_query(
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


def test_mem0_archive_conflict_records_out_of_order_conflicts() -> None:
    policy = mem0_archive_conflict_policy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            make_record(
                entry_id="newer",
                entity="user",
                attribute="home_city",
                value="Seattle",
                timestamp=5,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            make_record(
                entry_id="older",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=4,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
        ]
    )

    conflicts = policy.conflict_table[("user", "home_city")]
    conflict_values = {entry.value for entry in conflicts}

    assert {"Seattle", "Boston"} == conflict_values
