from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.types import MemoryEntry, Query


def test_memory_entry_defaults_and_helpers_reflect_current_schema() -> None:
    entry = MemoryEntry(
        entry_id="e1",
        entity="user",
        attribute="home_city",
        value="Boston",
        timestamp=1,
        session_id="s1",
    )

    assert entry.key == ("user", "home_city")
    assert entry.importance == 1.0
    assert entry.access_count == 0
    assert entry.status == MemoryStatus.ACTIVE
    assert entry.scope == "default"
    assert entry.supersedes_id is None
    assert entry.provenance == ""
    assert "entity=user" in entry.text()
    assert "attribute=home_city" in entry.text()


def test_memory_entry_allows_nondefault_validity_fields() -> None:
    entry = MemoryEntry(
        entry_id="e2",
        entity="user",
        attribute="home_city",
        value="Seattle",
        timestamp=2,
        session_id="s1",
        status=MemoryStatus.SUPERSEDED,
        scope="travel",
        supersedes_id="e1",
        provenance="unit-test",
    )

    assert entry.status == MemoryStatus.SUPERSEDED
    assert entry.scope == "travel"
    assert entry.supersedes_id == "e1"
    assert entry.provenance == "unit-test"


def test_query_defaults_and_key_reflect_current_schema() -> None:
    query = Query(
        query_id="q1",
        entity="user",
        attribute="home_city",
        question="Where do I live now?",
        answer="Boston",
        timestamp=3,
        session_id="s1",
    )

    assert query.key == ("user", "home_city")
    assert query.multi_attributes == ()
    assert query.query_mode == QueryMode.CURRENT_STATE
    assert query.supports_abstention is False


def test_query_allows_multihop_and_conflict_settings() -> None:
    query = Query(
        query_id="q2",
        entity="user",
        attribute="employer",
        question="Where do I work and live?",
        answer="OpenAI+San Francisco",
        timestamp=4,
        session_id="s1",
        multi_attributes=("home_city",),
        query_mode=QueryMode.CONFLICT_AWARE,
        supports_abstention=True,
    )

    assert query.multi_attributes == ("home_city",)
    assert query.query_mode == QueryMode.CONFLICT_AWARE
    assert query.supports_abstention is True
