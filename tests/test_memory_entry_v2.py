"""Tests for validity-aware memory and query domain fields."""
from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from tests.factories import make_query, make_record


def _base_entry(**kwargs) -> MemoryRecord:
    defaults = dict(entry_id="e1", entity="u", attribute="a", value="v", timestamp=1, session_id="s")
    defaults.update(kwargs)
    return make_record(**defaults)


def _base_query(**kwargs) -> RuntimeQuery:
    defaults = dict(query_id="q1", entity="u", attribute="a", question="?", answer="v", timestamp=2, session_id="s")
    defaults.update(kwargs)
    return make_query(**defaults)


# MemoryRecord fields

def test_memory_entry_default_status_is_active() -> None:
    e = _base_entry()
    assert e.status == MemoryStatus.ACTIVE


def test_memory_entry_default_scope_is_default() -> None:
    e = _base_entry()
    assert e.scope == "default"


def test_memory_entry_default_supersedes_id_is_none() -> None:
    e = _base_entry()
    assert e.supersedes_id is None


def test_memory_entry_default_provenance_is_empty() -> None:
    e = _base_entry()
    assert e.provenance == ""


def test_memory_entry_status_can_be_set() -> None:
    e = _base_entry(status=MemoryStatus.SUPERSEDED)
    assert e.status == MemoryStatus.SUPERSEDED


def test_memory_entry_supersedes_id_can_be_set() -> None:
    e = _base_entry(supersedes_id="e0")
    assert e.supersedes_id == "e0"


def test_memory_entry_scope_can_be_set() -> None:
    e = _base_entry(scope="boston")
    assert e.scope == "boston"


def test_memory_entry_text_unchanged_by_new_fields() -> None:
    """New fields should not break the text() method."""
    e = _base_entry(status=MemoryStatus.ACTIVE, scope="x", supersedes_id="y", provenance="z")
    text = e.text()
    assert "entity=u" in text
    assert "attribute=a" in text


# RuntimeQuery fields

def test_query_default_query_mode_is_current_state() -> None:
    q = _base_query()
    assert q.query_mode == QueryMode.CURRENT_STATE


def test_query_default_supports_abstention_is_false() -> None:
    q = _base_query()
    assert q.supports_abstention is False


def test_query_query_mode_can_be_set() -> None:
    q = _base_query(query_mode=QueryMode.CONFLICT_AWARE)
    assert q.query_mode == QueryMode.CONFLICT_AWARE


def test_query_supports_abstention_can_be_set() -> None:
    q = _base_query(supports_abstention=True)
    assert q.supports_abstention is True


def test_query_multi_attributes_still_works() -> None:
    q = _base_query(multi_attributes=("b", "c"))
    assert q.multi_attributes == ("b", "c")
