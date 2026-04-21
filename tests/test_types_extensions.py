from tests.factories import make_query, make_record


def test_memory_entry_has_default_importance() -> None:
    entry = make_record(
        entry_id="1", entity="u", attribute="a", value="v",
        timestamp=1, session_id="s",
    )
    assert entry.importance == 1.0
    assert entry.access_count == 0


def test_query_has_default_multi_attributes() -> None:
    query = make_query(
        query_id="q1", entity="u", attribute="a",
        question="?", answer="v", timestamp=2, session_id="s",
    )
    assert query.multi_attributes == ()


def test_query_multi_attributes_stored() -> None:
    query = make_query(
        query_id="q1", entity="u", attribute="a",
        question="?", answer="v", timestamp=2, session_id="s",
        multi_attributes=("b", "c"),
    )
    assert query.multi_attributes == ("b", "c")


def test_access_count_is_mutable() -> None:
    entry = make_record(
        entry_id="1", entity="u", attribute="a", value="v",
        timestamp=1, session_id="s",
    )
    entry.access_count += 1
    assert entry.access_count == 1


def test_multi_attributes_is_tuple_type() -> None:
    query = make_query(
        query_id="q1", entity="u", attribute="a",
        question="?", answer="v", timestamp=2, session_id="s",
        multi_attributes=("b", "c"),
    )
    assert isinstance(query.multi_attributes, tuple)
