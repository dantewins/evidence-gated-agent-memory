from memory_inference.domain.enums import UpdateType
from memory_inference.domain.memory import MemoryRecord
from memory_inference.llm.mock_consolidator import MockConsolidator
from tests.factories import make_record


def _entry(entry_id: str, value: str, timestamp: int, confidence: float = 1.0) -> MemoryRecord:
    return make_record(
        entry_id=entry_id, entity="u", attribute="a",
        value=value, timestamp=timestamp, session_id="s",
        confidence=confidence,
    )


def test_classify_supersession() -> None:
    c = MockConsolidator()
    existing = _entry("1", "old", timestamp=1)
    new = _entry("2", "new", timestamp=2)
    assert c.classify_update(new, existing) == UpdateType.SUPERSESSION


def test_classify_reinforcement() -> None:
    c = MockConsolidator()
    existing = _entry("1", "same", timestamp=1)
    new = _entry("2", "same", timestamp=2)
    assert c.classify_update(new, existing) == UpdateType.REINFORCEMENT


def test_classify_conflict_equal_timestamps() -> None:
    c = MockConsolidator()
    existing = _entry("1", "alpha", timestamp=5)
    new = _entry("2", "beta", timestamp=5)
    assert c.classify_update(new, existing) == UpdateType.CONFLICT


def test_classify_conflict_older_new_entry() -> None:
    c = MockConsolidator()
    existing = _entry("1", "new_val", timestamp=5)
    new = _entry("2", "old_val", timestamp=3)
    assert c.classify_update(new, existing) == UpdateType.CONFLICT


def test_merge_returns_highest_confidence() -> None:
    c = MockConsolidator()
    low = _entry("1", "low", timestamp=1, confidence=0.4)
    high = _entry("2", "high", timestamp=2, confidence=0.9)
    merged = c.merge_entries([low, high])
    assert "high" in merged.value and "low" in merged.value
    assert merged.confidence > 0.9  # boosted


def test_extract_facts_parses_key_value_pairs() -> None:
    c = MockConsolidator()
    facts = c.extract_facts(
        "home_city=Boston; preferred_language=Python",
        entity="user_01",
        session_id="s1",
        timestamp=5,
    )
    assert len(facts) == 2
    assert any(f.attribute == "home_city" and f.value == "Boston" for f in facts)
    assert any(f.attribute == "preferred_language" and f.value == "Python" for f in facts)
    assert all(f.entity == "user_01" and f.session_id == "s1" and f.timestamp == 5 for f in facts)


def test_consolidator_tracks_total_calls() -> None:
    c = MockConsolidator()
    assert c.total_calls == 0
    existing = _entry("1", "old", timestamp=1)
    new = _entry("2", "new", timestamp=2)
    c.classify_update(new, existing)
    assert c.total_calls == 1


def test_merge_entries_tracks_total_calls() -> None:
    c = MockConsolidator()
    c.merge_entries([_entry("1", "v", timestamp=1)])
    assert c.total_calls == 1


def test_extract_facts_tracks_total_calls() -> None:
    c = MockConsolidator()
    c.extract_facts("a=b", entity="u", session_id="s", timestamp=1)
    assert c.total_calls == 1


def test_extract_facts_returns_empty_for_no_key_value_pairs() -> None:
    c = MockConsolidator()
    facts = c.extract_facts("no equals signs here", entity="u", session_id="s", timestamp=1)
    assert facts == []
