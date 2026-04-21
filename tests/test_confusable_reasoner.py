from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.confusable import ConfusableReasoner
from tests.factories import make_query, make_record


def _entry(entry_id: str, entity: str, attribute: str, value: str, ts: int) -> MemoryRecord:
    return make_record(
        entry_id=entry_id, entity=entity, attribute=attribute,
        value=value, timestamp=ts, session_id="s",
    )


def _query(entity: str, attribute: str, answer: str) -> RuntimeQuery:
    return make_query(
        query_id="q", entity=entity, attribute=attribute,
        question="?", answer=answer, timestamp=99, session_id="s",
    )


def test_picks_majority_value() -> None:
    r = ConfusableReasoner()
    context = [
        _entry("1", "u", "a", "Boston", ts=1),
        _entry("2", "u", "a", "Boston", ts=2),
        _entry("3", "u", "a", "NYC", ts=3),
    ]
    assert r.answer(_query("u", "a", "Boston"), context) == "Boston"


def test_tiebreaks_by_recency() -> None:
    r = ConfusableReasoner()
    context = [
        _entry("1", "u", "a", "Boston", ts=1),
        _entry("2", "u", "a", "NYC", ts=2),
    ]
    assert r.answer(_query("u", "a", "NYC"), context) == "NYC"


def test_returns_unknown_for_empty_context() -> None:
    r = ConfusableReasoner()
    assert r.answer(_query("u", "a", "x"), []) == "UNKNOWN"


def test_confused_by_distractors() -> None:
    """3 wrong-entity entries outvote 1 correct entry — confirms confusability."""
    r = ConfusableReasoner()
    correct = _entry("c", "user_00", "home_city", "Boston", ts=3)
    distractors = [
        _entry(f"d{i}", f"user_{i:02d}", "home_city", "NYC", ts=1)
        for i in range(1, 4)
    ]
    result = r.answer(_query("user_00", "home_city", "Boston"), [correct] + distractors)
    assert result == "NYC"  # majority (distractors win) — this is the desired wrong answer


def test_does_not_filter_by_entity_or_attribute() -> None:
    """Reasoner votes on ALL entries in context regardless of entity/attribute."""
    r = ConfusableReasoner()
    context = [
        _entry("1", "user_00", "home_city", "Paris", ts=1),
        _entry("2", "user_00", "preferred_language", "French", ts=2),
        _entry("3", "user_00", "preferred_language", "French", ts=3),
    ]
    # 'French' appears twice vs 'Paris' once — majority wins
    result = r.answer(_query("user_00", "home_city", "Paris"), context)
    assert result == "French"
