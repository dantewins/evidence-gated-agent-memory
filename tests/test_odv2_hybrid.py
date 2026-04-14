from memory_inference.consolidation.odv2_hybrid import ODV2DenseMemoryPolicy, ODV2StrongMemoryPolicy
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.llm.mock_consolidator import MockConsolidator
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
        )


def _strong_policy() -> ODV2StrongMemoryPolicy:
    return ODV2StrongMemoryPolicy(consolidator=MockConsolidator())


def _dense_policy() -> ODV2DenseMemoryPolicy:
    return ODV2DenseMemoryPolicy(consolidator=MockConsolidator(), encoder=FakeDenseEncoder())


def test_odv2_strong_structured_query_returns_state_and_support_evidence() -> None:
    p = _strong_policy()
    support = MemoryEntry(
        entry_id="turn-1",
        entity="Alice",
        attribute="dialogue",
        value="I got a new job at Google.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    fact = MemoryEntry(
        entry_id="fact-1",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        scope="session_1",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-1",
            "support_text": "I got a new job at Google.",
            "memory_kind": "state",
        },
    )
    p.ingest([support, fact])
    p.maybe_consolidate()

    query = Query(
        query_id="q-structured",
        entity="Alice",
        attribute="employer",
        question="Where does Alice work now?",
        answer="Google",
        timestamp=2,
        session_id="s",
        query_mode=QueryMode.CURRENT_STATE,
    )
    result = p.retrieve_for_query(query)

    assert result.debug["retrieval_mode"] == "hybrid_state_evidence"
    assert result.debug["backbone"] == "strong"
    assert any(entry.attribute == "employer" and entry.value == "Google" for entry in result.entries)
    assert any(entry.entry_id == "turn-1" for entry in result.entries)


def test_odv2_dense_structured_query_returns_state_and_support_evidence() -> None:
    p = _dense_policy()
    support = MemoryEntry(
        entry_id="turn-1",
        entity="Alice",
        attribute="dialogue",
        value="I got a new job at Google.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    fact = MemoryEntry(
        entry_id="fact-1",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        scope="session_1",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-1",
            "support_text": "I got a new job at Google.",
            "memory_kind": "state",
        },
    )
    p.ingest([support, fact])
    p.maybe_consolidate()

    query = Query(
        query_id="q-structured",
        entity="Alice",
        attribute="employer",
        question="Where does Alice work now?",
        answer="Google",
        timestamp=2,
        session_id="s",
        query_mode=QueryMode.CURRENT_STATE,
    )
    result = p.retrieve_for_query(query)

    assert result.debug["retrieval_mode"] == "hybrid_state_evidence"
    assert result.debug["backbone"] == "dense"
    assert any(entry.attribute == "employer" and entry.value == "Google" for entry in result.entries)
    assert any(entry.entry_id == "turn-1" for entry in result.entries)


def test_odv2_dense_history_query_surfaces_prior_and_current_values() -> None:
    p = _dense_policy()
    p.ingest(
        [
            MemoryEntry(
                entry_id="turn-google",
                entity="Alice",
                attribute="dialogue",
                value="I got a new job at Google.",
                timestamp=1,
                session_id="s",
                scope="session_1",
            ),
            MemoryEntry(
                entry_id="fact-google",
                entity="Alice",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                scope="session_1",
                metadata={
                    "source_kind": "structured_fact",
                    "source_entry_id": "turn-google",
                    "support_text": "I got a new job at Google.",
                    "memory_kind": "state",
                },
            ),
            MemoryEntry(
                entry_id="turn-meta",
                entity="Alice",
                attribute="dialogue",
                value="I switched to Meta.",
                timestamp=2,
                session_id="s",
                scope="session_2",
            ),
            MemoryEntry(
                entry_id="fact-meta",
                entity="Alice",
                attribute="employer",
                value="Meta",
                timestamp=2,
                session_id="s",
                scope="session_2",
                metadata={
                    "source_kind": "structured_fact",
                    "source_entry_id": "turn-meta",
                    "support_text": "I switched to Meta.",
                    "memory_kind": "state",
                },
            ),
        ]
    )
    p.maybe_consolidate()

    query = Query(
        query_id="q-history",
        entity="Alice",
        attribute="employer",
        question="Where did Alice work before Meta?",
        answer="Google",
        timestamp=3,
        session_id="s",
        query_mode=QueryMode.HISTORY,
    )
    result = p.retrieve_for_query(query)
    returned_values = {entry.value for entry in result.entries if entry.attribute == "employer"}

    assert "Google" in returned_values
    assert "Meta" in returned_values


def test_odv2_hybrid_policy_names_are_explicit() -> None:
    assert _strong_policy().name == "odv2_strong"
    assert _dense_policy().name == "odv2_dense"
