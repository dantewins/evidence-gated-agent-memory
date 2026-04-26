from memory_inference.domain.enums import QueryMode
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies import (
    ODV2Policy,
    odv2_dense_compact_policy,
    odv2_dense_policy,
    odv2_strong_policy,
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
        )


def _strong_policy() -> ODV2Policy:
    return odv2_strong_policy(consolidator=MockConsolidator())


def _dense_policy() -> ODV2Policy:
    return odv2_dense_policy(consolidator=MockConsolidator(), encoder=FakeDenseEncoder())


def _dense_compact_policy() -> ODV2Policy:
    return odv2_dense_compact_policy(consolidator=MockConsolidator(), encoder=FakeDenseEncoder())


def test_odv2_strong_structured_query_returns_state_and_support_evidence() -> None:
    p = _strong_policy()
    support = make_record(
        entry_id="turn-1",
        entity="Alice",
        attribute="dialogue",
        value="I got a new job at Google.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    fact = make_record(
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

    query = make_query(
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
    support = make_record(
        entry_id="turn-1",
        entity="Alice",
        attribute="dialogue",
        value="I got a new job at Google.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    fact = make_record(
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

    query = make_query(
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
            make_record(
                entry_id="turn-google",
                entity="Alice",
                attribute="dialogue",
                value="I got a new job at Google.",
                timestamp=1,
                session_id="s",
                scope="session_1",
            ),
            make_record(
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
            make_record(
                entry_id="turn-meta",
                entity="Alice",
                attribute="dialogue",
                value="I switched to Meta.",
                timestamp=2,
                session_id="s",
                scope="session_2",
            ),
            make_record(
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

    query = make_query(
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


def test_odv2_dense_uses_soft_entity_matching_when_query_entity_is_wrong() -> None:
    p = _dense_policy()
    support = make_record(
        entry_id="turn-assistant",
        entity="assistant",
        attribute="dialogue",
        value="I started working at Google.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    fact = make_record(
        entry_id="fact-assistant-google",
        entity="assistant",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        scope="session_1",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-assistant",
            "support_text": "I started working at Google.",
            "memory_kind": "state",
        },
    )
    p.ingest([support, fact])
    p.maybe_consolidate()

    result = p.retrieve_for_query(
        make_query(
            query_id="q-cross-entity",
            entity="user",
            attribute="employer",
            question="Where does the assistant work now?",
            answer="Google",
            timestamp=2,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert any(entry.entry_id == "fact-assistant-google" for entry in result.entries)
    assert any(entry.entry_id == "turn-assistant" for entry in result.entries)


def test_odv2_strong_fallback_does_not_drop_cross_entity_raw_evidence() -> None:
    p = _strong_policy()
    support = make_record(
        entry_id="turn-assistant",
        entity="assistant",
        attribute="dialogue",
        value="The venue was Cafe Luna.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    p.ingest([support])
    p.maybe_consolidate()

    result = p.retrieve_for_query(
        make_query(
            query_id="q-raw-cross-entity",
            entity="user",
            attribute="dialogue",
            question="Which venue was Cafe Luna?",
            answer="Cafe Luna",
            timestamp=2,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert any(entry.entry_id == "turn-assistant" for entry in result.entries)


def test_odv2_dense_evidence_lane_prefers_semantic_match_over_wrong_anchor() -> None:
    p = _dense_policy()
    wrong_support = make_record(
        entry_id="turn-meta",
        entity="Alice",
        attribute="dialogue",
        value="I started working at Meta.",
        timestamp=1,
        session_id="s",
        scope="session_1",
    )
    wrong_fact = make_record(
        entry_id="fact-meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=1,
        session_id="s",
        scope="session_1",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-meta",
            "support_text": "I started working at Meta.",
            "memory_kind": "state",
        },
    )
    correct_raw = make_record(
        entry_id="turn-google",
        entity="Alice",
        attribute="dialogue",
        value="The correct employer is Google.",
        timestamp=2,
        session_id="s",
        scope="session_2",
    )
    p.ingest([wrong_support, wrong_fact, correct_raw])
    p.maybe_consolidate()

    result = p.retrieve_for_query(
        make_query(
            query_id="q-evidence-semantic",
            entity="Alice",
            attribute="employer",
            question="Where does Alice work at Google?",
            answer="Google",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )
    entry_ids = [entry.entry_id for entry in result.entries]

    assert entry_ids.index("turn-google") < entry_ids.index("turn-meta")


def test_odv2_hybrid_policy_names_are_explicit() -> None:
    assert _strong_policy().name == "odv2_strong"
    assert _dense_policy().name == "odv2_dense"
    assert _dense_compact_policy().name == "odv2_dense_compact"
