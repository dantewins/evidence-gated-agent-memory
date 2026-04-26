from memory_inference.domain.enums import QueryMode
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies.presets import odv2_recovery_policy
from tests.factories import make_query, make_record


class KeywordDenseEncoder:
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


def test_odv2_recovery_uses_history_aware_fallback_for_temporal_queries() -> None:
    policy = odv2_recovery_policy(
        consolidator=MockConsolidator(),
        encoder=KeywordDenseEncoder(),
    )
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
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q-history",
            entity="Alice",
            attribute="employer",
            question="Where did Alice work before Meta?",
            answer="Google",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.HISTORY,
        ),
        top_k=1,
    )

    assert result.debug["retrieval_mode"] == "odv2_recovery_history"
    assert result.entries[0].value == "Google"


def test_odv2_recovery_filters_stale_current_state_support() -> None:
    policy = odv2_recovery_policy(
        consolidator=MockConsolidator(),
        encoder=KeywordDenseEncoder(),
    )
    old_support = make_record(
        entry_id="turn-google",
        entity="Alice",
        attribute="dialogue",
        value="I got a job at Google.",
        timestamp=1,
        session_id="s",
    )
    old_fact = make_record(
        entry_id="fact-google",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-google",
            "support_text": "I got a job at Google.",
            "memory_kind": "state",
        },
    )
    new_support = make_record(
        entry_id="turn-meta",
        entity="Alice",
        attribute="dialogue",
        value="I switched to Meta.",
        timestamp=2,
        session_id="s",
    )
    new_fact = make_record(
        entry_id="fact-meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        metadata={
            "source_kind": "structured_fact",
            "source_entry_id": "turn-meta",
            "support_text": "I switched to Meta.",
            "memory_kind": "state",
        },
    )
    policy.ingest([old_support, old_fact, new_support, new_fact])
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q-current",
            entity="Alice",
            attribute="employer",
            question="Where does Alice work now?",
            answer="Meta",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )
    ids = {entry.entry_id for entry in result.entries}

    assert result.debug["retrieval_mode"] == "odv2_recovery_current_guard"
    assert "fact-meta" in ids
    assert "turn-meta" in ids
    assert "turn-google" not in ids
    assert "Google" not in {entry.value for entry in result.entries if entry.attribute == "employer"}
