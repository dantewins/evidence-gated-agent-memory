from memory_inference.memory.retrieval.semantic import DenseEncoder
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies.presets import (
    odv2_dense_policy,
    odv2_strong_policy,
    offline_delta_v2_policy,
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


def test_odv2_policy_presets_have_expected_names_and_backbones() -> None:
    offline = offline_delta_v2_policy(consolidator=MockConsolidator())
    strong = odv2_strong_policy(consolidator=MockConsolidator())
    dense = odv2_dense_policy(consolidator=MockConsolidator(), encoder=FakeDenseEncoder())

    assert offline.name == "offline_delta_v2"
    assert offline.hybrid_backbone is None
    assert strong.name == "odv2_strong"
    assert strong.hybrid_backbone.name == "strong"
    assert dense.name == "odv2_dense"
    assert dense.hybrid_backbone.name == "dense"


def test_odv2_dense_preset_returns_hybrid_state_and_support() -> None:
    policy = odv2_dense_policy(consolidator=MockConsolidator(), encoder=FakeDenseEncoder())
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
    policy.ingest([support, fact])
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q",
            entity="Alice",
            attribute="employer",
            question="Where does Alice work now?",
            answer="Google",
            timestamp=2,
            session_id="s",
        )
    )

    assert result.debug["retrieval_mode"] == "hybrid_state_evidence"
    assert result.debug["backbone"] == "dense"
    assert any(entry.attribute == "employer" and entry.value == "Google" for entry in result.entries)
    assert any(entry.entry_id == "turn-1" for entry in result.entries)
