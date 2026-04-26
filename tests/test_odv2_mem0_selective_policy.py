from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies.presets import odv2_mem0_selective_policy
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
            1.0 if "hiking" in lower else 0.0,
        )


def _policy():
    return odv2_mem0_selective_policy(
        consolidator=MockConsolidator(),
        encoder=KeywordDenseEncoder(),
    )


def test_selective_policy_passes_through_mem0_when_no_validity_signal() -> None:
    policy = _policy()
    policy.ingest(
        [
            make_record(
                entry_id="hiking",
                entity="Alice",
                attribute="dialogue",
                value="I enjoy hiking every weekend.",
                timestamp=1,
                session_id="s",
            )
        ]
    )
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q-dialogue",
            entity="Alice",
            attribute="dialogue",
            question="What does Alice enjoy?",
            timestamp=2,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert result.debug["retrieval_mode"] == "odv2_mem0_selective_passthrough"
    assert [entry.entry_id for entry in result.entries] == ["hiking"]


def test_selective_policy_removes_stale_state_only_when_mem0_has_current_evidence() -> None:
    policy = _policy()
    old_fact = make_record(
        entry_id="fact-google",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    current_fact = make_record(
        entry_id="fact-meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    policy.ingest([old_fact, current_fact])
    policy.maybe_consolidate()

    def retrieve_with_both_sides(_, top_k=5):
        return RetrievalBundle(
            records=[old_fact, current_fact][:top_k],
            debug={"retrieval_mode": "stub_mem0_with_conflict"},
        )

    policy.retriever.retrieve_for_query = retrieve_with_both_sides
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
        ),
        top_k=5,
    )
    values = [entry.value for entry in result.entries]

    assert result.debug["retrieval_mode"] == "odv2_mem0_selective_guard"
    assert values == ["Meta"]
    assert result.debug["validity_appended"] == "0"


def test_selective_policy_does_not_intervene_on_history_queries() -> None:
    policy = _policy()
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
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.HISTORY,
        ),
        top_k=2,
    )

    assert result.debug["retrieval_mode"] == "odv2_mem0_selective_passthrough"
    assert {entry.value for entry in result.entries} == {"Google", "Meta"}
