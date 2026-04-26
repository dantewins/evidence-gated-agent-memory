from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies.presets import odv2_mem0_temporal_prune_policy
from tests.factories import make_query, make_record
from tests.test_odv2_mem0_selective_policy import KeywordDenseEncoder


def _policy():
    return odv2_mem0_temporal_prune_policy(
        consolidator=MockConsolidator(),
        encoder=KeywordDenseEncoder(),
    )


def test_temporal_prune_removes_older_same_key_conflict_without_appending() -> None:
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

    def retrieve_with_conflict(_, top_k=5):
        return RetrievalBundle(
            records=[old_fact, current_fact][:top_k],
            debug={"retrieval_mode": "stub_mem0_conflict"},
        )

    policy.retriever.retrieve_for_query = retrieve_with_conflict
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

    assert [entry.value for entry in result.entries] == ["Meta"]
    assert result.debug["retrieval_mode"] == "odv2_mem0_temporal_prune"
    assert result.debug["temporal_pruned"] == "1"
    assert result.debug["decision_source"] == "ledger"


def test_temporal_prune_passes_through_history_queries() -> None:
    policy = _policy()
    old_fact = make_record(
        entry_id="old",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        metadata={"memory_kind": "state"},
    )
    current_fact = make_record(
        entry_id="new",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        metadata={"memory_kind": "state"},
    )
    policy.ingest([old_fact, current_fact])
    policy.maybe_consolidate()

    def retrieve_with_conflict(_, top_k=5):
        return RetrievalBundle(
            records=[old_fact, current_fact][:top_k],
            debug={"retrieval_mode": "stub_mem0_conflict"},
        )

    policy.retriever.retrieve_for_query = retrieve_with_conflict
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
        top_k=5,
    )

    assert [entry.value for entry in result.entries] == ["Google", "Meta"]
    assert result.debug["retrieval_mode"] == "odv2_mem0_temporal_passthrough"
