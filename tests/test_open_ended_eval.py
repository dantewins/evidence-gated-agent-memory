from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.llm.prompting import build_reasoning_prompt
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.open_ended_eval import answers_match
from memory_inference.types import MemoryEntry, Query


def test_answers_match_normalizes_short_span_predictions() -> None:
    assert answers_match("Business Administration.", "Business Administration")
    assert answers_match("She graduated with Business Administration", "Business Administration")
    assert not answers_match("Psychology", "Business Administration")


def test_append_only_uses_lexical_retrieval_for_dialogue_queries() -> None:
    policy = AppendOnlyMemoryPolicy()
    policy.ingest(
        [
            MemoryEntry(
                entry_id="1",
                entity="user",
                attribute="dialogue",
                value="I bought a Fitbit Inspire HR and want more steps.",
                timestamp=0,
                session_id="s",
            ),
            MemoryEntry(
                entry_id="2",
                entity="user",
                attribute="dialogue",
                value="I graduated with a degree in Business Administration.",
                timestamp=1,
                session_id="s",
            ),
        ]
    )
    query = Query(
        query_id="q",
        entity="user",
        attribute="dialogue",
        question="What degree did I graduate with?",
        answer="Business Administration",
        timestamp=2,
        session_id="s",
    )
    retrieved = policy.retrieve_for_query(query)
    assert retrieved.entries
    assert "Business Administration" in retrieved.entries[0].value


def test_prompt_includes_source_metadata() -> None:
    query = Query(
        query_id="q",
        entity="Caroline",
        attribute="dialogue",
        question="When did Caroline go to the support group?",
        answer="7 May 2023",
        timestamp=0,
        session_id="s",
    )
    prompt = build_reasoning_prompt(
        query,
        [
            MemoryEntry(
                entry_id="1",
                entity="Caroline",
                attribute="dialogue",
                value="I went to the LGBTQ support group.",
                timestamp=0,
                session_id="s",
                metadata={"source_date": "7 May 2023", "session_label": "session_1"},
            )
        ],
    )
    assert "source_date=7 May 2023" in prompt.user_prompt
    assert "session_label=session_1" in prompt.user_prompt


def test_offline_delta_open_ended_retrieval_prefers_current_scoped_entries() -> None:
    policy = OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator())
    policy.ingest(
        [
            MemoryEntry(
                entry_id="1",
                entity="user",
                attribute="dialogue",
                value="I lived in New York.",
                timestamp=0,
                session_id="s",
                scope="session_1",
            ),
            MemoryEntry(
                entry_id="2",
                entity="user",
                attribute="dialogue",
                value="I moved to Boston.",
                timestamp=1,
                session_id="s",
                scope="session_1",
            ),
            MemoryEntry(
                entry_id="3",
                entity="user",
                attribute="dialogue",
                value="I graduated with Business Administration.",
                timestamp=2,
                session_id="s",
                scope="session_2",
            ),
        ]
    )
    policy.maybe_consolidate()
    query = Query(
        query_id="q",
        entity="user",
        attribute="dialogue",
        question="What degree did I graduate with?",
        answer="Business Administration",
        timestamp=3,
        session_id="s",
    )
    retrieved = policy.retrieve_for_query(query)
    assert retrieved.entries
    assert retrieved.entries[0].scope == "session_2"
    assert "Business Administration" in retrieved.entries[0].value


def test_offline_delta_open_ended_retrieval_keeps_earlier_turns_within_scope() -> None:
    policy = OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator())
    policy.ingest(
        [
            MemoryEntry(
                entry_id="1",
                entity="user",
                attribute="dialogue",
                value="I graduated with Business Administration.",
                timestamp=0,
                session_id="s",
                scope="session_1",
            ),
            MemoryEntry(
                entry_id="2",
                entity="user",
                attribute="dialogue",
                value="Also I like hiking on weekends.",
                timestamp=1,
                session_id="s",
                scope="session_1",
            ),
        ]
    )
    policy.maybe_consolidate()
    query = Query(
        query_id="q2",
        entity="user",
        attribute="dialogue",
        question="What degree did I graduate with?",
        answer="Business Administration",
        timestamp=2,
        session_id="s",
    )
    retrieved = policy.retrieve_for_query(query)
    assert any("Business Administration" in entry.value for entry in retrieved.entries)
