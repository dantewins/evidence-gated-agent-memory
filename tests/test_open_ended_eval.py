from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
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


def test_recency_salience_can_diverge_from_append_only_on_equal_lexical_match() -> None:
    entries = [
        MemoryEntry(
            entry_id="older",
            entity="user",
            attribute="dialogue",
            value="I graduated with Business Administration in 2022.",
            timestamp=1,
            session_id="s",
            importance=1.7,
            confidence=0.95,
        ),
        MemoryEntry(
            entry_id="newer",
            entity="user",
            attribute="dialogue",
            value="I graduated with honors.",
            timestamp=2,
            session_id="s",
            importance=0.45,
            confidence=0.6,
        ),
    ]
    query = Query(
        query_id="q3",
        entity="user",
        attribute="dialogue",
        question="What did I graduate with?",
        answer="Business Administration",
        timestamp=3,
        session_id="s",
    )
    append_only = AppendOnlyMemoryPolicy()
    recency_salience = RecencySalienceMemoryPolicy()
    append_only.ingest(entries)
    recency_salience.ingest(entries)

    append_ids = [entry.entry_id for entry in append_only.retrieve_for_query(query).entries]
    recency_ids = [entry.entry_id for entry in recency_salience.retrieve_for_query(query).entries]

    assert append_ids[0] == "newer"
    assert recency_ids[0] == "older"


def test_policy_shortlist_keeps_salient_older_open_ended_evidence() -> None:
    entries = [
        MemoryEntry(
            entry_id="target",
            entity="user",
            attribute="dialogue",
            value="I graduated with Business Administration.",
            timestamp=0,
            session_id="s",
            importance=1.8,
            confidence=0.95,
        )
    ]
    entries.extend(
        MemoryEntry(
            entry_id=f"recent-{idx}",
            entity="user",
            attribute="dialogue",
            value=f"I went hiking on weekend {idx}.",
            timestamp=idx,
            session_id="s",
            importance=0.15,
            confidence=0.25,
        )
        for idx in range(1, 71)
    )
    query = Query(
        query_id="q4",
        entity="user",
        attribute="dialogue",
        question="What degree did I graduate with?",
        answer="Business Administration",
        timestamp=72,
        session_id="s",
    )

    append_only = AppendOnlyMemoryPolicy()
    recency_salience = RecencySalienceMemoryPolicy()
    append_only.ingest(entries)
    recency_salience.ingest(entries)

    append_top = append_only.retrieve_for_query(query).entries[0].value
    recency_top = recency_salience.retrieve_for_query(query).entries[0].value

    assert "Business Administration" not in append_top
    assert "Business Administration" in recency_top
