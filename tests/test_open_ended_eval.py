from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.llm.prompting import build_reasoning_prompt
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
