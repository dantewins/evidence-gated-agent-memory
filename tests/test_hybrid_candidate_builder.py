from memory_inference.domain.enums import QueryMode
from memory_inference.memory.retrieval import HybridCandidateBuilder, HybridMergeStrategy
from tests.factories import make_query, make_record


def test_hybrid_candidate_builder_uses_anchor_sources_and_scopes_for_evidence() -> None:
    builder = HybridCandidateBuilder(entity_matches=lambda left, right: left == right)
    query = make_query(
        query_id="q",
        entity="Alice",
        attribute="employer",
        question="Where does Alice work now?",
        answer="Google",
        timestamp=2,
        session_id="s",
        query_mode=QueryMode.CURRENT_STATE,
    )
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
        metadata={"source_kind": "structured_fact", "source_entry_id": "turn-1"},
    )
    unrelated = make_record(
        entry_id="turn-2",
        entity="Alice",
        attribute="dialogue",
        value="I also like hiking.",
        timestamp=3,
        session_id="s",
        scope="session_2",
    )

    anchor_ids = builder.anchor_source_ids([fact])
    anchor_scopes = builder.anchor_scopes([fact])
    evidence = builder.evidence_candidates(
        query,
        episodic_log=[support, fact, unrelated],
        anchor_source_ids=anchor_ids,
        anchor_scopes=anchor_scopes,
    )

    evidence_ids = {entry.entry_id for entry in evidence}
    assert "turn-1" in evidence_ids
    assert "fact-1" in evidence_ids
    assert "turn-2" in evidence_ids


def test_hybrid_merge_strategy_prioritizes_state_then_evidence_without_duplicates() -> None:
    merger = HybridMergeStrategy()
    state = [
        make_record("fact-1", "Alice", "employer", "Google", 1, "s"),
        make_record("fact-2", "Alice", "title", "Engineer", 2, "s"),
    ]
    evidence = [
        make_record("fact-1", "Alice", "employer", "Google", 1, "s"),
        make_record("turn-1", "Alice", "dialogue", "I work at Google.", 1, "s"),
    ]

    merged = merger.merge(state_entries=state, evidence_entries=evidence, top_k=3)

    assert [entry.entry_id for entry in merged] == ["fact-1", "fact-2", "turn-1"]
