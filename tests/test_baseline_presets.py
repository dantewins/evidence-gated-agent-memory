from memory_inference.memory.policies import AppendOnlyMemoryPolicy
from memory_inference.memory.policies import ExactMatchMemoryPolicy
from memory_inference.memory.policies import RecencySalienceMemoryPolicy
from memory_inference.memory.policies import StrongRetrievalMemoryPolicy
from memory_inference.memory.policies import SummaryOnlyMemoryPolicy
from tests.factories import make_query, make_record


def test_log_based_presets_preserve_shared_ingest_and_distinct_ranking() -> None:
    entries = [
        make_record(
            entry_id="older",
            entity="user",
            attribute="dialogue",
            value="I graduated with Business Administration in 2022.",
            timestamp=1,
            session_id="s",
            importance=1.7,
            confidence=0.95,
        ),
        make_record(
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
    query = make_query(
        query_id="q",
        entity="user",
        attribute="dialogue",
        question="What did I graduate with?",
        answer="Business Administration",
        timestamp=3,
        session_id="s",
    )

    append_only = AppendOnlyMemoryPolicy()
    strong = StrongRetrievalMemoryPolicy()
    recency_salience = RecencySalienceMemoryPolicy()
    for policy in (append_only, strong, recency_salience):
        policy.ingest(entries)
        assert policy.snapshot_size() == 2

    assert append_only.retrieve_for_query(query).entries[0].entry_id == "newer"
    assert strong.retrieve_for_query(query).entries[0].entry_id == "older"
    assert recency_salience.retrieve_for_query(query).entries[0].entry_id == "older"


def test_exact_and_summary_presets_keep_latest_reduced_state() -> None:
    updates = [
        make_record(
            entry_id="old",
            entity="user",
            attribute="home_city",
            value="Boston",
            timestamp=1,
            session_id="s",
            scope="default",
        ),
        make_record(
            entry_id="new",
            entity="user",
            attribute="home_city",
            value="Seattle",
            timestamp=2,
            session_id="s",
            scope="default",
        ),
    ]

    exact_match = ExactMatchMemoryPolicy()
    summary_only = SummaryOnlyMemoryPolicy()
    exact_match.ingest(updates)
    summary_only.ingest(updates)

    assert exact_match.snapshot_size() == 1
    assert summary_only.snapshot_size() == 1
    assert exact_match.retrieve("user", "home_city").entries[0].value == "Seattle"
    assert summary_only.retrieve("user", "home_city").entries[0].value == "Seattle"
