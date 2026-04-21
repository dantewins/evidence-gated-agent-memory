from memory_inference.annotation.fact_extractor import extract_structured_facts
from memory_inference.annotation.query_intent import infer_query_attributes


def test_extract_structured_facts_covers_benchmark_event_patterns() -> None:
    facts = extract_structured_facts(
        "I redeemed a $5 coupon on coffee creamer at Target and created a playlist on Spotify called Summer Vibes."
    )
    by_key = {(fact.attribute, fact.value): fact for fact in facts}

    assert by_key[("venue", "Target")].is_stateful is False
    assert by_key[("created_name", "Summer Vibes")].is_stateful is False


def test_extract_structured_facts_covers_identity_and_relationship_status() -> None:
    facts = extract_structured_facts(
        "I identify as a transgender woman and my relationship status is single."
    )
    by_key = {(fact.attribute, fact.value): fact for fact in facts}

    assert by_key[("identity", "transgender woman")].is_stateful is True
    assert by_key[("relationship_status", "single")].is_stateful is True


def test_infer_query_attributes_covers_benchmark_question_templates() -> None:
    assert "attended_event" in infer_query_attributes(
        "What play did I attend at the local community theater?"
    )
    assert "venue" in infer_query_attributes(
        "Where did I redeem a $5 coupon on coffee creamer?"
    )
    assert "created_name" in infer_query_attributes(
        "What is the name of the playlist I created on Spotify?"
    )
    assert "commute_duration" in infer_query_attributes(
        "How long is my daily commute to work?"
    )
