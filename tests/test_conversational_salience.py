from memory_inference.benchmarks.conversational_salience import estimate_confidence, estimate_importance


def test_personal_fact_scores_higher_than_generic_help_request() -> None:
    help_request = (
        "Can you help me solve this river crossing puzzle? "
        "Remember, strategic thinking and planning are key."
    )
    personal_fact = (
        "I've been tracking my progress with my new Fitbit Inspire HR, "
        "which I bought on February 15th."
    )
    assert estimate_importance(personal_fact, speaker="user", attribute="dialogue") > (
        estimate_importance(help_request, speaker="user", attribute="dialogue")
    )
    assert estimate_confidence(personal_fact, speaker="user", attribute="dialogue") > (
        estimate_confidence(help_request, speaker="user", attribute="dialogue")
    )


def test_event_summaries_receive_structured_bonus() -> None:
    dialogue = "I moved from Sweden four years ago."
    event = "Moved from Sweden four years ago"
    assert estimate_importance(event, speaker="Caroline", attribute="event") > (
        estimate_importance(dialogue, speaker="Caroline", attribute="dialogue")
    )
