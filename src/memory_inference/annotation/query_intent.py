from __future__ import annotations

import re
from typing import Sequence

from memory_inference.domain.enums import QueryMode

_LOCOMO_CATEGORY_TO_MODE = {
    "single-hop": QueryMode.CURRENT_STATE,
    "multi-hop": QueryMode.CURRENT_STATE,
    "temporal": QueryMode.HISTORY,
    "open-ended": QueryMode.CURRENT_STATE,
    "adversarial": QueryMode.CONFLICT_AWARE,
    "1": QueryMode.CURRENT_STATE,
    "2": QueryMode.HISTORY,
    "3": QueryMode.CURRENT_STATE,
    "4": QueryMode.CURRENT_STATE,
    "5": QueryMode.CONFLICT_AWARE,
}

_LONGMEMEVAL_QUESTION_TYPE_TO_MODE = {
    "single-session-user": QueryMode.CURRENT_STATE,
    "single-session-assistant": QueryMode.CURRENT_STATE,
    "single-session-preference": QueryMode.CURRENT_STATE,
    "multi-session": QueryMode.CURRENT_STATE,
    "temporal-reasoning": QueryMode.HISTORY,
    "knowledge-update": QueryMode.CURRENT_STATE,
    "temporal-ordering": QueryMode.HISTORY,
}


def infer_query_attributes(question: str) -> tuple[str, ...]:
    normalized = question.lower()
    if re.search(r"\b(when|what time|date|day|month|year)\b", normalized):
        return ()

    candidates: list[str] = []
    location_question = bool(re.search(r"\b(where|city|town|location)\b", normalized))
    if location_question and re.search(r"\b(move|moved)\s+from\b|\bfrom where\b", normalized):
        candidates.append("origin")
    elif location_question and re.search(r"\b(live|living|moved|move|based)\b", normalized):
        candidates.append("home_city")

    if re.search(r"\b(work|works|worked|job|employer|company)\b", normalized):
        candidates.append("employer")
    if re.search(r"\b(commute|travel to work|drive to work)\b", normalized):
        candidates.append("commute_duration")
    if re.search(r"\b(degree|graduate|graduated|study|studied|major)\b", normalized):
        candidates.append("education")
    if re.search(r"\b(prefer|preferred|favorite|like|love)\b", normalized):
        candidates.append("preference")
    if re.search(r"\b(bought|buy|purchased|purchase|own|owns|got)\b", normalized):
        candidates.append("possession")
    if re.search(r"\b(redeem|coupon|store|shop|bought .* at|purchase .* from)\b", normalized):
        candidates.append("venue")
    if re.search(r"\b(play|concert|show|movie|event)\b.*\b(attend|attended|watch|watched|see|saw)\b", normalized):
        candidates.append("attended_event")
    if re.search(r"\bname of\b|\bplaylist\b|\bcalled\b|\bnamed\b|\btitled\b", normalized):
        candidates.append("created_name")
    if re.search(r"\bidentity\b|\bidentify\b", normalized):
        candidates.append("identity")
    if re.search(r"\brelationship status\b|\bsingle\b|\bmarried\b|\bengaged\b|\bdivorced\b", normalized):
        candidates.append("relationship_status")
    if re.search(r"\bresearch\b|\bresearched\b|\blooking into\b|\blooked into\b|\bexploring\b", normalized):
        candidates.append("research_topic")
    return tuple(dict.fromkeys(candidates))


def choose_query_attribute(
    question: str,
    entity: str,
    updates: Sequence[object],
    *,
    default_attribute: str,
) -> str:
    candidates = infer_query_attributes(question)
    if not candidates:
        return default_attribute

    for attribute in candidates:
        if _has_matching_attribute(updates, entity=entity, attribute=attribute):
            return attribute
    return default_attribute


def locomo_query_mode(category: object) -> QueryMode:
    return _LOCOMO_CATEGORY_TO_MODE.get(str(category).strip().lower(), QueryMode.CURRENT_STATE)


def longmemeval_query_mode(question_type: str) -> QueryMode:
    return _LONGMEMEVAL_QUESTION_TYPE_TO_MODE.get(question_type, QueryMode.CURRENT_STATE)


def should_skip_locomo_question(category: object, answer: object) -> bool:
    normalized_category = str(category).strip().lower()
    if answer is not None and str(answer).strip():
        return False
    return normalized_category in {"5", "adversarial"}


def infer_locomo_query_entity(question: str, speakers: set[str]) -> str:
    question_lower = question.lower()
    matches = [
        speaker
        for speaker in speakers
        if speaker and speaker.lower() in question_lower
    ]
    if len(matches) == 1:
        return matches[0]
    return "conversation"


def infer_longmemeval_query_entity(question_type: str) -> str:
    if question_type == "single-session-assistant":
        return "assistant"
    return "user"


def _has_matching_attribute(updates: Sequence[object], *, entity: str, attribute: str) -> bool:
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute != attribute:
            continue
        if entity in {"conversation", "all"} or update_entity == entity:
            return True
    return False
