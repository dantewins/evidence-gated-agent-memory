from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_FACT_END_RE = re.compile(
    r"\s+(?:instead|now|though|actually|really|yesterday|today|tomorrow|"
    r"last\s+\w+|next\s+\w+|four\s+years\s+ago|years?\s+ago|months?\s+ago)\b.*$",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StructuredFact:
    attribute: str
    value: str


_LOCATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("home_city", re.compile(r"\b(?:live|lived|living|based)\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("home_city", re.compile(r"\b(?:move|moved|moving)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("origin", re.compile(r"\b(?:move|moved|moving)\s+from\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
)
_AFFILIATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("employer", re.compile(r"\b(?:job|work|worked)\s+(?:at|for)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\b(?:switched|switching)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\bjoined\s+(?P<value>[A-Z][^.!?;,]+)", re.IGNORECASE)),
)
_EDUCATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("education", re.compile(r"\bgraduated\s+with(?:\s+a\s+degree\s+in)?\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\bdegree\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\b(?:study|studied)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
)
_PREFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("preference", re.compile(r"\b(?:prefer|preferred|like|love)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\bfavorite\s+(?:[^.!?;,]+?)\s+is\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
)
_POSSESSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("possession", re.compile(r"\b(?:bought|purchased|own|owns)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("possession", re.compile(r"\bgot(?:\s+a\s+new)?\s+(?!job\b)(?P<value>[^.!?;,]+)", re.IGNORECASE)),
)

_ALL_PATTERNS = (
    _LOCATION_PATTERNS
    + _AFFILIATION_PATTERNS
    + _EDUCATION_PATTERNS
    + _PREFERENCE_PATTERNS
    + _POSSESSION_PATTERNS
)


def extract_structured_facts(text: str) -> list[StructuredFact]:
    """Extract lightweight slot/value facts from conversational text.

    These facts are intentionally coarse. The goal is not full IE fidelity;
    it is to give the memory layer attribute keys that can be revised over time
    instead of forcing every benchmark sample through a single `dialogue` slot.
    """
    content = text.strip()
    if not content or content.endswith("?"):
        return []

    facts: list[StructuredFact] = []
    seen: set[tuple[str, str]] = set()
    for attribute, pattern in _ALL_PATTERNS:
        for match in pattern.finditer(content):
            value = _clean_value(match.group("value"))
            if not value:
                continue
            key = (attribute, value.casefold())
            if key in seen:
                continue
            seen.add(key)
            facts.append(StructuredFact(attribute=attribute, value=value))
    return facts


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
    if re.search(r"\b(degree|graduate|graduated|study|studied|major)\b", normalized):
        candidates.append("education")
    if re.search(r"\b(prefer|preferred|favorite|like|love)\b", normalized):
        candidates.append("preference")
    if re.search(r"\b(bought|buy|purchased|purchase|own|owns|got)\b", normalized):
        candidates.append("possession")
    return tuple(dict.fromkeys(candidates))


def choose_query_attribute(
    question: str,
    entity: str,
    updates: Sequence[object],
    *,
    fallback: str,
) -> str:
    candidates = infer_query_attributes(question)
    if not candidates:
        return fallback

    for attribute in candidates:
        if _has_matching_attribute(updates, entity=entity, attribute=attribute):
            return attribute
    return fallback


def _has_matching_attribute(updates: Sequence[object], *, entity: str, attribute: str) -> bool:
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute != attribute:
            continue
        if entity in {"conversation", "all"} or update_entity == entity:
            return True
    return False


def _clean_value(value: str) -> str:
    cleaned = _FACT_END_RE.sub("", value)
    cleaned = cleaned.strip(" \t\n\r\"'`.,;:!?")
    cleaned = re.sub(r"^(?:a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) < 2:
        return ""
    return cleaned
