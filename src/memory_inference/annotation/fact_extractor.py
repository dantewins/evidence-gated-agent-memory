from __future__ import annotations

import re
from dataclasses import dataclass

_FACT_END_RE = re.compile(
    r"\s+(?:instead|now|though|actually|really|yesterday|today|tomorrow|"
    r"last\s+\w+|next\s+\w+|four\s+years\s+ago|years?\s+ago|months?\s+ago)\b.*$",
    re.IGNORECASE,
)
_CLAUSE_TAIL_RE = re.compile(
    r"\s+and\s+(?:i|we|you|he|she|they|my|our|your|his|her|their|created|made|started|went|watched|saw|researched)\b.*$",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StructuredFact:
    attribute: str
    value: str
    is_stateful: bool = True


_STATEFUL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("home_city", re.compile(r"\b(?:live|lived|living|based)\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("home_city", re.compile(r"\b(?:move|moved|moving)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("origin", re.compile(r"\b(?:move|moved|moving)\s+from\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\b(?:job|work|worked)\s+(?:at|for)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\b(?:switched|switching)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\bjoined\s+(?P<value>[A-Z][^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\bgraduated\s+with(?:\s+a\s+degree\s+in)?\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\bdegree\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\b(?:study|studied)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\b(?:prefer|preferred|like|love)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\bfavorite\s+(?:[^.!?;,]+?)\s+is\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("possession", re.compile(r"\b(?:bought|purchased|own|owns)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("possession", re.compile(r"\bgot(?:\s+a\s+new)?\s+(?!job\b)(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    (
        "commute_duration",
        re.compile(
            r"\bcommute(?:\s+to\s+work)?(?:\s+(?:is|was|takes?|took))?\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "identity",
        re.compile(
            r"\b(?:identify as|identifies as|identity is|i am a|i'm a|she is a|he is a|they are a)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "relationship_status",
        re.compile(
            r"\b(?:relationship status (?:is|was)\s+|(?:i am|i'm|she is|he is|they are)\s+)(?P<value>single|married|engaged|divorced|widowed|in a relationship)\b",
            re.IGNORECASE,
        ),
    ),
)
_EPISODIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "research_topic",
        re.compile(
            r"\b(?:research|researched|researching|looked into|looking into|exploring|explored)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    ("attended_event", re.compile(r"\b(?:attended|watched|saw)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("attended_event", re.compile(r"\bwent\s+to\s+(?:see\s+)?(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("venue", re.compile(r"\b(?:redeemed?|used)\b[^.!?;,]*\b(?:at|in)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    (
        "venue",
        re.compile(
            r"\b(?:bought|purchased|ordered|picked up|grabbed)\b[^.!?;,]*\b(?:at|from)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "created_name",
        re.compile(
            r"\b(?:created|made|started)\b[^.!?;,]*\b(?:called|named|titled)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "created_name",
        re.compile(
            r"\bnamed\s+(?:my|the)\s+(?:new\s+)?(?:playlist|list|document|spreadsheet|album|project)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
)

_ALL_PATTERNS: tuple[tuple[str, bool, re.Pattern[str]], ...] = tuple(
    (attribute, True, pattern) for attribute, pattern in _STATEFUL_PATTERNS
) + tuple(
    (attribute, False, pattern) for attribute, pattern in _EPISODIC_PATTERNS
)


def extract_structured_facts(text: str) -> list[StructuredFact]:
    content = text.strip()
    if not content or content.endswith("?"):
        return []

    facts: list[StructuredFact] = []
    seen: set[tuple[str, str, bool]] = set()
    for attribute, is_stateful, pattern in _ALL_PATTERNS:
        for match in pattern.finditer(content):
            value = _clean_value(match.group("value"))
            if not value:
                continue
            key = (attribute, value.casefold(), is_stateful)
            if key in seen:
                continue
            seen.add(key)
            facts.append(StructuredFact(attribute=attribute, value=value, is_stateful=is_stateful))
    return facts


def _clean_value(value: str) -> str:
    cleaned = _FACT_END_RE.sub("", value)
    cleaned = _CLAUSE_TAIL_RE.sub("", cleaned)
    cleaned = cleaned.strip(" \t\n\r\"'`.,;:!?")
    cleaned = re.sub(r"^(?:a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) < 2:
        return ""
    return cleaned
