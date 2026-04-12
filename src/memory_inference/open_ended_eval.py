from __future__ import annotations

import re
from typing import Iterable, Sequence

from memory_inference.types import MemoryEntry, Query, RetrievalResult

_CONVERSATIONAL_ATTRIBUTES = frozenset({"dialogue", "event"})
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "their",
        "they",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
        "would",
        "you",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def is_open_ended_query(query: Query) -> bool:
    return query.attribute in _CONVERSATIONAL_ATTRIBUTES


def lexical_retrieval(
    entries: Iterable[MemoryEntry],
    query: Query,
    *,
    top_k: int = 8,
    policy_name: str = "",
    secondary_score_fn=None,
) -> RetrievalResult:
    entry_list = list(entries)
    if query.entity and query.entity not in {"conversation", "all"}:
        entity_matches = [entry for entry in entry_list if entry.entity == query.entity]
        if entity_matches:
            entry_list = entity_matches
    ranked = sorted(
        entry_list,
        key=lambda entry: _combined_score(entry, query, secondary_score_fn),
        reverse=True,
    )
    unique: list[MemoryEntry] = []
    seen_ids: set[str] = set()
    for entry in ranked:
        if entry.entry_id in seen_ids:
            continue
        seen_ids.add(entry.entry_id)
        unique.append(entry)
        if len(unique) >= top_k:
            break
    return RetrievalResult(
        entries=unique,
        debug={
            "policy": policy_name,
            "retrieval_mode": "lexical_open_ended",
        },
    )


def lexical_score(entry: MemoryEntry, query: Query) -> tuple[float, ...]:
    return _score_entry(entry, query)


def answers_match(prediction: str, gold: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if normalized_prediction == normalized_gold:
        return True
    if not normalized_prediction or not normalized_gold:
        return False
    if _contains_as_span(normalized_prediction, normalized_gold):
        return True
    if _contains_as_span(normalized_gold, normalized_prediction):
        return True
    return False


def normalize_answer(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_as_span(container: str, span: str) -> bool:
    return bool(re.search(rf"(^|\s){re.escape(span)}($|\s)", container))


def _score_entry(entry: MemoryEntry, query: Query) -> tuple[float, ...]:
    question_tokens = _content_tokens(query.question)
    entry_tokens = _content_tokens(
        " ".join(
            [
                entry.entity,
                entry.attribute,
                entry.value,
                " ".join(entry.metadata.values()),
            ]
        )
    )

    lexical_overlap = len(question_tokens & entry_tokens)
    entity_bonus = 0.0
    question_lower = query.question.lower()
    if entry.entity and entry.entity.lower() in question_lower:
        entity_bonus = 1.0
    elif query.entity and entry.entity.lower() == query.entity.lower():
        entity_bonus = 0.5

    provenance_bonus = 0.2 if entry.provenance.endswith("event_summary") else 0.0
    attribute_bonus = 0.1 if entry.attribute in _CONVERSATIONAL_ATTRIBUTES else 0.0

    return (
        float(lexical_overlap) + entity_bonus + provenance_bonus + attribute_bonus,
    )


def _combined_score(entry: MemoryEntry, query: Query, secondary_score_fn) -> tuple[float, ...]:
    primary = _score_entry(entry, query)
    if secondary_score_fn is None:
        return primary
    secondary = secondary_score_fn(entry)
    if isinstance(secondary, tuple):
        return primary + secondary
    return primary + (secondary,)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }
