from __future__ import annotations

import re
from typing import Callable, Iterable

from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery

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


def lexical_retrieval(
    entries: Iterable[MemoryRecord],
    query: RuntimeQuery,
    *,
    top_k: int = 8,
    policy_name: str = "",
    secondary_score_fn=None,
) -> RetrievalBundle:
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
    unique: list[MemoryRecord] = []
    seen_ids: set[str] = set()
    for entry in ranked:
        if entry.entry_id in seen_ids:
            continue
        seen_ids.add(entry.entry_id)
        unique.append(entry)
        if len(unique) >= top_k:
            break
    return RetrievalBundle(
        records=unique,
        debug={
            "policy": policy_name,
            "retrieval_mode": "lexical_open_ended",
        },
    )


def shortlist_open_ended_candidates(
    entries: Iterable[MemoryRecord],
    query: RuntimeQuery,
    *,
    score_fn: Callable[[MemoryRecord], tuple[float, ...] | float],
    limit: int,
) -> list[MemoryRecord]:
    entry_list = list(entries)
    if query.entity and query.entity not in {"conversation", "all"}:
        entity_matches = [entry for entry in entry_list if entry.entity == query.entity]
        if entity_matches:
            entry_list = entity_matches
    ranked = sorted(
        entry_list,
        key=lambda entry: _coerce_score_key(score_fn(entry)),
        reverse=True,
    )
    unique: list[MemoryRecord] = []
    seen_ids: set[str] = set()
    for entry in ranked:
        if entry.entry_id in seen_ids:
            continue
        seen_ids.add(entry.entry_id)
        unique.append(entry)
        if len(unique) >= limit:
            break
    return unique


def _score_entry(entry: MemoryRecord, query: RuntimeQuery) -> tuple[float, ...]:
    question_tokens = _content_tokens(query.question)
    entry_tokens = _content_tokens(
        " ".join(
            [
                entry.entity,
                entry.attribute,
                entry.value,
                entry.support_text,
                entry.source_kind,
                entry.source_attribute,
                entry.memory_kind,
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
    attribute_bonus = 0.1 if entry.attribute in {"dialogue", "event"} else 0.0

    return (
        float(lexical_overlap) + entity_bonus + provenance_bonus + attribute_bonus,
    )


def _combined_score(entry: MemoryRecord, query: RuntimeQuery, secondary_score_fn) -> tuple[float, ...]:
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


def _coerce_score_key(score: tuple[float, ...] | float) -> tuple[float, ...]:
    if isinstance(score, tuple):
        return score
    return (float(score),)


class LexicalBackboneRanker:
    name = "strong"

    def index_entries(self, entries: Iterable[MemoryRecord]) -> None:
        del entries

    def rank(
        self,
        query: RuntimeQuery,
        candidates: Iterable[MemoryRecord],
        *,
        score_fn,
        limit: int,
    ) -> list[MemoryRecord]:
        shortlisted = shortlist_open_ended_candidates(
            candidates,
            query,
            score_fn=lambda entry: score_fn(entry, None),
            limit=limit,
        )
        result = lexical_retrieval(
            shortlisted,
            query,
            top_k=limit,
            secondary_score_fn=lambda entry: score_fn(entry, None),
        )
        return list(result.entries)

    def backbone_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context,
    ) -> tuple[float, ...]:
        del query_context
        exact_entity = 1.0 if query.entity in {"conversation", "all"} or entry.entity == query.entity else 0.0
        exact_attribute = 1.0 if entry.attribute == query.attribute else 0.0
        scope_bonus = 1.0 if entry.scope == "default" else 0.5
        return (
            exact_entity + exact_attribute,
            entry.importance,
            float(entry.access_count),
            scope_bonus,
            self._time_bias(entry, query),
        )

    def structured_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context,
        *,
        memory_kind: str,
    ) -> tuple[float, ...]:
        del query_context
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.5,
            MemoryStatus.SUPERSEDED: 0.3,
            MemoryStatus.ARCHIVED: 0.2,
        }.get(entry.status, 0.0)
        memory_kind_bonus = 0.4 if memory_kind == "state" else 0.0
        support_bonus = 0.2 if entry.source_entry_id else 0.0
        return self.backbone_score(entry, query, None) + (
            status_bonus,
            memory_kind_bonus,
            support_bonus,
        )

    def evidence_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context,
        *,
        anchor_source_ids: set[str],
        anchor_scopes: set[str],
    ) -> tuple[float, ...]:
        del query_context
        anchor_bonus = 1.5 if entry.entry_id in anchor_source_ids else 0.0
        scope_bonus = 1.0 if entry.scope in anchor_scopes else 0.0
        attribute_bonus = (
            1.0 if entry.attribute == query.attribute
            else 0.8 if entry.attribute in {"dialogue", "event"}
            else 0.0
        )
        return (
            anchor_bonus,
            scope_bonus,
            attribute_bonus,
            entry.importance,
            entry.confidence,
            self._time_bias(entry, query),
        )

    def open_ended_mode(self) -> str:
        return "lexical_open_ended"

    def backbone_mode(self) -> str:
        return "lexical_backbone"

    @staticmethod
    def _time_bias(entry: MemoryRecord, query: RuntimeQuery) -> float:
        if query.query_mode == QueryMode.HISTORY:
            return -float(entry.timestamp)
        return float(entry.timestamp)
