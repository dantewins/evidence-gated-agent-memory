from __future__ import annotations

from typing import Callable, Iterable, List

from memory_inference.memory.policies.interface import BaseMemoryPolicy, ScoreStrategy
from memory_inference.memory.retrieval import (
    has_structured_fact_candidates,
    is_open_ended_query,
    lexical_retrieval,
    rerank_structured_candidates,
    shortlist_open_ended_candidates,
)
from memory_inference.domain.memory import MemoryKey, MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery


class ScoredBaselinePolicy(BaseMemoryPolicy):
    def __init__(
        self,
        *,
        name: str,
        score_strategy: ScoreStrategy,
        open_ended_shortlist_factor: int = 16,
        open_ended_min_limit: int = 64,
        open_ended_top_k_floor: int = 8,
        structured_shortlist_factor: int = 12,
        structured_min_limit: int = 48,
        support_limit: int = 2,
    ) -> None:
        super().__init__(name=name)
        self._score_strategy = score_strategy
        self._open_ended_shortlist_factor = open_ended_shortlist_factor
        self._open_ended_min_limit = open_ended_min_limit
        self._open_ended_top_k_floor = open_ended_top_k_floor
        self._structured_shortlist_factor = structured_shortlist_factor
        self._structured_min_limit = structured_min_limit
        self._support_limit = support_limit

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        for update in updates:
            self._store_update(update)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        ranked = sorted(
            self._direct_candidates(entity, attribute),
            key=lambda entry: self._score(entry, entity, attribute),
            reverse=True,
        )
        return RetrievalBundle(records=ranked[:top_k], debug={"policy": self.name})

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self._candidate_pool(),
                query,
                score_fn=lambda entry: self._score(entry, query.entity, query.attribute),
                limit=max(top_k * self._open_ended_shortlist_factor, self._open_ended_min_limit),
            )
            return lexical_retrieval(
                candidates,
                query,
                top_k=max(top_k, self._open_ended_top_k_floor),
                policy_name=self.name,
                secondary_score_fn=lambda entry: self._score(entry, query.entity, query.attribute),
            )

        candidates = self._attribute_candidates(query.attribute)
        if has_structured_fact_candidates(candidates):
            return rerank_structured_candidates(
                candidates,
                query,
                top_k=top_k,
                policy_name=self.name,
                score_fn=lambda entry: self._score(entry, query.entity, query.attribute),
                support_entries=self._support_entries(),
                shortlist_limit=max(top_k * self._structured_shortlist_factor, self._structured_min_limit),
                support_limit=self._support_limit,
            )
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def snapshot_size(self) -> int:
        return self._snapshot_size()

    def _score(self, entry: MemoryRecord, entity: str, attribute: str) -> tuple[object, ...]:
        return self._score_strategy(entry, entity, attribute)

    def _candidate_pool(self) -> list[MemoryRecord]:
        return list(self._iter_entries())

    def _attribute_candidates(self, attribute: str) -> list[MemoryRecord]:
        return [entry for entry in self._iter_entries() if entry.attribute == attribute]

    def _direct_candidates(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return [
            entry
            for entry in self._iter_entries()
            if entry.entity == entity and entry.attribute == attribute
        ]

    def _support_entries(self) -> Iterable[MemoryRecord]:
        return self._iter_entries()

    def _store_update(self, update: MemoryRecord) -> None:
        raise NotImplementedError

    def _iter_entries(self) -> Iterable[MemoryRecord]:
        raise NotImplementedError

    def _snapshot_size(self) -> int:
        raise NotImplementedError


class ScoredLogPolicy(ScoredBaselinePolicy):
    def __init__(self, *, name: str, score_strategy: ScoreStrategy, **kwargs: object) -> None:
        super().__init__(name=name, score_strategy=score_strategy, **kwargs)
        self.entries: List[MemoryRecord] = []

    def _store_update(self, update: MemoryRecord) -> None:
        self.entries.append(update)

    def _iter_entries(self) -> Iterable[MemoryRecord]:
        return self.entries

    def _snapshot_size(self) -> int:
        return len(self.entries)


class ScopedLatestPolicy(ScoredBaselinePolicy):
    def __init__(self, *, name: str, score_strategy: ScoreStrategy, **kwargs: object) -> None:
        super().__init__(name=name, score_strategy=score_strategy, **kwargs)
        self.current: dict[tuple[str, str, str], MemoryRecord] = {}

    def _store_update(self, update: MemoryRecord) -> None:
        key = (update.entity, update.attribute, update.scope)
        existing = self.current.get(key)
        if existing is None or update.timestamp >= existing.timestamp:
            self.current[key] = update

    def _iter_entries(self) -> Iterable[MemoryRecord]:
        return self.current.values()

    def _snapshot_size(self) -> int:
        return len(self.current)


class LatestPerKeyPolicy(ScoredBaselinePolicy):
    def __init__(self, *, name: str, score_strategy: ScoreStrategy, **kwargs: object) -> None:
        super().__init__(name=name, score_strategy=score_strategy, **kwargs)
        self.current: dict[MemoryKey, MemoryRecord] = {}

    def _store_update(self, update: MemoryRecord) -> None:
        existing = self.current.get(update.key)
        if existing is None or update.timestamp >= existing.timestamp:
            self.current[update.key] = update

    def _iter_entries(self) -> Iterable[MemoryRecord]:
        return self.current.values()

    def _snapshot_size(self) -> int:
        return len(self.current)

    def _direct_candidates(self, entity: str, attribute: str) -> list[MemoryRecord]:
        entry = self.current.get((entity, attribute))
        return [entry] if entry else []


def append_only_score(entry: MemoryRecord, entity: str, attribute: str) -> tuple[float, ...]:
    return (float(entry.timestamp),)


def strong_retrieval_score(entry: MemoryRecord, entity: str, attribute: str) -> tuple[float, ...]:
    exact_entity = 1.0 if entry.entity == entity else 0.0
    exact_attribute = 1.0 if entry.attribute == attribute else 0.0
    scope_bonus = 1.0 if entry.scope == "default" else 0.5
    return (
        exact_entity + exact_attribute,
        entry.importance,
        float(entry.access_count),
        scope_bonus,
        float(entry.timestamp),
    )


def recency_salience_score(
    *,
    recency_weight: float = 1.0,
    importance_weight: float = 0.75,
) -> Callable[[MemoryRecord, str, str], tuple[float, ...]]:
    def score(entry: MemoryRecord, entity: str, attribute: str) -> tuple[float, ...]:
        entity_match = 1.0 if entry.entity == entity else 0.0
        attribute_match = 1.0 if entry.attribute == attribute else 0.0
        salience = (
            importance_weight * entry.importance
            + 0.25 * entry.confidence
            + 0.1 * entry.access_count
        )
        recency = recency_weight * entry.timestamp
        return (entity_match + attribute_match, salience, recency)

    return score


def exact_match_score(entry: MemoryRecord, entity: str, attribute: str) -> tuple[object, ...]:
    return (float(entry.timestamp), entry.scope)


def summary_only_score(entry: MemoryRecord, entity: str, attribute: str) -> tuple[float, ...]:
    return (float(entry.timestamp),)


class ExactKeyPolicy(ScopedLatestPolicy):
    def __init__(self) -> None:
        super().__init__(
            name="exact_match",
            score_strategy=exact_match_score,
            open_ended_shortlist_factor=8,
            open_ended_min_limit=32,
            structured_shortlist_factor=8,
            structured_min_limit=32,
            support_limit=1,
        )


class SummaryKeyPolicy(LatestPerKeyPolicy):
    def __init__(self) -> None:
        super().__init__(
            name="summary_only",
            score_strategy=summary_only_score,
            open_ended_shortlist_factor=8,
            open_ended_min_limit=32,
            structured_shortlist_factor=8,
            structured_min_limit=32,
            support_limit=1,
        )


class AppendOnlyMemoryPolicy(ScoredLogPolicy):
    def __init__(self) -> None:
        super().__init__(name="append_only", score_strategy=append_only_score)


class StrongRetrievalMemoryPolicy(ScoredLogPolicy):
    """Retrieval-only baseline with stronger ranking over raw history."""

    def __init__(self) -> None:
        super().__init__(name="strong_retrieval", score_strategy=strong_retrieval_score)


class RecencySalienceMemoryPolicy(ScoredLogPolicy):
    """Non-consolidating baseline using recency, confidence, and importance."""

    def __init__(self, recency_weight: float = 1.0, importance_weight: float = 0.75) -> None:
        super().__init__(
            name="recency_salience",
            score_strategy=recency_salience_score(
                recency_weight=recency_weight,
                importance_weight=importance_weight,
            ),
        )
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight


class ExactMatchMemoryPolicy(ExactKeyPolicy):
    """Symbolic baseline that only keeps the latest exact key+scope entry."""


class SummaryOnlyMemoryPolicy(SummaryKeyPolicy):
    """Keep only the current latest fact per key."""
