from __future__ import annotations

from typing import Iterable, List, Tuple

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.open_ended_eval import (
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, RetrievalResult
from memory_inference.types import Query


class StrongRetrievalMemoryPolicy(BaseMemoryPolicy):
    """Retrieval-only baseline with stronger ranking over raw history."""

    def __init__(self) -> None:
        super().__init__(name="strong_retrieval")
        self.entries: List[MemoryEntry] = []

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        self.entries.extend(updates)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        ranked = sorted(
            self.entries,
            key=lambda entry: self._score(entry, entity, attribute),
            reverse=True,
        )
        return RetrievalResult(entries=ranked[:top_k], debug={"policy": self.name})

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self.entries,
                query,
                score_fn=lambda entry: self._score(entry, query.entity, query.attribute),
                limit=max(top_k * 16, 64),
            )
            return lexical_retrieval(
                candidates,
                query,
                top_k=max(top_k, 8),
                policy_name=self.name,
                secondary_score_fn=lambda entry: self._score(entry, query.entity, query.attribute),
            )
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def snapshot_size(self) -> int:
        return len(self.entries)

    def _score(self, entry: MemoryEntry, entity: str, attribute: str) -> Tuple[float, ...]:
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
