from __future__ import annotations

from typing import Dict, Iterable, Tuple

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.open_ended_eval import (
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, RetrievalResult
from memory_inference.types import Query


class ExactMatchMemoryPolicy(BaseMemoryPolicy):
    """Symbolic baseline that only keeps the latest exact key+scope entry."""

    def __init__(self) -> None:
        super().__init__(name="exact_match")
        self.current: Dict[Tuple[str, str, str], MemoryEntry] = {}

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        for update in updates:
            key = (update.entity, update.attribute, update.scope)
            existing = self.current.get(key)
            if existing is None or update.timestamp >= existing.timestamp:
                self.current[key] = update

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        entries = [
            entry
            for (stored_entity, stored_attribute, _scope), entry in self.current.items()
            if stored_entity == entity and stored_attribute == attribute
        ]
        entries.sort(key=lambda item: (item.timestamp, item.scope), reverse=True)
        return RetrievalResult(entries=entries[:top_k], debug={"policy": self.name})

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self.current.values(),
                query,
                score_fn=lambda entry: (float(entry.timestamp),),
                limit=max(top_k * 8, 32),
            )
            return lexical_retrieval(
                candidates,
                query,
                top_k=max(top_k, 8),
                policy_name=self.name,
                secondary_score_fn=lambda entry: (float(entry.timestamp),),
            )
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def snapshot_size(self) -> int:
        return len(self.current)
