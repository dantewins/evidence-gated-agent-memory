from __future__ import annotations

from typing import Iterable, List

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.open_ended_eval import (
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, RetrievalResult
from memory_inference.types import Query


class AppendOnlyMemoryPolicy(BaseMemoryPolicy):
    def __init__(self) -> None:
        super().__init__(name="append_only")
        self.entries: List[MemoryEntry] = []

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        self.entries.extend(updates)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        matches = [
            entry for entry in self.entries if entry.entity == entity and entry.attribute == attribute
        ]
        matches.sort(key=lambda item: item.timestamp, reverse=True)
        return RetrievalResult(entries=matches[:top_k], debug={"policy": self.name})

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self.entries,
                query,
                score_fn=lambda entry: (float(entry.timestamp),),
                limit=max(top_k * 16, 64),
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
        return len(self.entries)
