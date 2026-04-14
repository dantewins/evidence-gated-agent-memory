from __future__ import annotations

from typing import Iterable, List

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.semantic_utils import (
    DenseEncoder,
    TransformerDenseEncoder,
    entry_search_text,
    query_search_text,
)
from memory_inference.open_ended_eval import expand_with_support_entries, is_open_ended_query
from memory_inference.types import MemoryEntry, Query, RetrievalResult


class DenseRetrievalMemoryPolicy(BaseMemoryPolicy):
    """Semantic retrieval baseline over the full episodic log."""

    def __init__(self, *, encoder: DenseEncoder | None = None) -> None:
        super().__init__(name="dense_retrieval")
        self.entries: List[MemoryEntry] = []
        self.encoder = encoder if encoder is not None else TransformerDenseEncoder()
        self._entry_vectors: dict[str, tuple[float, ...]] = {}

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        new_entries = list(updates)
        if not new_entries:
            return
        self.entries.extend(new_entries)
        entry_vectors = self.encoder.encode_passages([entry_search_text(entry) for entry in new_entries])
        for update, vector in zip(new_entries, entry_vectors):
            self._entry_vectors[update.entry_id] = vector

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        query = Query(
            query_id="dense-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            answer="",
            timestamp=max((entry.timestamp for entry in self.entries), default=0),
            session_id="dense-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        candidates = self._candidate_pool(query)
        ranked = self._rank(query, candidates)
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.entries,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalResult(
            entries=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "dense_semantic",
            },
        )

    def snapshot_size(self) -> int:
        return len(self.entries)

    def _candidate_pool(self, query: Query) -> list[MemoryEntry]:
        return list(self.entries)

    def _rank(self, query: Query, candidates: Iterable[MemoryEntry]) -> list[MemoryEntry]:
        query_vector = self.encoder.encode_query(query_search_text(query))
        unique: dict[str, MemoryEntry] = {}
        for entry in candidates:
            unique.setdefault(entry.entry_id, entry)
        return sorted(
            unique.values(),
            key=lambda entry: self._score(entry, query, query_vector),
            reverse=True,
        )

    def _score(
        self,
        entry: MemoryEntry,
        query: Query,
        query_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self._entry_vectors[entry.entry_id])
        structured_bonus = 1.0 if entry.metadata.get("source_kind") == "structured_fact" else 0.0
        time_bias = float(entry.timestamp)
        return (
            dense_similarity,
            structured_bonus,
            time_bias,
        )

    def _has_structured_fact(self, entries: Iterable[MemoryEntry]) -> bool:
        return any(entry.metadata.get("source_kind") == "structured_fact" for entry in entries)
