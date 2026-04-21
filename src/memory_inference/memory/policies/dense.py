from __future__ import annotations

from typing import Iterable, List

from memory_inference.memory.retrieval.semantic import (
    DenseEncoder,
    TransformerDenseEncoder,
    entry_search_text,
    query_search_text,
)
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.retrieval import expand_with_support_entries, is_open_ended_query


class DenseRetrievalMemoryPolicy(BaseMemoryPolicy):
    """Semantic retrieval baseline over the full episodic log."""

    def __init__(self, *, encoder: DenseEncoder | None = None) -> None:
        super().__init__(name="dense_retrieval")
        self.records: List[MemoryRecord] = []
        self.encoder = encoder if encoder is not None else TransformerDenseEncoder()
        self._record_vectors: dict[str, tuple[float, ...]] = {}

    @property
    def entries(self) -> list[MemoryRecord]:
        return self.records

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        new_records = list(updates)
        if not new_records:
            return
        self.records.extend(new_records)
        record_vectors = self.encoder.encode_passages([entry_search_text(record) for record in new_records])
        for update, vector in zip(new_records, record_vectors):
            self._record_vectors[update.record_id] = vector

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id="dense-retrieve",
            context_id="dense-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=max((record.timestamp for record in self.records), default=0),
            session_id="dense-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        candidates = self._candidate_pool(query)
        ranked = self._rank(query, candidates)
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_records = ranked[:limit]
        if self._has_structured_fact(top_records):
            top_records = expand_with_support_entries(
                top_records,
                self.records,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalBundle(
            records=top_records,
            debug={
                "policy": self.name,
                "retrieval_mode": "dense_semantic",
            },
        )

    def snapshot_size(self) -> int:
        return len(self.records)

    def _candidate_pool(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return list(self.records)

    def _rank(self, query: RuntimeQuery, candidates: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        query_vector = self.encoder.encode_query(query_search_text(query))
        unique: dict[str, MemoryRecord] = {}
        for record in candidates:
            unique.setdefault(record.record_id, record)
        return sorted(
            unique.values(),
            key=lambda record: self._score(record, query, query_vector),
            reverse=True,
        )

    def _score(
        self,
        record: MemoryRecord,
        query: RuntimeQuery,
        query_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self._record_vectors[record.record_id])
        structured_bonus = 1.0 if record.source_kind == "structured_fact" else 0.0
        return (
            dense_similarity,
            structured_bonus,
            float(record.timestamp),
        )

    def _has_structured_fact(self, records: Iterable[MemoryRecord]) -> bool:
        return any(record.source_kind == "structured_fact" for record in records)
