from __future__ import annotations

from typing import Callable, Iterable

from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.memory.retrieval.semantic import (
    DenseEncoder,
    TransformerDenseEncoder,
    entry_search_text,
    query_search_text,
)
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery


class DenseRanker:
    def __init__(
        self,
        *,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
    ) -> None:
        self.encoder = encoder if encoder is not None else TransformerDenseEncoder()
        self.write_top_k = write_top_k
        self.entry_vectors: dict[str, tuple[float, ...]] = {}

    def index(self, entry: MemoryRecord) -> None:
        self.entry_vectors[entry.entry_id] = self.encoder.encode_passage(entry_search_text(entry))

    def remove(self, entry_id: str) -> None:
        self.entry_vectors.pop(entry_id, None)

    def rank_query(
        self,
        query: RuntimeQuery,
        candidates: Iterable[MemoryRecord],
        *,
        entity_matches: Callable[[str, str], bool],
        history: bool = False,
    ) -> list[MemoryRecord]:
        query_vector = self.encoder.encode_query(query_search_text(query))
        unique: dict[str, MemoryRecord] = {}
        for entry in candidates:
            unique.setdefault(entry.entry_id, entry)
        return sorted(
            unique.values(),
            key=lambda entry: self._query_score(
                entry,
                query=query,
                query_vector=query_vector,
                entity_matches=entity_matches,
                history=history,
            ),
            reverse=True,
        )

    def nearest_neighbors(
        self,
        update: MemoryRecord,
        candidates: Iterable[MemoryRecord],
        *,
        is_state_memory: Callable[[MemoryRecord], bool],
    ) -> list[MemoryRecord]:
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        update_vector = self.encoder.encode_query(entry_search_text(update))
        return sorted(
            candidate_list,
            key=lambda entry: self._write_score(
                entry,
                update=update,
                update_vector=update_vector,
                is_state_memory=is_state_memory,
            ),
            reverse=True,
        )[: self.write_top_k]

    def entry_vector(self, entry: MemoryRecord) -> tuple[float, ...]:
        vector = self.entry_vectors.get(entry.entry_id)
        if vector is None:
            self.index(entry)
            vector = self.entry_vectors[entry.entry_id]
        return vector

    def _query_score(
        self,
        entry: MemoryRecord,
        *,
        query: RuntimeQuery,
        query_vector: tuple[float, ...],
        entity_matches: Callable[[str, str], bool],
        history: bool,
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self.entry_vector(entry))
        entity_bonus = 1.0 if entity_matches(entry.entity, query.entity) else 0.0
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        structured_bonus = 0.2 if entry.source_kind == "structured_fact" else 0.0
        time_bias = -float(entry.timestamp) if history else float(entry.timestamp)
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + structured_bonus,
            time_bias,
        )

    def _write_score(
        self,
        entry: MemoryRecord,
        *,
        update: MemoryRecord,
        update_vector: tuple[float, ...],
        is_state_memory: Callable[[MemoryRecord], bool],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(update_vector, self.entry_vector(entry))
        entity_bonus = 1.0 if entry.entity == update.entity else 0.0
        attribute_bonus = 1.0 if entry.attribute == update.attribute else 0.0
        state_bonus = 0.2 if is_state_memory(entry) == is_state_memory(update) else 0.0
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + state_bonus,
            float(entry.timestamp),
        )


class ODV2DenseBackboneRanker:
    name = "dense"

    def __init__(self, *, encoder: DenseEncoder | None = None) -> None:
        self.encoder = encoder if encoder is not None else TransformerDenseEncoder()
        self._entry_vectors: dict[str, tuple[float, ...]] = {}

    def index_entries(self, entries: Iterable[MemoryRecord]) -> None:
        entry_list = list(entries)
        if not entry_list:
            return
        vectors = self.encoder.encode_passages([entry_search_text(entry) for entry in entry_list])
        for entry, vector in zip(entry_list, vectors):
            self._entry_vectors[entry.entry_id] = vector

    def rank(
        self,
        query: RuntimeQuery,
        candidates: Iterable[MemoryRecord],
        *,
        score_fn,
        limit: int,
    ) -> list[MemoryRecord]:
        query_vector = self.encoder.encode_query(query_search_text(query))
        unique: dict[str, MemoryRecord] = {}
        for entry in candidates:
            unique.setdefault(entry.entry_id, entry)
        ranked = sorted(
            unique.values(),
            key=lambda entry: score_fn(entry, query_vector),
            reverse=True,
        )
        return ranked[:limit]

    def backbone_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_context, self._entry_vector(entry))
        entity_bonus = self._entity_signal(entry, query)
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        return (
            entity_bonus,
            attribute_bonus,
            dense_similarity,
            entry.importance,
            entry.confidence,
            self._time_bias(entry, query),
        )

    def structured_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context: tuple[float, ...],
        *,
        memory_kind: str,
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_context, self._entry_vector(entry))
        entity_bonus = self._entity_signal(entry, query)
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.5,
            MemoryStatus.SUPERSEDED: 0.3,
            MemoryStatus.ARCHIVED: 0.2,
        }.get(entry.status, 0.0)
        memory_kind_bonus = 0.4 if memory_kind == "state" else 0.0
        support_bonus = 0.2 if entry.source_entry_id else 0.0
        return (
            entity_bonus,
            attribute_bonus,
            status_bonus,
            memory_kind_bonus,
            support_bonus,
            dense_similarity,
            entry.importance,
            entry.confidence,
            self._time_bias(entry, query),
        )

    def evidence_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context: tuple[float, ...],
        *,
        anchor_source_ids: set[str],
        anchor_scopes: set[str],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_context, self._entry_vector(entry))
        entity_bonus = self._entity_signal(entry, query)
        anchor_bonus = 1.5 if entry.entry_id in anchor_source_ids else 0.0
        scope_bonus = 1.0 if entry.scope in anchor_scopes else 0.0
        attribute_bonus = (
            1.0 if entry.attribute == query.attribute
            else 0.8 if entry.attribute in {"dialogue", "event"}
            else 0.0
        )
        return (
            entity_bonus,
            attribute_bonus,
            dense_similarity,
            anchor_bonus,
            scope_bonus,
            entry.importance,
            entry.confidence,
            self._time_bias(entry, query),
        )

    def open_ended_mode(self) -> str:
        return "dense_open_ended"

    def backbone_mode(self) -> str:
        return "dense_backbone"

    def _entry_vector(self, entry: MemoryRecord) -> tuple[float, ...]:
        vector = self._entry_vectors.get(entry.entry_id)
        if vector is None:
            vector = self.encoder.encode_passage(entry_search_text(entry))
            self._entry_vectors[entry.entry_id] = vector
        return vector

    @staticmethod
    def _entity_signal(entry: MemoryRecord, query: RuntimeQuery) -> float:
        if query.entity in {"conversation", "all"} or entry.entity == query.entity:
            return 1.0
        if entry.entity and entry.entity.lower() in query.question.lower():
            return 0.9
        return 0.0

    @staticmethod
    def _time_bias(entry: MemoryRecord, query: RuntimeQuery) -> float:
        if query.query_mode == QueryMode.HISTORY:
            return -float(entry.timestamp)
        return float(entry.timestamp)
