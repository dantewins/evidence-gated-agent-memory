from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.semantic_utils import (
    DenseEncoder,
    TransformerDenseEncoder,
    entry_search_text,
    normalize_text,
    query_search_text,
)
from memory_inference.open_ended_eval import expand_with_support_entries, is_open_ended_query
from memory_inference.types import MemoryEntry, Query, RetrievalResult

_CONVERSATIONAL_ATTRIBUTES = frozenset({"dialogue", "event"})
_DELETE_MARKERS = frozenset({"delete", "deleted", "none", "n/a", "removed", "unknown"})


class Mem0MemoryPolicy(BaseMemoryPolicy):
    """Mem0-style flat active store with dense write-time updates."""

    def __init__(
        self,
        *,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
    ) -> None:
        super().__init__(name="mem0")
        self.encoder = encoder if encoder is not None else TransformerDenseEncoder()
        self.write_top_k = write_top_k
        self.active_store: dict[str, MemoryEntry] = {}
        self._entry_vectors: dict[str, tuple[float, ...]] = {}

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        for update in updates:
            prepared = self._prepare_entry(update)
            neighbors = self._similar_memories(prepared)
            self._apply_write(prepared, neighbors)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        query = Query(
            query_id="mem0-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            answer="",
            timestamp=max((entry.timestamp for entry in self.active_store.values()), default=0),
            session_id="mem0-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        ranked = self._rank_for_query(query, self.active_store.values())
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.active_store.values(),
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalResult(
            entries=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_active_dense",
            },
        )

    def snapshot_size(self) -> int:
        return len(self.active_store)

    def _apply_write(self, update: MemoryEntry, neighbors: Sequence[MemoryEntry]) -> None:
        same_key = [
            entry
            for entry in neighbors
            if entry.entity == update.entity and entry.attribute == update.attribute
        ]
        duplicate = next(
            (
                entry
                for entry in same_key
                if normalize_text(entry.value) == normalize_text(update.value)
            ),
            None,
        )
        if duplicate is not None:
            self._apply_noop(duplicate, update)
            return

        if self._is_delete_update(update):
            self._apply_delete(same_key)
            return

        if self._is_state_memory(update) and same_key:
            self._apply_update(update, same_key)
            return

        self._apply_add(update)

    def _apply_add(self, update: MemoryEntry) -> None:
        stored = dataclasses.replace(update, access_count=max(update.access_count, 1))
        self.active_store[stored.entry_id] = stored
        self._cache_entry(stored)

    def _apply_update(self, update: MemoryEntry, same_key: Sequence[MemoryEntry]) -> None:
        target = same_key[0]
        merged = dataclasses.replace(
            update,
            entry_id=target.entry_id,
            access_count=max(target.access_count, 1) + 1,
            importance=max(target.importance, update.importance),
            confidence=max(target.confidence, update.confidence),
            metadata={
                **target.metadata,
                **update.metadata,
            },
        )
        self.active_store[target.entry_id] = merged
        self._cache_entry(merged)

        for stale in same_key[1:]:
            if normalize_text(stale.value) != normalize_text(merged.value):
                self._drop_entry(stale.entry_id)

    def _apply_delete(self, same_key: Sequence[MemoryEntry]) -> None:
        for entry in same_key:
            self._drop_entry(entry.entry_id)

    def _apply_noop(self, existing: MemoryEntry, update: MemoryEntry) -> None:
        richer = update if len(entry_search_text(update)) >= len(entry_search_text(existing)) else existing
        merged = dataclasses.replace(
            richer,
            entry_id=existing.entry_id,
            timestamp=max(existing.timestamp, update.timestamp),
            access_count=max(existing.access_count, 1) + 1,
            importance=max(existing.importance, update.importance),
            confidence=max(existing.confidence, update.confidence),
            metadata={
                **existing.metadata,
                **update.metadata,
            },
        )
        self.active_store[existing.entry_id] = merged
        self._cache_entry(merged)

    def _rank_for_query(
        self,
        query: Query,
        candidates: Iterable[MemoryEntry],
    ) -> list[MemoryEntry]:
        query_vector = self.encoder.encode_query(query_search_text(query))
        unique: dict[str, MemoryEntry] = {}
        for entry in candidates:
            unique.setdefault(entry.entry_id, entry)
        return sorted(
            unique.values(),
            key=lambda entry: self._query_score(entry, query, query_vector),
            reverse=True,
        )

    def _query_score(
        self,
        entry: MemoryEntry,
        query: Query,
        query_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self._entry_vectors[entry.entry_id])
        entity_bonus = 1.0 if self._entity_matches(entry.entity, query.entity) else 0.0
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        structured_bonus = 0.2 if entry.metadata.get("source_kind") == "structured_fact" else 0.0
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + structured_bonus,
            float(entry.timestamp),
        )

    def _similar_memories(self, update: MemoryEntry) -> list[MemoryEntry]:
        if not self.active_store:
            return []
        update_vector = self.encoder.encode_query(entry_search_text(update))
        return sorted(
            self.active_store.values(),
            key=lambda entry: self._write_score(entry, update, update_vector),
            reverse=True,
        )[: self.write_top_k]

    def _write_score(
        self,
        entry: MemoryEntry,
        update: MemoryEntry,
        update_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(update_vector, self._entry_vectors[entry.entry_id])
        entity_bonus = 1.0 if entry.entity == update.entity else 0.0
        attribute_bonus = 1.0 if entry.attribute == update.attribute else 0.0
        state_bonus = 0.2 if self._is_state_memory(entry) == self._is_state_memory(update) else 0.0
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + state_bonus,
            float(entry.timestamp),
        )

    def _cache_entry(self, entry: MemoryEntry) -> None:
        self._entry_vectors[entry.entry_id] = self.encoder.encode_passage(entry_search_text(entry))

    def _drop_entry(self, entry_id: str) -> None:
        self.active_store.pop(entry_id, None)
        self._entry_vectors.pop(entry_id, None)

    def _prepare_entry(self, entry: MemoryEntry) -> MemoryEntry:
        memory_kind = entry.metadata.get("memory_kind")
        if memory_kind:
            return entry
        return dataclasses.replace(
            entry,
            metadata={
                **entry.metadata,
                "memory_kind": "state" if self._is_state_memory(entry) else "event",
            },
        )

    def _is_state_memory(self, entry: MemoryEntry) -> bool:
        if entry.metadata.get("memory_kind") == "state":
            return True
        if entry.metadata.get("source_kind") == "structured_fact":
            return True
        return entry.attribute not in _CONVERSATIONAL_ATTRIBUTES

    def _is_delete_update(self, entry: MemoryEntry) -> bool:
        normalized_value = normalize_text(entry.value)
        if normalized_value in _DELETE_MARKERS:
            return True
        support_text = normalize_text(entry.metadata.get("support_text", ""))
        return any(marker in support_text.split() for marker in _DELETE_MARKERS)

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    def _has_structured_fact(self, entries: Iterable[MemoryEntry]) -> bool:
        return any(entry.metadata.get("source_kind") == "structured_fact" for entry in entries)
