from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import DefaultDict, Iterable, Sequence

from memory_inference.consolidation.mem0 import Mem0MemoryPolicy
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.consolidation.semantic_utils import query_search_text
from memory_inference.open_ended_eval import expand_with_support_entries, is_open_ended_query
from memory_inference.types import MemoryEntry, MemoryKey, Query, RetrievalResult


class Mem0SupportLinksPolicy(Mem0MemoryPolicy):
    """Explicit alias for the current Mem0 baseline with support expansion enabled."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "mem0_support_links"


class Mem0FeatureAblationPolicy(Mem0MemoryPolicy):
    """Mem0 with optional validity-state bookkeeping and history-aware retrieval."""

    def __init__(
        self,
        *,
        name: str,
        enable_archive_conflict: bool = False,
        history_aware_retrieval: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.enable_archive_conflict = enable_archive_conflict
        self.history_aware_retrieval = history_aware_retrieval
        self.episodic_log: list[MemoryEntry] = []
        self.archive: DefaultDict[MemoryKey, list[MemoryEntry]] = defaultdict(list)
        self.conflict_table: DefaultDict[MemoryKey, list[MemoryEntry]] = defaultdict(list)

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        for update in updates:
            self.episodic_log.append(update)
            prepared = self._prepare_entry(update)
            neighbors = self._similar_memories(prepared)
            self._apply_write(prepared, neighbors)

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if self.history_aware_retrieval and query.query_mode == QueryMode.HISTORY:
            return self._retrieve_history(query, top_k=top_k)
        if self.enable_archive_conflict and query.query_mode in {
            QueryMode.STATE_WITH_PROVENANCE,
            QueryMode.CONFLICT_AWARE,
        }:
            return self._retrieve_augmented_state(query, top_k=top_k)
        return super().retrieve_for_query(query, top_k=top_k)

    def snapshot_size(self) -> int:
        size = super().snapshot_size()
        if self.enable_archive_conflict:
            size += sum(len(entries) for entries in self.archive.values())
            size += sum(len(entries) for entries in self.conflict_table.values())
        return size

    def _apply_update(self, update: MemoryEntry, same_key: Sequence[MemoryEntry]) -> None:
        if self.enable_archive_conflict:
            self._record_conflicts(update, same_key)
            self._archive_entries_if_replaced(update, same_key)
        super()._apply_update(update, same_key)

    def _apply_delete(self, same_key: Sequence[MemoryEntry]) -> None:
        if self.enable_archive_conflict:
            for entry in same_key:
                self._record_archive(entry, status=MemoryStatus.ARCHIVED)
        super()._apply_delete(same_key)

    def _retrieve_history(self, query: Query, *, top_k: int) -> RetrievalResult:
        candidates = [
            entry
            for entry in self.episodic_log
            if self._entity_matches(entry.entity, query.entity)
            and entry.attribute == query.attribute
        ]
        ranked = self._rank_history_candidates(query, candidates)
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.episodic_log,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalResult(
            entries=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_history_dense",
            },
        )

    def _retrieve_augmented_state(self, query: Query, *, top_k: int) -> RetrievalResult:
        active_ranked = self._rank_for_query(query, self.active_store.values())
        candidates = list(active_ranked)
        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            candidates.extend(self._archive_entries(query))
        if query.query_mode == QueryMode.CONFLICT_AWARE:
            candidates = self._conflict_entries(query) + candidates

        deduped: list[MemoryEntry] = []
        seen_ids: set[str] = set()
        for entry in candidates:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            deduped.append(entry)

        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = deduped[:limit]
        support_source = list(self.active_store.values()) + self._archive_entries(query)
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                support_source,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalResult(
            entries=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_state_augmented",
                "query_mode": query.query_mode.name,
            },
        )

    def _rank_history_candidates(
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
            key=lambda entry: self._history_score(entry, query, query_vector),
            reverse=True,
        )

    def _history_score(
        self,
        entry: MemoryEntry,
        query: Query,
        query_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self._entry_vector_for_history(entry))
        entity_bonus = 1.0 if self._entity_matches(entry.entity, query.entity) else 0.0
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        structured_bonus = 0.2 if entry.metadata.get("source_kind") == "structured_fact" else 0.0
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + structured_bonus,
            -float(entry.timestamp),
        )

    def _entry_vector_for_history(self, entry: MemoryEntry) -> tuple[float, ...]:
        vector = self._entry_vectors.get(entry.entry_id)
        if vector is None:
            self._cache_entry(entry)
            vector = self._entry_vectors[entry.entry_id]
        return vector

    def _archive_entries_if_replaced(
        self,
        update: MemoryEntry,
        same_key: Sequence[MemoryEntry],
    ) -> None:
        for entry in same_key:
            if self._normalized_value(entry) == self._normalized_value(update):
                continue
            self._record_archive(entry, status=MemoryStatus.SUPERSEDED)

    def _record_conflicts(
        self,
        update: MemoryEntry,
        same_key: Sequence[MemoryEntry],
    ) -> None:
        key = update.key
        conflicts = [
            entry
            for entry in same_key
            if entry.timestamp == update.timestamp
            and self._normalized_value(entry) != self._normalized_value(update)
        ]
        if not conflicts:
            return
        for entry in conflicts:
            self._append_unique(
                self.conflict_table[key],
                dataclasses.replace(entry, status=MemoryStatus.CONFLICTED),
            )
        self._append_unique(
            self.conflict_table[key],
            dataclasses.replace(update, status=MemoryStatus.CONFLICTED),
        )

    def _record_archive(self, entry: MemoryEntry, *, status: MemoryStatus) -> None:
        self._append_unique(
            self.archive[entry.key],
            dataclasses.replace(entry, status=status),
        )

    def _archive_entries(self, query: Query) -> list[MemoryEntry]:
        archived = [
            entry
            for (entity, attribute), entries in self.archive.items()
            if self._entity_matches(entity, query.entity) and attribute == query.attribute
            for entry in entries
        ]
        archived.sort(key=lambda entry: entry.timestamp, reverse=True)
        return archived

    def _conflict_entries(self, query: Query) -> list[MemoryEntry]:
        conflicts = [
            entry
            for (entity, attribute), entries in self.conflict_table.items()
            if self._entity_matches(entity, query.entity) and attribute == query.attribute
            for entry in entries
        ]
        conflicts.sort(key=lambda entry: entry.timestamp, reverse=True)
        return conflicts

    def _append_unique(self, entries: list[MemoryEntry], candidate: MemoryEntry) -> None:
        if any(existing.entry_id == candidate.entry_id for existing in entries):
            return
        entries.append(candidate)

    def _normalized_value(self, entry: MemoryEntry) -> str:
        return " ".join(entry.value.lower().split())

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity


class Mem0ArchiveConflictPolicy(Mem0FeatureAblationPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            name="mem0_archive_conflict",
            enable_archive_conflict=True,
            history_aware_retrieval=False,
            **kwargs,
        )


class Mem0HistoryAwarePolicy(Mem0FeatureAblationPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            name="mem0_history_aware",
            enable_archive_conflict=False,
            history_aware_retrieval=True,
            **kwargs,
        )


class Mem0AllFeaturesPolicy(Mem0FeatureAblationPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            name="mem0_all_features",
            enable_archive_conflict=True,
            history_aware_retrieval=True,
            **kwargs,
        )
