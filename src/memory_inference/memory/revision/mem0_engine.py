from __future__ import annotations

import dataclasses
from typing import Sequence

from memory_inference.domain.enums import MemoryStatus
from memory_inference.memory.retrieval.semantic import entry_search_text, normalize_text
from memory_inference.domain.memory import MemoryKey, MemoryRecord
from memory_inference.memory.stores.archive import ArchiveStore
from memory_inference.memory.stores.conflicts import ConflictStore
from memory_inference.memory.stores.current_state import CurrentStateStore

_CONVERSATIONAL_ATTRIBUTES = frozenset({"dialogue", "event"})
_DELETE_MARKERS = frozenset({"delete", "deleted", "none", "n/a", "removed", "unknown"})


class Mem0RevisionEngine:
    def prepare_entry(self, entry: MemoryRecord) -> MemoryRecord:
        if entry.memory_kind:
            return entry
        return dataclasses.replace(
            entry,
            memory_kind="state" if self.is_state_memory(entry) else "event",
        )

    def apply(
        self,
        update: MemoryRecord,
        *,
        state_store: CurrentStateStore,
        ranker,
        archive_store: ArchiveStore | None = None,
        conflict_store: ConflictStore | None = None,
    ) -> None:
        same_key = state_store.same_key(update.entity, update.attribute)
        duplicate = next(
            (
                entry
                for entry in same_key
                if normalize_text(entry.value) == normalize_text(update.value)
            ),
            None,
        )
        if duplicate is not None:
            merged = self._merge_noop(duplicate, update)
            state_store.put(merged)
            ranker.index(merged)
            self._resolve_conflicts_for_key(duplicate.key, conflict_store)
            return

        if self.is_delete_update(update):
            if archive_store is not None:
                for entry in same_key:
                    archive_store.add(entry.key, self.snapshot_entry(entry, status=MemoryStatus.ARCHIVED))
            self._resolve_conflicts_for_key(same_key[0].key if same_key else None, conflict_store)
            for entry in same_key:
                state_store.remove(entry.entry_id)
                ranker.remove(entry.entry_id)
            return

        if self.is_state_memory(update) and same_key:
            if archive_store is not None and conflict_store is not None:
                self._resolve_conflicts_if_superseded(update, same_key, conflict_store)
                self._record_conflicts(update, same_key, conflict_store)
                self._archive_entries_if_replaced(update, same_key, archive_store)

            target = same_key[0]
            merged = self._merge_update(target, update)
            state_store.put(merged)
            ranker.index(merged)
            for stale in same_key[1:]:
                if normalize_text(stale.value) != normalize_text(merged.value):
                    state_store.remove(stale.entry_id)
                    ranker.remove(stale.entry_id)
            return

        stored = dataclasses.replace(update, access_count=max(update.access_count, 1))
        state_store.put(stored)
        ranker.index(stored)

    def is_state_memory(self, entry: MemoryRecord) -> bool:
        if entry.memory_kind == "state":
            return True
        if entry.source_kind == "structured_fact":
            return True
        return entry.attribute not in _CONVERSATIONAL_ATTRIBUTES

    def is_delete_update(self, entry: MemoryRecord) -> bool:
        normalized_value = normalize_text(entry.value)
        if normalized_value in _DELETE_MARKERS:
            return True
        support_text = normalize_text(entry.support_text)
        return any(marker in support_text.split() for marker in _DELETE_MARKERS)

    def snapshot_entry(self, entry: MemoryRecord, *, status: MemoryStatus) -> MemoryRecord:
        suffix = self.normalized_value(entry).replace(" ", "_")[:24] or "value"
        snapshot_id = f"{entry.entry_id}::{status.name.lower()}::{entry.timestamp}::{suffix}"
        return dataclasses.replace(entry, record_id=snapshot_id, status=status)

    def normalized_value(self, entry: MemoryRecord) -> str:
        return " ".join(entry.value.lower().split())

    def _merge_update(self, target: MemoryRecord, update: MemoryRecord) -> MemoryRecord:
        return dataclasses.replace(
            update,
            record_id=target.record_id,
            access_count=max(target.access_count, 1) + 1,
            importance=max(target.importance, update.importance),
            confidence=max(target.confidence, update.confidence),
            metadata={
                **target.metadata,
                **update.metadata,
            },
            source_kind=update.source_kind or target.source_kind,
            source_attribute=update.source_attribute or target.source_attribute,
            memory_kind=update.memory_kind or target.memory_kind,
            source_entry_id=update.source_entry_id or target.source_entry_id,
            support_text=update.support_text or target.support_text,
        )

    def _merge_noop(self, existing: MemoryRecord, update: MemoryRecord) -> MemoryRecord:
        richer = update if len(entry_search_text(update)) >= len(entry_search_text(existing)) else existing
        return dataclasses.replace(
            richer,
            record_id=existing.record_id,
            timestamp=max(existing.timestamp, update.timestamp),
            access_count=max(existing.access_count, 1) + 1,
            importance=max(existing.importance, update.importance),
            confidence=max(existing.confidence, update.confidence),
            metadata={
                **existing.metadata,
                **update.metadata,
            },
            source_kind=update.source_kind or existing.source_kind,
            source_attribute=update.source_attribute or existing.source_attribute,
            memory_kind=update.memory_kind or existing.memory_kind,
            source_entry_id=update.source_entry_id or existing.source_entry_id,
            support_text=update.support_text or existing.support_text,
        )

    def _archive_entries_if_replaced(
        self,
        update: MemoryRecord,
        same_key: Sequence[MemoryRecord],
        archive_store: ArchiveStore,
    ) -> None:
        for entry in same_key:
            if self.normalized_value(entry) == self.normalized_value(update):
                continue
            archive_store.add(entry.key, self.snapshot_entry(entry, status=MemoryStatus.SUPERSEDED))

    def _record_conflicts(
        self,
        update: MemoryRecord,
        same_key: Sequence[MemoryRecord],
        conflict_store: ConflictStore,
    ) -> None:
        key = update.key
        conflicts = [
            entry
            for entry in same_key
            if entry.timestamp >= update.timestamp
            and self.normalized_value(entry) != self.normalized_value(update)
        ]
        if not conflicts:
            return
        for entry in conflicts:
            conflict_store.add(key, self.snapshot_entry(entry, status=MemoryStatus.CONFLICTED))
        conflict_store.add(key, self.snapshot_entry(update, status=MemoryStatus.CONFLICTED))

    def _resolve_conflicts_if_superseded(
        self,
        update: MemoryRecord,
        same_key: Sequence[MemoryRecord],
        conflict_store: ConflictStore,
    ) -> None:
        if any(entry.timestamp < update.timestamp for entry in same_key):
            self._resolve_conflicts_for_key(update.key, conflict_store)

    def _resolve_conflicts_for_key(
        self,
        key: MemoryKey | None,
        conflict_store: ConflictStore | None,
    ) -> None:
        if conflict_store is None:
            return
        conflict_store.clear(key)
