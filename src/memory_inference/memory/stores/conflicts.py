from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from memory_inference.domain.memory import MemoryKey, MemoryRecord


class ConflictStore:
    def __init__(self) -> None:
        self.entries: DefaultDict[MemoryKey, list[MemoryRecord]] = defaultdict(list)

    def add(self, key: MemoryKey, entry: MemoryRecord) -> None:
        bucket = self.entries[key]
        if any(existing.entry_id == entry.entry_id for existing in bucket):
            return
        bucket.append(entry)

    def clear(self, key: MemoryKey | None) -> None:
        if key is None:
            return
        self.entries.pop(key, None)

    def by_query(self, *, entity: str, attribute: str, entity_matches) -> list[MemoryRecord]:
        conflicts = [
            entry
            for (stored_entity, stored_attribute), entries in self.entries.items()
            if entity_matches(stored_entity, entity) and stored_attribute == attribute
            for entry in entries
        ]
        conflicts.sort(key=lambda entry: entry.timestamp, reverse=True)
        return conflicts

    def snapshot_size(self) -> int:
        return sum(len(entries) for entries in self.entries.values())
