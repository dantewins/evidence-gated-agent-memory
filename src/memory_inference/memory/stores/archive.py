from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from memory_inference.domain.memory import MemoryKey, MemoryRecord


class ArchiveStore:
    def __init__(self) -> None:
        self.entries: DefaultDict[MemoryKey, list[MemoryRecord]] = defaultdict(list)

    def add(self, key: MemoryKey, entry: MemoryRecord) -> None:
        bucket = self.entries[key]
        if any(existing.entry_id == entry.entry_id for existing in bucket):
            return
        bucket.append(entry)

    def by_query(self, *, entity: str, attribute: str, entity_matches) -> list[MemoryRecord]:
        archived = [
            entry
            for (stored_entity, stored_attribute), entries in self.entries.items()
            if entity_matches(stored_entity, entity) and stored_attribute == attribute
            for entry in entries
        ]
        archived.sort(key=lambda entry: entry.timestamp, reverse=True)
        return archived

    def snapshot_size(self) -> int:
        return sum(len(entries) for entries in self.entries.values())
