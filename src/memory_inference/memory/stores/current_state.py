from __future__ import annotations

from typing import Iterable

from memory_inference.domain.memory import MemoryRecord


class CurrentStateStore:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def values(self) -> list[MemoryRecord]:
        return list(self.records.values())

    def same_key(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return [
            entry
            for entry in self.records.values()
            if entry.entity == entity and entry.attribute == attribute
        ]

    def put(self, entry: MemoryRecord) -> None:
        self.records[entry.entry_id] = entry

    def remove(self, entry_id: str) -> None:
        self.records.pop(entry_id, None)

    def remove_all(self, entries: Iterable[MemoryRecord]) -> None:
        for entry in entries:
            self.remove(entry.entry_id)

    def latest_timestamp(self) -> int:
        return max((entry.timestamp for entry in self.records.values()), default=0)

    def snapshot_size(self) -> int:
        return len(self.records)


class ScopedCurrentStateStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str, str], MemoryRecord] = {}

    def values(self) -> list[MemoryRecord]:
        return list(self.records.values())

    def get(self, entity: str, attribute: str, scope: str) -> MemoryRecord | None:
        return self.records.get((entity, attribute, scope))

    def put(self, entry: MemoryRecord) -> None:
        self.records[(entry.entity, entry.attribute, entry.scope)] = entry

    def remove(self, entity: str, attribute: str, scope: str) -> None:
        self.records.pop((entity, attribute, scope), None)

    def remove_entry(self, entry: MemoryRecord) -> None:
        self.remove(entry.entity, entry.attribute, entry.scope)

    def same_key(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return [
            entry
            for (stored_entity, stored_attribute, _scope), entry in self.records.items()
            if stored_entity == entity and stored_attribute == attribute
        ]

    def existing_for_revision(self, entry: MemoryRecord) -> MemoryRecord | None:
        scoped = self.get(entry.entity, entry.attribute, entry.scope)
        if scoped is not None:
            return scoped
        same_key = self.same_key(entry.entity, entry.attribute)
        if not same_key:
            return None
        return max(same_key, key=lambda candidate: candidate.timestamp)

    def by_query(self, *, entity: str, attribute: str, entity_matches) -> list[MemoryRecord]:
        current = [
            entry
            for (stored_entity, stored_attribute, _scope), entry in self.records.items()
            if entity_matches(stored_entity, entity) and stored_attribute == attribute
        ]
        current.sort(key=lambda entry: entry.timestamp, reverse=True)
        return current

    def snapshot_size(self) -> int:
        return len(self.records)
