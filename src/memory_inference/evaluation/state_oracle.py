from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from memory_inference.domain.enums import MemoryStatus
from memory_inference.domain.memory import MemoryRecord


class StateOracle:
    """Evaluates the validity state implied by a collection of memory records."""

    def __init__(self, entries: Sequence[MemoryRecord]) -> None:
        self._entries = list(entries)

    def active_value(self, entity: str, attribute: str) -> Optional[MemoryRecord]:
        """Return the most-recent active entry for an entity/attribute pair."""
        candidates = [
            entry
            for entry in self._entries
            if entry.entity == entity
            and entry.attribute == attribute
            and entry.status in (MemoryStatus.ACTIVE, MemoryStatus.REINFORCED)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry.timestamp)

    def superseded_chain(self, entity: str, attribute: str) -> List[MemoryRecord]:
        """Return all superseded records for an entity/attribute pair."""
        return [
            entry
            for entry in self._entries
            if entry.entity == entity
            and entry.attribute == attribute
            and entry.status == MemoryStatus.SUPERSEDED
        ]

    def unresolved_conflicts(self, entity: str, attribute: str) -> List[MemoryRecord]:
        """Return all conflicted records for an entity/attribute pair."""
        return [
            entry
            for entry in self._entries
            if entry.entity == entity
            and entry.attribute == attribute
            and entry.status == MemoryStatus.CONFLICTED
        ]

    def scope_splits(self, entity: str, attribute: str) -> Dict[str, List[MemoryRecord]]:
        """Return active records grouped by scope for an entity/attribute pair."""
        active_statuses = {MemoryStatus.ACTIVE, MemoryStatus.REINFORCED}
        groups: Dict[str, List[MemoryRecord]] = defaultdict(list)
        for entry in self._entries:
            if entry.entity == entity and entry.attribute == attribute and entry.status in active_statuses:
                groups[entry.scope].append(entry)
        return dict(groups)

    def current_state_match(self, entity: str, attribute: str, gold_value: str) -> bool:
        """Check whether the active record's value matches the provided gold value."""
        active = self.active_value(entity, attribute)
        if active is None:
            return False
        return active.value == gold_value
