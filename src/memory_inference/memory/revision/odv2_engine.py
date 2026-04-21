from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import DefaultDict, Iterable, Set

from memory_inference.domain.enums import MemoryStatus, RevisionOp
from memory_inference.domain.memory import MemoryKey, MemoryRecord
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.stores import ArchiveStore, ConflictStore, ScopedCurrentStateStore


class ODV2RevisionEngine:
    def __init__(
        self,
        *,
        consolidator: BaseConsolidator,
        importance_threshold: float = 0.1,
    ) -> None:
        self.consolidator = consolidator
        self.importance_threshold = importance_threshold

    def consolidate(
        self,
        pending: list[MemoryRecord],
        *,
        state_store: ScopedCurrentStateStore,
        archive_store: ArchiveStore,
        conflict_store: ConflictStore,
        prior_values: DefaultDict[MemoryKey, Set[str]],
    ) -> None:
        if not pending:
            return
        for entry in pending:
            self.process_entry(
                entry,
                state_store=state_store,
                archive_store=archive_store,
                conflict_store=conflict_store,
                prior_values=prior_values,
            )
        pending.clear()
        self.apply_importance_threshold(
            state_store=state_store,
            archive_store=archive_store,
        )

    def process_entry(
        self,
        entry: MemoryRecord,
        *,
        state_store: ScopedCurrentStateStore,
        archive_store: ArchiveStore,
        conflict_store: ConflictStore,
        prior_values: DefaultDict[MemoryKey, Set[str]],
    ) -> None:
        existing = state_store.existing_for_revision(entry)
        prior = prior_values[entry.key] - {entry.value}
        op = self.consolidator.classify_revision(entry, existing, prior_values=prior)

        if op == RevisionOp.ADD:
            state_store.put(dataclasses.replace(entry, status=MemoryStatus.ACTIVE))
            return

        if op == RevisionOp.REINFORCE:
            reinforced = dataclasses.replace(
                entry,
                status=MemoryStatus.REINFORCED,
                importance=min(1.0, (existing.importance if existing else 1.0) + 0.1),
            )
            state_store.put(reinforced)
            return

        if op in {RevisionOp.REVISE, RevisionOp.REVERT}:
            if existing is not None:
                archive_store.add(
                    entry.key,
                    dataclasses.replace(
                        existing,
                        status=MemoryStatus.SUPERSEDED,
                        importance=max(0.0, existing.importance - 0.2),
                    ),
                )
                state_store.remove_entry(existing)
            state_store.put(
                dataclasses.replace(
                    entry,
                    status=MemoryStatus.ACTIVE,
                    supersedes_id=existing.entry_id if existing else None,
                )
            )
            return

        if op == RevisionOp.SPLIT_SCOPE:
            state_store.put(dataclasses.replace(entry, status=MemoryStatus.ACTIVE))
            return

        if op == RevisionOp.CONFLICT_UNRESOLVED:
            conflicted_new = dataclasses.replace(entry, status=MemoryStatus.CONFLICTED)
            if existing is not None:
                conflict_store.add(
                    entry.key,
                    dataclasses.replace(existing, status=MemoryStatus.CONFLICTED),
                )
                state_store.remove_entry(existing)
            conflict_store.add(entry.key, conflicted_new)
            return

        if op == RevisionOp.LOW_CONFIDENCE:
            archive_store.add(
                entry.key,
                dataclasses.replace(entry, status=MemoryStatus.ARCHIVED),
            )
            return

    def apply_importance_threshold(
        self,
        *,
        state_store: ScopedCurrentStateStore,
        archive_store: ArchiveStore,
    ) -> None:
        to_demote = [
            entry
            for entry in state_store.values()
            if entry.importance < self.importance_threshold
        ]
        for entry in to_demote:
            state_store.remove_entry(entry)
            archive_store.add(
                entry.key,
                dataclasses.replace(entry, status=MemoryStatus.ARCHIVED),
            )
