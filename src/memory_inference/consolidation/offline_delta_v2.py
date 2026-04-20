from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode, RevisionOp
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.open_ended_eval import (
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, MemoryKey, Query, RetrievalResult


class OfflineDeltaConsolidationPolicyV2(BaseMemoryPolicy):
    """Validity-aware Offline Delta Consolidation (ODC v2).

    Explicit stores:
    - episodic_log: append-only, every entry ever seen
    - current_state: one active entry per (entity, attribute, scope) triple
    - conflict_table: unresolved same-time contradictions, keyed by MemoryKey
    - archive: superseded / low-confidence entries, keyed by MemoryKey
    - _pending: entries awaiting offline consolidation

    Online path  (ingest)             — no LLM calls, O(1).
    Offline path (maybe_consolidate)  — LLM-assisted, fires at session boundary.
    """

    def __init__(
        self,
        consolidator: BaseConsolidator,
        importance_threshold: float = 0.1,
        support_history_limit: int = 3,
        maintenance_frequency: int = 1,
    ) -> None:
        super().__init__(name="offline_delta_v2")
        self.consolidator = consolidator
        self.importance_threshold = importance_threshold
        self.support_history_limit = support_history_limit
        self.maintenance_frequency = max(1, maintenance_frequency)

        self.episodic_log: List[MemoryEntry] = []
        # current_state: keyed by (entity, attribute, scope) for scope-split support
        self.current_state: Dict[tuple, MemoryEntry] = {}
        self.conflict_table: DefaultDict[MemoryKey, List[MemoryEntry]] = defaultdict(list)
        self.archive: DefaultDict[MemoryKey, List[MemoryEntry]] = defaultdict(list)
        # prior_values_seen: set of values ever seen per MemoryKey, for REVERT detection
        self._prior_values: DefaultDict[MemoryKey, Set[str]] = defaultdict(set)
        self._pending: List[MemoryEntry] = []

    # ------------------------------------------------------------------ #
    # Online path                                                          #
    # ------------------------------------------------------------------ #

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        """Append to episodic log and pending buffer. No consolidation yet."""
        for entry in updates:
            self.episodic_log.append(entry)
            self._prior_values[entry.key].add(entry.value)
            self._pending.append(entry)

    # ------------------------------------------------------------------ #
    # Offline path                                                         #
    # ------------------------------------------------------------------ #

    def maybe_consolidate(self) -> None:
        """Process all pending entries via classify_revision, updating stores."""
        if not self._pending:
            return

        for entry in self._pending:
            self._process_entry(entry)

        self._pending.clear()
        self._apply_importance_threshold()

    def _process_entry(self, entry: MemoryEntry) -> None:
        scope_key = (entry.entity, entry.attribute, entry.scope)
        existing: Optional[MemoryEntry] = self.current_state.get(scope_key)
        prior = self._prior_values[entry.key] - {entry.value}

        op = self.consolidator.classify_revision(entry, existing, prior_values=prior)

        if op == RevisionOp.ADD:
            activated = dataclasses.replace(entry, status=MemoryStatus.ACTIVE)
            self.current_state[scope_key] = activated

        elif op == RevisionOp.REINFORCE:
            reinforced = dataclasses.replace(
                entry,
                status=MemoryStatus.REINFORCED,
                importance=min(1.0, (existing.importance if existing else 1.0) + 0.1),
            )
            self.current_state[scope_key] = reinforced

        elif op in (RevisionOp.REVISE, RevisionOp.REVERT):
            if existing is not None:
                superseded = dataclasses.replace(
                    existing,
                    status=MemoryStatus.SUPERSEDED,
                    importance=max(0.0, existing.importance - 0.2),
                )
                self.archive[entry.key].append(superseded)
            revised = dataclasses.replace(
                entry,
                status=MemoryStatus.ACTIVE,
                supersedes_id=existing.entry_id if existing else None,
            )
            self.current_state[scope_key] = revised

        elif op == RevisionOp.SPLIT_SCOPE:
            # Different scope → create a parallel active entry, don't overwrite
            split = dataclasses.replace(entry, status=MemoryStatus.ACTIVE)
            new_scope_key = (entry.entity, entry.attribute, entry.scope)
            self.current_state[new_scope_key] = split

        elif op == RevisionOp.CONFLICT_UNRESOLVED:
            conflicted_new = dataclasses.replace(entry, status=MemoryStatus.CONFLICTED)
            if existing is not None:
                conflicted_existing = dataclasses.replace(existing, status=MemoryStatus.CONFLICTED)
                self.conflict_table[entry.key].append(conflicted_existing)
                # Remove from current_state — unresolved, don't surface as active
                self.current_state.pop(scope_key, None)
            self.conflict_table[entry.key].append(conflicted_new)

        elif op == RevisionOp.LOW_CONFIDENCE:
            # Do not update current_state; archive silently
            low = dataclasses.replace(entry, status=MemoryStatus.ARCHIVED)
            self.archive[entry.key].append(low)

        # RevisionOp.NO_OP: do nothing

    def _apply_importance_threshold(self) -> None:
        to_demote = [
            k for k, e in self.current_state.items()
            if e.importance < self.importance_threshold
        ]
        for k in to_demote:
            entry = self.current_state.pop(k)
            mem_key = (entry.entity, entry.attribute)
            self.archive[mem_key].append(dataclasses.replace(entry, status=MemoryStatus.ARCHIVED))

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        """Default retrieval: current-state entries for this (entity, attribute)."""
        entries = self._current_entries(entity, attribute)
        if not entries:
            entries = self._archive_entries(entity, attribute)[:top_k]
        return RetrievalResult(
            entries=entries[:top_k],
            debug={
                "policy": self.name,
                "conflict_count": str(len(self.conflict_table.get((entity, attribute), []))),
            },
        )

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            return self._retrieve_open_ended(query, top_k=max(top_k, 8))
        return self.retrieve_by_mode(query)

    def retrieve_by_mode(self, query: Query) -> RetrievalResult:
        """Mode-aware retrieval dispatched on query.query_mode."""
        entity, attribute = query.entity, query.attribute
        mode = query.query_mode

        if mode == QueryMode.CURRENT_STATE:
            entries = self._current_entries(entity, attribute)

        elif mode == QueryMode.STATE_WITH_PROVENANCE:
            current = self._current_entries(entity, attribute)
            archived = self.archive.get((entity, attribute), [])
            entries = current + archived

        elif mode == QueryMode.HISTORY:
            entries = [e for e in self.episodic_log if e.entity == entity and e.attribute == attribute]

        elif mode == QueryMode.CONFLICT_AWARE:
            conflicts = self.conflict_table.get((entity, attribute), [])
            current = self._current_entries(entity, attribute)
            entries = conflicts + current

        else:
            entries = self._current_entries(entity, attribute)

        conflict_count = len(self.conflict_table.get((entity, attribute), []))
        return RetrievalResult(
            entries=entries,
            debug={
                "policy": self.name,
                "mode": mode.name,
                "conflict_count": str(conflict_count),
            },
        )

    def _retrieve_open_ended(self, query: Query, *, top_k: int) -> RetrievalResult:
        if query.query_mode == QueryMode.HISTORY:
            candidates = [
                entry
                for entry in self.episodic_log
                if self._entity_matches(entry.entity, query.entity) and entry.attribute == query.attribute
            ]
            shortlisted = shortlist_open_ended_candidates(
                candidates,
                query,
                score_fn=lambda entry: self._open_ended_candidate_score(entry, query, anchor_scopes=set()),
                limit=max(top_k * 16, 64),
            )
            result = lexical_retrieval(
                shortlisted,
                query,
                top_k=top_k,
                policy_name=self.name,
                secondary_score_fn=lambda entry: self._open_ended_candidate_score(
                    entry,
                    query,
                    anchor_scopes=set(),
                ),
            )
            return RetrievalResult(
                entries=result.entries,
                debug={
                    **result.debug,
                    "retrieval_mode": "open_ended_history",
                },
            )

        anchor_pool = (
            self._current_entries(query.entity, query.attribute)
            or self._archive_entries(query.entity, query.attribute)
            or self._conflict_entries(query.entity, query.attribute)
        )
        if not anchor_pool:
            anchor_pool = [
                entry
                for entry in self.episodic_log
                if self._entity_matches(entry.entity, query.entity) and entry.attribute == query.attribute
            ]

        anchor_shortlist = shortlist_open_ended_candidates(
            anchor_pool,
            query,
            score_fn=lambda entry: self._open_ended_anchor_score(entry, query),
            limit=max(top_k * 8, 32),
        )
        anchor_ranked = lexical_retrieval(
            anchor_shortlist,
            query,
            top_k=max(2, min(top_k, 4)),
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._open_ended_anchor_score(entry, query),
        )
        anchor_scopes = {
            entry.scope
            for entry in anchor_ranked.entries
            if entry.scope and entry.scope != "default"
        }

        candidates = [
            entry
            for entry in self.episodic_log
            if self._entity_matches(entry.entity, query.entity)
            and entry.attribute == query.attribute
            and (
                not anchor_scopes
                or entry.scope in anchor_scopes
                or entry.scope == "default"
            )
        ]
        shortlisted = shortlist_open_ended_candidates(
            candidates,
            query,
            score_fn=lambda entry: self._open_ended_candidate_score(entry, query, anchor_scopes=anchor_scopes),
            limit=max(top_k * 16, 64),
        )
        result = lexical_retrieval(
            shortlisted,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._open_ended_candidate_score(
                entry,
                query,
                anchor_scopes=anchor_scopes,
            ),
        )
        return RetrievalResult(
            entries=result.entries,
            debug={
                **result.debug,
                "retrieval_mode": "open_ended_scoped",
            },
        )

    def _open_ended_anchor_score(self, entry: MemoryEntry, query: Query) -> tuple[float, ...]:
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.4,
            MemoryStatus.SUPERSEDED: 0.2,
            MemoryStatus.ARCHIVED: 0.1,
        }.get(entry.status, 0.0)
        scope_bonus = 1.0 if entry.scope != "default" else 0.5
        time_bias = -float(entry.timestamp) if query.query_mode == QueryMode.HISTORY else float(entry.timestamp)
        return (
            status_bonus,
            entry.importance,
            entry.confidence,
            scope_bonus,
            time_bias,
        )

    def _open_ended_candidate_score(
        self,
        entry: MemoryEntry,
        query: Query,
        *,
        anchor_scopes: Set[str],
    ) -> tuple[float, ...]:
        scope_bonus = 1.0 if anchor_scopes and entry.scope in anchor_scopes else 0.0
        time_bias = -float(entry.timestamp) if query.query_mode == QueryMode.HISTORY else float(entry.timestamp)
        return (
            scope_bonus,
            entry.importance,
            entry.confidence,
            time_bias,
        )

    def _current_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        current = [
            e for (ent, attr, _scope), e in self.current_state.items()
            if self._entity_matches(ent, entity) and attr == attribute
        ]
        current.sort(key=lambda entry: entry.timestamp, reverse=True)
        return current

    def _archive_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        archived = [
            entry
            for (ent, attr), entries in self.archive.items()
            if self._entity_matches(ent, entity) and attr == attribute
            for entry in entries
        ]
        archived.sort(key=lambda entry: entry.timestamp, reverse=True)
        return archived

    def _conflict_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        conflicts = [
            entry
            for (ent, attr), entries in self.conflict_table.items()
            if self._entity_matches(ent, entity) and attr == attribute
            for entry in entries
        ]
        conflicts.sort(key=lambda entry: entry.timestamp, reverse=True)
        return conflicts

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    def snapshot_size(self) -> int:
        return len(self.current_state) + sum(len(v) for v in self.archive.values())
