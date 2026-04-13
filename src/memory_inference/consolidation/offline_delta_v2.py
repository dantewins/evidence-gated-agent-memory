from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode, RevisionOp
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.open_ended_eval import (
    expand_with_support_entries,
    has_structured_fact_candidates,
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
        self._maintenance_ticks = 0

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
        self._maintenance_ticks += 1
        if self._maintenance_ticks % self.maintenance_frequency != 0:
            return
        pending_count = len(self._pending)
        before_calls = self.consolidator.total_calls

        for entry in self._pending:
            self._process_entry(entry)

        self._pending.clear()
        self._apply_importance_threshold()
        self.maintenance_calls += 1
        consolidator_calls = self.consolidator.total_calls - before_calls
        self.maintenance_tokens += max(1, pending_count) * 8 + consolidator_calls * 4
        self.maintenance_latency_ms += float(pending_count + consolidator_calls)

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
            archived = sorted(
                self._archive_entries(entity, attribute),
                key=lambda e: e.timestamp, reverse=True,
            )
            entries = archived[:top_k]
        return RetrievalResult(
            entries=entries[:top_k],
            debug={
                "policy": self.name,
                "conflict_count": str(len(self._conflict_entries(entity, attribute))),
            },
        )

    def retrieve_by_mode(self, query: Query) -> RetrievalResult:
        """Mode-aware retrieval dispatched on query.query_mode."""
        entity, attribute = query.entity, query.attribute
        mode = query.query_mode

        if mode == QueryMode.CURRENT_STATE:
            entries = self._current_entries(entity, attribute)

        elif mode == QueryMode.STATE_WITH_PROVENANCE:
            current = self._current_entries(entity, attribute)
            archived = self._archive_entries(entity, attribute)
            entries = current + archived

        elif mode == QueryMode.HISTORY:
            entries = [
                e for e in self.episodic_log
                if self._entity_matches(e.entity, entity) and e.attribute == attribute
            ]

        elif mode == QueryMode.CONFLICT_AWARE:
            conflicts = self._conflict_entries(entity, attribute)
            current = self._current_entries(entity, attribute)
            entries = conflicts + current

        else:
            entries = self._current_entries(entity, attribute)

        conflict_count = len(self._conflict_entries(entity, attribute))
        return RetrievalResult(
            entries=entries,
            debug={
                "policy": self.name,
                "mode": mode.name,
                "conflict_count": str(conflict_count),
            },
        )

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self._open_ended_candidates(query),
                query,
                score_fn=lambda entry: self._open_ended_secondary_score(entry, query),
                limit=max(top_k * 16, 64),
            )
            return lexical_retrieval(
                candidates,
                query,
                top_k=max(top_k, 6),
                policy_name=self.name,
                secondary_score_fn=lambda entry: self._open_ended_secondary_score(entry, query),
            )
        hybrid = self._retrieve_hybrid_structured_query(query, top_k=max(top_k, 8))
        if hybrid is not None:
            return hybrid
        return self.retrieve_by_mode(query)

    def _open_ended_candidates(self, query: Query) -> List[MemoryEntry]:
        scoped_current = self._current_entries(query.entity, query.attribute)
        archived = list(self.archive.get((query.entity, query.attribute), []))
        conflicts = list(self.conflict_table.get((query.entity, query.attribute), []))
        episodic = [entry for entry in self.episodic_log if entry.attribute == query.attribute]

        pools: List[List[MemoryEntry]] = [scoped_current, archived, conflicts, episodic]
        combined: List[MemoryEntry] = []
        seen_ids: Set[str] = set()
        for pool in pools:
            for entry in pool:
                if entry.entry_id in seen_ids:
                    continue
                seen_ids.add(entry.entry_id)
                combined.append(entry)
        return combined

    def _open_ended_secondary_score(self, entry: MemoryEntry, query: Query) -> tuple[float, ...]:
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.4,
            MemoryStatus.SUPERSEDED: 0.2,
            MemoryStatus.ARCHIVED: 0.1,
        }.get(entry.status, 0.0)
        if query.query_mode == QueryMode.HISTORY:
            time_bias = -float(entry.timestamp)
        else:
            time_bias = float(entry.timestamp)
        scope_match = 1.0 if entry.scope != "default" else 0.0
        return (
            status_bonus,
            scope_match,
            entry.importance,
            entry.confidence,
            time_bias,
        )

    def _structured_secondary_score(self, entry: MemoryEntry, query: Query) -> tuple[float, ...]:
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.5,
            MemoryStatus.SUPERSEDED: 0.3,
            MemoryStatus.ARCHIVED: 0.2,
        }.get(entry.status, 0.0)
        entity_bonus = 1.0 if query.entity in {"conversation", "all"} or entry.entity == query.entity else 0.0
        memory_kind_bonus = 0.4 if self._memory_kind(entry) == "state" else 0.0
        support_bonus = 0.2 if entry.metadata.get("source_entry_id") else 0.0
        if query.query_mode == QueryMode.HISTORY:
            time_bias = -float(entry.timestamp)
        else:
            time_bias = float(entry.timestamp)
        return (
            entity_bonus,
            status_bonus,
            memory_kind_bonus,
            support_bonus,
            entry.importance,
            entry.confidence,
            time_bias,
        )

    def _current_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        return [
            e for (ent, attr, _scope), e in self.current_state.items()
            if self._entity_matches(ent, entity) and attr == attribute
        ]

    def _archive_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        if entity in {"conversation", "all"}:
            entries: List[MemoryEntry] = []
            for (stored_entity, stored_attribute), archived in self.archive.items():
                if stored_attribute == attribute and self._entity_matches(stored_entity, entity):
                    entries.extend(archived)
            return entries
        return list(self.archive.get((entity, attribute), []))

    def _conflict_entries(self, entity: str, attribute: str) -> List[MemoryEntry]:
        if entity in {"conversation", "all"}:
            entries: List[MemoryEntry] = []
            for (stored_entity, stored_attribute), conflicts in self.conflict_table.items():
                if stored_attribute == attribute and self._entity_matches(stored_entity, entity):
                    entries.extend(conflicts)
            return entries
        return list(self.conflict_table.get((entity, attribute), []))

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    def _structured_candidates(self, query: Query) -> List[MemoryEntry]:
        if query.query_mode == QueryMode.HISTORY:
            entries = [
                entry
                for entry in self.episodic_log
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
            ]
        else:
            current = self._current_entries(query.entity, query.attribute)
            archived = self._archive_entries(query.entity, query.attribute)
            conflicts = self._conflict_entries(query.entity, query.attribute)
            episodic_structured = [
                entry
                for entry in self.episodic_log
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
                and self._is_structured_fact(entry)
            ]
            entries = current + archived + conflicts + episodic_structured

        combined: List[MemoryEntry] = []
        seen_ids: Set[str] = set()
        for entry in entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            combined.append(entry)
        return combined

    def _retrieve_hybrid_structured_query(
        self,
        query: Query,
        *,
        top_k: int,
    ) -> RetrievalResult | None:
        structured_candidates = self._structured_candidates(query)
        if not has_structured_fact_candidates(structured_candidates):
            return None

        structured_shortlist = shortlist_open_ended_candidates(
            structured_candidates,
            query,
            score_fn=lambda entry: self._structured_secondary_score(entry, query),
            limit=max(top_k * 12, 48),
        )
        structured_ranked = lexical_retrieval(
            structured_shortlist,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._structured_secondary_score(entry, query),
        )

        evidence_candidates = self._evidence_candidates(query, structured_ranked.entries)
        evidence_shortlist = shortlist_open_ended_candidates(
            evidence_candidates,
            query,
            score_fn=lambda entry: self._evidence_secondary_score(
                entry,
                query,
                anchor_source_ids=self._anchor_source_ids(structured_ranked.entries),
            ),
            limit=max(top_k * 16, 64),
        )
        evidence_ranked = lexical_retrieval(
            evidence_shortlist,
            query,
            top_k=max(top_k, 6),
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._evidence_secondary_score(
                entry,
                query,
                anchor_source_ids=self._anchor_source_ids(structured_ranked.entries),
            ),
        )

        merged = self._merge_hybrid_entries(
            state_entries=structured_ranked.entries,
            evidence_entries=evidence_ranked.entries,
            top_k=top_k,
        )
        expanded = expand_with_support_entries(
            merged,
            self.episodic_log,
            support_limit=self.support_history_limit,
            max_entries=top_k + self.support_history_limit,
        )
        return RetrievalResult(
            entries=expanded,
            debug={
                "policy": self.name,
                "retrieval_mode": "hybrid_state_evidence",
            },
        )

    def _evidence_candidates(
        self,
        query: Query,
        structured_entries: Iterable[MemoryEntry],
    ) -> List[MemoryEntry]:
        anchor_source_ids = self._anchor_source_ids(structured_entries)
        candidates: List[MemoryEntry] = []
        seen_ids: Set[str] = set()

        for entry in self.episodic_log:
            if entry.entry_id in anchor_source_ids:
                if entry.entry_id not in seen_ids:
                    seen_ids.add(entry.entry_id)
                    candidates.append(entry)
                continue
            if not self._entity_matches(entry.entity, query.entity):
                continue
            if entry.attribute in {"dialogue", "event", query.attribute}:
                if entry.entry_id in seen_ids:
                    continue
                seen_ids.add(entry.entry_id)
                candidates.append(entry)

        return candidates

    def _evidence_secondary_score(
        self,
        entry: MemoryEntry,
        query: Query,
        *,
        anchor_source_ids: Set[str],
    ) -> tuple[float, ...]:
        anchor_bonus = 1.5 if entry.entry_id in anchor_source_ids else 0.0
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.8 if entry.attribute in {"dialogue", "event"} else 0.0
        if query.query_mode == QueryMode.HISTORY:
            time_bias = -float(entry.timestamp)
        else:
            time_bias = float(entry.timestamp)
        return (
            anchor_bonus,
            attribute_bonus,
            entry.importance,
            entry.confidence,
            time_bias,
        )

    def _merge_hybrid_entries(
        self,
        *,
        state_entries: Iterable[MemoryEntry],
        evidence_entries: Iterable[MemoryEntry],
        top_k: int,
    ) -> List[MemoryEntry]:
        merged: List[MemoryEntry] = []
        seen_ids: Set[str] = set()

        for entry in state_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= top_k:
                return merged

        for entry in evidence_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= top_k:
                break
        return merged

    def _anchor_source_ids(self, entries: Iterable[MemoryEntry]) -> Set[str]:
        return {
            source_entry_id
            for entry in entries
            if (source_entry_id := entry.metadata.get("source_entry_id"))
        }

    def _is_structured_fact(self, entry: MemoryEntry) -> bool:
        return entry.metadata.get("source_kind") == "structured_fact"

    def _memory_kind(self, entry: MemoryEntry) -> str:
        return entry.metadata.get("memory_kind", "state")

    def snapshot_size(self) -> int:
        return len(self.current_state) + sum(len(v) for v in self.archive.values())
