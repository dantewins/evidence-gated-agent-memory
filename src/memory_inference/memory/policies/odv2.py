from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Set

from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryKey, MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.retrieval import (
    HybridRanker,
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.memory.revision import ODV2RevisionEngine
from memory_inference.memory.stores import ArchiveStore, ConflictStore, ScopedCurrentStateStore


class ODV2Policy(BaseMemoryPolicy):
    def __init__(
        self,
        *,
        name: str,
        consolidator: BaseConsolidator,
        importance_threshold: float = 0.1,
        support_history_limit: int = 3,
        hybrid_backbone=None,
        broad_candidate_pool: bool = False,
        prefer_exact_entity_when_available: bool = False,
        compact_current_state: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.support_history_limit = support_history_limit
        self.revision_engine = ODV2RevisionEngine(
            consolidator=consolidator,
            importance_threshold=importance_threshold,
        )
        self.state_store = ScopedCurrentStateStore()
        self.archive_store = ArchiveStore()
        self.conflict_store = ConflictStore()
        self.hybrid_backbone = hybrid_backbone
        self.hybrid_ranker = (
            HybridRanker(
                backbone=hybrid_backbone,
                support_history_limit=support_history_limit,
                entity_matches=self._entity_matches,
                broad_candidate_pool=broad_candidate_pool,
                prefer_exact_entity_when_available=prefer_exact_entity_when_available,
                compact_current_state=compact_current_state,
            )
            if hybrid_backbone is not None
            else None
        )
        self.episodic_log: list[MemoryRecord] = []
        self._pending: list[MemoryRecord] = []
        self._prior_values: DefaultDict[MemoryKey, Set[str]] = defaultdict(set)

    @property
    def current_state(self):
        return self.state_store.records

    @property
    def archive(self):
        return self.archive_store.entries

    @property
    def conflict_table(self):
        return self.conflict_store.entries

    def ingest(self, updates) -> None:
        new_entries = list(updates)
        if not new_entries:
            return
        self.episodic_log.extend(new_entries)
        for entry in new_entries:
            self._prior_values[entry.key].add(entry.value)
            self._pending.append(entry)
        if self.hybrid_backbone is not None:
            self.hybrid_backbone.index_entries(new_entries)

    def maybe_consolidate(self) -> None:
        self.revision_engine.consolidate(
            self._pending,
            state_store=self.state_store,
            archive_store=self.archive_store,
            conflict_store=self.conflict_store,
            prior_values=self._prior_values,
        )

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        entries = self._current_entries(entity, attribute)
        if not entries:
            entries = self._archive_entries(entity, attribute)[:top_k]
        return RetrievalBundle(
            records=entries[:top_k],
            debug={
                "policy": self.name,
                "conflict_count": str(len(self.conflict_table.get((entity, attribute), []))),
            },
        )

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        if self.hybrid_ranker is not None:
            hybrid_top_k = max(top_k, 8) if is_open_ended_query(query) else top_k
            return self.hybrid_ranker.retrieve(
                query,
                episodic_log=self.episodic_log,
                current_entries=self._current_entries(query.entity, query.attribute),
                archive_entries=self._archive_entries(query.entity, query.attribute),
                conflict_entries=self._conflict_entries(query.entity, query.attribute),
                top_k=hybrid_top_k,
                policy_name=self.name,
            )
        if is_open_ended_query(query):
            return self._retrieve_open_ended(query, top_k=max(top_k, 8))
        return self.retrieve_by_mode(query)

    def retrieve_by_mode(self, query: RuntimeQuery) -> RetrievalBundle:
        entity, attribute = query.entity, query.attribute
        mode = query.query_mode

        if mode == QueryMode.CURRENT_STATE:
            entries = self._current_entries(entity, attribute)
        elif mode == QueryMode.STATE_WITH_PROVENANCE:
            entries = self._current_entries(entity, attribute) + self._archive_entries(entity, attribute)
        elif mode == QueryMode.HISTORY:
            entries = [
                entry
                for entry in self.episodic_log
                if self._entity_matches(entry.entity, entity) and entry.attribute == attribute
            ]
        elif mode == QueryMode.CONFLICT_AWARE:
            entries = self._conflict_entries(entity, attribute) + self._current_entries(entity, attribute)
        else:
            entries = self._current_entries(entity, attribute)

        return RetrievalBundle(
            records=entries,
            debug={
                "policy": self.name,
                "mode": mode.name,
                "conflict_count": str(len(self.conflict_table.get((entity, attribute), []))),
            },
        )

    def current_entries_for_query(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self._current_entries(query.entity, query.attribute)

    def archive_entries_for_query(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self._archive_entries(query.entity, query.attribute)

    def conflict_entries_for_query(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self._conflict_entries(query.entity, query.attribute)

    def snapshot_size(self) -> int:
        return self.state_store.snapshot_size() + self.archive_store.snapshot_size()

    def _retrieve_open_ended(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
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
            return RetrievalBundle(
                records=result.records,
                debug={**result.debug, "retrieval_mode": "open_ended_history"},
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
            score_fn=lambda entry: self._open_ended_candidate_score(
                entry,
                query,
                anchor_scopes=anchor_scopes,
            ),
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
        return RetrievalBundle(
            records=result.records,
            debug={**result.debug, "retrieval_mode": "open_ended_scoped"},
        )

    def _open_ended_anchor_score(self, entry: MemoryRecord, query: RuntimeQuery) -> tuple[float, ...]:
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
        entry: MemoryRecord,
        query: RuntimeQuery,
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

    def _current_entries(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return self.state_store.by_query(
            entity=entity,
            attribute=attribute,
            entity_matches=self._entity_matches,
        )

    def _archive_entries(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return self.archive_store.by_query(
            entity=entity,
            attribute=attribute,
            entity_matches=self._entity_matches,
        )

    def _conflict_entries(self, entity: str, attribute: str) -> list[MemoryRecord]:
        return self.conflict_store.by_query(
            entity=entity,
            attribute=attribute,
            entity_matches=self._entity_matches,
        )

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity
