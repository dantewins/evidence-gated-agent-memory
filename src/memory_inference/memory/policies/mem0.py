from __future__ import annotations

from typing import Iterable

from memory_inference.memory.retrieval.semantic import DenseEncoder
from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.retrieval import DenseRanker, expand_with_support_entries, is_open_ended_query
from memory_inference.memory.revision import Mem0RevisionEngine
from memory_inference.memory.stores import ArchiveStore, ConflictStore, CurrentStateStore


class Mem0Policy(BaseMemoryPolicy):
    """Composed Mem0 policy with optional history and archive/conflict features."""

    def __init__(
        self,
        *,
        name: str,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
        history_enabled: bool = False,
        archive_conflict_enabled: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.revision_engine = Mem0RevisionEngine()
        self.state_store = CurrentStateStore()
        self.dense_ranker = DenseRanker(encoder=encoder, write_top_k=write_top_k)
        self.history_enabled = history_enabled
        self.archive_conflict_enabled = archive_conflict_enabled
        self.episodic_log: list[MemoryRecord] = []
        self.archive_store = ArchiveStore() if archive_conflict_enabled else None
        self.conflict_store = ConflictStore() if archive_conflict_enabled else None

    @property
    def encoder(self) -> DenseEncoder:
        return self.dense_ranker.encoder

    @property
    def write_top_k(self) -> int:
        return self.dense_ranker.write_top_k

    @property
    def active_store(self) -> dict[str, MemoryRecord]:
        return self.state_store.records

    @property
    def archive(self):
        return self.archive_store.entries if self.archive_store is not None else {}

    @property
    def conflict_table(self):
        return self.conflict_store.entries if self.conflict_store is not None else {}

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        for update in updates:
            self.episodic_log.append(update)
            prepared = self.revision_engine.prepare_entry(update)
            self.revision_engine.apply(
                prepared,
                state_store=self.state_store,
                ranker=self.dense_ranker,
                archive_store=self.archive_store,
                conflict_store=self.conflict_store,
            )

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id=f"{self.name}-retrieve",
            context_id=f"{self.name}-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=self.state_store.latest_timestamp(),
            session_id=f"{self.name}-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        if self.history_enabled and query.query_mode == QueryMode.HISTORY:
            return self._retrieve_history(query, top_k=top_k)
        if self.archive_conflict_enabled and query.query_mode != QueryMode.HISTORY:
            return self._retrieve_augmented_state(query, top_k=top_k)
        return self._retrieve_active_state(query, top_k=top_k)

    def snapshot_size(self) -> int:
        size = self.state_store.snapshot_size()
        if self.archive_store is not None:
            size += self.archive_store.snapshot_size()
        if self.conflict_store is not None:
            size += self.conflict_store.snapshot_size()
        return size

    def _retrieve_active_state(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
        ranked = self.dense_ranker.rank_query(
            query,
            self.state_store.values(),
            entity_matches=self._entity_matches,
        )
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.state_store.values(),
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalBundle(
            records=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_active_dense",
            },
        )

    def _retrieve_history(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
        candidates = [
            entry
            for entry in self.episodic_log
            if self._entity_matches(entry.entity, query.entity)
            and entry.attribute == query.attribute
        ]
        ranked = self.dense_ranker.rank_query(
            query,
            candidates,
            entity_matches=self._entity_matches,
            history=True,
        )
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.episodic_log,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalBundle(
            records=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_history_dense",
            },
        )

    def _retrieve_augmented_state(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
        active_ranked = self.dense_ranker.rank_query(
            query,
            self.state_store.values(),
            entity_matches=self._entity_matches,
        )
        candidates: list[MemoryRecord] = []
        if query.query_mode == QueryMode.CONFLICT_AWARE and self.conflict_store is not None:
            candidates.extend(self.conflict_store.by_query(
                entity=query.entity,
                attribute=query.attribute,
                entity_matches=self._entity_matches,
            )[:2])
        candidates.extend(active_ranked)
        if query.query_mode in {
            QueryMode.CURRENT_STATE,
            QueryMode.STATE_WITH_PROVENANCE,
            QueryMode.CONFLICT_AWARE,
        } and self.archive_store is not None:
            candidates.extend(self.archive_store.by_query(
                entity=query.entity,
                attribute=query.attribute,
                entity_matches=self._entity_matches,
            )[:2])

        deduped: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for entry in candidates:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            deduped.append(entry)

        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = deduped[:limit]
        support_source = self.state_store.values()
        if self.archive_store is not None:
            support_source = support_source + self.archive_store.by_query(
                entity=query.entity,
                attribute=query.attribute,
                entity_matches=self._entity_matches,
            )
        if self._has_structured_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                support_source,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalBundle(
            records=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "mem0_state_augmented",
                "query_mode": query.query_mode.name,
            },
        )

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    def _has_structured_fact(self, entries: Iterable[MemoryRecord]) -> bool:
        return any(entry.source_kind == "structured_fact" for entry in entries)
