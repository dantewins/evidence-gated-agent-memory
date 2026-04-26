from __future__ import annotations

from typing import Iterable

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.mem0 import Mem0Policy
from memory_inference.memory.policies.odv2 import ODV2Policy
from memory_inference.memory.retrieval import expand_with_support_entries, is_open_ended_query
from memory_inference.memory.retrieval.semantic import DenseEncoder


class ODV2RecoveryPolicy(BaseMemoryPolicy):
    """Validity-gated high-recall policy for real benchmark recovery runs.

    The policy keeps ODV2 as the state authority for current/conflict queries,
    but delegates broad recall to a history-aware Mem0-style retriever when the
    validity ledger has no decisive signal. This avoids making real QA depend on
    brittle structured extraction alone.
    """

    def __init__(
        self,
        *,
        name: str,
        consolidator: BaseConsolidator,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
        importance_threshold: float = 0.1,
        support_history_limit: int = 1,
    ) -> None:
        super().__init__(name=name)
        self.support_history_limit = support_history_limit
        self.validity = ODV2Policy(
            name=f"{name}::validity",
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=support_history_limit,
        )
        self.retriever = Mem0Policy(
            name=f"{name}::mem0",
            encoder=encoder,
            write_top_k=write_top_k,
            history_enabled=True,
            archive_conflict_enabled=False,
        )
        self.episodic_log: list[MemoryRecord] = []

    @property
    def current_state(self):
        return self.validity.current_state

    @property
    def archive(self):
        return self.validity.archive

    @property
    def conflict_table(self):
        return self.validity.conflict_table

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        update_list = list(updates)
        if not update_list:
            return
        self.episodic_log.extend(update_list)
        self.validity.ingest(update_list)
        self.retriever.ingest(update_list)

    def maybe_consolidate(self) -> None:
        self.validity.maybe_consolidate()
        self.retriever.maybe_consolidate()
        self.maintenance_tokens = self.validity.maintenance_tokens + self.retriever.maintenance_tokens
        self.maintenance_latency_ms = (
            self.validity.maintenance_latency_ms + self.retriever.maintenance_latency_ms
        )
        self.maintenance_calls = self.validity.maintenance_calls + self.retriever.maintenance_calls

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id=f"{self.name}-retrieve",
            context_id=f"{self.name}-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=max((record.timestamp for record in self.episodic_log), default=0),
            session_id=f"{self.name}-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        if query.query_mode == QueryMode.HISTORY:
            return self._history_retrieval(query, top_k=top_k)

        base = self.retriever.retrieve_for_query(query, top_k=top_k)
        if is_open_ended_query(query):
            return self._bundle(
                list(base.records),
                base=base,
                retrieval_mode="odv2_recovery_mem0_open_ended",
            )

        current_entries = self.validity.current_entries_for_query(query)
        archive_entries = self.validity.archive_entries_for_query(query)
        conflict_entries = self.validity.conflict_entries_for_query(query)

        if query.query_mode == QueryMode.CONFLICT_AWARE and conflict_entries:
            records = self._dedupe(
                conflict_entries
                + current_entries
                + self._filter_stale_state_records(
                    list(base.records),
                    current_entries=current_entries,
                    archive_entries=archive_entries,
                )
            )
            return self._bundle(
                self._cap(records, top_k=max(top_k, len(conflict_entries))),
                base=base,
                retrieval_mode="odv2_recovery_conflict_guard",
                conflict_count=len(conflict_entries),
            )

        if current_entries and self._has_decisive_current_state(query, current_entries):
            filtered_base = self._filter_stale_state_records(
                list(base.records),
                current_entries=current_entries,
                archive_entries=archive_entries,
            )
            anchored = expand_with_support_entries(
                current_entries,
                self.episodic_log,
                support_limit=self.support_history_limit,
                max_entries=len(current_entries) + self.support_history_limit,
            )
            records = self._dedupe(filtered_base + anchored)
            if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
                records = self._dedupe(records + archive_entries[:1])
            return self._bundle(
                self._cap(records, top_k=max(top_k, len(anchored))),
                base=base,
                retrieval_mode="odv2_recovery_current_guard",
                archive_count=len(archive_entries),
            )

        return self._bundle(
            list(base.records),
            base=base,
            retrieval_mode="odv2_recovery_mem0_fallback",
        )

    def snapshot_size(self) -> int:
        return self.validity.snapshot_size() + self.retriever.snapshot_size()

    def _history_retrieval(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
        base = self.retriever.retrieve_for_query(query, top_k=top_k)
        return self._bundle(
            list(base.records),
            base=base,
            retrieval_mode="odv2_recovery_history",
        )

    def _filter_stale_state_records(
        self,
        records: list[MemoryRecord],
        *,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if not current_entries:
            return records

        current_entry_ids = {entry.entry_id for entry in current_entries}
        current_values = {self._normalized_value(entry.value) for entry in current_entries}
        current_source_ids = self._source_ids(current_entries)
        archived_source_ids = self._source_ids(archive_entries) - current_source_ids

        filtered: list[MemoryRecord] = []
        for record in records:
            if record.entry_id in archived_source_ids:
                continue
            if self._is_state_record(record):
                if record.entry_id in current_entry_ids:
                    filtered.append(record)
                    continue
                if self._normalized_value(record.value) not in current_values:
                    continue
            filtered.append(record)
        return filtered

    def _has_decisive_current_state(
        self,
        query: RuntimeQuery,
        current_entries: list[MemoryRecord],
    ) -> bool:
        if query.entity not in {"conversation", "all"}:
            return True
        current_values = {self._normalized_value(entry.value) for entry in current_entries}
        return len(current_values) <= 1

    def _bundle(
        self,
        records: list[MemoryRecord],
        *,
        base: RetrievalBundle,
        retrieval_mode: str,
        conflict_count: int = 0,
        archive_count: int = 0,
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "validity_conflicts": str(conflict_count),
                "validity_archive": str(archive_count),
            },
        )

    @staticmethod
    def _source_ids(entries: Iterable[MemoryRecord]) -> set[str]:
        return {
            entry.source_entry_id
            for entry in entries
            if entry.source_entry_id
        }

    @staticmethod
    def _is_state_record(entry: MemoryRecord) -> bool:
        return entry.memory_kind == "state" or entry.source_kind == "structured_fact"

    @staticmethod
    def _normalized_value(value: str) -> str:
        return " ".join(value.lower().split())

    @staticmethod
    def _dedupe(records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        deduped: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.entry_id in seen_ids:
                continue
            seen_ids.add(record.entry_id)
            deduped.append(record)
        return deduped

    @staticmethod
    def _cap(records: list[MemoryRecord], *, top_k: int) -> list[MemoryRecord]:
        return records[:top_k]
