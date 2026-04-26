from __future__ import annotations

from typing import Iterable

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.mem0 import Mem0Policy
from memory_inference.memory.policies.odv2 import ODV2Policy
from memory_inference.memory.retrieval.semantic import DenseEncoder


class ODV2Mem0SelectivePolicy(BaseMemoryPolicy):
    """Mem0-first retrieval with conservative ODV2 stale-state suppression.

    This policy is intentionally asymmetric: Mem0 remains the default retrieval
    path, and ODV2 is allowed to remove stale same-key state only when Mem0 has
    already retrieved the corresponding current value. The policy never injects
    ODV2-only evidence into the answer context; this keeps Mem0 recall as the
    safety floor while reducing obvious contradictory state. It also removes
    redundant support turns when the retrieved state fact already carries the
    support text, which can reduce prompt cost without dropping the state value.
    """

    def __init__(
        self,
        *,
        name: str,
        consolidator: BaseConsolidator,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
        importance_threshold: float = 0.1,
        max_validity_appends: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.max_validity_appends = max_validity_appends
        self.retriever = Mem0Policy(
            name=f"{name}::mem0",
            encoder=encoder,
            write_top_k=write_top_k,
            history_enabled=False,
            archive_conflict_enabled=False,
        )
        self.validity = ODV2Policy(
            name=f"{name}::validity",
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=1,
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
        self.retriever.ingest(update_list)
        self.validity.ingest(update_list)

    def maybe_consolidate(self) -> None:
        self.retriever.maybe_consolidate()
        self.validity.maybe_consolidate()
        self.maintenance_tokens = self.retriever.maintenance_tokens + self.validity.maintenance_tokens
        self.maintenance_latency_ms = (
            self.retriever.maintenance_latency_ms + self.validity.maintenance_latency_ms
        )
        self.maintenance_calls = self.retriever.maintenance_calls + self.validity.maintenance_calls

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
        base = self.retriever.retrieve_for_query(query, top_k=top_k)
        base_records = list(base.records)

        if query.query_mode == QueryMode.HISTORY or query.attribute in {"dialogue", "event"}:
            return self._bundle(base_records, base=base, retrieval_mode="odv2_mem0_selective_passthrough")

        base_records, support_compacted = self._compact_redundant_support(base_records, query)

        current_entries = self.validity.current_entries_for_query(query)
        archive_entries = self.validity.archive_entries_for_query(query)
        conflict_entries = self.validity.conflict_entries_for_query(query)

        if query.query_mode == QueryMode.CONFLICT_AWARE and conflict_entries:
            records = self._dedupe(conflict_entries + base_records)
            return self._bundle(
                records[: max(top_k, len(conflict_entries))],
                base=base,
                retrieval_mode="odv2_mem0_selective_conflict",
                removed_count=0,
                appended_count=0,
                support_compacted=support_compacted,
            )

        decisive_current = self._decisive_current_entries(query, current_entries)
        if not decisive_current:
            return self._bundle(
                base_records,
                base=base,
                retrieval_mode=(
                    "odv2_mem0_selective_compact"
                    if support_compacted
                    else "odv2_mem0_selective_passthrough"
                ),
                support_compacted=support_compacted,
            )
        if not self._base_contains_current_same_key(base_records, query, decisive_current):
            return self._bundle(
                base_records,
                base=base,
                retrieval_mode=(
                    "odv2_mem0_selective_compact"
                    if support_compacted
                    else "odv2_mem0_selective_passthrough"
                ),
                support_compacted=support_compacted,
            )

        filtered, removed_count = self._remove_stale_same_key_records(
            base_records,
            query=query,
            current_entries=decisive_current,
            archive_entries=archive_entries,
        )
        return self._bundle(
            filtered[:top_k],
            base=base,
            retrieval_mode=(
                "odv2_mem0_selective_guard"
                if removed_count
                else (
                    "odv2_mem0_selective_compact"
                    if support_compacted
                    else "odv2_mem0_selective_passthrough"
                )
            ),
            removed_count=removed_count,
            appended_count=0,
            support_compacted=support_compacted,
        )

    def snapshot_size(self) -> int:
        return self.retriever.snapshot_size() + self.validity.snapshot_size()

    def _decisive_current_entries(
        self,
        query: RuntimeQuery,
        current_entries: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if not current_entries:
            return []
        if query.entity in {"conversation", "all"}:
            values_by_key: dict[tuple[str, str], set[str]] = {}
            for entry in current_entries:
                values_by_key.setdefault(entry.key, set()).add(self._normalized_value(entry.value))
            if len(values_by_key) != 1:
                return []
            if any(len(values) != 1 for values in values_by_key.values()):
                return []
        else:
            current_values = {self._normalized_value(entry.value) for entry in current_entries}
            if len(current_values) != 1:
                return []
        return current_entries[:1]

    def _remove_stale_same_key_records(
        self,
        records: list[MemoryRecord],
        *,
        query: RuntimeQuery,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
    ) -> tuple[list[MemoryRecord], int]:
        current_values = {self._normalized_value(entry.value) for entry in current_entries}
        current_ids = {entry.entry_id for entry in current_entries}
        stale_source_ids = self._source_ids(archive_entries) - self._source_ids(current_entries)
        archived_values = {self._normalized_value(entry.value) for entry in archive_entries}
        removed = 0
        filtered: list[MemoryRecord] = []
        for record in records:
            if record.entry_id in stale_source_ids:
                removed += 1
                continue
            if self._same_query_key(record, query) and self._is_state_record(record):
                if record.entry_id in current_ids:
                    filtered.append(record)
                    continue
                normalized_value = self._normalized_value(record.value)
                if normalized_value in archived_values and normalized_value not in current_values:
                    removed += 1
                    continue
                if normalized_value not in current_values:
                    removed += 1
                    continue
            filtered.append(record)
        return filtered, removed

    def _base_contains_current_same_key(
        self,
        records: list[MemoryRecord],
        query: RuntimeQuery,
        current_entries: list[MemoryRecord],
    ) -> bool:
        current_values = {self._normalized_value(entry.value) for entry in current_entries}
        return any(
            self._same_query_key(record, query)
            and self._is_state_record(record)
            and self._normalized_value(record.value) in current_values
            for record in records
        )

    def _compact_redundant_support(
        self,
        records: list[MemoryRecord],
        query: RuntimeQuery,
    ) -> tuple[list[MemoryRecord], int]:
        if query.query_mode not in {QueryMode.CURRENT_STATE, QueryMode.STATE_WITH_PROVENANCE}:
            return records, 0
        support_source_ids = {
            record.source_entry_id
            for record in records
            if self._same_query_key(record, query)
            and self._is_state_record(record)
            and record.source_entry_id
            and record.support_text
        }
        if not support_source_ids:
            return records, 0
        compacted: list[MemoryRecord] = []
        removed = 0
        for record in records:
            if record.entry_id in support_source_ids and record.attribute in {"dialogue", "event"}:
                removed += 1
                continue
            compacted.append(record)
        return compacted, removed

    def _bundle(
        self,
        records: list[MemoryRecord],
        *,
        base: RetrievalBundle,
        retrieval_mode: str,
        removed_count: int = 0,
        appended_count: int = 0,
        support_compacted: int = 0,
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "validity_removed": str(removed_count),
                "validity_appended": str(appended_count),
                "support_compacted": str(support_compacted),
            },
        )

    @staticmethod
    def _same_query_key(record: MemoryRecord, query: RuntimeQuery) -> bool:
        if query.entity not in {"conversation", "all"} and record.entity != query.entity:
            return False
        return record.attribute == query.attribute

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
