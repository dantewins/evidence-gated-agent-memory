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


class ODV2Mem0TemporalPrunePolicy(BaseMemoryPolicy):
    """Mem0 retrieval with monotonic pruning of retrieved same-key conflicts.

    The policy never adds records. For current-state questions, if Mem0 returns
    multiple state values for the queried entity/attribute, this policy keeps the
    ODV2-current value when the ledger agrees with one retrieved value; otherwise
    it keeps the uniquely latest retrieved value. This gives a validity-focused
    intervention that can reduce contradictory context without changing the
    evidence source or broad retrieval distribution.
    """

    def __init__(
        self,
        *,
        name: str,
        consolidator: BaseConsolidator,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
        importance_threshold: float = 0.1,
    ) -> None:
        super().__init__(name=name)
        self.retriever = Mem0Policy(
            name=f"{name}::mem0",
            encoder=encoder,
            write_top_k=write_top_k,
            history_enabled=True,
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
        if not self._can_prune(query):
            return self._bundle(base_records, base=base, retrieval_mode="odv2_mem0_temporal_passthrough")

        same_key_records = [
            record
            for record in base_records
            if self._same_query_key(record, query) and self._is_state_record(record)
        ]
        values = {self._normalized_value(record.value) for record in same_key_records}
        if len(values) < 2:
            return self._bundle(base_records, base=base, retrieval_mode="odv2_mem0_temporal_passthrough")

        selected_value = self._ledger_supported_value(query, same_key_records)
        decision_source = "ledger"
        if selected_value is None:
            selected_value = self._unique_latest_value(same_key_records)
            decision_source = "latest_timestamp"
        if selected_value is None:
            return self._bundle(
                base_records,
                base=base,
                retrieval_mode="odv2_mem0_temporal_ambiguous",
                conflict_values=len(values),
            )

        filtered: list[MemoryRecord] = []
        pruned = 0
        for record in base_records:
            if (
                self._same_query_key(record, query)
                and self._is_state_record(record)
                and self._normalized_value(record.value) != selected_value
            ):
                pruned += 1
                continue
            filtered.append(record)
        if not filtered:
            return self._bundle(base_records, base=base, retrieval_mode="odv2_mem0_temporal_passthrough")
        return self._bundle(
            filtered[:top_k],
            base=base,
            retrieval_mode="odv2_mem0_temporal_prune",
            temporal_pruned=pruned,
            conflict_values=len(values),
            decision_source=decision_source,
        )

    def snapshot_size(self) -> int:
        return self.retriever.snapshot_size() + self.validity.snapshot_size()

    def _ledger_supported_value(
        self,
        query: RuntimeQuery,
        records: list[MemoryRecord],
    ) -> str | None:
        retrieved_values = {self._normalized_value(record.value) for record in records}
        current_values = {
            self._normalized_value(entry.value)
            for entry in self.validity.current_entries_for_query(query)
        }
        supported = current_values & retrieved_values
        if len(supported) != 1:
            return None
        return next(iter(supported))

    def _unique_latest_value(self, records: list[MemoryRecord]) -> str | None:
        if not records:
            return None
        latest_timestamp = max(record.timestamp for record in records)
        latest_values = {
            self._normalized_value(record.value)
            for record in records
            if record.timestamp == latest_timestamp
        }
        if len(latest_values) != 1:
            return None
        return next(iter(latest_values))

    @staticmethod
    def _can_prune(query: RuntimeQuery) -> bool:
        return (
            query.query_mode in {QueryMode.CURRENT_STATE, QueryMode.STATE_WITH_PROVENANCE}
            and query.attribute not in {"dialogue", "event"}
        )

    @staticmethod
    def _same_query_key(record: MemoryRecord, query: RuntimeQuery) -> bool:
        if query.entity not in {"conversation", "all"} and record.entity != query.entity:
            return False
        return record.attribute == query.attribute

    @staticmethod
    def _is_state_record(entry: MemoryRecord) -> bool:
        return entry.memory_kind == "state" or entry.source_kind == "structured_fact"

    @staticmethod
    def _normalized_value(value: str) -> str:
        return " ".join(value.lower().split())

    def _bundle(
        self,
        records: list[MemoryRecord],
        *,
        base: RetrievalBundle,
        retrieval_mode: str,
        temporal_pruned: int = 0,
        conflict_values: int = 0,
        decision_source: str = "",
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "temporal_pruned": str(temporal_pruned),
                "conflict_values": str(conflict_values),
                "decision_source": decision_source,
            },
        )
