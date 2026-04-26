from __future__ import annotations

from typing import Iterable, Protocol

from memory_inference.domain.enums import QueryMode
from memory_inference.memory.retrieval.query_routing import (
    has_structured_fact_candidates,
    is_open_ended_query,
)
from memory_inference.memory.retrieval.support_expander import expand_with_support_entries
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery


class HybridBackbone(Protocol):
    name: str

    def index_entries(self, entries: Iterable[MemoryRecord]) -> None: ...

    def rank(
        self,
        query: RuntimeQuery,
        candidates: Iterable[MemoryRecord],
        *,
        score_fn,
        limit: int,
    ) -> list[MemoryRecord]: ...

    def backbone_score(self, entry: MemoryRecord, query: RuntimeQuery, query_context) -> tuple[float, ...]: ...

    def structured_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context,
        *,
        memory_kind: str,
    ) -> tuple[float, ...]: ...

    def evidence_score(
        self,
        entry: MemoryRecord,
        query: RuntimeQuery,
        query_context,
        *,
        anchor_source_ids: set[str],
        anchor_scopes: set[str],
    ) -> tuple[float, ...]: ...

    def open_ended_mode(self) -> str: ...

    def backbone_mode(self) -> str: ...


class HybridCandidateBuilder:
    def __init__(
        self,
        *,
        entity_matches,
        broad_candidate_pool: bool = False,
        prefer_exact_entity_when_available: bool = False,
    ) -> None:
        self._entity_matches = entity_matches
        self._broad_candidate_pool = broad_candidate_pool
        self._prefer_exact_entity_when_available = prefer_exact_entity_when_available

    def structured_candidates(
        self,
        query: RuntimeQuery,
        *,
        episodic_log: Iterable[MemoryRecord],
        current_entries: Iterable[MemoryRecord],
        archive_entries: Iterable[MemoryRecord],
        conflict_entries: Iterable[MemoryRecord],
    ) -> list[MemoryRecord]:
        if query.query_mode.name == "HISTORY":
            entries = [
                entry
                for entry in episodic_log
                if self._candidate_matches_query(entry, query)
            ]
        else:
            episodic_structured = [
                entry
                for entry in episodic_log
                if self.is_structured_fact(entry)
                and self._candidate_matches_query(entry, query)
            ]
            episodic_structured = self._prefer_exact_entity_pool(episodic_structured, query)
            entries = list(current_entries) + list(archive_entries) + list(conflict_entries) + episodic_structured
        return self._dedupe(entries)

    def evidence_candidates(
        self,
        query: RuntimeQuery,
        *,
        episodic_log: Iterable[MemoryRecord],
        anchor_source_ids: set[str],
        anchor_scopes: set[str],
    ) -> list[MemoryRecord]:
        exact_candidates: list[MemoryRecord] = []
        broad_candidates: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for entry in episodic_log:
            if not self._candidate_entity_matches(entry.entity, query.entity):
                continue
            if not self._evidence_attribute_matches(entry, query):
                continue
            if entry.entry_id in seen_ids:
                continue
            if (
                entry.entry_id in anchor_source_ids
                or entry.scope in anchor_scopes
                or entry.attribute in {"dialogue", "event"}
                or entry.attribute == query.attribute
            ):
                seen_ids.add(entry.entry_id)
                if self._entity_matches(entry.entity, query.entity):
                    exact_candidates.append(entry)
                broad_candidates.append(entry)
        return self._prefer_exact_entity_pool(
            exact_candidates if exact_candidates else broad_candidates,
            query,
        )

    def fallback_candidates(
        self,
        query: RuntimeQuery,
        *,
        episodic_log: Iterable[MemoryRecord],
    ) -> list[MemoryRecord]:
        candidates = self._dedupe(
            entry
            for entry in episodic_log
            if self._candidate_matches_query(entry, query)
        )
        return self._prefer_exact_entity_pool(candidates, query)

    @staticmethod
    def anchor_source_ids(entries: Iterable[MemoryRecord]) -> set[str]:
        return {
            source_entry_id
            for entry in entries
            if (source_entry_id := entry.source_entry_id)
        }

    @staticmethod
    def anchor_scopes(entries: Iterable[MemoryRecord]) -> set[str]:
        return {
            entry.scope
            for entry in entries
            if entry.scope and entry.scope != "default"
        }

    @staticmethod
    def is_structured_fact(entry: MemoryRecord) -> bool:
        return entry.source_kind == "structured_fact"

    @staticmethod
    def memory_kind(entry: MemoryRecord) -> str:
        return entry.memory_kind or "state"

    def _candidate_matches_query(self, entry: MemoryRecord, query: RuntimeQuery) -> bool:
        return (
            self._candidate_entity_matches(entry.entity, query.entity)
            and self._candidate_attribute_matches(entry, query)
        )

    def _candidate_entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        if self._broad_candidate_pool:
            return True
        return self._entity_matches(entry_entity, query_entity)

    def _candidate_attribute_matches(self, entry: MemoryRecord, query: RuntimeQuery) -> bool:
        if entry.attribute == query.attribute:
            return True
        if not self._broad_candidate_pool:
            return False
        return (
            self.is_structured_fact(entry)
            or entry.attribute in {"dialogue", "event"}
        )

    def _evidence_attribute_matches(self, entry: MemoryRecord, query: RuntimeQuery) -> bool:
        if entry.attribute in {"dialogue", "event", query.attribute}:
            return True
        return self._broad_candidate_pool and self.is_structured_fact(entry)

    @staticmethod
    def _dedupe(entries: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        combined: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            combined.append(entry)
        return combined

    def _prefer_exact_entity_pool(
        self,
        entries: list[MemoryRecord],
        query: RuntimeQuery,
    ) -> list[MemoryRecord]:
        if not self._prefer_exact_entity_when_available:
            return entries
        if query.entity in {"conversation", "all"}:
            return entries
        exact_matches = [
            entry
            for entry in entries
            if self._entity_matches(entry.entity, query.entity)
        ]
        if exact_matches:
            return exact_matches
        return entries


class HybridMergeStrategy:
    def merge(
        self,
        *,
        state_entries: Iterable[MemoryRecord],
        evidence_entries: Iterable[MemoryRecord],
        top_k: int,
        state_budget: int | None = None,
        evidence_budget: int | None = None,
    ) -> list[MemoryRecord]:
        resolved_state_budget = state_budget if state_budget is not None else max(2, min(3, top_k // 3 + 1))
        resolved_evidence_budget = (
            evidence_budget
            if evidence_budget is not None
            else max(1, top_k - resolved_state_budget)
        )
        merged: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for entry in state_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= resolved_state_budget:
                break

        evidence_count = 0
        for entry in evidence_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            evidence_count += 1
            if evidence_count >= resolved_evidence_budget:
                break

        if len(merged) >= top_k:
            return merged[:top_k]

        for source in (state_entries, evidence_entries):
            for entry in source:
                if entry.entry_id in seen_ids:
                    continue
                seen_ids.add(entry.entry_id)
                merged.append(entry)
                if len(merged) >= top_k:
                    return merged
        return merged


class HybridRanker:
    def __init__(
        self,
        *,
        backbone: HybridBackbone,
        support_history_limit: int,
        entity_matches,
        broad_candidate_pool: bool = False,
        prefer_exact_entity_when_available: bool = False,
        compact_current_state: bool = False,
    ) -> None:
        self.backbone = backbone
        self.support_history_limit = support_history_limit
        self.builder = HybridCandidateBuilder(
            entity_matches=entity_matches,
            broad_candidate_pool=broad_candidate_pool,
            prefer_exact_entity_when_available=prefer_exact_entity_when_available,
        )
        self.merge_strategy = HybridMergeStrategy()
        self._entity_matches = entity_matches
        self._compact_current_state = compact_current_state

    def retrieve(
        self,
        query: RuntimeQuery,
        *,
        episodic_log: list[MemoryRecord],
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
        conflict_entries: list[MemoryRecord],
        top_k: int,
        policy_name: str,
    ) -> RetrievalBundle:
        if is_open_ended_query(query):
            ranked = self.backbone.rank(
                query,
                episodic_log,
                score_fn=lambda entry, query_context: self.backbone.backbone_score(entry, query, query_context),
                limit=max(top_k * 16, 64),
            )
            return RetrievalBundle(
                records=ranked[:top_k],
                debug={
                    "policy": policy_name,
                    "retrieval_mode": self.backbone.open_ended_mode(),
                    "backbone": self.backbone.name,
                },
            )

        structured_candidates = self.builder.structured_candidates(
            query,
            episodic_log=episodic_log,
            current_entries=current_entries,
            archive_entries=archive_entries,
            conflict_entries=conflict_entries,
        )
        if not has_structured_fact_candidates(structured_candidates):
            candidates = self.builder.fallback_candidates(query, episodic_log=episodic_log)
            ranked = self.backbone.rank(
                query,
                candidates,
                score_fn=lambda entry, query_context: self.backbone.backbone_score(entry, query, query_context),
                limit=max(top_k * 16, 64),
            )
            return RetrievalBundle(
                records=ranked[:top_k],
                debug={
                    "policy": policy_name,
                    "retrieval_mode": self.backbone.backbone_mode(),
                    "backbone": self.backbone.name,
                },
            )

        structured_ranked = self.backbone.rank(
            query,
            structured_candidates,
            score_fn=lambda entry, query_context: self.backbone.structured_score(
                entry,
                query,
                query_context,
                memory_kind=self.builder.memory_kind(entry),
            ),
            limit=self._structured_limit(query, top_k),
        )[: self._state_budget(query, top_k)]

        anchor_source_ids = self.builder.anchor_source_ids(structured_ranked)
        anchor_scopes = self.builder.anchor_scopes(structured_ranked)
        evidence_budget = self._evidence_budget(query, top_k, structured_ranked)
        evidence_ranked: list[MemoryRecord] = []
        if evidence_budget > 0:
            evidence_candidates = self.builder.evidence_candidates(
                query,
                episodic_log=episodic_log,
                anchor_source_ids=anchor_source_ids,
                anchor_scopes=anchor_scopes,
            )
            evidence_ranked = self.backbone.rank(
                query,
                evidence_candidates,
                score_fn=lambda entry, query_context: self.backbone.evidence_score(
                    entry,
                    query,
                    query_context,
                    anchor_source_ids=anchor_source_ids,
                    anchor_scopes=anchor_scopes,
                ),
                limit=self._evidence_limit(query, top_k),
            )[:evidence_budget]

        merged = self.merge_strategy.merge(
            state_entries=structured_ranked,
            evidence_entries=evidence_ranked,
            top_k=top_k,
            state_budget=self._state_budget(query, top_k),
            evidence_budget=evidence_budget,
        )
        expanded = expand_with_support_entries(
            merged,
            episodic_log,
            support_limit=self._support_limit(query),
            max_entries=self._expanded_max_entries(query, top_k, merged),
        )
        return RetrievalBundle(
            records=expanded,
            debug={
                "policy": policy_name,
                "retrieval_mode": "hybrid_state_evidence",
                "backbone": self.backbone.name,
            },
        )

    def _structured_limit(self, query: RuntimeQuery, top_k: int) -> int:
        if self._compact_current_state and query.query_mode != QueryMode.HISTORY:
            return max(top_k * 4, 16)
        return max(top_k * 12, 48)

    def _evidence_limit(self, query: RuntimeQuery, top_k: int) -> int:
        if self._compact_current_state and query.query_mode != QueryMode.HISTORY:
            return max(top_k * 4, 16)
        return max(top_k * 16, 64)

    def _state_budget(self, query: RuntimeQuery, top_k: int) -> int:
        if not self._compact_current_state or query.query_mode == QueryMode.HISTORY:
            return min(top_k, max(2, min(3, top_k // 3 + 1)))
        if query.query_mode == QueryMode.CONFLICT_AWARE:
            return min(top_k, 3)
        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            return min(top_k, 3)
        return min(top_k, 2)

    def _evidence_budget(
        self,
        query: RuntimeQuery,
        top_k: int,
        structured_ranked: list[MemoryRecord],
    ) -> int:
        if not self._compact_current_state or query.query_mode == QueryMode.HISTORY:
            return max(1, top_k - self._state_budget(query, top_k))
        if query.query_mode == QueryMode.CONFLICT_AWARE:
            return min(2, max(0, top_k - self._state_budget(query, top_k)))
        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            return 1
        return 1

    def _support_limit(self, query: RuntimeQuery) -> int:
        if not self._compact_current_state or query.query_mode == QueryMode.HISTORY:
            return self.support_history_limit
        if query.query_mode == QueryMode.CONFLICT_AWARE:
            return min(2, self.support_history_limit)
        return 1

    def _expanded_max_entries(
        self,
        query: RuntimeQuery,
        top_k: int,
        merged: list[MemoryRecord],
    ) -> int:
        if not self._compact_current_state or query.query_mode == QueryMode.HISTORY:
            return top_k + self.support_history_limit
        support_limit = self._support_limit(query)
        return min(top_k, len(merged) + support_limit)
