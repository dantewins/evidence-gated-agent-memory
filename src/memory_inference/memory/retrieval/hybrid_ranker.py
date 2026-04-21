from __future__ import annotations

from typing import Iterable, Protocol

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
    def __init__(self, *, entity_matches) -> None:
        self._entity_matches = entity_matches

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
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
            ]
        else:
            episodic_structured = [
                entry
                for entry in episodic_log
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
                and self.is_structured_fact(entry)
            ]
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
        candidates: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for entry in episodic_log:
            if not self._entity_matches(entry.entity, query.entity):
                continue
            if entry.attribute not in {"dialogue", "event", query.attribute}:
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
                candidates.append(entry)
        return candidates

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


class HybridMergeStrategy:
    def merge(
        self,
        *,
        state_entries: Iterable[MemoryRecord],
        evidence_entries: Iterable[MemoryRecord],
        top_k: int,
    ) -> list[MemoryRecord]:
        state_budget = max(2, min(3, top_k // 3 + 1))
        evidence_budget = max(1, top_k - state_budget)
        merged: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for entry in state_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= state_budget:
                break

        evidence_count = 0
        for entry in evidence_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            evidence_count += 1
            if evidence_count >= evidence_budget:
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
    ) -> None:
        self.backbone = backbone
        self.support_history_limit = support_history_limit
        self.builder = HybridCandidateBuilder(entity_matches=entity_matches)
        self.merge_strategy = HybridMergeStrategy()
        self._entity_matches = entity_matches

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
            candidates = [
                entry
                for entry in episodic_log
                if self._entity_matches(entry.entity, query.entity)
                and entry.attribute in {query.attribute, "dialogue", "event"}
            ]
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
            limit=max(top_k * 12, 48),
        )[:top_k]

        anchor_source_ids = self.builder.anchor_source_ids(structured_ranked)
        anchor_scopes = self.builder.anchor_scopes(structured_ranked)
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
            limit=max(top_k * 16, 64),
        )[:top_k]

        merged = self.merge_strategy.merge(
            state_entries=structured_ranked,
            evidence_entries=evidence_ranked,
            top_k=top_k,
        )
        expanded = expand_with_support_entries(
            merged,
            episodic_log,
            support_limit=self.support_history_limit,
            max_entries=top_k + self.support_history_limit,
        )
        return RetrievalBundle(
            records=expanded,
            debug={
                "policy": policy_name,
                "retrieval_mode": "hybrid_state_evidence",
                "backbone": self.backbone.name,
            },
        )
