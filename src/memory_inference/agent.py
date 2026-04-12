from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.llm.base import BaseReasoner
from memory_inference.open_ended_eval import answers_match
from memory_inference.types import BenchmarkBatch, InferenceExample, MemoryEntry, Query


@dataclass(slots=True)
class AgentRunner:
    policy: BaseMemoryPolicy
    reasoner: BaseReasoner

    def run_batches(self, batches: Iterable[BenchmarkBatch]) -> List[InferenceExample]:
        results: List[InferenceExample] = []
        for batch in batches:
            self.policy.ingest(batch.updates)
            self.policy.maybe_consolidate()
            for query in batch.queries:
                retrieved = self._retrieve(query)
                trace = self.reasoner.answer_with_trace(query, retrieved)
                prediction = trace.answer
                if query.multi_attributes:
                    prediction = self._format_multihop(prediction, query, retrieved)
                results.append(
                    InferenceExample(
                        query=query,
                        retrieved=list(retrieved),
                        prediction=prediction,
                        correct=answers_match(prediction, query.answer),
                        policy_name=self.policy.name,
                        prompt_tokens=trace.prompt_tokens,
                        completion_tokens=trace.completion_tokens,
                        total_tokens=trace.total_tokens,
                        latency_ms=trace.latency_ms,
                        cache_hit=trace.cache_hit,
                        reasoner_name=trace.model_id,
                    )
                )
        return results

    def _retrieve(
        self, query: Query
    ) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = list(self._retrieve_for_query(query).entries)
        for attr in query.multi_attributes:
            subquery = dataclasses.replace(query, attribute=attr, multi_attributes=())
            entries.extend(self._retrieve_for_query(subquery).entries)
        return entries

    def _retrieve_for_query(self, query: Query):
        retrieve_for_query = getattr(self.policy, "retrieve_for_query", None)
        if callable(retrieve_for_query):
            return retrieve_for_query(query)
        retrieve_by_mode = getattr(self.policy, "retrieve_by_mode", None)
        if callable(retrieve_by_mode):
            return retrieve_by_mode(query)
        return self.policy.retrieve(query.entity, query.attribute)

    def _format_multihop(
        self, primary_prediction: str, query: Query, retrieved: Sequence[MemoryEntry]
    ) -> str:
        """For multi-hop queries, join values from each attribute with '+'."""
        parts = [primary_prediction]
        for attr in query.multi_attributes:
            candidates = [e for e in retrieved if e.attribute == attr]
            if candidates:
                parts.append(max(candidates, key=lambda e: e.timestamp).value)
            else:
                parts.append("UNKNOWN")
        return "+".join(parts)
