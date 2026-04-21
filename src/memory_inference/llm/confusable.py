from __future__ import annotations

from typing import Dict, Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.base import BaseReasoner


class ConfusableReasoner(BaseReasoner):
    """Majority-vote reasoner that does NOT filter by entity or attribute.

    Simulates an LLM that reads all retrieved context and can be confused
    by distractors and contradictions. Tiebreaker: most recent entry.
    """

    def answer(self, query: RuntimeQuery, context: Sequence[MemoryRecord]) -> str:
        if not context:
            return "UNKNOWN"

        value_counts: Dict[str, int] = {}
        value_latest: Dict[str, int] = {}
        for entry in context:
            value_counts[entry.value] = value_counts.get(entry.value, 0) + 1
            value_latest[entry.value] = max(
                value_latest.get(entry.value, 0), entry.timestamp
            )

        return max(
            value_counts.keys(),
            key=lambda v: (value_counts[v], value_latest[v]),
        )
