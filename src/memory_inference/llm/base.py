from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.domain.results import ReaderTrace

ReasonerTrace = ReaderTrace


class BaseReasoner(ABC):
    """Frozen inference interface.

    A real implementation can call an API or local model. For now we keep the base
    model logically frozen and only vary the memory policy.
    """

    @abstractmethod
    def answer(self, query: RuntimeQuery, context: Sequence[MemoryRecord]) -> str:
        raise NotImplementedError

    def answer_with_trace(
        self,
        query: RuntimeQuery,
        context: Sequence[MemoryRecord],
    ) -> ReasonerTrace:
        started = time.perf_counter()
        answer = self.answer(query, context)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ReasonerTrace(
            answer=answer,
            model_id=self.__class__.__name__,
            raw_output=answer,
            latency_ms=latency_ms,
        )
