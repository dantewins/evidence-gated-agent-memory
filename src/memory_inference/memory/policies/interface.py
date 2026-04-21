from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from typing import Any, Protocol

from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery


class BaseMemoryPolicy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.maintenance_tokens = 0
        self.maintenance_latency_ms = 0.0
        self.maintenance_calls = 0

    @abstractmethod
    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        raise NotImplementedError

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def maybe_consolidate(self) -> None:
        """Optional hook for policies that consolidate after ingestion."""

    def snapshot_size(self) -> int:
        return 0


class ScoreStrategy(Protocol):
    def __call__(self, entry: MemoryRecord, entity: str, attribute: str) -> tuple[Any, ...]:
        ...
