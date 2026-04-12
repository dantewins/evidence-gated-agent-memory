from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from memory_inference.types import MemoryEntry, RetrievalResult
from memory_inference.types import Query


class BaseMemoryPolicy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.maintenance_tokens = 0
        self.maintenance_latency_ms = 0.0
        self.maintenance_calls = 0

    @abstractmethod
    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        raise NotImplementedError

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def maybe_consolidate(self) -> None:
        """Optional hook for policies that consolidate periodically."""

    def snapshot_size(self) -> int:
        return 0
