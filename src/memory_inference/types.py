from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode

MemoryKey = Tuple[str, str]


@dataclass(slots=True)
class MemoryEntry:
    """Atomic memory item stored by the agent."""

    entry_id: str
    entity: str
    attribute: str
    value: str
    timestamp: int
    session_id: str
    confidence: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)
    importance: float = 1.0
    access_count: int = 0
    # Validity-state fields (Phase 1 extensions)
    status: MemoryStatus = field(default=MemoryStatus.ACTIVE)
    scope: str = "default"
    supersedes_id: Optional[str] = None
    provenance: str = ""

    @property
    def key(self) -> MemoryKey:
        return (self.entity, self.attribute)

    def text(self) -> str:
        return (
            f"entity={self.entity}; attribute={self.attribute}; value={self.value}; "
            f"timestamp={self.timestamp}; session={self.session_id}"
        )


@dataclass(slots=True)
class Query:
    query_id: str
    entity: str
    attribute: str
    question: str
    answer: str
    timestamp: int
    session_id: str
    multi_attributes: Tuple[str, ...] = ()
    # Validity-state fields (Phase 1 extensions)
    query_mode: QueryMode = field(default=QueryMode.CURRENT_STATE)
    supports_abstention: bool = False

    @property
    def key(self) -> MemoryKey:
        return (self.entity, self.attribute)


@dataclass(slots=True)
class RetrievalResult:
    entries: Sequence[MemoryEntry]
    debug: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceExample:
    query: Query
    retrieved: Sequence[MemoryEntry]
    prediction: str
    correct: bool
    policy_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    reasoner_name: str = ""


@dataclass(slots=True)
class BenchmarkBatch:
    session_id: str
    updates: List[MemoryEntry]
    queries: List[Query]


@dataclass(slots=True)
class ConsolidatedState:
    current: Dict[MemoryKey, MemoryEntry] = field(default_factory=dict)
    superseded_by: Dict[str, str] = field(default_factory=dict)
    history: Dict[MemoryKey, List[MemoryEntry]] = field(default_factory=dict)

    def get_current(self, key: MemoryKey) -> Optional[MemoryEntry]:
        return self.current.get(key)
