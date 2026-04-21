from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from memory_inference.domain.enums import MemoryStatus

MemoryKey = Tuple[str, str]


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    entity: str
    attribute: str
    value: str
    timestamp: int
    session_id: str
    confidence: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)
    importance: float = 1.0
    access_count: int = 0
    status: MemoryStatus = field(default=MemoryStatus.ACTIVE)
    scope: str = "default"
    supersedes_id: Optional[str] = None
    provenance: str = ""
    source_kind: str = ""
    source_attribute: str = ""
    memory_kind: str = ""
    source_entry_id: Optional[str] = None
    support_text: str = ""
    speaker: str = ""
    source_date: str = ""
    session_label: str = ""

    @property
    def entry_id(self) -> str:
        return self.record_id

    @property
    def key(self) -> MemoryKey:
        return (self.entity, self.attribute)

    def text(self) -> str:
        return (
            f"entity={self.entity}; attribute={self.attribute}; value={self.value}; "
            f"timestamp={self.timestamp}; session={self.session_id}"
        )


@dataclass(slots=True)
class RetrievalBundle:
    records: Sequence[MemoryRecord]
    debug: Dict[str, str] = field(default_factory=dict)

    @property
    def entries(self) -> Sequence[MemoryRecord]:
        return self.records
