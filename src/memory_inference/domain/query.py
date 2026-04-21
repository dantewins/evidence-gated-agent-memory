from __future__ import annotations

from dataclasses import dataclass, field

from memory_inference.domain.enums import QueryMode


@dataclass(slots=True)
class RuntimeQuery:
    query_id: str
    context_id: str
    entity: str
    attribute: str
    question: str
    timestamp: int
    session_id: str
    multi_attributes: tuple[str, ...] = ()
    query_mode: QueryMode = field(default=QueryMode.CURRENT_STATE)
    supports_abstention: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)
