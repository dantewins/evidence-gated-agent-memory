from __future__ import annotations

from typing import Iterable

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery

_CONVERSATIONAL_ATTRIBUTES = frozenset({"dialogue", "event"})


def is_open_ended_query(query: RuntimeQuery) -> bool:
    return query.attribute in _CONVERSATIONAL_ATTRIBUTES


def has_structured_fact_candidates(entries: Iterable[MemoryRecord]) -> bool:
    return any(entry.source_kind == "structured_fact" for entry in entries)
