from __future__ import annotations

from typing import Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery


def format_multihop_prediction(
    primary_prediction: str,
    query: RuntimeQuery,
    retrieved: Sequence[MemoryRecord],
) -> str:
    parts = [primary_prediction]
    for attr in query.multi_attributes:
        candidates = [record for record in retrieved if record.attribute == attr]
        if candidates:
            parts.append(max(candidates, key=lambda record: record.timestamp).value)
        else:
            parts.append("UNKNOWN")
    return "+".join(parts)
