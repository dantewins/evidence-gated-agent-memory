from __future__ import annotations

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery


def make_record(
    entry_id: str = "e",
    entity: str = "u",
    attribute: str = "a",
    value: str = "v",
    timestamp: int = 1,
    session_id: str = "s",
    **kwargs,
) -> MemoryRecord:
    if "record_id" in kwargs:
        entry_id = kwargs.pop("record_id")
    metadata = dict(kwargs.pop("metadata", {}) or {})
    for field_name in (
        "source_kind",
        "source_attribute",
        "memory_kind",
        "source_entry_id",
        "support_text",
        "speaker",
        "source_date",
        "session_label",
    ):
        if field_name in metadata and field_name not in kwargs:
            kwargs[field_name] = metadata.pop(field_name)
    kwargs["metadata"] = metadata
    return MemoryRecord(
        record_id=entry_id,
        entity=entity,
        attribute=attribute,
        value=value,
        timestamp=timestamp,
        session_id=session_id,
        **kwargs,
    )


def make_query(
    query_id: str = "q",
    entity: str = "u",
    attribute: str = "a",
    question: str = "?",
    timestamp: int = 1,
    session_id: str = "s",
    context_id: str = "ctx",
    answer: str | None = None,
    **kwargs,
) -> RuntimeQuery:
    del answer
    return RuntimeQuery(
        query_id=query_id,
        context_id=context_id,
        entity=entity,
        attribute=attribute,
        question=question,
        timestamp=timestamp,
        session_id=session_id,
        **kwargs,
    )


record = make_record
query = make_query
