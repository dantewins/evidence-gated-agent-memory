from __future__ import annotations


def compact_support_text(text: str, *, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def structured_fact_metadata(
    *,
    base_metadata: dict[str, str],
    source_attribute: str,
    source_entry_id: str,
    support_text: str,
    is_stateful: bool,
) -> dict[str, str]:
    return {
        **base_metadata,
        "source_kind": "structured_fact",
        "source_attribute": source_attribute,
        "source_entry_id": source_entry_id,
        "support_text": support_text,
        "memory_kind": "state" if is_stateful else "event",
    }
