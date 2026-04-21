from __future__ import annotations

from typing import Callable, Iterable, Sequence

from memory_inference.memory.retrieval.lexical_ranker import (
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery


def rerank_structured_candidates(
    entries: Iterable[MemoryRecord],
    query: RuntimeQuery,
    *,
    top_k: int,
    policy_name: str,
    score_fn: Callable[[MemoryRecord], tuple[float, ...] | float],
    support_entries: Iterable[MemoryRecord],
    shortlist_limit: int,
    support_limit: int = 2,
) -> RetrievalBundle:
    shortlisted = shortlist_open_ended_candidates(
        entries,
        query,
        score_fn=score_fn,
        limit=shortlist_limit,
    )
    base = lexical_retrieval(
        shortlisted,
        query,
        top_k=top_k,
        policy_name=policy_name,
        secondary_score_fn=score_fn,
    )
    expanded = expand_with_support_entries(
        base.entries,
        support_entries,
        support_limit=support_limit,
        max_entries=top_k + support_limit,
    )
    return RetrievalBundle(
        records=expanded,
        debug={
            **base.debug,
            "retrieval_mode": "structured_fact_rerank",
        },
    )


def expand_with_support_entries(
    entries: Sequence[MemoryRecord],
    support_entries: Iterable[MemoryRecord],
    *,
    support_limit: int = 2,
    max_entries: int | None = None,
) -> list[MemoryRecord]:
    result: list[MemoryRecord] = []
    seen_ids: set[str] = set()
    support_by_id = {entry.entry_id: entry for entry in support_entries}

    for entry in entries:
        if entry.entry_id in seen_ids:
            continue
        seen_ids.add(entry.entry_id)
        result.append(entry)

    supports_added = 0
    for entry in entries:
        source_entry_id = entry.source_entry_id
        if not source_entry_id:
            continue
        support = support_by_id.get(source_entry_id)
        if support is None or support.entry_id in seen_ids:
            continue
        seen_ids.add(support.entry_id)
        result.append(support)
        supports_added += 1
        if supports_added >= support_limit:
            break

    if max_entries is not None:
        return result[:max_entries]
    return result
