"""Raw LongMemEval adapter for the official HuggingFace dataset format.

Ingests the official LongMemEval JSON (xiaowu0162/longmemeval) and converts
dialogue sessions into BenchmarkBatch objects without LLM-based fact extraction.
Each dialogue turn becomes a MemoryEntry; QA pairs map to Query objects.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from memory_inference.benchmarks.normalized_schema import (
    SCHEMA_VERSION,
    NormalizedDataset,
    NormalizedRecord,
)
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query

logger = logging.getLogger(__name__)

_QUESTION_TYPE_TO_MODE = {
    "single-session-user": QueryMode.CURRENT_STATE,
    "single-session-assistant": QueryMode.CURRENT_STATE,
    "multi-session": QueryMode.CURRENT_STATE,
    "temporal-reasoning": QueryMode.HISTORY,
    "knowledge-update": QueryMode.CURRENT_STATE,
    "temporal-ordering": QueryMode.HISTORY,
}


def load_raw_longmemeval(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> List[BenchmarkBatch]:
    """Load official LongMemEval JSON and return BenchmarkBatch list."""
    dataset = preprocess_raw_longmemeval(path, split=split, limit=limit)
    return [rec.batch for rec in dataset.records]


def preprocess_raw_longmemeval(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> NormalizedDataset:
    """Preprocess raw LongMemEval into a NormalizedDataset with integrity stats."""
    raw_data = json.loads(Path(path).read_text())
    if not isinstance(raw_data, list):
        raise ValueError("LongMemEval raw format expects a JSON array of records")

    records: List[NormalizedRecord] = []
    dropped = 0
    warnings: List[str] = []
    total_updates = 0
    total_queries = 0

    for idx, item in enumerate(raw_data):
        if limit is not None and len(records) >= limit:
            break
        try:
            batch = _convert_record(item, idx)
            total_updates += len(batch.updates)
            total_queries += len(batch.queries)
            records.append(NormalizedRecord(
                schema_version=SCHEMA_VERSION,
                source_dataset="longmemeval",
                source_split=split,
                source_record_id=str(item.get("question_id", f"lme-{idx}")),
                batch=batch,
                preprocessing_metadata={
                    "question_type": str(item.get("question_type", "")),
                    "num_haystack_sessions": str(len(item.get("haystack_sessions", []))),
                },
            ))
        except (KeyError, ValueError, TypeError) as exc:
            dropped += 1
            warnings.append(f"Record {idx}: {exc}")
            logger.warning("Dropped LongMemEval record %d: %s", idx, exc)

    return NormalizedDataset(
        source_dataset="longmemeval",
        source_split=split,
        records=records,
        total_sessions=len(records),
        total_updates=total_updates,
        total_queries=total_queries,
        dropped_records=dropped,
        warnings=warnings,
    )


def _convert_record(item: dict, index: int) -> BenchmarkBatch:
    """Convert a single raw LongMemEval record to a BenchmarkBatch."""
    qid = str(item.get("question_id", f"lme-{index}"))
    sessions = item.get("haystack_sessions", [])
    question_type = str(item.get("question_type", ""))

    updates: List[MemoryEntry] = []
    for turn_idx, turn in enumerate(sessions):
        role = turn.get("role", "unknown")
        content = str(turn.get("content", ""))
        if not content.strip():
            continue
        updates.append(MemoryEntry(
            entry_id=f"{qid}-turn-{turn_idx}",
            entity=role,
            attribute="dialogue",
            value=content,
            timestamp=turn_idx,
            session_id=qid,
            confidence=1.0,
            provenance="longmemeval_raw",
        ))

    query_mode = _QUESTION_TYPE_TO_MODE.get(question_type, QueryMode.CURRENT_STATE)
    multi_attrs = tuple(item.get("multi_attributes", []) or [])

    query = Query(
        query_id=qid,
        entity="user",
        attribute="dialogue",
        question=str(item["question"]),
        answer=str(item["answer"]),
        timestamp=len(sessions),
        session_id=qid,
        multi_attributes=multi_attrs,
        query_mode=query_mode,
        supports_abstention=False,
    )

    return BenchmarkBatch(session_id=qid, updates=updates, queries=[query])
