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
from memory_inference.benchmarks.conversational_salience import estimate_confidence, estimate_importance
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query

logger = logging.getLogger(__name__)

_QUESTION_TYPE_TO_MODE = {
    "single-session-user": QueryMode.CURRENT_STATE,
    "single-session-assistant": QueryMode.CURRENT_STATE,
    "single-session-preference": QueryMode.CURRENT_STATE,
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
    haystack_dates = item.get("haystack_dates", []) or []
    haystack_session_ids = item.get("haystack_session_ids", []) or []
    question_type = str(item.get("question_type", ""))

    updates: List[MemoryEntry] = []
    turn_groups = _coerce_turn_groups(sessions)
    turn_counter = 0
    for session_idx, session in enumerate(turn_groups):
        source_date = str(haystack_dates[session_idx]) if session_idx < len(haystack_dates) else ""
        source_session_id = (
            str(haystack_session_ids[session_idx])
            if session_idx < len(haystack_session_ids)
            else ""
        )
        scope = source_session_id or f"session_{session_idx}"
        for turn in session:
            role = turn.get("role", "unknown")
            content = str(turn.get("content", ""))
            if not content.strip():
                continue
            importance = estimate_importance(content, speaker=str(role), attribute="dialogue")
            confidence = estimate_confidence(content, speaker=str(role), attribute="dialogue")
            updates.append(MemoryEntry(
                entry_id=f"{qid}-turn-{turn_counter}",
                entity=role,
                attribute="dialogue",
                value=content,
                timestamp=turn_counter,
                session_id=qid,
                scope=scope,
                confidence=confidence,
                importance=importance,
                metadata={
                    "source_date": source_date,
                    "session_label": source_session_id,
                    "speaker": str(role),
                    "has_answer": str(bool(turn.get("has_answer", False))).lower(),
                },
                provenance=f"longmemeval_raw_s{session_idx}",
            ))
            turn_counter += 1

    query_mode = _QUESTION_TYPE_TO_MODE.get(question_type, QueryMode.CURRENT_STATE)
    multi_attrs = tuple(item.get("multi_attributes", []) or [])
    query_entity = _query_entity_for_question_type(question_type)
    supports_abstention = qid.endswith("_abs")

    query = Query(
        query_id=qid,
        entity=query_entity,
        attribute="dialogue",
        question=str(item["question"]),
        answer=str(item["answer"]),
        timestamp=len(sessions),
        session_id=qid,
        multi_attributes=multi_attrs,
        query_mode=query_mode,
        supports_abstention=supports_abstention,
    )

    return BenchmarkBatch(session_id=qid, updates=updates, queries=[query])


def _coerce_turn_groups(raw_sessions: object) -> List[List[dict]]:
    """Accept either a flat turn list or a list of turn lists."""
    if not isinstance(raw_sessions, list):
        raise ValueError("haystack_sessions must be a list")
    if not raw_sessions:
        return []

    first = raw_sessions[0]
    if isinstance(first, dict):
        return [raw_sessions]  # Single flat session.
    if isinstance(first, list):
        normalized: List[List[dict]] = []
        for idx, session in enumerate(raw_sessions):
            if not isinstance(session, list):
                raise ValueError(f"haystack_sessions[{idx}] must be a list of turns")
            normalized.append(session)
        return normalized
    raise ValueError("haystack_sessions must contain turn dicts or lists of turn dicts")


def _query_entity_for_question_type(question_type: str) -> str:
    if question_type == "single-session-assistant":
        return "assistant"
    return "user"
