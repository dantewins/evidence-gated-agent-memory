"""Raw LoCoMo adapter for the official HuggingFace dataset format.

Ingests the official LoCoMo JSON (snap-research/locomo) and converts
multi-session dialogues into BenchmarkBatch objects. Event summaries
are used as structured memory entries; QA pairs map to Query objects.
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

_CATEGORY_TO_MODE = {
    "single-hop": QueryMode.CURRENT_STATE,
    "multi-hop": QueryMode.CURRENT_STATE,
    "temporal": QueryMode.HISTORY,
    "open-ended": QueryMode.CURRENT_STATE,
    "adversarial": QueryMode.CONFLICT_AWARE,
    "1": QueryMode.CURRENT_STATE,
    "2": QueryMode.HISTORY,
    "3": QueryMode.CURRENT_STATE,
}


def load_raw_locomo(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> List[BenchmarkBatch]:
    """Load official LoCoMo JSON and return BenchmarkBatch list."""
    dataset = preprocess_raw_locomo(path, split=split, limit=limit)
    return [rec.batch for rec in dataset.records]


def preprocess_raw_locomo(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> NormalizedDataset:
    """Preprocess raw LoCoMo into a NormalizedDataset with integrity stats."""
    raw_data = json.loads(Path(path).read_text())
    if not isinstance(raw_data, list):
        raise ValueError("LoCoMo raw format expects a JSON array of records")

    records: List[NormalizedRecord] = []
    dropped = 0
    warnings: List[str] = []
    total_updates = 0
    total_queries = 0

    for idx, item in enumerate(raw_data):
        if limit is not None and len(records) >= limit:
            break
        try:
            batches = _convert_sample(item, idx)
            for batch in batches:
                total_updates += len(batch.updates)
                total_queries += len(batch.queries)
                records.append(NormalizedRecord(
                    schema_version=SCHEMA_VERSION,
                    source_dataset="locomo",
                    source_split=split,
                    source_record_id=f"{item.get('sample_id', f'lc-{idx}')}-{batch.session_id}",
                    batch=batch,
                    preprocessing_metadata={
                        "sample_id": str(item.get("sample_id", "")),
                        "num_sessions": str(_count_sessions(item)),
                    },
                ))
        except (KeyError, ValueError, TypeError) as exc:
            dropped += 1
            warnings.append(f"Sample {idx}: {exc}")
            logger.warning("Dropped LoCoMo sample %d: %s", idx, exc)

    return NormalizedDataset(
        source_dataset="locomo",
        source_split=split,
        records=records,
        total_sessions=len(records),
        total_updates=total_updates,
        total_queries=total_queries,
        dropped_records=dropped,
        warnings=warnings,
    )


def _convert_sample(item: dict, index: int) -> List[BenchmarkBatch]:
    """Convert a single raw LoCoMo sample into one or more BenchmarkBatch objects."""
    sample_id = str(item.get("sample_id", f"lc-{index}"))
    conversation = item.get("conversation", {})
    event_summary = item.get("event_summary", {})
    qa_list = item.get("qa", [])

    # Build memory entries from event summaries (structured per-speaker facts)
    updates: List[MemoryEntry] = []
    ts_counter = 0

    # Extract from event summaries first (more structured)
    for speaker, events in event_summary.items():
        if not isinstance(events, list):
            continue
        for event_text in events:
            updates.append(MemoryEntry(
                entry_id=f"{sample_id}-evt-{ts_counter}",
                entity=str(speaker),
                attribute="event",
                value=str(event_text),
                timestamp=ts_counter,
                session_id=sample_id,
                confidence=0.9,
                metadata={"speaker": str(speaker)},
                provenance="locomo_event_summary",
            ))
            ts_counter += 1

    # Extract from dialogue sessions
    session_keys = sorted(
        k for k in conversation
        if k.startswith("session_") and not k.endswith("_date_time")
    )
    for sess_idx, sess_key in enumerate(session_keys):
        session_data = conversation[sess_key]
        session_date = str(conversation.get(f"{sess_key}_date_time", ""))
        if not isinstance(session_data, list):
            continue
        for turn in session_data:
            speaker = str(turn.get("speaker", "unknown"))
            text = str(turn.get("text", ""))
            if not text.strip():
                continue
            updates.append(MemoryEntry(
                entry_id=f"{sample_id}-{sess_key}-{turn.get('dia_id', ts_counter)}",
                entity=speaker,
                attribute="dialogue",
                value=text,
                timestamp=ts_counter,
                session_id=f"{sample_id}-{sess_key}",
                confidence=1.0,
                metadata={
                    "source_date": session_date,
                    "session_label": sess_key,
                    "speaker": speaker,
                },
                provenance="locomo_dialogue",
            ))
            ts_counter += 1

    # Build one batch per QA pair (each gets the full update context)
    batches: List[BenchmarkBatch] = []
    for qa_idx, qa in enumerate(qa_list):
        category = _normalize_category(qa.get("category", ""))
        query_mode = _CATEGORY_TO_MODE.get(category, QueryMode.CURRENT_STATE)
        query_entity = _infer_query_entity(
            str(qa.get("question", "")),
            speakers=set(event_summary.keys()) | {
                str(turn.get("speaker", ""))
                for sess_key in session_keys
                for turn in conversation.get(sess_key, [])
                if isinstance(turn, dict)
            },
        )
        query = Query(
            query_id=f"{sample_id}-q{qa_idx}",
            entity=query_entity,
            attribute="dialogue",
            question=str(qa["question"]),
            answer=str(qa["answer"]),
            timestamp=ts_counter,
            session_id=sample_id,
            query_mode=query_mode,
            supports_abstention=(category == "adversarial"),
        )
        batches.append(BenchmarkBatch(
            session_id=f"{sample_id}-q{qa_idx}",
            updates=list(updates),
            queries=[query],
        ))

    if not batches:
        # No QA pairs -- return updates-only batch
        batches.append(BenchmarkBatch(
            session_id=sample_id,
            updates=updates,
            queries=[],
        ))

    return batches


def _count_sessions(item: dict) -> int:
    conv = item.get("conversation", {})
    return sum(
        1 for k in conv
        if k.startswith("session_") and not k.endswith("_date_time")
    )


def _normalize_category(raw_category: object) -> str:
    return str(raw_category).strip().lower()


def _infer_query_entity(question: str, speakers: set[str]) -> str:
    question_lower = question.lower()
    matches = [
        speaker
        for speaker in speakers
        if speaker and speaker.lower() in question_lower
    ]
    if len(matches) == 1:
        return matches[0]
    return "conversation"
