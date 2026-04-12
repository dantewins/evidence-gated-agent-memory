"""Canonical normalized benchmark schema for cross-dataset evaluation.

Both LoCoMo and LongMemEval raw adapters normalize into this format,
enabling consistent preprocessing, caching, and comparison.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List

from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.types import BenchmarkBatch

SCHEMA_VERSION = "1.0.0"


@dataclass(slots=True)
class NormalizedRecord:
    schema_version: str
    source_dataset: str
    source_split: str
    source_record_id: str
    batch: BenchmarkBatch
    preprocessing_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDataset:
    schema_version: str = SCHEMA_VERSION
    source_dataset: str = ""
    source_split: str = ""
    records: List[NormalizedRecord] = field(default_factory=list)
    total_sessions: int = 0
    total_updates: int = 0
    total_queries: int = 0
    dropped_records: int = 0
    warnings: List[str] = field(default_factory=list)


def serialize_normalized(dataset: NormalizedDataset, path: str | Path) -> str:
    """Write normalized dataset to JSON and return its SHA-256 hash."""
    data = _json_ready(asdict(dataset))
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_normalized(path: str | Path) -> NormalizedDataset:
    """Load a normalized dataset from JSON."""
    from memory_inference.types import MemoryEntry, Query

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Normalized dataset must be a JSON object")
    raw_records = raw.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Normalized dataset is missing a valid 'records' list")

    records = []
    for rec in raw_records:
        batch_raw = rec["batch"]
        updates = [MemoryEntry(**_restore_memory_entry(u)) for u in batch_raw["updates"]]
        queries = [Query(**_restore_query(q)) for q in batch_raw["queries"]]
        batch = BenchmarkBatch(
            session_id=batch_raw["session_id"],
            updates=updates,
            queries=queries,
        )
        records.append(NormalizedRecord(
            schema_version=rec.get("schema_version", SCHEMA_VERSION),
            source_dataset=rec.get("source_dataset", ""),
            source_split=rec.get("source_split", ""),
            source_record_id=rec.get("source_record_id", ""),
            batch=batch,
            preprocessing_metadata=rec.get("preprocessing_metadata", {}),
        ))
    return NormalizedDataset(
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
        source_dataset=raw.get("source_dataset", ""),
        source_split=raw.get("source_split", ""),
        records=records,
        total_sessions=raw.get("total_sessions", 0),
        total_updates=raw.get("total_updates", 0),
        total_queries=raw.get("total_queries", 0),
        dropped_records=raw.get("dropped_records", 0),
        warnings=raw.get("warnings", []),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _restore_query(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    if "query_mode" in restored and isinstance(restored["query_mode"], str):
        restored["query_mode"] = QueryMode[restored["query_mode"]]
    if "multi_attributes" in restored and isinstance(restored["multi_attributes"], list):
        restored["multi_attributes"] = tuple(restored["multi_attributes"])
    if "attribute_hints" in restored and isinstance(restored["attribute_hints"], list):
        restored["attribute_hints"] = tuple(restored["attribute_hints"])
    return restored


def _restore_memory_entry(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    if "status" in restored and isinstance(restored["status"], str):
        restored["status"] = MemoryStatus[restored["status"]]
    return restored
