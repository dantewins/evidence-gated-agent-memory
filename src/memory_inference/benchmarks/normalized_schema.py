"""Canonical normalized benchmark schema for cross-dataset evaluation.

Both LoCoMo and LongMemEval raw adapters normalize into this format,
enabling consistent preprocessing, caching, and comparison.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

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
    from memory_inference.benchmarks.longmemeval_preprocess import _json_ready

    data = _json_ready(asdict(dataset))
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_normalized(path: str | Path) -> NormalizedDataset:
    """Load a normalized dataset from JSON."""
    from memory_inference.benchmarks.longmemeval_preprocess import (
        _restore_memory_entry,
        _restore_query,
    )

    raw = json.loads(Path(path).read_text())
    records = []
    for rec in raw.get("records", []):
        batch_raw = rec["batch"]
        updates = [_restore_memory_entry(u) for u in batch_raw["updates"]]
        queries = [_restore_query(q) for q in batch_raw["queries"]]
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
