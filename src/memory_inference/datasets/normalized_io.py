"""Canonical normalized benchmark schema for compiled experiment datasets."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.targets import EvalTarget

SCHEMA_VERSION = "2.0.0"
ANNOTATION_VERSION = "phase3-v1"
COMPILER_VERSION = "phase3-v1"
BENCHMARK_SOURCE_VERSION = "raw-v1"


@dataclass(slots=True)
class NormalizedRecord:
    schema_version: str
    source_dataset: str
    source_split: str
    source_record_id: str
    context: ExperimentContext
    cases: list[ExperimentCase]
    preprocessing_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDataset:
    schema_version: str = SCHEMA_VERSION
    source_dataset: str = ""
    source_split: str = ""
    records: list[NormalizedRecord] = field(default_factory=list)
    total_contexts: int = 0
    total_updates: int = 0
    total_cases: int = 0
    dropped_records: int = 0
    warnings: list[str] = field(default_factory=list)
    benchmark_source_version: str = BENCHMARK_SOURCE_VERSION
    annotation_version: str = ANNOTATION_VERSION
    compiler_version: str = COMPILER_VERSION

    @property
    def total_sessions(self) -> int:
        return self.total_contexts

    @property
    def total_queries(self) -> int:
        return self.total_cases


def serialize_normalized(dataset: NormalizedDataset, path: str | Path) -> str:
    data = _json_ready(asdict(dataset))
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_normalized(path: str | Path) -> NormalizedDataset:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Normalized dataset must be a JSON object")
    raw_records = raw.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Normalized dataset is missing a valid 'records' list")

    records: list[NormalizedRecord] = []
    for rec in raw_records:
        context_raw = rec.get("context")
        case_list_raw = rec.get("cases")
        if not isinstance(context_raw, dict):
            raise ValueError("Normalized record is missing a valid 'context' object")
        if not isinstance(case_list_raw, list):
            raise ValueError("Normalized record is missing a valid 'cases' list")

        context = ExperimentContext(
            context_id=str(context_raw["context_id"]),
            session_id=str(context_raw["session_id"]),
            updates=[_restore_memory_record(payload) for payload in context_raw.get("updates", [])],
            metadata=_restore_string_dict(context_raw.get("metadata", {})),
        )
        cases = [
            ExperimentCase(
                case_id=str(case_raw["case_id"]),
                context_id=str(case_raw["context_id"]),
                runtime_query=_restore_runtime_query(case_raw["runtime_query"]),
                eval_target=_restore_eval_target(case_raw["eval_target"]),
                metadata=_restore_string_dict(case_raw.get("metadata", {})),
            )
            for case_raw in case_list_raw
        ]
        records.append(
            NormalizedRecord(
                schema_version=str(rec.get("schema_version", SCHEMA_VERSION)),
                source_dataset=str(rec.get("source_dataset", "")),
                source_split=str(rec.get("source_split", "")),
                source_record_id=str(rec.get("source_record_id", "")),
                context=context,
                cases=cases,
                preprocessing_metadata=_restore_string_dict(rec.get("preprocessing_metadata", {})),
            )
        )

    total_contexts = int(raw.get("total_contexts", raw.get("total_sessions", len(records))))
    total_cases = int(raw.get("total_cases", raw.get("total_queries", sum(len(record.cases) for record in records))))
    return NormalizedDataset(
        schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        source_dataset=str(raw.get("source_dataset", "")),
        source_split=str(raw.get("source_split", "")),
        records=records,
        total_contexts=total_contexts,
        total_updates=int(raw.get("total_updates", 0)),
        total_cases=total_cases,
        dropped_records=int(raw.get("dropped_records", 0)),
        warnings=[str(item) for item in raw.get("warnings", [])],
        benchmark_source_version=str(raw.get("benchmark_source_version", BENCHMARK_SOURCE_VERSION)),
        annotation_version=str(raw.get("annotation_version", ANNOTATION_VERSION)),
        compiler_version=str(raw.get("compiler_version", COMPILER_VERSION)),
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


def _restore_runtime_query(payload: dict[str, Any]) -> RuntimeQuery:
    query_mode = payload.get("query_mode", QueryMode.CURRENT_STATE)
    if isinstance(query_mode, str):
        query_mode = QueryMode[query_mode]
    multi_attributes = payload.get("multi_attributes", ())
    if isinstance(multi_attributes, list):
        multi_attributes = tuple(str(item) for item in multi_attributes)
    return RuntimeQuery(
        query_id=str(payload["query_id"]),
        context_id=str(payload["context_id"]),
        entity=str(payload["entity"]),
        attribute=str(payload["attribute"]),
        question=str(payload["question"]),
        timestamp=int(payload["timestamp"]),
        session_id=str(payload["session_id"]),
        multi_attributes=multi_attributes,
        query_mode=query_mode,
        supports_abstention=bool(payload.get("supports_abstention", False)),
    )


def _restore_eval_target(payload: dict[str, Any]) -> EvalTarget:
    return EvalTarget(
        query_id=str(payload["query_id"]),
        gold_answer=str(payload["gold_answer"]),
        benchmark_name=str(payload.get("benchmark_name", "")),
        benchmark_category=str(payload.get("benchmark_category", "")),
        supports_abstention=bool(payload.get("supports_abstention", False)),
        scoring_policy=str(payload.get("scoring_policy", "exact_normalized_match")),
    )


def _restore_memory_record(payload: dict[str, Any]) -> MemoryRecord:
    status = payload.get("status", MemoryStatus.ACTIVE)
    if isinstance(status, str):
        status = MemoryStatus[status]
    metadata = _restore_string_dict(payload.get("metadata", {}))
    return MemoryRecord(
        record_id=str(payload["record_id"]),
        entity=str(payload["entity"]),
        attribute=str(payload["attribute"]),
        value=str(payload["value"]),
        timestamp=int(payload["timestamp"]),
        session_id=str(payload["session_id"]),
        confidence=float(payload.get("confidence", 1.0)),
        metadata=metadata,
        importance=float(payload.get("importance", 1.0)),
        access_count=int(payload.get("access_count", 0)),
        status=status,
        scope=str(payload.get("scope", "default")),
        supersedes_id=payload.get("supersedes_id"),
        provenance=str(payload.get("provenance", "")),
        source_kind=str(payload.get("source_kind", "")),
        source_attribute=str(payload.get("source_attribute", "")),
        memory_kind=str(payload.get("memory_kind", "")),
        source_entry_id=payload.get("source_entry_id"),
        support_text=str(payload.get("support_text", "")),
        speaker=str(payload.get("speaker", "")),
        source_date=str(payload.get("source_date", "")),
        session_label=str(payload.get("session_label", "")),
    )


def _restore_string_dict(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}
