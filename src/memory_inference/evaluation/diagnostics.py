from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from memory_inference.domain.results import EvaluatedCase
from memory_inference.evaluation.metrics import (
    case_has_proactive_interference,
    case_has_retrieval_hit,
    case_has_stale_state_exposure,
)
from memory_inference.evaluation.scoring import answers_exact_match, answers_span_match


def evaluated_case_to_diagnostic_row(
    evaluated: EvaluatedCase,
    *,
    benchmark: str,
) -> dict[str, Any]:
    case = evaluated.case
    query = case.runtime_query
    target = case.eval_target
    records = list(evaluated.retrieval_bundle.records)
    return {
        "benchmark": benchmark,
        "policy_name": evaluated.policy_name,
        "case_id": case.case_id,
        "context_id": case.context_id,
        "query_id": query.query_id,
        "query_mode": query.query_mode.name,
        "category": target.benchmark_category or case.metadata.get("question_category", ""),
        "entity": query.entity,
        "attribute": query.attribute,
        "question": query.question,
        "gold_answer": target.gold_answer,
        "prediction": evaluated.prediction,
        "correct": evaluated.correct,
        "exact_match": answers_exact_match(evaluated.prediction, target.gold_answer),
        "span_match": answers_span_match(evaluated.prediction, target.gold_answer),
        "supports_abstention": target.supports_abstention,
        "retrieval_hit": case_has_retrieval_hit(evaluated),
        "stale_state_exposure": case_has_stale_state_exposure(evaluated),
        "proactive_interference": case_has_proactive_interference(evaluated),
        "retrieved_items": len(records),
        "retrieved_context_tokens": sum(_token_count(record.text()) for record in records),
        "prompt_tokens": evaluated.reader_trace.prompt_tokens,
        "completion_tokens": evaluated.reader_trace.completion_tokens,
        "latency_ms": evaluated.reader_trace.latency_ms,
        "cache_hit": evaluated.reader_trace.cache_hit,
        "retrieval_mode": evaluated.retrieval_bundle.debug.get("retrieval_mode", ""),
        "base_retrieval_mode": evaluated.retrieval_bundle.debug.get("base_retrieval_mode", ""),
        "official_mem0_results": _debug_int(evaluated, "official_mem0_results"),
        "official_mem0_stored_count": _debug_int(evaluated, "official_mem0_stored_count"),
        "official_mem0_add_mode": evaluated.retrieval_bundle.debug.get("official_mem0_add_mode", ""),
        "official_mem0_raw_fallback": _debug_int(evaluated, "official_mem0_raw_fallback"),
        "official_mem0_add_batches": _debug_int(evaluated, "official_mem0_add_batches"),
        "official_mem0_add_messages": _debug_int(evaluated, "official_mem0_add_messages"),
        "validity_removed": _debug_int(evaluated, "validity_removed"),
        "validity_appended": _debug_int(evaluated, "validity_appended"),
        "support_compacted": _debug_int(evaluated, "support_compacted"),
        "temporal_pruned": _debug_int(evaluated, "temporal_pruned"),
        "conflict_values": _debug_int(evaluated, "conflict_values"),
        "decision_source": evaluated.retrieval_bundle.debug.get("decision_source", ""),
        "retrieved_records": [
            {
                "record_id": record.record_id,
                "entity": record.entity,
                "attribute": record.attribute,
                "value": record.value,
                "timestamp": record.timestamp,
                "status": record.status.name,
                "scope": record.scope,
                "source_kind": record.source_kind,
                "memory_kind": record.memory_kind,
            }
            for record in records
        ],
    }


def diagnostic_rows(
    evaluated_cases: Iterable[EvaluatedCase],
    *,
    benchmark: str,
) -> list[dict[str, Any]]:
    return [
        evaluated_case_to_diagnostic_row(evaluated, benchmark=benchmark)
        for evaluated in evaluated_cases
    ]


def write_diagnostic_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _token_count(text: str) -> int:
    return len(text.split())


def _debug_int(evaluated: EvaluatedCase, key: str) -> int:
    raw_value = evaluated.retrieval_bundle.debug.get(key, "0")
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0
