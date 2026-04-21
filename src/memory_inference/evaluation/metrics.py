from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from memory_inference.domain.results import EvaluatedCase

ABSTAIN_TOKEN = "ABSTAIN"


@dataclass(slots=True)
class ExperimentMetrics:
    policy_name: str
    total_queries: int
    correct_queries: int
    accuracy: float
    abstention_accuracy: float
    proactive_interference_rate: float
    avg_retrieved_items: float
    avg_retrieved_chars: float
    avg_context_tokens: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_snapshot_size: float
    maintenance_tokens: int
    maintenance_latency_ms: float
    amortized_end_to_end_tokens: float
    avg_query_latency_ms: float
    cache_hit_rate: float


def compute_metrics(
    policy_name: str,
    evaluated_cases: Iterable[EvaluatedCase],
    *,
    snapshot_sizes: Optional[Sequence[int]] = None,
    maintenance_tokens: int = 0,
    maintenance_latency_ms: float = 0.0,
) -> ExperimentMetrics:
    rows: List[EvaluatedCase] = list(evaluated_cases)
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    abstention_queries = [row for row in rows if row.case.eval_target.supports_abstention]
    abstention_correct = sum(
        1
        for row in abstention_queries
        if row.prediction == ABSTAIN_TOKEN
    )
    interference_count = sum(1 for row in rows if _has_proactive_interference(row))
    avg_items = sum(len(row.retrieval_bundle.records) for row in rows) / total if total else 0.0
    avg_chars = (
        sum(sum(len(record.text()) for record in row.retrieval_bundle.records) for row in rows) / total
        if total
        else 0.0
    )
    avg_prompt_tokens = (
        sum(row.reader_trace.prompt_tokens for row in rows) / total if total else 0.0
    )
    avg_context_tokens = (
        sum(sum(_token_count(record.text()) for record in row.retrieval_bundle.records) for row in rows) / total
        if total
        else 0.0
    )
    avg_completion_tokens = (
        sum(row.reader_trace.completion_tokens for row in rows) / total if total else 0.0
    )
    avg_query_latency_ms = (
        sum(row.reader_trace.latency_ms for row in rows) / total if total else 0.0
    )
    cache_hit_rate = (
        sum(1 for row in rows if row.reader_trace.cache_hit) / total if total else 0.0
    )
    snapshot_values = list(snapshot_sizes or [])
    avg_snapshot_size = (
        sum(snapshot_values) / len(snapshot_values) if snapshot_values else 0.0
    )
    return ExperimentMetrics(
        policy_name=policy_name,
        total_queries=total,
        correct_queries=correct,
        accuracy=(correct / total) if total else 0.0,
        abstention_accuracy=(
            abstention_correct / len(abstention_queries) if abstention_queries else 0.0
        ),
        proactive_interference_rate=(interference_count / total) if total else 0.0,
        avg_retrieved_items=avg_items,
        avg_retrieved_chars=avg_chars,
        avg_context_tokens=avg_context_tokens,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
        avg_snapshot_size=avg_snapshot_size,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
        amortized_end_to_end_tokens=(
            (avg_prompt_tokens if avg_prompt_tokens > 0 else avg_context_tokens)
            + avg_completion_tokens
            + (maintenance_tokens / total if total else 0.0)
        ),
        avg_query_latency_ms=avg_query_latency_ms,
        cache_hit_rate=cache_hit_rate,
    )


def _has_proactive_interference(row: EvaluatedCase) -> bool:
    mismatched = [
        record for record in row.retrieval_bundle.records
        if record.entity == row.case.runtime_query.entity
        and record.attribute == row.case.runtime_query.attribute
        and record.value != row.case.eval_target.gold_answer
    ]
    return bool(mismatched and row.prediction != row.case.eval_target.gold_answer)


def _token_count(text: str) -> int:
    return len(text.split())

