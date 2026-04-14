from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from memory_inference.types import InferenceExample

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
    avg_completion_tokens: float
    avg_snapshot_size: float
    maintenance_tokens: int
    maintenance_latency_ms: float
    amortized_end_to_end_tokens: float
    avg_query_latency_ms: float
    cache_hit_rate: float


def compute_metrics(
    policy_name: str,
    examples: Iterable[InferenceExample],
    *,
    snapshot_sizes: Optional[Sequence[int]] = None,
    maintenance_tokens: int = 0,
    maintenance_latency_ms: float = 0.0,
) -> ExperimentMetrics:
    rows: List[InferenceExample] = list(examples)
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    abstention_queries = [row for row in rows if row.query.supports_abstention]
    abstention_correct = sum(
        1
        for row in abstention_queries
        if row.prediction == ABSTAIN_TOKEN
    )
    interference_count = sum(1 for row in rows if _has_proactive_interference(row))
    avg_items = sum(len(row.retrieved) for row in rows) / total if total else 0.0
    avg_chars = (
        sum(sum(len(entry.text()) for entry in row.retrieved) for row in rows) / total if total else 0.0
    )
    avg_prompt_tokens = (
        sum(row.prompt_tokens for row in rows) / total if total else 0.0
    )
    fallback_context_tokens = (
        sum(sum(_token_count(entry.text()) for entry in row.retrieved) for row in rows) / total
        if total
        else 0.0
    )
    avg_context_tokens = avg_prompt_tokens if avg_prompt_tokens > 0 else fallback_context_tokens
    avg_completion_tokens = (
        sum(row.completion_tokens for row in rows) / total if total else 0.0
    )
    avg_query_latency_ms = (
        sum(row.latency_ms for row in rows) / total if total else 0.0
    )
    cache_hit_rate = (
        sum(1 for row in rows if row.cache_hit) / total if total else 0.0
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
        avg_completion_tokens=avg_completion_tokens,
        avg_snapshot_size=avg_snapshot_size,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
        amortized_end_to_end_tokens=(
            avg_context_tokens + avg_completion_tokens + (maintenance_tokens / total if total else 0.0)
        ),
        avg_query_latency_ms=avg_query_latency_ms,
        cache_hit_rate=cache_hit_rate,
    )


def _has_proactive_interference(row: InferenceExample) -> bool:
    mismatched = [
        entry for entry in row.retrieved
        if entry.entity == row.query.entity
        and entry.attribute == row.query.attribute
        and entry.value != row.query.answer
    ]
    return bool(mismatched and row.prediction != row.query.answer)


def _token_count(text: str) -> int:
    return len(text.split())
