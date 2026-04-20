from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

from memory_inference.agent import AgentRunner
from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.llm.base import BaseReasoner
from memory_inference.metrics import ExperimentMetrics, compute_metrics
from memory_inference.types import BenchmarkBatch, InferenceExample


@dataclass(slots=True)
class ExperimentResult:
    metrics: ExperimentMetrics
    examples: List[InferenceExample]

def evaluate_structured_policy_full(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    batches: Iterable[BenchmarkBatch],
) -> ExperimentResult:
    examples: List[InferenceExample] = []
    snapshot_sizes: List[int] = []
    maintenance_tokens = 0
    maintenance_latency_ms = 0.0
    policy_name = "unknown"
    cached_signature: tuple | None = None
    cached_runner: AgentRunner | None = None
    cached_policy: BaseMemoryPolicy | None = None

    for batch in batches:
        signature = _updates_signature(batch)
        if signature != cached_signature or cached_runner is None or cached_policy is None:
            policy = policy_factory()
            policy_name = policy.name
            runner = AgentRunner(policy=policy, reasoner=reasoner)
            runner.ingest_updates(batch.updates)
            cached_signature = signature
            cached_runner = runner
            cached_policy = policy
            maintenance_tokens += policy.maintenance_tokens
            maintenance_latency_ms += policy.maintenance_latency_ms
        else:
            runner = cached_runner
            policy = cached_policy
            policy_name = policy.name
        batch_examples = runner.answer_queries(batch.queries)
        examples.extend(batch_examples)
        snapshot_sizes.append(policy.snapshot_size())

    metrics = compute_metrics(
        policy_name,
        examples,
        snapshot_sizes=snapshot_sizes,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
    )
    return ExperimentResult(metrics=metrics, examples=examples)


def _updates_signature(batch: BenchmarkBatch) -> tuple:
    return tuple(
        (
            entry.entry_id,
            entry.entity,
            entry.attribute,
            entry.value,
            entry.timestamp,
            entry.session_id,
            entry.confidence,
            entry.importance,
            entry.access_count,
            entry.status.name,
            entry.scope,
            entry.supersedes_id,
            entry.provenance,
            tuple(sorted(entry.metadata.items())),
        )
        for entry in batch.updates
    )
