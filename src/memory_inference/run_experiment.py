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


def evaluate_structured_policy(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    batches: Iterable[BenchmarkBatch],
) -> ExperimentMetrics:
    return evaluate_structured_policy_full(policy_factory, reasoner, batches).metrics


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

    for batch in batches:
        policy = policy_factory()
        policy_name = policy.name
        runner = AgentRunner(policy=policy, reasoner=reasoner)
        batch_examples = runner.run_batches([batch])
        examples.extend(batch_examples)
        snapshot_sizes.append(policy.snapshot_size())
        maintenance_tokens += policy.maintenance_tokens
        maintenance_latency_ms += policy.maintenance_latency_ms

    metrics = compute_metrics(
        policy_name,
        examples,
        snapshot_sizes=snapshot_sizes,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
    )
    return ExperimentResult(metrics=metrics, examples=examples)
