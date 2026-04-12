from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from memory_inference.agent import AgentRunner
from memory_inference.benchmarks.revision_synthetic import RevisionScenario
from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.experiment_registry import build_synthetic_batches, default_policy_factories
from memory_inference.llm.base import BaseReasoner
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.fixed_prompt_reader import FixedPromptReader
from memory_inference.metrics import ExperimentMetrics, attach_state_metrics, compute_metrics
from memory_inference.types import InferenceExample


@dataclass(slots=True)
class ExperimentResult:
    """Wraps metrics and raw examples for post-hoc analysis (CIs, failure bucketing)."""
    metrics: ExperimentMetrics
    examples: List[InferenceExample]


def main() -> None:
    scenarios = build_synthetic_batches()
    reasoners = build_reasoners()
    policy_factories = build_policy_factories()

    print("Running revised validity-aware synthetic benchmark...\n")
    for reasoner in reasoners:
        print(f"Reader: {reasoner.__class__.__name__}")
        for factory in policy_factories:
            result = evaluate_policy_full(factory, reasoner, scenarios)
            _print_metrics(result.metrics)
        print()


def build_reasoners() -> Sequence[BaseReasoner]:
    return [
        DeterministicValidityReader(),
        FixedPromptReader(),
    ]


def build_policy_factories() -> Sequence[Callable[[], BaseMemoryPolicy]]:
    return list(default_policy_factories())


def evaluate_policy(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    scenarios: Sequence[RevisionScenario],
) -> ExperimentMetrics:
    """Evaluate a policy and return metrics only (backward-compatible)."""
    return evaluate_policy_full(policy_factory, reasoner, scenarios).metrics


def evaluate_policy_full(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    scenarios: Sequence[RevisionScenario],
) -> ExperimentResult:
    """Evaluate a policy and return both metrics and raw examples."""
    examples: List[InferenceExample] = []
    used_policies: List[BaseMemoryPolicy] = []
    snapshot_sizes: List[int] = []
    maintenance_tokens = 0
    maintenance_latency_ms = 0.0

    for scenario in scenarios:
        policy = policy_factory()
        runner = AgentRunner(policy=policy, reasoner=reasoner)
        scenario_examples = runner.run_batches([scenario.batch])
        examples.extend(scenario_examples)
        used_policies.append(policy)
        snapshot_sizes.append(policy.snapshot_size())
        maintenance_tokens += policy.maintenance_tokens
        maintenance_latency_ms += policy.maintenance_latency_ms

    base = compute_metrics(
        used_policies[0].name if used_policies else "unknown",
        examples,
        snapshot_sizes=snapshot_sizes,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
    )
    metrics = attach_state_metrics(base, scenarios, used_policies)
    return ExperimentResult(metrics=metrics, examples=examples)


def _print_metrics(metrics: ExperimentMetrics) -> None:
    print(f"Policy: {metrics.policy_name}")
    print(f"  accuracy                    = {metrics.accuracy:.3f}")
    print(f"  current_state_exact_match   = {metrics.current_state_exact_match:.3f}")
    print(f"  supersession_precision      = {metrics.supersession_precision:.3f}")
    print(f"  supersession_recall         = {metrics.supersession_recall:.3f}")
    print(f"  conflict_detection_f1       = {metrics.conflict_detection_f1:.3f}")
    print(f"  scope_split_accuracy        = {metrics.scope_split_accuracy:.3f}")
    print(f"  abstention_accuracy         = {metrics.abstention_accuracy:.3f}")
    print(f"  temporal_query_accuracy     = {metrics.temporal_query_accuracy:.3f}")
    print(f"  proactive_interference_rate = {metrics.proactive_interference_rate:.3f}")
    print(f"  state_table_edit_distance   = {metrics.state_table_edit_distance:.3f}")
    print(f"  avg_context_tokens          = {metrics.avg_context_tokens:.2f}")
    print(f"  avg_completion_tokens       = {metrics.avg_completion_tokens:.2f}")
    print(f"  avg_snapshot_size           = {metrics.avg_snapshot_size:.2f}")
    print(f"  avg_query_latency_ms        = {metrics.avg_query_latency_ms:.2f}")
    print(f"  cache_hit_rate              = {metrics.cache_hit_rate:.3f}")
    print(f"  maintenance_tokens         = {metrics.maintenance_tokens}")
    print(f"  maintenance_latency_ms     = {metrics.maintenance_latency_ms:.2f}")
    print(f"  amortized_end_to_end_tokens = {metrics.amortized_end_to_end_tokens:.2f}")
    print()


if __name__ == "__main__":
    main()
