from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List, Sequence

from memory_inference.datasets.normalized_io import NormalizedDataset, NormalizedRecord
from memory_inference.domain.results import EvaluatedCase
from memory_inference.evaluation.manifests import RunManifest, build_manifest, write_manifest
from memory_inference.evaluation.metrics import ExperimentMetrics, compute_metrics
from memory_inference.evaluation.scoring import evaluate_executed_cases
from memory_inference.llm.base import BaseReasoner
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.orchestration.cache import RunnerCache


@dataclass(slots=True)
class ExperimentResult:
    metrics: ExperimentMetrics
    evaluated_cases: List[EvaluatedCase]


@dataclass(slots=True)
class DatasetExperimentResult:
    benchmark: str
    metrics: list[ExperimentMetrics]
    manifest: RunManifest | None = None


def evaluate_structured_policy_full(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    records: Iterable[NormalizedRecord],
    *,
    runner_cache: RunnerCache | None = None,
) -> ExperimentResult:
    evaluated_cases: List[EvaluatedCase] = []
    snapshot_sizes: List[int] = []
    maintenance_tokens = 0
    maintenance_latency_ms = 0.0
    policy_name = "unknown"
    cache = runner_cache if runner_cache is not None else RunnerCache()

    for record in records:
        policy = policy_factory()
        policy_name = policy.name
        runner = cache.build_runner(policy=policy, reasoner=reasoner)
        executed_cases = runner.run_cases_for_context(record.context, record.cases)
        evaluated_cases.extend(evaluate_executed_cases(executed_cases))
        snapshot_sizes.append(policy.snapshot_size())
        maintenance_tokens += policy.maintenance_tokens
        maintenance_latency_ms += policy.maintenance_latency_ms

    metrics = compute_metrics(
        policy_name,
        evaluated_cases,
        snapshot_sizes=snapshot_sizes,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
    )
    return ExperimentResult(metrics=metrics, evaluated_cases=evaluated_cases)


def run_dataset_experiment(
    *,
    benchmark_name: str,
    dataset: NormalizedDataset,
    reasoner: BaseReasoner,
    policy_factories: Sequence[Callable[[], BaseMemoryPolicy]],
    manifest_config: dict[str, object] | None = None,
    manifest_output: str = "",
    include_environment: bool = True,
) -> DatasetExperimentResult:
    metric_rows: list[ExperimentMetrics] = []
    for factory in policy_factories:
        result = evaluate_structured_policy_full(factory, reasoner, dataset.records)
        metric_rows.append(result.metrics)

    manifest = None
    if manifest_output:
        manifest = build_manifest(
            benchmark=benchmark_name,
            reasoner=reasoner.__class__.__name__,
            policy_names=[row.policy_name for row in metric_rows],
            metrics=[asdict(row) for row in metric_rows],
            config=manifest_config or {},
            include_environment=include_environment,
        )
        write_manifest(manifest_output, manifest)

    return DatasetExperimentResult(
        benchmark=benchmark_name,
        metrics=metric_rows,
        manifest=manifest,
    )
