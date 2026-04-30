from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Iterable, List, Sequence

from memory_inference.datasets.normalized_io import NormalizedDataset, NormalizedRecord
from memory_inference.domain.benchmark import ExperimentCase
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.domain.results import EvaluatedCase, ExecutedCase, ReaderTrace
from memory_inference.evaluation.diagnostics import evaluated_case_to_diagnostic_row
from memory_inference.evaluation.manifests import RunManifest, build_manifest, write_manifest
from memory_inference.evaluation.metrics import ExperimentMetrics, compute_metrics
from memory_inference.evaluation.scoring import evaluate_executed_case
from memory_inference.llm.base import BaseReasoner
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.orchestration.cache import RunnerCache
from memory_inference.orchestration.postprocess import format_multihop_prediction


ProgressCallback = Callable[["ProgressEvent"], None]
CaseCallback = Callable[[EvaluatedCase], None]


@dataclass(slots=True)
class ExperimentResult:
    metrics: ExperimentMetrics
    evaluated_cases: List[EvaluatedCase]


@dataclass(slots=True)
class DatasetExperimentResult:
    benchmark: str
    metrics: list[ExperimentMetrics]
    manifest: RunManifest | None = None


@dataclass(slots=True)
class ProgressEvent:
    phase: str
    benchmark: str
    policy_name: str
    policy_index: int
    policy_total: int
    case_index: int = 0
    case_total: int = 0
    case_id: str = ""
    context_id: str = ""
    retrieved_items: int = 0
    correct: bool | None = None
    cache_hit: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reader_batch_size: str = ""
    context_index: int = 0
    context_total: int = 0
    update_count: int = 0
    elapsed_ms: float = 0.0
    snapshot_size: int = 0
    metrics: ExperimentMetrics | None = None


@dataclass(slots=True)
class _PendingCase:
    case: ExperimentCase
    retrieval_bundle: RetrievalBundle
    policy_name: str
    case_index: int


def evaluate_structured_policy_full(
    policy_factory: Callable[[], BaseMemoryPolicy],
    reasoner: BaseReasoner,
    records: Iterable[NormalizedRecord],
    *,
    runner_cache: RunnerCache | None = None,
    benchmark_name: str = "",
    policy_index: int = 1,
    policy_total: int = 1,
    reader_flush_size: int = 0,
    progress_callback: ProgressCallback | None = None,
    case_callback: CaseCallback | None = None,
) -> ExperimentResult:
    evaluated_cases: List[EvaluatedCase] = []
    snapshot_sizes: List[int] = []
    maintenance_tokens = 0
    maintenance_latency_ms = 0.0
    policy_name = "unknown"
    cache = runner_cache if runner_cache is not None else RunnerCache()
    record_list = list(records)
    total_cases = sum(len(record.cases) for record in record_list)
    pending_cases: list[_PendingCase] = []
    case_index = 0

    def flush_pending() -> None:
        if not pending_cases:
            return
        traces = reasoner.answer_many_with_traces(
            [pending.case.runtime_query for pending in pending_cases],
            [pending.retrieval_bundle.records for pending in pending_cases],
        )
        if len(traces) != len(pending_cases):
            raise ValueError(
                f"Reasoner returned {len(traces)} traces for {len(pending_cases)} cases."
            )
        for pending, trace in zip(pending_cases, traces):
            prediction = trace.answer
            if pending.case.runtime_query.multi_attributes:
                prediction = format_multihop_prediction(
                    prediction,
                    pending.case.runtime_query,
                    pending.retrieval_bundle.records,
                )
            evaluated = evaluate_executed_case(
                ExecutedCase(
                    case=pending.case,
                    retrieval_bundle=pending.retrieval_bundle,
                    reader_trace=trace,
                    prediction=prediction,
                    policy_name=pending.policy_name,
                )
            )
            if case_callback is not None:
                case_callback(evaluated)
            evaluated_cases.append(_lightweight_evaluated_case(evaluated))
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="case_finished",
                        benchmark=benchmark_name,
                        policy_name=pending.policy_name,
                        policy_index=policy_index,
                        policy_total=policy_total,
                        case_index=pending.case_index,
                        case_total=total_cases,
                        case_id=pending.case.case_id,
                        context_id=pending.case.context_id,
                        retrieved_items=len(pending.retrieval_bundle.records),
                        correct=evaluated.correct,
                        cache_hit=trace.cache_hit,
                        prompt_tokens=trace.prompt_tokens,
                        completion_tokens=trace.completion_tokens,
                        reader_batch_size=trace.metadata.get("batch_size", ""),
                    )
                )
        pending_cases.clear()

    context_total = len(record_list)
    for context_index, record in enumerate(record_list, start=1):
        policy = policy_factory()
        policy_name = policy.name
        runner = cache.build_runner(policy=policy, reasoner=reasoner)
        if progress_callback is not None:
            progress_callback(
                ProgressEvent(
                    phase="context_started",
                    benchmark=benchmark_name,
                    policy_name=policy_name,
                    policy_index=policy_index,
                    policy_total=policy_total,
                    context_index=context_index,
                    context_total=context_total,
                    context_id=record.context.context_id,
                    update_count=len(record.context.updates),
                )
            )
        started = time.perf_counter()
        runner.prepare_context(record.context)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        snapshot_size = policy.snapshot_size()
        snapshot_sizes.append(snapshot_size)
        maintenance_tokens += policy.maintenance_tokens
        maintenance_latency_ms += policy.maintenance_latency_ms
        if progress_callback is not None:
            progress_callback(
                ProgressEvent(
                    phase="context_finished",
                    benchmark=benchmark_name,
                    policy_name=policy_name,
                    policy_index=policy_index,
                    policy_total=policy_total,
                    context_index=context_index,
                    context_total=context_total,
                    context_id=record.context.context_id,
                    update_count=len(record.context.updates),
                    elapsed_ms=elapsed_ms,
                    snapshot_size=snapshot_size,
                )
            )
        for case in record.cases:
            case_index += 1
            runtime_query = runner._query_with_case_metadata(case)
            retrieval_bundle = runner._retrieve(runtime_query)
            pending_cases.append(
                _PendingCase(
                    case=case,
                    retrieval_bundle=retrieval_bundle,
                    policy_name=policy_name,
                    case_index=case_index,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="case_prepared",
                        benchmark=benchmark_name,
                        policy_name=policy_name,
                        policy_index=policy_index,
                        policy_total=policy_total,
                        case_index=case_index,
                        case_total=total_cases,
                        case_id=case.case_id,
                        context_id=case.context_id,
                        retrieved_items=len(retrieval_bundle.records),
                    )
                )
            if reader_flush_size > 0 and len(pending_cases) >= reader_flush_size:
                flush_pending()
    flush_pending()

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
    cases_output: str = "",
    include_environment: bool = True,
    reader_flush_size: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> DatasetExperimentResult:
    metric_rows: list[ExperimentMetrics] = []
    case_writer = _StreamingDiagnosticWriter(cases_output, benchmark=benchmark_name)
    try:
        for policy_index, factory in enumerate(policy_factories, start=1):
            result = evaluate_structured_policy_full(
                factory,
                reasoner,
                dataset.records,
                benchmark_name=benchmark_name,
                policy_index=policy_index,
                policy_total=len(policy_factories),
                reader_flush_size=reader_flush_size,
                progress_callback=progress_callback,
                case_callback=case_writer.write_case if cases_output else None,
            )
            metric_rows.append(result.metrics)
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="policy_finished",
                        benchmark=benchmark_name,
                        policy_name=result.metrics.policy_name,
                        policy_index=policy_index,
                        policy_total=len(policy_factories),
                        case_total=result.metrics.total_queries,
                        metrics=result.metrics,
                    )
                )
    finally:
        case_writer.close()

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


class _StreamingDiagnosticWriter:
    def __init__(self, path: str | Path, *, benchmark: str) -> None:
        self.benchmark = benchmark
        self.opened = False
        self._handle = None
        if not path:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = output_path.open("w")
        self.opened = True

    def write_case(self, evaluated: EvaluatedCase) -> None:
        if self._handle is None:
            return
        row = evaluated_case_to_diagnostic_row(evaluated, benchmark=self.benchmark)
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None


def _lightweight_evaluated_case(evaluated: EvaluatedCase) -> EvaluatedCase:
    trace = evaluated.reader_trace
    return EvaluatedCase(
        case=evaluated.case,
        retrieval_bundle=evaluated.retrieval_bundle,
        reader_trace=ReaderTrace(
            answer=trace.answer,
            model_id=trace.model_id,
            prompt="",
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            latency_ms=trace.latency_ms,
            cache_hit=trace.cache_hit,
            raw_output="",
            metadata=dict(trace.metadata),
        ),
        prediction=evaluated.prediction,
        correct=evaluated.correct,
        policy_name=evaluated.policy_name,
    )
