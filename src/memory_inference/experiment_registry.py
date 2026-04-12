from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from memory_inference.benchmarks.revision_synthetic import RevisionBenchmarkConfig, build_revision_benchmark
from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.exact_match import ExactMatchMemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
from memory_inference.consolidation.strong_retrieval import StrongRetrievalMemoryPolicy
from memory_inference.consolidation.summary_only import SummaryOnlyMemoryPolicy
from memory_inference.consolidation.ablation import (
    NoArchiveConsolidator,
    NoConflictConsolidator,
    NoRevertConsolidator,
    NoScopeConsolidator,
)
from memory_inference.llm.mock_consolidator import MockConsolidator


PolicyFactory = Callable[[], BaseMemoryPolicy]


def default_policy_factories() -> list[PolicyFactory]:
    return [
        AppendOnlyMemoryPolicy,
        RecencySalienceMemoryPolicy,
        SummaryOnlyMemoryPolicy,
        ExactMatchMemoryPolicy,
        StrongRetrievalMemoryPolicy,
        lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
    ]


def _named_odv2(name: str, consolidator):
    """Create an ODV2 policy with a custom display name."""
    policy = OfflineDeltaConsolidationPolicyV2(consolidator=consolidator)
    policy.name = name
    return policy


def ablation_policy_factories() -> list[PolicyFactory]:
    """Policy factories for ablation studies: ODV2 with one component disabled each."""
    return [
        lambda: _named_odv2("offline_delta_v2", MockConsolidator()),
        lambda: _named_odv2("odv2_no_revert", NoRevertConsolidator()),
        lambda: _named_odv2("odv2_no_conflict", NoConflictConsolidator()),
        lambda: _named_odv2("odv2_no_scope", NoScopeConsolidator()),
        lambda: _named_odv2("odv2_no_archive", NoArchiveConsolidator()),
    ]


_ABLATION_NAMES = {
    "odv2_no_revert": NoRevertConsolidator,
    "odv2_no_conflict": NoConflictConsolidator,
    "odv2_no_scope": NoScopeConsolidator,
    "odv2_no_archive": NoArchiveConsolidator,
}


def policy_factory_by_name(name: str) -> PolicyFactory:
    lookup: dict[str, PolicyFactory] = {
        "append_only": AppendOnlyMemoryPolicy,
        "recency_salience": RecencySalienceMemoryPolicy,
        "summary_only": SummaryOnlyMemoryPolicy,
        "exact_match": ExactMatchMemoryPolicy,
        "strong_retrieval": StrongRetrievalMemoryPolicy,
        "offline_delta_v2": lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
    }
    for abl_name, consolidator_cls in _ABLATION_NAMES.items():
        lookup[abl_name] = (lambda n, cls: lambda: _named_odv2(n, cls()))(abl_name, consolidator_cls)
    if name not in lookup:
        raise KeyError(f"Unknown policy preset: {name}")
    return lookup[name]


@dataclass(slots=True)
class ExperimentPreset:
    name: str
    description: str


def default_presets() -> list[ExperimentPreset]:
    return [
        ExperimentPreset(
            name="synthetic_revision",
            description="Structured synthetic validity-maintenance benchmark with oracle state labels.",
        ),
        ExperimentPreset(
            name="longmemeval",
            description="LongMemEval evaluation using the canonical raw or normalized benchmark pipeline.",
        ),
    ]


def build_synthetic_batches():
    return build_revision_benchmark(RevisionBenchmarkConfig())
