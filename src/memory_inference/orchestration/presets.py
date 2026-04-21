from __future__ import annotations

from typing import Callable

from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies import (
    AppendOnlyMemoryPolicy,
    mem0_all_features_policy,
    mem0_archive_conflict_policy,
    mem0_history_aware_policy,
    mem0_policy,
    odv2_dense_policy,
    odv2_strong_policy,
    offline_delta_v2_policy,
    ExactMatchMemoryPolicy,
    RecencySalienceMemoryPolicy,
    StrongRetrievalMemoryPolicy,
    SummaryOnlyMemoryPolicy,
)
from memory_inference.memory.policies.dense import DenseRetrievalMemoryPolicy
from memory_inference.memory.policies.interface import BaseMemoryPolicy

PolicyFactory = Callable[[], BaseMemoryPolicy]

PAPER_POLICY_NAMES: tuple[str, ...] = (
    "strong_retrieval",
    "dense_retrieval",
    "mem0",
    "mem0_archive_conflict",
    "mem0_history_aware",
    "mem0_all_features",
    "offline_delta_v2",
    "odv2_strong",
    "odv2_dense",
)

DEBUG_POLICY_NAMES: tuple[str, ...] = (
    "append_only",
    "recency_salience",
    "summary_only",
    "exact_match",
)

TEST_POLICY_NAMES: tuple[str, ...] = (
    "append_only",
    "offline_delta_v2",
)

ALL_POLICY_NAMES: tuple[str, ...] = PAPER_POLICY_NAMES + DEBUG_POLICY_NAMES


def all_policy_factories() -> list[PolicyFactory]:
    return policy_factories_for_names(ALL_POLICY_NAMES)


def paper_policy_factories() -> list[PolicyFactory]:
    return policy_factories_for_names(PAPER_POLICY_NAMES)


def debug_policy_factories() -> list[PolicyFactory]:
    return policy_factories_for_names(DEBUG_POLICY_NAMES)


def test_policy_factories() -> list[PolicyFactory]:
    return policy_factories_for_names(TEST_POLICY_NAMES)


def policy_factories_for_names(names: tuple[str, ...] | list[str]) -> list[PolicyFactory]:
    return [policy_factory_by_name(name) for name in names]


def policy_factory_by_name(name: str) -> PolicyFactory:
    lookup: dict[str, PolicyFactory] = {
        "append_only": AppendOnlyMemoryPolicy,
        "recency_salience": RecencySalienceMemoryPolicy,
        "summary_only": SummaryOnlyMemoryPolicy,
        "exact_match": ExactMatchMemoryPolicy,
        "strong_retrieval": StrongRetrievalMemoryPolicy,
        "dense_retrieval": DenseRetrievalMemoryPolicy,
        "mem0": mem0_policy,
        "mem0_archive_conflict": mem0_archive_conflict_policy,
        "mem0_history_aware": mem0_history_aware_policy,
        "mem0_all_features": mem0_all_features_policy,
        "offline_delta_v2": _offline_delta_factory,
        "odv2_strong": _odv2_strong_factory,
        "odv2_dense": _odv2_dense_factory,
    }
    if name not in lookup:
        raise KeyError(f"Unknown policy preset: {name}")
    return lookup[name]


def _offline_delta_factory():
    return offline_delta_v2_policy(consolidator=MockConsolidator())


def _odv2_strong_factory():
    return odv2_strong_policy(consolidator=MockConsolidator())


def _odv2_dense_factory():
    return odv2_dense_policy(consolidator=MockConsolidator())
