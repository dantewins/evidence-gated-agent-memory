from __future__ import annotations

from typing import Callable

from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.dense_retrieval import DenseRetrievalMemoryPolicy
from memory_inference.consolidation.exact_match import ExactMatchMemoryPolicy
from memory_inference.consolidation.mem0 import Mem0MemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.odv2_hybrid import ODV2HybridMemoryPolicy
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
from memory_inference.consolidation.strong_retrieval import StrongRetrievalMemoryPolicy
from memory_inference.consolidation.summary_only import SummaryOnlyMemoryPolicy
from memory_inference.llm.mock_consolidator import MockConsolidator

PolicyFactory = Callable[[], BaseMemoryPolicy]


def default_policy_factories() -> list[PolicyFactory]:
    return [
        AppendOnlyMemoryPolicy,
        RecencySalienceMemoryPolicy,
        SummaryOnlyMemoryPolicy,
        ExactMatchMemoryPolicy,
        StrongRetrievalMemoryPolicy,
        DenseRetrievalMemoryPolicy,
        Mem0MemoryPolicy,
        lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
        lambda: ODV2HybridMemoryPolicy(consolidator=MockConsolidator()),
    ]


def policy_factory_by_name(name: str) -> PolicyFactory:
    lookup: dict[str, PolicyFactory] = {
        "append_only": AppendOnlyMemoryPolicy,
        "recency_salience": RecencySalienceMemoryPolicy,
        "summary_only": SummaryOnlyMemoryPolicy,
        "exact_match": ExactMatchMemoryPolicy,
        "strong_retrieval": StrongRetrievalMemoryPolicy,
        "dense_retrieval": DenseRetrievalMemoryPolicy,
        "mem0": Mem0MemoryPolicy,
        "offline_delta_v2": lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
        "odv2_hybrid": lambda: ODV2HybridMemoryPolicy(consolidator=MockConsolidator()),
    }
    if name not in lookup:
        raise KeyError(f"Unknown policy preset: {name}")
    return lookup[name]
