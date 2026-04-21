from __future__ import annotations

from memory_inference.memory.retrieval.semantic import DenseEncoder
from memory_inference.memory.policies.mem0 import Mem0Policy
from memory_inference.memory.policies.odv2 import ODV2Policy
from memory_inference.memory.retrieval import LexicalBackboneRanker, ODV2DenseBackboneRanker


def build_mem0_policy(
    *,
    name: str,
    encoder: DenseEncoder | None = None,
    write_top_k: int = 10,
    history_enabled: bool = False,
    archive_conflict_enabled: bool = False,
) -> Mem0Policy:
    return Mem0Policy(
        name=name,
        encoder=encoder,
        write_top_k=write_top_k,
        history_enabled=history_enabled,
        archive_conflict_enabled=archive_conflict_enabled,
    )


def mem0_policy(
    *,
    encoder: DenseEncoder | None = None,
    write_top_k: int = 10,
) -> Mem0Policy:
    return build_mem0_policy(
        name="mem0",
        encoder=encoder,
        write_top_k=write_top_k,
    )


def mem0_archive_conflict_policy(
    *,
    encoder: DenseEncoder | None = None,
    write_top_k: int = 10,
) -> Mem0Policy:
    return build_mem0_policy(
        name="mem0_archive_conflict",
        encoder=encoder,
        write_top_k=write_top_k,
        archive_conflict_enabled=True,
    )


def mem0_history_aware_policy(
    *,
    encoder: DenseEncoder | None = None,
    write_top_k: int = 10,
) -> Mem0Policy:
    return build_mem0_policy(
        name="mem0_history_aware",
        encoder=encoder,
        write_top_k=write_top_k,
        history_enabled=True,
    )


def mem0_all_features_policy(
    *,
    encoder: DenseEncoder | None = None,
    write_top_k: int = 10,
) -> Mem0Policy:
    return build_mem0_policy(
        name="mem0_all_features",
        encoder=encoder,
        write_top_k=write_top_k,
        history_enabled=True,
        archive_conflict_enabled=True,
    )


def build_odv2_policy(
    *,
    name: str,
    consolidator,
    importance_threshold: float = 0.1,
    support_history_limit: int = 3,
    hybrid_backbone=None,
) -> ODV2Policy:
    return ODV2Policy(
        name=name,
        consolidator=consolidator,
        importance_threshold=importance_threshold,
        support_history_limit=support_history_limit,
        hybrid_backbone=hybrid_backbone,
    )


def offline_delta_v2_policy(
    *,
    consolidator,
    importance_threshold: float = 0.1,
    support_history_limit: int = 3,
) -> ODV2Policy:
    return build_odv2_policy(
        name="offline_delta_v2",
        consolidator=consolidator,
        importance_threshold=importance_threshold,
        support_history_limit=support_history_limit,
    )


def odv2_strong_policy(
    *,
    consolidator,
    importance_threshold: float = 0.1,
    support_history_limit: int = 3,
) -> ODV2Policy:
    return build_odv2_policy(
        name="odv2_strong",
        consolidator=consolidator,
        importance_threshold=importance_threshold,
        support_history_limit=support_history_limit,
        hybrid_backbone=LexicalBackboneRanker(),
    )


def odv2_dense_policy(
    *,
    consolidator,
    importance_threshold: float = 0.1,
    support_history_limit: int = 3,
    encoder: DenseEncoder | None = None,
) -> ODV2Policy:
    return build_odv2_policy(
        name="odv2_dense",
        consolidator=consolidator,
        importance_threshold=importance_threshold,
        support_history_limit=support_history_limit,
        hybrid_backbone=ODV2DenseBackboneRanker(encoder=encoder),
    )
