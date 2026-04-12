"""Ablation consolidators for component-level analysis.

Each consolidator subclasses MockConsolidator and suppresses exactly one
revision operation, allowing controlled ablation studies that isolate
the contribution of each policy component.
"""
from __future__ import annotations

from typing import Optional, Set

from memory_inference.consolidation.revision_types import RevisionOp
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.types import MemoryEntry


class NoRevertConsolidator(MockConsolidator):
    """Suppress REVERT detection: treat reversions as ordinary revisions."""

    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        op = super().classify_revision(new_entry, existing, prior_values)
        if op == RevisionOp.REVERT:
            return RevisionOp.REVISE
        return op


class NoConflictConsolidator(MockConsolidator):
    """Suppress conflict detection: resolve equal-timestamp contradictions as revisions."""

    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        op = super().classify_revision(new_entry, existing, prior_values)
        if op == RevisionOp.CONFLICT_UNRESOLVED:
            return RevisionOp.REVISE
        return op


class NoScopeConsolidator(MockConsolidator):
    """Suppress scope-split handling: collapse all scopes to default."""

    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        op = super().classify_revision(new_entry, existing, prior_values)
        if op == RevisionOp.SPLIT_SCOPE:
            return RevisionOp.REVISE
        return op


class NoArchiveConsolidator(MockConsolidator):
    """Suppress low-confidence archival: promote all entries regardless of confidence."""

    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        op = super().classify_revision(new_entry, existing, prior_values)
        if op == RevisionOp.LOW_CONFIDENCE:
            return RevisionOp.ADD
        return op
