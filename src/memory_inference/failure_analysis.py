"""Failure analysis and bucketing for qualitative paper analysis."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Sequence

from memory_inference.consolidation.revision_types import MemoryStatus
from memory_inference.types import InferenceExample


@dataclass(slots=True)
class FailureRecord:
    scenario_family: str
    failure_mode: str
    query_id: str
    entity: str
    attribute: str
    gold_answer: str
    prediction: str
    retrieved_count: int
    retrieved_summary: str


# Failure mode constants
STALE_RETRIEVAL = "stale_retrieval"
CONFLICT_LEAK = "conflict_leak"
SCOPE_CONFUSION = "scope_confusion"
NOISE_LEAK = "noise_leak"
MISSING_ENTRY = "missing_entry"
REASONER_ERROR = "reasoner_error"


def classify_failure(ex: InferenceExample) -> str:
    """Determine failure mode for an incorrect prediction."""
    retrieved = list(ex.retrieved)
    query = ex.query

    if not retrieved:
        return MISSING_ENTRY

    # Check for noise leak: low-confidence entry surfaced
    for entry in retrieved:
        if entry.confidence < 0.3 and entry.value == ex.prediction:
            return NOISE_LEAK

    # Check for conflict leak: CONFLICTED entries surfaced
    conflicted = [e for e in retrieved if e.status == MemoryStatus.CONFLICTED]
    if conflicted:
        return CONFLICT_LEAK

    # Check for scope confusion: retrieved entry has wrong scope
    scopes = {e.scope for e in retrieved if e.entity == query.entity and e.attribute == query.attribute}
    if len(scopes) > 1:
        return SCOPE_CONFUSION

    # Check for stale retrieval: correct entity/attribute but old value
    matching = [
        e for e in retrieved
        if e.entity == query.entity and e.attribute == query.attribute
    ]
    if matching:
        latest = max(matching, key=lambda e: e.timestamp)
        if latest.value != query.answer:
            return STALE_RETRIEVAL

    # Correct entries retrieved but wrong prediction
    return REASONER_ERROR


def bucket_failures(
    examples: Sequence[InferenceExample],
) -> list[FailureRecord]:
    """Categorize all incorrect predictions by failure mode."""
    failures: list[FailureRecord] = []
    for ex in examples:
        if ex.correct:
            continue
        qid = ex.query.query_id
        dash_parts = qid.split("-", 2)
        family = dash_parts[0].upper() if dash_parts else "unknown"
        mode = classify_failure(ex)
        summary = "; ".join(
            f"{e.entity}.{e.attribute}={e.value}(t={e.timestamp},s={e.status.name})"
            for e in ex.retrieved
        )
        failures.append(FailureRecord(
            scenario_family=family,
            failure_mode=mode,
            query_id=qid,
            entity=ex.query.entity,
            attribute=ex.query.attribute,
            gold_answer=ex.query.answer,
            prediction=ex.prediction,
            retrieved_count=len(ex.retrieved),
            retrieved_summary=summary,
        ))
    return failures


def failure_summary(failures: Sequence[FailureRecord]) -> dict[str, int]:
    """Count failures per mode."""
    counts: dict[str, int] = {}
    for f in failures:
        counts[f.failure_mode] = counts.get(f.failure_mode, 0) + 1
    return counts


def export_failures_csv(failures: Sequence[FailureRecord]) -> str:
    """Export failures to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "scenario_family", "failure_mode", "query_id", "entity",
        "attribute", "gold_answer", "prediction", "retrieved_count",
        "retrieved_summary",
    ])
    for f in failures:
        writer.writerow([
            f.scenario_family, f.failure_mode, f.query_id, f.entity,
            f.attribute, f.gold_answer, f.prediction, f.retrieved_count,
            f.retrieved_summary,
        ])
    return buf.getvalue()
