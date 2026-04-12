"""Bootstrap confidence intervals and paired statistical tests.

Uses only stdlib (random.choices). No numpy dependency.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from memory_inference.types import InferenceExample


@dataclass(slots=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    alpha: float


def bootstrap_ci(
    values: Sequence[float | bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for the mean of *values*."""
    vals = [float(v) for v in values]
    if not vals:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0, alpha=alpha)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(
        sum(rng.choices(vals, k=n)) / n for _ in range(n_boot)
    )
    lo = int((alpha / 2) * n_boot)
    hi = int((1 - alpha / 2) * n_boot) - 1
    return ConfidenceInterval(
        mean=sum(vals) / n,
        lower=means[lo],
        upper=means[min(hi, len(means) - 1)],
        alpha=alpha,
    )


def paired_bootstrap_test(
    a: Sequence[bool],
    b: Sequence[bool],
    n_boot: int = 10000,
    seed: int | None = 42,
) -> float:
    """Two-sided paired permutation test via sign-flipping.

    Tests whether the accuracy of *a* differs from *b* on the same queries.
    Under the null hypothesis (no difference), each delta's sign is equally
    likely to be positive or negative.
    """
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length")
    n = len(a)
    if n == 0:
        return 1.0
    deltas = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    observed = abs(sum(deltas) / n)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_boot):
        flipped = sum(d * rng.choice((-1, 1)) for d in deltas) / n
        if abs(flipped) >= observed:
            count += 1
    return count / n_boot


def scenario_family_breakdown(
    examples: Sequence[InferenceExample],
) -> dict[str, list[bool]]:
    """Group per-query correctness by scenario family prefix (S1, S2, ...).

    Extracts the family from query_id which follows the pattern
    ``s{n}-q-user_NN-attribute`` in the synthetic benchmark.
    """
    families: dict[str, list[bool]] = {}
    for ex in examples:
        qid = ex.query.query_id
        # Extract scenario family: "s1-q-..." -> "S1", "s2-q-..." -> "S2", etc.
        dash_parts = qid.split("-", 2)
        family = dash_parts[0].upper() if dash_parts else "unknown"
        families.setdefault(family, []).append(ex.correct)
    return families
