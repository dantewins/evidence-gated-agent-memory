from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvalTarget:
    query_id: str
    gold_answer: str
    benchmark_name: str = ""
    benchmark_category: str = ""
    supports_abstention: bool = False
    scoring_policy: str = "exact_normalized_match"
