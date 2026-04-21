from __future__ import annotations

import re

from memory_inference.domain.results import EvaluatedCase, ExecutedCase


def answers_match(prediction: str, gold: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if not normalized_prediction or not normalized_gold:
        return False
    return normalized_prediction == normalized_gold


def normalize_answer(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def evaluate_executed_case(executed: ExecutedCase) -> EvaluatedCase:
    return EvaluatedCase(
        case=executed.case,
        retrieval_bundle=executed.retrieval_bundle,
        reader_trace=executed.reader_trace,
        prediction=executed.prediction,
        correct=answers_match(executed.prediction, executed.case.eval_target.gold_answer),
        policy_name=executed.policy_name,
    )


def evaluate_executed_cases(executed_cases: list[ExecutedCase]) -> list[EvaluatedCase]:
    return [evaluate_executed_case(executed) for executed in executed_cases]
