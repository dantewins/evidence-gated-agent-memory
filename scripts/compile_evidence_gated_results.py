from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASELINE_POLICY = "mem0"
TARGET_POLICY = "odv2_mem0_selective"


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: python scripts/compile_evidence_gated_results.py "
            "<cases.jsonl> [<cases.jsonl> ...]",
            file=sys.stderr,
        )
        return 1
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "benchmark",
            "category",
            "scope",
            "n",
            "mem0_accuracy",
            "odv2_accuracy",
            "delta_accuracy",
            "delta_gold_mismatch_exposure",
            "delta_prompt_tokens",
            "wins",
            "losses",
            "verdict",
            "interpretation",
        ],
    )
    writer.writeheader()
    for path_arg in argv:
        for row in _compile_file(Path(path_arg)):
            writer.writerow(row)
    return 0


def _compile_file(path: Path) -> list[dict[str, object]]:
    rows = _load_rows(path)
    benchmark = _first_value(rows, "benchmark", "longmemeval")
    category = _first_value(rows, "category", path.stem.replace("_cases", ""))
    by_policy_case = _by_policy_case(rows)
    baseline_cases = by_policy_case.get(BASELINE_POLICY, {})
    target_cases = by_policy_case.get(TARGET_POLICY, {})
    paired_ids = set(baseline_cases) & set(target_cases)
    if not paired_ids:
        raise ValueError(f"{path} has no paired {BASELINE_POLICY}/{TARGET_POLICY} rows")

    summaries = [
        ("all paired cases", _paired_summary(baseline_cases, target_cases, paired_ids)),
        (
            "ODV2 intervened",
            _paired_summary(
                baseline_cases,
                target_cases,
                {
                    case_id
                    for case_id in paired_ids
                    if _intervened(target_cases[case_id])
                },
            ),
        ),
        (
            "Gold-mismatched same-key state exposed",
            _paired_summary(
                baseline_cases,
                target_cases,
                {
                    case_id
                    for case_id in paired_ids
                    if bool(baseline_cases[case_id].get("stale_state_exposure"))
                },
            ),
        ),
    ]
    return [
        _csv_row(
            benchmark=str(benchmark),
            category=str(category),
            scope=scope,
            summary=summary,
        )
        for scope, summary in summaries
    ]


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def _by_policy_case(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        policy = str(row.get("policy_name", ""))
        if policy not in {BASELINE_POLICY, TARGET_POLICY}:
            continue
        case_id = str(row["case_id"])
        if case_id in grouped[policy]:
            raise ValueError(f"duplicate row for policy={policy} case_id={case_id}")
        grouped[policy][case_id] = row
    return grouped


def _paired_summary(
    baseline_cases: dict[str, dict[str, object]],
    target_cases: dict[str, dict[str, object]],
    case_ids: set[str],
) -> dict[str, object]:
    paired = [(baseline_cases[case_id], target_cases[case_id]) for case_id in sorted(case_ids)]
    if not paired:
        return {
            "n": 0,
            "baseline_acc": None,
            "target_acc": None,
            "delta_acc": 0.0,
            "delta_stale": 0.0,
            "delta_prompt": 0.0,
            "wins": 0,
            "losses": 0,
        }
    baseline_acc = _mean(_as_float_bool(base.get("correct")) for base, _ in paired)
    target_acc = _mean(_as_float_bool(target.get("correct")) for _, target in paired)
    baseline_stale = _mean(_as_float_bool(base.get("stale_state_exposure")) for base, _ in paired)
    target_stale = _mean(_as_float_bool(target.get("stale_state_exposure")) for _, target in paired)
    baseline_prompt = _mean(float(base.get("prompt_tokens", 0.0)) for base, _ in paired)
    target_prompt = _mean(float(target.get("prompt_tokens", 0.0)) for _, target in paired)
    wins = sum(
        1
        for base, target in paired
        if bool(target.get("correct")) and not bool(base.get("correct"))
    )
    losses = sum(
        1
        for base, target in paired
        if not bool(target.get("correct")) and bool(base.get("correct"))
    )
    return {
        "n": len(paired),
        "baseline_acc": baseline_acc,
        "target_acc": target_acc,
        "delta_acc": target_acc - baseline_acc,
        "delta_stale": target_stale - baseline_stale,
        "delta_prompt": target_prompt - baseline_prompt,
        "wins": wins,
        "losses": losses,
    }


def _csv_row(
    *,
    benchmark: str,
    category: str,
    scope: str,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "category": category,
        "scope": scope,
        "n": int(summary["n"]),
        "mem0_accuracy": _format_optional(summary["baseline_acc"]),
        "odv2_accuracy": _format_optional(summary["target_acc"]),
        "delta_accuracy": f"{float(summary['delta_acc']):.3f}",
        "delta_gold_mismatch_exposure": f"{float(summary['delta_stale']):.3f}",
        "delta_prompt_tokens": f"{float(summary['delta_prompt']):.2f}",
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "verdict": _verdict(summary),
        "interpretation": _interpretation(summary),
    }


def _intervened(row: dict[str, object]) -> bool:
    retrieval_mode = str(row.get("retrieval_mode", ""))
    return (
        "guard" in retrieval_mode
        or "compact" in retrieval_mode
        or _as_int(row.get("validity_removed")) > 0
        or _as_int(row.get("support_compacted")) > 0
    )


def _verdict(summary: dict[str, object]) -> str:
    if int(summary["n"]) == 0:
        return "no_cases"
    if float(summary["delta_acc"]) >= 0.0 and int(summary["losses"]) == 0 and float(summary["delta_prompt"]) < 0:
        return "usable"
    if int(summary["losses"]) > 0 or float(summary["delta_acc"]) < 0.0:
        return "not_usable"
    return "neutral"


def _interpretation(summary: dict[str, object]) -> str:
    if int(summary["n"]) == 0:
        return "No paired cases in this slice."
    if _verdict(summary) == "usable":
        return "No paired losses and lower prompt context relative to Mem0-style baseline."
    if _verdict(summary) == "not_usable":
        return "Introduces paired losses or lower accuracy relative to Mem0-style baseline."
    return "Neutral paired result."


def _first_value(rows: list[dict[str, object]], key: str, fallback: str) -> object:
    return next((row[key] for row in rows if row.get(key)), fallback)


def _format_optional(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def _as_float_bool(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
