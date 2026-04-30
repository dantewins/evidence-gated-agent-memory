from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


Row = dict[str, object]
BASELINE_POLICY = "mem0"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compile reviewer-facing paired results for ODV2 ablations."
    )
    parser.add_argument("cases", nargs="+", help="Per-case diagnostic JSONL files.")
    parser.add_argument("--baseline-policy", default=BASELINE_POLICY)
    parser.add_argument(
        "--target-policy",
        action="append",
        default=[],
        help="Target policy to compare. Defaults to every non-baseline policy in the files.",
    )
    parser.add_argument(
        "--audit-output",
        default="",
        help="Optional JSONL output for intervention/gold-mismatch/discordant case audit.",
    )
    args = parser.parse_args(argv)

    try:
        summaries, audit_rows = compile_paths(
            [Path(path) for path in args.cases],
            baseline_policy=args.baseline_policy,
            target_policies=set(args.target_policy),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_summary_csv(summaries, sys.stdout)
    if args.audit_output:
        write_audit_jsonl(Path(args.audit_output), audit_rows)
    return 0


def compile_paths(
    paths: list[Path],
    *,
    baseline_policy: str = BASELINE_POLICY,
    target_policies: set[str] | None = None,
) -> tuple[list[Row], list[Row]]:
    summaries: list[Row] = []
    audit_rows: list[Row] = []
    requested_targets = target_policies or set()
    for path in paths:
        file_summaries, file_audit = _compile_file(
            path,
            baseline_policy=baseline_policy,
            target_policies=requested_targets,
        )
        summaries.extend(file_summaries)
        audit_rows.extend(file_audit)
    return summaries, audit_rows


def write_summary_csv(rows: list[Row], handle) -> None:
    fieldnames = [
        "benchmark",
        "category",
        "baseline_policy",
        "target_policy",
        "scope",
        "n",
        "baseline_accuracy",
        "target_accuracy",
        "delta_accuracy",
        "baseline_gold_mismatch_exposure",
        "target_gold_mismatch_exposure",
        "delta_gold_mismatch_exposure",
        "baseline_prompt_tokens",
        "target_prompt_tokens",
        "delta_prompt_tokens",
        "wins",
        "losses",
        "both_wrong",
        "target_interventions",
        "verdict",
        "reviewer_note",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def write_audit_jsonl(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _compile_file(
    path: Path,
    *,
    baseline_policy: str,
    target_policies: set[str],
) -> tuple[list[Row], list[Row]]:
    rows = _load_rows(path)
    benchmark = _first_value(rows, "benchmark", "longmemeval")
    category = _first_value(rows, "category", path.stem.replace("_cases", ""))
    by_policy_case = _by_policy_case(rows)
    if baseline_policy not in by_policy_case:
        raise ValueError(f"{path} has no baseline policy rows for {baseline_policy!r}")

    available_targets = set(by_policy_case) - {baseline_policy}
    selected_targets = sorted(target_policies & available_targets if target_policies else available_targets)
    if target_policies and not selected_targets:
        requested = ", ".join(sorted(target_policies))
        raise ValueError(f"{path} has none of the requested target policies: {requested}")

    summaries: list[Row] = []
    audit_rows: list[Row] = []
    for target_policy in selected_targets:
        baseline_cases = by_policy_case[baseline_policy]
        target_cases = by_policy_case[target_policy]
        paired_ids = set(baseline_cases) & set(target_cases)
        if not paired_ids:
            continue
        scopes = [
            ("all paired cases", paired_ids),
            (
                "Predeclared validity-sensitive union",
                {
                    case_id
                    for case_id in paired_ids
                    if _predeclared_validity_sensitive(baseline_cases[case_id])
                },
            ),
            (
                "Current-state same-key evidence retrieved",
                {
                    case_id
                    for case_id in paired_ids
                    if _has_current_state_same_key_evidence(baseline_cases[case_id])
                },
            ),
            (
                "ODV2 intervened",
                {
                    case_id
                    for case_id in paired_ids
                    if _intervened(target_cases[case_id])
                },
            ),
            (
                "Gold-mismatched same-key state exposed",
                {
                    case_id
                    for case_id in paired_ids
                    if bool(baseline_cases[case_id].get("stale_state_exposure"))
                },
            ),
            (
                "Same-key state conflict exposed",
                {
                    case_id
                    for case_id in paired_ids
                    if _has_same_key_state_conflict(baseline_cases[case_id])
                },
            ),
        ]
        for scope, case_ids in scopes:
            summary = _paired_summary(baseline_cases, target_cases, case_ids)
            summaries.append(
                _summary_csv_row(
                    benchmark=str(benchmark),
                    category=str(category),
                    baseline_policy=baseline_policy,
                    target_policy=target_policy,
                    scope=scope,
                    summary=summary,
                )
            )
        audit_rows.extend(
            _audit_rows(
                benchmark=str(benchmark),
                category=str(category),
                baseline_policy=baseline_policy,
                target_policy=target_policy,
                baseline_cases=baseline_cases,
                target_cases=target_cases,
                case_ids=paired_ids,
            )
        )
    return summaries, audit_rows


def _load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    rows: list[Row] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def _by_policy_case(rows: Iterable[Row]) -> dict[str, dict[str, Row]]:
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        policy = str(row.get("policy_name", ""))
        if not policy:
            continue
        case_id = str(row["case_id"])
        if case_id in grouped[policy]:
            raise ValueError(f"duplicate row for policy={policy} case_id={case_id}")
        grouped[policy][case_id] = row
    return grouped


def _paired_summary(
    baseline_cases: dict[str, Row],
    target_cases: dict[str, Row],
    case_ids: set[str],
) -> Row:
    paired = [(baseline_cases[case_id], target_cases[case_id]) for case_id in sorted(case_ids)]
    if not paired:
        return {
            "n": 0,
            "baseline_acc": None,
            "target_acc": None,
            "delta_acc": 0.0,
            "baseline_gold_mismatch": None,
            "target_gold_mismatch": None,
            "delta_gold_mismatch": 0.0,
            "baseline_prompt": None,
            "target_prompt": None,
            "delta_prompt": 0.0,
            "wins": 0,
            "losses": 0,
            "both_wrong": 0,
            "target_interventions": 0,
        }

    baseline_acc = _mean(_as_float_bool(base.get("correct")) for base, _ in paired)
    target_acc = _mean(_as_float_bool(target.get("correct")) for _, target in paired)
    baseline_gold_mismatch = _mean(
        _as_float_bool(base.get("stale_state_exposure")) for base, _ in paired
    )
    target_gold_mismatch = _mean(
        _as_float_bool(target.get("stale_state_exposure")) for _, target in paired
    )
    baseline_prompt = _mean(float(base.get("prompt_tokens", 0.0)) for base, _ in paired)
    target_prompt = _mean(float(target.get("prompt_tokens", 0.0)) for _, target in paired)
    wins = sum(1 for base, target in paired if bool(target.get("correct")) and not bool(base.get("correct")))
    losses = sum(1 for base, target in paired if not bool(target.get("correct")) and bool(base.get("correct")))
    both_wrong = sum(1 for base, target in paired if not bool(target.get("correct")) and not bool(base.get("correct")))
    interventions = sum(1 for _, target in paired if _intervened(target))
    return {
        "n": len(paired),
        "baseline_acc": baseline_acc,
        "target_acc": target_acc,
        "delta_acc": target_acc - baseline_acc,
        "baseline_gold_mismatch": baseline_gold_mismatch,
        "target_gold_mismatch": target_gold_mismatch,
        "delta_gold_mismatch": target_gold_mismatch - baseline_gold_mismatch,
        "baseline_prompt": baseline_prompt,
        "target_prompt": target_prompt,
        "delta_prompt": target_prompt - baseline_prompt,
        "wins": wins,
        "losses": losses,
        "both_wrong": both_wrong,
        "target_interventions": interventions,
    }


def _summary_csv_row(
    *,
    benchmark: str,
    category: str,
    baseline_policy: str,
    target_policy: str,
    scope: str,
    summary: Row,
) -> Row:
    return {
        "benchmark": benchmark,
        "category": category,
        "baseline_policy": baseline_policy,
        "target_policy": target_policy,
        "scope": scope,
        "n": int(summary["n"]),
        "baseline_accuracy": _format_optional(summary["baseline_acc"]),
        "target_accuracy": _format_optional(summary["target_acc"]),
        "delta_accuracy": f"{float(summary['delta_acc']):.3f}",
        "baseline_gold_mismatch_exposure": _format_optional(summary["baseline_gold_mismatch"]),
        "target_gold_mismatch_exposure": _format_optional(summary["target_gold_mismatch"]),
        "delta_gold_mismatch_exposure": f"{float(summary['delta_gold_mismatch']):.3f}",
        "baseline_prompt_tokens": _format_optional(summary["baseline_prompt"], places=2),
        "target_prompt_tokens": _format_optional(summary["target_prompt"], places=2),
        "delta_prompt_tokens": f"{float(summary['delta_prompt']):.2f}",
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "both_wrong": int(summary["both_wrong"]),
        "target_interventions": int(summary["target_interventions"]),
        "verdict": _verdict(summary),
        "reviewer_note": _reviewer_note(scope, summary),
    }


def _audit_rows(
    *,
    benchmark: str,
    category: str,
    baseline_policy: str,
    target_policy: str,
    baseline_cases: dict[str, Row],
    target_cases: dict[str, Row],
    case_ids: set[str],
) -> list[Row]:
    rows: list[Row] = []
    for case_id in sorted(case_ids):
        baseline = baseline_cases[case_id]
        target = target_cases[case_id]
        flags = {
            "target_intervened": _intervened(target),
            "predeclared_validity_sensitive": _predeclared_validity_sensitive(baseline),
            "current_state_same_key_evidence_retrieved": _has_current_state_same_key_evidence(baseline),
            "baseline_gold_mismatch_exposure": bool(baseline.get("stale_state_exposure")),
            "same_key_state_conflict_exposed": _has_same_key_state_conflict(baseline),
            "discordant_correctness": bool(baseline.get("correct")) != bool(target.get("correct")),
        }
        if not any(flags.values()):
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "category": category,
                "case_id": case_id,
                "baseline_policy": baseline_policy,
                "target_policy": target_policy,
                **flags,
                "question": baseline.get("question", ""),
                "gold_answer": baseline.get("gold_answer", ""),
                "baseline_prediction": baseline.get("prediction", ""),
                "target_prediction": target.get("prediction", ""),
                "baseline_correct": bool(baseline.get("correct")),
                "target_correct": bool(target.get("correct")),
                "baseline_prompt_tokens": baseline.get("prompt_tokens", 0),
                "target_prompt_tokens": target.get("prompt_tokens", 0),
                "delta_prompt_tokens": float(target.get("prompt_tokens", 0.0))
                - float(baseline.get("prompt_tokens", 0.0)),
                "target_retrieval_mode": target.get("retrieval_mode", ""),
                "target_validity_removed": target.get("validity_removed", 0),
                "target_support_compacted": target.get("support_compacted", 0),
                "baseline_retrieved_records": baseline.get("retrieved_records", []),
                "target_retrieved_records": target.get("retrieved_records", []),
            }
        )
    return rows


def _verdict(summary: Row) -> str:
    if int(summary["n"]) == 0:
        return "no_cases"
    if int(summary["losses"]) > 0 or float(summary["delta_acc"]) < 0.0:
        return "not_usable"
    if int(summary["wins"]) > 0 and float(summary["delta_prompt"]) < 0.0:
        return "accuracy_plus_efficiency"
    if float(summary["delta_prompt"]) < 0.0 or float(summary["delta_gold_mismatch"]) < 0.0:
        return "efficiency_no_regression"
    return "neutral"


def _reviewer_note(scope: str, summary: Row) -> str:
    n = int(summary["n"])
    if n == 0:
        return "No paired cases in this scope."
    if int(summary["losses"]) > 0:
        return "Contains paired losses; do not headline this result."
    if scope == "ODV2 intervened" and n < 50:
        return "Small intervention slice; useful but insufficient alone for a strong claim."
    if scope == "Predeclared validity-sensitive union":
        return "Predeclared union: current-state same-key evidence, gold-mismatch exposure, or same-key conflict."
    if int(summary["wins"]) == 1 and int(summary["losses"]) == 0:
        return "Single discordant win with no losses; report as preliminary."
    if float(summary["delta_prompt"]) < 0.0 and float(summary["delta_acc"]) >= 0.0:
        return "No-regression efficiency result relative to baseline."
    return "Neutral paired result."


def _intervened(row: Row) -> bool:
    retrieval_mode = str(row.get("retrieval_mode", ""))
    return (
        "guard" in retrieval_mode
        or "compact" in retrieval_mode
        or _as_int(row.get("validity_removed")) > 0
        or _as_int(row.get("support_compacted")) > 0
    )


def _has_same_key_state_conflict(row: Row) -> bool:
    return len(_same_key_state_values(row)) > 1


def _predeclared_validity_sensitive(row: Row) -> bool:
    return (
        _has_current_state_same_key_evidence(row)
        or bool(row.get("stale_state_exposure"))
        or _has_same_key_state_conflict(row)
    )


def _has_current_state_same_key_evidence(row: Row) -> bool:
    query_mode = str(row.get("query_mode", ""))
    attribute = str(row.get("attribute", ""))
    if query_mode not in {"CURRENT_STATE", "STATE_WITH_PROVENANCE"}:
        return False
    if attribute in {"dialogue", "event"}:
        return False
    return bool(_same_key_state_values(row))


def _same_key_state_values(row: Row) -> set[str]:
    attribute = str(row.get("attribute", ""))
    entity = str(row.get("entity", ""))
    if attribute in {"dialogue", "event"}:
        return set()
    values: set[str] = set()
    for record in row.get("retrieved_records", []):
        if not isinstance(record, dict):
            continue
        if str(record.get("attribute", "")) != attribute:
            continue
        record_entity = str(record.get("entity", ""))
        if entity not in {"conversation", "all"} and record_entity != entity:
            continue
        memory_kind = str(record.get("memory_kind", ""))
        source_kind = str(record.get("source_kind", ""))
        if memory_kind != "state" and source_kind != "structured_fact":
            continue
        values.add(" ".join(str(record.get("value", "")).lower().split()))
    values.discard("")
    return values


def _first_value(rows: list[Row], key: str, fallback: str) -> object:
    return next((row[key] for row in rows if row.get(key)), fallback)


def _format_optional(value: object, *, places: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{places}f}"


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
