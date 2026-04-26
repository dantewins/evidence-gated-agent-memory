from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASELINE_POLICY = "mem0"
TARGET_POLICY = "odv2_mem0_selective"
Row = dict[str, object]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python scripts/compile_boss_results.py <cases.jsonl> [<cases.jsonl> ...]")
        return 1
    try:
        print(render_report([Path(path) for path in argv]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def render_report(paths: list[Path]) -> str:
    reports = [_compile_file(path) for path in paths]
    lines: list[str] = [
        "# ODV2 Results Brief",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(_executive_summary(reports))
    lines.extend(
        [
            "",
            "## Main Comparison",
            "",
            "| result | n | mem0 acc | odv2 acc | delta acc | delta stale | delta ctx | wins | losses | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for report in reports:
        lines.append(_summary_row(report["label"], report["all_paired"]))

    lines.extend(["", "## Validity-Sensitive Slices", ""])
    for report in reports:
        lines.extend(_slice_section(report))
    lines.extend(
        [
            "",
            "## Claim To Use",
            "",
            _claim_text(reports),
            "",
            "## What Not To Claim",
            "",
            "- Do not claim ODV2 beats Mem0 on broad benchmark accuracy unless delta acc is positive.",
            "- Do not use the negative ablation variants as the headline result.",
            "- Do not claim stale-state reduction unless delta stale is below 0.000 on the relevant slice.",
        ]
    )
    return "\n".join(lines)


def _compile_file(path: Path) -> dict[str, object]:
    rows = _load_rows(path)
    label = _label_for(path, rows)
    by_policy_case = _by_policy_case(rows)
    if BASELINE_POLICY not in by_policy_case:
        raise ValueError(f"{path} has no '{BASELINE_POLICY}' rows")
    if TARGET_POLICY not in by_policy_case:
        raise ValueError(f"{path} has no '{TARGET_POLICY}' rows")
    slices = {
        "all_paired": set(by_policy_case[BASELINE_POLICY]) & set(by_policy_case[TARGET_POLICY]),
        "validity_intervened": {
            case_id
            for case_id, row in by_policy_case[TARGET_POLICY].items()
            if _intervened(row) and case_id in by_policy_case[BASELINE_POLICY]
        },
        "baseline_stale_exposure": {
            case_id
            for case_id, row in by_policy_case[BASELINE_POLICY].items()
            if bool(row.get("stale_state_exposure")) and case_id in by_policy_case[TARGET_POLICY]
        },
        "baseline_same_key_conflict": {
            case_id
            for case_id, row in by_policy_case[BASELINE_POLICY].items()
            if _has_same_key_state_conflict(row) and case_id in by_policy_case[TARGET_POLICY]
        },
    }
    return {
        "path": path,
        "label": label,
        "by_policy_case": by_policy_case,
        **{
            name: _paired_summary(by_policy_case, case_ids)
            for name, case_ids in slices.items()
        },
    }


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


def _label_for(path: Path, rows: list[Row]) -> str:
    categories = sorted({str(row.get("category", "")) for row in rows if row.get("category")})
    benchmark = next((str(row.get("benchmark", "")) for row in rows if row.get("benchmark")), "")
    if len(categories) == 1:
        return f"{benchmark}:{categories[0]}" if benchmark else categories[0]
    return path.stem.replace("_cases", "")


def _by_policy_case(rows: Iterable[Row]) -> dict[str, dict[str, Row]]:
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        policy = str(row.get("policy_name", ""))
        if policy in {BASELINE_POLICY, TARGET_POLICY}:
            grouped[policy][str(row["case_id"])] = row
    return grouped


def _paired_summary(
    by_policy_case: dict[str, dict[str, Row]],
    case_ids: set[str],
) -> dict[str, object]:
    paired = [
        (
            by_policy_case[TARGET_POLICY][case_id],
            by_policy_case[BASELINE_POLICY][case_id],
        )
        for case_id in sorted(case_ids)
    ]
    if not paired:
        return {
            "n": 0,
            "baseline_acc": 0.0,
            "target_acc": 0.0,
            "delta_acc": 0.0,
            "delta_stale": 0.0,
            "delta_ctx": 0.0,
            "wins": 0,
            "losses": 0,
            "both_wrong": 0,
            "baseline_ctx": 0.0,
            "target_ctx": 0.0,
            "baseline_stale": 0.0,
            "target_stale": 0.0,
        }
    wins = sum(1 for target, base in paired if _bool(target, "correct") and not _bool(base, "correct"))
    losses = sum(1 for target, base in paired if not _bool(target, "correct") and _bool(base, "correct"))
    both_wrong = sum(1 for target, base in paired if not _bool(target, "correct") and not _bool(base, "correct"))
    baseline_acc = _mean(_as_float_bool(base.get("correct")) for _, base in paired)
    target_acc = _mean(_as_float_bool(target.get("correct")) for target, _ in paired)
    baseline_ctx = _mean(float(base.get("prompt_tokens", 0.0)) for _, base in paired)
    target_ctx = _mean(float(target.get("prompt_tokens", 0.0)) for target, _ in paired)
    baseline_stale = _mean(_as_float_bool(base.get("stale_state_exposure")) for _, base in paired)
    target_stale = _mean(_as_float_bool(target.get("stale_state_exposure")) for target, _ in paired)
    return {
        "n": len(paired),
        "baseline_acc": baseline_acc,
        "target_acc": target_acc,
        "delta_acc": target_acc - baseline_acc,
        "delta_stale": target_stale - baseline_stale,
        "delta_ctx": target_ctx - baseline_ctx,
        "wins": wins,
        "losses": losses,
        "both_wrong": both_wrong,
        "baseline_ctx": baseline_ctx,
        "target_ctx": target_ctx,
        "baseline_stale": baseline_stale,
        "target_stale": target_stale,
    }


def _executive_summary(reports: list[dict[str, object]]) -> list[str]:
    usable = [report for report in reports if _is_usable(report["all_paired"])]
    if usable:
        best = min(usable, key=lambda report: float(report["all_paired"]["delta_ctx"]))
        summary = best["all_paired"]
        return [
            "- Status: usable efficiency result against Mem0.",
            (
                f"- Best headline: {best['label']} matches Mem0 accuracy "
                f"with {abs(float(summary['delta_ctx'])):.2f} fewer prompt tokens per query "
                f"and {int(summary['losses'])} paired losses."
            ),
            "- Interpretation: ODV2 is not an accuracy win; it is a Mem0-safe validity gate that can reduce context cost.",
        ]
    return [
        "- Status: no usable positive result yet.",
        "- Requirement for a defensible claim: delta acc >= 0.000, losses = 0, and delta ctx < 0 or delta stale < 0.",
    ]


def _slice_section(report: dict[str, object]) -> list[str]:
    lines = [
        f"### {report['label']}",
        "",
        "| slice | n | delta acc | delta stale | delta ctx | wins | losses | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, label in (
        ("validity_intervened", "ODV2 intervened"),
        ("baseline_stale_exposure", "Mem0 exposed stale state"),
        ("baseline_same_key_conflict", "Mem0 same-key conflict"),
    ):
        lines.append(_slice_row(label, report[key]))
    lines.append("")
    return lines


def _summary_row(label: object, summary: dict[str, object]) -> str:
    return (
        f"| {label} | {int(summary['n'])} | "
        f"{float(summary['baseline_acc']):.3f} | "
        f"{float(summary['target_acc']):.3f} | "
        f"{float(summary['delta_acc']):.3f} | "
        f"{float(summary['delta_stale']):.3f} | "
        f"{float(summary['delta_ctx']):.2f} | "
        f"{int(summary['wins'])} | "
        f"{int(summary['losses'])} | "
        f"{_verdict(summary)} |"
    )


def _slice_row(label: str, summary: dict[str, object]) -> str:
    return (
        f"| {label} | {int(summary['n'])} | "
        f"{float(summary['delta_acc']):.3f} | "
        f"{float(summary['delta_stale']):.3f} | "
        f"{float(summary['delta_ctx']):.2f} | "
        f"{int(summary['wins'])} | "
        f"{int(summary['losses'])} | "
        f"{_verdict(summary)} |"
    )


def _claim_text(reports: list[dict[str, object]]) -> str:
    if any(_is_usable(report["all_paired"]) for report in reports):
        return (
            "ODV2 selective retrieval can be attached to Mem0 as a conservative "
            "validity gate: on the evaluated validity-sensitive slice, it preserves "
            "Mem0 answer accuracy and introduces no paired losses while reducing "
            "prompt context."
        )
    return (
        "Current results do not support a positive ODV2 claim. The honest result is "
        "that broad ODV2 variants can hurt Mem0, and only strictly gated variants "
        "are safe enough to continue evaluating."
    )


def _verdict(summary: dict[str, object]) -> str:
    if int(summary["n"]) == 0:
        return "no cases"
    if _is_usable(summary):
        return "usable"
    if float(summary["delta_acc"]) < 0 or int(summary["losses"]) > 0:
        return "not usable"
    if float(summary["delta_ctx"]) < 0 or float(summary["delta_stale"]) < 0:
        return "promising"
    return "neutral"


def _is_usable(summary: dict[str, object]) -> bool:
    return (
        int(summary["n"]) > 0
        and float(summary["delta_acc"]) >= 0.0
        and int(summary["losses"]) == 0
        and (
            float(summary["delta_ctx"]) < 0.0
            or float(summary["delta_stale"]) < 0.0
        )
    )


def _intervened(row: Row) -> bool:
    retrieval_mode = str(row.get("retrieval_mode", ""))
    return (
        "guard" in retrieval_mode
        or "compact" in retrieval_mode
        or _as_int(row.get("validity_removed")) > 0
        or _as_int(row.get("support_compacted")) > 0
    )


def _has_same_key_state_conflict(row: Row) -> bool:
    values = _same_key_state_values(row)
    return len(values) > 1


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


def _bool(row: Row, key: str) -> bool:
    return bool(row.get(key, False))


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
