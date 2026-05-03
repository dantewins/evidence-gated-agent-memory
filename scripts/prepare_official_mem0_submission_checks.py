#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE_POLICY = "official_mem0"
TARGET_POLICY = "official_mem0_odv2_selective"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare submission-facing checks from an official Mem0 result directory: "
            "cost-normalized metrics, paired bootstrap intervals, manual audit sample, "
            "and state-guard isolation diagnostics."
        )
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--output-dir", help="Directory for generated check files.")
    parser.add_argument("--audit-size", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument(
        "--extra-policy",
        action="append",
        default=[
            "official_mem0_same_evidence_adaptive",
            "official_mem0_top1",
            "official_mem0_top2",
            "official_mem0_top3",
            "official_mem0_top4",
        ],
        help="Additional policy to summarize against official_mem0 when present.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.paths]
    rows = _load_rows(input_paths)
    if not rows:
        raise SystemExit("No JSONL rows found.")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(input_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_key = _by_key(rows)
    policy_names = [TARGET_POLICY] + list(dict.fromkeys(args.extra_policy))
    efficiency_rows = _efficiency_rows(by_key, policy_names)
    bootstrap_rows: list[dict[str, Any]] = []
    for policy_name in policy_names:
        bootstrap_rows.extend(
            _bootstrap_rows(
                by_key,
                policy_name,
                samples=max(1, args.bootstrap_samples),
                seed=args.seed,
            )
        )
    audit_rows = _manual_audit_rows(by_key, args.audit_size)
    guard_rows = _state_guard_rows(by_key)

    _write_csv(output_dir / "policy_efficiency.csv", efficiency_rows)
    _write_csv(output_dir / "paired_bootstrap_summary.csv", bootstrap_rows)
    _write_csv(output_dir / "manual_audit_sample.csv", audit_rows)
    _write_csv(output_dir / "state_guard_isolation.csv", guard_rows)
    _write_manual_instructions(output_dir / "manual_audit_instructions.md")
    _write_summary(
        output_dir / "submission_checks_summary.md",
        efficiency_rows=efficiency_rows,
        bootstrap_rows=bootstrap_rows,
        audit_rows=audit_rows,
        guard_rows=guard_rows,
        source_paths=input_paths,
    )

    print(f"wrote {output_dir}/policy_efficiency.csv")
    print(f"wrote {output_dir}/paired_bootstrap_summary.csv")
    print(f"wrote {output_dir}/manual_audit_sample.csv")
    print(f"wrote {output_dir}/state_guard_isolation.csv")
    print(f"wrote {output_dir}/submission_checks_summary.md")
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*_cases.jsonl")))
        else:
            files.append(path)
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source_file"] = str(path)
                    rows.append(row)
    return rows


def _default_output_dir(input_paths: list[Path]) -> Path:
    if len(input_paths) == 1 and input_paths[0].is_dir():
        return input_paths[0] / "submission_checks"
    return Path("results") / "submission_checks"


def _by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("category") or ""),
            str(row.get("case_id") or ""),
            str(row.get("policy_name") or ""),
        )
        grouped[key] = row
    return grouped


def _case_keys(by_key: dict[tuple[str, str, str], dict[str, Any]], policy: str) -> set[tuple[str, str]]:
    return {(category, case_id) for category, case_id, row_policy in by_key if row_policy == policy}


def _pairs(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policy: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    shared = sorted(_case_keys(by_key, BASE_POLICY) & _case_keys(by_key, policy))
    return [(by_key[(category, case_id, BASE_POLICY)], by_key[(category, case_id, policy)]) for category, case_id in shared]


def _efficiency_rows(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policies: list[str],
) -> list[dict[str, Any]]:
    base_cases = sorted(_case_keys(by_key, BASE_POLICY))
    if not base_cases:
        raise SystemExit("No official_mem0 rows found.")
    policy_order = list(dict.fromkeys([BASE_POLICY] + policies))
    base_rows = [by_key[(category, case_id, BASE_POLICY)] for category, case_id in base_cases]
    base_total = sum(_reader_tokens(row) for row in base_rows)

    rows: list[dict[str, Any]] = []
    for policy in policy_order:
        cases = sorted(_case_keys(by_key, policy))
        if not cases:
            continue
        policy_rows = [by_key[(category, case_id, policy)] for category, case_id in cases]
        total_tokens = sum(_reader_tokens(row) for row in policy_rows)
        correct = sum(_correct(row) for row in policy_rows)
        delta_pct = ((total_tokens - base_total) / base_total * 100.0) if base_total else 0.0
        tokens_per_correct = (total_tokens / correct) if correct else 0.0
        rows.append(
            {
                "policy_name": policy,
                "paired_with_official_mem0": len(_pairs(by_key, policy)) if policy != BASE_POLICY else len(cases),
                "rows": len(policy_rows),
                "correct": correct,
                "accuracy": f"{correct / len(policy_rows):.6f}" if policy_rows else "0.000000",
                "reader_total_tokens": total_tokens,
                "delta_reader_tokens_vs_official_mem0_pct": "---" if policy == BASE_POLICY else f"{delta_pct:.2f}",
                "tokens_per_correct": f"{tokens_per_correct:.2f}" if correct else "",
                "correct_per_100k_reader_tokens": f"{(correct / total_tokens * 100000.0):.2f}" if total_tokens else "0.00",
                "retrieved_context_tokens": sum(int(row.get("retrieved_context_tokens") or 0) for row in policy_rows),
                "retrieved_items": sum(int(row.get("retrieved_items") or 0) for row in policy_rows),
            }
        )
    return rows


def _bootstrap_rows(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policy: str,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    pairs = _pairs(by_key, policy)
    if not pairs:
        return []
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    n = len(pairs)
    indexes = list(range(n))
    for _ in range(samples):
        sample = [pairs[rng.choice(indexes)] for _ in indexes]
        base_tokens = sum(_reader_tokens(base) for base, _ in sample)
        target_tokens = sum(_reader_tokens(target) for _, target in sample)
        base_correct = sum(_correct(base) for base, _ in sample)
        target_correct = sum(_correct(target) for _, target in sample)
        draws["accuracy_delta_pct_points"].append((target_correct - base_correct) / n * 100.0)
        draws["reader_token_reduction_pct"].append(
            ((base_tokens - target_tokens) / base_tokens * 100.0) if base_tokens else 0.0
        )
        draws["target_tokens_per_correct"].append(target_tokens / target_correct if target_correct else 0.0)
        draws["base_tokens_per_correct"].append(base_tokens / base_correct if base_correct else 0.0)
        draws["tokens_per_correct_reduction_pct"].append(
            ((base_tokens / base_correct) - (target_tokens / target_correct))
            / (base_tokens / base_correct)
            * 100.0
            if base_correct and target_correct
            else 0.0
        )

    rows: list[dict[str, Any]] = []
    for metric, values in sorted(draws.items()):
        lo, med, hi = _ci(values)
        rows.append(
            {
                "comparison": f"{policy}_minus_{BASE_POLICY}",
                "metric": metric,
                "bootstrap_samples": samples,
                "ci_low_2.5": f"{lo:.4f}",
                "median": f"{med:.4f}",
                "ci_high_97.5": f"{hi:.4f}",
            }
        )
    return rows


def _manual_audit_rows(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    audit_size: int,
) -> list[dict[str, Any]]:
    pairs = _pairs(by_key, TARGET_POLICY)
    top2_by_case = {
        (category, case_id): by_key[(category, case_id, "official_mem0_top2")]
        for category, case_id in _case_keys(by_key, "official_mem0_top2")
    }
    top3_by_case = {
        (category, case_id): by_key[(category, case_id, "official_mem0_top3")]
        for category, case_id in _case_keys(by_key, "official_mem0_top3")
    }

    priority: list[tuple[int, str, str, dict[str, Any], dict[str, Any]]] = []
    for base, target in pairs:
        category = str(base.get("category") or "")
        case_id = str(base.get("case_id") or "")
        base_correct = _correct(base)
        target_correct = _correct(target)
        top2 = top2_by_case.get((category, case_id))
        top3 = top3_by_case.get((category, case_id))
        score = 50
        if base_correct != target_correct:
            score = 0 if target_correct else 1
        elif top2 is not None and _correct(top2) != target_correct:
            score = 5
        elif top3 is not None and _correct(top3) != target_correct:
            score = 8
        elif category in {"multi-session", "single-session-preference"}:
            score = 20
        priority.append((score, category, case_id, base, target))

    selected = sorted(priority)[: max(0, audit_size)]
    rows: list[dict[str, Any]] = []
    for _, category, case_id, base, target in selected:
        top2 = top2_by_case.get((category, case_id), {})
        top3 = top3_by_case.get((category, case_id), {})
        rows.append(
            {
                "category": category,
                "case_id": case_id,
                "question": _clean_cell(base.get("question")),
                "gold_answer": _clean_cell(base.get("gold_answer")),
                "official_mem0_prediction": _clean_cell(base.get("prediction")),
                "official_mem0_auto_correct": _correct(base),
                "odv2_prediction": _clean_cell(target.get("prediction")),
                "odv2_auto_correct": _correct(target),
                "top2_prediction": _clean_cell(top2.get("prediction")),
                "top2_auto_correct": _correct(top2) if top2 else "",
                "top3_prediction": _clean_cell(top3.get("prediction")),
                "top3_auto_correct": _correct(top3) if top3 else "",
                "official_mem0_reader_tokens": _reader_tokens(base),
                "odv2_reader_tokens": _reader_tokens(target),
                "official_mem0_evidence": _records_summary(base),
                "odv2_evidence": _records_summary(target),
                "manual_official_mem0_correct": "",
                "manual_odv2_correct": "",
                "manual_notes": "",
            }
        )
    return rows


def _state_guard_rows(by_key: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (category, case_id, policy), target in sorted(by_key.items()):
        if policy != TARGET_POLICY:
            continue
        if int(target.get("validity_removed") or 0) <= 0:
            continue
        base = by_key.get((category, case_id, BASE_POLICY), {})
        top2 = by_key.get((category, case_id, "official_mem0_top2"), {})
        rows.append(
            {
                "category": category,
                "case_id": case_id,
                "validity_removed": int(target.get("validity_removed") or 0),
                "official_mem0_prediction": _clean_cell(base.get("prediction")),
                "official_mem0_correct": _correct(base) if base else "",
                "top2_prediction": _clean_cell(top2.get("prediction")),
                "top2_correct": _correct(top2) if top2 else "",
                "odv2_prediction": _clean_cell(target.get("prediction")),
                "odv2_correct": _correct(target),
                "official_mem0_evidence": _records_summary(base),
                "odv2_evidence": _records_summary(target),
            }
        )
    return rows


def _write_manual_instructions(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Manual Audit Instructions",
                "",
                "Grade each prediction using the question, gold answer, and visible retrieved evidence.",
                "Use `1` for correct, `0` for incorrect, and `?` only when the gold answer is ambiguous.",
                "Do not give credit for unsupported answers unless the prediction matches the gold answer.",
                "Fill `manual_official_mem0_correct`, `manual_odv2_correct`, and `manual_notes` in `manual_audit_sample.csv`.",
                "",
                "Report two numbers in the paper if time permits:",
                "",
                "- Manual agreement with the automatic span scorer.",
                "- Any corrected accuracy difference between official Mem0 and ODV2 on this 50-case audit.",
            ]
        )
        + "\n"
    )


def _write_summary(
    path: Path,
    *,
    efficiency_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    source_paths: list[Path],
) -> None:
    cache_free_path = path.parent / "cache_free_reader_systems.csv"
    systems_line = (
        "- Systems sanity claim: supported by `cache_free_reader_systems.csv`."
        if cache_free_path.exists()
        else "- Systems claim: still needs cache-free wall-clock benchmark output."
    )
    lines = [
        "# Official Mem0 Submission Checks",
        "",
        f"Sources: {', '.join(str(path) for path in source_paths)}",
        "",
        "## Policy Efficiency",
        "",
        "| policy | rows | correct | reader tokens | delta tokens | tokens/correct | correct/100k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in efficiency_rows:
        lines.append(
            "| {policy_name} | {rows} | {correct} | {reader_total_tokens} | "
            "{delta_reader_tokens_vs_official_mem0_pct} | {tokens_per_correct} | "
            "{correct_per_100k_reader_tokens} |".format(**row)
        )
    lines.extend(["", "## Bootstrap CIs", ""])
    if bootstrap_rows:
        lines.extend(
            [
                "| metric | 2.5% | median | 97.5% |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in bootstrap_rows:
            lines.append(
                f"| {row['metric']} | {row['ci_low_2.5']} | {row['median']} | {row['ci_high_97.5']} |"
            )
    else:
        lines.append("No paired rows available for bootstrap.")
    lines.extend(
        [
            "",
            "## Manual Audit",
            "",
            f"Wrote {len(audit_rows)} prioritized cases to `manual_audit_sample.csv`.",
            "Use this to test whether the local span scorer is undercounting correctness.",
            "",
            "## State Guard Isolation",
            "",
            f"Wrote {len(guard_rows)} rows where ODV2 removed stale official Mem0 evidence.",
            "If this count is tiny, claim token savings as ranked evidence compaction rather than validity reasoning.",
            "",
            "## What This Supports",
            "",
            "- Token-savings claim: supported by paired reader-token totals.",
            "- Cost-normalized utility claim: supported by tokens per correct answer.",
            "- Accuracy-improvement claim: not supported unless additional validation changes the result.",
            systems_line,
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ci(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return (
        ordered[int((len(ordered) - 1) * 0.025)],
        ordered[int((len(ordered) - 1) * 0.500)],
        ordered[int((len(ordered) - 1) * 0.975)],
    )


def _reader_tokens(row: dict[str, Any]) -> int:
    return int(row.get("reader_total_tokens") or 0) or (
        int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
    )


def _correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def _clean_cell(value: Any, *, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text[: limit - 3] + "..." if len(text) > limit else text


def _records_summary(row: dict[str, Any], *, max_records: int = 5, limit: int = 900) -> str:
    if not row:
        return ""
    chunks: list[str] = []
    for index, record in enumerate((row.get("retrieved_records") or [])[:max_records], start=1):
        if not isinstance(record, dict):
            continue
        value = record.get("value") or record.get("text") or record.get("support_text") or ""
        attribute = record.get("attribute") or ""
        chunks.append(f"[{index}] {attribute}: {value}")
    return _clean_cell(" | ".join(chunks), limit=limit)


if __name__ == "__main__":
    raise SystemExit(main())
