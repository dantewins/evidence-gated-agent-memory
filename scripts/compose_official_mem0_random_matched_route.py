#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any


BASE_POLICY = "official_mem0"
LOW_POLICY = "official_mem0_top1"
HIGH_POLICY = "official_mem0_top3"
MATCH_POLICY = "official_mem0_same_evidence_adaptive"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a matched-budget random top1/top3 control from existing official Mem0 "
            "top-k replay rows, and optionally summarize many random route draws."
        )
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--output", required=True, help="Output JSONL for the deterministic random route.")
    parser.add_argument("--summary", required=True, help="Markdown summary output.")
    parser.add_argument("--distribution", required=True, help="CSV output for random route distribution.")
    parser.add_argument("--policy-name", default="official_mem0_random_matched_top1_top3")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--top1-count", type=int, help="Number of cases randomly routed to top1.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    summary = Path(args.summary)
    distribution = Path(args.distribution)
    for path in (output, summary, distribution):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {path}. Pass --overwrite to replace it.")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows([Path(path) for path in args.paths])
    by_key = _rows_by_key(rows)
    case_keys = _shared_case_keys(by_key, [BASE_POLICY, LOW_POLICY, HIGH_POLICY, MATCH_POLICY])
    if not case_keys:
        raise SystemExit("No shared official_mem0/top1/top3/adaptive cases found.")

    top1_count = args.top1_count
    if top1_count is None:
        top1_count = sum(
            1
            for category, case_id in case_keys
            if str(by_key[(category, case_id, MATCH_POLICY)].get("source_policy_name") or "") == LOW_POLICY
        )
    if top1_count < 0 or top1_count > len(case_keys):
        raise SystemExit(f"Invalid top1 count {top1_count} for {len(case_keys)} cases.")

    selected_top1 = _random_top1_cases(case_keys, top1_count=top1_count, seed=args.seed)
    routed_rows = [
        _routed_row(
            by_key[(category, case_id, LOW_POLICY if (category, case_id) in selected_top1 else HIGH_POLICY)],
            policy_name=args.policy_name,
            source_policy_name=LOW_POLICY if (category, case_id) in selected_top1 else HIGH_POLICY,
            seed=args.seed,
            top1_count=top1_count,
            top3_count=len(case_keys) - top1_count,
        )
        for category, case_id in case_keys
    ]
    _write_jsonl(output, routed_rows)

    distribution_rows = []
    for draw_seed in range(args.draws):
        draw_top1 = _random_top1_cases(case_keys, top1_count=top1_count, seed=draw_seed)
        distribution_rows.append(
            _stats(
                case_keys,
                by_key,
                selected_top1=draw_top1,
                policy_name=f"random_seed_{draw_seed}",
            )
        )
    _write_csv(distribution, distribution_rows)

    deterministic = _stats(case_keys, by_key, selected_top1=selected_top1, policy_name=args.policy_name)
    adaptive = _policy_stats(case_keys, by_key, MATCH_POLICY)
    base = _policy_stats(case_keys, by_key, BASE_POLICY)
    top1 = _policy_stats(case_keys, by_key, LOW_POLICY)
    top3 = _policy_stats(case_keys, by_key, HIGH_POLICY)
    _write_summary(
        summary,
        deterministic=deterministic,
        adaptive=adaptive,
        base=base,
        top1=top1,
        top3=top3,
        distribution_rows=distribution_rows,
        output=output,
        distribution=distribution,
        seed=args.seed,
        top1_count=top1_count,
        total_cases=len(case_keys),
    )

    print(f"wrote {output}")
    print(f"wrote {distribution}")
    print(f"wrote {summary}")
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
                    rows.append(json.loads(line))
    return rows


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        grouped[
            (
                str(row.get("category") or ""),
                str(row.get("case_id") or ""),
                str(row.get("policy_name") or ""),
            )
        ] = row
    return grouped


def _shared_case_keys(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policies: list[str],
) -> list[tuple[str, str]]:
    shared: set[tuple[str, str]] | None = None
    for policy in policies:
        cases = {(category, case_id) for category, case_id, row_policy in by_key if row_policy == policy}
        shared = cases if shared is None else shared & cases
    return sorted(shared or set())


def _random_top1_cases(
    case_keys: list[tuple[str, str]],
    *,
    top1_count: int,
    seed: int,
) -> set[tuple[str, str]]:
    rng = random.Random(seed)
    return set(rng.sample(case_keys, top1_count))


def _routed_row(
    selected: dict[str, Any],
    *,
    policy_name: str,
    source_policy_name: str,
    seed: int,
    top1_count: int,
    top3_count: int,
) -> dict[str, Any]:
    row = dict(selected)
    row.update(
        {
            "policy_name": policy_name,
            "source_policy_name": source_policy_name,
            "retrieval_mode": "official_mem0_random_matched_route",
            "matched_random_seed": seed,
            "matched_random_top1_count": top1_count,
            "matched_random_top3_count": top3_count,
        }
    )
    return row


def _policy_stats(
    case_keys: list[tuple[str, str]],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policy_name: str,
) -> dict[str, Any]:
    rows = [by_key[(category, case_id, policy_name)] for category, case_id in case_keys]
    return _stats_from_rows(rows, policy_name=policy_name, case_keys=case_keys, by_key=by_key)


def _stats(
    case_keys: list[tuple[str, str]],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    selected_top1: set[tuple[str, str]],
    policy_name: str,
) -> dict[str, Any]:
    rows = [
        by_key[(category, case_id, LOW_POLICY if (category, case_id) in selected_top1 else HIGH_POLICY)]
        for category, case_id in case_keys
    ]
    return _stats_from_rows(rows, policy_name=policy_name, case_keys=case_keys, by_key=by_key)


def _stats_from_rows(
    rows: list[dict[str, Any]],
    *,
    policy_name: str,
    case_keys: list[tuple[str, str]],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    base_rows = [by_key[(category, case_id, BASE_POLICY)] for category, case_id in case_keys]
    correct = sum(bool(row.get("correct")) for row in rows)
    base_correct = sum(bool(row.get("correct")) for row in base_rows)
    reader_tokens = sum(_reader_tokens(row) for row in rows)
    base_reader_tokens = sum(_reader_tokens(row) for row in base_rows)
    context_tokens = sum(_context_tokens(row) for row in rows)
    base_context_tokens = sum(_context_tokens(row) for row in base_rows)
    wins = sum((not bool(base.get("correct"))) and bool(row.get("correct")) for base, row in zip(base_rows, rows))
    losses = sum(bool(base.get("correct")) and not bool(row.get("correct")) for base, row in zip(base_rows, rows))
    return {
        "policy_name": policy_name,
        "cases": len(rows),
        "correct": correct,
        "base_correct": base_correct,
        "wins_vs_official_mem0": wins,
        "losses_vs_official_mem0": losses,
        "reader_total_tokens": reader_tokens,
        "delta_reader_tokens_vs_official_mem0_pct": _reduction_pct(reader_tokens, base_reader_tokens),
        "reader_visible_retrieved_context_tokens": context_tokens,
        "delta_reader_visible_context_vs_official_mem0_pct": _reduction_pct(context_tokens, base_context_tokens),
        "tokens_per_correct": reader_tokens / correct if correct else 0.0,
    }


def _reader_tokens(row: dict[str, Any]) -> int:
    return int(row.get("reader_total_tokens") or 0) or (
        int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
    )


def _context_tokens(row: dict[str, Any]) -> int:
    return int(row.get("retrieved_context_tokens") or 0)


def _reduction_pct(value: int, base: int) -> float:
    if not base:
        return 0.0
    return (base - value) / base * 100.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(
    path: Path,
    *,
    deterministic: dict[str, Any],
    adaptive: dict[str, Any],
    base: dict[str, Any],
    top1: dict[str, Any],
    top3: dict[str, Any],
    distribution_rows: list[dict[str, Any]],
    output: Path,
    distribution: Path,
    seed: int,
    top1_count: int,
    total_cases: int,
) -> None:
    lines = [
        "# Matched-Budget Random Top1/Top3 Control",
        "",
        f"Deterministic control seed: {seed}",
        f"Route budget: {top1_count} top1 cases and {total_cases - top1_count} top3 cases.",
        f"Case JSONL: `{output}`",
        f"Random draw distribution: `{distribution}`",
        "",
        "## Point Comparisons",
        "",
        "| policy | correct | reader tokens | reader-token reduction | reader-visible context | context reduction | tokens/correct | wins/losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (base, top1, deterministic, adaptive, top3):
        lines.append(
            "| {policy_name} | {correct}/{cases} | {reader_total_tokens} | {delta_reader_tokens_vs_official_mem0_pct:.2f}% | "
            "{reader_visible_retrieved_context_tokens} | {delta_reader_visible_context_vs_official_mem0_pct:.2f}% | "
            "{tokens_per_correct:.1f} | {wins_vs_official_mem0}/{losses_vs_official_mem0} |".format(**row)
        )
    lines.extend(["", "## Random Route Distribution", ""])
    for key, label in (
        ("correct", "Correct answers"),
        ("reader_total_tokens", "Reader total tokens"),
        ("reader_visible_retrieved_context_tokens", "Reader-visible retrieved-context tokens"),
    ):
        values = sorted(float(row[key]) for row in distribution_rows)
        lines.append(
            f"- {label}: mean={statistics.mean(values):.2f}, median={_percentile(values, 0.50):.0f}, "
            f"95% interval=[{_percentile(values, 0.025):.0f}, {_percentile(values, 0.975):.0f}], "
            f"min={values[0]:.0f}, max={values[-1]:.0f}."
        )
    adaptive_correct = float(adaptive["correct"])
    pct_leq = sum(float(row["correct"]) <= adaptive_correct for row in distribution_rows) / len(distribution_rows)
    lines.extend(
        [
            "",
            f"Adaptive correct count is greater than or equal to {pct_leq:.2%} of random matched-budget draws.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = int((len(values) - 1) * q)
    return values[index]


if __name__ == "__main__":
    sys.exit(main())
