#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare official_mem0 vs official_mem0_odv2_selective token spend."
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument(
        "--extra-policy",
        action="append",
        default=["official_mem0_top2"],
        help="Also compare this policy against official_mem0 when present.",
    )
    args = parser.parse_args()

    rows = _load_rows([Path(path) for path in args.paths])
    pairs = _paired_rows(rows)
    if not pairs:
        raise SystemExit("No paired official_mem0/official_mem0_odv2_selective rows found.")

    print(f"rows={len(rows)} paired_cases={len(pairs)}")
    _print_metric("prompt_tokens", pairs, lambda row: int(row.get("prompt_tokens") or 0))
    _print_metric(
        "completion_tokens",
        pairs,
        lambda row: int(row.get("completion_tokens") or 0),
    )
    _print_metric(
        "reader_total_tokens",
        pairs,
        lambda row: int(row.get("prompt_tokens") or 0)
        + int(row.get("completion_tokens") or 0),
    )
    _print_metric(
        "retrieved_context_tokens",
        pairs,
        lambda row: int(row.get("retrieved_context_tokens") or 0),
    )
    _print_metric("retrieved_items", pairs, lambda row: int(row.get("retrieved_items") or 0))

    gate_rows = [odv2 for _, odv2 in pairs]
    print(f"compact_rows={sum(_is_compact_row(row) for row in gate_rows)}")
    print(f"guard_rows={sum(row.get('retrieval_mode') == 'official_mem0_odv2_guard' for row in gate_rows)}")
    print(f"validity_removed={sum(int(row.get('validity_removed') or 0) for row in gate_rows)}")
    print(f"validity_appended={sum(int(row.get('validity_appended') or 0) for row in gate_rows)}")
    print(f"support_compacted={sum(int(row.get('support_compacted') or 0) for row in gate_rows)}")
    print(
        "same_predictions="
        f"{sum(base.get('prediction') == odv2.get('prediction') for base, odv2 in pairs)}/{len(pairs)}"
    )
    print(
        "accuracy="
        f"official_mem0={_accuracy(base for base, _ in pairs):.3f} "
        f"official_mem0_odv2_selective={_accuracy(odv2 for _, odv2 in pairs):.3f}"
    )

    by_category: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for base, odv2 in pairs:
        by_category[str(base.get("category") or "")].append((base, odv2))
    print("by_category")
    for category, category_pairs in sorted(by_category.items()):
        base_tokens = sum(int(base.get("prompt_tokens") or 0) for base, _ in category_pairs)
        odv2_tokens = sum(int(odv2.get("prompt_tokens") or 0) for _, odv2 in category_pairs)
        delta = odv2_tokens - base_tokens
        pct = (delta / base_tokens * 100.0) if base_tokens else 0.0
        compact = sum(_is_compact_row(odv2) for _, odv2 in category_pairs)
        print(
            f"  {category}: n={len(category_pairs)} prompt_delta={delta} "
            f"prompt_delta_pct={pct:.2f}% compact_rows={compact}"
        )

    for policy_name in args.extra_policy:
        _print_extra_policy_comparison(rows, policy_name)
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


def _paired_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return _paired_rows_for(rows, "official_mem0_odv2_selective")


def _paired_rows_for(
    rows: list[dict[str, Any]],
    comparison_policy: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[
            (
                str(row.get("category") or ""),
                str(row.get("case_id") or ""),
                str(row.get("policy_name") or ""),
            )
        ] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for (category, case_id, policy_name), base in by_key.items():
        if policy_name != "official_mem0":
            continue
        odv2 = by_key.get((category, case_id, comparison_policy))
        if odv2 is not None:
            pairs.append((base, odv2))
    return pairs


def _print_metric(
    label: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    getter: Callable[[dict[str, Any]], int],
) -> None:
    base_total = sum(getter(base) for base, _ in pairs)
    odv2_total = sum(getter(odv2) for _, odv2 in pairs)
    delta = odv2_total - base_total
    pct = (delta / base_total * 100.0) if base_total else 0.0
    print(
        f"{label}: official_mem0={base_total} "
        f"official_mem0_odv2_selective={odv2_total} delta={delta} delta_pct={pct:.2f}%"
    )


def _print_extra_policy_comparison(rows: list[dict[str, Any]], policy_name: str) -> None:
    pairs = _paired_rows_for(rows, policy_name)
    if not pairs:
        return
    print(f"comparison={policy_name} paired_cases={len(pairs)}")
    for label, getter in (
        ("prompt_tokens", lambda row: int(row.get("prompt_tokens") or 0)),
        (
            "reader_total_tokens",
            lambda row: int(row.get("prompt_tokens") or 0)
            + int(row.get("completion_tokens") or 0),
        ),
        ("retrieved_context_tokens", lambda row: int(row.get("retrieved_context_tokens") or 0)),
        ("retrieved_items", lambda row: int(row.get("retrieved_items") or 0)),
    ):
        base_total = sum(getter(base) for base, _ in pairs)
        comparison_total = sum(getter(comparison) for _, comparison in pairs)
        delta = comparison_total - base_total
        pct = (delta / base_total * 100.0) if base_total else 0.0
        print(
            f"  {label}: official_mem0={base_total} {policy_name}={comparison_total} "
            f"delta={delta} delta_pct={pct:.2f}%"
        )
    print(
        f"  accuracy=official_mem0={_accuracy(base for base, _ in pairs):.3f} "
        f"{policy_name}={_accuracy(comparison for _, comparison in pairs):.3f}"
    )
    print(
        "  wins_losses="
        f"wins={sum((not bool(base.get('correct'))) and bool(comparison.get('correct')) for base, comparison in pairs)} "
        f"losses={sum(bool(base.get('correct')) and not bool(comparison.get('correct')) for base, comparison in pairs)} "
        f"same_predictions={sum(base.get('prediction') == comparison.get('prediction') for base, comparison in pairs)}/{len(pairs)}"
    )


def _is_compact_row(row: dict[str, Any]) -> bool:
    return str(row.get("retrieval_mode") or "").startswith("official_mem0_odv2_compact")


def _accuracy(rows) -> float:
    row_list = list(rows)
    if not row_list:
        return 0.0
    return sum(bool(row.get("correct")) for row in row_list) / len(row_list)


if __name__ == "__main__":
    raise SystemExit(main())
