from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python scripts/summarize_diagnostics.py <cases.jsonl> [baseline_policy]")
        return 1

    rows = _load_rows(Path(argv[0]))
    if not rows:
        print("No diagnostic rows found.")
        return 1
    baseline_policy = argv[1] if len(argv) > 1 else "mem0"

    _print_policy_summary(rows)
    print()
    _print_pairwise_summary(rows, baseline_policy=baseline_policy)
    print()
    _print_category_summary(rows)
    return 0


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _print_policy_summary(rows: list[dict[str, object]]) -> None:
    print("| policy | n | accuracy | exact | retrieval_hit | stale_exposure | ctx_tokens | memory_tokens | latency_ms | acc_per_1k_ctx |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for policy, policy_rows in sorted(_group_by(rows, "policy_name").items()):
        context_tokens = _mean(policy_rows, "prompt_tokens")
        accuracy = _mean_bool(policy_rows, "correct")
        print(
            f"| {policy} | {len(policy_rows)} | "
            f"{accuracy:.3f} | "
            f"{_mean_bool(policy_rows, 'exact_match'):.3f} | "
            f"{_mean_bool(policy_rows, 'retrieval_hit'):.3f} | "
            f"{_mean_bool(policy_rows, 'stale_state_exposure'):.3f} | "
            f"{context_tokens:.2f} | "
            f"{_mean(policy_rows, 'retrieved_context_tokens'):.2f} | "
            f"{_mean(policy_rows, 'latency_ms'):.2f} | "
            f"{((accuracy / context_tokens) * 1000.0 if context_tokens else 0.0):.3f} |"
        )


def _print_pairwise_summary(rows: list[dict[str, object]], *, baseline_policy: str) -> None:
    by_policy_case: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_policy_case[str(row["policy_name"])][str(row["case_id"])] = row

    baseline_rows = by_policy_case.get(baseline_policy, {})
    if not baseline_rows:
        print(f"No baseline rows found for policy '{baseline_policy}'.")
        return

    print(f"| policy | paired_n | delta_acc | delta_stale | delta_ctx | wins | losses | both_wrong |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for policy, case_rows in sorted(by_policy_case.items()):
        if policy == baseline_policy:
            continue
        paired = [
            (row, baseline_rows[case_id])
            for case_id, row in case_rows.items()
            if case_id in baseline_rows
        ]
        if not paired:
            continue
        wins = sum(1 for row, base in paired if bool(row["correct"]) and not bool(base["correct"]))
        losses = sum(1 for row, base in paired if not bool(row["correct"]) and bool(base["correct"]))
        both_wrong = sum(1 for row, base in paired if not bool(row["correct"]) and not bool(base["correct"]))
        delta_acc = _mean_values(
            (1.0 if bool(row["correct"]) else 0.0) - (1.0 if bool(base["correct"]) else 0.0)
            for row, base in paired
        )
        delta_stale = _mean_values(
            (1.0 if bool(row["stale_state_exposure"]) else 0.0)
            - (1.0 if bool(base["stale_state_exposure"]) else 0.0)
            for row, base in paired
        )
        delta_ctx = _mean_values(
            float(row.get("prompt_tokens", 0.0)) - float(base.get("prompt_tokens", 0.0))
            for row, base in paired
        )
        print(
            f"| {policy} | {len(paired)} | {delta_acc:.3f} | {delta_stale:.3f} | "
            f"{delta_ctx:.2f} | {wins} | {losses} | {both_wrong} |"
        )


def _print_category_summary(rows: list[dict[str, object]]) -> None:
    print("| policy | category | n | accuracy | stale_exposure | memory_tokens |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or row.get("query_mode") or "unknown")
        grouped[(str(row["policy_name"]), category)].append(row)

    for (policy, category), group_rows in sorted(grouped.items()):
        print(
            f"| {policy} | {category} | {len(group_rows)} | "
            f"{_mean_bool(group_rows, 'correct'):.3f} | "
            f"{_mean_bool(group_rows, 'stale_state_exposure'):.3f} | "
            f"{_mean(group_rows, 'retrieved_context_tokens'):.2f} |"
        )


def _group_by(rows: Iterable[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return _mean_values(float(row.get(key, 0.0)) for row in rows)


def _mean_bool(rows: list[dict[str, object]], key: str) -> float:
    return _mean_values(1.0 if bool(row.get(key, False)) else 0.0 for row in rows)


def _mean_values(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
