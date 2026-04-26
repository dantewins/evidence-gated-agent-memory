from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


Row = dict[str, object]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python scripts/summarize_validity_slices.py <cases.jsonl> [baseline_policy]")
        return 1
    rows = _load_rows(Path(argv[0]))
    if not rows:
        print("No diagnostic rows found.")
        return 1
    baseline_policy = argv[1] if len(argv) > 1 else "mem0"
    by_case_policy = _by_case_policy(rows)
    slices = _build_slices(by_case_policy, baseline_policy=baseline_policy)
    for name, case_ids in slices.items():
        if not case_ids:
            continue
        slice_rows = [row for row in rows if str(row["case_id"]) in case_ids]
        print(f"## {name} ({len(case_ids)} cases)")
        _print_policy_summary(slice_rows)
        print()
        _print_pairwise_summary(slice_rows, baseline_policy=baseline_policy)
        print()
    return 0


def _load_rows(path: Path) -> list[Row]:
    loaded: list[Row] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                loaded.append(json.loads(line))
    return loaded


def _by_case_policy(rows: Iterable[Row]) -> dict[str, dict[str, Row]]:
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["case_id"])][str(row["policy_name"])] = row
    return grouped


def _build_slices(
    by_case_policy: dict[str, dict[str, Row]],
    *,
    baseline_policy: str,
) -> dict[str, set[str]]:
    return {
        "all_paired": {
            case_id
            for case_id, policy_rows in by_case_policy.items()
            if baseline_policy in policy_rows
        },
        "baseline_stale_exposure": _baseline_cases(
            by_case_policy,
            baseline_policy,
            lambda row: bool(row.get("stale_state_exposure")),
        ),
        "baseline_same_key_conflict": _baseline_cases(
            by_case_policy,
            baseline_policy,
            _has_same_key_state_conflict,
        ),
        "baseline_hit_and_stale": _baseline_cases(
            by_case_policy,
            baseline_policy,
            lambda row: bool(row.get("retrieval_hit")) and bool(row.get("stale_state_exposure")),
        ),
        "validity_intervened": {
            case_id
            for case_id, policy_rows in by_case_policy.items()
            if any(_intervened(row) for row in policy_rows.values())
        },
    }


def _baseline_cases(
    by_case_policy: dict[str, dict[str, Row]],
    baseline_policy: str,
    predicate: Callable[[Row], bool],
) -> set[str]:
    return {
        case_id
        for case_id, policy_rows in by_case_policy.items()
        if baseline_policy in policy_rows and predicate(policy_rows[baseline_policy])
    }


def _intervened(row: Row) -> bool:
    retrieval_mode = str(row.get("retrieval_mode", ""))
    return (
        "guard" in retrieval_mode
        or "prune" in retrieval_mode
        or "compact" in retrieval_mode
        or _as_int(row.get("validity_removed")) > 0
        or _as_int(row.get("support_compacted")) > 0
        or _as_int(row.get("temporal_pruned")) > 0
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
        values.add(_normalize_value(str(record.get("value", ""))))
    values.discard("")
    return values


def _print_policy_summary(rows: list[Row]) -> None:
    print("| policy | n | accuracy | exact | retrieval_hit | stale_exposure | ctx_tokens | memory_tokens | interventions |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for policy, policy_rows in sorted(_group_by(rows, "policy_name").items()):
        print(
            f"| {policy} | {len(policy_rows)} | "
            f"{_mean_bool(policy_rows, 'correct'):.3f} | "
            f"{_mean_bool(policy_rows, 'exact_match'):.3f} | "
            f"{_mean_bool(policy_rows, 'retrieval_hit'):.3f} | "
            f"{_mean_bool(policy_rows, 'stale_state_exposure'):.3f} | "
            f"{_mean(policy_rows, 'prompt_tokens'):.2f} | "
            f"{_mean(policy_rows, 'retrieved_context_tokens'):.2f} | "
            f"{sum(1 for row in policy_rows if _intervened(row))} |"
        )


def _print_pairwise_summary(rows: list[Row], *, baseline_policy: str) -> None:
    by_policy_case: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        by_policy_case[str(row["policy_name"])][str(row["case_id"])] = row
    baseline_rows = by_policy_case.get(baseline_policy, {})
    if not baseline_rows:
        print(f"No baseline rows found for policy '{baseline_policy}'.")
        return
    print("| policy | paired_n | delta_acc | delta_stale | delta_ctx | wins | losses | both_wrong |")
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
        print(
            f"| {policy} | {len(paired)} | "
            f"{_mean_values(_bool(row, 'correct') - _bool(base, 'correct') for row, base in paired):.3f} | "
            f"{_mean_values(_bool(row, 'stale_state_exposure') - _bool(base, 'stale_state_exposure') for row, base in paired):.3f} | "
            f"{_mean_values(float(row.get('prompt_tokens', 0.0)) - float(base.get('prompt_tokens', 0.0)) for row, base in paired):.2f} | "
            f"{wins} | {losses} | {both_wrong} |"
        )


def _group_by(rows: Iterable[Row], key: str) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _mean(rows: list[Row], key: str) -> float:
    return _mean_values(float(row.get(key, 0.0)) for row in rows)


def _mean_bool(rows: list[Row], key: str) -> float:
    return _mean_values(_bool(row, key) for row in rows)


def _mean_values(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _bool(row: Row, key: str) -> float:
    return 1.0 if bool(row.get(key, False)) else 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_value(value: str) -> str:
    return " ".join(value.lower().split())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
