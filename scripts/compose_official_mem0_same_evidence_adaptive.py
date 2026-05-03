#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REVISION_SIGNALS = (
    "now",
    "currently",
    "recently",
    "new",
    "changed",
    "updated",
    "revised",
    "latest",
    "moved",
    "switched",
    "instead",
    "no longer",
    "previously",
    "formerly",
    "used to",
    "current",
)

ANSWER_VALUE_RE = re.compile(
    r"(?:\$\s*)?\b\d+(?:[,:.]\d+)?"
    r"(?:\s?(?:am|pm|kg|lb|lbs|%|percent|minutes?|hours?|days?|weeks?|months?|years?|miles?|km))?\b",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compose an exact same-evidence adaptive reader-budget policy from official Mem0 "
            "top-k replay rows. Routing uses only the official_mem0 top-5 retrieved records, "
            "then selects a precomputed official_mem0_top1 or official_mem0_top3 reader row."
        )
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--policy-name", default="official_mem0_same_evidence_adaptive")
    parser.add_argument("--low-risk-policy", default="official_mem0_top1")
    parser.add_argument("--high-risk-policy", default="official_mem0_top3")
    parser.add_argument("--risk-candidate-k", type=int, default=4)
    parser.add_argument("--revision-signal-threshold", type=int, default=1)
    parser.add_argument("--distinct-value-threshold", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.risk_candidate_k < 1:
        parser.error("--risk-candidate-k must be >= 1")
    if args.revision_signal_threshold < 1:
        parser.error("--revision-signal-threshold must be >= 1")
    if args.distinct_value_threshold < 1:
        parser.error("--distinct-value-threshold must be >= 1")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows([Path(path) for path in args.paths])
    by_key = _rows_by_key(rows)
    base_rows = _policy_rows(rows, "official_mem0")
    if not base_rows:
        raise SystemExit("No official_mem0 rows found.")

    routed_rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    for base in base_rows:
        route = _risk_route(
            base,
            risk_candidate_k=args.risk_candidate_k,
            revision_signal_threshold=args.revision_signal_threshold,
            distinct_value_threshold=args.distinct_value_threshold,
        )
        selected_policy = args.high_risk_policy if route["same_evidence_high_risk"] else args.low_risk_policy
        selected = by_key.get(_row_key(base, selected_policy))
        if selected is None:
            missing[selected_policy] += 1
            selected_policy = "official_mem0"
            selected = base
        route_counts[selected_policy] += 1
        routed = dict(selected)
        routed.update(
            {
                "policy_name": args.policy_name,
                "source_policy_name": selected_policy,
                "retrieval_mode": (
                    "official_mem0_same_evidence_high_risk"
                    if route["same_evidence_high_risk"]
                    else "official_mem0_same_evidence_low_risk"
                ),
                "same_evidence_gate_high_risk_policy": args.high_risk_policy,
                "same_evidence_gate_low_risk_policy": args.low_risk_policy,
                "same_evidence_gate_source_policy": selected_policy,
                "same_evidence_risk_candidate_k": args.risk_candidate_k,
                **route,
            }
        )
        routed_rows.append(routed)

    with output.open("w") as handle:
        for row in routed_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _print_summary(
        output=output,
        base_rows=base_rows,
        routed_rows=routed_rows,
        route_counts=route_counts,
        missing=missing,
        policy_name=args.policy_name,
    )
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
    return {_row_key(row, str(row.get("policy_name") or "")): row for row in rows}


def _policy_rows(rows: list[dict[str, Any]], policy_name: str) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("policy_name") != policy_name:
            continue
        keyed[(str(row.get("category") or ""), str(row.get("case_id") or ""))] = row
    return [keyed[key] for key in sorted(keyed)]


def _row_key(row: dict[str, Any], policy_name: str) -> tuple[str, str, str]:
    return (
        str(row.get("category") or ""),
        str(row.get("case_id") or ""),
        policy_name,
    )


def _risk_route(
    row: dict[str, Any],
    *,
    risk_candidate_k: int,
    revision_signal_threshold: int,
    distinct_value_threshold: int,
) -> dict[str, Any]:
    values = [
        str(record.get("value") or "")
        for record in (row.get("retrieved_records") or [])[:risk_candidate_k]
        if isinstance(record, dict)
    ]
    text = " ".join(value.lower() for value in values)
    revision_signals = [signal for signal in REVISION_SIGNALS if signal in text]
    answer_values = sorted(
        {
            match.group(0).strip().lower()
            for value in values
            for match in ANSWER_VALUE_RE.finditer(value)
        }
    )
    high_risk = (
        len(revision_signals) >= revision_signal_threshold
        or len(answer_values) >= distinct_value_threshold
    )
    return {
        "same_evidence_high_risk": high_risk,
        "same_evidence_revision_signal_count": len(revision_signals),
        "same_evidence_revision_signals": revision_signals,
        "same_evidence_distinct_answer_value_count": len(answer_values),
        "same_evidence_answer_values": answer_values,
    }


def _print_summary(
    *,
    output: Path,
    base_rows: list[dict[str, Any]],
    routed_rows: list[dict[str, Any]],
    route_counts: Counter[str],
    missing: Counter[str],
    policy_name: str,
) -> None:
    by_category: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for base, routed in zip(base_rows, routed_rows):
        by_category[str(base.get("category") or "")].append((base, routed))

    base_prompt = sum(_as_int(row.get("prompt_tokens")) for row in base_rows)
    prompt = sum(_as_int(row.get("prompt_tokens")) for row in routed_rows)
    base_total = sum(_reader_tokens(row) for row in base_rows)
    total = sum(_reader_tokens(row) for row in routed_rows)
    correct = sum(bool(row.get("correct")) for row in routed_rows)
    base_correct = sum(bool(row.get("correct")) for row in base_rows)
    wins = sum((not bool(base.get("correct"))) and bool(routed.get("correct")) for base, routed in zip(base_rows, routed_rows))
    losses = sum(bool(base.get("correct")) and not bool(routed.get("correct")) for base, routed in zip(base_rows, routed_rows))

    print(f"wrote {output} rows={len(routed_rows)} policy={policy_name}")
    print(f"routes={dict(sorted(route_counts.items()))}")
    if missing:
        print(f"missing_policy_fallbacks={dict(sorted(missing.items()))}")
    print(f"accuracy={correct}/{len(routed_rows)} ({correct / len(routed_rows):.3f}) base={base_correct}/{len(base_rows)}")
    print(f"wins_losses=wins={wins} losses={losses}")
    print(f"prompt_tokens={prompt} base={base_prompt} delta={prompt - base_prompt} delta_pct={_pct(prompt, base_prompt):.2f}%")
    print(f"reader_total_tokens={total} base={base_total} delta={total - base_total} delta_pct={_pct(total, base_total):.2f}%")
    print("by_category")
    for category, pairs in sorted(by_category.items()):
        category_correct = sum(bool(routed.get("correct")) for _, routed in pairs)
        category_base_correct = sum(bool(base.get("correct")) for base, _ in pairs)
        category_total = sum(_reader_tokens(routed) for _, routed in pairs)
        category_base_total = sum(_reader_tokens(base) for base, _ in pairs)
        print(
            f"  {category}: n={len(pairs)} correct={category_correct}/{len(pairs)} "
            f"base={category_base_correct}/{len(pairs)} reader_delta_pct={_pct(category_total, category_base_total):.2f}%"
        )


def _reader_tokens(row: dict[str, Any]) -> int:
    return _as_int(row.get("prompt_tokens")) + _as_int(row.get("completion_tokens"))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pct(value: int, base: int) -> float:
    if not base:
        return 0.0
    return (value - base) / base * 100.0


if __name__ == "__main__":
    sys.exit(main())
