#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from memory_inference.domain.enums import MemoryStatus, QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.scoring import (
    answers_exact_match,
    answers_match,
    answers_span_match,
)
from memory_inference.llm.local_config import LocalModelConfig
from memory_inference.llm.local_hf_reasoner import LocalHFReasoner


DEFAULT_POLICIES = [
    "official_mem0",
    "official_mem0_same_evidence_adaptive",
    "official_mem0_top1",
    "official_mem0_top2",
    "official_mem0_top3",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cache-free reader-only systems benchmark over existing official Mem0 "
            "case JSONL rows. This measures wall-clock reader replay without rerunning Mem0 ingestion."
        )
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--output-dir", help="Directory for benchmark CSV/JSONL outputs.")
    parser.add_argument("--policy", action="append", default=[], help="Policy to benchmark.")
    parser.add_argument("--limit", type=int, default=64, help="Number of shared cases to benchmark.")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-template-id", default="validity-v1")
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.paths]
    rows = _load_rows(input_paths)
    if not rows:
        raise SystemExit("No JSONL rows found.")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(input_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = args.policy or DEFAULT_POLICIES
    by_key = _by_key(rows)
    selected_cases = _select_shared_cases(by_key, policies, limit=args.limit, seed=args.seed)
    if not selected_cases:
        raise SystemExit(f"No shared cases found for policies: {', '.join(policies)}")

    reasoner = LocalHFReasoner(
        LocalModelConfig(
            model_id=args.model_id,
            cache_dir=None,
            inference_batch_size=args.inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            context_window=args.context_window,
            device=args.device,
            dtype=args.dtype,
            prompt_template_id=args.prompt_template_id,
            use_chat_template=not args.no_chat_template,
        )
    )
    reasoner._ensure_loaded()
    torch = getattr(reasoner, "_torch", None)

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = [by_key[(category, case_id, policy)] for category, case_id in selected_cases]
        queries = [_row_to_query(row) for row in policy_rows]
        contexts = [_row_to_records(row) for row in policy_rows]
        prompt_tokens_expected = sum(_reader_tokens(row) for row in policy_rows)

        _reset_gpu_peak(torch)
        _sync(torch)
        started = time.perf_counter()
        traces = reasoner.answer_many_with_traces(queries, contexts)
        _sync(torch)
        wall_ms = (time.perf_counter() - started) * 1000.0
        peak_memory_mib = _peak_memory_mib(torch)

        correct = 0
        trace_prompt_tokens = 0
        trace_completion_tokens = 0
        with (output_dir / f"{policy}_cache_free_traces.jsonl").open("w") as handle:
            for row, trace in zip(policy_rows, traces):
                gold = str(row.get("gold_answer") or "")
                is_correct = answers_match(trace.answer, gold)
                correct += int(is_correct)
                trace_prompt_tokens += trace.prompt_tokens
                trace_completion_tokens += trace.completion_tokens
                payload = {
                    "policy_name": policy,
                    "category": row.get("category"),
                    "case_id": row.get("case_id"),
                    "question": row.get("question"),
                    "gold_answer": gold,
                    "prediction": trace.answer,
                    "correct": is_correct,
                    "exact_match": answers_exact_match(trace.answer, gold),
                    "span_match": answers_span_match(trace.answer, gold),
                    "prompt_tokens": trace.prompt_tokens,
                    "completion_tokens": trace.completion_tokens,
                    "total_tokens": trace.total_tokens,
                    "latency_ms_trace": trace.latency_ms,
                    "cache_hit": trace.cache_hit,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                trace_rows.append(payload)

        total_trace_tokens = trace_prompt_tokens + trace_completion_tokens
        summary_rows.append(
            {
                "policy_name": policy,
                "cases": len(policy_rows),
                "correct": correct,
                "accuracy": f"{correct / len(policy_rows):.6f}",
                "wall_ms": f"{wall_ms:.2f}",
                "examples_per_second": f"{(len(policy_rows) / (wall_ms / 1000.0)):.4f}" if wall_ms else "0.0000",
                "tokens_per_second": f"{(total_trace_tokens / (wall_ms / 1000.0)):.2f}" if wall_ms else "0.00",
                "mean_wall_ms_per_case": f"{wall_ms / len(policy_rows):.2f}",
                "trace_prompt_tokens": trace_prompt_tokens,
                "trace_completion_tokens": trace_completion_tokens,
                "trace_total_tokens": total_trace_tokens,
                "input_row_reader_tokens": prompt_tokens_expected,
                "peak_allocated_memory_mib": f"{peak_memory_mib:.1f}" if peak_memory_mib is not None else "",
                "inference_batch_size": args.inference_batch_size,
                "context_window": args.context_window,
                "cache_dir": "disabled",
            }
        )

    _write_csv(output_dir / "cache_free_reader_systems.csv", summary_rows)
    _write_csv(output_dir / "cache_free_reader_traces.csv", trace_rows)
    _write_summary(output_dir / "cache_free_reader_systems.md", summary_rows, selected_cases)
    print(f"wrote {output_dir}/cache_free_reader_systems.csv")
    print(f"wrote {output_dir}/cache_free_reader_systems.md")
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


def _default_output_dir(input_paths: list[Path]) -> Path:
    if len(input_paths) == 1 and input_paths[0].is_dir():
        return input_paths[0] / "submission_checks"
    return Path("results") / "submission_checks"


def _by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
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


def _select_shared_cases(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    policies: list[str],
    *,
    limit: int,
    seed: int,
) -> list[tuple[str, str]]:
    shared: set[tuple[str, str]] | None = None
    for policy in policies:
        cases = {(category, case_id) for category, case_id, row_policy in by_key if row_policy == policy}
        shared = cases if shared is None else shared & cases
    case_list = sorted(shared or set())
    rng = random.Random(seed)
    rng.shuffle(case_list)
    return sorted(case_list[:limit])


def _row_to_query(row: dict[str, Any]) -> RuntimeQuery:
    records = _row_to_records(row)
    return RuntimeQuery(
        query_id=str(row.get("query_id") or row.get("case_id") or "cache_free_reader"),
        context_id=str(row.get("context_id") or row.get("case_id") or ""),
        entity=str(row.get("entity") or "user"),
        attribute=str(row.get("attribute") or "preference"),
        question=str(row.get("question") or ""),
        timestamp=max((record.timestamp for record in records), default=0),
        session_id=str(row.get("context_id") or row.get("case_id") or ""),
        query_mode=_query_mode(str(row.get("query_mode") or "CURRENT_STATE")),
        supports_abstention=_as_bool(row.get("supports_abstention")),
        metadata={"benchmark_category": str(row.get("category") or "")},
    )


def _row_to_records(row: dict[str, Any]) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for index, payload in enumerate(row.get("retrieved_records") or []):
        if not isinstance(payload, dict):
            continue
        records.append(
            MemoryRecord(
                record_id=str(payload.get("record_id") or f"official-mem0-cache-free-{index}"),
                entity=str(payload.get("entity") or row.get("entity") or "user"),
                attribute=str(payload.get("attribute") or row.get("attribute") or "preference"),
                value=str(payload.get("value") or payload.get("text") or payload.get("support_text") or ""),
                timestamp=_as_int(payload.get("timestamp"), default=0),
                session_id=str(row.get("context_id") or row.get("case_id") or ""),
                status=_memory_status(str(payload.get("status") or "ACTIVE")),
                scope=str(payload.get("scope") or "default"),
                source_kind=str(payload.get("source_kind") or "official_mem0"),
                memory_kind=str(payload.get("memory_kind") or "retrieved_memory"),
            )
        )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    selected_cases: list[tuple[str, str]],
) -> None:
    lines = [
        "# Cache-Free Reader Systems Benchmark",
        "",
        f"Shared cases replayed: {len(selected_cases)}",
        "",
        "| policy | cases | correct | wall ms | examples/s | tokens/s | mean ms/case | peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy_name} | {cases} | {correct} | {wall_ms} | {examples_per_second} | "
            "{tokens_per_second} | {mean_wall_ms_per_case} | {peak_allocated_memory_mib} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Cache is disabled in `LocalHFReasoner` for this benchmark. The rows replay existing retrieved records and do not rerun Mem0 ingestion or extraction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _reset_gpu_peak(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats()


def _sync(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def _peak_memory_mib(torch: Any) -> float | None:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None or not cuda.is_available():
        return None
    return float(cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def _reader_tokens(row: dict[str, Any]) -> int:
    return int(row.get("reader_total_tokens") or 0) or (
        int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
    )


def _query_mode(value: str) -> QueryMode:
    normalized = value.strip().upper().replace("-", "_")
    return QueryMode.__members__.get(normalized, QueryMode.CURRENT_STATE)


def _memory_status(value: str) -> MemoryStatus:
    normalized = value.strip().upper()
    return MemoryStatus.__members__.get(normalized, MemoryStatus.ACTIVE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    sys.exit(main())
