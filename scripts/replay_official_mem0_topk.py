#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an official_mem0 top-k reader baseline from existing case JSONL. "
            "This avoids rerunning expensive Mem0 ingestion/extraction."
        )
    )
    parser.add_argument("paths", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--policy-name", default="official_mem0_top2")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-dir", default=".cache/memory_inference_official_mem0_topk_replay")
    parser.add_argument("--prompt-template-id", default="validity-v1")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional max official_mem0 rows to replay.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    base_rows = _official_mem0_rows(_load_rows([Path(path) for path in args.paths]))
    if args.limit:
        base_rows = base_rows[: args.limit]
    if not base_rows:
        raise SystemExit("No official_mem0 rows found to replay.")

    reasoner = LocalHFReasoner(
        LocalModelConfig(
            model_id=args.model_id,
            cache_dir=Path(args.cache_dir),
            inference_batch_size=args.inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            context_window=args.context_window,
            device=args.device,
            dtype=args.dtype,
            prompt_template_id=args.prompt_template_id,
            use_chat_template=not args.no_chat_template,
        )
    )

    queries = [_row_to_query(row) for row in base_rows]
    contexts = [_row_to_records(row)[: args.top_k] for row in base_rows]
    traces = reasoner.answer_many_with_traces(queries, contexts)

    with output.open("w") as handle:
        for row, records, trace in zip(base_rows, contexts, traces):
            replayed = _replayed_row(
                row,
                records=records,
                trace=trace,
                policy_name=args.policy_name,
                top_k=args.top_k,
            )
            handle.write(json.dumps(replayed, sort_keys=True) + "\n")

    correct = sum(bool(row.get("correct")) for row in _load_rows([output]))
    print(
        f"wrote {output} rows={len(base_rows)} policy={args.policy_name} "
        f"top_k={args.top_k} accuracy={correct}/{len(base_rows)}"
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


def _official_mem0_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("policy_name") != "official_mem0":
            continue
        key = (str(row.get("category") or ""), str(row.get("case_id") or ""))
        by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _row_to_query(row: dict[str, Any]) -> RuntimeQuery:
    records = _row_to_records(row)
    return RuntimeQuery(
        query_id=str(row.get("query_id") or row.get("case_id") or "official_mem0_topk"),
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
                record_id=str(payload.get("record_id") or f"official-mem0-replay-{index}"),
                entity=str(payload.get("entity") or row.get("entity") or "user"),
                attribute=str(payload.get("attribute") or row.get("attribute") or "preference"),
                value=str(payload.get("value") or ""),
                timestamp=_as_int(payload.get("timestamp"), default=0),
                session_id=str(row.get("context_id") or row.get("case_id") or ""),
                status=_memory_status(str(payload.get("status") or "ACTIVE")),
                scope=str(payload.get("scope") or "default"),
                source_kind=str(payload.get("source_kind") or "official_mem0"),
                memory_kind=str(payload.get("memory_kind") or "retrieved_memory"),
            )
        )
    return records


def _replayed_row(
    row: dict[str, Any],
    *,
    records: list[MemoryRecord],
    trace,
    policy_name: str,
    top_k: int,
) -> dict[str, Any]:
    prediction = trace.answer
    gold = str(row.get("gold_answer") or "")
    replayed = dict(row)
    replayed.update(
        {
            "policy_name": policy_name,
            "prediction": prediction,
            "correct": answers_match(prediction, gold),
            "exact_match": answers_exact_match(prediction, gold),
            "span_match": answers_span_match(prediction, gold),
            "retrieved_items": len(records),
            "retrieved_context_tokens": sum(_token_count(record.text()) for record in records),
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "latency_ms": trace.latency_ms,
            "cache_hit": trace.cache_hit,
            "retrieval_mode": f"official_mem0_top{top_k}_replay",
            "base_retrieval_mode": row.get("retrieval_mode", ""),
            "official_mem0_topk_replay": top_k,
            "retrieved_records": [_record_payload(record) for record in records],
        }
    )
    return replayed


def _record_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "entity": record.entity,
        "attribute": record.attribute,
        "value": record.value,
        "timestamp": record.timestamp,
        "status": record.status.name,
        "scope": record.scope,
        "source_kind": record.source_kind,
        "memory_kind": record.memory_kind,
    }


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


def _token_count(text: str) -> int:
    return len(text.split())


if __name__ == "__main__":
    sys.exit(main())
