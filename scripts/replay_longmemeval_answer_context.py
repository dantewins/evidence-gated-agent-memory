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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LongMemEval questions with oracle answer-session context. "
            "This is a cheap reader/scorer sanity check, not a memory policy."
        )
    )
    parser.add_argument("cases", nargs="+", help="Result directories or *_cases.jsonl files.")
    parser.add_argument("--longmemeval", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--output-dir", help="Directory for sanity-check outputs.")
    parser.add_argument("--limit", type=int, default=64)
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

    case_paths = [Path(path) for path in args.cases]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(case_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = _official_mem0_rows(_load_rows(case_paths))
    raw_by_id = _load_longmemeval(Path(args.longmemeval))
    selected_rows = _select_rows_with_answer_context(result_rows, raw_by_id, limit=args.limit, seed=args.seed)
    if not selected_rows:
        raise SystemExit("No selected rows have answer_session_ids in the LongMemEval file.")

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

    queries = [_row_to_query(row) for row in selected_rows]
    contexts = [_answer_session_records(raw_by_id[str(row["case_id"])]) for row in selected_rows]
    started = time.perf_counter()
    traces = reasoner.answer_many_with_traces(queries, contexts)
    wall_ms = (time.perf_counter() - started) * 1000.0

    output_rows: list[dict[str, Any]] = []
    for row, records, trace in zip(selected_rows, contexts, traces):
        gold = str(row.get("gold_answer") or "")
        output_rows.append(
            {
                "policy_name": "oracle_answer_session_context",
                "category": row.get("category"),
                "case_id": row.get("case_id"),
                "question": row.get("question"),
                "gold_answer": gold,
                "prediction": trace.answer,
                "correct": answers_match(trace.answer, gold),
                "exact_match": answers_exact_match(trace.answer, gold),
                "span_match": answers_span_match(trace.answer, gold),
                "answer_session_records": len(records),
                "retrieved_context_tokens": sum(len(record.text().split()) for record in records),
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
                "latency_ms": trace.latency_ms,
            }
        )

    jsonl_path = output_dir / "oracle_answer_context_cases.jsonl"
    with jsonl_path.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_rows = _summary_rows(output_rows, wall_ms)
    _write_csv(output_dir / "oracle_answer_context_summary.csv", summary_rows)
    _write_markdown(output_dir / "oracle_answer_context_summary.md", summary_rows, output_rows, wall_ms)
    print(f"wrote {jsonl_path}")
    print(f"wrote {output_dir / 'oracle_answer_context_summary.md'}")
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
    by_case: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("policy_name") != "official_mem0":
            continue
        by_case[(str(row.get("category") or ""), str(row.get("case_id") or ""))] = row
    return [by_case[key] for key in sorted(by_case)]


def _load_longmemeval(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text())
    return {str(record["question_id"]): record for record in records}


def _select_rows_with_answer_context(
    rows: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("case_id") or "") in raw_by_id
        and raw_by_id[str(row.get("case_id") or "")].get("answer_session_ids")
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return sorted(eligible[:limit], key=lambda row: (str(row.get("category") or ""), str(row.get("case_id") or "")))


def _row_to_query(row: dict[str, Any]) -> RuntimeQuery:
    return RuntimeQuery(
        query_id=str(row.get("query_id") or row.get("case_id") or "oracle_answer_context"),
        context_id=str(row.get("context_id") or row.get("case_id") or ""),
        entity=str(row.get("entity") or "user"),
        attribute=str(row.get("attribute") or "dialogue"),
        question=str(row.get("question") or ""),
        timestamp=0,
        session_id=str(row.get("context_id") or row.get("case_id") or ""),
        query_mode=_query_mode(str(row.get("query_mode") or "CURRENT_STATE")),
        supports_abstention=_as_bool(row.get("supports_abstention")),
        metadata={"benchmark_category": str(row.get("category") or "")},
    )


def _answer_session_records(raw: dict[str, Any]) -> list[MemoryRecord]:
    answer_ids = set(str(session_id) for session_id in raw.get("answer_session_ids") or [])
    records: list[MemoryRecord] = []
    for index, (session_id, session) in enumerate(
        zip(raw.get("haystack_session_ids") or [], raw.get("haystack_sessions") or []),
        start=1,
    ):
        if str(session_id) not in answer_ids:
            continue
        records.append(
            MemoryRecord(
                record_id=f"{raw['question_id']}-answer-session-{index}",
                entity="user",
                attribute="answer_session",
                value=_session_text(session),
                timestamp=index,
                session_id=str(session_id),
                status=MemoryStatus.ACTIVE,
                scope="oracle_answer_context",
                source_kind="longmemeval_answer_session",
                memory_kind="event",
            )
        )
    return records


def _session_text(session: Any) -> str:
    if isinstance(session, str):
        return session
    if not isinstance(session, list):
        return json.dumps(session, ensure_ascii=True)
    answer_turns = [
        turn
        for turn in session
        if isinstance(turn, dict) and bool(turn.get("has_answer"))
    ]
    if answer_turns:
        session = answer_turns
    turns: list[str] = []
    for turn in session:
        if isinstance(turn, dict):
            role = str(turn.get("role") or turn.get("speaker") or "turn")
            content = str(turn.get("content") or turn.get("text") or "")
            turns.append(f"{role}: {content}")
        else:
            turns.append(str(turn))
    return "\n".join(turns)


def _summary_rows(rows: list[dict[str, Any]], wall_ms: float) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category") or ""), []).append(row)
    summaries: list[dict[str, Any]] = []
    for category, category_rows in [("all", rows)] + sorted(by_category.items()):
        correct = sum(bool(row.get("correct")) for row in category_rows)
        tokens = sum(int(row.get("total_tokens") or 0) for row in category_rows)
        summaries.append(
            {
                "category": category,
                "cases": len(category_rows),
                "correct": correct,
                "accuracy": f"{correct / len(category_rows):.6f}" if category_rows else "0.000000",
                "total_tokens": tokens,
                "tokens_per_correct": f"{tokens / correct:.2f}" if correct else "",
                "wall_ms_total_run": f"{wall_ms:.2f}" if category == "all" else "",
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    wall_ms: float,
) -> None:
    lines = [
        "# Oracle Answer-Session Context Replay",
        "",
        "This sanity check gives the reader only LongMemEval sessions marked as containing the answer.",
        "It is not a deployable memory policy and should be reported only as a reader/scorer sanity check.",
        "",
        f"Wall-clock replay time: {wall_ms:.2f} ms",
        "",
        "| category | cases | correct | accuracy | total tokens | tokens/correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {category} | {cases} | {correct} | {accuracy} | {total_tokens} | {tokens_per_correct} |".format(**row)
        )
    examples = [row for row in output_rows if bool(row.get("correct"))][:3]
    if examples:
        lines.extend(["", "## Correct Examples", ""])
        for row in examples:
            lines.append(
                f"- `{row['case_id']}`: gold={row['gold_answer']!r}, prediction={row['prediction']!r}"
            )
    path.write_text("\n".join(lines) + "\n")


def _default_output_dir(input_paths: list[Path]) -> Path:
    if len(input_paths) == 1 and input_paths[0].is_dir():
        return input_paths[0] / "submission_checks"
    return Path("results") / "submission_checks"


def _query_mode(value: str) -> QueryMode:
    normalized = value.strip().upper().replace("-", "_")
    return QueryMode.__members__.get(normalized, QueryMode.CURRENT_STATE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    sys.exit(main())
