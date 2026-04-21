from __future__ import annotations

import json
from pathlib import Path

from memory_inference.domain.benchmark import (
    RawConversationSession,
    RawConversationTurn,
    RawLongMemEvalRecord,
)


def load_longmemeval_records(path: str | Path, *, limit: int | None = None) -> list[RawLongMemEvalRecord]:
    raw_data = json.loads(Path(path).read_text())
    if not isinstance(raw_data, list):
        raise ValueError("LongMemEval raw format expects a JSON array of records")

    records: list[RawLongMemEvalRecord] = []
    for idx, item in enumerate(raw_data):
        if limit is not None and len(records) >= limit:
            break
        records.append(_parse_longmemeval_record(item, idx))
    return records


def _parse_longmemeval_record(item: dict, index: int) -> RawLongMemEvalRecord:
    question_id = str(item.get("question_id", f"lme-{index}"))
    question_type = str(item.get("question_type", ""))
    question = str(item.get("question", ""))
    answer = str(item.get("answer", ""))
    multi_attributes = tuple(item.get("multi_attributes", []) or [])
    raw_sessions = item.get("haystack_sessions", [])
    haystack_dates = item.get("haystack_dates", []) or []
    haystack_session_ids = item.get("haystack_session_ids", []) or []

    sessions: list[RawConversationSession] = []
    for session_idx, session in enumerate(_coerce_turn_groups(raw_sessions)):
        turns = [
            RawConversationTurn(
                speaker=str(turn.get("role", "unknown")),
                text=str(turn.get("content", "")),
                has_answer=bool(turn.get("has_answer", False)),
            )
            for turn in session
            if isinstance(turn, dict)
        ]
        sessions.append(
            RawConversationSession(
                label=str(haystack_session_ids[session_idx]) if session_idx < len(haystack_session_ids) else f"session_{session_idx}",
                date=str(haystack_dates[session_idx]) if session_idx < len(haystack_dates) else "",
                turns=turns,
            )
        )

    return RawLongMemEvalRecord(
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        sessions=sessions,
        multi_attributes=multi_attributes,
    )


def _coerce_turn_groups(raw_sessions: object) -> list[list[dict]]:
    if not isinstance(raw_sessions, list):
        raise ValueError("haystack_sessions must be a list")
    if not raw_sessions:
        return []

    first = raw_sessions[0]
    if isinstance(first, dict):
        return [raw_sessions]
    if isinstance(first, list):
        normalized: list[list[dict]] = []
        for idx, session in enumerate(raw_sessions):
            if not isinstance(session, list):
                raise ValueError(f"haystack_sessions[{idx}] must be a list of turns")
            normalized.append(session)
        return normalized
    raise ValueError("haystack_sessions must contain turn dicts or lists of turn dicts")
