from __future__ import annotations

import json
from pathlib import Path

from memory_inference.domain.benchmark import (
    RawConversationSession,
    RawConversationTurn,
    RawLoCoMoQuestion,
    RawLoCoMoSample,
)


def load_locomo_samples(path: str | Path, *, limit: int | None = None) -> list[RawLoCoMoSample]:
    raw_data = json.loads(Path(path).read_text())
    if not isinstance(raw_data, list):
        raise ValueError("LoCoMo raw format expects a JSON array of records")

    samples: list[RawLoCoMoSample] = []
    for idx, item in enumerate(raw_data):
        if limit is not None and len(samples) >= limit:
            break
        samples.append(_parse_locomo_sample(item, idx))
    return samples


def _parse_locomo_sample(item: dict, index: int) -> RawLoCoMoSample:
    sample_id = str(item.get("sample_id", f"lc-{index}"))
    conversation = item.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation must be a JSON object")
    raw_event_summary = item.get("event_summary", {})
    if not isinstance(raw_event_summary, dict):
        raise ValueError("LoCoMo event_summary must be a JSON object")
    raw_qa = item.get("qa", [])
    if not isinstance(raw_qa, list):
        raise ValueError("LoCoMo qa must be a JSON array")

    session_keys = sorted(
        key for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    sessions: list[RawConversationSession] = []
    for session_key in session_keys:
        session_data = conversation.get(session_key, [])
        if not isinstance(session_data, list):
            continue
        turns = [
            RawConversationTurn(
                speaker=str(turn.get("speaker", "unknown")),
                text=str(turn.get("text", "")),
                turn_id=str(turn.get("dia_id", "")),
            )
            for turn in session_data
            if isinstance(turn, dict)
        ]
        sessions.append(
            RawConversationSession(
                label=session_key,
                date=str(conversation.get(f"{session_key}_date_time", "")),
                turns=turns,
            )
        )

    questions = [
        RawLoCoMoQuestion(
            question=str(qa.get("question", "")),
            answer="" if qa.get("answer") is None else str(qa.get("answer", "")),
            category=str(qa.get("category", "")),
        )
        for qa in raw_qa
        if isinstance(qa, dict)
    ]
    event_summary = {
        str(speaker): [str(event) for event in events] if isinstance(events, list) else []
        for speaker, events in raw_event_summary.items()
    }

    return RawLoCoMoSample(
        sample_id=sample_id,
        sessions=sessions,
        event_summary=event_summary,
        questions=questions,
    )
