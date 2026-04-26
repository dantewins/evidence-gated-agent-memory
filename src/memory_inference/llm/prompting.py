from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.metrics import ABSTAIN_TOKEN


@dataclass(slots=True)
class PromptPackage:
    prompt: str
    template_id: str
    system_prompt: str
    user_prompt: str
    messages: tuple[dict[str, str], ...]


def build_reasoning_prompt(
    query: RuntimeQuery,
    context: Sequence[MemoryRecord],
    *,
    template_id: str = "validity-v1",
    system_prompt: str = (
        "You are evaluating a frozen-weight memory system. "
        "Answer strictly from the supplied external memory."
    ),
) -> PromptPackage:
    instruction = _instruction_for_query(query)
    validity_rule = _validity_rule_for_query(query)
    memory_block = _format_context(context)
    user_prompt = (
        f"Task: {instruction}\n"
        f"Validity rule: {validity_rule}\n"
        f"Question: {query.question}\n"
        f"Memory:\n{memory_block}\n"
        "Answer with a short span only.\n"
    )
    if query.supports_abstention or query.query_mode == QueryMode.CONFLICT_AWARE:
        user_prompt += (
            f"If the memory does not contain enough evidence to answer, answer exactly {ABSTAIN_TOKEN}.\n"
        )
    prompt = f"System: {system_prompt}\nUser:\n{user_prompt}"
    return PromptPackage(
        prompt=prompt,
        template_id=template_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
    )


def render_prompt(
    package: PromptPackage,
    *,
    tokenizer: Any | None = None,
    use_chat_template: bool = False,
) -> str:
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(package.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return package.prompt
    return package.prompt

def _instruction_for_query(query: RuntimeQuery) -> str:
    if query.query_mode == QueryMode.HISTORY:
        return "Return the historically relevant value from the provided timeline."
    if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
        return "Return the current value supported by the most relevant provenance."
    if query.query_mode == QueryMode.CONFLICT_AWARE:
        return "Return the current value only if the latest evidence is not in conflict."
    return "Return the current valid value from memory."


def _validity_rule_for_query(query: RuntimeQuery) -> str:
    if query.query_mode == QueryMode.HISTORY:
        return (
            "Use the historically relevant record; older, superseded, or archived "
            "records may be correct when the question asks about the past."
        )
    if query.query_mode == QueryMode.CONFLICT_AWARE:
        return (
            "If same-entity records conflict, answer only when ACTIVE or REINFORCED "
            "latest evidence clearly resolves the conflict; otherwise ABSTAIN."
        )
    return (
        "For current-state questions, prefer ACTIVE or REINFORCED records over "
        "SUPERSEDED, ARCHIVED, or CONFLICTED records; when same-entity records "
        "conflict, use the latest non-stale value."
    )


def _format_context(context: Sequence[MemoryRecord]) -> str:
    if not context:
        return "(no memory retrieved)"
    return "\n".join(
        f"- entity={entry.entity}; relation={entry.attribute}; value={entry.value}; "
        f"scope={entry.scope}; timestamp={entry.timestamp}; status={entry.status.name}"
        f"{_format_metadata(entry)}"
        for entry in context
    )


def _format_metadata(entry: MemoryRecord) -> str:
    visible_items = []
    typed_values = {
        "source_date": getattr(entry, "source_date", ""),
        "session_label": getattr(entry, "session_label", ""),
        "session_id": entry.session_id,
        "speaker": getattr(entry, "speaker", ""),
        "source_kind": getattr(entry, "source_kind", ""),
        "memory_kind": getattr(entry, "memory_kind", ""),
    }
    for key, typed_value in typed_values.items():
        value = typed_value
        if value:
            visible_items.append(f"{key}={value}")
    support_text = getattr(entry, "support_text", "")
    if support_text:
        visible_items.append(f"support={_compact_support_text(support_text)}")
    if not visible_items:
        return ""
    return "; " + "; ".join(visible_items)


def _compact_support_text(text: str, *, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
