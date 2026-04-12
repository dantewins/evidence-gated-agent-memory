from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.metrics import ABSTAIN_TOKEN
from memory_inference.types import MemoryEntry, Query


@dataclass(slots=True)
class PromptPackage:
    prompt: str
    template_id: str
    system_prompt: str
    user_prompt: str
    messages: tuple[dict[str, str], ...]


def build_reasoning_prompt(
    query: Query,
    context: Sequence[MemoryEntry],
    *,
    template_id: str = "validity-v1",
    system_prompt: str = (
        "You are evaluating a frozen-weight memory system. "
        "Answer strictly from the supplied external memory."
    ),
) -> PromptPackage:
    instruction = _instruction_for_query(query)
    memory_block = _format_context(context)
    user_prompt = (
        f"Task: {instruction}\n"
        f"Question: {query.question}\n"
        f"Memory:\n{memory_block}\n"
        "Answer with a short span only.\n"
    )
    if query.supports_abstention or query.query_mode == QueryMode.CONFLICT_AWARE:
        user_prompt += (
            f"If the latest evidence is unresolved or contradictory, answer exactly {ABSTAIN_TOKEN}.\n"
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


def _instruction_for_query(query: Query) -> str:
    if query.query_mode == QueryMode.HISTORY:
        return "Return the historically relevant value from the provided timeline."
    if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
        return "Return the current value supported by the most relevant provenance."
    if query.query_mode == QueryMode.CONFLICT_AWARE:
        return "Return the current value only if the latest evidence is not in conflict."
    return "Return the current valid value from memory."


def _format_context(context: Sequence[MemoryEntry]) -> str:
    if not context:
        return "(no memory retrieved)"
    return "\n".join(
        f"- entity={entry.entity}; relation={entry.attribute}; value={entry.value}; "
        f"scope={entry.scope}; timestamp={entry.timestamp}; status={entry.status.name}"
        f"{_format_metadata(entry)}"
        for entry in context
    )


def _format_metadata(entry: MemoryEntry) -> str:
    if not entry.metadata:
        return ""
    visible_items = []
    for key in ("source_date", "session_label", "session_id", "speaker"):
        value = entry.metadata.get(key)
        if value:
            visible_items.append(f"{key}={value}")
    if not visible_items:
        return ""
    return "; " + "; ".join(visible_items)
