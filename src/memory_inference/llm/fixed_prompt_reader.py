from __future__ import annotations

from typing import Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.base import BaseReasoner
from memory_inference.llm.deterministic_reader import DeterministicValidityReader


class FixedPromptReader(BaseReasoner):
    """Stable reader stub representing a frozen prompt-based LLM policy."""

    def __init__(self, prompt_template: str = "Answer from the provided memory context.") -> None:
        self.prompt_template = prompt_template
        self._deterministic_reader = DeterministicValidityReader()

    def answer(self, query: RuntimeQuery, context: Sequence[MemoryRecord]) -> str:
        # The prompt is held fixed across experiments; the answer heuristic is deterministic
        # so the memory layer remains the manipulated variable in this scaffold.
        return self._deterministic_reader.answer(query, context)
