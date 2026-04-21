from __future__ import annotations

from dataclasses import dataclass, field

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.targets import EvalTarget


@dataclass(slots=True)
class RawConversationTurn:
    speaker: str
    text: str
    turn_id: str = ""
    has_answer: bool = False


@dataclass(slots=True)
class RawConversationSession:
    label: str
    date: str
    turns: list[RawConversationTurn]


@dataclass(slots=True)
class RawLoCoMoQuestion:
    question: str
    answer: str
    category: str


@dataclass(slots=True)
class RawLoCoMoSample:
    sample_id: str
    sessions: list[RawConversationSession]
    event_summary: dict[str, list[str]]
    questions: list[RawLoCoMoQuestion]


@dataclass(slots=True)
class RawLongMemEvalRecord:
    question_id: str
    question_type: str
    question: str
    answer: str
    sessions: list[RawConversationSession]
    multi_attributes: tuple[str, ...] = ()


@dataclass(slots=True)
class ExperimentContext:
    context_id: str
    session_id: str
    updates: list[MemoryRecord]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentCase:
    case_id: str
    context_id: str
    runtime_query: RuntimeQuery
    eval_target: EvalTarget
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def query_id(self) -> str:
        return self.runtime_query.query_id
