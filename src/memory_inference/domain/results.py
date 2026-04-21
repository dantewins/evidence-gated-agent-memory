from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from memory_inference.domain.benchmark import ExperimentCase
from memory_inference.domain.memory import RetrievalBundle


@dataclass(slots=True)
class ReaderTrace:
    answer: str
    model_id: str = "unknown"
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    raw_output: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutedCase:
    case: ExperimentCase
    retrieval_bundle: RetrievalBundle
    reader_trace: ReaderTrace
    prediction: str
    policy_name: str


@dataclass(slots=True)
class EvaluatedCase:
    case: ExperimentCase
    retrieval_bundle: RetrievalBundle
    reader_trace: ReaderTrace
    prediction: str
    correct: bool
    policy_name: str
