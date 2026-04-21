from __future__ import annotations

from dataclasses import dataclass

from memory_inference.llm.base import BaseReasoner
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.orchestration.runner import ContextCaseRunner


@dataclass(slots=True)
class RunnerCache:
    """Central hook for runner reuse and future cache coordination."""

    def build_runner(
        self,
        *,
        policy: BaseMemoryPolicy,
        reasoner: BaseReasoner,
    ) -> ContextCaseRunner:
        return ContextCaseRunner(policy=policy, reasoner=reasoner)
