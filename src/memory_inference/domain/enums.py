from __future__ import annotations

from enum import Enum, auto


class MemoryStatus(Enum):
    ACTIVE = auto()
    REINFORCED = auto()
    SUPERSEDED = auto()
    CONFLICTED = auto()
    ARCHIVED = auto()


class RevisionOp(Enum):
    ADD = auto()
    REINFORCE = auto()
    REVISE = auto()
    REVERT = auto()
    SPLIT_SCOPE = auto()
    CONFLICT_UNRESOLVED = auto()
    LOW_CONFIDENCE = auto()
    NO_OP = auto()


class QueryMode(Enum):
    CURRENT_STATE = auto()
    STATE_WITH_PROVENANCE = auto()
    HISTORY = auto()
    CONFLICT_AWARE = auto()


class UpdateType(Enum):
    NEW = auto()
    REINFORCEMENT = auto()
    SUPERSESSION = auto()
    CONFLICT = auto()


__all__ = ["MemoryStatus", "QueryMode", "RevisionOp", "UpdateType"]
