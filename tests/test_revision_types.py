from memory_inference.domain.enums import MemoryStatus, QueryMode, RevisionOp


def test_memory_status_has_all_required_values() -> None:
    expected = {"ACTIVE", "REINFORCED", "SUPERSEDED", "CONFLICTED", "ARCHIVED"}
    assert {s.name for s in MemoryStatus} == expected


def test_revision_op_has_all_required_values() -> None:
    expected = {"ADD", "REINFORCE", "REVISE", "REVERT", "SPLIT_SCOPE", "CONFLICT_UNRESOLVED", "LOW_CONFIDENCE", "NO_OP"}
    assert {op.name for op in RevisionOp} == expected


def test_query_mode_has_all_required_values() -> None:
    expected = {"CURRENT_STATE", "STATE_WITH_PROVENANCE", "HISTORY", "CONFLICT_AWARE"}
    assert {m.name for m in QueryMode} == expected
