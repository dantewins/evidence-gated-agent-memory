from __future__ import annotations

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.memory.policies.official_mem0 import OfficialMem0Policy


def main() -> int:
    policy = OfficialMem0Policy(name="official_mem0_smoke")
    policy.ingest([
        MemoryRecord(
            record_id="smoke-1",
            entity="user",
            attribute="dialogue",
            value="My project codename is Redwood.",
            timestamp=1,
            session_id="smoke",
            speaker="user",
            memory_kind="event",
        )
    ])
    query = RuntimeQuery(
        query_id="smoke-q",
        context_id="smoke",
        entity="user",
        attribute="project_codename",
        question="What is my project codename?",
        timestamp=2,
        session_id="smoke",
        query_mode=QueryMode.CURRENT_STATE,
    )
    result = policy.retrieve_for_query(query, top_k=3)
    retrieved_text = "\n".join(record.value for record in result.records)
    if not result.records or "redwood" not in retrieved_text.lower():
        raise RuntimeError(
            "Official Mem0 smoke test failed: stored/searchable memory was not retrieved. "
            f"debug={result.debug!r}; retrieved={retrieved_text!r}"
        )
    print(
        "official_mem0 smoke ok: "
        f"records={len(result.records)} "
        f"stored={result.debug.get('official_mem0_stored_count', '')} "
        f"mode={result.debug.get('official_mem0_add_mode', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
