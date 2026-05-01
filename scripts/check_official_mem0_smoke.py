from __future__ import annotations

import os

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
    if (
        result.debug.get("official_mem0_raw_fallback") == "1"
        and not _env_bool("MEM0_ALLOW_RAW_FALLBACK_SMOKE", False)
    ):
        raise RuntimeError(
            "Official Mem0 smoke test used raw fallback. This run would not validate the "
            "configured Mem0 extraction LLM, so the benchmark is likely invalid. Reduce "
            "MEM0_ADD_BATCH_SIZE or MEM0_ADD_MAX_MESSAGE_CHARS, lower MEM0_LLM_MAX_TOKENS, "
            "or increase the vLLM --max-model-len. Set MEM0_ALLOW_RAW_FALLBACK_SMOKE=1 "
            f"only for explicit raw-storage debugging. debug={result.debug!r}"
        )
    print(
        "official_mem0 smoke ok: "
        f"records={len(result.records)} "
        f"stored={result.debug.get('official_mem0_stored_count', '')} "
        f"mode={result.debug.get('official_mem0_add_mode', '')}"
    )
    return 0


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
