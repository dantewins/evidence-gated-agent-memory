from __future__ import annotations

import os
import re
import uuid
from typing import Any, Iterable

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.odv2 import ODV2Policy


class OfficialMem0Policy(BaseMemoryPolicy):
    """Adapter around Mem0 OSS that fits the repository policy interface.

    The adapter is optional: importing the repository should not require
    mem0ai, Qdrant, Ollama, or vLLM. Real runs instantiate Mem0 lazily from a
    local-first config, while callers may inject an already configured client.
    """

    def __init__(
        self,
        *,
        name: str = "official_mem0",
        client: Any | None = None,
        config: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.user_id = user_id or f"{name}-{uuid.uuid4().hex}"
        self.client = client
        self._config = config
        self.episodic_log: list[MemoryRecord] = []
        self._ingested = False

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        update_list = list(updates)
        if not update_list:
            return
        self.episodic_log.extend(update_list)
        messages = _records_to_messages(update_list)
        if not messages:
            return
        self._add_messages(messages)
        self._ingested = True

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id=f"{self.name}-retrieve",
            context_id=f"{self.name}-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=max((record.timestamp for record in self.episodic_log), default=0),
            session_id=f"{self.name}-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        raw_results = self._search(query.question, top_k=top_k)
        records = [
            _result_to_record(result, query=query, index=index)
            for index, result in enumerate(_normalize_mem0_results(raw_results))
        ]
        return RetrievalBundle(
            records=records[:top_k],
            debug={
                "policy": self.name,
                "retrieval_mode": "official_mem0_search",
                "official_mem0_results": str(len(records)),
            },
        )

    def snapshot_size(self) -> int:
        if not self._ingested:
            return 0
        client = self._ensure_client()
        get_all = getattr(client, "get_all", None)
        if get_all is None:
            return len(self.episodic_log)
        try:
            return len(_normalize_mem0_results(get_all(user_id=self.user_id)))
        except Exception:
            return len(self.episodic_log)

    def _add_messages(self, messages: list[dict[str, str]]) -> None:
        client = self._ensure_client()
        try:
            client.add(
                messages,
                user_id=self.user_id,
                metadata={"source": "validity-aware-memory"},
            )
        except TypeError:
            client.add(messages, user_id=self.user_id)

    def _search(self, query: str, *, top_k: int) -> Any:
        client = self._ensure_client()
        try:
            return client.search(query, user_id=self.user_id, limit=top_k)
        except TypeError:
            return client.search(query, user_id=self.user_id)

    def _ensure_client(self) -> Any:
        if self.client is None:
            self.client = _build_mem0_client(self._config)
        return self.client


class OfficialMem0ODV2SelectivePolicy(BaseMemoryPolicy):
    """Official Mem0 retrieval with ODV2 as a conservative post-retrieval gate."""

    def __init__(
        self,
        *,
        name: str = "official_mem0_odv2_selective",
        consolidator: BaseConsolidator,
        client: Any | None = None,
        config: dict[str, Any] | None = None,
        importance_threshold: float = 0.1,
        user_id: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.retriever = OfficialMem0Policy(
            name=f"{name}::official_mem0",
            client=client,
            config=config,
            user_id=user_id,
        )
        self.validity = ODV2Policy(
            name=f"{name}::validity",
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=1,
        )

    @property
    def current_state(self):
        return self.validity.current_state

    @property
    def archive(self):
        return self.validity.archive

    @property
    def conflict_table(self):
        return self.validity.conflict_table

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        update_list = list(updates)
        if not update_list:
            return
        self.retriever.ingest(update_list)
        self.validity.ingest(update_list)

    def maybe_consolidate(self) -> None:
        self.validity.maybe_consolidate()
        self.maintenance_tokens = self.retriever.maintenance_tokens + self.validity.maintenance_tokens
        self.maintenance_latency_ms = (
            self.retriever.maintenance_latency_ms + self.validity.maintenance_latency_ms
        )
        self.maintenance_calls = self.retriever.maintenance_calls + self.validity.maintenance_calls

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        return self.retriever.retrieve(entity, attribute, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        base = self.retriever.retrieve_for_query(query, top_k=top_k)
        records = list(base.records)
        if not _query_allows_official_mem0_gate(query):
            return self._bundle(records, base=base, retrieval_mode="official_mem0_odv2_passthrough")

        current_entries = self.validity.current_entries_for_query(query)
        archive_entries = self.validity.archive_entries_for_query(query)
        decisive_current = _decisive_current_entries(query, current_entries)
        if not decisive_current or not archive_entries:
            return self._bundle(records, base=base, retrieval_mode="official_mem0_odv2_passthrough")
        if not _official_records_contain_any(records, decisive_current):
            return self._bundle(records, base=base, retrieval_mode="official_mem0_odv2_passthrough")

        filtered, removed = _remove_archived_values_from_official_records(
            records,
            query=query,
            current_entries=decisive_current,
            archive_entries=archive_entries,
        )
        return self._bundle(
            filtered[:top_k],
            base=base,
            retrieval_mode=(
                "official_mem0_odv2_guard"
                if removed
                else "official_mem0_odv2_passthrough"
            ),
            removed_count=removed,
        )

    def snapshot_size(self) -> int:
        return self.retriever.snapshot_size() + self.validity.snapshot_size()

    def _bundle(
        self,
        records: list[MemoryRecord],
        *,
        base: RetrievalBundle,
        retrieval_mode: str,
        removed_count: int = 0,
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "validity_removed": str(removed_count),
                "validity_appended": "0",
                "support_compacted": "0",
            },
        )


def official_mem0_local_config_from_env() -> dict[str, Any]:
    """Build a local-first Mem0 OSS config from environment variables."""

    collection_name = _safe_collection_name(
        os.getenv("MEM0_COLLECTION_NAME", f"official_mem0_{uuid.uuid4().hex}")
    )
    embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "768"))
    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "ollama")
    embedder_provider = os.getenv("MEM0_EMBEDDER_PROVIDER", "ollama")
    vector_provider = os.getenv("MEM0_VECTOR_STORE_PROVIDER", "qdrant")

    return {
        "vector_store": {
            "provider": vector_provider,
            "config": _vector_store_config(
                provider=vector_provider,
                collection_name=collection_name,
                embedding_dims=embedding_dims,
            ),
        },
        "llm": {
            "provider": llm_provider,
            "config": _llm_config(provider=llm_provider),
        },
        "embedder": {
            "provider": embedder_provider,
            "config": _embedder_config(provider=embedder_provider),
        },
    }


def _build_mem0_client(config: dict[str, Any] | None = None) -> Any:
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise ImportError(
            "official_mem0 requires the optional Mem0 OSS dependency. "
            "Install with `pip install mem0ai qdrant-client ollama`, then run "
            "with local providers such as MEM0_LLM_PROVIDER=ollama."
        ) from exc
    return Memory.from_config(config or official_mem0_local_config_from_env())


def _vector_store_config(
    *,
    provider: str,
    collection_name: str,
    embedding_dims: int,
) -> dict[str, Any]:
    if provider == "qdrant":
        config: dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_model_dims": embedding_dims,
        }
        if os.getenv("MEM0_QDRANT_URL"):
            config["url"] = os.getenv("MEM0_QDRANT_URL")
        elif os.getenv("MEM0_QDRANT_HOST"):
            config["host"] = os.getenv("MEM0_QDRANT_HOST")
            config["port"] = int(os.getenv("MEM0_QDRANT_PORT", "6333"))
        else:
            config["path"] = os.getenv(
                "MEM0_QDRANT_PATH",
                ".cache/memory_inference_official_mem0_qdrant",
            )
        return config
    return {"collection_name": collection_name}


def _llm_config(*, provider: str) -> dict[str, Any]:
    model = os.getenv("MEM0_LLM_MODEL", "llama3.1:8b")
    config: dict[str, Any] = {
        "model": model,
        "temperature": float(os.getenv("MEM0_LLM_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("MEM0_LLM_MAX_TOKENS", "2000")),
    }
    if provider == "ollama":
        config["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "vllm":
        config["vllm_base_url"] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        config["api_key"] = os.getenv("VLLM_API_KEY", "local-vllm")
    elif provider == "lmstudio":
        config["lmstudio_base_url"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    return config


def _embedder_config(*, provider: str) -> dict[str, Any]:
    if provider == "ollama":
        return {
            "model": os.getenv("MEM0_EMBEDDER_MODEL", "nomic-embed-text:latest"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
    if provider == "huggingface":
        return {
            "model": os.getenv("MEM0_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        }
    if provider == "lmstudio":
        return {
            "model": os.getenv("MEM0_EMBEDDER_MODEL", "text-embedding-nomic-embed-text-v1.5"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        }
    return {"model": os.getenv("MEM0_EMBEDDER_MODEL", "nomic-embed-text:latest")}


def _records_to_messages(records: list[MemoryRecord]) -> list[dict[str, str]]:
    dialogue_records = [
        record for record in records if record.attribute in {"dialogue", "event"}
    ]
    source_records = dialogue_records or records
    messages: list[dict[str, str]] = []
    for record in sorted(source_records, key=lambda entry: (entry.timestamp, entry.record_id)):
        content = record.value.strip()
        if not content:
            continue
        if record.attribute not in {"dialogue", "event"}:
            content = f"{record.entity}'s {record.attribute} is {record.value}."
        messages.append(
            {
                "role": _speaker_to_role(record.speaker),
                "content": content,
            }
        )
    return messages


def _speaker_to_role(speaker: str) -> str:
    normalized = speaker.strip().lower()
    if normalized in {"assistant", "agent", "bot"}:
        return "assistant"
    return "user"


def _normalize_mem0_results(raw_results: Any) -> list[Any]:
    if raw_results is None:
        return []
    if isinstance(raw_results, dict):
        for key in ("results", "memories", "data"):
            value = raw_results.get(key)
            if isinstance(value, list):
                return value
        return [raw_results]
    if isinstance(raw_results, list):
        return raw_results
    return [raw_results]


def _result_to_record(result: Any, *, query: RuntimeQuery, index: int) -> MemoryRecord:
    result_id = ""
    metadata: dict[str, str] = {}
    if isinstance(result, dict):
        result_id = str(result.get("id") or result.get("memory_id") or index)
        value = str(
            result.get("memory")
            or result.get("text")
            or result.get("value")
            or result.get("content")
            or ""
        )
        raw_metadata = result.get("metadata") or {}
        if isinstance(raw_metadata, dict):
            metadata = {str(key): str(value) for key, value in raw_metadata.items()}
        if result.get("score") is not None:
            metadata["score"] = str(result["score"])
    else:
        result_id = str(index)
        value = str(result)
    if not value:
        value = str(result)
    metadata["official_mem0_id"] = result_id
    return MemoryRecord(
        record_id=f"official-mem0-{query.query_id}-{index}-{result_id}",
        entity=query.entity,
        attribute=query.attribute,
        value=value,
        timestamp=query.timestamp,
        session_id=query.session_id,
        metadata=metadata,
        source_kind="official_mem0",
        memory_kind="retrieved_memory",
    )


def _query_allows_official_mem0_gate(query: RuntimeQuery) -> bool:
    if query.query_mode != QueryMode.CURRENT_STATE:
        return False
    if query.multi_attributes:
        return False
    if query.entity in {"conversation", "all"}:
        return False
    return query.attribute not in {"dialogue", "event"}


def _decisive_current_entries(
    query: RuntimeQuery,
    current_entries: list[MemoryRecord],
) -> list[MemoryRecord]:
    if not current_entries:
        return []
    if query.entity not in {"conversation", "all"}:
        current_values = {_normalize(entry.value) for entry in current_entries}
        if len(current_values) != 1:
            return []
    return current_entries[:1]


def _remove_archived_values_from_official_records(
    records: list[MemoryRecord],
    *,
    query: RuntimeQuery,
    current_entries: list[MemoryRecord],
    archive_entries: list[MemoryRecord],
) -> tuple[list[MemoryRecord], int]:
    current_values = {_normalize(entry.value) for entry in current_entries}
    current_timestamp = max(entry.timestamp for entry in current_entries)
    archived_values = {
        _normalize(entry.value)
        for entry in archive_entries
        if entry.attribute == query.attribute
        and entry.entity == query.entity
        and entry.timestamp < current_timestamp
    }
    if not archived_values:
        return records, 0
    filtered: list[MemoryRecord] = []
    removed = 0
    for record in records:
        text = _normalize(record.value)
        contains_current = any(value and value in text for value in current_values)
        contains_archived = any(value and value in text for value in archived_values)
        if contains_archived and not contains_current:
            removed += 1
            continue
        filtered.append(record)
    return filtered, removed


def _official_records_contain_any(
    records: list[MemoryRecord],
    entries: list[MemoryRecord],
) -> bool:
    values = {_normalize(entry.value) for entry in entries}
    return any(
        value and value in _normalize(record.value)
        for record in records
        for value in values
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _safe_collection_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return safe or "official_mem0"
