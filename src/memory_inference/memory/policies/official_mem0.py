from __future__ import annotations

import contextlib
import io
import json
import os
import re
import uuid
from typing import Any, Iterable, Iterator

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.odv2 import ODV2Policy


_DEFAULT_COLLECTION_NAME = f"official_mem0_{uuid.uuid4().hex}"
_MEM0_CLIENT_CACHE: dict[str, Any] = {}
_OFFICIAL_GATE_MODES = {"guard", "compact"}
_TERM_RE = re.compile(r"[a-z0-9]+")
_RANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "now",
    "of",
    "on",
    "or",
    "some",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
}


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
        self._last_add_debug: dict[str, str] = {}

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
                **self._last_add_debug,
            },
        )

    def snapshot_size(self) -> int:
        if not self._ingested:
            return 0
        stored_count = self._stored_memory_count()
        return stored_count if stored_count is not None else len(self.episodic_log)

    def _add_messages(self, messages: list[dict[str, str]]) -> None:
        client = self._ensure_client()
        prepared_messages = _prepare_messages_for_mem0(
            messages,
            max_chars=_env_int("MEM0_ADD_MAX_MESSAGE_CHARS", 2000),
        )
        if not prepared_messages:
            return

        batch_size = max(1, _env_int("MEM0_ADD_BATCH_SIZE", 8))
        batches = list(_batched(prepared_messages, batch_size))
        infer = _env_bool("MEM0_ADD_INFER", True)
        raw_fallback = _env_bool("MEM0_RAW_FALLBACK_ON_EMPTY", False)
        require_nonempty = _env_bool("MEM0_REQUIRE_NONEMPTY", False)
        fail_on_raw_fallback = _env_bool("MEM0_FAIL_ON_RAW_FALLBACK", False)

        for batch in batches:
            self._client_add(client, batch, infer=infer, add_mode="infer" if infer else "raw")

        stored_count = self._stored_memory_count()
        fallback_used = False
        fallback_blocked = False
        add_mode = "infer" if infer else "raw"
        if infer and raw_fallback and stored_count == 0:
            if fail_on_raw_fallback:
                fallback_blocked = True
                add_mode = "infer_failed_raw_fallback_blocked"
            else:
                for batch in batches:
                    self._client_add(client, batch, infer=False, add_mode="raw_fallback")
                stored_count = self._stored_memory_count()
                fallback_used = True
                add_mode = "infer_then_raw_fallback"

        self._last_add_debug = {
            "official_mem0_add_mode": add_mode,
            "official_mem0_add_batches": str(len(batches)),
            "official_mem0_add_messages": str(len(prepared_messages)),
            "official_mem0_raw_fallback": "1" if fallback_used else "0",
            "official_mem0_raw_fallback_blocked": "1" if fallback_blocked else "0",
            "official_mem0_stored_count": str(stored_count if stored_count is not None else -1),
        }
        if fallback_blocked:
            raise RuntimeError(
                "Official Mem0 used raw fallback after infer=True stored zero memories. "
                "This usually means the configured Mem0 LLM failed extraction, often because "
                "the Mem0 add batch is too large for the serving context window. Reduce "
                "MEM0_ADD_BATCH_SIZE or MEM0_ADD_MAX_MESSAGE_CHARS, lower "
                "MEM0_LLM_MAX_TOKENS, or restart vLLM with a larger --max-model-len. "
                f"debug={self._last_add_debug!r}"
            )
        if require_nonempty and stored_count == 0:
            raise RuntimeError(
                "Official Mem0 stored zero memories for a non-empty context. "
                "This makes the benchmark invalid. Check that the configured Mem0 LLM "
                "is running and can extract memories, or rerun with "
                "MEM0_RAW_FALLBACK_ON_EMPTY=true to store raw messages with infer=False."
            )

    def _client_add(
        self,
        client: Any,
        messages: list[dict[str, str]],
        *,
        infer: bool,
        add_mode: str,
    ) -> None:
        metadata = {
            "source": "validity-aware-memory",
            "add_mode": add_mode,
        }
        try:
            with _quiet_mem0_output():
                client.add(
                    messages,
                    user_id=self.user_id,
                    metadata=metadata,
                    infer=infer,
                )
        except TypeError:
            try:
                with _quiet_mem0_output():
                    client.add(messages, user_id=self.user_id, metadata=metadata)
            except TypeError:
                with _quiet_mem0_output():
                    client.add(messages, user_id=self.user_id)

    def _search(self, query: str, *, top_k: int) -> Any:
        client = self._ensure_client()
        attempts = (
            lambda: client.search(query, filters={"user_id": self.user_id}, limit=top_k),
            lambda: client.search(query, filters={"AND": [{"user_id": self.user_id}]}, limit=top_k),
            lambda: client.search(query, user_id=self.user_id, limit=top_k),
            lambda: client.search(query, filters={"user_id": self.user_id}),
            lambda: client.search(query, user_id=self.user_id),
        )
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                with _quiet_mem0_output():
                    return attempt()
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    def _stored_memory_count(self) -> int | None:
        client = self._ensure_client()
        get_all = getattr(client, "get_all", None)
        if get_all is None:
            return None
        attempts = (
            lambda: get_all(filters={"user_id": self.user_id}),
            lambda: get_all(filters={"AND": [{"user_id": self.user_id}]}),
            lambda: get_all(user_id=self.user_id),
        )
        for attempt in attempts:
            try:
                with _quiet_mem0_output():
                    return len(_normalize_mem0_results(attempt()))
            except (TypeError, ValueError):
                continue
        return None

    def _ensure_client(self) -> Any:
        if self.client is None:
            self.client = _build_mem0_client(self._config)
        return self.client


class OfficialMem0ODV2SelectivePolicy(BaseMemoryPolicy):
    """Official Mem0 retrieval with ODV2 as a post-retrieval gate.

    The default ``guard`` mode preserves the original conservative behavior:
    Mem0 remains the reader context unless ODV2 can remove a stale same-key
    value. ``compact`` mode is intended for the official Mem0 token-spend
    comparison: when ODV2 has relevant current-state evidence, it replaces the
    verbose Mem0 reader context with compact ODV2 state records.
    """

    def __init__(
        self,
        *,
        name: str = "official_mem0_odv2_selective",
        consolidator: BaseConsolidator,
        client: Any | None = None,
        config: dict[str, Any] | None = None,
        importance_threshold: float = 0.1,
        user_id: str | None = None,
        gate_mode: str | None = None,
        compact_top_k: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.gate_mode = _official_gate_mode(gate_mode)
        self.compact_top_k = max(
            1,
            compact_top_k
            if compact_top_k is not None
            else _env_int("OFFICIAL_MEM0_ODV2_COMPACT_TOP_K", 5),
        )
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
        if self.gate_mode == "compact":
            compact_records = _compact_records_for_query(
                query,
                current_entries,
                self.validity.episodic_log,
                limit=min(top_k, self.compact_top_k),
            )
            if compact_records:
                return self._bundle(
                    compact_records,
                    base=base,
                    retrieval_mode="official_mem0_odv2_compact_current",
                    removed_count=max(0, len(records) - len(compact_records)),
                    appended_count=len(compact_records),
                    support_compacted=max(0, len(records) - len(compact_records)),
                    base_record_count=len(records),
                )

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
            base_record_count=len(records),
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
        appended_count: int = 0,
        support_compacted: int = 0,
        base_record_count: int | None = None,
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "official_mem0_odv2_gate_mode": self.gate_mode,
                "official_mem0_odv2_base_records": str(
                    len(base.records) if base_record_count is None else base_record_count
                ),
                "official_mem0_odv2_returned_records": str(len(records)),
                "validity_removed": str(removed_count),
                "validity_appended": str(appended_count),
                "support_compacted": str(support_compacted),
            },
        )


def official_mem0_local_config_from_env() -> dict[str, Any]:
    """Build a local-first Mem0 OSS config from environment variables."""

    collection_name = _safe_collection_name(
        os.getenv("MEM0_COLLECTION_NAME", _DEFAULT_COLLECTION_NAME)
    )
    embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "384"))
    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "ollama")
    embedder_provider = os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface")
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
            "Install with `pip install -e \".[official-mem0]\"`, then run "
            "with local providers such as MEM0_LLM_PROVIDER=ollama."
        ) from exc

    effective_config = config or official_mem0_local_config_from_env()
    _patch_mem0_vllm_response_format_if_needed(effective_config)
    reuse_client = _env_bool("MEM0_REUSE_CLIENT", True)
    cache_key = _mem0_config_cache_key(effective_config)
    if reuse_client and cache_key in _MEM0_CLIENT_CACHE:
        return _MEM0_CLIENT_CACHE[cache_key]

    with _quiet_mem0_output():
        client = Memory.from_config(effective_config)
    if reuse_client:
        _MEM0_CLIENT_CACHE[cache_key] = client
    return client


def _mem0_config_cache_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _patch_mem0_vllm_response_format_if_needed(config: dict[str, Any]) -> None:
    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict) or llm_config.get("provider") != "vllm":
        return
    if not _env_bool("MEM0_VLLM_DISABLE_RESPONSE_FORMAT", True):
        return
    try:
        from mem0.llms.vllm import VllmLLM
    except ImportError:
        return
    if getattr(VllmLLM, "_memory_inference_response_format_patch", False):
        return

    original_generate_response = VllmLLM.generate_response

    def generate_response_without_response_format(
        self,
        messages,
        response_format=None,
        tools=None,
        tool_choice="auto",
        **kwargs,
    ):
        return original_generate_response(
            self,
            messages,
            response_format=None,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    VllmLLM.generate_response = generate_response_without_response_format
    VllmLLM._memory_inference_response_format_patch = True


@contextlib.contextmanager
def _quiet_mem0_output():
    if not _env_bool("MEM0_QUIET", True):
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


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


def _prepare_messages_for_mem0(
    messages: list[dict[str, str]],
    *,
    max_chars: int,
) -> list[dict[str, str]]:
    if max_chars <= 0:
        return messages
    prepared: list[dict[str, str]] = []
    for message in messages:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = str(message.get("role", "user") or "user")
        for chunk in _split_text(content, max_chars=max_chars):
            prepared.append({"role": role, "content": chunk})
    return prepared


def _split_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _batched(items: list[dict[str, str]], batch_size: int) -> Iterator[list[dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


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
            if key not in raw_results:
                continue
            return _normalize_mem0_results(raw_results.get(key))
        return [raw_results] if _looks_like_mem0_memory(raw_results) else []
    if isinstance(raw_results, list):
        normalized: list[Any] = []
        for item in raw_results:
            normalized.extend(_normalize_mem0_results(item))
        return normalized
    return [raw_results] if str(raw_results).strip() else []


def _looks_like_mem0_memory(result: dict[str, Any]) -> bool:
    return any(
        str(result.get(key) or "").strip()
        for key in ("memory", "text", "value", "content")
    )


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


def _compact_records_for_query(
    query: RuntimeQuery,
    current_entries: list[MemoryRecord],
    episodic_log: list[MemoryRecord],
    *,
    limit: int,
) -> list[MemoryRecord]:
    decisive = _decisive_current_entries(query, current_entries)
    ranked_candidates = _rank_compact_candidates(query, episodic_log)

    records: list[MemoryRecord] = []
    if decisive and _current_record_query_score(query, decisive[0]) >= 2.0:
        records.extend(decisive[:1])
    records.extend(entry for _, entry in ranked_candidates)
    records = _dedupe_records_by_value(records)
    if not records:
        return []
    return records[:limit]


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


def _current_record_query_score(query: RuntimeQuery, entry: MemoryRecord) -> float:
    query_terms = _rank_terms(query.question)
    entry_terms = (
        _rank_terms(entry.value)
        | _rank_terms(entry.support_text)
        | _rank_terms(entry.attribute.replace("_", " "))
    )
    overlap = len(query_terms & entry_terms)
    if overlap <= 0:
        return 0.0
    status_bonus = 0.25 if entry.status.name in {"ACTIVE", "REINFORCED"} else 0.0
    support_bonus = 0.2 if entry.support_text else 0.0
    return float(overlap) + status_bonus + support_bonus + (entry.timestamp * 0.0001)


def _rank_compact_candidates(
    query: RuntimeQuery,
    records: Iterable[MemoryRecord],
) -> list[tuple[float, MemoryRecord]]:
    scored: list[tuple[float, MemoryRecord]] = []
    for entry in records:
        if not _record_matches_compact_query(entry, query):
            continue
        score = _current_record_query_score(query, entry)
        if score < 2.0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _record_matches_compact_query(record: MemoryRecord, query: RuntimeQuery) -> bool:
    if query.entity not in {"conversation", "all"} and record.entity != query.entity:
        return False
    if record.attribute != query.attribute:
        return False
    if record.attribute in {"dialogue", "event"}:
        return False
    return record.source_kind == "structured_fact" or record.memory_kind in {"state", "event"}


def _rank_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw_term in _TERM_RE.findall(text.lower()):
        if raw_term in _RANK_STOPWORDS or len(raw_term) <= 1:
            continue
        terms.add(raw_term)
        if len(raw_term) > 3 and raw_term.endswith("s"):
            terms.add(raw_term[:-1])
    return terms


def _dedupe_records_by_value(records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
    deduped: list[MemoryRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.entity, record.attribute, _normalize(record.value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


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


def _official_gate_mode(value: str | None = None) -> str:
    mode = (value or os.getenv("OFFICIAL_MEM0_ODV2_GATE_MODE", "guard")).strip().lower()
    if mode not in _OFFICIAL_GATE_MODES:
        allowed = ", ".join(sorted(_OFFICIAL_GATE_MODES))
        raise ValueError(
            f"Unknown OFFICIAL_MEM0_ODV2_GATE_MODE={mode!r}; expected one of: {allowed}"
        )
    return mode


def _safe_collection_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return safe or "official_mem0"


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default
