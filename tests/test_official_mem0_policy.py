from memory_inference.domain.enums import QueryMode
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.memory.policies.official_mem0 import (
    OfficialMem0ODV2SelectivePolicy,
    OfficialMem0Policy,
    official_mem0_local_config_from_env,
)
from tests.factories import make_query, make_record


class FakeMem0Client:
    def __init__(self, search_results=None):
        self.add_calls = []
        self.search_calls = []
        self.search_results = search_results or []

    def add(self, messages, user_id, metadata=None):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata or {},
            }
        )

    def search(self, query, user_id=None, filters=None, limit=None):
        self.search_calls.append(
            {
                "query": query,
                "user_id": user_id or (filters or {}).get("user_id"),
                "filters": filters or {},
                "limit": limit,
            }
        )
        return {"results": self.search_results}

    def get_all(self, user_id):
        return {"results": self.search_results}


def test_official_mem0_config_defaults_to_local_llama_and_ollama(monkeypatch) -> None:
    monkeypatch.delenv("MEM0_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)
    monkeypatch.delenv("MEM0_EMBEDDER_PROVIDER", raising=False)
    monkeypatch.delenv("MEM0_EMBEDDER_MODEL", raising=False)

    config = official_mem0_local_config_from_env()

    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["config"]["model"] == "llama3.1:8b"
    assert config["embedder"]["provider"] == "huggingface"
    assert config["embedder"]["config"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert config["vector_store"]["provider"] == "qdrant"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 384


def test_official_mem0_policy_adapts_add_and_search_to_repo_records() -> None:
    client = FakeMem0Client(search_results=[{"id": "m1", "memory": "Alice lives in Boston"}])
    policy = OfficialMem0Policy(client=client, user_id="u")
    policy.ingest(
        [
            make_record(
                entry_id="turn-1",
                entity="Alice",
                attribute="dialogue",
                value="I moved to Boston.",
                speaker="user",
            )
        ]
    )

    result = policy.retrieve_for_query(
        make_query(
            query_id="q",
            entity="Alice",
            attribute="home_city",
            question="Where does Alice live now?",
            timestamp=2,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert client.add_calls[0]["messages"] == [{"role": "user", "content": "I moved to Boston."}]
    assert client.search_calls[0]["query"] == "Where does Alice live now?"
    assert result.debug["retrieval_mode"] == "official_mem0_search"
    assert result.entries[0].value == "Alice lives in Boston"
    assert result.entries[0].source_kind == "official_mem0"


def test_official_mem0_odv2_gate_removes_archived_value_when_current_is_present() -> None:
    client = FakeMem0Client(
        search_results=[
            {"id": "old", "memory": "Alice used to work at Google."},
            {"id": "new", "memory": "Alice now works at Meta."},
        ]
    )
    policy = OfficialMem0ODV2SelectivePolicy(
        client=client,
        consolidator=MockConsolidator(),
        user_id="u",
    )
    old_fact = make_record(
        entry_id="fact-google",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    current_fact = make_record(
        entry_id="fact-meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    policy.ingest([old_fact, current_fact])
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q",
            entity="Alice",
            attribute="employer",
            question="Where does Alice work now?",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert [entry.value for entry in result.entries] == ["Alice now works at Meta."]
    assert result.debug["retrieval_mode"] == "official_mem0_odv2_guard"
    assert result.debug["validity_removed"] == "1"


def test_official_mem0_odv2_gate_keeps_mem0_output_when_current_is_absent() -> None:
    client = FakeMem0Client(search_results=[{"id": "old", "memory": "Alice used to work at Google."}])
    policy = OfficialMem0ODV2SelectivePolicy(
        client=client,
        consolidator=MockConsolidator(),
        user_id="u",
    )
    old_fact = make_record(
        entry_id="fact-google",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    current_fact = make_record(
        entry_id="fact-meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        metadata={"source_kind": "structured_fact", "memory_kind": "state"},
    )
    policy.ingest([old_fact, current_fact])
    policy.maybe_consolidate()

    result = policy.retrieve_for_query(
        make_query(
            query_id="q",
            entity="Alice",
            attribute="employer",
            question="Where does Alice work now?",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        )
    )

    assert [entry.value for entry in result.entries] == ["Alice used to work at Google."]
    assert result.debug["retrieval_mode"] == "official_mem0_odv2_passthrough"
    assert result.debug["validity_removed"] == "0"
