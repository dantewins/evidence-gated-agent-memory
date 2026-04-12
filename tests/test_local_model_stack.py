import builtins
import json

import pytest

from memory_inference.benchmarks.longmemeval_preprocess import preprocess_longmemeval
from memory_inference.benchmarks.longmemeval_preprocess import load_preprocessed_longmemeval
from memory_inference.benchmarks.locomo_preprocess import load_preprocessed_locomo, preprocess_locomo
from memory_inference.cli import main as cli_main
from memory_inference.experiment_registry import policy_factory_by_name
from memory_inference.llm.base import BaseReasoner
from memory_inference.llm.cache import ResponseCache, cache_key
from memory_inference.llm.local_config import LocalModelConfig
from memory_inference.llm.local_hf_reasoner import LocalHFReasoner
from memory_inference.llm.prompting import build_reasoning_prompt, render_prompt
from memory_inference.llm.token_accounting import TokenUsage, count_tokens
from memory_inference.results import build_manifest
from memory_inference.types import MemoryEntry, Query


class DummyReasoner(BaseReasoner):
    def answer(self, query: Query, context):
        return "Boston"


def _query() -> Query:
    return Query(
        query_id="q1",
        entity="user_a",
        attribute="home_city",
        question="Where does user_a live now?",
        answer="Boston",
        timestamp=2,
        session_id="s1",
    )


def _context() -> list[MemoryEntry]:
    return [
        MemoryEntry(
            entry_id="e1",
            entity="user_a",
            attribute="home_city",
            value="Boston",
            timestamp=1,
            session_id="s1",
        )
    ]


def _raw_longmemeval_payload() -> list[dict]:
    return [
        {
            "question_id": "q_001",
            "question_type": "knowledge-update",
            "question": "What city does the user live in now?",
            "answer": "Boston",
            "haystack_sessions": [
                {"role": "user", "content": "I moved to Boston."},
            ],
            "haystack_session_ids": ["sess_1"],
            "haystack_dates": ["2024-01-20"],
        }
    ]


def _raw_locomo_payload() -> list[dict]:
    return [
        {
            "sample_id": "sample_001",
            "conversation": {
                "session_1": [
                    {"dia_id": 0, "speaker": "Alice", "text": "I got a new job at Google."},
                ],
                "session_1_date_time": "2024-01-10",
            },
            "event_summary": {
                "Alice": ["Got a job at Google"],
            },
            "qa": [
                {
                    "question": "Where does Alice work now?",
                    "answer": "Google",
                    "category": 1,
                }
            ],
        }
    ]


def test_base_reasoner_default_trace_wraps_answer() -> None:
    trace = DummyReasoner().answer_with_trace(_query(), _context())
    assert trace.answer == "Boston"
    assert trace.model_id == "DummyReasoner"


def test_prompt_builder_includes_question_and_memory() -> None:
    prompt = build_reasoning_prompt(_query(), _context())
    assert "Where does user_a live now?" in prompt.prompt
    assert "entity=user_a" in prompt.prompt
    rendered = render_prompt(prompt)
    assert "System:" in rendered


def test_token_count_falls_back_without_tokenizer() -> None:
    assert count_tokens("a b c") == 3
    usage = TokenUsage(prompt_tokens=2, completion_tokens=3)
    assert usage.total_tokens == 5


def test_response_cache_round_trip(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key("model", "prompt")
    from memory_inference.llm.base import ReasonerTrace

    cache.save(key, ReasonerTrace(answer="Boston", model_id="test"))
    loaded = cache.load(key)
    assert loaded is not None
    assert loaded.answer == "Boston"


def test_local_hf_reasoner_raises_helpful_error_without_dependencies() -> None:
    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake-model"))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError("missing dependency")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError):
            reasoner._ensure_loaded()
    finally:
        builtins.__import__ = real_import


def test_local_hf_reasoner_extracts_first_line_answer() -> None:
    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    answer = reasoner._extract_answer("System: x\nUser:y\nAnswer: Boston\nMore", "System: x\nUser:y\n")
    assert answer == "Boston"


def test_local_hf_reasoner_extracts_abstain() -> None:
    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    answer = reasoner._extract_answer("ABSTAIN due to conflict", "")
    assert answer == "ABSTAIN"


def test_local_hf_reasoner_decodes_completion_only() -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in token_ids)

    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    reasoner._tokenizer = DummyTokenizer()
    assert reasoner._decode_completion([10, 11, 12]) == "10,11,12"


def test_local_hf_reasoner_extracts_answer_after_assistant_label() -> None:
    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    answer = reasoner._extract_answer("assistant\nBoston\n", "")
    assert answer == "Boston"


def test_local_hf_reasoner_uses_dtype_load_kwarg() -> None:
    class DummyTorch:
        bfloat16 = "bf16"

    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake", dtype="bfloat16"))
    assert reasoner._model_load_kwargs(DummyTorch()) == {"device_map": "auto", "dtype": "bf16"}


def test_local_hf_reasoner_omits_sampling_kwargs_for_greedy_generation() -> None:
    class DummyTokenizer:
        eos_token_id = 7

    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    reasoner._tokenizer = DummyTokenizer()
    kwargs = reasoner._generate_kwargs()
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_local_hf_reasoner_resets_sampling_defaults_for_greedy_generation() -> None:
    class DummyGenerationConfig:
        do_sample = True
        max_new_tokens = None
        repetition_penalty = 1.0
        temperature = 0.7
        top_p = 0.9
        top_k = 20

    class DummyModel:
        generation_config = DummyGenerationConfig()

    reasoner = LocalHFReasoner(LocalModelConfig(model_id="fake"))
    reasoner._model = DummyModel()
    reasoner._configure_generation_defaults()
    config = reasoner._model.generation_config
    assert config.do_sample is False
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.top_k == 50


def test_preprocess_longmemeval_writes_cached_json(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "processed.json"
    source.write_text(json.dumps(_raw_longmemeval_payload()))
    batches = preprocess_longmemeval(source, output)
    assert len(batches) == 1
    payload = json.loads(output.read_text())
    assert payload["records"][0]["batch"]["queries"][0]["attribute"] == "home_city"
    reloaded = load_preprocessed_longmemeval(output)
    assert reloaded[0].queries[0].answer == "Boston"


def test_cli_preprocess_longmemeval(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "processed.json"
    source.write_text(json.dumps(_raw_longmemeval_payload()))
    cli_main(["preprocess-longmemeval", "--input", str(source), "--output", str(output)])
    assert output.exists()


def test_preprocess_locomo_writes_cached_json(tmp_path) -> None:
    source = tmp_path / "locomo_source.json"
    output = tmp_path / "locomo_processed.json"
    source.write_text(json.dumps(_raw_locomo_payload()))
    batches = preprocess_locomo(source, output)
    assert len(batches) == 1
    payload = json.loads(output.read_text())
    assert payload["records"][0]["batch"]["queries"][0]["attribute"] == "employer"
    reloaded = load_preprocessed_locomo(output)
    assert reloaded[0].queries[0].answer == "Google"


def test_cli_preprocess_locomo(tmp_path) -> None:
    source = tmp_path / "locomo_source.json"
    output = tmp_path / "locomo_processed.json"
    source.write_text(json.dumps(_raw_locomo_payload()))
    cli_main(["preprocess-locomo", "--input", str(source), "--output", str(output)])
    assert output.exists()


def test_cli_synthetic_writes_manifest(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    cli_main(["synthetic", "--reasoner", "deterministic", "--output", str(output)])
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "synthetic_revision"
    assert "metrics" in payload
    assert "created_at_utc" in payload


def test_cli_longmemeval_writes_manifest(tmp_path) -> None:
    source = tmp_path / "longmemeval.json"
    output = tmp_path / "longmemeval_manifest.json"
    source.write_text(json.dumps(_raw_longmemeval_payload()))
    cli_main(["longmemeval", "--input", str(source), "--input-format", "raw", "--output", str(output)])
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "longmemeval"


def test_cli_policy_filtering(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    cli_main([
        "synthetic",
        "--reasoner",
        "deterministic",
        "--policy",
        "offline_delta_v2",
        "--output",
        str(output),
    ])
    payload = json.loads(output.read_text())
    assert payload["policy_names"] == ["offline_delta_v2"]


def test_policy_registry_returns_expected_factory() -> None:
    policy = policy_factory_by_name("offline_delta_v2")()
    assert policy.name == "offline_delta_v2"


def test_build_manifest_adds_timestamp() -> None:
    manifest = build_manifest(
        benchmark="synthetic_revision",
        reasoner="DeterministicValidityReader",
        policy_names=["append_only"],
        metrics=[],
        config={},
    )
    assert manifest.created_at_utc
    assert manifest.environment is not None
    assert manifest.environment.python_version
