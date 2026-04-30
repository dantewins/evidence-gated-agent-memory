import json

from memory_inference.evaluation.manifests import build_manifest, write_manifest
from memory_inference.cli.main import manifest_config


class Args:
    reasoner = "local-hf"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    policy = ["mem0"]
    category = ["multi-session"]
    query_mode = []
    input = ""
    input_format = "raw"
    cache_dir = ".cache/test"
    cases_output = "results/test_cases.jsonl"
    max_new_tokens = 32
    temperature = 0.0
    top_p = 1.0
    do_sample = False
    repetition_penalty = 1.0
    device = "cuda"
    dtype = "bfloat16"
    prompt_template_id = "default"
    trust_remote_code = False
    no_chat_template = False


def test_manifest_build_can_skip_environment_capture() -> None:
    manifest = build_manifest(
        benchmark="longmemeval",
        reasoner="DeterministicValidityReader",
        policy_names=["append_only"],
        metrics=[],
        config={},
        include_environment=False,
    )

    assert manifest.created_at_utc
    assert manifest.environment is None
    assert manifest.git_commit is None
    assert manifest.git_dirty is False


def test_manifest_write_serializes_payload(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = build_manifest(
        benchmark="locomo",
        reasoner="DeterministicValidityReader",
        policy_names=["mem0"],
        metrics=[{"policy_name": "mem0", "accuracy": 1.0}],
        config={"reasoner": "deterministic"},
        include_environment=False,
    )

    write_manifest(path, manifest)
    payload = json.loads(path.read_text())

    assert payload["benchmark"] == "locomo"
    assert payload["policy_names"] == ["mem0"]
    assert payload["environment"] is None


def test_cli_manifest_config_includes_artifact_metadata(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"ok": true}')
    args = Args()
    args.input = str(input_path)
    monkeypatch.setenv("MEM0_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "nomic-embed-text:latest")

    config = manifest_config(args, ["longmemeval", "--input", str(input_path)])

    assert config["command_line"].startswith("memory-inference longmemeval")
    assert config["input_sha256"]
    assert config["official_mem0"]["llm_model"] == "llama3.1:8b"
    assert config["official_mem0"]["embedder_model"] == "nomic-embed-text:latest"
