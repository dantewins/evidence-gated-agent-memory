import json

from memory_inference.evaluation.manifests import build_manifest, write_manifest


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

