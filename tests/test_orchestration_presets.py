from memory_inference.orchestration.presets import (
    DEBUG_POLICY_NAMES,
    PAPER_POLICY_NAMES,
    TEST_POLICY_NAMES,
    paper_policy_factories,
    policy_factory_by_name,
)


def test_policy_preset_groups_are_purpose_specific() -> None:
    assert PAPER_POLICY_NAMES != DEBUG_POLICY_NAMES
    assert PAPER_POLICY_NAMES != TEST_POLICY_NAMES
    assert "append_only" not in PAPER_POLICY_NAMES
    assert "strong_retrieval" in PAPER_POLICY_NAMES
    assert "append_only" in DEBUG_POLICY_NAMES
    assert "offline_delta_v2" in TEST_POLICY_NAMES


def test_paper_policy_factories_resolve_named_policies() -> None:
    factories = paper_policy_factories()
    policy_names = [factory().name for factory in factories]

    assert policy_names == list(PAPER_POLICY_NAMES)
    assert policy_factory_by_name("odv2_dense")().name == "odv2_dense"

