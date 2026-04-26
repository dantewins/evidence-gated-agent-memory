from memory_inference.orchestration.presets import (
    DEBUG_POLICY_NAMES,
    PAPER_POLICY_NAMES,
    TEST_POLICY_NAMES,
    paper_policy_factories,
    policy_factory_by_name,
)
from memory_inference.memory.policies.odv2_mem0_hybrid import ODV2Mem0HybridPolicy
from memory_inference.memory.policies.odv2_mem0_selective import ODV2Mem0SelectivePolicy
from memory_inference.memory.policies.odv2_recovery import ODV2RecoveryPolicy


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
    assert policy_factory_by_name("odv2_dense_compact")().name == "odv2_dense_compact"
    assert policy_factory_by_name("mem0_validity_guard")().name == "mem0_validity_guard"
    assert policy_factory_by_name("odv2_mem0_hybrid")().name == "odv2_mem0_hybrid"
    assert policy_factory_by_name("odv2_mem0_selective")().name == "odv2_mem0_selective"
    assert policy_factory_by_name("odv2_recovery")().name == "odv2_recovery"
    assert isinstance(policy_factory_by_name("odv2_mem0_hybrid")(), ODV2Mem0HybridPolicy)
    assert isinstance(policy_factory_by_name("odv2_mem0_selective")(), ODV2Mem0SelectivePolicy)
    assert isinstance(policy_factory_by_name("odv2_recovery")(), ODV2RecoveryPolicy)
