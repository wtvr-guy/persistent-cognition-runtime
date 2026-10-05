from prometheist.native_policy import (
    NATIVE_RESOURCE_POLICY_VERSION,
    native_resource_safety_policy,
)


def test_native_resource_policy_freezes_calibrated_gates():
    policy = native_resource_safety_policy()

    assert policy.policy_version == NATIVE_RESOURCE_POLICY_VERSION
    assert policy.cpu_system_headroom_percent == 10
    assert policy.memory_system_headroom_percent == 10
    assert policy.uncertainty_headroom_percent == 5
    assert policy.default_llm_process_memory_mib == 3_072
    assert policy.default_process_memory_mib == 512
    assert policy.llm_concurrency_limit == 1
