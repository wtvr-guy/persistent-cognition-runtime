"""Environment-calibrated resource policy for the current native v0.7 target.

The generic ResourceSafetyPolicy remains deliberately conservative for unknown
hosts. Production native execution uses this explicit, versioned calibration so
measured target-host assumptions are never confused with universal defaults.
"""
from __future__ import annotations

from prometheist.attention_observation import ResourceSafetyPolicy


NATIVE_RESOURCE_POLICY_VERSION = "v0.7-native-calibration-v1"


def native_resource_safety_policy() -> ResourceSafetyPolicy:
    """Return the frozen native policy currently under empirical calibration."""

    policy = ResourceSafetyPolicy(
        policy_version=NATIVE_RESOURCE_POLICY_VERSION,
        cpu_system_headroom_percent=10,
        memory_system_headroom_percent=10,
        memory_system_headroom_min_mib=1024,
        uncertainty_headroom_percent=5,
        default_llm_process_memory_mib=3072,
        default_process_memory_mib=512,
        default_process_cpu_units=1,
        llm_concurrency_limit=1,
    )

    from prometheist.runtime_settings import job_settings, local_ollama
    settings = job_settings()
    if settings is not None:
        values = {key: value for key, value in settings.resources.model_dump().items()
                  if key in ResourceSafetyPolicy.model_fields}
        import os
        cpu = int(os.environ.get("PROMETHEIST_GUI_MODEL_CPU_UNITS", "1"))
        if cpu < 1:
            raise ValueError("model CPU requirement must be positive")
        values["default_process_cpu_units"] = cpu
        from prometheist.contract_registry import STAGE_CONTRACTS
        selections = [settings.selection_for(stage) for stage, contract in STAGE_CONTRACTS.items()
                      if contract.kinds and stage.startswith("V2_")]
        if selections and all(selection.provider == "openai" or not local_ollama(settings) for selection in selections):
            values["default_llm_process_memory_mib"] = values["default_process_memory_mib"]
        else:
            floor = int(os.environ.get("PROMETHEIST_GUI_MODEL_MEMORY_FLOOR_MIB", "0"))
            if floor < 0:
                raise ValueError("model memory floor must be nonnegative")
            values["default_llm_process_memory_mib"] = max(values["default_llm_process_memory_mib"], floor)
        policy = ResourceSafetyPolicy.model_validate({**policy.model_dump(), **values,
                                                     "policy_version": "app-resource-policy/v1"})
    return policy
