from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from persistent_cognition.capability_registry import DEFAULT_REGISTRY
from persistent_cognition.interaction_contracts import DurableInteraction
from persistent_cognition.models import MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import (
    PerceptStage,
    ResponseMemoryPackage,
    _execute_stage,
    _validated_response_policy,
)
from persistent_cognition.percept_response_worker import (
    USER_PROMPT_STAGE_SPECIALIST_ROLES,
    UserPromptLLM,
    _ALLOWED_LLM_KINDS_BY_STAGE,
)
from persistent_cognition.response_policy import (
    RESPONSE_POLICY_VERSION,
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
    source_types_for_scope,
)


def _policy() -> ResponsePolicy:
    return ResponsePolicy(
        evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )


def _envelope(stage: PerceptStage):
    return SimpleNamespace(step=SimpleNamespace(step_key=stage.value))


def test_every_stage_has_one_named_specialist_role() -> None:
    assert set(USER_PROMPT_STAGE_SPECIALIST_ROLES) == set(PerceptStage)
    assert len(set(USER_PROMPT_STAGE_SPECIALIST_ROLES.values())) == len(PerceptStage)
    assert set(_ALLOWED_LLM_KINDS_BY_STAGE) == set(PerceptStage)
    assert _ALLOWED_LLM_KINDS_BY_STAGE[PerceptStage.RESOLVE_REFERENCES] == set()
    assert _ALLOWED_LLM_KINDS_BY_STAGE[PerceptStage.EXECUTE_WORK] == set()
    assert _ALLOWED_LLM_KINDS_BY_STAGE[PerceptStage.PERSIST_RESULT] == set()


@pytest.mark.parametrize("protocol_version", [
    "v0.7-interaction-v9", "v0.8-interaction-v10", "v0.8-interaction-v11",
])
def test_pre_split_interaction_protocol_cannot_resume_under_new_stage_graph(
    protocol_version: str,
) -> None:
    with pytest.raises(ValueError, match="interaction protocol version is unsupported"):
        DurableInteraction(
            protocol_version=protocol_version,
            interaction_id=uuid4(),
            conversation_id=uuid4(),
            correlation_id=uuid4(),
            user_prompt_event_id=uuid4(),
            before_global_seq=1,
            task_id=uuid4(),
            assignment_id=uuid4(),
            user_text="unfinished old interaction",
        )


def test_guarded_worker_rejects_another_specialists_llm_role() -> None:
    worker = UserPromptLLM(stage=PerceptStage.EVIDENCE_POLICY)

    with pytest.raises(RuntimeError, match="cannot invoke LLM role FINAL_RESPONSE_V2"):
        worker._require_stage_specialization("FINAL_RESPONSE_V2")


def test_evidence_policy_is_a_separate_durable_stage() -> None:
    policy = _policy()
    llm = SimpleNamespace(_response_policy=lambda _percept: policy)

    output, refs = _execute_stage(
        None,
        llm,
        _envelope(PerceptStage.EVIDENCE_POLICY),
        SimpleNamespace(user_text="What constraint did I give you?"),
        scheduler_key="test",
        registry=DEFAULT_REGISTRY,
    )

    assert output == {
        "response_policy_version": RESPONSE_POLICY_VERSION,
        "response_policy": policy.model_dump(mode="json"),
        "response_source_types": [
            event_type.value
            for event_type in source_types_for_scope(policy.evidence_scope)
        ],
    }
    assert refs == []


def test_response_stage_inherits_exact_policy_without_reclassification(monkeypatch) -> None:
    policy = _policy()
    package = ResponseMemoryPackage(
        memory_packet=MemoryPacket(
            memory_request_id=uuid4(),
            need=MemoryNeed(query_text="remembered constraint"),
            supported=False,
            items=[],
        ),
        adaptive_recall_rounds=0,
    )
    results = {
        PerceptStage.EVIDENCE_POLICY: {
            "response_policy_version": RESPONSE_POLICY_VERSION,
            "response_policy": policy.model_dump(mode="json"),
            "response_source_types": [
                event_type.value
                for event_type in source_types_for_scope(policy.evidence_scope)
            ],
        },
        PerceptStage.PRECOGNITIVE: {
            "disposition": {"response_required": True, "capability_indices": []},
        },
        PerceptStage.RETRIEVE_MEMORY: {
            "memory_package": package.model_dump(mode="json"),
        },
        PerceptStage.EXECUTE_WORK: {"work_results": []},
    }
    monkeypatch.setattr(
        "persistent_cognition.percept_response_runtime._stage_result",
        lambda _conn, _interaction, stage, _scheduler_key: results[stage],
    )
    received: list[ResponsePolicy] = []

    def generate(_percept, _package, _work_results, *, response_policy):
        received.append(response_policy)
        return "inherited policy"

    llm = SimpleNamespace(generate_final_response=generate)
    output, refs = _execute_stage(
        None,
        llm,
        _envelope(PerceptStage.RESPOND),
        SimpleNamespace(user_text="What constraint did I give you?"),
        scheduler_key="test",
        registry=DEFAULT_REGISTRY,
    )

    assert received == [policy]
    assert output == {
        "response_required": True,
        "response_text": "inherited policy",
        "skipped": False,
    }
    assert refs == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"response_policy_version": "obsolete"},
        {"response_source_types": ["INTERACTION_RESPONSE"]},
    ],
)
def test_committed_policy_fails_closed_on_version_or_allowlist_drift(mutation) -> None:
    policy = _policy()
    stage_output = {
        "response_policy_version": RESPONSE_POLICY_VERSION,
        "response_policy": policy.model_dump(mode="json"),
        "response_source_types": [
            event_type.value
            for event_type in source_types_for_scope(policy.evidence_scope)
        ],
    }
    stage_output.update(mutation)

    with pytest.raises(RuntimeError, match="persisted response policy"):
        _validated_response_policy(stage_output)
