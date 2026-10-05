from __future__ import annotations

import os

import pytest

from prometheist.percept_response_worker import (
    _INTERACTIVE_PERSONALITY_PROMPT,
    UserPromptLLM,
    UserPromptWorkSelection,
)
from prometheist.percept_response_runtime import (
    _RESPONSE_POLICY_PROMPT,
)
from prometheist.models import EventType
from prometheist.response_policy import (
    HistoricalEvidenceScope,
    ResponseSurfaceMode,
    source_types_for_scope,
)


def test_user_prompt_work_schema_cannot_decide_response_requirement() -> None:
    schema = UserPromptWorkSelection.model_json_schema()
    assert "response_required" not in schema.get("properties", {})
    assert set(schema.get("properties", {})) == {"capability_indices"}










def test_response_policy_defaults_ordinary_questions_to_natural_language() -> None:
    normalized = " ".join(_RESPONSE_POLICY_PROMPT.split()).casefold()
    assert "natural_language is the default for ordinary questions" in normalized
    assert "only when the current user explicitly requires exact raw output" in normalized
    assert "a request to answer naturally" in normalized
    assert "self_model" in normalized
    assert "what do i usually prefer?" in normalized
    assert "fifth-grade teacher" in normalized
    assert "specific remembered personal facts" in normalized


@pytest.mark.parametrize(
    ("scope", "expected_types"),
    [
        (HistoricalEvidenceScope.USER_AUTHORED, {EventType.USER_PROMPT}),
        (
            HistoricalEvidenceScope.SELF_MODEL,
            {
                EventType.USER_PROMPT,
                EventType.TOOL_RESULT,
                EventType.PERCEPT_OBSERVATION,
                EventType.SYSTEM_EVENT,
            },
        ),
        (
            HistoricalEvidenceScope.MODEL_OUTPUT,
            {
                EventType.INTERACTION_RESPONSE,
                EventType.AGENT_RESPONSE,
                EventType.AGENT_RESULT,
            },
        ),
        (
            HistoricalEvidenceScope.MIXED_CONVERSATION,
            {
                EventType.USER_PROMPT,
                EventType.INTERACTION_RESPONSE,
                EventType.AGENT_RESPONSE,
                EventType.AGENT_RESULT,
            },
        ),
        (
            HistoricalEvidenceScope.EXTERNAL_TOOL,
            {EventType.TOOL_RESULT, EventType.PERCEPT_OBSERVATION},
        ),
        (
            HistoricalEvidenceScope.SYSTEM_RECORD,
            {
                EventType.SYSTEM_EVENT,
                EventType.ERROR,
                EventType.INTERACTION_WORKING_STATE,
            },
        ),
        (
            HistoricalEvidenceScope.DERIVED_INTERNAL,
            {
                EventType.DERIVED_REPRESENTATION,
                EventType.AGENT_DECISION,
                EventType.RETRIEVAL_REQUEST,
                EventType.RETRIEVAL_RESULT,
                EventType.MEMORY_REQUEST,
                EventType.MEMORY_PACKET,
                EventType.AGENT_DELEGATION,
                EventType.TOOL_REQUEST,
                EventType.CAPABILITY_REQUEST,
                EventType.CAPABILITY_PACKET,
                EventType.CAPABILITY_RESULT,
            },
        ),
        (
            HistoricalEvidenceScope.GENERAL_OR_CURRENT,
            {
                EventType.USER_PROMPT,
                EventType.TOOL_RESULT,
                EventType.SYSTEM_EVENT,
            },
        ),
    ],
)
def test_response_scope_maps_to_exact_retrieval_event_roles(scope, expected_types) -> None:
    assert set(source_types_for_scope(scope)) == expected_types


@pytest.mark.parametrize(
    ("prompt", "scope"),
    [
        ("What constraint did I give you?", HistoricalEvidenceScope.USER_AUTHORED),
        ("What do I usually prefer?", HistoricalEvidenceScope.SELF_MODEL),
        ("Quote the assistant's prior response verbatim.", HistoricalEvidenceScope.MODEL_OUTPUT),
        ("Summarize our prior conversation.", HistoricalEvidenceScope.MIXED_CONVERSATION),
        ("What is the current status?", HistoricalEvidenceScope.GENERAL_OR_CURRENT),
    ],
)
def test_response_policy_worker_selects_scope_without_historical_evidence(
    monkeypatch,
    prompt,
    scope,
) -> None:
    llm = UserPromptLLM()
    calls: list[tuple[str, str]] = []

    def fake_structured(kind, system, current_user, evidence, schema, max_tokens):
        del schema, max_tokens
        calls.append((current_user, evidence))
        assert kind == "V2_RESPONSE_POLICY"
        assert prompt in current_user
        assert evidence.endswith("none")
        return (
            '{"evidence_scope":"'
            f"{scope.value}"
            '","surface_mode":"NATURAL_LANGUAGE","insufficient_literal":null}'
        )

    monkeypatch.setattr(llm, "_structured_with_evidence", fake_structured)

    policy = llm._response_policy(prompt)

    assert policy.evidence_scope is scope
    assert policy.surface_mode is ResponseSurfaceMode.NATURAL_LANGUAGE
    assert len(calls) == 1


def test_explicit_prior_assistant_reference_uses_mixed_scope_without_model_classification(
    monkeypatch,
) -> None:
    llm = UserPromptLLM()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("explicit prior-assistant references must use the deterministic path")

    monkeypatch.setattr(llm, "_structured_with_evidence", fail_if_called)

    policy = llm._response_policy("What did you tell me earlier about PostgreSQL?")

    assert policy.evidence_scope is HistoricalEvidenceScope.MIXED_CONVERSATION
    assert set(source_types_for_scope(policy.evidence_scope)) == {
        EventType.USER_PROMPT,
        EventType.INTERACTION_RESPONSE,
        EventType.AGENT_RESPONSE,
        EventType.AGENT_RESULT,
    }


def test_interactive_identity_belongs_to_prometheist_not_disposable_llm() -> None:
    normalized = " ".join(_INTERACTIVE_PERSONALITY_PROMPT.split()).casefold()
    assert "you are prometheist" in normalized
    assert "language model is a fresh, disposable semantic worker" in normalized
    assert "not prometheist's identity" in normalized


def test_user_prompt_llm_always_installs_core_interactive_personality(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PROMETHEIST_PERSONALITY_PROMPT", raising=False)

    UserPromptLLM()

    resolved = os.environ["PROMETHEIST_PERSONALITY_PROMPT"]
    assert resolved == _INTERACTIVE_PERSONALITY_PROMPT.strip()
    assert "A historical USER_PROMPT is direct evidence" in resolved


def test_user_configured_personality_extends_core_accuracy_contract(monkeypatch) -> None:
    configured = "Use concise dry humor when appropriate, without sacrificing precision."
    monkeypatch.setenv("PROMETHEIST_PERSONALITY_PROMPT", configured)

    UserPromptLLM()

    resolved = os.environ["PROMETHEIST_PERSONALITY_PROMPT"]
    assert resolved.startswith(_INTERACTIVE_PERSONALITY_PROMPT.strip())
    assert "[User-configured personality]" in resolved
    assert configured in resolved
    assert "A historical USER_PROMPT is direct evidence" in resolved
