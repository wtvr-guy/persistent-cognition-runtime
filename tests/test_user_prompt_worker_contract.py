from __future__ import annotations

import pytest

from persistent_cognition.percept_response_worker import (
    UserPromptLLM,
    UserPromptWorkSelection,
)
from persistent_cognition.percept_response_runtime import (
    _RESPONSE_POLICY_PROMPT,
)
from persistent_cognition.models import EventType
from persistent_cognition.response_policy import (
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
    assert "specific remembered personal facts" in normalized


@pytest.mark.parametrize(
    ("scope", "expected_types"),
    [
        (HistoricalEvidenceScope.USER_AUTHORED, {EventType.USER_PROMPT}),
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


@pytest.mark.parametrize("prompt", [
    "What did you tell me earlier about PostgreSQL?",
    "What did Persistent Cognition tell me earlier about PostgreSQL?",
    "Persistent Cognition recommended PostgreSQL. Why?",
    "PERSISTENT COGNITION just said to use PostgreSQL. Why?",
    "persistent_cognition mentioned PostgreSQL. Explain.",
])
def test_explicit_prior_assistant_reference_uses_mixed_scope_without_model_classification(
    monkeypatch,
    prompt,
) -> None:
    llm = UserPromptLLM()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("explicit prior-assistant references must use the deterministic path")

    monkeypatch.setattr(llm, "_structured_with_evidence", fail_if_called)

    policy = llm._response_policy(prompt)

    assert policy.evidence_scope is HistoricalEvidenceScope.MIXED_CONVERSATION
    assert set(source_types_for_scope(policy.evidence_scope)) == {
        EventType.USER_PROMPT,
        EventType.INTERACTION_RESPONSE,
        EventType.AGENT_RESPONSE,
        EventType.AGENT_RESULT,
    }
