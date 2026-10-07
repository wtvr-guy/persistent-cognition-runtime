"""Exact-value replies copy canonical source ranges and never invoke prose generation."""
from datetime import datetime, timezone
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import ResponseMemoryPackage
from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope, ResponseAnswerKind, ResponsePolicy, ResponseSurfaceMode,
)
from persistent_cognition.source_value_response import (
    IndexedValueSource, SourceValuePlan, SourceValueSelection, format_value_sources,
    render_value_response, resolve_value_plan,
    structured_value_sources,
)


def source(text):
    return IndexedValueSource.from_text(text, "DIRECT_USER_TESTIMONY")


def selection(index, first, last):
    return SourceValueSelection(source_index=index, first_token=first, last_token=last)


@pytest.mark.parametrize(("text", "first", "last", "expected"), [
    ("The identifier is BlueHarbor-68C19B.", 3, 3, "BlueHarbor-68C19B"),
    ("My name is Renée  O’Connor.", 3, 4, "Renée  O’Connor"),
    ("The code is 00001234.", 3, 3, "00001234"),
    ("The rate is 12.50.", 3, 3, "12.50"),
])
def test_resolves_original_source_characters_without_value_regeneration(text, first, last, expected):
    sources = (source(text),)
    plan = SourceValuePlan(selections=[selection(0, first, last)])
    assert resolve_value_plan(sources, plan) == (expected,)
    assert render_value_response((expected,)) == f"The requested value is {expected}."


@pytest.mark.parametrize(("index", "first", "last"), [(1, 0, 0), (0, 2, 1), (0, 0, 99)])
def test_invalid_source_or_token_indices_fail_closed(index, first, last):
    with pytest.raises(ValueError):
        resolve_value_plan((source("Original value."),), SourceValuePlan(
            selections=[selection(index, first, last)],
        ))


def test_models_cannot_submit_replacement_values_or_response_prose():
    with pytest.raises(ValidationError):
        SourceValuePlan.model_validate_json(json.dumps({"selections": [{
            "source_index": 0, "first_token": 3, "last_token": 3, "value": "Kestrel-01",
        }]}))
    with pytest.raises(ValidationError):
        SourceValuePlan.model_validate_json('{"selections":[],"answer":"Kestrel-01"}')


def evidence(text, kind, seq):
    return MemoryEvidence(
        source_event_id=uuid4(), event_type=kind, source="test",
        created_at=datetime.now(timezone.utc), conversation_id=uuid4(),
        conversation_seq=seq, global_seq=seq, content=text,
    )


def test_natural_factual_reply_filters_roles_and_bypasses_free_form_model(monkeypatch):
    client = UserPromptLLM()
    original = evidence("The identifier is BlueHarbor-68C19B.", EventType.USER_PROMPT, 1)
    poison = evidence("The identifier is Kestrel-01.", EventType.INTERACTION_RESPONSE, 2)
    packet = MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=True,
                          items=[poison, original])
    before = packet.model_dump_json()

    def select(kind, system, prompt, historical, schema, max_tokens):
        assert kind == "V2_SOURCE_VALUE_SELECTION"
        assert "BlueHarbor-68C19B" in historical
        assert "Kestrel-01" not in historical
        assert "3: \"BlueHarbor-68C19B\"" in historical
        return '{"selections":[{"source_index":0,"first_token":3,"last_token":3}]}'

    def forbid_prose(*args, **kwargs):
        raise AssertionError("exact values must bypass free-form language generation")

    monkeypatch.setattr(client, "_structured_with_evidence", select)
    monkeypatch.setattr(client, "_text_with_evidence", forbid_prose)
    result = client.generate_final_response(
        "What identifier did I give you? Answer naturally.",
        ResponseMemoryPackage(memory_packet=packet), (),
        response_policy=ResponsePolicy(
            evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
            surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            answer_kind=ResponseAnswerKind.EXTRACTIVE_VALUES,
        ),
    )
    assert result == "The requested value is BlueHarbor-68C19B."
    assert client._artifact_evidence_refs == (f"event:{original.source_event_id}",)
    assert packet.model_dump_json() == before


def test_missing_value_abstains_without_switching_to_free_form_generation(monkeypatch):
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_structured_with_evidence", lambda *args: '{"selections":[]}')
    monkeypatch.setattr(client, "_select_current_fallback_literal", lambda prompt: "UNKNOWN")
    monkeypatch.setattr(client, "_text_with_evidence", lambda *args: pytest.fail("no prose fallback"))
    packet = MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=True,
                          items=[evidence("My work messages are concise.", EventType.USER_PROMPT, 1)])
    assert client.generate_final_response(
        "What was my teacher's name? Otherwise answer UNKNOWN.",
        ResponseMemoryPackage(memory_packet=packet), (),
        response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
            surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            answer_kind=ResponseAnswerKind.EXTRACTIVE_VALUES),
    ) == "UNKNOWN"


def test_multiple_values_preserve_selection_order_and_original_spelling():
    sources = (source("Nickname BlueHarbor-68C19B."), source("Ruled out Docker Compose."))
    values = resolve_value_plan(sources, SourceValuePlan(
        selections=[selection(0, 1, 1), selection(1, 2, 3)],
    ))
    assert render_value_response(values) == "The requested values are BlueHarbor-68C19B; Docker Compose."
    assert "MODEL_OUTPUT_ONLY" in format_value_sources((IndexedValueSource.from_text(
        "Ruled out Docker Compose.", "MODEL_OUTPUT_ONLY",
    ),))


def test_omitted_policy_answer_kind_cannot_silently_bypass_extraction(monkeypatch):
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_structured_with_evidence", lambda *args:
        '{"evidence_scope":"USER_AUTHORED","surface_mode":"NATURAL_LANGUAGE"}')
    with pytest.raises(ValueError, match="omitted answer_kind"):
        client._response_policy("What identifier did I give you?")


def test_prior_assistant_scope_guard_keeps_factual_answer_kind(monkeypatch):
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_structured_with_evidence", lambda *args:
        '{"evidence_scope":"USER_AUTHORED","surface_mode":"NATURAL_LANGUAGE",'
        '"answer_kind":"EXTRACTIVE_VALUES"}')
    policy = client._response_policy("What nickname are we using, and what did you rule out?")
    assert policy.evidence_scope is HistoricalEvidenceScope.MIXED_CONVERSATION
    assert policy.answer_kind is ResponseAnswerKind.EXTRACTIVE_VALUES


def test_invalid_selection_never_falls_back_to_free_form_answer(monkeypatch):
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_structured_with_evidence", lambda *args:
        '{"selections":[{"source_index":99,"first_token":0,"last_token":0}]}')
    monkeypatch.setattr(client, "_text_with_evidence", lambda *args: pytest.fail("prose bypass"))
    packet = MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=True,
                          items=[evidence("Identifier BlueHarbor-68C19B.", EventType.USER_PROMPT, 1)])
    with pytest.raises(ValueError, match="unknown source"):
        client.generate_final_response(
            "What identifier did I give you?", ResponseMemoryPackage(memory_packet=packet), (),
            response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
                surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
                answer_kind=ResponseAnswerKind.EXTRACTIVE_VALUES),
        )


def test_structured_executor_values_preserve_decoded_unicode_and_path_characters(monkeypatch):
    result = {"capability_id": "inspection", "result_data": {
        "identifier": "BlueHarbor-68C19B", "path": "C:\\Users\\Renée",
    }}
    sources = structured_value_sources(result)
    path_index = next(i for i, item in enumerate(sources) if item.context == "result_data.path")
    original = result["result_data"]["path"]
    plan = SourceValuePlan(selections=[selection(path_index, 0, len(sources[path_index].tokens) - 1)])
    assert resolve_value_plan(sources, plan) == (original,)
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_structured_with_evidence", lambda *args: plan.model_dump_json())
    monkeypatch.setattr(client, "_text_with_evidence", lambda *args: pytest.fail("prose bypass"))
    packet = MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=False, items=[])
    assert client.generate_final_response(
        "What path did the inspection return?", ResponseMemoryPackage(memory_packet=packet),
        (result,), response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.EXTERNAL_TOOL,
            surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            answer_kind=ResponseAnswerKind.EXTRACTIVE_VALUES),
    ) == f"The requested value is {original}."
