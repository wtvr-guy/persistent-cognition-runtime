from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from uuid import uuid4

import pytest

from persistent_cognition import llm
from persistent_cognition.llm import _strip_thinking
from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import ResponseMemoryPackage, _FINAL_RESPONSE_PROMPT
from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
)


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self._content}}


class _FakeHTTPClient:
    def __init__(self, contents):
        self._contents = iter(contents)
        self.calls = []

    def post(self, path, *, json):
        self.calls.append((path, json))
        return _FakeResponse(next(self._contents))


def _evidence(event_type: EventType, content: str, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=event_type,
        source="test",
        created_at=datetime.now(timezone.utc),
        conversation_id=uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=content,
    )


def _packet(*contents: str) -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="Question"),
        supported=bool(contents),
        items=[
            _evidence(EventType.USER_PROMPT, content, index + 1)
            for index, content in enumerate(contents)
        ],
    )


def _package(packet: MemoryPacket | None = None) -> ResponseMemoryPackage:
    return ResponseMemoryPackage(
        memory_packet=packet or _packet(),
        adaptive_recall_rounds=0,
    )


def _natural_policy(
    scope: HistoricalEvidenceScope = HistoricalEvidenceScope.GENERAL_OR_CURRENT,
) -> ResponsePolicy:
    return ResponsePolicy(
        evidence_scope=scope,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )


def test_strip_thinking_preserves_plain_answer():
    assert _strip_thinking("final answer") == "final answer"


def test_transport_envelope_redacts_hidden_thinking_but_preserves_diagnostics():
    secret = "private reasoning that must not be persisted"

    redacted = llm._redact_hidden_thinking(
        {
            "message": {"content": '{"answer":"ok"}', "thinking": secret},
            "eval_count": 17,
        }
    )

    assert redacted["message"]["content"] == '{"answer":"ok"}'
    assert redacted["eval_count"] == 17
    assert redacted["message"]["thinking"] == {
        "redacted": True,
        "utf8_bytes": len(secret.encode("utf-8")),
        "sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
    }
    assert secret not in str(redacted)


def test_strip_thinking_removes_complete_think_block():
    assert _strip_thinking("<think>hidden reasoning</think>final answer") == "final answer"


def test_strip_thinking_uses_content_after_unmatched_closing_tag():
    assert _strip_thinking("hidden reasoning</think>final answer") == "final answer"


@pytest.mark.parametrize(
    "content",
    ["hidden reasoning</think>", "<think>hidden reasoning</think>", "   "],
)
def test_strip_thinking_rejects_outputs_without_answer_content(content):
    with pytest.raises(ValueError, match="no answer content"):
        _strip_thinking(content)


def test_response_prompt_does_not_leak_acceptance_scenario_facts():
    prompt = _FINAL_RESPONSE_PROMPT.casefold()
    assert "project kestrel" not in prompt
    assert "virtualization is disabled" not in prompt
    assert "blueharbor" not in prompt
    assert "vx-" not in prompt


def test_user_facing_answers_use_expressive_temperature_with_structured_envelope(
    monkeypatch,
):
    monkeypatch.delenv("PCR_RESPONSE_TEMPERATURE", raising=False)
    client = UserPromptLLM(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(['{"answer":"Final answer only."}'])
    client._client = fake_http

    assert client.generate_final_response(
        "Question", _package(), (), response_policy=_natural_policy()
    ) == "Final answer only."
    path, payload = fake_http.calls[0]
    assert path == "/api/chat"
    assert payload["format"]["required"] == ["answer"]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"] == {"num_predict": 256, "temperature": 0.65}


def test_explicit_ollama_keep_alive_is_sent_without_changing_default(
    monkeypatch,
):
    monkeypatch.setenv("PCR_OLLAMA_KEEP_ALIVE", "30m")
    client = llm.OllamaClient(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(['{"ok":true}'])
    client._client = fake_http

    client._structured(
        "V2_MEMORY_SUFFICIENCY_USER_PROMPT",
        "system",
        "user",
        {"type": "object"},
        32,
    )

    assert fake_http.calls[0][1]["keep_alive"] == "30m"


def test_control_llm_kinds_remain_deterministic_when_response_temperature_is_high(
    monkeypatch,
):
    monkeypatch.setenv("PCR_RESPONSE_TEMPERATURE", "1.25")
    client = llm.OllamaClient(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(['{"ok":true}'])
    client._client = fake_http
    client._structured(
        "V2_MEMORY_SUFFICIENCY_USER_PROMPT",
        "system",
        "user",
        {"type": "object"},
        32,
    )
    assert fake_http.calls[0][1]["options"] == {
        "num_predict": 32,
        "temperature": 0.0,
    }


def test_user_facing_answer_retries_invalid_structured_output():
    client = llm.OllamaClient(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(["not-json", '{"answer":"Recovered answer."}'])
    client._client = fake_http
    assert client._text("FINAL_RESPONSE_V2", "system", "Question") == "Recovered answer."
    assert len(fake_http.calls) == 2


def test_user_facing_answer_fails_closed_after_two_invalid_outputs():
    client = llm.OllamaClient(base_url="http://ollama.test", model="model:test")
    client._client = _FakeHTTPClient(["not-json", '{"answer":"   "}'])
    with pytest.raises(ValueError, match="model answer failed to validate"):
        client._text("FINAL_RESPONSE_V2", "system", "Question")


def test_history_dependent_response_fails_closed_without_admissible_evidence():
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="What was my fifth-grade teacher's name?"),
        supported=False,
        items=[],
    )
    package = ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=1,
    )
    client = UserPromptLLM(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(['{"verbatim_value":null}'])
    client._client = fake_http

    answer = client.generate_final_response(
        "What was my fifth-grade teacher's name?",
        package,
        (),
        response_policy=_natural_policy(HistoricalEvidenceScope.USER_AUTHORED),
    )

    assert answer == "Persisted evidence is insufficient."
    assert len(fake_http.calls) == 1
    assert "current-fallback selector" in fake_http.calls[0][1]["messages"][0]["content"]


def test_verbatim_placeholders_prevent_model_from_respelling_opaque_literals():
    exact_code = "A66673AD"
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="What codename did I give Project Oriole?"),
        supported=True,
        items=[
            _evidence(
                EventType.USER_PROMPT,
                f"The codename for Project Oriole is {exact_code}.",
                1,
            )
        ],
    )
    client = UserPromptLLM(base_url="http://ollama.test", model="model:test")
    fake_http = _FakeHTTPClient(
        [
            '{"answer":"The codename is [[VERBATIM_0]]."}',
        ]
    )
    client._client = fake_http

    answer = client.generate_final_response(
        "What codename did I give Project Oriole?",
        _package(packet),
        (),
        response_policy=_natural_policy(HistoricalEvidenceScope.USER_AUTHORED),
    )
    assert answer == f"The codename is {exact_code}."
    response_messages = fake_http.calls[0][1]["messages"]
    assert [message["role"] for message in response_messages] == [
        "system",
        "tool",
        "user",
    ]
    assert exact_code not in response_messages[1]["content"]
    assert "[[VERBATIM_0]]" in response_messages[1]["content"]


def test_verbatim_placeholders_cover_hyphenated_labels_and_current_input():
    exact_label = "BlueHarbor-4E0FF9"
    literal_to_placeholder, placeholder_to_literal = llm._build_verbatim_placeholder_maps(
        f"Call this plan {exact_label}."
    )
    assert literal_to_placeholder == {exact_label: "[[VERBATIM_0]]"}
    masked = llm._mask_verbatim_literals(
        f"Call this plan {exact_label}.",
        literal_to_placeholder,
    )
    assert masked == "Call this plan [[VERBATIM_0]]."
    assert llm._restore_verbatim_literals(masked, placeholder_to_literal) == (
        f"Call this plan {exact_label}."
    )


def test_verbatim_restore_accepts_unambiguous_bare_application_token():
    assert llm._restore_verbatim_literals(
        "VERBATIM_0",
        {"[[VERBATIM_0]]": "ASTER-1234ABCD"},
    ) == "ASTER-1234ABCD"


def test_verbatim_restore_does_not_confuse_double_digit_placeholder_indices():
    mapping = {
        "[[VERBATIM_1]]": "FIRST-1234",
        "[[VERBATIM_10]]": "TENTH-5678",
    }
    assert llm._restore_verbatim_literals("VERBATIM_10", mapping) == "TENTH-5678"


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
