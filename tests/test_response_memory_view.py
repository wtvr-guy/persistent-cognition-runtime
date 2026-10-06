from __future__ import annotations

from tests.http_fakes import StreamingHTTPFake

import pytest

from datetime import datetime, timezone
from uuid import uuid4

from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import ResponseMemoryPackage
from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": self._content}}


class _FakeHTTPClient(StreamingHTTPFake):
    def __init__(self, contents: list[str]) -> None:
        self._contents = iter(contents)
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, *, json: dict) -> _FakeResponse:
        self.calls.append((path, json))
        return _FakeResponse(next(self._contents))


def _evidence(
    *,
    event_type: EventType,
    content: str,
    conversation_id,
    conversation_seq: int,
    global_seq: int,
) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=event_type,
        source="test",
        created_at=datetime.now(timezone.utc),
        conversation_id=conversation_id,
        conversation_seq=conversation_seq,
        global_seq=global_seq,
        content=content,
    )


def test_final_responder_receives_compact_oldest_to_newest_evidence_timeline():
    historical_conversation = uuid4()
    active_conversation = uuid4()
    historical = _evidence(
        event_type=EventType.USER_PROMPT,
        content="Historical rule under profile VX-ABC123.",
        conversation_id=historical_conversation,
        conversation_seq=1,
        global_seq=1,
    )
    turn1 = _evidence(
        event_type=EventType.USER_PROMPT,
        content="Call the active plan BlueHarbor-ABC123.",
        conversation_id=active_conversation,
        conversation_seq=1,
        global_seq=20,
    )
    turn2_response = _evidence(
        event_type=EventType.INTERACTION_RESPONSE,
        content="Docker Compose | VX-ABC123",
        conversation_id=active_conversation,
        conversation_seq=13,
        global_seq=32,
    )
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="What nickname and ruled-out approach?"),
        supported=True,
        items=[turn2_response, historical, turn1],
    )
    package = ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=0,
    )
    client = UserPromptLLM(base_url="https://ollama.test", model="model:test")
    fake = _FakeHTTPClient(['{"answer":"ok"}'])
    client._client = fake

    policy = ResponsePolicy(
        evidence_scope=HistoricalEvidenceScope.MIXED_CONVERSATION,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )
    assert client.generate_final_response(
        "Answer from the timeline.", package, (), response_policy=policy
    ) == "ok"
    path, payload = fake.calls[0]
    assert path == "/api/chat"
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "tool",
        "user",
    ]
    model_input = payload["messages"][1]["content"]
    assert "[Evidence timeline: oldest to newest]" in model_input
    assert model_input.index("Historical rule") < model_input.index("Call the active plan")
    assert model_input.index("Call the active plan") < model_input.index("Docker Compose")
    assert "conversation_scope: historical_context" in model_input
    assert "conversation_scope: recent_conversation" in model_input
    assert str(packet.memory_request_id) not in model_input
    assert str(historical.source_event_id) not in model_input
    assert str(turn1.source_event_id) not in model_input
    assert str(turn2_response.source_event_id) not in model_input
    assert "global_seq:" not in model_input
    assert "conversation_seq:" not in model_input
    assert "created_at:" not in model_input
    assert "retrieval_reasons:" not in model_input
    assert payload["messages"][2]["content"] == "Answer from the timeline."


def test_retrieval_completion_does_not_claim_semantic_sufficiency():
    unrelated = _evidence(
        event_type=EventType.USER_PROMPT,
        content="My work messages are direct and concise.",
        conversation_id=uuid4(),
        conversation_seq=1,
        global_seq=1,
    )
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="What was my teacher's name?"),
        supported=True,
        items=[unrelated],
    )
    package = ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=1,
    )
    client = UserPromptLLM(base_url="https://ollama.test", model="model:test")
    fake = _FakeHTTPClient(['{"answer":"I do not know the name."}'])
    client._client = fake

    policy = ResponsePolicy(
        evidence_scope=HistoricalEvidenceScope.GENERAL_OR_CURRENT,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )
    assert client.generate_final_response(
        "What was my teacher's name?", package, (), response_policy=policy
    ) == "I do not know the name."
    evidence = fake.calls[0][1]["messages"][1]["content"]
    assert "My work messages" in evidence
    assert "does not establish answerability" in evidence
    assert "memory_sufficient" not in evidence
    assert client._artifact_evidence_refs == (f"event:{unrelated.source_event_id}",)


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
