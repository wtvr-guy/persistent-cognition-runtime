from __future__ import annotations

from tests.http_fakes import StreamingHTTPFake

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from persistent_cognition.capability_registry import CapabilityDescriptor, CapabilityKind
from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_worker import UserPromptLLM


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHTTPClient(StreamingHTTPFake):
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = iter(payloads)
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, *, json: dict) -> _FakeResponse:
        self.calls.append((path, json))
        return _FakeResponse(next(self._payloads))


class _SnapshotHTTPClient(StreamingHTTPFake):
    def __init__(self, payloads: dict[str, dict]) -> None:
        self._payloads = payloads

    def get(self, path: str) -> _FakeResponse:
        return _FakeResponse(self._payloads[path])


def _packet() -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="Which established constraint applies?"),
        supported=True,
        items=[
            MemoryEvidence(
                source_event_id=uuid4(),
                event_type=EventType.USER_PROMPT,
                source="test",
                created_at=datetime.now(timezone.utc),
                conversation_id=uuid4(),
                conversation_seq=1,
                global_seq=1,
                content="Project Kestrel uses PostgreSQL directly on Windows.",
            )
        ],
    )


def _catalog() -> tuple[CapabilityDescriptor, ...]:
    return (
        CapabilityDescriptor(
            capability_id="external.inspect",
            kind=CapabilityKind.WORKFLOW,
            description="Inspect an external source when required.",
        ),
    )


def _chat_payload() -> dict:
    return {
        "message": {"content": '{"capability_indices":[]}'},
        "done_reason": "stop",
        "eval_count": 12,
    }


def _generate_payload() -> dict:
    return {
        "response": '{"capability_indices":[]}',
        "done_reason": "stop",
        "eval_count": 12,
    }


def test_runtime_snapshot_captures_ollama_version_model_digest_and_loaded_record():
    model = "qwen3:4b-instruct-2507-q4_K_M"
    client = UserPromptLLM(base_url="https://ollama.test", model=model)
    client._client = _SnapshotHTTPClient(
        {
            "/api/version": {"version": "0.12.3"},
            "/api/tags": {
                "models": [
                    {
                        "name": model,
                        "model": model,
                        "digest": "d" * 64,
                        "details": {"quantization_level": "Q4_K_M"},
                    }
                ]
            },
            "/api/ps": {
                "models": [
                    {
                        "name": model,
                        "model": model,
                        "digest": "d" * 64,
                        "size_vram": 3_000_000_000,
                    }
                ]
            },
        }
    )

    snapshot = client.runtime_snapshot()

    assert snapshot["status"] == "COMPLETE"
    assert snapshot["ollama_version"] == "0.12.3"
    assert snapshot["model"]["digest"] == "d" * 64
    assert snapshot["running_model"]["size_vram"] == 3_000_000_000
    assert snapshot["errors"] == []


def test_qwen3_structured_call_appends_latest_no_think_soft_switch():
    client = UserPromptLLM(base_url="https://ollama.test", model="qwen3:4b")
    fake = _FakeHTTPClient([_chat_payload()])
    client._client = fake

    decision = client.decide_disposition(
        "Answer from established memory.",
        _packet(),
        _catalog(),
    )
    assert decision.response_required is True
    assert decision.capability_indices == []
    assert fake.calls[0][0] == "/api/chat"
    assert fake.calls[0][1]["think"] is False
    messages = fake.calls[0][1]["messages"]
    assert [message["role"] for message in messages] == ["system", "tool", "user"]
    assert messages[2]["content"].endswith("/no_think")
    assert "Project Kestrel uses" in messages[1]["content"]
    assert "Project Kestrel uses" not in messages[2]["content"]


def test_qwen3_instruct_call_uses_raw_structured_generate_transport():
    client = UserPromptLLM(
        base_url="https://ollama.test",
        model="qwen3:4b-instruct-2507-q4_K_M",
    )
    fake = _FakeHTTPClient([_generate_payload()])
    client._client = fake

    decision = client.decide_disposition(
        "Answer from established memory.",
        _packet(),
        _catalog(),
    )
    assert decision.response_required is True
    path, request = fake.calls[0]
    assert path == "/api/generate"
    assert request["raw"] is True
    assert "think" not in request
    assert "messages" not in request
    assert request["format"]
    assert request["prompt"].startswith("<|im_start|>system\n")
    assert "<|im_end|>\n<|im_start|>user\n" in request["prompt"]
    assert "<tool_response>" in request["prompt"]
    assert request["prompt"].index("<tool_response>") < request["prompt"].index(
        "Answer from established memory."
    )
    assert "Answer from established memory." in request["prompt"]
    assert request["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert "/no_think" not in request["prompt"]


def test_precognitive_worker_receives_evidence_without_memory_transport_metadata():
    packet = _packet()
    item = packet.items[0]
    client = UserPromptLLM(
        base_url="https://ollama.test",
        model="qwen3:4b-instruct-2507-q4_K_M",
    )
    fake = _FakeHTTPClient([_generate_payload()])
    client._client = fake
    client.decide_disposition("Answer from established memory.", packet, _catalog())

    prompt = fake.calls[0][1]["prompt"]
    assert item.content in prompt
    assert item.event_type.value in prompt
    assert str(packet.memory_request_id) not in prompt
    assert str(item.source_event_id) not in prompt
    assert str(item.conversation_id) not in prompt
    assert "conversation_seq:" not in prompt
    assert "global_seq:" not in prompt
    assert "created_at:" not in prompt
    assert "retrieval_reasons:" not in prompt


def test_qwen3_empty_or_thinking_only_output_retries_with_larger_budget():
    client = UserPromptLLM(base_url="https://ollama.test", model="qwen3:4b")
    fake = _FakeHTTPClient(
        [
            {
                "message": {"content": "", "thinking": "bounded hidden trace"},
                "done_reason": "length",
                "prompt_eval_count": 900,
                "eval_count": 48,
            },
            {
                "message": {
                    "content": '{"capability_indices":[]}',
                    "thinking": "",
                },
                "done_reason": "stop",
                "prompt_eval_count": 900,
                "eval_count": 12,
            },
        ]
    )
    client._client = fake

    decision = client.decide_disposition(
        "Answer from established memory.",
        _packet(),
        _catalog(),
    )
    assert decision.capability_indices == []
    assert [call[1]["options"]["num_predict"] for call in fake.calls] == [48, 96]
    assert all(call[1]["messages"][2]["content"].endswith("/no_think") for call in fake.calls)


def test_qwen3_empty_output_failure_reports_metadata_without_reasoning_text():
    secret_reasoning = "THIS_REASONING_MUST_NOT_ESCAPE"
    client = UserPromptLLM(base_url="https://ollama.test", model="qwen3:4b")
    client._client = _FakeHTTPClient(
        [
            {
                "message": {"content": "", "thinking": secret_reasoning},
                "done_reason": "length",
                "prompt_eval_count": 900,
                "eval_count": 48,
            },
            {
                "message": {
                    "content": "<think>still hidden</think>",
                    "thinking": secret_reasoning,
                },
                "done_reason": "length",
                "prompt_eval_count": 900,
                "eval_count": 96,
            },
        ]
    )

    with pytest.raises(ValueError, match="pre-cognitive work selection failed") as exc_info:
        client.decide_disposition(
            "Answer from established memory.",
            _packet(),
            _catalog(),
        )
    message = str(exc_info.value)
    assert secret_reasoning not in message
    assert '"thinking_length"' in message
    assert '"max_tokens":96' in message
    assert '"done_reason":"length"' in message


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
