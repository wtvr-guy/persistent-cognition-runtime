from __future__ import annotations

from datetime import datetime, timezone
import uuid

from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import ResponseMemoryPackage
from persistent_cognition.percept_response_worker import (
    UserPromptLLM,
    _cognitive_memory_packet,
)
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
)


def _packet(*items: MemoryEvidence) -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid.uuid4(),
        need=MemoryNeed(query_text="remembered phrase"),
        supported=bool(items),
        items=list(items),
    )


def _evidence(
    *,
    event_type: EventType,
    source: str,
    content: str,
    global_seq: int,
) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid.uuid4(),
        event_type=event_type,
        source=source,
        created_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        conversation_id=uuid.uuid4(),
        conversation_seq=global_seq,
        global_seq=global_seq,
        content=content,
    )


def test_empty_external_catalog_bypasses_precognitive_llm(monkeypatch) -> None:
    llm = UserPromptLLM()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("pre-cognitive LLM must not run for an empty catalog")

    monkeypatch.setattr(llm, "_structured", fail_if_called)
    decision = llm.decide_disposition(
        "Explain why your last answer contradicted my request.",
        _packet(),
        (),
    )

    assert decision.response_required is True
    assert decision.capability_indices == []


def test_recursive_response_trace_is_not_model_visible() -> None:
    trace = _evidence(
        event_type=EventType.SYSTEM_EVENT,
        source="percept_response_v2/response_input_trace",
        content="recursive trace poison " * 100,
        global_seq=1,
    )
    user = _evidence(
        event_type=EventType.USER_PROMPT,
        source="user",
        content='Remember the phrase "cat boy rock Virginia."',
        global_seq=2,
    )

    filtered = _cognitive_memory_packet(_packet(trace, user))

    assert [item.source for item in filtered.items] == ["user"]
    assert filtered.items[0].content == user.content
    assert filtered.supported is True
    assert filtered.retrieval_trace["cognitive_visibility_filter"]["excluded_item_count"] == 1


def test_final_responder_gets_user_evidence_authority_and_no_trace(monkeypatch) -> None:
    llm = UserPromptLLM()
    captured: dict[str, str] = {}

    def capture_text(kind, system, current_user, evidence, max_tokens=None):
        del kind, max_tokens
        captured["system"] = system
        captured["current_user"] = current_user
        captured["evidence"] = evidence
        return "The prior response was mistaken."

    monkeypatch.setattr(llm, "_text_with_evidence", capture_text)
    direct_user_evidence = _evidence(
        event_type=EventType.USER_PROMPT,
        source="user",
        content='Remember the phrase "cat boy rock Virginia."',
        global_seq=1,
    )
    bad_prior_response = _evidence(
        event_type=EventType.INTERACTION_RESPONSE,
        source="percept_response_v2",
        content="You never asked me to remember that phrase.",
        global_seq=2,
    )
    recursive_trace = _evidence(
        event_type=EventType.SYSTEM_EVENT,
        source="percept_response_v2/response_input_trace",
        content="nested response trace should never reach the responder",
        global_seq=3,
    )
    package = ResponseMemoryPackage(
        memory_packet=_packet(direct_user_evidence, bad_prior_response, recursive_trace),
        adaptive_recall_rounds=0,
    )

    answer = llm.generate_final_response(
        "Why did you say I never asked you to remember it?",
        package,
        (),
        response_policy=ResponsePolicy(
            evidence_scope=HistoricalEvidenceScope.MIXED_CONVERSATION,
            surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
        ),
    )

    assert answer == "The prior response was mistaken."
    assert "nested response trace" not in captured["evidence"]
    assert 'Remember the phrase "cat boy rock Virginia."' in captured["evidence"]
    assert "You never asked me to remember that phrase." in captured["evidence"]
    assert "cat boy rock Virginia" not in captured["current_user"]
    assert "QUARANTINED_EVIDENCE" in captured["evidence"]
    assert "authority_class: DIRECT_USER_TESTIMONY" in captured["evidence"]
    assert "authority_class: MODEL_OUTPUT_ONLY" in captured["evidence"]
    assert "A historical USER_PROMPT is direct evidence" in captured["system"]
    assert "they never\nnegate a user-authored event" in captured["system"]
