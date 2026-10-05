from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist.response_policy import (
    ExactSourceComposition,
    ExactSourceSelection,
    HistoricalEvidenceScope,
    allowed_event_types,
    explicit_prior_assistant_reference,
    filter_memory_packet_for_scope,
    source_types_for_scope,
    validate_current_literal,
    validate_exact_source_composition,
    validate_exact_source_selection,
)


def _evidence(event_type: EventType, content: str, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=event_type,
        source="response-policy-test",
        created_at=datetime.now(timezone.utc),
        conversation_id=uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=content,
    )


def _packet(*items: MemoryEvidence) -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="historical question"),
        supported=bool(items),
        items=list(items),
    )


def test_user_authored_scope_physically_removes_model_output():
    user = _evidence(EventType.USER_PROMPT, "My favorite color is green.", 1)
    assistant = _evidence(
        EventType.INTERACTION_RESPONSE,
        "The user's favorite color is poison-blue.",
        2,
    )

    filtered = filter_memory_packet_for_scope(
        _packet(user, assistant),
        HistoricalEvidenceScope.USER_AUTHORED,
    )

    assert filtered is not None
    assert [item.source_event_id for item in filtered.items] == [user.source_event_id]


def test_exact_source_selection_returns_source_bytes_not_generated_wrapper():
    source_texts = ("The launch key is ASTER-1234ABCD.",)
    selection = ExactSourceSelection(source_index=0, verbatim_value="ASTER-1234ABCD")
    assert validate_exact_source_selection(source_texts, selection) == "ASTER-1234ABCD"

    with pytest.raises(ValueError, match="verbatim substring"):
        validate_exact_source_selection(
            source_texts,
            ExactSourceSelection(source_index=0, verbatim_value="Key: ASTER-1234ABCD"),
        )


def test_exact_source_composition_uses_source_values_and_current_formatting():
    prompt = "Return exactly '<approach> | <profile>' and nothing else."
    source_texts = (
        "I am choosing between Docker Compose and PostgreSQL directly on Windows.",
        "I track that constraint under profile VX-1234ABCD.",
    )
    composition = ExactSourceComposition(
        selections=[
            ExactSourceSelection(source_index=0, verbatim_value="Docker Compose"),
            ExactSourceSelection(source_index=1, verbatim_value="VX-1234ABCD"),
        ],
        separator=" | ",
    )

    assert (
        validate_exact_source_composition(prompt, source_texts, composition)
        == "Docker Compose | VX-1234ABCD"
    )


def test_current_fallback_must_be_verbatim_current_text():
    prompt = "USER_PROMPT only; otherwise INSUFFICIENT."
    assert validate_current_literal(prompt, "INSUFFICIENT") == "INSUFFICIENT"
    with pytest.raises(ValueError, match="exact substring"):
        validate_current_literal(prompt, "UNKNOWN")


def test_default_retrieval_scope_excludes_model_outputs():
    assert source_types_for_scope(HistoricalEvidenceScope.USER_AUTHORED) == [
        EventType.USER_PROMPT
    ]
    assert EventType.INTERACTION_RESPONSE not in source_types_for_scope(
        HistoricalEvidenceScope.GENERAL_OR_CURRENT
    )
    assert allowed_event_types(HistoricalEvidenceScope.GENERAL_OR_CURRENT) == {
        EventType.USER_PROMPT,
        EventType.TOOL_RESULT,
        EventType.SYSTEM_EVENT,
    }


def test_general_scope_final_filter_is_fail_closed():
    user = _evidence(EventType.USER_PROMPT, "Current historical fact.", 1)
    assistant = _evidence(EventType.INTERACTION_RESPONSE, "Unrequested model claim.", 2)

    filtered = filter_memory_packet_for_scope(
        _packet(user, assistant),
        HistoricalEvidenceScope.GENERAL_OR_CURRENT,
    )

    assert filtered is not None
    assert [item.source_event_id for item in filtered.items] == [user.source_event_id]


def test_mixed_conversation_scope_allows_user_and_model_outputs():
    source_types = source_types_for_scope(HistoricalEvidenceScope.MIXED_CONVERSATION)
    assert EventType.USER_PROMPT in source_types
    assert EventType.INTERACTION_RESPONSE in source_types


def test_only_explicit_prior_assistant_references_opt_into_mixed_history():
    assert explicit_prior_assistant_reference("What did you tell me earlier?") is True
    assert explicit_prior_assistant_reference("What is my favorite color?") is False
