"""Narrow real-Ollama smoke gate for the v2 final responder."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import PerceptStage, ResponseMemoryPackage
from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseAnswerKind,
    ResponseSurfaceMode,
)
from tests._cli_helpers import ollama_available, print_transcript
from tests._native_artifact_assertions import (
    assert_recalled_literal,
    assert_response_evidence_receipt,
    print_artifact_receipt,
)


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not ollama_available(), reason="Ollama is not reachable"),
]


def _evidence(
    event_type: EventType,
    content: str,
    *,
    conversation_id,
    conversation_seq: int,
    global_seq: int,
) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=event_type,
        source="native-response-smoke",
        created_at=datetime.now(timezone.utc),
        conversation_id=conversation_id,
        conversation_seq=conversation_seq,
        global_seq=global_seq,
        content=content,
    )


def test_real_ollama_response_reconciles_recent_relational_evidence():
    historical_conversation = uuid4()
    active_conversation = uuid4()
    profile_token = "VX-NATIVE01"
    plan_label = f"BlueHarbor-{uuid4().hex[:6].upper()}"
    prior_response = _evidence(
        EventType.INTERACTION_RESPONSE,
        f"Docker Compose | {profile_token}",
        conversation_id=active_conversation,
        conversation_seq=13,
        global_seq=32,
    )
    prior_question = _evidence(
        EventType.USER_PROMPT,
        "Which approach conflicts with my established Kestrel rule?",
        conversation_id=active_conversation,
        conversation_seq=8,
        global_seq=27,
    )
    plan_name = _evidence(
        EventType.USER_PROMPT,
        f"Call the active Kestrel plan {plan_label}.",
        conversation_id=active_conversation,
        conversation_seq=1,
        global_seq=20,
    )
    historical_rule = _evidence(
        EventType.USER_PROMPT,
        "For Project Kestrel, never use Docker; deploy PostgreSQL directly on Windows "
        f"under profile {profile_token}.",
        conversation_id=historical_conversation,
        conversation_seq=1,
        global_seq=1,
    )
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="What nickname and which approach was just ruled out?"),
        supported=True,
        items=[prior_response, prior_question, plan_name, historical_rule],
    )
    package = ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=0,
    )

    interaction_id = uuid4()
    interaction = SimpleNamespace(
        interaction_id=interaction_id,
        conversation_id=active_conversation,
        correlation_id=uuid4(),
        task_id=uuid4(),
        assignment_id=uuid4(),
    )
    prompt = (
        "What nickname are we using for this plan, and which approach did you just rule "
        "out? Answer naturally and briefly."
    )
    answer = UserPromptLLM(
        interaction=interaction,
        stage=PerceptStage.RESPOND,
        claim_id=uuid4(),
    ).generate_final_response(
        prompt,
        package,
        (),
        response_policy=ResponsePolicy(
            evidence_scope=HistoricalEvidenceScope.MIXED_CONVERSATION,
            surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            answer_kind=ResponseAnswerKind.EXTRACTIVE_VALUES,
        ),
    )
    assert answer.strip()
    print_transcript(f"\nRelational response smoke — User:\n{prompt}")
    print_transcript(f"\nRelational response smoke — Persistent Cognition:\n{answer}")
    print_artifact_receipt(
        "Relational response smoke",
        assert_response_evidence_receipt(
            interaction_id=interaction_id,
            required_event_ids=(plan_name.source_event_id, prior_response.source_event_id),
            require_complete=False,
        ),
    )
    assert_recalled_literal(answer, plan_label, label="Relational response smoke")
    print_transcript("HUMAN REVIEW REQUIRED: judge the relational response above.")


@pytest.mark.parametrize(("prompt", "expected_scope", "expected_kind"), [
    (
        "Tell me the codename for Project Oriole from persistent memory.",
        HistoricalEvidenceScope.USER_AUTHORED,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "List the source event IDs in the last MEMORY_PACKET retrieval record.",
        HistoricalEvidenceScope.DERIVED_INTERNAL,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "The codename for Project Oriole is 2FF0372B.",
        HistoricalEvidenceScope.GENERAL_OR_CURRENT,
        ResponseAnswerKind.SYNTHESIS,
    ),
])
def test_real_ollama_policy_distinguishes_stored_facts_from_internal_records(
    prompt, expected_scope, expected_kind,
):
    policy = UserPromptLLM()._response_policy(prompt)
    print_transcript(f"\nSource-policy regression — User:\n{prompt}")
    print_transcript(f"Source-policy regression — Scope: {policy.evidence_scope.value}")
    assert policy.evidence_scope is expected_scope
    assert policy.answer_kind is expected_kind
    assert policy.surface_mode is ResponseSurfaceMode.NATURAL_LANGUAGE
