from __future__ import annotations

from datetime import UTC, datetime
import uuid

import pytest

from persistent_cognition.model_evidence_budget import ModelEvidenceBudgetExceeded
from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.percept_response_runtime import ResponseMemoryPackage
from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
)


class _NoModelCallAllowed:
    def post(self, *args, **kwargs):
        raise AssertionError("oversized evidence reached the model transport")


def test_single_oversized_memory_event_cannot_expand_v2_model_context(monkeypatch):
    huge_content = "OVERSIZED-CANONICAL-EVIDENCE " + ("x" * 2_000_000)
    packet = MemoryPacket(
        memory_request_id=uuid.uuid4(),
        need=MemoryNeed(query_text="What does the stored evidence say?", limit=1),
        supported=True,
        items=[
            MemoryEvidence(
                source_event_id=uuid.uuid4(),
                event_type=EventType.USER_PROMPT,
                source="user",
                created_at=datetime.now(UTC),
                conversation_id=uuid.uuid4(),
                conversation_seq=1,
                global_seq=1,
                content=huge_content,
            )
        ],
    )
    package = ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=0,
    )
    client = UserPromptLLM(base_url="http://ollama.test", model="model:test")
    client._client = _NoModelCallAllowed()
    monkeypatch.setenv("PCR_MAX_MODEL_EVIDENCE_ITEM_BYTES", "16384")
    monkeypatch.setenv("PCR_MAX_MODEL_EVIDENCE_TOTAL_BYTES", "65536")

    with pytest.raises(ModelEvidenceBudgetExceeded, match="item exceeds"):
        client.generate_final_response(
            "What does the stored evidence say?",
            package,
            (),
            response_policy=ResponsePolicy(
                evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
                surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            ),
        )


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
