"""Narrow real-Ollama smoke gate for the v2 pre-cognitive work selector."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from prometheist.capability_registry import CapabilityDescriptor, CapabilityKind
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist.percept_response_runtime import PreCognitiveDisposition
from prometheist.percept_response_worker import UserPromptLLM
from tests._cli_helpers import ollama_available, print_transcript


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not ollama_available(), reason="Ollama is not reachable"),
]


def _evidence(content: str, *, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=EventType.USER_PROMPT,
        source="native-pre-cognitive-smoke",
        created_at=datetime.now(timezone.utc),
        conversation_id=uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=content,
    )


def test_real_ollama_precognitive_selector_returns_valid_work_requirement():
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="Which established Kestrel constraint applies?"),
        supported=True,
        items=[
            _evidence(
                "For Project Kestrel, never use Docker; deploy PostgreSQL directly on Windows.",
                seq=1,
            )
        ],
    )
    catalog = (
        CapabilityDescriptor(
            capability_id="external.inspect",
            kind=CapabilityKind.WORKFLOW,
            description="Inspect an external source only when required.",
        ),
    )

    prompt = (
        "Using only the supplied established memory, answer which Kestrel constraint "
        "applies. Do not inspect or consult any outside source."
    )
    decision = UserPromptLLM().decide_disposition(
        prompt,
        packet,
        catalog,
    )
    print_transcript(f"\nPre-cognitive work smoke — User:\n{prompt}")
    print_transcript(
        "\nPre-cognitive work smoke — capability_indices:\n"
        f"{decision.capability_indices}"
    )
    assert isinstance(decision, PreCognitiveDisposition)
    assert decision.response_required is True
    assert decision.capability_indices == []
