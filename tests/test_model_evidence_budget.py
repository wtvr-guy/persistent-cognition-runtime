from __future__ import annotations

from datetime import UTC, datetime
import uuid

import pytest

from prometheist.model_evidence_budget import (
    ModelEvidenceBudget,
    ModelEvidenceBudgetExceeded,
    configured_model_evidence_budget,
    validate_capability_result_content,
    validate_memory_packet_content,
)
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket


def _evidence(content: str, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid.uuid4(),
        event_type=EventType.USER_PROMPT,
        source="test",
        created_at=datetime.now(UTC),
        conversation_id=uuid.uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=content,
    )


def _packet(*contents: str) -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid.uuid4(),
        need=MemoryNeed(query_text="budget test", limit=max(1, len(contents))),
        supported=bool(contents),
        items=[_evidence(content, index + 1) for index, content in enumerate(contents)],
    )


def test_single_memory_item_fails_closed_before_exceeding_item_budget():
    budget = ModelEvidenceBudget(max_item_bytes=16, max_total_bytes=64)
    with pytest.raises(ModelEvidenceBudgetExceeded, match="item exceeds"):
        validate_memory_packet_content(_packet("x" * 17), budget=budget)


def test_multiple_memory_items_cannot_exceed_aggregate_budget():
    budget = ModelEvidenceBudget(max_item_bytes=16, max_total_bytes=24)
    with pytest.raises(ModelEvidenceBudgetExceeded, match="aggregate"):
        validate_memory_packet_content(_packet("a" * 12, "b" * 12, "c"), budget=budget)


def test_capability_results_share_the_same_aggregate_evidence_budget():
    budget = ModelEvidenceBudget(max_item_bytes=128, max_total_bytes=140)
    results = ({"payload": "a" * 60}, {"payload": "b" * 60})
    with pytest.raises(ModelEvidenceBudgetExceeded, match="combined evidence"):
        validate_capability_result_content(
            results,
            budget=budget,
            prior_evidence_bytes=40,
        )


def test_budget_environment_values_are_governed_and_validated(monkeypatch):
    monkeypatch.setenv("PROMETHEIST_MAX_MODEL_EVIDENCE_ITEM_BYTES", "8192")
    monkeypatch.setenv("PROMETHEIST_MAX_MODEL_EVIDENCE_TOTAL_BYTES", "32768")
    budget = configured_model_evidence_budget()
    assert budget.max_item_bytes == 8192
    assert budget.max_total_bytes == 32768


def test_invalid_budget_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("PROMETHEIST_MAX_MODEL_EVIDENCE_ITEM_BYTES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        configured_model_evidence_budget()


def test_item_budget_cannot_exceed_total_budget():
    with pytest.raises(ValueError, match="must not exceed"):
        ModelEvidenceBudget(max_item_bytes=65, max_total_bytes=64)
