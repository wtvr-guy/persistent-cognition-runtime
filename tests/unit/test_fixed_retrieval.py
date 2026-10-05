from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from prometheist import jit_memory
from prometheist.fixed_retrieval import merge_evidence
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist import percept_response_runtime as runtime
from prometheist.percept_response_worker import UserPromptLLM
from prometheist.response_policy import HistoricalEvidenceScope, ResponsePolicy, ResponseSurfaceMode


def context():
    return SimpleNamespace(interaction_id=uuid4(), conversation_id=uuid4(),
                           task_id=uuid4(), correlation_id=uuid4(),
                           before_global_seq=100, user_text="How have my priorities changed?")


def evidence(seq, *, kind=EventType.USER_PROMPT, content=None, reasons=()):
    return MemoryEvidence(source_event_id=UUID(int=seq), event_type=kind, source="user",
                          created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                          conversation_id=UUID(int=1), conversation_seq=seq, global_seq=seq,
                          content=content or f"exact evidence {seq}", retrieval_reasons=list(reasons))


def packet(*items):
    return MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=bool(items), items=list(items))


def test_empty_broad_route_does_not_stop_association_and_relation(monkeypatch):
    ctx = context()
    initial = packet(evidence(1), evidence(2))
    monkeypatch.setattr(runtime, "expand_canonical_neighbors", lambda conn, p, **kw: p)
    calls = []
    def request(conn, **kwargs):
        calls.append(kwargs)
        # The first route adds nothing; later routes must still run.
        return packet() if len(calls) == 1 else packet(evidence(10 + len(calls)))
    monkeypatch.setattr(jit_memory, "request_memory", request)
    result = runtime.retrieve_memory_package(None, ctx, initial, [EventType.USER_PROMPT])
    assert [c["recall_stage"] for c in calls] == list(jit_memory.AdaptiveRecallStage)
    assert all(c["before_global_seq"] == ctx.before_global_seq for c in calls)
    assert all(c["need"].query_text == ctx.user_text for c in calls)
    assert result.adaptive_recall_rounds == 4
    assert result.stop_reason == "routes_exhausted"
    assert {i.global_seq for i in result.memory_packet.items} >= {1, 2, 12, 13, 14}
    assert "sufficient" not in result.model_dump()
    assert "memory_deficit" not in result.model_dump()


def test_no_focus_records_skips_instead_of_repeating_broad(monkeypatch):
    calls = []
    monkeypatch.setattr(jit_memory, "request_memory", lambda conn, **kw: calls.append(kw) or packet())
    result = runtime.retrieve_memory_package(None, context(), packet(), [EventType.USER_PROMPT])
    assert len(calls) == 1
    assert [r["status"] for r in result.route_results] == ["completed", "missing_focus", "missing_focus", "missing_focus"]
    assert not result.memory_packet.items


def test_replay_produces_same_route_ids_and_package(monkeypatch):
    ctx = context()
    monkeypatch.setattr(jit_memory, "request_memory", lambda conn, **kw: MemoryPacket(
        memory_request_id=kw["memory_request_id"], need=kw["need"], supported=False, items=[]))
    initial = packet()
    one = runtime.retrieve_memory_package(None, ctx, initial, [EventType.USER_PROMPT])
    two = runtime.retrieve_memory_package(None, ctx, initial, [EventType.USER_PROMPT])
    assert one.model_dump_json() == two.model_dump_json()


def test_merge_preserves_temporal_route_and_opposition_diversity():
    ctx = context()
    saturated = packet(*(evidence(i) for i in range(1, 21)))
    opposing = packet(evidence(30, reasons=("OPPOSES",)), evidence(31, reasons=("SUPPORTS",)))
    result = merge_evidence(ctx, [saturated, opposing], [EventType.USER_PROMPT], 4)
    assert {i.global_seq for i in result.items} == {20, 30, 31, 1}
    assert len(saturated.items) == 20


def test_source_cutoff_filter_precedes_budget_and_records_exclusion():
    ctx = context()
    result = merge_evidence(ctx, [packet(evidence(100), evidence(101),
        evidence(99, kind=EventType.INTERACTION_RESPONSE), evidence(2))], [EventType.USER_PROMPT], 1)
    assert [i.global_seq for i in result.items] == [2]
    assert len(result.retrieval_trace["excluded"]) == 3


def test_merge_keeps_exact_bytes_and_combines_duplicate_provenance():
    ctx = context()
    source = evidence(1, content="é\n  exact bytes", reasons=("SUPPORTS",))
    duplicate = source.model_copy(update={"retrieval_reasons": ["BROAD"], "provenance_event_ids": [UUID(int=8)]})
    result = merge_evidence(ctx, [packet(source), packet(duplicate)], [EventType.USER_PROMPT], 2)
    assert len(result.items) == 1
    assert result.items[0].content == source.content
    assert result.items[0].retrieval_reasons == ["BROAD", "SUPPORTS"]
    assert result.items[0].provenance_event_ids == [UUID(int=8)]
    with pytest.raises(ValueError, match="conflicting canonical"):
        merge_evidence(ctx, [packet(source), packet(source.model_copy(update={"content": "altered"}))], [EventType.USER_PROMPT], 2)


def test_byte_bound_skips_oversized_views_without_truncating_sources(monkeypatch):
    monkeypatch.setenv("PROMETHEIST_MAX_MODEL_EVIDENCE_ITEM_BYTES", "20")
    monkeypatch.setenv("PROMETHEIST_MAX_MODEL_EVIDENCE_TOTAL_BYTES", "20")
    too_big = evidence(3, content="é" * 11)
    result = merge_evidence(context(), [packet(too_big, evidence(1, content="retained"))], [EventType.USER_PROMPT], 2)
    assert result.items[0].content == "retained"
    assert too_big.content == "é" * 11
    assert result.retrieval_trace["excluded"][str(too_big.source_event_id)] == "byte_budget"


def test_final_responder_gets_authoritative_work_with_empty_retrieval(monkeypatch):
    client = UserPromptLLM()
    received = []
    monkeypatch.setattr(client, "_text_with_evidence", lambda *a: received.append(a) or "result")
    result = client.generate_final_response("What happened?", runtime.ResponseMemoryPackage(memory_packet=packet()),
        ({"capability_id": "inspection", "result_data": {"status": "complete"}},),
        response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.EXTERNAL_TOOL,
                                       surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE))
    assert result == "result"
    assert "complete" in received[0][3]


def test_retrieval_stage_rejects_every_model_kind():
    client = UserPromptLLM(stage=runtime.PerceptStage.RETRIEVE_MEMORY)
    for kind in ("V2_MEMORY_REQUIREMENTS", "V2_MEMORY_SUFFICIENCY_USER_PROMPT", "FINAL_RESPONSE_V2"):
        with pytest.raises(RuntimeError, match="cannot invoke"):
            client._require_stage_specialization(kind)


def test_empty_historical_evidence_retains_explicit_abstention(monkeypatch):
    client = UserPromptLLM()
    monkeypatch.setattr(client, "_select_current_fallback_literal", lambda _: None)
    result = client.generate_final_response("What was my teacher's name?", runtime.ResponseMemoryPackage(memory_packet=packet()), (),
        response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
                                       surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE))
    assert result == "Persisted evidence is insufficient."
