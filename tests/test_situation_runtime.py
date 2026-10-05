from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from prometheist import db, event_store
from prometheist.action_outcomes import issue_action, observe_action_outcome
from prometheist.attention_observation import (
    HostResourceMetrics,
    ReservationCreditHostResourceProbe,
)
from prometheist.attention_store import load_scheduler
from prometheist.cognitive_store import get_record, list_records, put_record, rebuild_heads
from prometheist.consolidation import ConsolidationSchedule, consolidate_page, emit_due_consolidations, schedule_consolidation
from prometheist.expectations import Expectation
from prometheist.models import EventType
from prometheist.ollama_runtime import OllamaRuntimeState
from prometheist.percept_context import Observation, PerceptContext
from prometheist.percept_intake import ingest_percept, install_source_policy
from prometheist.perception import PerceptKind, PerceptModality, PerceptSource
from prometheist.percept_triage import SourcePolicy
from prometheist.semantic_memory import (
    current_semantic_resolution,
    selected_assertion,
    semantic_evidence,
)
from prometheist.situation_runtime import (
    SituationStage,
    drain_situations,
    run_situation_task,
    submit_situation_page,
)
from prometheist.situations import register_expectation


class FixedProbe:
    def __init__(self, available=12000):
        self.available = available

    def capture(self):
        return HostResourceMetrics(platform="test", logical_cpu_count=8, cpu_utilization_percent=10,
                                   load_1m=0, memory_total_mib=16384, memory_available_mib=self.available)


class FixedOllamaRuntimeProbe:
    def __init__(self, *, resident: bool):
        self.resident = resident

    def capture(self):
        return OllamaRuntimeState(
            model="qwen3:4b-instruct-2507-q4_K_M",
            probe_ok=True,
            resident=self.resident,
            reported_name=(
                "qwen3:4b-instruct-2507-q4_K_M"
                if self.resident
                else None
            ),
            size_bytes=3_000 * 1024 * 1024 if self.resident else None,
            size_vram_bytes=0 if self.resident else None,
        )


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


def source_setup(conn):
    source = PerceptSource(source_id="test:monitor", kind=PerceptKind.SYSTEM_OBSERVATION, modality=PerceptModality.STRUCTURED)
    install_source_policy(conn, SourcePolicy(source_id=source.source_id, kind=source.kind, modalities=(source.modality,)))
    return source


def add_observation(conn, source, property_name="memory", value=300, expected=None, delivery_id=None, subject="worker:1"):
    return ingest_percept(conn, source=source, observation={property_name: value}, observed_at=datetime.now(timezone.utc),
                          delivery_id=delivery_id or str(uuid4()), context=PerceptContext(
                              entity_refs=(subject,), expectation_refs=(expected,) if expected else (),
                              observations=(Observation(subject=subject, property=property_name, value=value),)))


def consolidation_task_id(conn, task_ids):
    matches = []
    for task_id in task_ids:
        task = get_record(conn, "situation_task", str(task_id))
        if (
            task is not None
            and task["percept"]["source"]["source_id"] == "scheduler:consolidation"
        ):
            matches.append(task_id)
    assert len(matches) == 1
    return matches[0]


def register_memory_expectation(conn):
    now = datetime.now(timezone.utc)
    expectation = Expectation(expectation_id=uuid4(), subject="worker:1", property="memory",
                              expected_range=(500, 16000), normalization_scale=200, source="test-policy",
                              provenance=(uuid4(),), confidence=1.0, valid_from=now - timedelta(minutes=1),
                              valid_until=now + timedelta(hours=1))
    register_expectation(conn, expectation)
    return expectation.expectation_id


def test_heads_rebuild_exactly_without_mutating_events(conn):
    put_record(conn, "test", "a", {"value": 1}, revision="first")
    put_record(conn, "test", "a", {"value": 2}, revision="second")
    original = conn.execute("SELECT event_id, payload FROM events ORDER BY global_seq").fetchall()
    rebuild_heads(conn)
    assert get_record(conn, "test", "a") == {"value": 2}
    assert conn.execute("SELECT event_id, payload FROM events ORDER BY global_seq").fetchall() == original


def test_intake_replay_and_conflicting_delivery(conn):
    source = source_setup(conn)
    args = dict(source=source, observation={"value": 1}, observed_at=datetime.now(timezone.utc), delivery_id="delivery:1")
    first = ingest_percept(conn, **args)
    before = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    assert ingest_percept(conn, **args) == first
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == before
    with pytest.raises(ValueError, match="conflicting deterministic event retry"):
        ingest_percept(conn, **{**args, "observation": {"value": 2}})


def test_intake_crash_recovers_post_snapshot_receipt(conn, monkeypatch):
    from prometheist import percept_intake
    source = source_setup(conn)
    args = dict(source=source, observation={"value": 1}, observed_at=datetime.now(timezone.utc), delivery_id="interrupted")
    original = percept_intake.put_record
    def interrupted(connection, kind, *positional, **kwargs):
        if kind == "intake_receipt":
            raise RuntimeError("simulated process loss")
        return original(connection, kind, *positional, **kwargs)
    monkeypatch.setattr(percept_intake, "put_record", interrupted)
    with pytest.raises(RuntimeError):
        ingest_percept(conn, **args)
    monkeypatch.setattr(percept_intake, "put_record", original)
    ingest_percept(conn, **args)
    assert len(list_records(conn, "situation")) == 1
    assert len(list_records(conn, "situation")[0][1]["percept_ids"]) == 1


def test_source_configuration_can_revert_without_reusing_old_head(conn):
    a = SourcePolicy(source_id="test", kind=PerceptKind.SYSTEM_OBSERVATION, modalities=(PerceptModality.STRUCTURED,))
    b = a.model_copy(update={"response_required": True})
    for value in (a, b, a):
        install_source_policy(conn, value)
        assert get_record(conn, "source_policy", "test") == value.model_dump(mode="json")


def test_four_percepts_one_guarded_task_and_finite_action_feedback(conn, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    source = source_setup(conn)
    expected = register_memory_expectation(conn)
    for property_name, value in (("cpu", 98), ("memory", 300), ("heartbeat", False), ("lease", "expired")):
        add_observation(conn, source, property_name, value, expected if property_name == "memory" else None)
    candidates = list_records(conn, "situation_candidate")
    assert len(candidates) == 1
    assert len(candidates[0][1]["situation"]["percept_ids"]) == 4
    ids = submit_situation_page(conn, probe=FixedProbe())
    assert len(ids) == 1
    result = run_situation_task(conn, ids[0], probe=FixedProbe())
    assert result == {"response_required": False, "response_text": None}
    assert len(list_records(conn, "action_execution")) == 1
    outcomes = conn.execute("SELECT payload FROM events WHERE event_type = %s", (EventType.PERCEPT_OBSERVATION.value,)).fetchall()
    assert any(row[0]["percept"]["source"]["kind"] == "ACTION_OUTCOME" for row in outcomes)
    # Subsequent scheduler pages may process bookkeeping for the matched outcome,
    # but must never manufacture more actions for an unchanged old discrepancy.
    for _ in range(4):
        drain_situations(conn, probe=FixedProbe())
    assert len(list_records(conn, "action_execution")) == 1
    from prometheist import artifact_journal
    artifacts = artifact_journal.interaction_artifacts(ids[0])
    assert not any(value["artifact_type"] == "LLM_INVOCATION" for value in artifacts)
    assert len([value for value in artifacts if value["artifact_type"] == "STAGE_RESULT"]) == 8
    verification = artifact_journal.verify_interaction_chain(ids[0])
    assert verification["valid"] and verification["complete"]
    final = verification["last_artifact"]["payload"]
    assert final["last_completed_stage"] == "SITUATION_PERSIST"
    assert final["response_required"] is False and final["response_text"] is None
    assert len([item for item in final["artifact_chain"] if item["artifact_type"] == "STAGE_RESULT"]) == 8


def _capture_first_situation_claim_probe(
    conn,
    monkeypatch,
    *,
    model_stage: bool,
):
    from prometheist import situation_runtime

    source = source_setup(conn)
    add_observation(conn, source, value=500)
    now = datetime.now(timezone.utc)
    schedule_consolidation(
        conn,
        ConsolidationSchedule(schedule_id=uuid4(), due_at=now),
    )
    emit_due_consolidations(conn, now=now)
    task_ids = submit_situation_page(
        conn,
        probe=FixedProbe(),
        ollama_runtime_probe=FixedOllamaRuntimeProbe(resident=False),
    )
    task_id = consolidation_task_id(conn, task_ids)
    captured = {}

    class CapturingLauncher:
        def __init__(
            self,
            connection_factory,
            *,
            probe,
            policy,
            scheduler_key,
            **_kwargs,
        ):
            del connection_factory, policy, scheduler_key
            captured["probe"] = probe

        def launch(self, **_kwargs):
            raise RuntimeError("stop after launcher construction")

    monkeypatch.setattr(
        situation_runtime,
        "GuardedWorkerLauncher",
        CapturingLauncher,
    )
    if model_stage:
        monkeypatch.setattr(
            situation_runtime,
            "situation_stage_uses_model",
            lambda *_args, **_kwargs: True,
        )

    with pytest.raises(RuntimeError, match="stop after launcher construction"):
        run_situation_task(
            conn,
            task_id,
            probe=FixedProbe(),
            ollama_runtime_probe=FixedOllamaRuntimeProbe(resident=False),
        )
    return captured["probe"]


def test_situation_deterministic_stage_credits_unused_llm_reservation(
    conn,
    monkeypatch,
):
    probe = _capture_first_situation_claim_probe(
        conn,
        monkeypatch,
        model_stage=False,
    )

    assert isinstance(probe, ReservationCreditHostResourceProbe)
    assert probe.scheduled_memory_mib == 3_072
    assert probe.required_memory_mib == 512
    physical = probe.base_probe.capture()
    adjusted = probe.capture()
    assert adjusted.memory_available_mib == min(
        adjusted.memory_total_mib,
        physical.memory_available_mib + 2_560,
    )


def test_situation_model_stage_rechecks_ollama_residency(conn, monkeypatch):
    from prometheist.ollama_runtime import OllamaClaimHostResourceProbe

    probe = _capture_first_situation_claim_probe(
        conn,
        monkeypatch,
        model_stage=True,
    )

    assert isinstance(probe, OllamaClaimHostResourceProbe)
    assert probe.scheduled_memory_mib == 3_072
    probe.capture()
    assert probe.last_effective_required_memory_mib == 3_072
    assert probe.last_memory_credit_mib == 0


def test_nonresponse_consolidation_tail_stages_are_model_free(conn):
    source = source_setup(conn)
    add_observation(conn, source, value=500)
    now = datetime.now(timezone.utc)
    schedule_consolidation(
        conn,
        ConsolidationSchedule(schedule_id=uuid4(), due_at=now),
    )
    emit_due_consolidations(conn, now=now)
    task_ids = submit_situation_page(
        conn,
        probe=FixedProbe(),
        ollama_runtime_probe=FixedOllamaRuntimeProbe(resident=False),
    )
    task_id = consolidation_task_id(conn, task_ids)
    task_data = get_record(conn, "situation_task", str(task_id))
    task = __import__(
        "prometheist.situation_runtime",
        fromlist=["SituationTask"],
    ).SituationTask.model_validate(task_data)

    from prometheist.situation_runtime import situation_stage_uses_model

    assert (
        situation_stage_uses_model(
            conn,
            task,
            SituationStage.RETRIEVE,
        )
        is False
    )
    assert (
        situation_stage_uses_model(
            conn,
            task,
            SituationStage.RESPOND,
        )
        is False
    )
    assert (
        situation_stage_uses_model(
            conn,
            task,
            SituationStage.PERSIST,
        )
        is False
    )


def test_clean_worker_exit_without_a_durable_result_does_not_complete_task(conn):
    source = source_setup(conn)
    add_observation(conn, source)
    task_id, = submit_situation_page(conn, probe=FixedProbe())
    launches = []

    def launch(**kwargs):
        launches.append(kwargs["step_id"])
        return SimpleNamespace(process=SimpleNamespace(wait=lambda **_: 0))

    with pytest.raises(RuntimeError, match="missing durable situation result: SITUATION_MEMORY"):
        run_situation_task(conn, task_id, probe=FixedProbe(), launcher=SimpleNamespace(launch=launch))
    assert len(launches) == 1
    assert load_scheduler(conn).tasks[task_id].status.value != "COMPLETED"
    assert get_record(conn, "situation_completion", str(task_id)) is None
    assert not list_records(conn, "situation_progress")


@pytest.mark.parametrize("failure_boundary", ["manifest", "progress"])
def test_situation_finalization_recovers_without_repeating_work(conn, monkeypatch, failure_boundary):
    from prometheist import artifact_journal, situation_runtime

    source = source_setup(conn)
    add_observation(conn, source, expected=register_memory_expectation(conn))
    task_id, = submit_situation_page(conn, probe=FixedProbe())
    original_task = get_record(conn, "situation_task", str(task_id))
    snapshot_id = original_task["situation"]["snapshot_id"]
    write_manifest = artifact_journal.write_final_disposition_artifact
    put = situation_runtime.put_record

    def fail_manifest(**kwargs):
        raise OSError("simulated manifest interruption")

    def fail_progress(connection, kind, *args, **kwargs):
        if kind == "situation_progress":
            raise OSError("simulated progress interruption")
        return put(connection, kind, *args, **kwargs)

    if failure_boundary == "manifest":
        monkeypatch.setattr(artifact_journal, "write_final_disposition_artifact", fail_manifest)
    else:
        monkeypatch.setattr(situation_runtime, "put_record", fail_progress)
    with pytest.raises(OSError, match="simulated"):
        run_situation_task(conn, task_id, probe=FixedProbe())

    before = artifact_journal.interaction_artifacts(task_id)
    assert len([value for value in before if value["artifact_type"] == "STAGE_RESULT"]) == 8
    assert len(list_records(conn, "action_execution")) == 1
    assert not list_records(conn, "situation_progress")
    if failure_boundary == "manifest":
        assert load_scheduler(conn).tasks[task_id].status.value != "COMPLETED"

    monkeypatch.setattr(artifact_journal, "write_final_disposition_artifact", write_manifest)
    monkeypatch.setattr(situation_runtime, "put_record", put)
    if failure_boundary == "progress":
        # The normal polling path must find completed tasks even though their
        # worker assignments have already been released and action feedback has
        # replaced the candidate snapshot. The first page wraps the cursor.
        drain_situations(conn, probe=FixedProbe())
        drain_situations(conn, probe=FixedProbe())
        repaired = conn.execute(
            "SELECT payload FROM events WHERE payload->>'record_kind' = 'situation_progress' "
            "AND payload->'data'->>'snapshot_id' = %s", (snapshot_id,),
        ).fetchall()
        assert len(repaired) == 1
    progress_before_retry = list_records(conn, "situation_progress")

    def unexpected_launch(**kwargs):
        pytest.fail("completed situation work must not be executed again")

    for _ in range(2):
        result = run_situation_task(
            conn, task_id, probe=FixedProbe(), launcher=SimpleNamespace(launch=unexpected_launch),
        )
        assert result == {"response_required": False, "response_text": None}
    after = artifact_journal.interaction_artifacts(task_id)
    assert [entry["artifact_hash"] for entry in after[:len(before)]] == [entry["artifact_hash"] for entry in before]
    assert len([entry for entry in after if entry["artifact_type"] == "FINAL_DISPOSITION"]) == 1
    assert artifact_journal.verify_interaction_chain(task_id)["valid"]
    assert load_scheduler(conn).tasks[task_id].status.value == "COMPLETED"
    assert conn.execute(
        "SELECT count(*) FROM events WHERE payload->>'record_kind' = 'situation_progress' "
        "AND payload->'data'->>'snapshot_id' = %s", (snapshot_id,),
    ).fetchone()[0] == 1
    if failure_boundary == "progress":
        # Retrying an older completed task must not roll back the latest head.
        assert list_records(conn, "situation_progress") == progress_before_retry
    assert len(list_records(conn, "action_execution")) == 1


def test_situation_priority_cannot_override_memory_admission(conn):
    source = source_setup(conn)
    add_observation(conn, source, expected=register_memory_expectation(conn))
    ids = submit_situation_page(conn, probe=FixedProbe(10))
    assert len(ids) == 1
    assert not load_scheduler(conn).worker_visible_assignments()
    assert run_situation_task(conn, ids[0], probe=FixedProbe(10)) is None


@pytest.mark.parametrize(
    ("resident", "expected_memory_mib"),
    [(True, 512), (False, 3_072)],
)
def test_consolidation_admission_uses_ollama_incremental_memory(
    conn,
    resident,
    expected_memory_mib,
):
    source = source_setup(conn)
    add_observation(conn, source, value=500)
    now = datetime.now(timezone.utc)
    schedule_consolidation(
        conn,
        ConsolidationSchedule(schedule_id=uuid4(), due_at=now),
    )
    emit_due_consolidations(conn, now=now)

    task_ids = submit_situation_page(
        conn,
        probe=FixedProbe(),
        ollama_runtime_probe=FixedOllamaRuntimeProbe(resident=resident),
    )

    task_id = consolidation_task_id(conn, task_ids)
    task = load_scheduler(conn).tasks[task_id]
    estimate = task.metadata.process_resource_estimate
    assert estimate.memory_mib == expected_memory_mib
    assert estimate.llm_slots == 1
    assert task.resumable_state["resource_admission"]["memory_mib"] == (
        expected_memory_mib
    )
    assert task.resumable_state["resource_admission"]["requires_model"] is True
    assert (
        task.resumable_state["resource_admission"]["ollama_runtime_state"][
            "resident"
        ]
        is resident
    )


def test_scheduled_consolidation_executes_separately_and_preserves_sources(conn):
    source = source_setup(conn)
    add_observation(conn, source, value=500)
    original = conn.execute("SELECT event_id, payload FROM events ORDER BY global_seq").fetchall()
    now = datetime.now(timezone.utc)
    schedule = ConsolidationSchedule(schedule_id=uuid4(), due_at=now)
    schedule_consolidation(conn, schedule)
    assert not list_records(conn, "consolidation")
    emit_due_consolidations(conn, now=now)
    emit_due_consolidations(conn, now=now)
    for _ in range(4):
        drain_situations(conn, probe=FixedProbe())
    projections = list_records(conn, "consolidation")
    assert len(projections) == 1
    assert projections[0][1]["canonical_records_modified"] is False
    assert projections[0][1]["projections"][0]["observed_values"][0]["support_percept_ids"]
    assert len(projections[0][1]["semantic_updates"]) == 1
    for event_id, payload in original:
        assert event_store.get_event_by_id(conn, event_id).payload == payload

    resolution = current_semantic_resolution(conn, "worker:1", "memory")
    assertion = selected_assertion(conn, resolution)
    assert resolution is not None
    assert assertion is not None and assertion.value == 500
    assert len(semantic_evidence(conn, "worker:1", "memory")) == 1


def test_consolidation_records_variation_as_evidence_instead_of_skipping_it(conn):
    source = source_setup(conn)
    add_observation(conn, source, value=500, delivery_id="first")
    add_observation(conn, source, value=700, delivery_id="second")
    projection = consolidate_page(conn, action_id=uuid4())

    assert projection["projections"][0]["variation_present"] is True
    assert len(projection["semantic_updates"]) == 2
    assert len(semantic_evidence(conn, "worker:1", "memory")) == 2
    resolution = current_semantic_resolution(conn, "worker:1", "memory")
    assert resolution is not None
    assertion = selected_assertion(conn, resolution)
    assert assertion is not None and assertion.value == 700


def test_consolidation_retry_reuses_frozen_input_page(conn):
    source = source_setup(conn)
    add_observation(conn, source, value=500, delivery_id="first")
    action_id = uuid4()

    first = consolidate_page(conn, action_id=action_id)
    assert len(first["semantic_updates"]) == 1

    add_observation(conn, source, value=700, delivery_id="arrived-after-freeze")
    retried = consolidate_page(conn, action_id=action_id)

    assert retried == first
    assert len(semantic_evidence(conn, "worker:1", "memory")) == 1
    resolution = current_semantic_resolution(conn, "worker:1", "memory")
    assertion = selected_assertion(conn, resolution)
    assert assertion is not None and assertion.value == 500


def test_consolidation_keeps_subject_evidence_separate(conn):
    source = source_setup(conn)
    add_observation(conn, source, value=500, subject="worker:1", delivery_id="w1")
    add_observation(conn, source, value=900, subject="worker:2", delivery_id="w2")
    projection = consolidate_page(conn, action_id=uuid4())

    subjects = {item["subject"] for item in projection["semantic_updates"]}
    assert subjects == {"worker:1", "worker:2"}
    one = selected_assertion(
        conn, current_semantic_resolution(conn, "worker:1", "memory")
    )
    two = selected_assertion(
        conn, current_semantic_resolution(conn, "worker:2", "memory")
    )
    assert one is not None and one.value == 500
    assert two is not None and two.value == 900
    assert len(semantic_evidence(conn, "worker:1", "memory")) == 1
    assert len(semantic_evidence(conn, "worker:2", "memory")) == 1


def test_forged_action_success_receipt_fails_closed(conn):
    now = datetime.now(timezone.utc)
    action_id = uuid4()
    issue_action(conn, action_id=action_id, task_id=uuid4(), at=now)
    unrelated = put_record(conn, "test", "unrelated", {"observed_status": "SUCCEEDED"}, revision="1")
    with pytest.raises(ValueError, match="matching registered executor receipt"):
        observe_action_outcome(conn, action_id=action_id, receipt_event_id=unrelated, status="SUCCEEDED", observed_at=now)


def test_media_quarantine_revalidates_content_not_supplied_metadata(conn, tmp_path):
    from prometheist.percept_adapters import preserve_media, verify_media
    from prometheist.reflexes import quarantine_corrupt_media
    original = tmp_path / "input.bin"
    original.write_bytes(b"valid content")
    reference = preserve_media(original, mime_type="application/octet-stream")
    path = verify_media(reference)
    wrong_metadata = reference.model_copy(update={"byte_length": 1})
    assert not quarantine_corrupt_media(conn, wrong_metadata)
    assert verify_media(reference) == path
    path.write_bytes(b"corrupt content")
    assert quarantine_corrupt_media(conn, reference)
    assert path.exists()
    with pytest.raises(ValueError, match="quarantined"):
        verify_media(reference)
