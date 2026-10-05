from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from prometheist import artifact_journal, db
from prometheist.attention_store import load_scheduler
from prometheist.interaction_contracts import DurableInteraction
from prometheist.interaction_store import load_interaction, save_interaction
from prometheist.perception import (
    AdvisorySemanticClassification,
    PerceptKind,
    PerceptModality,
    PerceptSource,
    SalienceDisposition,
    evaluate_salience,
    normalize_anomaly_percept,
    normalize_percept,
    normalize_scheduled_percept,
    normalize_user_interaction_percept,
)
from prometheist.percept_response_runtime import PerceptStage, begin_percept
from prometheist.percept_response_worker import UserPromptLLM, _execute_claimed_user_prompt_step
from prometheist.worker_protocol import deterministic_worker_step_id
from prometheist.worker_store import guarded_claim_worker_step, load_worker_result


NOW = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)


class FixedProbe:
    def capture(self):
        from prometheist.attention_observation import HostResourceMetrics

        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=10,
            load_1m=0,
            memory_total_mib=16_384,
            memory_available_mib=12_000,
        )


def test_normalize_percept_supports_structured_observations_deterministically() -> None:
    correlation_id = uuid.uuid4()
    source = PerceptSource(
        source_id="sensor:health",
        kind=PerceptKind.SYSTEM_OBSERVATION,
        modality=PerceptModality.STRUCTURED,
        interface="monitor",
    )

    percept = normalize_percept(
        source=source,
        observation={"status": "failed", "component": "disk", "count": 2},
        observed_at=NOW,
        correlation_id=correlation_id,
    )

    assert percept.source.modality is PerceptModality.STRUCTURED
    assert percept.normalized_text == '{"component":"disk","count":2,"status":"failed"}'
    assert percept.input_buffer.segments == (percept.normalized_text,)

    repeated = normalize_percept(
        source=source,
        observation={"component": "disk", "count": 2, "status": "failed"},
        observed_at=NOW,
        correlation_id=correlation_id,
    )
    assert repeated == percept


def test_percept_id_distinguishes_percept_class_contracts() -> None:
    correlation_id = uuid.uuid4()
    observed_at = NOW
    scheduled = normalize_percept(
        source=PerceptSource(
            source_id="shared-source",
            kind=PerceptKind.SCHEDULED_EVENT,
            modality=PerceptModality.STRUCTURED,
            interface="scheduler",
        ),
        observation={"value": 1},
        observed_at=observed_at,
        correlation_id=correlation_id,
    )
    anomaly = normalize_percept(
        source=PerceptSource(
            source_id="shared-source",
            kind=PerceptKind.ANOMALY_ALERT,
            modality=PerceptModality.STRUCTURED,
            interface="anomaly-detector",
        ),
        observation={"value": 1},
        observed_at=observed_at,
        correlation_id=correlation_id,
    )

    assert scheduled.percept_id != anomaly.percept_id


def test_normalize_percept_rejects_modality_mismatch() -> None:
    with pytest.raises(ValueError):
        normalize_percept(
            source=PerceptSource(
                source_id="sensor:health",
                kind=PerceptKind.SYSTEM_OBSERVATION,
                modality=PerceptModality.TEXT,
                interface="monitor",
            ),
            observation={"status": "failed"},
            observed_at=NOW,
            correlation_id=uuid.uuid4(),
        )


def test_structured_scalar_boolean_payload_is_allowed() -> None:
    percept = normalize_percept(
        source=PerceptSource(
            source_id="sensor:flag",
            kind=PerceptKind.EXTERNAL_OBSERVATION,
            modality=PerceptModality.STRUCTURED,
            interface="webhook",
        ),
        observation=True,
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assert percept.normalized_text == "true"
    assert percept.features.contains_structured_payload is True


def test_metric_scalar_is_not_marked_as_structured_payload() -> None:
    percept = normalize_percept(
        source=PerceptSource(
            source_id="sensor:temperature",
            kind=PerceptKind.EXTERNAL_OBSERVATION,
            modality=PerceptModality.METRIC,
            interface="telemetry",
        ),
        observation=1.5,
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assert percept.features.contains_structured_payload is False


def test_metric_scalar_uses_stable_json_normalization() -> None:
    percept = normalize_percept(
        source=PerceptSource(
            source_id="sensor:temperature",
            kind=PerceptKind.EXTERNAL_OBSERVATION,
            modality=PerceptModality.METRIC,
            interface="telemetry",
        ),
        observation=1.5,
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assert percept.normalized_text == "1.5"


def test_metric_percept_rejects_boolean_payload() -> None:
    with pytest.raises(ValueError):
        normalize_percept(
            source=PerceptSource(
                source_id="sensor:temperature",
                kind=PerceptKind.EXTERNAL_OBSERVATION,
                modality=PerceptModality.METRIC,
                interface="telemetry",
            ),
            observation=True,
            observed_at=NOW,
            correlation_id=uuid.uuid4(),
        )


def test_metric_percept_rejects_null_payload() -> None:
    with pytest.raises(ValueError):
        normalize_percept(
            source=PerceptSource(
                source_id="sensor:temperature",
                kind=PerceptKind.EXTERNAL_OBSERVATION,
                modality=PerceptModality.METRIC,
                interface="telemetry",
            ),
            observation=None,
            observed_at=NOW,
            correlation_id=uuid.uuid4(),
        )


def test_scheduled_event_is_a_first_class_non_user_percept() -> None:
    percept = normalize_scheduled_percept(
        source_id="scheduler:daily-summary",
        observation={"kind": "scheduled-review", "window": "daily"},
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assessment = evaluate_salience(percept)

    assert percept.source.kind is PerceptKind.SCHEDULED_EVENT
    assert percept.response_required is False
    assert assessment.disposition is SalienceDisposition.IGNORE


def test_scheduled_percept_rejects_raw_text_payload() -> None:
    with pytest.raises(ValueError):
        normalize_scheduled_percept(
            source_id="scheduler:daily-summary",
            observation="plain text is not a structured scheduled payload",
            observed_at=NOW,
            correlation_id=uuid.uuid4(),
        )


def test_structured_percept_rejects_non_json_payload() -> None:
    with pytest.raises(ValueError):
        normalize_scheduled_percept(
            source_id="scheduler:daily-summary",
            observation={"when": NOW},
            observed_at=NOW,
            correlation_id=uuid.uuid4(),
        )


def test_anomaly_alert_is_a_first_class_non_user_percept() -> None:
    percept = normalize_anomaly_percept(
        source_id="integrity:watchdog",
        observation={"status": "error", "detail": "crash", "severity": "urgent"},
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assessment = evaluate_salience(percept)

    assert percept.source.kind is PerceptKind.ANOMALY_ALERT
    assert percept.response_required is False
    assert assessment.disposition is SalienceDisposition.IGNORE


def test_user_percept_builds_bounded_input_buffer() -> None:
    conversation_id = uuid.uuid4()
    correlation_id = uuid.uuid4()
    source_event_id = uuid.uuid4()
    user_text = "A" * 700

    percept = normalize_user_interaction_percept(
        user_text=user_text,
        observed_at=NOW,
        correlation_id=correlation_id,
        source_event_id=source_event_id,
        conversation_id=conversation_id,
    )

    assert percept.response_required is True
    assert percept.input_buffer.truncated is True
    assert percept.input_buffer.total_segments == 5
    assert percept.input_buffer.retained_segment_indices == (0, 1, 3, 4)
    assert len(percept.input_buffer.segments) == 4


def test_advisory_semantic_classification_has_no_policy_authority() -> None:
    percept = normalize_user_interaction_percept(
        user_text="Please review this roadmap update.",
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
        source_event_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
    )

    assessment = evaluate_salience(
        percept,
        advisory_classification=AdvisorySemanticClassification(
            label="REFLEX",
            confidence=1.0,
            source_model="test-model",
        ),
    )

    assert assessment.disposition is SalienceDisposition.DELIBERATE
    assert assessment.advisory_only is True
    assert assessment.advisory_classification is not None
    assert assessment.advisory_classification.authoritative is False


def test_salience_does_not_use_lexical_substrings() -> None:
    percept = normalize_user_interaction_percept(
        user_text="Explain how this address changed.",
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
        source_event_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
    )

    assessment = evaluate_salience(percept)

    assert "plan" not in assessment.trigger_terms
    assert "add" not in assessment.trigger_terms


def test_unstructured_integrity_words_cannot_authorize_reflex() -> None:
    percept = normalize_percept(
        source=PerceptSource(
            source_id="sensor:integrity",
            kind=PerceptKind.SYSTEM_OBSERVATION,
            modality=PerceptModality.TEXT,
            interface="daemon",
        ),
        observation="error failure crash",
        observed_at=NOW,
        correlation_id=uuid.uuid4(),
    )

    assessment = evaluate_salience(percept)

    assert assessment.disposition is SalienceDisposition.IGNORE
    assert assessment.preauthorized_reflexes == ()


def test_begin_percept_persists_normalized_percept_and_salience() -> None:
    connection = db.get_connection()
    try:
        conversation_id = uuid.uuid4()
        interaction = begin_percept(
            connection,
            "Please investigate this error failure crash immediately.",
            conversation_id,
            probe=FixedProbe(),
            clock=lambda: NOW,
        )

        stored = load_interaction(connection, interaction.interaction_id)
        assert stored.percept is not None
        assert stored.percept.source.kind is PerceptKind.USER_INTERACTION
        assert stored.percept.source_event_id == stored.user_prompt_event_id
        assert stored.salience_assessment is not None
        assert stored.salience_assessment.percept_id == stored.percept.percept_id
        assert stored.salience_assessment.disposition is SalienceDisposition.DELIBERATE

        scheduler = load_scheduler(connection)
        task = scheduler.tasks[interaction.task_id]
        assert task.resumable_state["percept_id"] == str(stored.percept.percept_id)
        assert task.resumable_state["salience_disposition"] == "DELIBERATE"
    finally:
        connection.close()


def test_reference_stage_journals_percept_and_salience_before_completion() -> None:
    connection = db.get_connection()
    now = datetime.now(timezone.utc)
    try:
        interaction = begin_percept(
            connection,
            "Please investigate this error failure crash immediately.",
            uuid.uuid4(),
            probe=FixedProbe(),
            clock=lambda: now,
        )
        step_id = deterministic_worker_step_id(interaction.assignment_id, "V2_RESOLVE_REFERENCES")
        attempt = guarded_claim_worker_step(
            connection,
            step_id=step_id,
            worker_id="test-resolve-worker",
            probe=FixedProbe(),
            clock=lambda: now,
        )
        assert attempt.envelope is not None
        _execute_claimed_user_prompt_step(
            connection,
            UserPromptLLM(stage=PerceptStage.RESOLVE_REFERENCES),
            claim_id=attempt.envelope.claim.claim_id,
            worker_id="test-resolve-worker",
            scheduler_key="default",
        )
        result = load_worker_result(connection, step_id)
        assert result is not None
        assert result.output["percept"]["source"]["kind"] == "USER_INTERACTION"
        assert result.output["salience_assessment"]["disposition"] == "DELIBERATE"
        recovered = artifact_journal.load_stage_result_artifact(
            interaction.interaction_id, PerceptStage.RESOLVE_REFERENCES.value
        )
        assert recovered is not None
        assert recovered["output"] == result.output
    finally:
        connection.close()


def test_save_interaction_backfills_missing_percept_payloads() -> None:
    connection = db.get_connection()
    try:
        interaction = DurableInteraction(
            interaction_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            correlation_id=uuid.uuid4(),
            user_prompt_event_id=uuid.uuid4(),
            before_global_seq=1,
            task_id=uuid.uuid4(),
            assignment_id=uuid.uuid4(),
            user_text="user text",
            percept=normalize_user_interaction_percept(
                user_text="user text",
                observed_at=NOW,
                correlation_id=uuid.uuid4(),
                source_event_id=uuid.uuid4(),
                conversation_id=uuid.uuid4(),
            ),
            salience_assessment=None,
        )
        # This contract test only exercises the storage retry/backfill logic on an
        # already-present row shape; use the live path for end-to-end relational setup.
        with connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (conversation_id) VALUES (%s)
                ON CONFLICT (conversation_id) DO NOTHING
                """,
                (interaction.conversation_id,),
            )
            cur.execute(
                """
                INSERT INTO attention_tasks (
                    task_id, task_key, created_seq, criticality, service_class,
                    interruption_policy, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    interaction.task_id,
                    "test:interaction",
                    999_001,
                    "USER_BLOCKING",
                    "INTERACTIVE",
                    "CHECKPOINT_ONLY",
                    "QUEUED",
                ),
            )
            cur.execute(
                """
                INSERT INTO attention_scheduler_state (scheduler_key)
                VALUES ('default')
                ON CONFLICT (scheduler_key) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO attention_scheduling_epochs (
                    scheduler_key, epoch_id, epoch_sequence, scheduler_cycle,
                    admission_policy_version, assignment_policy_version, status
                ) VALUES ('default', %s, 1, 0, 'test', 'test', 'COMMITTED')
                ON CONFLICT (scheduler_key, epoch_id) DO NOTHING
                """,
                (uuid.uuid4(),),
            )
            cur.execute(
                """
                INSERT INTO attention_assignments (
                    scheduler_key, assignment_id, task_id, task_revision,
                    created_epoch_sequence, status
                ) VALUES ('default', %s, %s, 0, 1, 'READY')
                """,
                (interaction.assignment_id, interaction.task_id),
            )
            cur.execute(
                """
                INSERT INTO events (
                    event_id, conversation_id, correlation_id, conversation_seq,
                    event_type, source, payload, payload_text
                ) VALUES (%s, %s, %s, 1, 'USER_PROMPT', 'user', '{}'::jsonb, 'user text')
                """,
                (
                    interaction.user_prompt_event_id,
                    interaction.conversation_id,
                    interaction.correlation_id,
                ),
            )
            cur.execute(
                """
                INSERT INTO attention_interactions (
                    scheduler_key, interaction_id, protocol_version,
                    conversation_id, correlation_id, user_prompt_event_id,
                    before_global_seq, task_id, assignment_id, user_text,
                    percept_payload, salience_assessment_payload
                ) VALUES (
                    'default', %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL
                )
                """,
                (
                    interaction.interaction_id,
                    interaction.protocol_version,
                    interaction.conversation_id,
                    interaction.correlation_id,
                    interaction.user_prompt_event_id,
                    interaction.before_global_seq,
                    interaction.task_id,
                    interaction.assignment_id,
                    interaction.user_text,
                ),
            )
        connection.commit()

        stored = save_interaction(connection, interaction)
        assert stored.percept is not None
        reloaded = load_interaction(connection, interaction.interaction_id)
        assert reloaded.percept is not None
        assert reloaded.percept.percept_id == interaction.percept.percept_id
    finally:
        connection.close()
