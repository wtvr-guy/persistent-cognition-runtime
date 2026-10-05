"""Persistent input adapters, situation formation, and attention candidacy."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid5

import psycopg

from prometheist import event_store
from prometheist.cognitive_store import COGNITIVE_NAMESPACE, SYSTEM_CONVERSATION, get_record, put_record, record_lock
from prometheist.models import EventType
from prometheist.percept_context import PerceptContext
from prometheist.perception import Percept, PerceptKind, PerceptSource, normalize_percept
from prometheist.percept_triage import SourcePolicy
from prometheist.situations import persist_situations


def install_source_policy(conn: psycopg.Connection, policy: SourcePolicy) -> None:
    """Local application configuration API. Input observations cannot call this."""
    import hashlib
    revision = hashlib.sha256(policy.model_dump_json().encode()).hexdigest()
    with record_lock(conn, f"source-policy:{policy.source_id}"):
        if get_record(conn, "source_policy", policy.source_id) == policy.model_dump(mode="json"):
            return
        predecessor = conn.execute(
            "SELECT event_id FROM cognitive_heads WHERE record_kind = 'source_policy' AND record_key = %s",
            (policy.source_id,),
        ).fetchone()
        if predecessor:
            revision = f"{predecessor[0]}:{revision}"
        put_record(conn, "source_policy", policy.source_id, policy.model_dump(mode="json"), revision=revision)


def ingest_percept(
    conn: psycopg.Connection, *, source: PerceptSource, observation: Any,
    observed_at: datetime, delivery_id: str, context: PerceptContext | None = None,
    conversation_id: UUID = SYSTEM_CONVERSATION, correlation_id: UUID | None = None,
) -> Percept:
    """Persist exact admitted JSON/text, then derive bounded situation candidates.

    Delivery identity belongs to the source adapter (message ID, sample ID, etc.).
    Reusing it for different content fails closed through deterministic events.
    """
    if not delivery_id.strip():
        raise ValueError("delivery_id is required")
    if source.kind is PerceptKind.USER_INTERACTION:
        raise ValueError("explicit user prompts enter through begin_percept")
    policy_data = get_record(conn, "source_policy", source.source_id)
    if policy_data is None:
        raise ValueError("source has no installed application policy")
    policy = SourcePolicy.model_validate(policy_data)
    if source.kind is not policy.kind or source.modality not in policy.modalities:
        raise ValueError("source contract differs from installed policy")
    source_event_id = uuid5(COGNITIVE_NAMESPACE, f"intake:{source.source_id}:{delivery_id}")
    correlation = correlation_id or source_event_id
    try:
        percept = normalize_percept(
            source=source, observation=observation, observed_at=observed_at,
            source_event_id=source_event_id, conversation_id=conversation_id,
            correlation_id=correlation, response_required=policy.response_required, context=context,
        )
    except ValueError:
        from prometheist.percept_adapters import MediaReference
        from prometheist.reflexes import quarantine_corrupt_media
        from prometheist.perception import PerceptModality
        if source.modality in {PerceptModality.IMAGE, PerceptModality.AUDIO, PerceptModality.VIDEO,
                               PerceptModality.FILE, PerceptModality.DOCUMENT}:
            quarantine_corrupt_media(conn, MediaReference.model_validate(observation))
        raise
    with record_lock(conn, f"intake:{source_event_id}"):
        event_store.start_conversation(conn, conversation_id)
        event_store.record_event(
            conn, conversation_id=conversation_id, correlation_id=correlation,
            event_id=source_event_id, event_type=EventType.PERCEPT_OBSERVATION,
            source=source.source_id, payload={"observation": observation, "percept": percept.model_dump(mode="json")},
            payload_text=observation if isinstance(observation, str) else None,
        )
        completed = get_record(conn, "intake_receipt", str(source_event_id))
        if completed:
            return percept
        # Resume all post-admission steps if interrupted; each boundary is durable.
        for situation in persist_situations(conn, percept):
            key = str(situation.situation_id)
            with record_lock(conn, f"candidate:{key}"):
                current = get_record(conn, "situation", key)
                # An older retry cannot replace a candidate built from newer evidence.
                if current and current["snapshot_id"] == str(situation.snapshot_id):
                    put_record(conn, "situation_candidate", key,
                               {"situation": situation.model_dump(mode="json"),
                                "percept": percept.model_dump(mode="json"), "policy": policy.model_dump(mode="json")},
                               revision=str(situation.snapshot_id))
        put_record(conn, "intake_receipt", str(source_event_id), {"percept_id": str(percept.percept_id)}, revision="1")
    return percept
