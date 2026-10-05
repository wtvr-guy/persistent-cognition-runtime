"""Action issuance is an intention; only subsequent evidence records an outcome."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID, uuid5

import psycopg

from prometheist import event_store
from prometheist.cognitive_store import get_record, put_record
from prometheist.expectations import Expectation
from prometheist.percept_context import Observation, PerceptContext
from prometheist.percept_intake import ingest_percept, install_source_policy
from prometheist.perception import Percept, PerceptKind, PerceptModality, PerceptSource
from prometheist.percept_triage import SourcePolicy
from prometheist.situations import register_expectation

ACTION_EXPECTATION_SECONDS = 3600
OUTCOME_SOURCE = "system:action-outcome"


def issue_action(conn: psycopg.Connection, *, action_id: UUID, task_id: UUID, at: datetime,
                 entity_refs: tuple[str, ...] = ()) -> UUID:
    event_id = put_record(conn, "action_intention", str(action_id),
                         {"action_id": str(action_id), "task_id": str(task_id),
                          "issued_at": at.isoformat(), "status": "ISSUED", "entity_refs": list(entity_refs)}, revision="1")
    expectation_id = uuid5(action_id, "completion-expectation")
    register_expectation(conn, Expectation(
        expectation_id=expectation_id, subject=f"action:{action_id}", property="completion",
        expected_value="SUCCEEDED", source="application:registered-task", provenance=(event_id,),
        confidence=1.0, valid_from=at, valid_until=at + timedelta(seconds=ACTION_EXPECTATION_SECONDS),
    ))
    return expectation_id


def observe_action_outcome(
    conn: psycopg.Connection, *, action_id: UUID, receipt_event_id: UUID,
    status: Literal["SUCCEEDED", "FAILED", "UNKNOWN"], observed_at: datetime,
) -> Percept:
    if status not in {"SUCCEEDED", "FAILED", "UNKNOWN"}:
        raise ValueError("issued is not an observed outcome")
    intention = get_record(conn, "action_intention", str(action_id))
    receipt = event_store.get_event_by_id(conn, receipt_event_id)
    if (intention is None or receipt is None or receipt.source != "cognitive_runtime"
        or receipt.payload.get("record_kind") != "action_execution"
        or receipt.payload.get("record_key") != str(action_id)
        or receipt.payload.get("data", {}).get("observed_status") != status):
        raise ValueError("action outcomes require the matching registered executor receipt")
    install_source_policy(conn, SourcePolicy(source_id=OUTCOME_SOURCE, kind=PerceptKind.ACTION_OUTCOME,
                                             modalities=(PerceptModality.STRUCTURED,)))
    return ingest_percept(
        conn, source=PerceptSource(source_id=OUTCOME_SOURCE, kind=PerceptKind.ACTION_OUTCOME,
                                   modality=PerceptModality.STRUCTURED, interface="action-executor"),
        observation={"action_id": str(action_id), "receipt_event_id": str(receipt_event_id), "status": status},
        observed_at=observed_at, delivery_id=f"{action_id}:{receipt_event_id}", correlation_id=action_id,
        context=PerceptContext(
            entity_refs=tuple(intention["entity_refs"]), expectation_refs=(uuid5(action_id, "completion-expectation"),),
            observations=(Observation(subject=f"action:{action_id}", property="completion", value=status),),
            uncertainty=3 if status == "UNKNOWN" else 0,
        ),
    )
