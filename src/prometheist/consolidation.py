"""Scheduled derivation over repeated observations; never rewrite canonical history."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from uuid import UUID

import psycopg

from prometheist import event_store
from prometheist.cognitive_store import get_record, list_records, put_record
from prometheist.percept_context import FrozenRecord, aware
from prometheist.percept_intake import ingest_percept, install_source_policy
from prometheist.perception import PerceptKind, PerceptModality, PerceptSource
from prometheist.percept_triage import SourcePolicy, TaskClass
from prometheist.semantic_memory import record_semantic_evidence
from prometheist.situations import Situation

DERIVATION_METHOD = "consolidation/v2"


def _record_page_semantic_evidence(
    conn: psycopg.Connection,
    by_percept: dict,
    *,
    derived_at: datetime,
) -> list[dict]:
    """Persist each unique observation as evidence, independent of page shape.

    A storage page is only a bounded execution unit. It must never decide
    whether evidence is semantically admissible. Stable evidence identities
    deduplicate overlapping situation snapshots and repeated consolidation
    passes, while the semantic resolver compares observation time rather than
    page order.
    """

    updates: list[dict] = []
    for state in sorted(
        by_percept.values(),
        key=lambda item: (
            item.observation.subject,
            item.observation.property,
            item.observed_at,
            str(item.percept_id),
        ),
    ):
        observation = state.observation
        if state.source_event_id is not None:
            source_event = event_store.get_event_by_id(conn, state.source_event_id)
            if source_event is None:
                raise RuntimeError(
                    f"semantic evidence source event is missing: {state.source_event_id}"
                )
            known_at = source_event.created_at
        else:
            # Compatibility for pre-v2 derived situation snapshots. New
            # ObservedState records always retain source_event_id.
            known_at = derived_at
        result = record_semantic_evidence(
            conn,
            subject=observation.subject,
            property=observation.property,
            value=observation.value,
            unit=observation.unit,
            confidence=observation.confidence,
            source_id=state.percept_id,
            observed_at=state.observed_at,
            known_at=known_at,
            resolved_at=derived_at,
            derivation_method=DERIVATION_METHOD,
        )
        updates.append(
            {
                "subject": observation.subject,
                "property": observation.property,
                "assertion_id": str(result.assertion.assertion_id),
                "evidence_id": str(result.evidence.evidence_id),
            }
        )
    return updates


def _load_or_create_consolidation_input(
    conn: psycopg.Connection,
    *,
    action_id: UUID,
    after_key: str,
) -> tuple[list[tuple[str, dict]], datetime, str | None]:
    """Freeze one action's exact input page and derivation time.

    A retry must never read a moving page after partially publishing semantic
    evidence. The input record is therefore durable before derivation begins.
    """

    input_key = str(action_id)
    existing = get_record(conn, "consolidation_input", input_key)
    if existing is None:
        rows = list_records(conn, "situation_ingest", after_key=after_key)
        payload = {
            "after_key": after_key,
            "record_keys": [key for key, _ in rows],
            "derived_at": datetime.now(timezone.utc).isoformat(),
            "next_cursor": rows[-1][0] if rows else None,
        }
        put_record(
            conn,
            "consolidation_input",
            input_key,
            payload,
            revision="1",
        )
        existing = payload
    elif existing.get("after_key", "") != after_key:
        raise ValueError("consolidation action replayed with a different cursor")

    derived_at = datetime.fromisoformat(str(existing["derived_at"]))
    aware(derived_at)
    rows: list[tuple[str, dict]] = []
    for key in existing.get("record_keys", []):
        data = get_record(conn, "situation_ingest", str(key))
        if data is None:
            raise RuntimeError(
                f"frozen consolidation input is missing situation_ingest record: {key}"
            )
        rows.append((str(key), data))
    return rows, derived_at, existing.get("next_cursor")


def consolidate_page(
    conn: psycopg.Connection,
    *,
    action_id: UUID,
    after_key: str = "",
) -> dict:
    """Process one frozen bounded page without making the page epistemic."""

    snapshots, derived_at, next_cursor = _load_or_create_consolidation_input(
        conn,
        action_id=action_id,
        after_key=after_key,
    )
    groups = defaultdict(dict)
    unique_states = {}

    for _key, data in snapshots:
        situation = Situation.model_validate(data)
        for state in situation.observed_state:
            observation = state.observation
            grouping_key = (observation.property, observation.unit)
            groups[grouping_key][(state.percept_id, observation.subject)] = state
            unique_states[
                (state.percept_id, observation.subject, observation.property)
            ] = state

    projections = []
    for (property_name, unit), by_percept in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        values = defaultdict(list)
        for state in by_percept.values():
            values[json.dumps(state.observation.value, sort_keys=True)].append(
                str(state.percept_id)
            )
        projections.append(
            {
                "property": property_name,
                "unit": unit,
                "observed_values": [
                    {
                        "value": json.loads(value),
                        "support_percept_ids": refs,
                    }
                    for value, refs in sorted(values.items())
                ],
                "epistemic_status": "DERIVED",
                "scope": "BOUNDED_SITUATION_PAGE",
                "variation_present": len(values) > 1,
                "universal_claim": False,
                "subject_refs": sorted(
                    {
                        state.observation.subject
                        for state in by_percept.values()
                    }
                ),
            }
        )

    semantic_updates = _record_page_semantic_evidence(
        conn,
        {index: state for index, state in enumerate(unique_states.values())},
        derived_at=derived_at,
    )
    result = {
        "projections": projections,
        "source_snapshot_ids": [
            data["snapshot_id"] for _, data in snapshots
        ],
        "next_cursor": next_cursor,
        "canonical_records_modified": False,
        "semantic_updates": semantic_updates,
        "derived_at": derived_at.isoformat(),
    }
    put_record(conn, "consolidation", str(action_id), result, revision="2")
    return result


class ConsolidationSchedule(FrozenRecord):
    schedule_id: UUID
    due_at: datetime
    after_key: str = ""


def schedule_consolidation(
    conn: psycopg.Connection, schedule: ConsolidationSchedule
) -> None:
    aware(schedule.due_at)
    put_record(
        conn,
        "consolidation_schedule",
        str(schedule.schedule_id),
        schedule.model_dump(mode="json"),
        revision="1",
    )


def emit_due_consolidations(
    conn: psycopg.Connection,
    *,
    now: datetime,
    after_key: str = "",
) -> str | None:
    """One scheduler poll page, safe to replay across process restarts."""

    aware(now)
    source_id = "scheduler:consolidation"
    install_source_policy(
        conn,
        SourcePolicy(
            source_id=source_id,
            kind=PerceptKind.SCHEDULED_EVENT,
            modalities=(PerceptModality.STRUCTURED,),
            deterministic_task=TaskClass.CONSOLIDATE,
            allowed_task_classes=(TaskClass.CONSOLIDATE,),
        ),
    )
    rows = list_records(conn, "consolidation_schedule", after_key=after_key)
    for key, data in rows:
        schedule = ConsolidationSchedule.model_validate(data)
        if schedule.due_at > now or get_record(conn, "schedule_delivery", key):
            continue
        percept = ingest_percept(
            conn,
            source=PerceptSource(
                source_id=source_id,
                kind=PerceptKind.SCHEDULED_EVENT,
                modality=PerceptModality.STRUCTURED,
                interface="scheduler",
            ),
            observation={
                "schedule_id": key,
                "after_key": schedule.after_key,
            },
            observed_at=schedule.due_at,
            delivery_id=key,
            correlation_id=schedule.schedule_id,
        )
        put_record(
            conn,
            "schedule_delivery",
            key,
            {"percept_id": str(percept.percept_id)},
            revision="1",
        )
    return rows[-1][0] if rows else None
