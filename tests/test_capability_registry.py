from __future__ import annotations

import json
import uuid

import pytest

from prometheist import db, event_store
from prometheist.capability_registry import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityNeed,
    CapabilityRegistry,
    RegisteredCapability,
    deterministic_capability_request_id,
    request_capability,
)
from prometheist.models import EventType


def _registration(
    capability_id: str,
    kind: CapabilityKind,
    *terms: str,
    executor: str = "test.executor",
    execution_priority: int = 100,
    depends_on_capability_ids: tuple[str, ...] = (),
    selectable_as_external_work: bool = True,
) -> RegisteredCapability:
    return RegisteredCapability(
        descriptor=CapabilityDescriptor(
            capability_id=capability_id,
            kind=kind,
            description=f"Public description for {capability_id}.",
        ),
        routing_terms=terms,
        executor=executor,
        execution_priority=execution_priority,
        depends_on_capability_ids=depends_on_capability_ids,
        selectable_as_external_work=selectable_as_external_work,
    )


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def test_discovery_is_bounded_deterministic_and_canonical_first():
    registry = CapabilityRegistry(
        (
            _registration("weather_lookup", CapabilityKind.TOOL, "weather", "forecast"),
            _registration(
                "compare_records",
                CapabilityKind.WORKFLOW,
                "compare records",
                "reconcile records",
            ),
        )
    )
    need = CapabilityNeed(
        query_text="Please compare records from the two systems.",
        supplemental_query_texts=["weather forecast"],
        limit=1,
    )

    first = registry.discover_with_trace(need)
    second = registry.discover_with_trace(need)

    assert first == second
    matches, role, selected_text = first
    assert [item.descriptor.capability_id for item in matches] == ["compare_records"]
    assert role == "canonical"
    assert selected_text == need.query_text


def test_supplemental_query_runs_only_after_canonical_abstention():
    registry = CapabilityRegistry(
        (_registration("weather_lookup", CapabilityKind.TOOL, "weather forecast"),)
    )

    matches, role, selected_text = registry.discover_with_trace(
        CapabilityNeed(
            query_text="This canonical task has no installed match.",
            supplemental_query_texts=["weather forecast"],
        )
    )

    assert [item.descriptor.capability_id for item in matches] == ["weather_lookup"]
    assert role == "supplemental"
    assert selected_text == "weather forecast"


def test_discovery_abstains_and_honors_kind_and_exclusion_filters():
    registry = CapabilityRegistry(
        (
            _registration("weather_model", CapabilityKind.MODEL, "weather"),
            _registration("weather_tool", CapabilityKind.TOOL, "weather"),
        )
    )
    need = CapabilityNeed(
        query_text="weather",
        kinds=[CapabilityKind.TOOL],
        exclude_capability_ids=["weather_tool"],
    )

    assert registry.discover(need) == []


def test_registry_rejects_duplicate_ids_and_exposes_only_public_descriptors():
    private = _registration(
        "internal_memory",
        CapabilityKind.SERVICE,
        "secret routing phrase",
        executor="private.executor",
    )
    registry = CapabilityRegistry((private,))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(private)

    rendered = json.dumps(
        registry.discover(CapabilityNeed(query_text="secret routing phrase"))[0].model_dump(
            mode="json"
        )
    )
    assert "secret routing phrase" not in rendered
    assert "private.executor" not in rendered
    assert set(CapabilityKind) == {
        CapabilityKind.SERVICE,
        CapabilityKind.TOOL,
        CapabilityKind.MODEL,
        CapabilityKind.WORKFLOW,
    }


def test_work_plan_ignores_model_order_and_expands_dependencies():
    registry = CapabilityRegistry(
        (
            _registration(
                "collect_external_evidence",
                CapabilityKind.TOOL,
                "collect",
                execution_priority=40,
            ),
            _registration(
                "reconcile_evidence",
                CapabilityKind.WORKFLOW,
                "reconcile",
                execution_priority=10,
                depends_on_capability_ids=("collect_external_evidence",),
            ),
            _registration(
                "independent_check",
                CapabilityKind.TOOL,
                "check",
                execution_priority=20,
            ),
        )
    )
    catalog = registry.capability_catalog()
    index = {item.capability_id: position for position, item in enumerate(catalog)}

    # The model deliberately returns a non-execution order. Prometheist owns the plan.
    plan = registry.plan_execution(
        catalog,
        [index["reconcile_evidence"], index["independent_check"]]
    )

    assert plan.requested_catalog_indices == [
        index["reconcile_evidence"],
        index["independent_check"],
    ]
    # independent_check is runnable at priority 20 while collect_external_evidence
    # is runnable at priority 40. reconcile_evidence cannot become runnable until
    # collect_external_evidence completes, even though reconcile has priority 10.
    assert [item.capability_id for item in plan.items] == [
        "independent_check",
        "collect_external_evidence",
        "reconcile_evidence",
    ]


def test_work_plan_stably_breaks_ready_ties_by_priority_then_id():
    registry = CapabilityRegistry(
        (
            _registration("zeta", CapabilityKind.TOOL, "zeta", execution_priority=30),
            _registration("alpha", CapabilityKind.TOOL, "alpha", execution_priority=30),
            _registration("middle", CapabilityKind.TOOL, "middle", execution_priority=20),
        )
    )
    catalog = registry.capability_catalog()
    indices = list(reversed(range(len(catalog))))

    plan = registry.plan_execution(catalog, indices)

    assert [item.capability_id for item in plan.items] == ["middle", "alpha", "zeta"]


def test_work_plan_rejects_cycles():
    registry = CapabilityRegistry(
        (
            _registration(
                "alpha",
                CapabilityKind.WORKFLOW,
                "alpha",
                depends_on_capability_ids=("beta",),
            ),
            _registration(
                "beta",
                CapabilityKind.WORKFLOW,
                "beta",
                depends_on_capability_ids=("alpha",),
            ),
        )
    )
    catalog = registry.capability_catalog()
    alpha_index = next(
        index for index, item in enumerate(catalog) if item.capability_id == "alpha"
    )

    with pytest.raises(ValueError, match="cycle"):
        registry.plan_execution(catalog, [alpha_index])


def test_persisted_capability_lookup_is_idempotent_and_task_neutral(conn):
    conversation_id = uuid.uuid4()
    correlation_id = uuid.uuid4()
    task_id = uuid.uuid4()
    step_id = uuid.uuid4()
    event_store.start_conversation(conn, conversation_id)
    registry = CapabilityRegistry(
        (_registration("weather_lookup", CapabilityKind.TOOL, "weather"),)
    )
    need = CapabilityNeed(query_text="weather")

    first = request_capability(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        requester_task_id=task_id,
        requester_step_id=step_id,
        need=need,
        registry=registry,
    )
    repeated = request_capability(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        requester_task_id=task_id,
        requester_step_id=step_id,
        need=need,
        registry=registry,
    )

    assert repeated == first
    assert first.capability_request_id == deterministic_capability_request_id(step_id)
    assert "agent" not in first.model_dump_json().casefold()
    events = event_store.get_events_by_conversation(conn, conversation_id)
    assert [event.event_type for event in events] == [
        EventType.CAPABILITY_REQUEST,
        EventType.CAPABILITY_PACKET,
    ]
