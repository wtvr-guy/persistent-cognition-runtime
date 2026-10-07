"""Live percept-to-response runtime for the 2026-08-29 Persistent Cognition architecture.

This is the authoritative interactive path. Every LLM call is stateless. The
pre-cognitive role commits work/response disposition, fixed deterministic retrieval
collects bounded canonical evidence, and
the final responder receives memory plus authoritative work results directly.
"""
from __future__ import annotations

from persistent_cognition.contract_registry import SEMANTIC_CONTRACTS

from persistent_cognition.prompt_registry import (
    _FINAL_RESPONSE_PROMPT,
    _RESPONSE_POLICY_PROMPT,
    _CURRENT_FALLBACK_SELECTION_PROMPT,
    _EXACT_SOURCE_SELECTION_PROMPT,
    _EXACT_SOURCE_COMPOSITION_PROMPT,
)

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from itertools import islice
import json
import os
import subprocess
import sys
from typing import Any
from uuid import UUID, uuid4, uuid5

import psycopg
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from persistent_cognition import db, event_store, jit_memory
from persistent_cognition.canonical_neighborhood import expand_canonical_neighbors
from persistent_cognition.attention import (
    AttentionTask,
    InterruptionPolicy,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
)
from persistent_cognition.attention_aperture import ATTENTION_APERTURE_VERSION, open_attention_aperture
from persistent_cognition.attention_observation import (
    HostResourceProbe,
    LocalResourceAdmissionController,
    ResourceSafetyPolicy,
    SystemHostResourceProbe,
)
from persistent_cognition.attention_resources import ProcessResourceEstimate, ResourceEstimateSource
from persistent_cognition.attention_store import (
    DEFAULT_SCHEDULER_KEY,
    allocate_created_seq,
    load_scheduler,
    save_scheduler,
)
from persistent_cognition.capability_registry import (
    DEFAULT_REGISTRY,
    CapabilityDescriptor,
    CapabilityExecutionPlan,
    CapabilityRegistry,
)
from persistent_cognition.capability_runtime import CapabilityExecution, execute_registered_capability
from persistent_cognition.epistemic_authority import format_authority_bound_memory_packet
from persistent_cognition.interaction_contracts import (
    DurableInteraction,
    MemoryContext,
    deterministic_capability_memory_request_id,
    deterministic_interaction_event_id,
    deterministic_interaction_id,
    deterministic_interaction_task_id,
)
from persistent_cognition.interaction_store import save_interaction
from persistent_cognition.interaction_working_state import activate_working_state, load_working_state
from persistent_cognition.llm import (
    OllamaClient,
    _base_text_max_tokens,
    _build_verbatim_placeholder_maps,
    _format_capability_result_data,
    _mask_verbatim_literals,
    _quarantined_evidence,
    _restore_verbatim_literals,
    _retry_token_caps,
    _verbatim_source_texts,
)
from persistent_cognition.models import EventType, MemoryPacket
from persistent_cognition.model_evidence_budget import (
    configured_model_evidence_budget,
    memory_packet_content_bytes,
    validate_capability_result_content,
    validate_memory_packet_content,
    validate_rendered_evidence,
)
from persistent_cognition.native_policy import native_resource_safety_policy
from persistent_cognition.ollama_runtime import (
    OllamaClaimHostResourceProbe,
    OllamaRuntimeProbe,
    OllamaRuntimeState,
)
from persistent_cognition.perception import (
    normalize_user_interaction_percept,
)
from persistent_cognition.response_policy import (
    RESPONSE_POLICY_VERSION,
    CurrentFallbackSelection,
    ExactSourceComposition,
    ExactSourceSelection,
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
    filter_memory_packet_for_scope,
    explicit_prior_assistant_reference,
    scope_requires_historical_support,
    source_types_for_scope,
    validate_current_literal,
    validate_exact_source_composition,
    validate_exact_source_selection,
)
from persistent_cognition.worker_protocol import WorkerClaimEnvelope, WorkerEffectPolicy, deterministic_worker_step_id
from persistent_cognition.worker_runtime import GuardedWorkerLauncher
from persistent_cognition.worker_store import load_worker_result, register_worker_step

SOURCE = "percept_response_v2"
DEFAULT_RESPONSE_MEMORY_ITEM_LIMIT = 20
DEFAULT_ADAPTIVE_RECALL_ITEM_LIMIT = 10
DEFAULT_PERCEPT_WORKER_LEASE_SECONDS = 600
DEFAULT_PERCEPT_WORKER_TIMEOUT_SECONDS = 660
DEFAULT_POST_KILL_WAIT_SECONDS = 10.0
_RESPONSE_MEMORY_ITEM_LIMIT = int(
    os.environ.get("PCR_RESPONSE_MEMORY_ITEMS", str(DEFAULT_RESPONSE_MEMORY_ITEM_LIMIT))
)
_ADAPTIVE_RECALL_ITEM_LIMIT = int(
    os.environ.get("PCR_ADAPTIVE_RECALL_ITEMS", str(DEFAULT_ADAPTIVE_RECALL_ITEM_LIMIT))
)
PERCEPT_LLM_SLOTS = 1
_INTERNAL_MEMORY_EXECUTORS = {"jit_memory"}
_ADAPTIVE_RECALL_STAGES = (
    jit_memory.AdaptiveRecallStage.BROAD,
    jit_memory.AdaptiveRecallStage.ASSOCIATIVE,
    jit_memory.AdaptiveRecallStage.RELATIONAL,
    jit_memory.AdaptiveRecallStage.FOCUSED,
)



class PerceptStage(str, Enum):
    RESOLVE_REFERENCES = "V2_RESOLVE_REFERENCES"
    EVIDENCE_POLICY = "V2_EVIDENCE_POLICY"
    PRECOGNITIVE = "V2_PRECOGNITIVE"
    EXECUTE_WORK = "V2_EXECUTE_WORK"
    RETRIEVE_MEMORY = "V3_RETRIEVE_MEMORY"
    RESPOND = "V2_RESPOND"
    PERSIST_RESULT = "V2_PERSIST_RESULT"

    @property
    def capability(self) -> str:
        from persistent_cognition.contract_registry import STAGE_CONTRACTS
        return STAGE_CONTRACTS[self.value].capability


PERCEPT_STAGES = tuple(PerceptStage)
PERCEPT_CAPABILITIES = tuple(stage.capability for stage in PERCEPT_STAGES)


class PreCognitiveDisposition(BaseModel):
    """Closed semantic disposition; execution mechanics remain system-owned."""

    model_config = ConfigDict(extra="forbid")
    response_required: bool
    capability_indices: list[int] = Field(default_factory=list)

    @field_validator("capability_indices")
    @classmethod
    def validate_indices(cls, values: list[int]) -> list[int]:
        if any(index < 0 for index in values):
            raise ValueError("capability_indices must be non-negative")
        if len(values) != len(set(values)):
            raise ValueError("capability_indices must not contain duplicates")
        return values


class ResponseMemoryPackage(BaseModel):
    """Retrieved evidence; completion never certifies semantic sufficiency."""

    model_config = ConfigDict(extra="forbid")
    memory_packet: MemoryPacket
    retrieval_policy: str = "fixed-retrieval/v1"
    stop_reason: str = "routes_exhausted"
    adaptive_recall_rounds: int = 0
    route_results: list[dict[str, Any]] = Field(default_factory=list)








_GENERIC_INSUFFICIENT_RESPONSE = "Persisted evidence is insufficient."


def _admitted_capability_results(
    scope: HistoricalEvidenceScope,
    work_results: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if scope in {
        HistoricalEvidenceScope.EXTERNAL_TOOL,
        HistoricalEvidenceScope.GENERAL_OR_CURRENT,
    }:
        return work_results
    return ()


def _exact_source_texts(
    packet: MemoryPacket | None,
    work_results: tuple[dict[str, Any], ...],
    *,
    current_percept: str | None = None,
) -> tuple[str, ...]:
    texts: list[str] = []
    if packet is not None:
        texts.extend(item.content for item in packet.items)
    texts.extend(
        json.dumps(result, sort_keys=True, default=str, separators=(",", ":"))
        for result in work_results
    )
    if current_percept is not None:
        texts.append(current_percept)
    return tuple(texts)


def _format_exact_source_candidates(
    source_texts: tuple[str, ...],
    literal_to_placeholder: dict[str, str],
) -> str:
    blocks = [
        f"source_index: {index}\n"
        f"content: {_mask_verbatim_literals(source, literal_to_placeholder)}"
        for index, source in enumerate(source_texts)
    ]
    return "\n\n[Admitted exact-source candidates]\n" + "\n\n".join(blocks)


def _memory_evidence_refs(packet: MemoryPacket | None) -> tuple[str, ...]:
    """Return canonical source references for the model-facing memory view."""

    if packet is None:
        return ()
    return tuple(f"event:{item.source_event_id}" for item in packet.items)


class PerceptLLM(OllamaClient):
    """Shared transport helpers for independent stateless specialist roles."""

    def _set_artifact_evidence_refs(self, refs: tuple[str, ...]) -> None:
        """Expose causal evidence refs to artifact-aware subclasses."""

        del refs


class PerceptSpecialists(PerceptLLM):
    """Explicit semantic contracts, separate from the transport-only base."""

    def _response_policy(self, percept: str) -> ResponsePolicy:
        """Classify source and surface requirements from current authority only."""

        self._set_artifact_evidence_refs(())
        if explicit_prior_assistant_reference(percept):
            return ResponsePolicy(
                evidence_scope=HistoricalEvidenceScope.MIXED_CONVERSATION,
                surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
            )

        def validate_policy(content: str) -> ResponsePolicy:
            policy = ResponsePolicy.model_validate_json(content)
            validate_current_literal(percept, policy.insufficient_literal)
            return policy.model_copy(update={"insufficient_literal": None})

        last_error: Exception | None = None
        for token_cap in _retry_token_caps(_base_text_max_tokens()):
            try:
                content = self._structured_with_evidence(
                    "V2_RESPONSE_POLICY",
                    _RESPONSE_POLICY_PROMPT,
                    percept,
                    _quarantined_evidence(),
                    SEMANTIC_CONTRACTS["V2_RESPONSE_POLICY"].output_schema(),
                    token_cap,
                )
                return self._validated_model_output(
                    kind="V2_RESPONSE_POLICY",
                    raw_output=content,
                    validator=lambda: validate_policy(content),
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"response policy failed to validate: {last_error}")

    def _select_current_fallback_literal(self, percept: str) -> str | None:
        """Select a no-support literal without exposing historical evidence."""

        self._set_artifact_evidence_refs(())
        for token_cap in _retry_token_caps(_base_text_max_tokens()):
            try:
                content = self._structured_with_evidence(
                    "V2_CURRENT_FALLBACK_SELECTION",
                    _CURRENT_FALLBACK_SELECTION_PROMPT,
                    percept,
                    _quarantined_evidence(),
                    SEMANTIC_CONTRACTS["V2_CURRENT_FALLBACK_SELECTION"].output_schema(),
                    token_cap,
                )
                return self._validated_model_output(
                    kind="V2_CURRENT_FALLBACK_SELECTION",
                    raw_output=content,
                    validator=lambda: validate_current_literal(
                        percept,
                        CurrentFallbackSelection.model_validate_json(content).verbatim_value,
                    ),
                )
            except (ValidationError, ValueError):
                continue
        return None

    def _select_exact_source_substring(
        self,
        percept: str,
        packet: MemoryPacket | None,
        work_results: tuple[dict[str, Any], ...],
        *,
        include_current: bool,
    ) -> str:
        source_texts = _exact_source_texts(
            packet,
            work_results,
            current_percept=percept if include_current else None,
        )
        if not source_texts:
            raise ValueError("exact-source response has no admitted source candidates")
        self._set_artifact_evidence_refs(_memory_evidence_refs(packet))
        literal_to_placeholder, placeholder_to_literal = _build_verbatim_placeholder_maps(
            *source_texts
        )
        evidence = _quarantined_evidence(
            _format_exact_source_candidates(source_texts, literal_to_placeholder)
        )
        validate_rendered_evidence(
            (evidence,),
            budget=configured_model_evidence_budget(),
        )
        last_error: Exception | None = None

        def validate_selection(content: str) -> str:
            selection = ExactSourceSelection.model_validate_json(content)
            restored = selection.model_copy(
                update={
                    "verbatim_value": _restore_verbatim_literals(
                        selection.verbatim_value,
                        placeholder_to_literal,
                    )
                }
            )
            return validate_exact_source_selection(source_texts, restored)

        for token_cap in _retry_token_caps(_base_text_max_tokens()):
            try:
                content = self._structured_with_evidence(
                    "V2_EXACT_SOURCE_SELECTION",
                    _EXACT_SOURCE_SELECTION_PROMPT,
                    _mask_verbatim_literals(percept, literal_to_placeholder),
                    evidence,
                    SEMANTIC_CONTRACTS["V2_EXACT_SOURCE_SELECTION"].output_schema(),
                    token_cap,
                )
                return self._validated_model_output(
                    kind="V2_EXACT_SOURCE_SELECTION",
                    raw_output=content,
                    validator=lambda: validate_selection(content),
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"exact-source selection failed to validate: {last_error}")

    def _select_exact_source_composition(
        self,
        percept: str,
        packet: MemoryPacket | None,
        work_results: tuple[dict[str, Any], ...],
        *,
        include_current: bool,
    ) -> str:
        source_texts = _exact_source_texts(
            packet,
            work_results,
            current_percept=percept if include_current else None,
        )
        if not source_texts:
            raise ValueError("exact-source composition has no admitted source candidates")
        self._set_artifact_evidence_refs(_memory_evidence_refs(packet))
        literal_to_placeholder, placeholder_to_literal = _build_verbatim_placeholder_maps(
            *source_texts
        )
        evidence = _quarantined_evidence(
            _format_exact_source_candidates(source_texts, literal_to_placeholder)
        )
        validate_rendered_evidence(
            (evidence,),
            budget=configured_model_evidence_budget(),
        )
        last_error: Exception | None = None

        def validate_composition(content: str) -> str:
            composition = ExactSourceComposition.model_validate_json(content)
            restored = composition.model_copy(
                update={
                    "selections": [
                        selection.model_copy(
                            update={
                                "verbatim_value": _restore_verbatim_literals(
                                    selection.verbatim_value,
                                    placeholder_to_literal,
                                )
                            }
                        )
                        for selection in composition.selections
                    ]
                }
            )
            return validate_exact_source_composition(percept, source_texts, restored)

        for token_cap in _retry_token_caps(_base_text_max_tokens()):
            try:
                content = self._structured_with_evidence(
                    "V2_EXACT_SOURCE_COMPOSITION",
                    _EXACT_SOURCE_COMPOSITION_PROMPT,
                    _mask_verbatim_literals(percept, literal_to_placeholder),
                    evidence,
                    SEMANTIC_CONTRACTS["V2_EXACT_SOURCE_COMPOSITION"].output_schema(),
                    token_cap,
                )
                return self._validated_model_output(
                    kind="V2_EXACT_SOURCE_COMPOSITION",
                    raw_output=content,
                    validator=lambda: validate_composition(content),
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"exact-source composition failed to validate: {last_error}")

    def generate_final_response(
        self,
        percept: str,
        package: ResponseMemoryPackage,
        work_results: tuple[dict[str, Any], ...],
        *,
        response_policy: ResponsePolicy,
    ) -> str:
        packet = package.memory_packet
        budget = configured_model_evidence_budget()
        validate_memory_packet_content(packet, budget=budget)
        prior_evidence_bytes = memory_packet_content_bytes(packet)
        validate_capability_result_content(
            work_results,
            budget=budget,
            prior_evidence_bytes=prior_evidence_bytes,
        )

        policy = response_policy.model_copy(deep=True)
        admitted_packet = filter_memory_packet_for_scope(packet, policy.evidence_scope)
        admitted_results = _admitted_capability_results(policy.evidence_scope, work_results)
        has_admitted_history = bool(admitted_packet and admitted_packet.items)
        has_admitted_result = bool(admitted_results)
        if (
            scope_requires_historical_support(policy.evidence_scope)
            and not has_admitted_history
            and not has_admitted_result
        ):
            self._set_artifact_evidence_refs(())
            return self._select_current_fallback_literal(percept) or (
                _GENERIC_INSUFFICIENT_RESPONSE
            )

        include_current = policy.evidence_scope is HistoricalEvidenceScope.GENERAL_OR_CURRENT
        if policy.surface_mode is ResponseSurfaceMode.EXACT_SOURCE_SUBSTRING:
            return self._select_exact_source_substring(
                percept,
                admitted_packet,
                admitted_results,
                include_current=include_current,
            )
        if policy.surface_mode is ResponseSurfaceMode.EXACT_SOURCE_COMPOSITION:
            return self._select_exact_source_composition(
                percept,
                admitted_packet,
                admitted_results,
                include_current=include_current,
            )

        result_source_texts = _exact_source_texts(None, admitted_results)
        literal_to_placeholder, placeholder_to_literal = _build_verbatim_placeholder_maps(
            *_verbatim_source_texts(percept, admitted_packet),
            *result_source_texts,
        )
        masked_percept = _mask_verbatim_literals(percept, literal_to_placeholder)
        status_view = ("\n\n[Retrieval status]\n" + package.retrieval_policy
                       + ": " + package.stop_reason
                       + ". This is route execution status, not an answerability verdict. "
                       "Assess the admitted content below.")
        memory_view = format_authority_bound_memory_packet(
            admitted_packet,
            literal_to_placeholder=literal_to_placeholder,
        )
        work_view = _format_capability_result_data(admitted_results)
        validate_rendered_evidence(
            (status_view, memory_view, work_view),
            budget=budget,
        )
        refs = list(_memory_evidence_refs(admitted_packet))
        self._set_artifact_evidence_refs(tuple(refs))
        answer = self._text_with_evidence(
            "FINAL_RESPONSE_V2",
            _FINAL_RESPONSE_PROMPT,
            masked_percept,
            _quarantined_evidence(
                status_view,
                memory_view,
                work_view,
            ),
        )
        return _restore_verbatim_literals(answer, placeholder_to_literal)


def _external_capability_catalog(registry: CapabilityRegistry) -> tuple[CapabilityDescriptor, ...]:
    """Memory expansion is cognitive substrate, never pre-cognitive selectable work."""

    return tuple(
        descriptor
        for descriptor in registry.capability_catalog()
        if registry.get(descriptor.capability_id).executor not in _INTERNAL_MEMORY_EXECUTORS
    )


def _packet_event_ids(packet: MemoryPacket) -> tuple[UUID, ...]:
    return tuple(item.source_event_id for item in packet.items)


def _effective_adaptive_stage(
    requested: jit_memory.AdaptiveRecallStage,
    packet: MemoryPacket,
) -> tuple[jit_memory.AdaptiveRecallStage, list[UUID]]:
    """Choose the strongest stage whose focus preconditions are actually met."""

    ids = [item.source_event_id for item in packet.items]
    if requested is jit_memory.AdaptiveRecallStage.BROAD or not ids:
        return jit_memory.AdaptiveRecallStage.BROAD, []
    if requested is jit_memory.AdaptiveRecallStage.ASSOCIATIVE:
        return requested, list(islice(ids, len(_ADAPTIVE_RECALL_STAGES)))
    if requested is jit_memory.AdaptiveRecallStage.RELATIONAL:
        iterator = iter(ids)
        first = next(iterator, None)
        second = next(iterator, None)
        if first is not None and second is not None:
            return requested, [first, second]
        if first is not None:
            return jit_memory.AdaptiveRecallStage.FOCUSED, [first]
        return jit_memory.AdaptiveRecallStage.BROAD, []
    return jit_memory.AdaptiveRecallStage.FOCUSED, [ids[0]]


def retrieve_memory_package(
    conn: psycopg.Connection,
    interaction: MemoryContext,
    initial_packet: MemoryPacket,
    source_types: list[EventType],
) -> ResponseMemoryPackage:
    """Fixed bounded routes. No LLM, generated deficit, or coverage judge."""
    from persistent_cognition.fixed_retrieval import merge_evidence

    packets = [initial_packet]
    packet = merge_evidence(interaction, packets, source_types, _RESPONSE_MEMORY_ITEM_LIMIT)
    route_results = []
    if packet.items and source_types:
        # A separate route prevents a saturated aperture from starving neighbors.
        neighbors = expand_canonical_neighbors(
            conn, packet, source_types=source_types,
            before_global_seq=interaction.before_global_seq,
            item_limit=_RESPONSE_MEMORY_ITEM_LIMIT + len(packet.items),
            memory_request_id=uuid5(interaction.interaction_id, "fixed-neighbors:v1"),
        )
        initial_ids = set(_packet_event_ids(packet))
        neighbors = neighbors.model_copy(update={
            "items": [item for item in neighbors.items if item.source_event_id not in initial_ids]
        })
        packets.append(neighbors)
        packet = merge_evidence(interaction, packets, source_types, _RESPONSE_MEMORY_ITEM_LIMIT)
    for round_index, requested in enumerate(_ADAPTIVE_RECALL_STAGES):
        stage, focus_ids = _effective_adaptive_stage(requested, packet)
        # Do not silently run BROAD four times when a route has no focus roots.
        if stage is not requested:
            route_results.append({"route": requested.value, "status": "missing_focus"})
            continue
        need = jit_memory.build_memory_need(
            interaction.user_text,
            focus_event_ids=focus_ids, include_persisted_history=True,
            conversation_id=None, limit=_ADAPTIVE_RECALL_ITEM_LIMIT,
            source_types=source_types,
        )
        expansion = jit_memory.request_memory(
            conn, conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            requesting_component=f"task:{interaction.task_id}/fixed-retrieval:{requested.value}",
            need=need, before_global_seq=interaction.before_global_seq,
            memory_request_id=uuid5(interaction.interaction_id, f"fixed-retrieval:v1:{requested.value}"),
            recall_stage=requested,
        )
        previous = set(_packet_event_ids(packet))
        packets.append(expansion)
        packet = merge_evidence(interaction, packets, source_types, _RESPONSE_MEMORY_ITEM_LIMIT)
        route_results.append({"route": requested.value, "status": "completed",
                              "new_event_ids": sorted(str(i) for i in set(_packet_event_ids(packet)) - previous),
                              "memory_request_id": str(expansion.memory_request_id)})
    return ResponseMemoryPackage(
        memory_packet=packet,
        adaptive_recall_rounds=sum(r["status"] == "completed" for r in route_results),
        route_results=route_results,
    )


def begin_percept(
    conn: psycopg.Connection,
    user_text: str,
    conversation_id: UUID,
    *,
    correlation_id: UUID | None = None,
    probe: HostResourceProbe | None = None,
    policy: ResourceSafetyPolicy | None = None,
    ollama_runtime_state: OllamaRuntimeState | None = None,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> DurableInteraction:
    from persistent_cognition.resource_limits import validate_text_intake
    validate_text_intake(user_text)
    normalized = user_text.strip()
    if not normalized:
        raise ValueError("user_text must not be empty")
    effective_policy = policy or native_resource_safety_policy()
    interaction_memory_mib = (
        ollama_runtime_state.incremental_process_memory_mib(effective_policy)
        if ollama_runtime_state is not None
        else effective_policy.default_llm_process_memory_mib
    )
    residency_label = (
        ollama_runtime_state.residency_label
        if ollama_runtime_state is not None
        else "unobserved-cold-fallback"
    )
    event_store.start_conversation(conn, conversation_id)
    correlation = correlation_id or uuid4()
    interaction_id = deterministic_interaction_id(conversation_id, correlation)
    prompt = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": user_text},
        payload_text=user_text,
        event_id=deterministic_interaction_event_id(interaction_id, "user-prompt"),
        journal_id=interaction_id,
    )
    percept = normalize_user_interaction_percept(
        user_text=user_text,
        observed_at=prompt.created_at,
        correlation_id=correlation,
        source_event_id=prompt.event_id,
        conversation_id=conversation_id,
    )
    from persistent_cognition.situations import persist_situations
    situations = persist_situations(conn, percept)
    salience_assessment = situations[0].salience
    scheduler = load_scheduler(conn, scheduler_key=scheduler_key)
    task_id = deterministic_interaction_task_id(interaction_id)
    scheduler.submit(
        AttentionTask(
            task_id=task_id,
            task_key=f"percept-v2:{interaction_id}",
            created_seq=allocate_created_seq(conn),
            metadata=SchedulingMetadata(
                criticality=TaskCriticality.USER_BLOCKING,
                service_class=ServiceClass.INTERACTIVE,
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
                required_capabilities=list(PERCEPT_CAPABILITIES),
                process_resource_estimate=ProcessResourceEstimate(
                    cpu_units=effective_policy.default_process_cpu_units,
                    memory_mib=interaction_memory_mib,
                    llm_slots=PERCEPT_LLM_SLOTS,
                    source=ResourceEstimateSource.CONSERVATIVE_DEFAULT,
                    basis=(
                        f"{effective_policy.policy_version}: bounded stateless percept pipeline; "
                        f"Ollama admission state={residency_label}"
                    ),
                ),
            ),
            resumable_state={
                "interaction_id": str(interaction_id),
                "percept_id": str(percept.percept_id),
                "situation_snapshot_ids": [str(value.snapshot_id) for value in situations],
                "salience_disposition": salience_assessment.disposition.value,
                "resource_admission": {
                    "memory_mib": interaction_memory_mib,
                    "ollama_runtime_state": (
                        ollama_runtime_state.model_dump(mode="json")
                        if ollama_runtime_state is not None
                        else None
                    ),
                },
            },
        )
    )
    controller = LocalResourceAdmissionController(
        scheduler,
        probe=probe,
        policy=effective_policy,
        clock=clock,
    )
    controller.plan_scheduling_epoch()
    save_scheduler(conn, scheduler, scheduler_key=scheduler_key)
    assignments = [
        value for value in scheduler.worker_visible_assignments() if value.task_id == task_id
    ]
    if len(assignments) != PERCEPT_LLM_SLOTS:
        raise RuntimeError("interaction was not safely admitted to one assignment")
    interaction = DurableInteraction(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation,
        user_prompt_event_id=prompt.event_id,
        before_global_seq=prompt.global_seq,
        task_id=task_id,
        assignment_id=assignments[0].assignment_id,
        user_text=normalized,
        percept=percept,
        salience_assessment=salience_assessment,
    )
    save_interaction(conn, interaction, scheduler_key=scheduler_key)
    _ensure_steps(conn, interaction, scheduler_key=scheduler_key)
    return interaction


def _ensure_steps(
    conn: psycopg.Connection,
    interaction: DurableInteraction,
    *,
    scheduler_key: str,
) -> None:
    for stage in PERCEPT_STAGES:
        effect_policy = (
            WorkerEffectPolicy.IDEMPOTENT_WITH_KEY
            if stage in {PerceptStage.EXECUTE_WORK, PerceptStage.PERSIST_RESULT}
            else WorkerEffectPolicy.NO_EXTERNAL_EFFECT
        )
        register_worker_step(
            conn,
            assignment_id=interaction.assignment_id,
            step_key=stage.value,
            capability=stage.capability,
            input_refs=[f"event:{interaction.user_prompt_event_id}"],
            effect_policy=effect_policy,
            scheduler_key=scheduler_key,
        )


def _stage_result(
    conn: psycopg.Connection,
    interaction: DurableInteraction,
    stage: PerceptStage,
    scheduler_key: str,
) -> dict[str, Any]:
    step_id = deterministic_worker_step_id(interaction.assignment_id, stage.value)
    result = load_worker_result(conn, step_id, scheduler_key=scheduler_key)
    if result is None:
        raise RuntimeError(f"percept stage {stage.value} is incomplete")
    return dict(result.output)


def _structured_execution(execution: CapabilityExecution) -> dict[str, Any]:
    return {
        "plan_position": execution.plan_position,
        "capability_id": execution.capability_id,
        "executor": execution.executor,
        "result_data": dict(execution.result_data),
        "memory_request_id": (
            str(execution.memory_packet.memory_request_id)
            if execution.memory_packet is not None
            else None
        ),
    }


def _validated_response_policy(stage_output: dict[str, Any]) -> ResponsePolicy:
    """Rehydrate one committed policy only when its version and allowlist agree."""

    if stage_output.get("response_policy_version") != RESPONSE_POLICY_VERSION:
        raise RuntimeError("persisted response policy version is unsupported")
    policy = ResponsePolicy.model_validate(stage_output.get("response_policy"))
    expected_source_types = [
        event_type.value for event_type in source_types_for_scope(policy.evidence_scope)
    ]
    if stage_output.get("response_source_types") != expected_source_types:
        raise RuntimeError("persisted response policy/source allowlist mismatch")
    return policy


def _execute_stage(
    conn: psycopg.Connection,
    llm: PerceptSpecialists,
    envelope: WorkerClaimEnvelope,
    interaction: DurableInteraction,
    *,
    scheduler_key: str,
    registry: CapabilityRegistry,
) -> tuple[dict[str, Any], list[str]]:
    stage = PerceptStage(envelope.step.step_key)
    if stage is not PerceptStage.RESOLVE_REFERENCES:
        bootstrap = _stage_result(conn, interaction, PerceptStage.RESOLVE_REFERENCES,
                                  scheduler_key=scheduler_key)
        expected = bootstrap.get("capability_snapshot")
        if expected is not None and expected != registry.snapshot():
            raise RuntimeError("capability bindings changed during the durable interaction")

    if stage is PerceptStage.RESOLVE_REFERENCES:
        return {
            "capability_snapshot": registry.snapshot(),
            "working_state_available": load_working_state(conn, interaction.conversation_id)
            is not None,
            "percept": (
                interaction.percept.model_dump(mode="json")
                if interaction.percept is not None
                else None
            ),
            "salience_assessment": (
                interaction.salience_assessment.model_dump(mode="json")
                if interaction.salience_assessment is not None
                else None
            ),
        }, []

    if stage is PerceptStage.EVIDENCE_POLICY:
        response_policy = llm._response_policy(interaction.user_text)
        source_types = source_types_for_scope(response_policy.evidence_scope)
        return {
            "response_policy_version": RESPONSE_POLICY_VERSION,
            "response_policy": response_policy.model_dump(mode="json"),
            "response_source_types": [event_type.value for event_type in source_types],
        }, []

    if stage is PerceptStage.PRECOGNITIVE:
        evidence_policy = _stage_result(
            conn,
            interaction,
            PerceptStage.EVIDENCE_POLICY,
            scheduler_key,
        )
        response_policy = _validated_response_policy(evidence_policy)
        source_types = source_types_for_scope(response_policy.evidence_scope)
        aperture_packet = open_attention_aperture(
            conn,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            requester_task_id=interaction.task_id,
            user_text=interaction.user_text,
            before_global_seq=interaction.before_global_seq,
            source_types=source_types,
        )
        catalog = _external_capability_catalog(registry)
        disposition = llm.decide_disposition(
            interaction.user_text,
            aperture_packet,
            catalog,
            interaction.salience_assessment,
        )
        disposition = disposition.model_copy(update={"response_required": True})
        plan = registry.plan_execution(catalog, disposition.capability_indices)
        return {
            "attention_aperture_version": ATTENTION_APERTURE_VERSION,
            "percept": (
                interaction.percept.model_dump(mode="json")
                if interaction.percept is not None
                else None
            ),
            "salience_assessment": (
                interaction.salience_assessment.model_dump(mode="json")
                if interaction.salience_assessment is not None
                else None
            ),
            "aperture_packet": aperture_packet.model_dump(mode="json"),
            "disposition": disposition.model_dump(mode="json"),
            "capability_catalog": [item.model_dump(mode="json") for item in catalog],
            "execution_plan": plan.model_dump(mode="json"),
        }, [f"memory-request:{aperture_packet.memory_request_id}"]

    if stage is PerceptStage.EXECUTE_WORK:
        precognitive = _stage_result(conn, interaction, PerceptStage.PRECOGNITIVE, scheduler_key)
        plan = CapabilityExecutionPlan.model_validate(precognitive["execution_plan"])
        executions: list[CapabilityExecution] = []
        for position, item in enumerate(plan.items):
            registration = registry.get(item.capability_id)
            if registration.executor in _INTERNAL_MEMORY_EXECUTORS:
                raise RuntimeError("memory retrieval cannot execute as pre-cognitive work")
            execution = execute_registered_capability(
                conn,
                registration=registration,
                capability_execution_id=uuid5(
                    envelope.step.step_id, f"work:{position}:{item.capability_id}"
                ),
                requester_task_id=interaction.task_id,
                requester_step_id=envelope.step.step_id,
                plan_position=position,
                conversation_id=interaction.conversation_id,
                correlation_id=interaction.correlation_id,
                task_text=interaction.user_text,
                before_global_seq=interaction.before_global_seq,
                memory_request_id=deterministic_capability_memory_request_id(
                    interaction.interaction_id, position, item.capability_id
                ),
            )
            executions.append(execution)
        return {
            "executions": [item.model_dump(mode="json") for item in executions],
            "work_results": [_structured_execution(item) for item in executions],
        }, []

    if stage is PerceptStage.RETRIEVE_MEMORY:
        precognitive = _stage_result(conn, interaction, PerceptStage.PRECOGNITIVE, scheduler_key)
        evidence_policy = _stage_result(
            conn,
            interaction,
            PerceptStage.EVIDENCE_POLICY,
            scheduler_key,
        )
        disposition = PreCognitiveDisposition.model_validate(precognitive["disposition"])
        if not disposition.response_required:
            return {"skipped": True, "reason": "response_not_required"}, []
        initial_packet = MemoryPacket.model_validate(precognitive["aperture_packet"])
        response_policy = _validated_response_policy(evidence_policy)
        package = retrieve_memory_package(
            conn,
            interaction,
            initial_packet,
            source_types_for_scope(response_policy.evidence_scope),
        )
        return {"skipped": False, "memory_package": package.model_dump(mode="json")}, [
            f"memory-request:{package.memory_packet.memory_request_id}"
        ]

    if stage is PerceptStage.RESPOND:
        precognitive = _stage_result(conn, interaction, PerceptStage.PRECOGNITIVE, scheduler_key)
        evidence_policy = _stage_result(
            conn,
            interaction,
            PerceptStage.EVIDENCE_POLICY,
            scheduler_key,
        )
        disposition = PreCognitiveDisposition.model_validate(precognitive["disposition"])
        if not disposition.response_required:
            return {"response_required": False, "response_text": None, "skipped": True}, []
        compose = _stage_result(conn, interaction, PerceptStage.RETRIEVE_MEMORY, scheduler_key)
        package = ResponseMemoryPackage.model_validate(compose["memory_package"])
        work = _stage_result(conn, interaction, PerceptStage.EXECUTE_WORK, scheduler_key)
        response_text = llm.generate_final_response(
            interaction.user_text,
            package,
            tuple(work["work_results"]),
            response_policy=_validated_response_policy(evidence_policy),
        )
        return {"response_required": True, "response_text": response_text, "skipped": False}, []

    if stage is PerceptStage.PERSIST_RESULT:
        response = _stage_result(conn, interaction, PerceptStage.RESPOND, scheduler_key)
        response_required = bool(response["response_required"])
        response_text = response.get("response_text")
        output_refs: list[str] = []
        response_event_id: UUID | None = None
        if response_required:
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError("response-required percept produced no final response text")
            response_event = event_store.record_event(
                conn,
                conversation_id=interaction.conversation_id,
                correlation_id=interaction.correlation_id,
                event_type=EventType.INTERACTION_RESPONSE,
                source=SOURCE,
                payload={"text": response_text},
                payload_text=response_text,
                event_id=deterministic_interaction_event_id(interaction.interaction_id, "response"),
            )
            response_event_id = response_event.event_id
            output_refs.append(f"event:{response_event.event_id}")
        activated = [interaction.user_prompt_event_id]
        if response_event_id is not None:
            activated.insert(0, response_event_id)
        activate_working_state(
            conn,
            interaction_id=interaction.interaction_id,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            activated_event_ids=activated,
        )
        return {
            "response_required": response_required,
            "response_text": response_text,
            "response_event_id": str(response_event_id) if response_event_id else None,
        }, output_refs

    raise AssertionError(f"unhandled percept stage {stage.value}")


def finish_percept(
    conn: psycopg.Connection,
    interaction: DurableInteraction,
    *,
    probe: HostResourceProbe | None = None,
    policy: ResourceSafetyPolicy | None = None,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> str | None:
    persisted = _stage_result(conn, interaction, PerceptStage.PERSIST_RESULT, scheduler_key)
    response_text = persisted.get("response_text")
    scheduler = load_scheduler(conn, scheduler_key=scheduler_key)
    if scheduler.tasks[interaction.task_id].status.value != "COMPLETED":
        scheduler.complete_task(
            interaction.task_id,
            {
                "interaction_id": str(interaction.interaction_id),
                "response_required": bool(persisted["response_required"]),
                "response_text": response_text,
            },
        )
        observation = scheduler.current_resource_observation()
        effective_policy = (
            policy
            or (observation.policy.model_copy(deep=True) if observation is not None else None)
            or native_resource_safety_policy()
        )
        controller = LocalResourceAdmissionController(
            scheduler,
            probe=probe,
            policy=effective_policy,
            clock=clock,
        )
        controller.plan_scheduling_epoch()
        save_scheduler(conn, scheduler, scheduler_key=scheduler_key)
    return str(response_text) if response_text is not None else None


def handle_percept_in_worker_processes(
    conn: psycopg.Connection,
    user_text: str,
    conversation_id: UUID,
    *,
    probe: HostResourceProbe | None = None,
    policy: ResourceSafetyPolicy | None = None,
    ollama_runtime_probe: OllamaRuntimeProbe | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
    worker_lease_seconds: int | None = None,
    worker_timeout_seconds: int | None = None,
    progress: Callable[[str], None] | None = None,
    registry: CapabilityRegistry = DEFAULT_REGISTRY,
    worker_command: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Run every architectural stage in a separately guarded fresh process."""

    effective_lease = worker_lease_seconds or int(
        os.environ.get(
            "PCR_WORKER_LEASE_SECONDS",
            str(DEFAULT_PERCEPT_WORKER_LEASE_SECONDS),
        )
    )
    effective_timeout = worker_timeout_seconds or int(
        os.environ.get(
            "PCR_WORKER_TIMEOUT_SECONDS",
            str(DEFAULT_PERCEPT_WORKER_TIMEOUT_SECONDS),
        )
    )
    effective_policy = policy or native_resource_safety_policy()
    physical_probe = probe or SystemHostResourceProbe()
    if ollama_runtime_probe is None:
        from persistent_cognition.model_runtime import configured_runtime_probe
        ollama_runtime_probe = configured_runtime_probe()
    runtime_probe = ollama_runtime_probe or OllamaRuntimeProbe()
    admission_runtime_state = runtime_probe.capture()
    scheduled_memory_mib = admission_runtime_state.incremental_process_memory_mib(effective_policy)
    interaction = begin_percept(
        conn,
        user_text,
        conversation_id,
        probe=physical_probe,
        policy=effective_policy,
        ollama_runtime_state=admission_runtime_state,
        scheduler_key=scheduler_key,
    )
    claim_probe = OllamaClaimHostResourceProbe(
        base_probe=physical_probe,
        runtime_probe=runtime_probe,
        policy=effective_policy,
        scheduled_memory_mib=scheduled_memory_mib,
    )
    launcher = GuardedWorkerLauncher(
        db.get_connection,
        probe=claim_probe,
        policy=effective_policy,
        scheduler_key=scheduler_key,
    )
    for stage in PERCEPT_STAGES:
        if progress is not None:
            progress(stage.value)
        step_id = deterministic_worker_step_id(interaction.assignment_id, stage.value)
        worker_id = f"percept-v2-{interaction.interaction_id}-{stage.value.casefold()}"
        launched = launcher.launch(
            step_id=step_id,
            worker_id=worker_id,
            command=worker_command if worker_command is not None else
                    [sys.executable, "-m", "persistent_cognition.percept_response_worker"],
            env={**os.environ, "PCR_CAPABILITY_SNAPSHOT_SHA256": registry.snapshot()["sha256"]},
            lease_seconds=effective_lease,
        )
        try:
            return_code = launched.process.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            from persistent_cognition.reflexes import terminate_owned_worker
            terminate_owned_worker(conn, launched, scheduler_key=scheduler_key)
            raise RuntimeError(f"percept worker timed out at stage {stage.value}") from None
        if return_code != 0:
            raise RuntimeError(
                f"percept worker failed at stage {stage.value} with exit code {return_code}"
            )
    return finish_percept(
        conn,
        interaction,
        probe=physical_probe,
        policy=effective_policy,
        scheduler_key=scheduler_key,
    )
