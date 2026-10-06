"""Fresh-process entry point for one v2 user-prompt response worker step.

Explicit user prompts always receive a response. Every stage result is atomically
written to the independent artifact journal before its database worker claim is
completed. A replacement worker can therefore rehydrate a completed stage from
JSON instead of repeating an LLM/tool call after interruption.

Every v2 LLM invocation also persists the exact stateless request contract and
normalized constrained output so later inspection can establish precisely what a
model worker was given rather than inferring it from downstream behavior.
"""
from __future__ import annotations

from persistent_cognition.diagnostics import exception_summary

from persistent_cognition.contract_registry import SEMANTIC_CONTRACTS

from persistent_cognition.contract_registry import STAGE_CONTRACTS

from persistent_cognition.prompt_registry import (
    _USER_PROMPT_WORK_SELECTION,
)

from enum import Enum
import hashlib
from itertools import count
import os
import sys
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from persistent_cognition import artifact_journal, db, event_store, llm_artifact_store
from persistent_cognition.capability_registry import DEFAULT_REGISTRY, CapabilityDescriptor
from persistent_cognition.epistemic_authority import format_authority_bound_memory_packet
from persistent_cognition.interaction_store import load_interaction_by_task
from persistent_cognition.models import EventType, MemoryPacket
from persistent_cognition.perception import SalienceAssessment, format_salience_context
from persistent_cognition.model_evidence_budget import (
    configured_model_evidence_budget,
    validate_memory_packet_content,
    validate_rendered_evidence,
)
from persistent_cognition.llm import _evidence_transport_layout, _quarantined_evidence
from persistent_cognition.percept_response_runtime import (
    PerceptSpecialists,
    PerceptStage,
    PreCognitiveDisposition,
    ResponseMemoryPackage,
    _execute_stage,
    _memory_evidence_refs,
    _stage_result,
)
from persistent_cognition.response_policy import ResponsePolicy
from persistent_cognition.worker_store import (
    complete_worker_claim,
    load_worker_claim_envelope,
    release_worker_claim,
)

_NON_COGNITIVE_MEMORY_SOURCES = frozenset(
    {
        "percept_response_v2/response_input_trace",
    }
)


USER_PROMPT_STAGE_SPECIALIST_ROLES = {
    stage: STAGE_CONTRACTS[stage.value].role for stage in PerceptStage
}
_ALLOWED_LLM_KINDS_BY_STAGE = {
    stage: STAGE_CONTRACTS[stage.value].kinds for stage in PerceptStage
}


class UserPromptWorkSelection(BaseModel):
    """The only model-authored pre-cognitive control output for a user prompt."""

    model_config = ConfigDict(extra="forbid")
    capability_indices: list[int] = Field(default_factory=list)

    @field_validator("capability_indices")
    @classmethod
    def validate_indices(cls, values: list[int]) -> list[int]:
        if any(index < 0 for index in values):
            raise ValueError("capability_indices must be non-negative")
        if len(values) != len(set(values)):
            raise ValueError("capability_indices must not contain duplicates")
        return values


def _cognitive_memory_packet(packet: MemoryPacket) -> MemoryPacket:
    """Remove audit-only artifacts from model-visible semantic memory.

    The independent artifact journal remains the complete audit/recovery record.
    Historical response-input trace SYSTEM_EVENTs created before that separation
    are durable, but must never recursively re-enter a later LLM context.
    """

    kept = [
        item.model_copy(deep=True)
        for item in packet.items
        if item.source not in _NON_COGNITIVE_MEMORY_SOURCES
    ]
    if len(kept) == len(packet.items):
        return packet.model_copy(deep=True)
    trace = dict(packet.retrieval_trace)
    trace["cognitive_visibility_filter"] = {
        "excluded_sources": sorted(_NON_COGNITIVE_MEMORY_SOURCES),
        "excluded_item_count": len(packet.items) - len(kept),
    }
    return packet.model_copy(
        update={
            "items": kept,
            "supported": bool(kept),
            "retrieval_trace": trace,
        },
        deep=True,
    )


class UserPromptLLM(PerceptSpecialists):
    """Artifact-aware transport constrained to one stage-specialist role."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        interaction=None,
        # A stage label is any worker-stage enum. The user pipeline supplies
        # PerceptStage; the v0.8 situation pipeline supplies SituationStage and
        # enforces its own allowlist in a subclass.
        stage: Enum | None = None,
        claim_id: UUID | None = None,
    ) -> None:
        from persistent_cognition.runtime_settings import job_settings
        configured = job_settings()
        if configured is None:
            super().__init__(base_url=base_url, model=model)
        else:
            super().__init__(base_url=base_url, model=model,
                             selection=configured.selection_for(stage.value if stage else None))
        self._artifact_interaction = interaction
        self._artifact_stage = stage
        self._artifact_claim_id = claim_id
        self._artifact_invocations = count()
        self._artifact_evidence_refs: tuple[str, ...] = ()
        self._artifact_invocation_records: dict[int, dict[str, Any]] = {}
        self._pending_validation: tuple[int, str] | None = None

    def _require_stage_specialization(self, kind: str) -> None:
        """Fail closed if one guarded process attempts another specialist's role."""

        stage = self._artifact_stage
        if stage is None:
            return
        if not isinstance(stage, PerceptStage):
            raise RuntimeError(f"{stage} is not a user-prompt stage specialist")
        if kind not in _ALLOWED_LLM_KINDS_BY_STAGE[stage]:
            role = USER_PROMPT_STAGE_SPECIALIST_ROLES[stage]
            raise RuntimeError(
                f"{stage.value} ({role}) cannot invoke LLM role {kind}"
            )

    def _set_artifact_evidence_refs(self, refs: tuple[str, ...]) -> None:
        self._artifact_evidence_refs = tuple(refs)

    def _structured(
        self,
        kind: str,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
    ) -> str:
        self._require_stage_specialization(kind)
        if self._pending_validation is not None:
            raise RuntimeError("prior LLM invocation has no resolved validation outcome")
        invocation_index = next(self._artifact_invocations)
        try:
            output = super()._structured(
                kind,
                system,
                user,
                schema,
                max_tokens,
            )
        except Exception as exc:
            transport_diagnostics = self._consume_invocation_diagnostics()
            validation_status = (
                "TRANSPORT_ERROR"
                if not isinstance(transport_diagnostics, dict)
                or transport_diagnostics.get("transport_error_type") is not None
                else "INVALID"
            )
            invocation = self._journal_llm_invocation(
                invocation_index=invocation_index,
                kind=kind,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
                output=None,
                error=exc,
                evidence=None,
                transport_diagnostics=transport_diagnostics,
            )
            self._journal_llm_validation(
                invocation_index=invocation_index,
                kind=kind,
                invocation=invocation,
                raw_output=None,
                parsed_output=None,
                error=exc,
                status=validation_status,
            )
            raise
        invocation = self._journal_llm_invocation(
            invocation_index=invocation_index,
            kind=kind,
            system=system,
            user=user,
            schema=schema,
            max_tokens=max_tokens,
            output=output,
            error=None,
            evidence=None,
            transport_diagnostics=self._consume_invocation_diagnostics(),
        )
        if invocation is not None:
            self._pending_validation = (invocation_index, kind)
        return output

    def _structured_with_evidence(
        self,
        kind: str,
        system: str,
        current_user: str,
        evidence: str,
        schema: dict,
        max_tokens: int,
    ) -> str:
        self._require_stage_specialization(kind)
        if self._pending_validation is not None:
            raise RuntimeError("prior LLM invocation has no resolved validation outcome")
        invocation_index = next(self._artifact_invocations)
        try:
            output = super()._structured_with_evidence(
                kind,
                system,
                current_user,
                evidence,
                schema,
                max_tokens,
            )
        except Exception as exc:
            transport_diagnostics = self._consume_invocation_diagnostics()
            validation_status = (
                "TRANSPORT_ERROR"
                if not isinstance(transport_diagnostics, dict)
                or transport_diagnostics.get("transport_error_type") is not None
                else "INVALID"
            )
            invocation = self._journal_llm_invocation(
                invocation_index=invocation_index,
                kind=kind,
                system=system,
                user=current_user,
                schema=schema,
                max_tokens=max_tokens,
                output=None,
                error=exc,
                evidence=evidence,
                transport_diagnostics=transport_diagnostics,
            )
            self._journal_llm_validation(
                invocation_index=invocation_index,
                kind=kind,
                invocation=invocation,
                raw_output=None,
                parsed_output=None,
                error=exc,
                status=validation_status,
            )
            raise
        invocation = self._journal_llm_invocation(
            invocation_index=invocation_index,
            kind=kind,
            system=system,
            user=current_user,
            schema=schema,
            max_tokens=max_tokens,
            output=output,
            error=None,
            evidence=evidence,
            transport_diagnostics=self._consume_invocation_diagnostics(),
        )
        if invocation is not None:
            self._pending_validation = (invocation_index, kind)
        return output

    def _journal_llm_invocation(
        self,
        *,
        invocation_index: int,
        kind: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        output: str | None,
        error: Exception | None,
        evidence: str | None,
        transport_diagnostics: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        interaction = self._artifact_interaction
        stage = self._artifact_stage
        claim_id = self._artifact_claim_id
        if interaction is None or stage is None or claim_id is None:
            return None
        artifact = llm_artifact_store.write_llm_invocation(
            interaction_id=interaction.interaction_id,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            task_id=interaction.task_id,
            assignment_id=interaction.assignment_id,
            stage=stage.value,
            claim_id=claim_id,
            invocation_index=invocation_index,
            kind=kind,
            model=self.model,
            base_url=self.base_url,
            system_prompt=system,
            user_prompt=user,
            schema=schema,
            max_tokens=max_tokens,
            temperature=self.temperature_for_kind(kind),
            output=output,
            error_type=type(error).__name__ if error is not None else None,
            error_message=exception_summary(error) if error is not None else None,
            evidence_prompt=evidence,
            transport_layout=(
                ("responses:instructions,evidence,current-user" if getattr(self, "selection", None) is not None and self.selection.provider == "openai"
                 else _evidence_transport_layout(self.model)) if evidence is not None else None
            ),
            evidence_refs=self._artifact_evidence_refs,
            transport_diagnostics=transport_diagnostics,
            provider=self.selection.provider if getattr(self, "selection", None) is not None else "ollama",
        )
        self._artifact_invocation_records[invocation_index] = artifact
        return artifact

    def _journal_llm_validation(
        self,
        *,
        invocation_index: int,
        kind: str,
        invocation: dict[str, Any] | None,
        raw_output: str | None,
        parsed_output: Any,
        error: Exception | None,
        status: str,
    ) -> None:
        interaction = self._artifact_interaction
        stage = self._artifact_stage
        claim_id = self._artifact_claim_id
        if interaction is None or stage is None or claim_id is None or invocation is None:
            return
        llm_artifact_store.write_llm_validation(
            interaction_id=interaction.interaction_id,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            task_id=interaction.task_id,
            assignment_id=interaction.assignment_id,
            stage=stage.value,
            claim_id=claim_id,
            invocation_index=invocation_index,
            kind=kind,
            invocation_artifact_id=str(invocation["artifact_id"]),
            invocation_artifact_hash=str(invocation["artifact_hash"]),
            status=status,
            raw_output_sha256=(
                hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
                if raw_output is not None
                else None
            ),
            parsed_output=parsed_output,
            error_type=type(error).__name__ if error is not None else None,
            error_message=exception_summary(error) if error is not None else None,
        )

    def _record_validation_outcome(
        self,
        *,
        kind: str,
        raw_output: str | None,
        parsed_output: Any,
        error: Exception | None,
        status: str,
    ) -> None:
        pending = self._pending_validation
        if pending is None:
            return
        invocation_index, expected_kind = pending
        if kind != expected_kind:
            raise RuntimeError(
                f"validation kind {kind} does not match pending invocation {expected_kind}"
            )
        invocation = self._artifact_invocation_records[invocation_index]
        if status != "VALID":
            self._journal_llm_validation(
                invocation_index=invocation_index,
                kind=kind,
                invocation=invocation,
                raw_output=raw_output,
                parsed_output=parsed_output,
                error=error,
                status=status,
            )
        self._pending_validation = None

    def decide_disposition(
        self,
        percept: str,
        memory_packet: MemoryPacket,
        capability_catalog: tuple[CapabilityDescriptor, ...],
        salience_assessment: SalienceAssessment | None = None,
    ) -> PreCognitiveDisposition:
        # There is no semantic decision to delegate when no external action is
        # executable. Calling a small model here only creates an opportunity for
        # prompt contamination to manufacture impossible capability selections.
        if not capability_catalog:
            return PreCognitiveDisposition(response_required=True, capability_indices=[])

        visible_packet = _cognitive_memory_packet(memory_packet)
        self._set_artifact_evidence_refs(_memory_evidence_refs(visible_packet))
        budget = configured_model_evidence_budget()
        validate_memory_packet_content(visible_packet, budget=budget)
        catalog_text = "\n".join(
            f"{index}: {item.capability_id} | {item.kind.value} | {item.description}"
            for index, item in enumerate(capability_catalog)
        )
        memory_text = format_authority_bound_memory_packet(visible_packet)
        if salience_assessment is not None:
            memory_text += "\n\n[Advisory deterministic salience]\n" + format_salience_context(
                salience_assessment
            )
        validate_rendered_evidence((memory_text,), budget=budget)
        current_user = (
            f"[Current user prompt]\n{percept}\n\n"
            f"[Executable capability catalog]\n{catalog_text}"
        )

        def validate_selection(content: str) -> UserPromptWorkSelection:
            selection = UserPromptWorkSelection.model_validate_json(content)
            if any(index >= len(capability_catalog) for index in selection.capability_indices):
                raise ValueError("pre-cognitive worker selected an unavailable capability")
            return selection

        last_error: ValueError | None = None
        for token_cap in (48, 96):
            try:
                content = self._structured_with_evidence(
                    "PRECOGNITIVE_USER_PROMPT_WORK",
                    _USER_PROMPT_WORK_SELECTION,
                    current_user,
                    _quarantined_evidence(memory_text),
                    SEMANTIC_CONTRACTS["PRECOGNITIVE_USER_PROMPT_WORK"].output_schema(),
                    token_cap,
                )
                selection = self._validated_model_output(
                    kind="PRECOGNITIVE_USER_PROMPT_WORK",
                    raw_output=content,
                    validator=lambda: validate_selection(content),
                )
                return PreCognitiveDisposition(
                    response_required=True,
                    capability_indices=selection.capability_indices,
                )
            except ValueError as exc:
                last_error = exc
        raise ValueError(f"pre-cognitive work selection failed to validate: {last_error}")

    def generate_final_response(
        self,
        percept: str,
        package: ResponseMemoryPackage,
        work_results: tuple[dict[str, Any], ...],
        *,
        response_policy: ResponsePolicy,
    ) -> str:
        visible_package = package.model_copy(
            update={"memory_packet": _cognitive_memory_packet(package.memory_packet)},
            deep=True,
        )

        return super().generate_final_response(
            percept,
            visible_package,
            work_results,
            response_policy=response_policy,
        )


def _ensure_percept_artifact(interaction) -> None:
    artifact_journal.write_percept_artifact(
        interaction_id=interaction.interaction_id,
        conversation_id=interaction.conversation_id,
        correlation_id=interaction.correlation_id,
        task_id=interaction.task_id,
        user_text=interaction.user_text,
        user_prompt_event_id=interaction.user_prompt_event_id,
        percept=interaction.percept,
        salience_assessment=interaction.salience_assessment,
    )


def _execute_claimed_user_prompt_step(
    conn,
    llm: PerceptSpecialists,
    *,
    claim_id: UUID,
    worker_id: str,
    scheduler_key: str,
    registry=DEFAULT_REGISTRY,
) -> PerceptStage:
    """Execute or rehydrate one stage with artifact-before-terminal ordering."""

    envelope = load_worker_claim_envelope(
        conn,
        claim_id,
        worker_id=worker_id,
        scheduler_key=scheduler_key,
    )
    interaction = load_interaction_by_task(
        conn,
        envelope.step.task_id,
        scheduler_key=scheduler_key,
    )
    stage = PerceptStage(envelope.step.step_key)
    _ensure_percept_artifact(interaction)
    try:
        recovered = artifact_journal.load_stage_result_artifact(
            interaction.interaction_id,
            stage.value,
        )
        if recovered is None:
            output, output_refs = _execute_stage(
                conn,
                llm,
                envelope,
                interaction,
                scheduler_key=scheduler_key,
                registry=registry,
            )
            artifact_journal.write_stage_result_artifact(
                interaction_id=interaction.interaction_id,
                conversation_id=interaction.conversation_id,
                correlation_id=interaction.correlation_id,
                task_id=interaction.task_id,
                assignment_id=interaction.assignment_id,
                stage=stage.value,
                output=output,
                output_refs=output_refs,
            )
        else:
            output = dict(recovered["output"])
            output_refs = list(recovered.get("output_refs", []))

        complete_worker_claim(
            conn,
            claim_id=claim_id,
            worker_id=worker_id,
            output=output,
            output_refs=output_refs,
            scheduler_key=scheduler_key,
        )
        return stage
    except Exception as exc:
        artifact_journal.write_stage_error_artifact(
            interaction_id=interaction.interaction_id,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            task_id=interaction.task_id,
            assignment_id=interaction.assignment_id,
            stage=stage.value,
            claim_id=claim_id,
            error_type=type(exc).__name__,
            message=exception_summary(exc),
        )
        event_store.record_event(
            conn,
            conversation_id=interaction.conversation_id,
            correlation_id=interaction.correlation_id,
            event_type=EventType.ERROR,
            source="percept_response_v2",
            payload={
                "stage": stage.value,
                "error_type": type(exc).__name__,
                "message": exception_summary(exc),
            },
            event_id=uuid5(claim_id, "error-event"),
        )
        release_worker_claim(
            conn,
            claim_id=claim_id,
            worker_id=worker_id,
            scheduler_key=scheduler_key,
        )
        raise


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing guarded worker environment value: {name}")
    return value


def main(*, registry=DEFAULT_REGISTRY) -> None:
    _configure_utf8_streams()
    expected = os.environ.get("PCR_CAPABILITY_SNAPSHOT_SHA256")
    if expected is not None and expected != registry.snapshot()["sha256"]:
        raise RuntimeError("worker capability bootstrap differs from the parent registration snapshot")
    claim_id = UUID(_required_environment("PCR_WORKER_CLAIM_ID"))
    worker_id = _required_environment("PCR_WORKER_ID")
    scheduler_key = _required_environment("PCR_WORKER_SCHEDULER_KEY")
    with db.get_connection() as conn:
        envelope = load_worker_claim_envelope(
            conn,
            claim_id,
            worker_id=worker_id,
            scheduler_key=scheduler_key,
        )
        interaction = load_interaction_by_task(
            conn,
            envelope.step.task_id,
            scheduler_key=scheduler_key,
        )
        stage = PerceptStage(envelope.step.step_key)
        stage = _execute_claimed_user_prompt_step(
            conn,
            UserPromptLLM(
                interaction=interaction,
                stage=stage,
                claim_id=claim_id,
            ),
            claim_id=claim_id,
            worker_id=worker_id,
            scheduler_key=scheduler_key,
            registry=registry,
        )
        if stage is PerceptStage.PERSIST_RESULT:
            persisted = _stage_result(
                conn,
                interaction,
                PerceptStage.PERSIST_RESULT,
                scheduler_key,
            )
            artifact_journal.write_final_disposition_artifact(
                interaction_id=interaction.interaction_id,
                conversation_id=interaction.conversation_id,
                correlation_id=interaction.correlation_id,
                task_id=interaction.task_id,
                assignment_id=interaction.assignment_id,
                response_required=bool(persisted["response_required"]),
                response_text=(
                    str(persisted["response_text"])
                    if persisted.get("response_text") is not None
                    else None
                ),
            )


if __name__ == "__main__":
    from persistent_cognition.diagnostics import run_command
    run_command(main)
