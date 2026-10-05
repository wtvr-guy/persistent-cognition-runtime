"""Immutable interaction artifacts for exact stateless LLM invocation envelopes."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal
from uuid import UUID

from persistent_cognition import artifact_journal


def write_llm_invocation(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    assignment_id: UUID,
    stage: str,
    claim_id: UUID,
    invocation_index: int,
    kind: str,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    max_tokens: int,
    temperature: float | None,
    output: str | None,
    error_type: str | None,
    error_message: str | None,
    evidence_prompt: str | None = None,
    transport_layout: str | None = None,
    evidence_refs: Iterable[str] = (),
    transport_diagnostics: dict[str, Any] | None = None,
    provider: Literal["ollama", "openai"] = "ollama",
) -> dict[str, Any]:
    """Persist the exact request contract and resulting normalized model output."""

    payload: dict[str, Any] = {
        "provider": provider,
        "claim_id": str(claim_id),
        "invocation_index": invocation_index,
        "kind": kind,
        "model": model,
        "base_url": base_url,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "schema": schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "output": output,
        "error_type": error_type,
        "error_message": error_message,
    }
    from persistent_cognition.contract_registry import contract_manifest
    registered = contract_manifest()["semantic_contracts"].get(kind)
    if registered is not None:
        payload["contract_identity"] = registered
    resolved_evidence_refs = list(evidence_refs)
    if resolved_evidence_refs:
        payload["evidence_refs"] = resolved_evidence_refs
    if evidence_prompt is not None:
        payload["evidence_prompt"] = evidence_prompt
    if transport_layout is not None:
        payload["transport_layout"] = transport_layout
    if transport_diagnostics is not None:
        payload["transport_diagnostics"] = transport_diagnostics

    return artifact_journal.write_interaction_artifact(
        artifact_key=(
            f"llm-invocation:{stage}:{claim_id}:{invocation_index}:{kind}"
        ),
        artifact_type="LLM_INVOCATION",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage=stage,
        producer=f"percept_response_v2/{provider}",
        payload=payload,
    )


def write_llm_validation(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    assignment_id: UUID,
    stage: str,
    claim_id: UUID,
    invocation_index: int,
    kind: str,
    invocation_artifact_id: str,
    invocation_artifact_hash: str,
    status: str,
    raw_output_sha256: str | None,
    parsed_output: Any,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any] | None:
    """Persist exceptional outcomes; a committed stage result proves routine success.

    The invocation still holds the exact input and raw output. An invocation with
    no validation or stage result is an incomplete attempt, never an accepted one.
    """

    if status == "VALID":
        return None
    if status not in {"INVALID", "TRANSPORT_ERROR"}:
        raise ValueError(f"unknown LLM validation status: {status}")

    return artifact_journal.write_interaction_artifact(
        artifact_key=f"llm-validation:{stage}:{claim_id}:{invocation_index}:{kind}",
        artifact_type="LLM_VALIDATION",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage=stage,
        producer="percept_response_v2/validation",
        payload={
            "claim_id": str(claim_id),
            "invocation_index": invocation_index,
            "kind": kind,
            "invocation_artifact_id": invocation_artifact_id,
            "invocation_artifact_hash": invocation_artifact_hash,
            "status": status,
            "raw_output_sha256": raw_output_sha256,
            "parsed_output": parsed_output,
            "error_type": error_type,
            "error_message": error_message,
        },
    )
