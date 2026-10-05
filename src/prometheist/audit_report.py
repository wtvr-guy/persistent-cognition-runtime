"""Deterministic human-readable audit rendering over the artifact journal.

This module is a *projection*, not authoritative history. Every fact in a
rendered report already exists in ``artifact_journal``/``blob_store``; this
module only formats it for a person. Deleting a rendered report therefore
loses nothing, and regenerating it for an unchanged journal state must always
reproduce the same text.

    prometheist audit --interaction-id <uuid>
    prometheist audit --latest

The renderer must degrade gracefully rather than fail when it encounters a
record type it does not specifically know how to summarize, since new
journal record types are expected to be added over time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from prometheist import artifact_journal, blob_store

_PREVIEW_LIMIT = 96


def _preview(text: str | None, *, limit: int = _PREVIEW_LIMIT) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "\u2026"


def _parse_timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def _summarize_artifact(artifact: dict[str, Any]) -> str:
    artifact_type = artifact.get("artifact_type")
    payload = artifact.get("payload") or {}
    stage = artifact.get("stage")
    prefix = f"stage={stage} " if stage else ""

    if artifact_type == "PERCEPT":
        return f'user_text="{_preview(payload.get("user_text"))}"'
    if artifact_type == "STAGE_RESULT":
        refs = payload.get("output_refs") or []
        return f"{prefix}output_refs={len(refs)}"
    if artifact_type == "STAGE_ERROR":
        return (
            f"{prefix}error_type={payload.get('error_type')} "
            f'message="{_preview(payload.get("message"))}"'
        )
    if artifact_type == "LLM_INVOCATION":
        kind = payload.get("kind")
        model = payload.get("model")
        error_type = payload.get("error_type")
        if error_type:
            return f"{prefix}kind={kind} model={model} FAILED error_type={error_type}"
        output = payload.get("output")
        output_chars = len(output) if isinstance(output, str) else 0
        return f"{prefix}kind={kind} model={model} output_chars={output_chars}"
    if artifact_type == "FINAL_DISPOSITION":
        response_text = payload.get("response_text")
        suffix = f' response="{_preview(response_text)}"' if response_text else ""
        return (
            f"status={payload.get('status')} "
            f"last_completed_stage={payload.get('last_completed_stage')} "
            f"response_required={payload.get('response_required')}{suffix}"
        )
    # Forward-compatible fallback: an unrecognized future record type is still
    # listed in the timeline, just without a specialized one-line summary.
    return f"{prefix}(no summary renderer for artifact_type={artifact_type!r})"


def _referenced_blob_descriptors(
    artifacts: list[dict[str, Any]],
) -> list[blob_store.BlobDescriptor]:
    descriptors: list[blob_store.BlobDescriptor] = []
    seen: set[str] = set()
    for artifact in artifacts:
        for descriptor in blob_store.iter_blob_descriptors(artifact.get("payload")):
            if descriptor.digest in seen:
                continue
            seen.add(descriptor.digest)
            descriptors.append(descriptor)
    return descriptors


def render_interaction_audit(interaction_id: UUID) -> str:
    """Render a deterministic forensic timeline for one percept/interaction.

    Raises ``ValueError`` when no artifacts exist for ``interaction_id`` so a
    caller cannot mistake a missing interaction for a genuinely empty one.
    """

    artifacts = artifact_journal.interaction_artifacts(interaction_id)
    if not artifacts:
        raise ValueError(f"no artifacts found for interaction {interaction_id}")

    verification = artifact_journal.verify_interaction_chain(interaction_id)
    opened_at = _parse_timestamp(artifacts[0]["created_at"])
    final = next(
        (item for item in artifacts if item.get("artifact_type") == "FINAL_DISPOSITION"),
        None,
    )
    disposition = "INCOMPLETE"
    if final is not None:
        final_payload = final.get("payload") or {}
        disposition = (
            f"{final_payload.get('status')} ({final_payload.get('last_completed_stage')})"
        )

    descriptors = _referenced_blob_descriptors(artifacts)
    verified_blobs = sum(
        1 for descriptor in descriptors if blob_store.verify_descriptor(descriptor)
    )

    lines: list[str] = [
        "# Percept Audit",
        "",
        f"Interaction: {interaction_id}",
        f"Opened: {artifacts[0]['created_at']}",
        f"Finalized: {final['created_at'] if final else '(incomplete)'}",
        f"Disposition: {disposition}",
        "",
        "## Integrity",
        f"Journal records: {verification['artifact_count']}",
        f"Hash chain: {'PASS' if verification['valid'] else 'FAIL'}",
    ]
    for error in verification["errors"]:
        lines.append(f"  - {error}")
    if descriptors:
        lines.append(f"Referenced blobs: {verified_blobs}/{len(descriptors)} verified")
    lines.append("")
    lines.append("## Timeline")
    for artifact in artifacts:
        elapsed = (_parse_timestamp(artifact["created_at"]) - opened_at).total_seconds()
        sequence = artifact.get("journal_sequence")
        artifact_type = str(artifact.get("artifact_type"))
        summary = _summarize_artifact(artifact)
        lines.append(f"+{elapsed:8.3f}s  [{sequence:>3}] {artifact_type:<18} {summary}")

    return "\n".join(lines) + "\n"
