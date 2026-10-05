"""Artifact-first structural oracles for native model acceptance.

Native prose is printed for human review. These helpers establish the machine-
verifiable part of the claim: a fresh response worker received the required
canonical evidence through an immutable, valid invocation-artifact chain.
"""
from __future__ import annotations

from collections.abc import Iterable
import json
from uuid import UUID

from prometheist import artifact_journal
from prometheist.interaction_contracts import deterministic_interaction_id
from tests._cli_helpers import print_transcript


_RESPONSE_REALIZATION_KINDS = frozenset(
    {
        "FINAL_RESPONSE_V2",
        "V2_CURRENT_FALLBACK_SELECTION",
        "V2_EXACT_SOURCE_COMPOSITION",
        "V2_EXACT_SOURCE_SELECTION",
    }
)


def interaction_id_for_prompt(conversation_id: UUID, correlation_id: UUID) -> UUID:
    return deterministic_interaction_id(conversation_id, correlation_id)


def assert_response_evidence_receipt(
    *,
    interaction_id: UUID,
    required_event_ids: Iterable[UUID] = (),
    forbidden_event_ids: Iterable[UUID] = (),
    require_complete: bool = True,
) -> dict[str, object]:
    """Verify the successful response realization's canonical evidence links."""

    verification = artifact_journal.verify_interaction_chain(interaction_id)
    assert verification["valid"] is True, verification
    if require_complete:
        assert verification["complete"] is True, verification

    invocations = [
        artifact
        for artifact in artifact_journal.interaction_artifacts(interaction_id)
        if artifact.get("artifact_type") == "LLM_INVOCATION"
        and artifact.get("stage") == "V2_RESPOND"
        and artifact.get("payload", {}).get("kind") in _RESPONSE_REALIZATION_KINDS
        and artifact.get("payload", {}).get("error_type") is None
        and artifact.get("payload", {}).get("output") is not None
    ]
    assert invocations, (
        "No successful response-realization invocation artifact was recorded for "
        f"interaction {interaction_id}"
    )
    realization = invocations[-1]
    payload = realization["payload"]
    evidence_refs = set(payload.get("evidence_refs", []))
    required_refs = {f"event:{event_id}" for event_id in required_event_ids}
    forbidden_refs = {f"event:{event_id}" for event_id in forbidden_event_ids}
    assert required_refs <= evidence_refs, {
        "missing_required_refs": sorted(required_refs - evidence_refs),
        "recorded_evidence_refs": sorted(evidence_refs),
    }
    assert evidence_refs.isdisjoint(forbidden_refs), {
        "forbidden_refs_admitted": sorted(evidence_refs & forbidden_refs),
        "recorded_evidence_refs": sorted(evidence_refs),
    }
    assert isinstance(payload.get("evidence_prompt"), str)

    return {
        "artifact_chain_valid": True,
        "artifact_id": realization["artifact_id"],
        "evidence_refs": sorted(evidence_refs),
        "interaction_id": str(interaction_id),
        "model": payload["model"],
        "response_kind": payload["kind"],
        "transport_layout": payload.get("transport_layout"),
    }


def print_artifact_receipt(label: str, receipt: dict[str, object]) -> None:
    print_transcript(
        f"\n{label} — structural artifact receipt:\n"
        + json.dumps(receipt, indent=2, sort_keys=True)
    )
