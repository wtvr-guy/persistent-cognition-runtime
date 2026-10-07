"""Artifact-first structural oracles for native model acceptance.

Native prose is printed for human review. These helpers establish the machine-
verifiable part of the claim: a fresh response worker received the required
canonical evidence through an immutable, valid invocation-artifact chain.
"""
from __future__ import annotations

from collections.abc import Iterable
import json
from uuid import UUID

from persistent_cognition import artifact_journal
from persistent_cognition.interaction_contracts import deterministic_interaction_id
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


def assert_recalled_literal(answer: str, expected: str, *, label: str) -> None:
    """Check a test-generated opaque value without grading natural prose.

    Provenance receipts establish delivery, not successful recall. A known
    randomized identifier must survive in the answer; explanations and other
    semantic judgments still require human review. This is a test oracle only.
    """
    import re

    assert expected, "the recall oracle requires a nonempty expected identifier"
    assert re.search(rf"(?<![\w-]){re.escape(expected)}(?![\w-])", answer), (
        f"{label}: answer did not reproduce the stored identifier {expected!r}. "
        f"Actual answer: {answer!r}"
    )


def assert_response_evidence_receipt(
    *,
    interaction_id: UUID,
    required_event_ids: Iterable[UUID] = (),
    forbidden_event_ids: Iterable[UUID] = (),
    require_complete: bool = True,
    require_deterministic: bool = False,
) -> dict[str, object]:
    """Verify the successful response realization's canonical evidence links."""

    verification = artifact_journal.verify_interaction_chain(interaction_id)
    assert verification["valid"] is True, verification
    if require_complete:
        assert verification["complete"] is True, verification

    artifacts = artifact_journal.interaction_artifacts(interaction_id)
    invocations = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "LLM_INVOCATION"
        and artifact.get("stage") == "V2_RESPOND"
        and artifact.get("payload", {}).get("kind") in _RESPONSE_REALIZATION_KINDS
        and artifact.get("payload", {}).get("error_type") is None
        and artifact.get("payload", {}).get("output") is not None
    ]
    renders = [artifact for artifact in artifacts
               if artifact.get("artifact_type") == "RESPONSE_RENDER"
               and artifact.get("stage") == "V2_RESPOND"]
    assert invocations or renders, (
        "No successful response-realization artifact was recorded for "
        f"interaction {interaction_id}"
    )
    if renders:
        # A deterministic response must not hide a semantic evidence-selection
        # call in the same stage. Replay the immutable inputs against the renderer.
        assert not [artifact for artifact in artifacts
                    if artifact.get("artifact_type") == "LLM_INVOCATION"
                    and artifact.get("stage") == "V2_RESPOND"]
        realization = renders[-1]
        _assert_deterministic_render(realization["payload"])
    else:
        assert not require_deterministic, (
            "Factual evidence recall unexpectedly invoked a response model"
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
    if not renders:
        assert isinstance(payload.get("evidence_prompt"), str)

    return {
        "artifact_chain_valid": True,
        "artifact_id": realization["artifact_id"],
        "evidence_refs": sorted(evidence_refs),
        "interaction_id": str(interaction_id),
        "model": payload.get("model"),
        "response_kind": payload["kind"],
        "transport_layout": ("application:canonical-evidence" if renders
                             else payload.get("transport_layout")),
    }


def _assert_deterministic_render(payload: dict) -> None:
    from persistent_cognition.response_policy import (
        ExactSourceComposition, ExactSourceSelection, ResponsePolicy, ResponseSurfaceMode,
        validate_exact_source_composition,
    )
    from persistent_cognition.source_value_response import (
        EVIDENCE_RENDERER_VERSION, ResponseEvidenceSource, SourceValueBinding,
        render_source_evidence, render_value_response, resolve_bound_values,
    )

    assert payload["renderer_version"] == EVIDENCE_RENDERER_VERSION
    sources = tuple(ResponseEvidenceSource(**source) for source in payload["sources"])
    assert {source.source_ref for source in sources if source.source_ref.startswith("event:")} == (
        set(payload["evidence_refs"])
    )
    bindings = tuple(SourceValueBinding.model_validate_json(json.dumps(binding))
                     for binding in payload["bindings"])
    if bindings:
        assert payload["kind"] == "DETERMINISTIC_SOURCE_VALUES"
        values = resolve_bound_values(sources, bindings)
        policy = ResponsePolicy.model_validate(payload["response_policy"])
        if policy.surface_mode is ResponseSurfaceMode.EXACT_SOURCE_SUBSTRING:
            assert len(values) == 1
            expected = values[0]
        elif policy.surface_mode is ResponseSurfaceMode.EXACT_SOURCE_COMPOSITION:
            expected = validate_exact_source_composition(
                payload["current_user_prompt"], values, ExactSourceComposition(
                    selections=[ExactSourceSelection(source_index=index, verbatim_value=value)
                                for index, value in enumerate(values)],
                    separator=payload["separator"],
                ),
            )
        else:
            expected = render_value_response(values)
    else:
        assert payload["kind"] == "DETERMINISTIC_EVIDENCE_DISPLAY"
        expected = render_source_evidence(sources)
    assert payload["output"] == expected


def print_artifact_receipt(label: str, receipt: dict[str, object]) -> None:
    print_transcript(
        f"\n{label} — structural artifact receipt:\n"
        + json.dumps(receipt, indent=2, sort_keys=True)
    )
