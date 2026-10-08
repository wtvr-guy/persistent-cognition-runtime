"""Frozen person-fidelity benchmark contracts and structural evaluation.

The benchmark deliberately separates two questions:

* did the response worker receive the exact canonical life evidence it needed?;
* did the resulting answer actually resemble the individual?

The first question is mechanically verifiable from immutable artifact provenance.
The second remains an independent human judgment. This module never converts natural
language resemblance into a keyword-matching oracle.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from persistent_cognition.models import EventType


BENCHMARK_ID = "PERSON-FIDELITY-001"
BENCHMARK_VERSION = "person-fidelity-public-baseline-v1"
HOLDOUT_BENCHMARK_ID = "PERSON-FIDELITY-002-HOLDOUT"
HOLDOUT_BENCHMARK_VERSION = "person-fidelity-holdout-v1"
BENCHMARK_NAMESPACE = UUID("505d839e-2e5a-5c19-870d-69d6a8f8913c")

# Each registered fixture pins its identity, version, and frozen role together so
# a holdout cannot be silently relabelled as the public baseline, or the reverse.
_REGISTERED_FIXTURES: dict[str, tuple[str, str]] = {
    BENCHMARK_ID: (BENCHMARK_VERSION, "FROZEN_BASELINE"),
    HOLDOUT_BENCHMARK_ID: (HOLDOUT_BENCHMARK_VERSION, "FROZEN_HOLDOUT"),
}


class FidelityDimension(str, Enum):
    AUTOBIOGRAPHICAL_MEANING = "AUTOBIOGRAPHICAL_MEANING"
    RELATIONAL_JUDGMENT = "RELATIONAL_JUDGMENT"
    TEMPORAL_BELIEF_CHANGE = "TEMPORAL_BELIEF_CHANGE"
    SELF_REPORT_BEHAVIOR_CONTRADICTION = "SELF_REPORT_BEHAVIOR_CONTRADICTION"
    CONTEXT_DEPENDENT_PREFERENCE = "CONTEXT_DEPENDENT_PREFERENCE"
    NOVEL_DECISION = "NOVEL_DECISION"
    CHARACTERISTIC_EXPRESSION = "CHARACTERISTIC_EXPRESSION"
    IDENTITY_INTEGRITY = "IDENTITY_INTEGRITY"
    LEGITIMATE_UNKNOWN = "LEGITIMATE_UNKNOWN"


REQUIRED_BASELINE_DIMENSIONS = frozenset(FidelityDimension)


class BenchmarkSubject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str
    name: str
    fictional: Literal[True]
    description: str


class LifeEventFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    conversation_id: str
    event_type: EventType
    source: str
    occurred_at: datetime
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)


class HumanOracle(BaseModel):
    """Evaluator-only semantic reference that must never enter system memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_outcome: str
    faithful_elements: tuple[str, ...]
    unfaithful_elements: tuple[str, ...]


class FidelityProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str
    dimension: FidelityDimension
    prompt: str
    required_event_ids: tuple[str, ...] = ()
    forbidden_event_ids: tuple[str, ...] = ()
    expect_no_seeded_evidence: bool = False
    human_oracle: HumanOracle

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> "FidelityProbe":
        required = set(self.required_event_ids)
        forbidden = set(self.forbidden_event_ids)
        if len(required) != len(self.required_event_ids):
            raise ValueError(f"{self.probe_id}: required_event_ids contains duplicates")
        if len(forbidden) != len(self.forbidden_event_ids):
            raise ValueError(f"{self.probe_id}: forbidden_event_ids contains duplicates")
        if required & forbidden:
            raise ValueError(f"{self.probe_id}: evidence cannot be required and forbidden")
        if self.expect_no_seeded_evidence and required:
            raise ValueError(
                f"{self.probe_id}: an unknown probe cannot require seeded evidence"
            )
        if not self.expect_no_seeded_evidence and not required:
            raise ValueError(f"{self.probe_id}: answerable probes require evidence")
        if self.expect_no_seeded_evidence != (
            self.dimension is FidelityDimension.LEGITIMATE_UNKNOWN
        ):
            raise ValueError(
                f"{self.probe_id}: only LEGITIMATE_UNKNOWN may be an open-world probe"
            )
        return self


class ReviewProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    separation_rule: str
    score_anchors: dict[str, str]
    aggregation_rule: str

    @model_validator(mode="after")
    def validate_score_anchors(self) -> "ReviewProtocol":
        if set(self.score_anchors) != {"0", "1", "2", "3", "4"}:
            raise ValueError("review protocol must define exactly the frozen 0-4 anchors")
        return self


class PersonFidelityCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    benchmark_id: str
    benchmark_version: str
    status: Literal["FROZEN_BASELINE", "FROZEN_HOLDOUT"]
    fixture_sha256: str
    subject: BenchmarkSubject
    life_events: tuple[LifeEventFixture, ...]
    probes: tuple[FidelityProbe, ...]
    review_protocol: ReviewProtocol

    @model_validator(mode="after")
    def validate_registered_identity(self) -> "PersonFidelityCorpus":
        registered = _REGISTERED_FIXTURES.get(self.benchmark_id)
        if registered is None:
            raise ValueError(f"unregistered person-fidelity benchmark: {self.benchmark_id}")
        expected_version, expected_status = registered
        if self.benchmark_version != expected_version:
            raise ValueError(
                f"{self.benchmark_id} must declare benchmark_version {expected_version!r}"
            )
        if self.status != expected_status:
            raise ValueError(
                f"{self.benchmark_id} must declare status {expected_status!r}"
            )
        return self

    @property
    def is_holdout(self) -> bool:
        return self.status == "FROZEN_HOLDOUT"

    @model_validator(mode="after")
    def validate_fixture_graph(self) -> "PersonFidelityCorpus":
        event_ids = [event.event_id for event in self.life_events]
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("life-event fixture IDs must be unique")
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("probe IDs must be unique")

        known_events = set(event_ids)
        for probe in self.probes:
            referenced = set(probe.required_event_ids) | set(probe.forbidden_event_ids)
            missing = referenced - known_events
            if missing:
                raise ValueError(
                    f"{probe.probe_id}: unknown evidence fixture IDs {sorted(missing)}"
                )

        represented = {probe.dimension for probe in self.probes}
        missing_dimensions = REQUIRED_BASELINE_DIMENSIONS - represented
        if missing_dimensions:
            names = sorted(dimension.value for dimension in missing_dimensions)
            raise ValueError(f"missing baseline fidelity dimensions: {names}")
        return self


class StructuralProbeResult(BaseModel):
    """Machine verdict about evidence delivery, never about identity resemblance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str
    response_nonempty: bool
    artifact_chain_valid: bool
    artifact_chain_complete: bool
    admitted_evidence_refs: tuple[str, ...]
    required_evidence_refs: tuple[str, ...]
    missing_required_refs: tuple[str, ...]
    forbidden_evidence_refs: tuple[str, ...]
    admitted_forbidden_refs: tuple[str, ...]
    unexpected_seeded_refs: tuple[str, ...]
    passed: bool


_RESPONSE_REALIZATION_KINDS = frozenset(
    {
        "FINAL_RESPONSE_V2",
        "V2_CURRENT_FALLBACK_SELECTION",
        "V2_EXACT_SOURCE_COMPOSITION",
        "V2_EXACT_SOURCE_SELECTION",
    }
)


def fixture_digest(document: dict[str, Any]) -> str:
    """Hash the complete fixture while excluding only its self-referential digest."""

    canonical = deepcopy(document)
    canonical.pop("fixture_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_person_fidelity_corpus(path: str | Path) -> PersonFidelityCorpus:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("person-fidelity corpus must be a JSON object")
    expected = str(raw.get("fixture_sha256", ""))
    actual = fixture_digest(raw)
    if expected != actual:
        raise ValueError(
            "person-fidelity fixture digest mismatch; publish a new benchmark version "
            f"or update the declared digest explicitly (expected={expected!r}, actual={actual})"
        )
    return PersonFidelityCorpus.model_validate(raw)


def deterministic_fixture_uuid(
    corpus: PersonFidelityCorpus,
    kind: Literal["event", "conversation", "correlation"],
    fixture_id: str,
) -> UUID:
    return uuid5(
        BENCHMARK_NAMESPACE,
        f"{corpus.benchmark_version}:{kind}:{fixture_id}",
    )


def chronological_life_events(
    corpus: PersonFidelityCorpus,
) -> tuple[LifeEventFixture, ...]:
    """Return deterministic event order without trusting JSON presentation order."""

    return tuple(
        sorted(
            corpus.life_events,
            key=lambda event: (event.occurred_at, event.event_id),
        )
    )


def canonical_seed_payload(
    corpus: PersonFidelityCorpus,
    event: LifeEventFixture,
) -> dict[str, Any]:
    """Return the only fixture data admitted to canonical memory.

    Probe prompts, reference outcomes, and human scoring criteria are intentionally
    unavailable to this function, making oracle leakage visible in code review.
    """

    return {
        **event.payload,
        "text": event.text,
        "benchmark_fixture": {
            "benchmark_id": corpus.benchmark_id,
            "benchmark_version": corpus.benchmark_version,
            "fixture_event_id": event.event_id,
            "occurred_at": event.occurred_at.isoformat(),
        },
    }
