"""Non-user operational triage. No retrieval, actions, prose, or scheduler powers."""
from __future__ import annotations

from persistent_cognition.prompt_registry import (
    TRIAGE_PROMPT,
)

from enum import Enum
from typing import Callable

from pydantic import Field, field_validator, model_validator

from persistent_cognition.percept_context import FrozenRecord, Reference
from persistent_cognition.perception import Percept, PerceptKind, PerceptModality
from persistent_cognition.response_policy import HistoricalEvidenceScope
from persistent_cognition.situations import Situation


class TaskClass(str, Enum):
    OBSERVE = "OBSERVE"
    RECONCILE = "RECONCILE"
    CONSOLIDATE = "CONSOLIDATE"


class UrgencyClass(str, Enum):
    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"


class SourcePolicy(FrozenRecord):
    """Installed by application configuration, never parsed from observation text."""
    source_id: Reference
    kind: PerceptKind
    modalities: tuple[PerceptModality, ...] = Field(min_length=1, max_length=10)
    response_required: bool = False
    natural_language_response: bool = False
    semantic_triage: bool = False
    deterministic_task: TaskClass | None = None
    allowed_task_classes: tuple[TaskClass, ...] = (TaskClass.OBSERVE, TaskClass.RECONCILE)
    evidence_domains: tuple[HistoricalEvidenceScope, ...] = (HistoricalEvidenceScope.DERIVED_INTERNAL,)
    maximum_urgency: UrgencyClass = UrgencyClass.ROUTINE
    trusted_executor: str | None = None
    trusted_executor_revision: str = Field(default="1", min_length=1, max_length=128)

    @field_validator("trusted_executor")
    @classmethod
    def validated_executor(cls, value: str | None) -> str | None:
        from persistent_cognition.trusted_executors import validate_executor_name
        return validate_executor_name(value) if value is not None else None

    @model_validator(mode="after")
    def coherent_policy(self) -> "SourcePolicy":
        if self.kind is PerceptKind.USER_INTERACTION:
            raise ValueError("explicit user prompts use the mandatory-response v2 pipeline")
        if self.deterministic_task and self.deterministic_task not in self.allowed_task_classes:
            raise ValueError("deterministic task must be allowed by its source policy")
        if TaskClass.CONSOLIDATE in self.allowed_task_classes and self.kind is not PerceptKind.SCHEDULED_EVENT:
            raise ValueError("consolidation must be a scheduled task")
        if len(set(self.evidence_domains)) != len(self.evidence_domains):
            raise ValueError("duplicate evidence domain")
        if self.trusted_executor is not None and (
                self.deterministic_task is None or self.deterministic_task is TaskClass.CONSOLIDATE):
            raise ValueError("trusted executors require an explicit non-consolidation deterministic task")
        return self


class TriageDecision(FrozenRecord):
    task_required: bool
    candidate_task_class: TaskClass | None
    evidence_domains: tuple[HistoricalEvidenceScope, ...] = Field(max_length=8)
    urgency_class: UrgencyClass

    @model_validator(mode="after")
    def coherent_decision(self) -> "TriageDecision":
        if self.task_required != (self.candidate_task_class is not None):
            raise ValueError("task class is present exactly when work is required")
        return self


def validate_triage(decision: TriageDecision, policy: SourcePolicy) -> TriageDecision:
    if decision.candidate_task_class and decision.candidate_task_class not in policy.allowed_task_classes:
        raise ValueError("triage proposed a task outside source policy")
    if not set(decision.evidence_domains).issubset(policy.evidence_domains):
        raise ValueError("triage broadened the committed evidence policy")
    if decision.urgency_class is UrgencyClass.ELEVATED and policy.maximum_urgency is UrgencyClass.ROUTINE:
        raise ValueError("triage exceeded the source urgency ceiling")
    return decision


def deterministic_triage(percept: Percept, situation: Situation, policy: SourcePolicy) -> TriageDecision | None:
    if percept.source.kind is PerceptKind.USER_INTERACTION:
        raise ValueError("user prompts must bypass non-user triage")
    if percept.source.kind is PerceptKind.ACTION_OUTCOME:
        own_errors = [error for error in situation.prediction_errors if error.percept_id == percept.percept_id]
        if own_errors and all(error.magnitude == 0 for error in own_errors):
            return TriageDecision(task_required=False, candidate_task_class=None,
                                  evidence_domains=policy.evidence_domains, urgency_class=UrgencyClass.ROUTINE)
    task_class = policy.deterministic_task
    if task_class is None and situation.prediction_error > 0:
        task_class = TaskClass.RECONCILE
    if task_class is None and (percept.context.active_goal_refs or percept.context.task_refs):
        task_class = TaskClass.OBSERVE
    # No opaque semantics remain when an adapter supplied structured observations.
    if task_class is None and not percept.context.observations and policy.semantic_triage:
        return None
    if task_class not in policy.allowed_task_classes:
        task_class = None
    decision = TriageDecision(
        task_required=task_class is not None, candidate_task_class=task_class,
        evidence_domains=policy.evidence_domains, urgency_class=policy.maximum_urgency,
    )
    return validate_triage(decision, policy)




def semantic_triage(
    policy: SourcePolicy, evidence: str, infer: Callable[[str, str, dict], str],
) -> TriageDecision:
    if not policy.semantic_triage:
        raise ValueError("semantic triage is not enabled")
    result = infer(TRIAGE_PROMPT + "\nApplication policy:\n" + policy.model_dump_json(),
                   evidence, TriageDecision.model_json_schema())
    return validate_triage(TriageDecision.model_validate_json(result), policy)
