"""One claimed situation stage per fresh process, with exact artifact handoffs."""
from __future__ import annotations

from prometheist.contract_registry import SEMANTIC_CONTRACTS

from prometheist.contract_registry import STAGE_CONTRACTS

from datetime import datetime
import json
import os
from uuid import UUID, uuid5

from prometheist import artifact_journal, db, event_store, jit_memory
from prometheist.action_outcomes import issue_action, observe_action_outcome
from prometheist.cognitive_store import get_record, put_record
from prometheist.consolidation import consolidate_page
from prometheist.epistemic_authority import format_authority_bound_memory_packet
from prometheist.llm import _quarantined_evidence
from prometheist.model_evidence_budget import configured_model_evidence_budget, validate_rendered_evidence
from prometheist.models import MemoryPacket
from prometheist.percept_response_runtime import ResponseMemoryPackage, retrieve_memory_package
from prometheist.percept_response_worker import UserPromptLLM
from prometheist.percept_triage import (
    TRIAGE_PROMPT,
    TriageDecision,
    TaskClass,
    deterministic_triage,
    validate_triage,
)
from prometheist.response_policy import (
    HistoricalEvidenceScope,
    ResponsePolicy,
    ResponseSurfaceMode,
    source_types_for_scope,
)
from prometheist.self_memory import (
    SelfEvidenceRelation,
    get_self_representation,
    self_evidence,
)
from prometheist.self_reflection import (
    SELF_SCHEMA_PROPOSAL_PROMPT,
    SELF_SCHEMA_REVIEW_PROMPT,
    SelfSchemaProposalBatch,
    SelfSchemaReview,
    apply_review,
    collect_consolidation_evidence,
    materialize_proposal,
    reflection_batches,
    render_reflection_evidence,
    render_candidate_support_evidence,
    review_memory_packet,
)
from prometheist.situation_runtime import (
    SITUATION_PROTOCOL,
    SituationStage,
    SituationTask,
    situation_stage_uses_model,
)
from prometheist.worker_protocol import deterministic_worker_step_id
from prometheist.worker_store import complete_worker_claim, load_worker_claim_envelope, load_worker_result, release_worker_claim

TRIAGE_MAX_TOKENS = 512
SELF_PROPOSAL_MAX_TOKENS = 768
SELF_REVIEW_MAX_TOKENS = 384
SITUATION_MODEL_ITEMS = 8

_ALLOWED_LLM_KINDS = {stage: STAGE_CONTRACTS[stage.value].kinds for stage in SituationStage}


class SituationLLM(UserPromptLLM):
    def _require_stage_specialization(self, kind: str) -> None:
        stage = self._artifact_stage
        if not isinstance(stage, SituationStage) or kind not in _ALLOWED_LLM_KINDS[stage]:
            raise RuntimeError(f"{stage} cannot invoke LLM role {kind}")

    def propose_self_schemas(self, evidence: str) -> SelfSchemaProposalBatch:
        raw = self._structured_with_evidence(
            "SELF_SCHEMA_PROPOSAL",
            SELF_SCHEMA_PROPOSAL_PROMPT,
            "Propose only self representations supported by the supplied evidence.",
            _quarantined_evidence(evidence),
            SEMANTIC_CONTRACTS["SELF_SCHEMA_PROPOSAL"].output_schema(),
            SELF_PROPOSAL_MAX_TOKENS,
        )
        return self._validated_model_output(
            kind="SELF_SCHEMA_PROPOSAL",
            raw_output=raw,
            validator=lambda: SelfSchemaProposalBatch.model_validate_json(raw),
        )

    def review_self_schema(
        self,
        *,
        candidate: str,
        support_evidence: str,
        related_evidence: str,
    ) -> SelfSchemaReview:
        payload = (
            "[Candidate]\n"
            + candidate
            + "\n\n"
            + support_evidence
            + "\n\n[Related/counterevidence]\n"
            + related_evidence
        )
        raw = self._structured_with_evidence(
            "SELF_SCHEMA_REVIEW",
            SELF_SCHEMA_REVIEW_PROMPT,
            "Review the supplied self representation against all supplied evidence.",
            _quarantined_evidence(payload),
            SEMANTIC_CONTRACTS["SELF_SCHEMA_REVIEW"].output_schema(),
            SELF_REVIEW_MAX_TOKENS,
        )
        return self._validated_model_output(
            kind="SELF_SCHEMA_REVIEW",
            raw_output=raw,
            validator=lambda: SelfSchemaReview.model_validate_json(raw),
        )

    def triage(self, policy, evidence: str) -> TriageDecision:
        if not policy.semantic_triage:
            raise ValueError("semantic triage is not enabled")
        system = TRIAGE_PROMPT + "\nApplication policy:\n" + policy.model_dump_json()
        current = "Classify the operational need under the application policy."
        raw = self._structured_with_evidence(
            "PERCEPT_TRIAGE",
            system,
            current,
            _quarantined_evidence(evidence),
            SEMANTIC_CONTRACTS["PERCEPT_TRIAGE"].output_schema(),
            TRIAGE_MAX_TOKENS,
        )
        return self._validated_model_output(
            kind="PERCEPT_TRIAGE",
            raw_output=raw,
            validator=lambda: validate_triage(TriageDecision.model_validate_json(raw), policy),
        )


def _assignment_id(task: SituationTask) -> UUID:
    """A claimed situation stage always runs under one durable assignment."""

    if task.assignment_id is None:
        raise RuntimeError("situation task has no durable assignment")
    return task.assignment_id


def _output(conn, task: SituationTask, stage: SituationStage, scheduler_key: str) -> dict:
    result = load_worker_result(conn, deterministic_worker_step_id(_assignment_id(task), stage.value), scheduler_key=scheduler_key)
    if result is None:
        raise RuntimeError(f"missing durable prerequisite: {stage.value}")
    return result.output


def _source_types(domains) -> list:
    return sorted({event_type for domain in domains for event_type in source_types_for_scope(domain)}, key=lambda item: item.value)


def _situation_view(task: SituationTask) -> dict:
    situation = task.situation
    return {"epistemic_status": "DERIVED", "snapshot_id": str(situation.snapshot_id),
            "prediction_error": situation.prediction_error, "retention_hint": situation.retention_hint,
            "observed_state": [value.model_dump(mode="json") for value in situation.observed_state[-SITUATION_MODEL_ITEMS:]],
            "prediction_errors": [value.model_dump(mode="json") for value in situation.prediction_errors[-SITUATION_MODEL_ITEMS:]],
            "percept_excerpt": list(task.percept.input_buffer.segments),
            "excerpt_incomplete": task.percept.input_buffer.truncated,
            "state_incomplete": len(situation.observed_state) > SITUATION_MODEL_ITEMS,
            "provenance": [str(value) for value in situation.provenance[-SITUATION_MODEL_ITEMS:]]}


def execute_situation_stage(conn, task: SituationTask, stage: SituationStage, *, llm=None, scheduler_key: str) -> dict:
    if stage is SituationStage.MEMORY:
        packet = jit_memory.request_attention_activation(
            conn, conversation_id=task.conversation_id, correlation_id=task.correlation_id,
            requesting_component=f"situation-aperture:{task.task_id}",
            need=jit_memory.build_memory_need(
                "\n".join(task.percept.input_buffer.segments), include_persisted_history=True,
                conversation_id=None, source_types=_source_types(task.policy.evidence_domains),
            ), before_global_seq=task.before_global_seq, memory_request_id=uuid5(task.task_id, "memory-aperture"),
        )
        return {"memory_packet": packet.model_dump(mode="json"), "source_policy": task.policy.model_dump(mode="json")}
    if stage is SituationStage.TRIAGE:
        memory = _output(conn, task, SituationStage.MEMORY, scheduler_key)
        decision = deterministic_triage(task.percept, task.situation, task.policy)
        if decision is not None:
            return {"decision": decision.model_dump(mode="json"), "inference_used": False}
        if llm is None:
            raise RuntimeError("opaque non-user semantics require a guarded triage specialist")
        packet = MemoryPacket.model_validate(memory["memory_packet"])
        evidence = json.dumps(_situation_view(task), ensure_ascii=False) + "\n" + format_authority_bound_memory_packet(packet)
        validate_rendered_evidence((evidence,), budget=configured_model_evidence_budget())
        llm._set_artifact_evidence_refs(tuple(f"event:{item.source_event_id}" for item in packet.items) +
                                      (f"situation:{task.situation.snapshot_id}",))
        decision = llm.triage(task.policy, evidence)
        return {"decision": decision.model_dump(mode="json"), "inference_used": True}
    if stage is SituationStage.EXECUTE:
        decision = TriageDecision.model_validate(_output(conn, task, SituationStage.TRIAGE, scheduler_key)["decision"])
        task_class = decision.candidate_task_class
        if not decision.task_required or task_class is None:
            return {"work_results": [], "task_required": False}
        action_id = uuid5(task.task_id, "registered-action")
        issue_action(conn, action_id=action_id, task_id=task.task_id, at=task.created_at,
                     entity_refs=task.situation.entity_refs[:16])
        saved = get_record(conn, "action_execution", str(action_id))
        if saved is None:
            if task_class is TaskClass.CONSOLIDATE:
                cursor = json.loads(task.percept.normalized_text).get("after_key", "")
                projection = get_record(conn, "consolidation", str(action_id)) or consolidate_page(conn, action_id=action_id, after_key=cursor)
                result_data = {"consolidation_id": str(action_id), "projection_count": len(projection["projections"]),
                               "next_cursor": projection["next_cursor"], "canonical_records_modified": False}
            else:
                result_data = _situation_view(task)
                result_data["operation"] = task_class.value
                result_data["meaning"] = "Observed discrepancies recorded for reconciliation; external state was not changed."
            saved = {"work_results": [{"capability_id": f"situation.{task_class.value.casefold()}",
                                        "executor": "application", "result_data": result_data}], "task_required": True, "observed_status": "SUCCEEDED"}
        receipt_id = put_record(conn, "action_execution", str(action_id), saved, revision="1")
        receipt = event_store.get_event_by_id(conn, receipt_id)
        # Read-after-write confirms our registered local operation. It makes no
        # claim that an external command has completed or a physical fault healed.
        if receipt is None or get_record(conn, "action_execution", str(action_id)) != saved:
            raise RuntimeError("action completion could not be observed")
        observe_action_outcome(conn, action_id=action_id, receipt_event_id=receipt_id,
                               status="SUCCEEDED", observed_at=receipt.created_at)
        return saved
    if stage is SituationStage.SELF_PROPOSE:
        decision = TriageDecision.model_validate(
            _output(conn, task, SituationStage.TRIAGE, scheduler_key)["decision"]
        )
        if decision.candidate_task_class is not TaskClass.CONSOLIDATE:
            return {"skipped": True, "candidates": []}

        action_id = uuid5(task.task_id, "registered-action")
        projection = get_record(conn, "consolidation", str(action_id))
        if projection is None:
            raise RuntimeError("self reflection requires completed semantic consolidation")
        derived_at = projection["derived_at"]
        evidence = collect_consolidation_evidence(conn, action_id)
        if not evidence:
            return {
                "skipped": True,
                "candidates": [],
                "root_event_count": 0,
                "derived_at": derived_at,
            }
        if llm is None:
            raise RuntimeError("self-schema proposal requires its own guarded specialist")
        candidates = []
        for batch in reflection_batches(evidence):
            proposal_batch = llm.propose_self_schemas(
                render_reflection_evidence(batch)
            )
            for proposal in proposal_batch.proposals:
                representation, resolution = materialize_proposal(
                    conn,
                    proposal=proposal,
                    evidence=batch,
                    derived_at=datetime.fromisoformat(str(derived_at)),
                )
                candidates.append(
                    {
                        "representation_id": str(representation.representation_id),
                        "proposal": proposal.model_dump(mode="json"),
                        "candidate_resolution_id": str(resolution.resolution_id),
                    }
                )
        return {
            "skipped": False,
            "candidates": candidates,
            "root_event_count": len(evidence),
            "derived_at": derived_at,
        }

    if stage is SituationStage.SELF_REVIEW:
        decision = TriageDecision.model_validate(
            _output(conn, task, SituationStage.TRIAGE, scheduler_key)["decision"]
        )
        if decision.candidate_task_class is not TaskClass.CONSOLIDATE:
            return {"skipped": True, "reviews": []}

        proposal_output = _output(
            conn,
            task,
            SituationStage.SELF_PROPOSE,
            scheduler_key,
        )
        if proposal_output.get("skipped") or not proposal_output.get("candidates"):
            return {"skipped": True, "reviews": []}
        if llm is None:
            raise RuntimeError("self-schema review requires its own guarded specialist")
        reviews = []
        seen = set()
        for candidate in proposal_output.get("candidates", []):
            representation_id = UUID(candidate["representation_id"])
            if representation_id in seen:
                continue
            seen.add(representation_id)
            representation = get_self_representation(conn, representation_id)
            if representation is None:
                raise RuntimeError(
                    f"self proposal disappeared before review: {representation_id}"
                )
            packet = review_memory_packet(
                conn,
                representation=representation,
                conversation_id=task.conversation_id,
                correlation_id=task.correlation_id,
                before_global_seq=task.before_global_seq,
            )
            support_evidence = render_candidate_support_evidence(
                conn,
                representation,
            )
            related_evidence = format_authority_bound_memory_packet(packet)
            validate_rendered_evidence(
                (support_evidence, related_evidence),
                budget=configured_model_evidence_budget(),
            )
            llm._set_artifact_evidence_refs(
                tuple(
                    f"event:{item.root_event_id}"
                    for item in self_evidence(
                        conn,
                        representation.representation_id,
                    )
                    if item.relation is SelfEvidenceRelation.SUPPORTS
                )
                + tuple(f"event:{item.source_event_id}" for item in packet.items)
            )
            review = llm.review_self_schema(
                candidate=representation.model_dump_json(),
                support_evidence=support_evidence,
                related_evidence=related_evidence,
            )
            resolution = apply_review(
                conn,
                representation=representation,
                review=review,
                review_packet=packet,
                resolved_at=datetime.fromisoformat(
                    str(proposal_output["derived_at"])
                ),
            )
            reviews.append(
                {
                    "representation_id": str(representation_id),
                    "review": review.model_dump(mode="json"),
                    "resolution_id": str(resolution.resolution_id),
                    "status": resolution.status.value,
                }
            )
        return {"skipped": False, "reviews": reviews}

    if stage is SituationStage.RETRIEVE:
        packet = MemoryPacket.model_validate(_output(conn, task, SituationStage.MEMORY, scheduler_key)["memory_packet"])
        if not task.policy.response_required or not task.policy.natural_language_response:
            return {"skipped": True}
        package = retrieve_memory_package(conn, task, packet, _source_types(task.policy.evidence_domains))
        return {"memory_package": package.model_dump(mode="json")}
    if stage is SituationStage.RESPOND:
        execution = _output(conn, task, SituationStage.EXECUTE, scheduler_key)
        if not task.policy.response_required:
            return {"response_required": False, "response_text": None}
        if task.policy.natural_language_response:
            if llm is None:
                raise RuntimeError("natural response requires a separate final responder")
            package = ResponseMemoryPackage.model_validate(_output(conn, task, SituationStage.RETRIEVE, scheduler_key)["memory_package"])
            text = llm.generate_final_response(
                task.user_text, package, tuple(execution["work_results"]) or
                ({"capability_id": "situation.observation", "result_data": _situation_view(task)},),
                response_policy=ResponsePolicy(evidence_scope=HistoricalEvidenceScope.DERIVED_INTERNAL,
                                               surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE),
            )
        else:
            text = json.dumps({"situation": _situation_view(task), "work": execution}, ensure_ascii=False, sort_keys=True)
        return {"response_required": True, "response_text": text}
    if stage is SituationStage.PERSIST:
        response = _output(conn, task, SituationStage.RESPOND, scheduler_key)
        put_record(conn, "situation_completion", str(task.task_id), response, revision="1")
        return response
    raise ValueError("unsupported situation stage")


def execute_claimed_situation_step(conn, *, claim_id: UUID, worker_id: str, scheduler_key: str, llm_factory=SituationLLM) -> dict:
    envelope = load_worker_claim_envelope(conn, claim_id, worker_id=worker_id, scheduler_key=scheduler_key)
    task = SituationTask.model_validate(get_record(conn, "situation_task", str(envelope.step.task_id)))
    if task.protocol_version != SITUATION_PROTOCOL or task.assignment_id != envelope.step.assignment_id:
        raise ValueError("situation task protocol or assignment mismatch")
    stage = SituationStage(envelope.step.step_key)
    artifact_journal.write_percept_artifact(
        interaction_id=task.task_id, conversation_id=task.conversation_id, correlation_id=task.correlation_id,
        task_id=task.task_id, user_text=task.user_text, user_prompt_event_id=task.user_prompt_event_id,
        percept=task.percept, salience_assessment=task.situation.salience,
    )
    try:
        recovered = artifact_journal.load_stage_result_artifact(task.task_id, stage.value)
        if recovered:
            output = recovered["output"]
        else:
            uses_model = situation_stage_uses_model(
                conn,
                task,
                stage,
                scheduler_key=scheduler_key,
            )
            llm = (
                llm_factory(interaction=task, stage=stage, claim_id=claim_id)
                if uses_model
                else None
            )
            output = execute_situation_stage(conn, task, stage, llm=llm, scheduler_key=scheduler_key)
            artifact_journal.write_stage_result_artifact(
                interaction_id=task.task_id, conversation_id=task.conversation_id, correlation_id=task.correlation_id,
                task_id=task.task_id, assignment_id=_assignment_id(task), stage=stage.value,
                output=output, output_refs=[f"situation:{task.situation.snapshot_id}"],
            )
        complete_worker_claim(conn, claim_id=claim_id, worker_id=worker_id, output=output,
                              output_refs=[f"situation:{task.situation.snapshot_id}"], scheduler_key=scheduler_key)
        return output
    except Exception as exc:
        artifact_journal.write_stage_error_artifact(
            interaction_id=task.task_id, conversation_id=task.conversation_id, correlation_id=task.correlation_id,
            task_id=task.task_id, assignment_id=_assignment_id(task), stage=stage.value, claim_id=claim_id,
            error_type=type(exc).__name__, message=str(exc),
        )
        release_worker_claim(conn, claim_id=claim_id, worker_id=worker_id, scheduler_key=scheduler_key)
        raise


def main() -> None:
    with db.get_connection() as conn:
        execute_claimed_situation_step(conn, claim_id=UUID(os.environ["PROMETHEIST_WORKER_CLAIM_ID"]),
                                       worker_id=os.environ["PROMETHEIST_WORKER_ID"],
                                       scheduler_key=os.environ["PROMETHEIST_WORKER_SCHEDULER_KEY"])


if __name__ == "__main__":
    main()
