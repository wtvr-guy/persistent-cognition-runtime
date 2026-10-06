"""Headless chat job process using the guarded worker pipeline."""
from __future__ import annotations

from persistent_cognition.diagnostics import exception_summary

import os
from pathlib import Path
import sys
from uuid import UUID

from persistent_cognition.runtime_settings import job_settings
from persistent_cognition.operator_state import write_private_policy


def run(directory):
    from persistent_cognition import db
    from persistent_cognition.resource_limits import read_json_file
    job = read_json_file(directory / "job.json")
    action, payload = job["action"], job["payload"]
    settings = job_settings()
    if settings is None:
        raise ValueError("Headless jobs require PCR_GUI_JOB_CONFIG_FILE or PCR_GUI_JOB_CONFIG")
    from persistent_cognition.resource_limits import validate_text_intake
    validate_text_intake(payload["text"])

    def report(value):
        write_private_policy(directory / "progress.json", value)

    if action != "chat":
        raise ValueError("Unregistered job action")
    from persistent_cognition.model_admission import prepare_task
    from persistent_cognition.model_residency import prepare_local_models
    from persistent_cognition.advisory_lock import SchedulerOwnershipError, scheduler_ownership
    scheduler_key = "gui-chat"
    with db.get_connection() as conn:
        # Take single-owner authority over this scheduler namespace BEFORE the
        # remaining host-visible effects (worker-environment mutation, model
        # validation, and the chat execution-state reset) so a non-owner process
        # can never perform them for this namespace.
        try:
            with scheduler_ownership(conn, scheduler_key):
                # Do not unload models belonging to a live namespace owner.
                # Admission still precedes any cognitive execution-state writes.
                report({"status": "Unloading local models before measuring available capacity"})
                residency = prepare_local_models(settings)
                report({"status": "Matching installed models to the task and measured host capacity"})
                plan = prepare_task(settings, payload["text"], payload.get("task", "auto"), fallback=payload.get("fallback", "review"))
                plan = plan.model_copy(update={"residency": residency})
                write_private_policy(directory / "admission.json", plan.model_dump(mode="json"))
                if plan.status != "eligible":
                    raise ValueError("; ".join(plan.reasons))
                return _run_chat_job(
                    conn,
                    directory=directory,
                    settings=settings,
                    payload=payload,
                    report=report,
                    plan=plan,
                    scheduler_key=scheduler_key,
                )
        except SchedulerOwnershipError as exc:
            raise ValueError("Another owner is already running a headless chat task") from exc


def _run_chat_job(conn, *, directory, settings, payload, report, plan, scheduler_key):
    from persistent_cognition.runtime_settings import AppSettings
    from persistent_cognition.model_catalog import validate_local_selection
    settings = AppSettings.model_validate({**settings.model_dump(),
        "routes": {**settings.routes, **plan.stages}})
    effective_path = directory / "effective-settings.json"
    write_private_policy(effective_path, settings.model_dump(mode="json"))
    os.environ.update(settings.worker_environment(effective_path))
    from persistent_cognition.contract_registry import STAGE_CONTRACTS
    selections = {settings.selection_for(stage).model_dump_json(): settings.selection_for(stage)
                  for stage, contract in STAGE_CONTRACTS.items() if contract.kinds and stage.startswith("V2_")}
    for selection in selections.values():
        if selection.provider == "ollama":
            report({"status": "Checking " + selection.model})
            validate_local_selection(settings.ollama_url, selection)
        else:
            from persistent_cognition.network_consent import NetworkPurpose, require_destination
            from persistent_cognition.openai_transport import OPENAI_ORIGIN
            require_destination(OPENAI_ORIGIN, NetworkPurpose.MODEL)
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Connect your OpenAI API key before starting this task")
    floor = plan.required_memory_mib
    os.environ["PCR_GUI_MODEL_MEMORY_FLOOR_MIB"] = str(floor)
    os.environ["PCR_GUI_MODEL_CPU_UNITS"] = str(plan.required_cpu_units)
    report({"status": "Checking local capacity", "model_memory_requirement_mib": floor,
            "response_model": plan.stages["V2_RESPOND"].model, "route_reason": plan.route_reason})
    from persistent_cognition.chat_startup import reset_chat_execution_state
    from persistent_cognition.api import respond
    reset_chat_execution_state(conn, scheduler_key=scheduler_key)
    text = respond(conn, payload["text"], UUID(payload["conversation_id"]),
        scheduler_key=scheduler_key, progress=lambda stage: report({"status": "Cognitive worker", "stage": stage}))
    return {"text": text, "conversation_id": payload["conversation_id"], "model_memory_requirement_mib": floor,
            "response_model": plan.stages["V2_RESPOND"].model, "task": plan.task}


def main():
    from persistent_cognition.cli import _configure_utf8_streams
    _configure_utf8_streams()
    directory = Path(sys.argv[1])
    try:
        result = run(directory)
        write_private_policy(directory / "result.json", {"result": result})
    except Exception as exc:
        message = exception_summary(exc)
        for name in ("OPENAI_API_KEY", "DATABASE_URL"):
            secret = os.environ.get(name)
            if secret:
                message = message.replace(secret, "[redacted]")
        write_private_policy(directory / "result.json", {"error": message[:2000]})
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
