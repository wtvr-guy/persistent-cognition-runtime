"""Headless chat job process using the guarded worker pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from uuid import UUID

from prometheist.runtime_settings import job_settings
from prometheist.operator_state import write_private_policy


def run(directory):
    from prometheist import db
    from prometheist.model_catalog import validate_local_selection
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    action, payload = job["action"], job["payload"]
    settings = job_settings()

    def report(value):
        write_private_policy(directory / "progress.json", value)

    if action != "chat":
        raise ValueError("Unregistered job action")
    from prometheist.runtime_settings import AppSettings
    from prometheist.model_admission import prepare_task
    from prometheist.model_residency import prepare_local_models
    report({"status": "Unloading local models before measuring available capacity"})
    residency = prepare_local_models(settings)
    report({"status": "Matching installed models to the task and measured host capacity"})
    plan = prepare_task(settings, payload["text"], payload.get("task", "auto"), fallback=payload.get("fallback", "review"))
    plan = plan.model_copy(update={"residency": residency})
    write_private_policy(directory / "admission.json", plan.model_dump(mode="json"))
    if plan.status != "eligible":
        raise ValueError("; ".join(plan.reasons))
    settings = AppSettings.model_validate({**settings.model_dump(),
        "routes": {**settings.routes, **plan.stages}})
    effective_path = directory / "effective-settings.json"
    write_private_policy(effective_path, settings.model_dump(mode="json"))
    os.environ.update(settings.worker_environment(effective_path))
    from prometheist.contract_registry import STAGE_CONTRACTS
    selections = {settings.selection_for(stage).model_dump_json(): settings.selection_for(stage)
                  for stage, contract in STAGE_CONTRACTS.items() if contract.kinds and stage.startswith("V2_")}
    for selection in selections.values():
        if selection.provider == "ollama":
            report({"status": "Checking " + selection.model})
            validate_local_selection(settings.ollama_url, selection)
        else:
            from prometheist.network_consent import NetworkPurpose, require_destination
            from prometheist.openai_transport import OPENAI_ORIGIN
            require_destination(OPENAI_ORIGIN, NetworkPurpose.MODEL)
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Connect your OpenAI API key before starting this task")
    floor = plan.required_memory_mib
    os.environ["PROMETHEIST_GUI_MODEL_MEMORY_FLOOR_MIB"] = str(floor)
    os.environ["PROMETHEIST_GUI_MODEL_CPU_UNITS"] = str(plan.required_cpu_units)
    report({"status": "Checking local capacity", "model_memory_requirement_mib": floor,
            "response_model": plan.stages["V2_RESPOND"].model, "route_reason": plan.route_reason})
    from prometheist.chat_startup import reset_chat_execution_state
    from prometheist.cli import _handle_with_admission_diagnostics
    with db.get_connection() as conn:
        acquired = conn.execute("SELECT pg_try_advisory_lock(hashtext('prometheist.gui.chat'))").fetchone()[0]
        conn.commit()
        if not acquired:
            raise ValueError("Another Prometheist GUI is already running a chat task")
        scheduler_key = "gui-chat"
        reset_chat_execution_state(conn, scheduler_key=scheduler_key)
        text = _handle_with_admission_diagnostics(conn, payload["text"], UUID(payload["conversation_id"]),
            scheduler_key=scheduler_key, progress=lambda stage: report({"status": "Cognitive worker", "stage": stage}))
    return {"text": text, "conversation_id": payload["conversation_id"], "model_memory_requirement_mib": floor,
            "response_model": plan.stages["V2_RESPOND"].model, "task": plan.task}


def main():
    from prometheist.cli import _configure_utf8_streams
    _configure_utf8_streams()
    directory = Path(sys.argv[1])
    try:
        result = run(directory)
        write_private_policy(directory / "result.json", {"result": result})
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for name in ("OPENAI_API_KEY", "DATABASE_URL"):
            secret = os.environ.get(name)
            if secret:
                message = message.replace(secret, "[redacted]")
        write_private_policy(directory / "result.json", {"error": message[:2000]})
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
