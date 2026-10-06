"""Replayable CPU/RAM eligibility and installed-only specialist routing.

No model calls, downloads, semantic classifiers, cloud fallbacks or VRAM credits.
The memory model is an explicit provisional engineering envelope, not an OOM
guarantee. Unknown architectures require a registered estimator before admission.
"""
from __future__ import annotations

from persistent_cognition.diagnostics import exception_summary

from datetime import datetime, timezone
import math
from types import MappingProxyType
from typing import Literal

from pydantic import Field

from persistent_cognition.attention_observation import (
    HOST_CPU_RESOURCE_ID, HOST_MEMORY_RESOURCE_ID, ResourceObservationSnapshot,
    ResourceSafetyPolicy, ResourceObservationError, SystemHostResourceProbe, build_resource_observation,
    discover_local_execution_resources,
)
from persistent_cognition.content_digest import content_digest
from persistent_cognition.runtime_settings import AppSettings, ModelSelection, SettingsRecord, local_ollama
from persistent_cognition.ollama_runtime import _canonical_model_name

ADMISSION_VERSION = "model-admission/cpu-v2"
ROUTING_VERSION = "task-routing/v1"
MIB_BYTES = 1024 * 1024
KV_ELEMENT_BYTES = 4  # conservative f32 envelope; no quantized-cache savings
MAX_ESTIMATED_BATCH = 128
# starcoder2 (e.g. the "dolphincoder" coding specialist) reports the same
# generic GQA/RoPE fields the estimator already reads (attention.head_count,
# attention.head_count_kv, block_count, context_length, embedding_length), so
# the same weights+KV-cache formula applies. Like the already-registered
# gemma2, it alternates sliding-window and global attention, so this
# estimator's dense (every layer, full context) formula is a safe overestimate
# rather than an underestimate.
DENSE_ARCHITECTURES = frozenset({"llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "starcoder2"})
TASKS = MappingProxyType({
    "general": {"label": "General", "requires": ["completion"], "input": "text", "implemented": True},
    "coding": {"label": "Coding", "requires": ["completion"], "input": "text", "implemented": True},
    "vision": {"label": "Vision", "requires": ["completion", "vision"], "input": "image", "implemented": False},
})
CONTROL_MESSAGES = MappingProxyType({
    "specialist_offer": "No eligible installed {task} specialist. Search the catalog, then review OpenAI if needed; the local default is the final option.",
    "specialist_required": "Choose an installed {task} specialist, enable default fallback, or review the model library.",
    "local_unavailable": "No usable local route. Search for an installed or downloadable {task} model? You choose all downloads.",
    "openai_offer": "If no local option suits this task, would you like to review an OpenAI route?",
})
TaskKind = Literal["auto", "general", "coding", "vision"]
Status = Literal["eligible", "temporarily_blocked", "unsupported", "unverified", "remote", "needs_choice", "needs_unload"]


class ModelEligibility(SettingsRecord):
    schema_version: Literal["model-eligibility/v1"] = "model-eligibility/v1"
    policy_version: str = ADMISSION_VERSION
    selection: ModelSelection
    status: Status
    reasons: list[str]
    required_memory_mib: int | None = Field(default=None, ge=0)
    required_cpu_units: int = Field(default=1, ge=1)
    components: dict[str, int] = Field(default_factory=dict)
    effective_selection: ModelSelection | None = None
    evidence: dict = Field(default_factory=dict)


class TaskPlan(SettingsRecord):
    schema_version: Literal["task-plan/v1"] = "task-plan/v1"
    routing_version: str = ROUTING_VERSION
    task: str
    classification_reason: str
    route_reason: str
    status: Status
    reasons: list[str]
    assessments: list[ModelEligibility]
    stages: dict[str, ModelSelection]
    observation: ResourceObservationSnapshot
    required_memory_mib: int = Field(ge=0)
    required_cpu_units: int = Field(ge=1)
    settings_digest: str
    specialist_offer: str | None = None
    routing_exclusions: dict[str, str] = Field(default_factory=dict)
    openai_offer: bool = False
    fallback_steps: list[str] = Field(default_factory=list)
    capacity_basis: Literal["measured", "after_unload_estimate"] = "measured"
    physical_observation: ResourceObservationSnapshot | None = None
    residency: dict = Field(default_factory=dict)


def resource_policy(settings: AppSettings) -> ResourceSafetyPolicy:
    values = {key: value for key, value in settings.resources.model_dump().items()
              if key in ResourceSafetyPolicy.model_fields}
    return ResourceSafetyPolicy(policy_version=ADMISSION_VERSION, llm_concurrency_limit=1, **values)


def capture_observation(settings: AppSettings) -> ResourceObservationSnapshot:
    policy = resource_policy(settings)
    errors = []
    try:
        metrics = SystemHostResourceProbe().capture()
    except Exception as exc:
        metrics = None
        errors = [f"Host probe failed: {type(exc).__name__}"]
    return build_resource_observation(scheduler_cycle=1, captured_at=datetime.now(timezone.utc),
        resources=discover_local_execution_resources(metrics, policy=policy), reservations=[],
        policy=policy, metrics=metrics, probe_errors=errors)


def classify_task(text: str, requested: TaskKind = "auto") -> tuple[str, str]:
    if requested != "auto":
        return requested, "Explicit task selected by the operator"
    leading = text.lstrip()
    command_match = False
    for prefix in ("/code", "/coding"):
        if leading.startswith(prefix) and (len(leading) == len(prefix) or
                not (leading[len(prefix)].isalnum() or leading[len(prefix)] == "_")):
            command_match = True
            break
    if command_match or "```" in text:
        return "coding", "Registered rule: /code prefix or fenced code block"
    return "general", "No explicit coding marker; default text task"


def capacity(observation: ResourceObservationSnapshot, resource_id: str) -> int:
    return next(item.available_for_new_work for item in observation.capacities if item.resource_id == resource_id)


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Missing or invalid {label} metadata")
    return value


def assess_model(settings: AppSettings, selection: ModelSelection, installed: dict | None,
                 details: dict, observation: ResourceObservationSnapshot, *, at: datetime,
                 task: str = "general") -> ModelEligibility:
    """Pure decision: identical serialized inputs produce identical output."""
    evidence = {"installed": installed, "details": details, "observation_id": str(observation.observation_id),
                "evaluated_at": at.isoformat(), "resource_settings": settings.resources.model_dump(), "task": task}
    values = {"selection": selection, "evidence": evidence}
    def result(status, reason, **kwargs):
        return ModelEligibility(**values, status=status, reasons=[reason], **kwargs)
    try:
        observation.assert_fresh(at=at)
    except ResourceObservationError:
        return result("unverified", "Host observation is stale; measure again")
    if not observation.healthy:
        return result("unverified", "Host CPU/RAM could not be measured")
    if task not in TASKS:
        return result("unsupported", "Task has no registered capability contract")
    if not TASKS[task]["implemented"]:
        return result("unsupported", "Image input is not connected to the guarded chat pipeline yet")
    worker = settings.resources.default_process_memory_mib
    if selection.provider == "openai" or not local_ollama(settings):
        return result("remote", "Remote capacity is not observable here; explicit route and destination consent required",
                      required_memory_mib=worker, effective_selection=selection)
    if details.get("error"):
        return result("unverified", details["error"])
    if installed is None:
        return result("unsupported", "Model is not installed. Only the operator can request a download")
    if not set(TASKS[task]["requires"]) <= set(details.get("capabilities", [])):
        return result("unsupported", "Model does not report the capabilities required by this task")
    parameters = dict(selection.parameters)
    from persistent_cognition.model_parameters import parameter_catalog
    controls = parameter_catalog("ollama", selection.model, details)
    for name, value in parameters.items():
        if name not in controls:
            return result("unsupported", f"Model does not report support for {name}")
        maximum = controls[name].get("maximum")
        if maximum is not None and isinstance(value, (int, float)) and value > maximum:
            return result("unsupported", f"Requested {name} exceeds this model's reported limit")
    if parameters.get("num_gpu", 0) != 0 or parameters.get("draft_num_predict", 0) != 0:
        return result("unverified", "This estimator covers CPU inference without speculative decoding; GPU profiles need calibration")
    if parameters.get("num_batch", MAX_ESTIMATED_BATCH) > MAX_ESTIMATED_BATCH:
        return result("unverified", "Batch exceeds the registered CPU memory estimator's supported limit")
    info = details.get("model_info", {})
    architecture = info.get("general.architecture")
    if architecture not in DENSE_ARCHITECTURES:
        return result("unverified", "No registered memory estimator for this model architecture")
    try:
        weight_bytes = _positive(installed.get("size"), "installed weight size")
        maximum_context = _positive(info.get(f"{architecture}.context_length"), "context length")
        context = parameters.get("num_ctx", min(settings.resources.model_context_tokens, maximum_context))
        if context > maximum_context:
            return result("unsupported", "Requested context exceeds the model's reported limit")
        layers = _positive(info.get(f"{architecture}.block_count"), "layer count")
        heads = _positive(info.get(f"{architecture}.attention.head_count"), "attention heads")
        kv_heads = _positive(info.get(f"{architecture}.attention.head_count_kv"), "KV heads")
        embedding = _positive(info.get(f"{architecture}.embedding_length"), "embedding length")
        if embedding % heads or kv_heads > heads:
            return result("unverified", "Attention dimensions do not match this estimator")
        key = _positive(info.get(f"{architecture}.attention.key_length", embedding // heads), "key dimension")
        value = _positive(info.get(f"{architecture}.attention.value_length", embedding // heads), "value dimension")
    except ValueError as exc:
        return result("unverified", exception_summary(exc))
    weights = math.ceil(weight_bytes / MIB_BYTES)
    kv = math.ceil(context * layers * kv_heads * (key + value) * KV_ELEMENT_BYTES / MIB_BYTES)
    buffers = max(settings.resources.model_runtime_headroom_min_mib,
                  math.ceil(weights * settings.resources.model_runtime_headroom_percent / 100))
    required = max(settings.resources.default_llm_process_memory_mib, weights + kv + buffers + worker)
    threads = parameters.get("num_thread", 1)
    parameters = {"num_ctx": context, "num_gpu": 0, "num_batch": MAX_ESTIMATED_BATCH,
                  "num_thread": threads, **parameters, "keep_alive": 0}
    cost = {"required_memory_mib": required, "required_cpu_units": threads,
            "components": {"weights_mib": weights, "kv_mib": kv, "buffers_mib": buffers, "worker_mib": worker},
            "effective_selection": ModelSelection(provider=selection.provider, model=selection.model, parameters=parameters)}
    memory_pool = next(item for item in observation.capacities if item.resource_id == HOST_MEMORY_RESOURCE_ID)
    if required > memory_pool.configured_capacity - memory_pool.configured_headroom_units:
        return result("unsupported", "Model and context exceed this host's total RAM envelope", **cost)
    if required > capacity(observation, HOST_MEMORY_RESOURCE_ID) or threads > capacity(observation, HOST_CPU_RESOURCE_ID):
        return result("temporarily_blocked", "Current CPU/RAM availability is below the reserved-headroom requirement", **cost)
    return result("eligible", "Fits the measured CPU/RAM envelope using the registered provisional estimator", **cost)


def read_model_evidence(settings: AppSettings, selections: list[ModelSelection]) -> dict:
    """Metadata only. Importantly, absence never invokes the pull endpoint."""
    from persistent_cognition import model_catalog
    if not local_ollama(settings) or not any(item.provider == "ollama" for item in selections):
        return {}
    try:
        installed = {_canonical_model_name(item["name"]): item for item in model_catalog.installed_models(settings.ollama_url)}
    except Exception as exc:
        return {_canonical_model_name(item.model): {"installed": None, "details": {
                    "error": f"Installed model inventory unavailable ({type(exc).__name__}); check Ollama"}}
                for item in selections if item.provider == "ollama"}
    evidence = {}
    for selection in sorted(selections, key=lambda item: item.model):
        if selection.provider != "ollama":
            continue
        name = _canonical_model_name(selection.model)
        if name in evidence:
            continue
        record = installed.get(name)
        details = {}
        if record:
            try:
                details = model_catalog.model_details(settings.ollama_url, selection.model)
            except Exception as exc:
                details = {"error": f"Metadata unavailable ({type(exc).__name__}); inspect the model service"}
        evidence[name] = {"installed": record, "details": details}
    return evidence


def plan_task(settings: AppSettings, text: str, requested: TaskKind, evidence: dict,
              observation: ResourceObservationSnapshot, *, at: datetime,
              fallback: Literal["review", "default"] = "review") -> TaskPlan:
    """Route only the response stage; explicit stage routes always win."""
    from persistent_cognition.contract_registry import STAGE_CONTRACTS
    task, classification = classify_task(text, requested)
    stages = {stage: settings.selection_for(stage) for stage, contract in STAGE_CONTRACTS.items()
              if contract.kinds and stage.startswith("V2_")}
    assessments = {}
    def assess(selection, kind="general"):
        key = selection.model_dump_json() + kind
        if key not in assessments:
            record = evidence.get(_canonical_model_name(selection.model), {})
            assessments[key] = assess_model(settings, selection, record.get("installed"), record.get("details", {}),
                                             observation, at=at, task=kind)
        return assessments[key]
    route_reason = "Using the selected default model"
    offer = None
    exclusions = {}
    failure = []
    needs_choice = False
    worker = settings.resources.default_process_memory_mib
    def requirements(items):
        # A worker completes and verified unloading finishes before another
        # model may load. Each estimate already includes worker overhead.
        costs = [item.required_memory_mib for item in items
                 if item.status != "remote" and item.required_memory_mib is not None]
        return max([worker, *costs]), max(item.required_cpu_units for item in items)
    if "V2_RESPOND" in settings.routes:
        route_reason = "Explicit final-response stage route takes precedence"
    elif settings.task_routing.enabled and local_ollama(settings) and settings.selection.provider == "ollama":
        candidates = []
        base = [assess(selection) for stage, selection in sorted(stages.items()) if stage != "V2_RESPOND"]
        for entry in sorted(settings.task_routing.specialists, key=lambda item: _canonical_model_name(item.selection.model)):
            if task not in entry.tasks:
                continue
            outcome = assess(entry.selection, task)
            if outcome.status == "eligible":
                memory, cpu = requirements([*base, outcome])
                if memory <= capacity(observation, HOST_MEMORY_RESOURCE_ID) and cpu <= capacity(observation, HOST_CPU_RESOURCE_ID):
                    candidates.append((entry.priority, outcome.required_memory_mib, _canonical_model_name(entry.selection.model), entry.selection))
                else:
                    exclusions[entry.selection.model] = "Largest sequential stage requirement exceeds current CPU/RAM capacity"
            else:
                exclusions[entry.selection.model] = outcome.reasons[0]
        if candidates:
            chosen = min(candidates, key=lambda item: item[:3])[-1]
            stages["V2_RESPOND"] = chosen
            route_reason = "Installed eligible specialist: priority, then memory requirement, then canonical model ID"
        elif fallback == "review":
            route_reason = "No eligible installed specialist; review catalog search, then OpenAI, then the local default"
            offer = CONTROL_MESSAGES["specialist_offer"].format(task=task)
            needs_choice = True
        elif settings.task_routing.allow_default_fallback:
            route_reason = "Operator chose the local default after catalog and OpenAI review"
        else:
            failure.append("No eligible specialist for this task and default fallback is disabled")
            offer = CONTROL_MESSAGES["specialist_required"].format(task=task)
    elif settings.selection.provider == "openai" or not local_ollama(settings):
        route_reason = "Explicit remote selection; automatic specialist routing stays local"
    chosen_assessments = [assess(selection, task if stage == "V2_RESPOND" else "general") for stage, selection in sorted(stages.items())]
    bad = [item for item in chosen_assessments if item.status not in {"eligible", "remote"}]
    failure.extend(f"{item.selection.model}: {item.reasons[0]}" for item in bad)
    required, cpu = requirements(chosen_assessments)
    if required > capacity(observation, HOST_MEMORY_RESOURCE_ID) or cpu > capacity(observation, HOST_CPU_RESOURCE_ID):
        failure.append("Largest sequential stage requirement exceeds current CPU/RAM capacity after headroom")
    if needs_choice and not failure:
        failure.append("Choose a fallback before inference; the local default passes the current capacity check")
    status = "needs_choice" if needs_choice else bad[0].status if bad else "temporarily_blocked" if failure else "eligible"
    if failure and offer is None and any(item.selection.provider == "ollama" for item in chosen_assessments):
        offer = CONTROL_MESSAGES["local_unavailable"].format(task=task)
    effective = {stage: (assess(selection, task if stage == "V2_RESPOND" else "general").effective_selection or selection)
                 for stage, selection in sorted(stages.items())}
    return TaskPlan(task=task, classification_reason=classification, route_reason=route_reason,
        status=status, reasons=list(dict.fromkeys(failure)) or ["All selected stages passed current admission checks"],
        assessments=list(assessments.values()), stages=effective, observation=observation,
        required_memory_mib=required, required_cpu_units=cpu, settings_digest=content_digest(settings.model_dump(mode="json")),
        specialist_offer=offer, routing_exclusions=exclusions,
        openai_offer=bool((needs_choice or failure) and TASKS[task]["implemented"] and all(item.selection.provider == "ollama" for item in chosen_assessments)),
        fallback_steps=["catalog_search", "openai_review", "local_default"] if needs_choice else [])


def prepare_task(settings: AppSettings, text: str, requested: TaskKind = "auto", *,
                 fallback: Literal["review", "default"] = "review", preview: bool = False) -> TaskPlan:
    task, _ = classify_task(text, requested)
    selections = [settings.selection, *settings.routes.values()]
    if settings.task_routing.enabled and "V2_RESPOND" not in settings.routes and settings.selection.provider == "ollama":
        selections += [item.selection for item in settings.task_routing.specialists if task in item.tasks]
    evidence = read_model_evidence(settings, selections)
    observation = capture_observation(settings)
    plan = plan_task(settings, text, requested, evidence, observation, at=datetime.now(timezone.utc), fallback=fallback)
    if preview and local_ollama(settings) and any(item.provider == "ollama" for item in selections):
        from persistent_cognition.model_residency import preview_after_unload
        return preview_after_unload(settings, plan, lambda projected: plan_task(
            settings, text, requested, evidence, projected, at=datetime.now(timezone.utc), fallback=fallback))
    return plan


def admission_manifest():
    return {"version": ADMISSION_VERSION, "routing_version": ROUTING_VERSION, "tasks": dict(TASKS),
            "architectures": sorted(DENSE_ARCHITECTURES), "kv_element_bytes": KV_ELEMENT_BYTES,
            "max_batch": MAX_ESTIMATED_BATCH, "downloads": "operator only", "automatic_routes": "installed local specialists only",
            "precedence": ["explicit stage", "eligible installed specialist priority/memory/name", "capability catalog search", "optional OpenAI review", "operator-approved local default"],
            "classifier": "explicit task; otherwise /code or fenced code → coding; otherwise general",
            "specialist_search": {"version": "specialist-search/v1", "coding_query": "coder", "vision_filter": "vision",
                                  "order": "canonical model ID", "verification": "advertised catalog metadata only"},
            "control_messages": {"version": "control-messages/v1", "templates": dict(CONTROL_MESSAGES), "model_calls": False},
            "residency_policy": "sequential-cold-models/v1; one local model per call, verified unload at boundaries",
            "memory_aggregation": "maximum sequential stage requirement including worker overhead",
            "calibration": "LOCAL-APP-001; provisional CPU estimate; unload forecasts never authorize execution"}
