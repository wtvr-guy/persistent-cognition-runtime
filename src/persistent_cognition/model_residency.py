"""Verified cold boundaries for the local app's sequential Ollama workers.

The scheduler limits worker concurrency; this module controls the separate model
server's residency. No weights are deleted. Remote services are not managed here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import time

import httpx

from persistent_cognition.runtime_settings import job_settings, local_ollama
from persistent_cognition.model_catalog import MODEL_HTTP_TIMEOUT_SECONDS
from persistent_cognition.ollama_runtime import _canonical_model_name

RESIDENCY_VERSION = "sequential-cold-models/v1"
UNLOAD_TIMEOUT_SECONDS = 10
UNLOAD_POLL_SECONDS = 0.1


class ModelResidencyError(RuntimeError):
    """No new inference is permitted when a cold boundary cannot be verified."""


def running_models(client, *, timeout=MODEL_HTTP_TIMEOUT_SECONDS):
    from persistent_cognition.network_consent import NetworkPurpose, require_destination
    require_destination(str(client.base_url), NetworkPurpose.MODEL)
    response = client.get("/api/ps", timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("models"), list):
        raise ModelResidencyError("Ollama did not provide a valid running-model inventory")
    records = {}
    for item in body["models"]:
        if not isinstance(item, dict):
            raise ModelResidencyError("Invalid running-model record")
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            raise ModelResidencyError("Running-model record has no model name")
        records[_canonical_model_name(name)] = {**item, "name": name}
    return [records[key] for key in sorted(records)]


def unload_and_verify(client, receipt):
    """Request unload once per observed model; poll until the daemon is empty."""
    deadline = time.monotonic() + UNLOAD_TIMEOUT_SECONDS
    requested = set()
    receipt.update({"unload_requested": [], "verified_empty": False})

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ModelResidencyError("Ollama did not unload in time; no next model will be loaded")
        return seconds

    while True:
        models = running_models(client, timeout=remaining())
        receipt["last_running"] = [item["name"] for item in models]
        if not models:
            receipt["verified_empty"] = True
            return
        for item in models:
            key = _canonical_model_name(item["name"])
            if key in requested:
                continue
            response = client.post("/api/generate", json={"model": item["name"], "keep_alive": 0,
                                                         "stream": False}, timeout=remaining())
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or body.get("error") or body.get("done") is not True:
                raise ModelResidencyError("Ollama did not acknowledge the unload request")
            requested.add(key)
            receipt["unload_requested"].append(item["name"])
        time.sleep(min(UNLOAD_POLL_SECONDS, remaining()))


@contextmanager
def residency_lock(base_url):
    # Separate from the app-instance lock: workers are separate processes. All
    # loopback spellings for a given port use the same lock in this private root.
    from urllib.parse import urlsplit
    from persistent_cognition.artifact_journal import artifact_root
    from persistent_cognition.process_lock import instance_lock
    parsed = urlsplit(str(base_url))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    key = hashlib.sha256(f"{parsed.scheme}:loopback:{port}".encode()).hexdigest()
    with instance_lock(artifact_root() / "operator" / "model-residency" / key):
        yield


def prepare_local_models(settings):
    """At actual job start, unload first; admission then measures real free RAM."""
    from persistent_cognition.contract_registry import STAGE_CONTRACTS
    if not local_ollama(settings) or not any(
        settings.selection_for(stage).provider == "ollama"
        for stage, contract in STAGE_CONTRACTS.items() if contract.kinds and stage.startswith("V2_")
    ):
        return {}
    receipt = {"policy": RESIDENCY_VERSION}
    with residency_lock(settings.ollama_url), httpx.Client(
        base_url=settings.ollama_url, trust_env=False, follow_redirects=False,
        timeout=MODEL_HTTP_TIMEOUT_SECONDS,
    ) as client:
        unload_and_verify(client, receipt)
    return receipt


def preview_after_unload(settings, plan, replan):
    """Read-only forecast; actual jobs must unload and pass physical admission.

Cached models consume RAM in the OS observation. Without this separate forecast,
the Send button could refuse to start the job that would release those models.
The forecast is explicitly NOT an execution entitlement or a warm-model credit.
"""
    if plan.status == "eligible" or not plan.observation.healthy:
        return plan
    from persistent_cognition.attention_observation import build_resource_observation, discover_local_execution_resources
    try:
        with httpx.Client(base_url=settings.ollama_url, trust_env=False, follow_redirects=False,
                          timeout=MODEL_HTTP_TIMEOUT_SECONDS) as client:
            models = running_models(client)
    except Exception as exc:
        return plan.model_copy(update={"residency": {"policy": RESIDENCY_VERSION,
            "preview_error": f"Running-model inventory unavailable ({type(exc).__name__})"}})
    physical = plan.observation
    reclaim = 0
    for model in models:
        size, vram = model.get("size"), model.get("size_vram")
        if type(size) is int and type(vram) is int and 0 <= vram <= size:
            reclaim += (size - vram) // (1024 * 1024)
    reclaim = min(reclaim, physical.metrics.memory_total_mib - physical.metrics.memory_available_mib)
    if reclaim <= 0:
        return plan
    metrics = physical.metrics.model_copy(update={
        "memory_available_mib": physical.metrics.memory_available_mib + reclaim})
    observation = build_resource_observation(scheduler_cycle=physical.scheduler_cycle,
        captured_at=physical.captured_at, policy=physical.policy, metrics=metrics,
        resources=discover_local_execution_resources(metrics, policy=physical.policy), reservations=[])
    projected = replan(observation)
    updates = {"capacity_basis": "after_unload_estimate", "physical_observation": physical,
               "residency": {"policy": RESIDENCY_VERSION, "running_models": models,
                             "estimated_reclaim_mib": reclaim, "execution_authorized": False}}
    if projected.status == "eligible":
        updates.update(status="needs_unload", reasons=[
            "Unload local models, then remeasure CPU/RAM before execution; this preview is an estimate"])
    return projected.model_copy(update=updates)


@contextmanager
def managed_inference(client, request, diagnostics):
    settings = job_settings()
    if settings is None or not local_ollama(settings):
        yield
        return
    # Apply after model/environment overrides. A retained general model would
    # invalidate the sequential cold-memory budget for the next worker.
    request["keep_alive"] = 0
    receipt = {"policy": RESIDENCY_VERSION, "before": {}, "after": {}}
    diagnostics["model_residency"] = receipt
    with residency_lock(client.base_url):
        unload_and_verify(client, receipt["before"])
        try:
            yield
        finally:
            # Covers completed calls, HTTP errors, and transport timeouts. A
            # killed process cannot run finally; the next job repeats preflight.
            unload_and_verify(client, receipt["after"])
