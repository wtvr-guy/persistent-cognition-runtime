from datetime import datetime, timezone
import json

import httpx
import pytest

from prometheist import model_residency as residency
from prometheist.runtime_settings import AppSettings, GUI_CONFIG_ENV, ModelSelection
from prometheist.llm import OllamaClient


@pytest.fixture
def managed(monkeypatch, tmp_path):
    settings = AppSettings(selection=ModelSelection(model="general"))
    monkeypatch.setenv(GUI_CONFIG_ENV, settings.model_dump_json())
    monkeypatch.delenv("PROMETHEIST_GUI_CONFIG_FILE", raising=False)
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(residency.time, "sleep", lambda seconds: None)
    return settings


def test_general_unloads_before_coder_loads_and_receipts_survive(managed):
    running, events = set(), []

    def handler(request):
        if request.url.path == "/api/ps":
            events.append(("inventory", sorted(running)))
            return httpx.Response(200, json={"models": [{"name": name} for name in running]})
        body = json.loads(request.content)
        if request.url.path == "/api/generate" and "prompt" not in body:
            events.append(("unload", body["model"]))
            running.remove(body["model"])
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        assert not running, "A new model loaded before the old model unloaded"
        assert body["keep_alive"] == 0
        events.append(("inference", body["model"]))
        # Simulate a daemon that has not yet honored keep_alive=0: the lifecycle
        # must still explicitly unload and verify before returning to scheduler.
        running.add(body["model"])
        return httpx.Response(200, json={"message": {"content": '{"answer":"ok"}'}})

    for model in ("general", "coder"):
        client = OllamaClient(selection=ModelSelection(model=model, parameters={"keep_alive": -1}))
        client._client.close()
        with httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler)) as transport:
            client._client = transport
            assert client._structured("FINAL_RESPONSE_V2", "system", "request", {}, 256) == '{"answer":"ok"}'
            diagnostics = client._consume_invocation_diagnostics()
            assert diagnostics["model_residency"]["before"]["verified_empty"]
            assert diagnostics["model_residency"]["after"]["verified_empty"]
            assert diagnostics["request_body"]["keep_alive"] == 0
            assert not running
    assert [item for item in events if item[0] != "inventory"] == [
        ("inference", "general"), ("unload", "general"), ("inference", "coder"), ("unload", "coder")]


@pytest.mark.parametrize("failure", ["http", "timeout"])
def test_failed_inference_still_unloads_before_releasing_lock(managed, failure):
    running = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": name} for name in running]})
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            running.clear()
            return httpx.Response(200, json={"done": True})
        running.append(body["model"])
        if failure == "timeout":
            raise httpx.ReadTimeout("inference timed out")
        return httpx.Response(500, json={"error": "failed generation"})

    client = OllamaClient(model="general")
    client._client.close()
    with httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler)) as transport:
        client._client = transport
        with pytest.raises(httpx.HTTPError):
            client._structured("FINAL_RESPONSE_V2", "system", "request", {}, 256)
        assert not running
        assert client._consume_invocation_diagnostics()["model_residency"]["after"]["verified_empty"]


@pytest.mark.parametrize("response", [{}, {"models": "unknown"}, {"models": [None]}, {"models": [{}]}])
def test_invalid_inventory_cannot_be_treated_as_empty(managed, response):
    requests = []
    def handler(request):
        requests.append(request.method)
        return httpx.Response(200, json=response)
    with httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(residency.ModelResidencyError):
            with residency.managed_inference(client, {}, {}):
                pytest.fail("Inference must not start")
    assert requests == ["GET"]


def test_unload_timeout_never_starts_the_next_model(managed, monkeypatch):
    ticks = iter([0, 0, 0, 11])
    monkeypatch.setattr(residency.time, "monotonic", lambda: next(ticks))
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "general"}]})
        return httpx.Response(200, json={"done": True})
    receipt = {}
    with httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(residency.ModelResidencyError, match="did not unload"):
            with residency.managed_inference(client, {}, receipt):
                pytest.fail("A load must not proceed on an unload acknowledgement alone")
    assert not receipt["model_residency"]["before"]["verified_empty"]


def test_profile_endpoint_lock_serializes_disposable_workers(managed):
    with residency.residency_lock("http://localhost:11434"):
        with pytest.raises(RuntimeError):
            with residency.residency_lock("http://127.0.0.1:11434"):
                pytest.fail("Second worker must not enter")
    with residency.residency_lock("http://[::1]:11434"):
        pass


def test_remote_services_are_not_unloaded(managed, monkeypatch):
    settings = AppSettings(ollama_url="https://models.example")
    monkeypatch.setenv(GUI_CONFIG_ENV, settings.model_dump_json())
    assert residency.prepare_local_models(settings) == {}
    diagnostics, request = {}, {"keep_alive": 60}
    with residency.managed_inference(None, request, diagnostics):
        pass
    assert request["keep_alive"] == 60
    assert "model_residency" not in diagnostics


def test_preview_forecast_is_read_only_and_not_execution_authority(managed, monkeypatch):
    settings, plan, replan = cold_plan()
    assert plan.status == "temporarily_blocked"
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"models": [
            {"name": "cached", "size": 6000 * 1024 * 1024, "size_vram": 0}]})
    real_client = httpx.Client
    monkeypatch.setattr(residency.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    forecast = residency.preview_after_unload(settings, plan, replan)
    assert forecast.status == "needs_unload"
    assert forecast.capacity_basis == "after_unload_estimate"
    assert forecast.physical_observation == plan.observation
    assert not forecast.residency["execution_authorized"]
    assert calls == [("GET", "/api/ps")]
    # Speculative reclaimed memory never mutates the physical observation.
    assert replan(plan.observation).status == "temporarily_blocked"


def cold_plan():
    from prometheist.attention_observation import HostResourceMetrics, build_resource_observation, discover_local_execution_resources
    from prometheist.model_admission import plan_task, resource_policy
    settings = AppSettings(selection=ModelSelection(model="general"), task_routing={"enabled": False})
    policy = resource_policy(settings)
    at = datetime.now(timezone.utc)
    metrics = HostResourceMetrics(platform="test", logical_cpu_count=8, cpu_utilization_percent=10,
                                  memory_total_mib=16384, memory_available_mib=5000)
    observation = build_resource_observation(scheduler_cycle=1, captured_at=at, policy=policy, metrics=metrics,
        resources=discover_local_execution_resources(metrics, policy=policy), reservations=[])
    records = {"general:latest": {"installed": {"size": 4096 * 1024 * 1024}, "details": {
        "capabilities": ["completion"], "model_info": {"general.architecture": "qwen3",
            "qwen3.context_length": 32768, "qwen3.block_count": 24, "qwen3.embedding_length": 2048,
            "qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 4}}}}
    def replan(observed):
        return plan_task(settings, "hello", "general", records, observed, at=at)
    return settings, replan(observation), replan


def test_actual_job_unloads_then_rechecks_without_preview_credit(managed, monkeypatch, tmp_path):
    from prometheist import runtime_job, model_admission
    settings, plan, _ = cold_plan()
    order = []
    monkeypatch.setattr(runtime_job, "job_settings", lambda: settings)
    monkeypatch.setattr(residency, "prepare_local_models", lambda settings: order.append("unload") or {"verified_empty": True})
    def actual_plan(*args, **kwargs):
        assert not kwargs.get("preview", False)
        order.append("measure")
        return plan
    monkeypatch.setattr(model_admission, "prepare_task", actual_plan)
    (tmp_path / "job.json").write_text(json.dumps({"action": "chat", "payload": {"text": "hello"}}))
    with pytest.raises(ValueError, match="CPU/RAM"):
        runtime_job.run(tmp_path)
    assert order == ["unload", "measure"]
    recorded = json.loads((tmp_path / "admission.json").read_text())
    assert recorded["status"] == "temporarily_blocked"
    assert recorded["capacity_basis"] == "measured"
