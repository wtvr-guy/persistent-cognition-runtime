from datetime import datetime, timedelta, timezone

import httpx
import pytest

from persistent_cognition.attention_observation import HostResourceMetrics, build_resource_observation, discover_local_execution_resources
from persistent_cognition.runtime_settings import AppSettings, ModelSelection, SpecialistModel, TaskRoutingSettings
from persistent_cognition.model_admission import assess_model, classify_task, plan_task, read_model_evidence, resource_policy

AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def observation(settings=None, *, free=12288, total=16384, cpu=10, healthy=True):
    policy = resource_policy(settings or AppSettings())
    metrics = HostResourceMetrics(platform="test", logical_cpu_count=8, cpu_utilization_percent=cpu,
                                  memory_total_mib=total, memory_available_mib=free) if healthy else None
    return build_resource_observation(scheduler_cycle=1, captured_at=AT, policy=policy, metrics=metrics,
        resources=discover_local_execution_resources(metrics, policy=policy), reservations=[])


def evidence(name="default", *, weights=2048):
    return {"installed": {"name": name, "size": weights * 1024 * 1024, "digest": "digest-" + name},
            "details": {"capabilities": ["completion"], "model_info": {"general.architecture": "qwen3",
                "qwen3.context_length": 32768, "qwen3.block_count": 24, "qwen3.embedding_length": 2048,
                "qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 4}}}


def starcoder_evidence(name="coder", *, weights=2048):
    # Field shapes mirror a real installed coding specialist ("dolphincoder:7b"
    # reports exactly these model_info values from `ollama show`).
    return {"installed": {"name": name, "size": weights * 1024 * 1024, "digest": "digest-" + name},
            "details": {"capabilities": ["completion"], "model_info": {"general.architecture": "starcoder2",
                "starcoder2.context_length": 16384, "starcoder2.block_count": 32, "starcoder2.embedding_length": 4608,
                "starcoder2.attention.head_count": 36, "starcoder2.attention.head_count_kv": 4}}}


def check(settings=None, record=None, host=None, *, at=AT, selection=None, task="general"):
    settings = settings or AppSettings(selection=ModelSelection(model="default"))
    record = record or evidence()
    return assess_model(settings, selection or settings.selection, record["installed"], record["details"],
                        host or observation(settings), at=at, task=task)


def test_pure_replay_explicit_context_and_cost_components():
    first = check()
    assert first == check()
    assert first.status == "eligible"
    assert first.required_memory_mib == 3456
    assert first.components == {"weights_mib": 2048, "kv_mib": 384, "buffers_mib": 512, "worker_mib": 512}
    assert first.effective_selection.parameters == {"num_ctx": 4096, "num_gpu": 0, "num_batch": 128, "num_thread": 1, "keep_alive": 0}
    larger = check(selection=ModelSelection(model="default", parameters={"num_ctx": 32768}))
    assert larger.required_memory_mib > first.required_memory_mib
    assert larger.effective_selection.parameters["num_ctx"] == 32768
    assert check(selection=ModelSelection(model="default", parameters={"num_ctx": 65536})).status == "unsupported"


def test_host_changes_unknowns_and_gpu_are_not_optimistically_admitted():
    assert check(host=observation(free=2500)).status == "temporarily_blocked"
    assert check(host=observation(cpu=99)).status == "temporarily_blocked"
    assert check(host=observation(free=4000, total=4000)).status == "unsupported"
    assert check(host=observation(healthy=False)).status == "unverified"
    assert check(at=AT + timedelta(minutes=1)).status == "unverified"
    unknown = evidence()
    unknown["details"]["model_info"]["general.architecture"] = "unregistered-hybrid"
    assert check(record=unknown).status == "unverified"
    incomplete = evidence()
    del incomplete["details"]["model_info"]["qwen3.attention.head_count_kv"]
    assert check(record=incomplete).status == "unverified"
    assert check(selection=ModelSelection(model="default", parameters={"num_gpu": -1})).status == "unverified"
    assert check(selection=ModelSelection(model="default", parameters={"num_batch": 512})).status == "unverified"
    non_text = evidence()
    non_text["details"]["capabilities"] = ["embedding"]
    assert check(record=non_text).status == "unsupported"
    # Modality support is distinct from having an image transport in this pipeline.
    visual = evidence()
    visual["details"]["capabilities"].append("vision")
    assert check(record=visual, task="vision").status == "unsupported"


def test_starcoder2_coding_specialists_are_registered_and_eligible():
    # Regression test: a "starcoder2"-family model (e.g. the "dolphincoder"
    # coding specialist) must be estimable, not stuck "unverified" forever.
    result = check(record=starcoder_evidence(), task="coding")
    assert result.status == "eligible"
    assert result.required_memory_mib == 3584
    assert result.components == {"weights_mib": 2048, "kv_mib": 512, "buffers_mib": 512, "worker_mib": 512}


def configured(*names, **kwargs):
    return AppSettings(selection=ModelSelection(model="default"), task_routing=TaskRoutingSettings(
        specialists=[SpecialistModel(selection=ModelSelection(model=name), tasks=["coding"]) for name in names]), **kwargs)


def test_installed_specialist_order_is_stable_and_stage_pin_wins():
    records = {name + ":latest": evidence(name) for name in ("default", "alpha", "zeta")}
    a, b = configured("zeta", "alpha"), configured("alpha", "zeta")
    one = plan_task(a, "/code fix this", "auto", records, observation(), at=AT)
    two = plan_task(b, "/code fix this", "auto", dict(reversed(list(records.items()))), observation(), at=AT)
    assert one.status == two.status == "eligible"
    assert one.stages == two.stages
    assert one.stages["V2_RESPOND"].model == "alpha"
    assert one.stages["V2_EVIDENCE_POLICY"].model == "default"
    pinned = configured("alpha", routes={"V2_RESPOND": ModelSelection(model="zeta")})
    three = plan_task(pinned, "```python\nx=1\n```", "auto", records, observation(), at=AT)
    assert three.stages["V2_RESPOND"].model == "zeta"
    assert "precedence" in three.route_reason


def test_registered_starcoder2_specialist_is_actually_chosen_for_coding():
    # End-to-end regression for the reported bug: registering an installed
    # starcoder2-family specialist (e.g. "dolphincoder") for the coding task
    # must route chat to it instead of silently keeping the default model.
    settings = configured("dolphincoder")
    records = {"default:latest": evidence(), "dolphincoder:latest": starcoder_evidence("dolphincoder")}
    plan = plan_task(settings, "/code write a function", "auto", records, observation(), at=AT)
    assert plan.status == "eligible"
    assert plan.stages["V2_RESPOND"].model == "dolphincoder"
    assert not plan.routing_exclusions


def test_default_fallback_search_offer_and_individual_route_budget():
    settings = configured("missing", "large")
    records = {"default:latest": evidence(), "large:latest": evidence("large", weights=10000)}
    review = plan_task(settings, "write a function", "coding", records, observation(), at=AT)
    assert review.status == "needs_choice"
    assert review.fallback_steps == ["catalog_search", "openai_review", "local_default"]
    assert review.openai_offer
    plan = plan_task(settings, "write a function", "coding", records, observation(), at=AT, fallback="default")
    assert plan.status == "eligible"
    assert plan.stages["V2_RESPOND"].model == "default"
    assert "Combined" not in plan.reasons[0]
    assert "CPU/RAM availability" in plan.routing_exclusions["large"]
    assert "not installed" in plan.routing_exclusions["missing"]
    assert not plan.openai_offer
    blocked = plan_task(settings, "write a function", "coding", records, observation(free=2200), at=AT, fallback="default")
    assert blocked.status == "temporarily_blocked" and blocked.openai_offer
    assert blocked.specialist_offer


def test_coding_routes_fit_sequentially_without_combined_residency():
    settings = configured("coder")
    records = {"default:latest": evidence(), "coder:latest": starcoder_evidence(weights=4096)}
    plan = plan_task(settings, "write a function", "coding", records, observation(free=10000), at=AT)
    assert plan.status == "eligible"
    assert plan.stages["V2_RESPOND"].model == "coder"
    assert plan.stages["V2_EVIDENCE_POLICY"].model == "default"
    # 3,456 + 6,144 would fail this host's envelope. Only 6,144 is reserved.
    assert plan.required_memory_mib == 6144
    assert not plan.routing_exclusions
    assert all(route.parameters["keep_alive"] == 0 for route in plan.stages.values())


def test_saved_keep_alive_cannot_defeat_sequential_admission():
    result = check(selection=ModelSelection(model="default", parameters={"keep_alive": -1}))
    assert result.status == "eligible"
    assert result.effective_selection.parameters["keep_alive"] == 0


def test_remote_host_is_not_measured_as_local_and_never_selected_implicitly():
    remote = AppSettings(ollama_url="https://models.example", selection=ModelSelection(model="remote"))
    assert check(settings=remote).status == "remote"
    assert check(settings=remote).required_memory_mib == 512
    cloud = AppSettings(selection=ModelSelection(provider="openai", model="gpt-5"))
    assert check(settings=cloud).status == "remote"
    with pytest.raises(ValueError, match="local Ollama"):
        SpecialistModel(selection=cloud.selection, tasks=["coding"])
    assert classify_task("can you help", "auto")[0] == "general"
    assert classify_task("/code solve this", "general")[0] == "general"
    assert classify_task(" " * 32768 + "/code: fix", "auto")[0] == "coding"
    assert classify_task("/codebase", "auto")[0] == "general"
    assert classify_task("/coding2", "auto")[0] == "general"


def test_missing_model_discovery_never_downloads(monkeypatch):
    from persistent_cognition import model_catalog
    requests = []
    monkeypatch.setattr(model_catalog, "ollama_request", lambda *args, **kwargs: requests.append((args, kwargs)) or {"models": []})
    result = read_model_evidence(AppSettings(), [ModelSelection(model="not-installed")])
    assert result["not-installed:latest"]["installed"] is None
    assert [args[1] for args, _ in requests] == ["/api/tags"]


def test_specialist_catalog_is_fixed_bounded_sorted_and_consent_gated(monkeypatch, tmp_path):
    from persistent_cognition import model_catalog
    from persistent_cognition.network_consent import NetworkPurpose, consent_proposal, grant_consent
    from persistent_cognition.content_digest import content_digest
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    page = '''<a href="/library/zeta"><p>Coding assistant</p><span class="rounded-md">3b</span></a>
              <a href="/library/alpha"><p>Code generation</p><span class="rounded-md">1.5b</span></a>
              <a href="/library/remote"><p>Coder</p><span class="rounded-md">cloud</span></a>
              <a href="/library/irrelevant"><p>Weather model</p></a>'''
    requests = []
    real_client = httpx.Client
    def handle(request):
        requests.append(request)
        return httpx.Response(200, text=page)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(PermissionError):
        model_catalog.search_specialists("coding")
    assert requests == []
    proposal = consent_proposal(model_catalog.CATALOG_ORIGIN, NetworkPurpose.MODEL_CATALOG)
    grant_consent(model_catalog.CATALOG_ORIGIN, NetworkPurpose.MODEL_CATALOG, accepted_digest=content_digest(proposal), root=tmp_path)
    result = model_catalog.search_specialists("coding")
    assert [item["name"] for item in result["models"]] == ["alpha", "zeta"]
    # Coding fans out across every registered keyword ("coder", "code") and
    # sort order ("popular", "newest") -- ollama.com has no pagination param
    # that actually works, so this fixed fan-out is the only way to see more
    # than one page's worth of results. "o=popular" is omitted like an empty
    # "q=": ollama.com 303-redirects it too, since "popular" is the default.
    assert [str(request.url) for request in requests] == [
        "https://ollama.com/search?q=coder",
        "https://ollama.com/search?q=coder&o=newest",
        "https://ollama.com/search?q=code",
        "https://ollama.com/search?q=code&o=newest",
    ]
    assert result["openai_offer"]
    assert all(request.method == "GET" for request in requests)
    visual = model_catalog.search_specialists("vision")
    assert visual["models"] == []  # cannot trust the remote filter alone
    # An empty keyword is omitted rather than sent as "q=", since ollama.com
    # redirects (303) an explicit-but-empty q instead of returning results.
    assert [str(request.url) for request in requests[-2:]] == [
        "https://ollama.com/search?c=vision",
        "https://ollama.com/search?c=vision&o=newest",
    ]
    # New browsing categories are independent of chat-routing TASKS: they do
    # not require an implemented chat task and never offer an OpenAI fallback.
    for category in ("tools", "thinking", "embedding"):
        extra = model_catalog.search_specialists(category)
        assert extra["models"] == []
        assert not extra["openai_offer"]
    base = model_catalog.search_specialists("base")
    # "base" has no capability filter, so client-side filtering only removes
    # cloud-only/no-size entries; real relevance to "base"/"pretrained"/
    # "foundation" depends on ollama.com's own search, which this fixture
    # (a fixed page regardless of query) cannot exercise.
    assert [item["name"] for item in base["models"]] == ["alpha", "irrelevant", "zeta"]
    assert all("not a verified base/foundation flag" in item["verification"] for item in base["models"])
    with pytest.raises(ValueError, match="Unregistered catalog category"):
        model_catalog.search_specialists("unknown")


def test_capacity_filter_is_dynamic_and_never_excludes_on_missing_evidence(monkeypatch):
    from persistent_cognition import model_catalog
    settings = AppSettings(selection=ModelSelection(model="default"))
    records = [
        {"name": "small", "advertised_sizes": ["1b"], "advertised_capabilities": []},
        {"name": "huge", "advertised_sizes": ["4000b"], "advertised_capabilities": []},
        {"name": "no-size-reported", "advertised_sizes": [], "advertised_capabilities": []},
    ]
    # No installed models to calibrate from yet: nothing is excluded, and the
    # UI is told filtering is not active rather than silently hiding results.
    monkeypatch.setattr(model_catalog, "installed_models", lambda url: [])
    kept, info = model_catalog._apply_capacity_filter([dict(r) for r in records], settings)
    assert [item["name"] for item in kept] == ["small", "huge", "no-size-reported"]
    assert info["calibration_sample_count"] == 0
    assert all(item["resource_check"] == "unavailable" for item in kept)

    # Calibrate from a real installed model's own reported size/parameter
    # count (here: 1 billion parameters occupying 1024 MiB, i.e. 1024
    # MiB/billion): architecture- and capability-agnostic by construction.
    monkeypatch.setattr(model_catalog, "installed_models", lambda url: [
        {"name": "calibrator", "size": 1024 * 1024 * 1024, "details": {"parameter_size": "1.0B"}},
    ])
    monkeypatch.setattr("persistent_cognition.model_admission.capture_observation", lambda settings: observation(settings, free=8192, total=16384))
    kept, info = model_catalog._apply_capacity_filter([dict(r) for r in records], settings)
    assert info["calibration_sample_count"] == 1
    assert info["calibration_mib_per_billion_parameters"] == pytest.approx(1024, rel=0.01)
    names = {item["name"]: item for item in kept}
    assert "huge" not in names  # 4000B parameters is never plausible on 16 GiB
    assert names["small"]["resource_check"] == "fits"
    assert names["no-size-reported"]["resource_check"] == "unknown_size"  # absence of evidence never excludes


def test_installed_weight_ratio_ignores_architecture_and_capability(monkeypatch):
    from persistent_cognition import model_catalog
    monkeypatch.setattr(model_catalog, "installed_models", lambda url: [
        {"name": "a", "size": 500 * 1024 * 1024, "details": {"parameter_size": "0.5B"}},  # unrecognized arch/capability irrelevant here
        {"name": "b", "size": None, "details": {"parameter_size": "3B"}},  # no size: ignored, not a zero
        {"name": "c", "size": 100, "details": {"parameter_size": ""}},  # unparsed size label: ignored
    ])
    ratio, count = model_catalog._installed_weight_ratio("http://localhost:11434")
    assert count == 1
    assert ratio == pytest.approx(1000, rel=0.01)  # 500 MiB / 0.5B parameters


def test_worker_records_choice_without_starting_cognition(monkeypatch, tmp_path):
    import json
    from contextlib import contextmanager, nullcontext
    from persistent_cognition import advisory_lock, db, runtime_job, model_admission, model_residency
    settings = configured()
    plan = plan_task(settings, "code task", "coding", {"default:latest": evidence()}, observation(), at=AT)
    assert plan.status == "needs_choice"
    (tmp_path / "job.json").write_text(json.dumps({"action":"chat", "payload":{"text":"code task", "task":"coding"}}))
    monkeypatch.setattr(runtime_job, "job_settings", lambda: settings)
    monkeypatch.setattr(model_residency, "prepare_local_models", lambda settings: {})
    monkeypatch.setattr(model_admission, "prepare_task", lambda *args, **kwargs: plan)
    connection = object()
    ownership = []
    monkeypatch.setattr(db, "get_connection", lambda: nullcontext(connection))
    @contextmanager
    def own_namespace(conn, scheduler_key):
        assert conn is connection and scheduler_key == "gui-chat"
        ownership.append("acquired")
        try:
            yield
        finally:
            ownership.append("released")
    monkeypatch.setattr(advisory_lock, "scheduler_ownership", own_namespace)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No cognitive work may start while a fallback choice is required")
    monkeypatch.setattr(runtime_job, "_run_chat_job", forbidden)
    with pytest.raises(ValueError, match="Choose a fallback"):
        runtime_job.run(tmp_path)
    assert ownership == ["acquired", "released"]
    assert json.loads((tmp_path / "admission.json").read_text())["status"] == "needs_choice"
    assert not (tmp_path / "effective-settings.json").exists()


def test_inventory_outage_yields_reviewable_fallback_instead_of_download(monkeypatch):
    from persistent_cognition import model_catalog
    def unavailable(endpoint):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(model_catalog, "installed_models", unavailable)
    settings = configured()
    records = read_model_evidence(settings, [settings.selection])
    plan = plan_task(settings, "hello", "general", records, observation(), at=AT)
    assert plan.status == "needs_choice"
    assert plan.assessments[0].status == "unverified"
    assert plan.openai_offer
