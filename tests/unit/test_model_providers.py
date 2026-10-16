import json

import httpx
import pytest

from persistent_cognition.runtime_settings import AppSettings, GUI_CONFIG_ENV, ModelSelection, load_settings, save_settings
from persistent_cognition.model_parameters import parameter_catalog
from persistent_cognition.native_policy import native_resource_safety_policy
from persistent_cognition.openai_transport import response_request, strict_schema


def test_typed_settings_reject_unknown_routes_and_unsafe_parameters(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        ModelSelection(parameters={"shell": "cmd"})
    with pytest.raises(ValueError, match="finite"):
        ModelSelection(parameters={"temperature": float("nan")})
    with pytest.raises(ValueError):
        ModelSelection(model="https://evil.example/model")
    with pytest.raises(ValueError, match="cloud aliases"):
        ModelSelection(model="gpt-oss:120b-cloud")
    with pytest.raises(ValueError, match="stage"):
        AppSettings(routes={"invented": ModelSelection()})
    with pytest.raises(ValueError, match="evidence"):
        AppSettings(memory={"max_item_bytes": 90000})
    with pytest.raises(ValueError):
        AppSettings(ollama_url="https://secret:password@example.com")
    settings = AppSettings(selection=ModelSelection(model="qwen3:4b", parameters={"seed": 42}))
    save_settings(tmp_path, settings)
    assert load_settings(tmp_path) == settings
    assert "OPENAI_API_KEY" not in settings.worker_environment()


def test_model_specific_controls_and_reasoning_requests():
    controls = parameter_catalog("ollama", "small", {"capabilities": ["completion"],
                                                    "model_info": {"small.context_length": 8192}})
    assert "think" not in controls
    assert controls["num_ctx"]["maximum"] == 8192
    cloud = ModelSelection(provider="openai", model="gpt-5", parameters={"reasoning_effort": "low", "max_output_tokens": 8192})
    request = response_request(cloud, kind="FINAL_RESPONSE_V2", system="policy", user="now", evidence="untrusted",
                               schema={"type": "object", "properties": {"answer": {"type": "string", "default": ""}}}, max_tokens=256)
    assert request["store"] is False
    assert request["max_output_tokens"] == 8192
    assert "temperature" not in request
    assert request["input"] == [{"role": "user", "content": "untrusted"}, {"role": "user", "content": "now"}]
    schema = request["text"]["format"]["schema"]
    assert schema["required"] == ["answer"] and schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["answer"]
    with pytest.raises(ValueError, match="open mapping"):
        strict_schema({"type": "object", "additionalProperties": {"type": "string"}})


def test_routed_resource_policy_never_reuses_another_models_credit(monkeypatch):
    from persistent_cognition.model_runtime import configured_runtime_probe
    local = AppSettings(selection=ModelSelection(model="small"), routes={"V2_RESPOND": ModelSelection(model="large")})
    monkeypatch.setenv(GUI_CONFIG_ENV, local.model_dump_json())
    assert not configured_runtime_probe().capture().probe_ok
    assert native_resource_safety_policy().default_llm_process_memory_mib == 3072
    cloud = AppSettings(selection=ModelSelection(provider="openai", model="gpt-5"))
    monkeypatch.setenv(GUI_CONFIG_ENV, cloud.model_dump_json())
    policy = native_resource_safety_policy()
    assert policy.default_llm_process_memory_mib == policy.default_process_memory_mib == 512
    assert configured_runtime_probe().capture().resident is False


def test_final_sampler_overrides_leave_control_calls_deterministic(monkeypatch):
    from persistent_cognition.llm import OllamaClient
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"answer":"ok"}'}})
    selection = ModelSelection(model="local-model", parameters={"temperature": 1.2, "top_p": .8,
                                                               "num_ctx": 8192, "num_predict": 777})
    client = OllamaClient(selection=selection)
    client._client.close()
    client._client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    client._structured("V2_RESPONSE_POLICY", "system", "user", {}, 256)
    client._structured("FINAL_RESPONSE_V2", "system", "user", {}, 256)
    assert requests[0]["options"] == {"temperature": 0.0, "num_predict": 256, "num_ctx": 8192}
    assert requests[1]["options"] == {"temperature": 1.2, "num_predict": 777, "num_ctx": 8192, "top_p": .8}


def test_openai_requires_current_consent_and_preserves_bounded_artifacts(monkeypatch, tmp_path):
    from persistent_cognition.llm import OllamaClient
    from persistent_cognition.network_consent import NetworkPurpose, consent_proposal, grant_consent, revoke_consent
    from persistent_cognition.content_digest import content_digest
    from persistent_cognition.openai_transport import OPENAI_ORIGIN
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-persist")
    requests = []
    real_client = httpx.Client
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "response-id", "status": "completed", "model": "gpt-5",
            "output": [{"type": "reasoning", "summary": "hidden"}, {"type": "message", "content": [{"type": "output_text", "text": '{"answer":"ok"}'}]}], "usage": {"output_tokens": 10}})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    client = OllamaClient(selection=ModelSelection(provider="openai", model="gpt-5"))
    with pytest.raises(PermissionError):
        client._structured("FINAL_RESPONSE_V2", "system", "user", {}, 256)
    assert not requests
    proposal = consent_proposal(OPENAI_ORIGIN, NetworkPurpose.MODEL)
    grant_consent(OPENAI_ORIGIN, NetworkPurpose.MODEL, accepted_digest=content_digest(proposal))
    assert client._structured("FINAL_RESPONSE_V2", "system", "user", {}, 256) == '{"answer":"ok"}'
    diagnostics = client._consume_invocation_diagnostics()
    encoded = json.dumps(diagnostics)
    assert "test-secret-never-persist" not in encoded and "hidden" not in encoded
    assert requests[0].headers["Authorization"] == "Bearer test-secret-never-persist"
    assert requests[0].url == "https://api.openai.com/v1/responses"
    monkeypatch.setenv("PCR_MAX_MODEL_INPUT_BYTES", "100")
    from persistent_cognition.model_evidence_budget import ModelEvidenceBudgetExceeded
    with pytest.raises(ModelEvidenceBudgetExceeded, match="byte budget"):
        client._structured("FINAL_RESPONSE_V2", "system", "x" * 200, {}, 256)
    assert len(requests) == 1
    revoke_consent(OPENAI_ORIGIN, NetworkPurpose.MODEL)
    with pytest.raises(PermissionError):
        client._structured("FINAL_RESPONSE_V2", "system", "user", {}, 256)
    assert len(requests) == 1


def test_catalog_parser_does_not_turn_links_into_external_download_targets():
    from persistent_cognition.model_catalog import CatalogParser
    parser = CatalogParser()
    parser.feed('<a href="/library/qwen3">x</a><a href="/library/qwen3">x</a><a href="https://evil.example/library/evil">e</a><a href="/library/name/tags">t</a><a href="/library/foo-cloud">cloud</a>')
    assert parser.names == ["qwen3"]


def test_installed_huggingface_model_names_are_valid_but_download_scopes_stay_exact(monkeypatch):
    from persistent_cognition.model_catalog import pull_model
    name = "hf.co/example/small-model-GGUF:Q4_K_M"
    assert ModelSelection(model=name).model == name
    with pytest.raises(ValueError, match="registry names"):
        pull_model("http://localhost:11434", name, lambda value: None)


def test_measured_cost_and_remote_ollama_capacity_are_local_host_scoped(monkeypatch):
    from persistent_cognition.model_runtime import configured_runtime_probe
    local = AppSettings(selection=ModelSelection(model="large"))
    floor = 8704
    monkeypatch.setenv(GUI_CONFIG_ENV, local.model_dump_json())
    monkeypatch.setenv("PCR_GUI_MODEL_MEMORY_FLOOR_MIB", str(floor))
    assert native_resource_safety_policy().default_llm_process_memory_mib == floor
    remote = AppSettings(ollama_url="https://192.0.2.1:11434", selection=local.selection)
    monkeypatch.setenv(GUI_CONFIG_ENV, remote.model_dump_json())
    assert not configured_runtime_probe().capture().resident
    assert native_resource_safety_policy().default_llm_process_memory_mib == 512


def test_reasoning_final_overrides_never_change_control_budgets():
    from persistent_cognition.openai_transport import DEFAULT_REASONING_BUDGET
    selection = ModelSelection(provider="openai", model="gpt-5", parameters={"max_output_tokens":256, "reasoning_effort":"low", "verbosity":"high"})
    args = {"system":"policy", "user":"task", "evidence":"", "schema":{"type":"object", "properties":{}}, "max_tokens":128}
    control = response_request(selection, kind="V2_RESPONSE_POLICY", **args)
    final = response_request(selection, kind="FINAL_RESPONSE_V2", **args)
    assert control["max_output_tokens"] == DEFAULT_REASONING_BUDGET
    assert "reasoning" not in control and "verbosity" not in control["text"]
    assert final["max_output_tokens"] == 256
    assert final["reasoning"] == {"effort":"low"} and final["text"]["verbosity"] == "high"


def test_invocation_artifact_records_real_provider(monkeypatch):
    from uuid import uuid4
    from persistent_cognition import artifact_journal, llm_artifact_store
    captured = {}
    def write(**kwargs):
        captured.update(kwargs)
        return kwargs
    monkeypatch.setattr(artifact_journal, "write_interaction_artifact", write)
    llm_artifact_store.write_llm_invocation(interaction_id=uuid4(),conversation_id=uuid4(),correlation_id=uuid4(),
        task_id=uuid4(),assignment_id=uuid4(),stage="V2_RESPOND",claim_id=uuid4(),invocation_index=1,
        kind="FINAL_RESPONSE_V2",model="gpt-5",base_url="https://api.openai.com/v1",system_prompt="s",user_prompt="u",
        schema={},max_tokens=256,temperature=None,output="{}",error_type=None,error_message=None,provider="openai")
    assert captured["producer"] == "percept_response_v2/openai"
    assert captured["payload"]["provider"] == "openai"
