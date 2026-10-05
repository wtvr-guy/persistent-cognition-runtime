import json
import os
import stat

import pytest

from persistent_cognition.content_digest import content_digest
from persistent_cognition.network_consent import (
    NetworkPurpose, consent_path, consent_proposal, grant_consent,
    normalize_destination, require_database_destination, require_destination, revoke_consent,
)


@pytest.fixture(autouse=True)
def private_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    for key in ("PGHOST", "PGHOSTADDR", "PGPORT", "PGSERVICE"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_loopback_services_work_without_external_grant(url):
    require_destination(url, NetworkPurpose.MODEL)


def test_exact_informed_consent_revocation_and_purpose_separation():
    url, purpose = "https://model.example/api", NetworkPurpose.MODEL
    with pytest.raises(PermissionError):
        require_destination(url, purpose)
    with pytest.raises(ValueError):
        grant_consent(url, purpose, accepted_digest="yes")
    digest = content_digest(consent_proposal(url, purpose))
    grant_consent(url, purpose, accepted_digest=digest)
    require_destination(url, purpose)
    for other in ("https://different.example", "http://model.example", "https://model.example:444"):
        with pytest.raises(PermissionError):
            require_destination(other, purpose)
    with pytest.raises(PermissionError):
        require_destination(url, NetworkPurpose.CONNECTIVITY)
    revoke_consent(url, purpose)
    with pytest.raises(PermissionError):
        require_destination(url, purpose)


def test_modified_disclosure_does_not_authorize_send():
    url, purpose = "https://model.example", NetworkPurpose.MODEL
    grant_consent(url, purpose, accepted_digest=content_digest(consent_proposal(url, purpose)))
    data = json.loads(consent_path().read_text())
    data["grants"][0]["disclosure"] = "approved by model"
    consent_path().write_text(json.dumps(data))
    with pytest.raises(PermissionError):
        require_destination(url, purpose)


@pytest.mark.parametrize("dsn", [
    "host=localhost hostaddr=198.51.100.1 dbname=private", "host=localhost,remote.example dbname=private",
    "service=private dbname=private", "postgresql://person:secret@localhost.remote.example/private",
])
def test_database_alternate_routes_cannot_bypass_consent(dsn):
    with pytest.raises(PermissionError):
        require_database_destination(dsn)


def test_database_environment_route_checked(monkeypatch):
    require_database_destination("host=localhost dbname=private")
    monkeypatch.setenv("PGHOSTADDR", "198.51.100.1")
    with pytest.raises(PermissionError):
        require_database_destination("host=localhost dbname=private")


def test_active_connectivity_requires_exact_https_url():
    url, purpose = "https://example.test/health?scope=network", NetworkPurpose.CONNECTIVITY
    with pytest.raises(ValueError):
        normalize_destination("http://example.test", purpose)
    with pytest.raises(PermissionError):
        require_destination("https://localhost/", purpose)
    grant_consent(url, purpose, accepted_digest=content_digest(consent_proposal(url, purpose)))
    require_destination(url, purpose)
    with pytest.raises(PermissionError):
        require_destination("https://example.test/different", purpose)


def test_all_model_transports_check_actual_destination_before_io(monkeypatch):
    import httpx
    from persistent_cognition.llm import OllamaClient
    from persistent_cognition.ollama_runtime import OllamaRuntimeProbe
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, json={}))
    with httpx.Client(base_url="https://remote.example", transport=transport) as http:
        client = OllamaClient(base_url="http://localhost:11434", model="model:test")
        client._client.close()
        client._client = http
        with pytest.raises(PermissionError):
            client._perform_ollama_request(kind="test", request_path="/api/chat", request_json={}, transport="test")
        client.runtime_snapshot()
        assert not OllamaRuntimeProbe(client=http, model="model:test").capture().probe_ok
    assert calls == []


def test_connectivity_does_not_follow_redirects_or_send_inventory(monkeypatch):
    import httpx
    from persistent_cognition.network_consent import check_connectivity
    purpose, url = NetworkPurpose.CONNECTIVITY, "https://endpoint.example/health"
    grant_consent(url, purpose, accepted_digest=content_digest(consent_proposal(url, purpose)))
    client_type = httpx.Client
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://not-consented.example"})
    def factory(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return client_type(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", factory)
    assert check_connectivity(url)["http_status"] == 302
    assert len(calls) == 1 and calls[0].method == "HEAD" and calls[0].content == b""


def test_concurrent_grant_and_revocation_cannot_restore_an_old_grant():
    from concurrent.futures import ThreadPoolExecutor
    purpose = NetworkPurpose.MODEL
    old = "https://revoked.example"
    grant_consent(old, purpose, accepted_digest=content_digest(consent_proposal(old, purpose)))
    urls = [f"https://destination-{i}.example" for i in range(12)]
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(grant_consent, url, purpose,
                   accepted_digest=content_digest(consent_proposal(url, purpose))) for url in urls]
        futures.append(pool.submit(revoke_consent, old, purpose))
        for future in futures:
            future.result()
    with pytest.raises(PermissionError):
        require_destination(old, purpose)
    for url in urls:
        require_destination(url, purpose)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes; Windows depends on the private directory ACL")
def test_revocation_keeps_remaining_grants_owner_only():
    purpose = NetworkPurpose.MODEL
    urls = ("https://revoked.example", "https://retained.example")
    for url in urls:
        grant_consent(url, purpose, accepted_digest=content_digest(consent_proposal(url, purpose)))
    assert stat.S_IMODE(consent_path().stat().st_mode) == 0o600
    revoke_consent(urls[0], purpose)
    assert stat.S_IMODE(consent_path().stat().st_mode) == 0o600
    require_destination(urls[1], purpose)
