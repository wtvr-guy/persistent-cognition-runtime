"""Operator-owned outbound consent. Observations and models cannot grant access."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

from pydantic import Field

from prometheist.artifact_journal import artifact_root
from prometheist.content_digest import content_digest
from prometheist.percept_context import FrozenRecord
from prometheist.operator_state import policy_lock, write_private_policy

CONSENT_VERSION = "network-consent/v1"
CONNECTIVITY_TIMEOUT_SECONDS = 5


class NetworkPurpose(str, Enum):
    MODEL = "model_inference"
    DATABASE = "database_storage"
    CONNECTIVITY = "connectivity_check"
    MODEL_CATALOG = "model_catalog"
    MODEL_DOWNLOAD = "model_download"


DISCLOSURES = {
    NetworkPurpose.MODEL_CATALOG: "Sends your model search terms and IP address to ollama.com. No chat, personal evidence, host inventory or API key is included.",
    NetworkPurpose.MODEL_DOWNLOAD: "Asks your configured Ollama service to download a selected model from the Ollama registry and its storage/CDN providers. Those services receive the model name, network address and request timing. The local service writes model weights to its own storage. No chat or personal evidence is included. The Ollama daemon, not Prometheist, performs the transfer and may finish an in-flight transfer after cancellation.",
    NetworkPurpose.MODEL: "Sends prompts, retrieved personal evidence, and model requests to this model endpoint. The endpoint operator can receive and retain them.",
    NetworkPurpose.DATABASE: "Sends database credentials and private canonical/derived records to this PostgreSQL endpoint. Its operator can receive and retain them.",
    NetworkPurpose.CONNECTIVITY: "Sends an HTTPS HEAD request with no body or personal inventory to this exact URL. The destination and DNS resolver can observe your IP address, hostname lookup, request path and timing. No redirects are followed.",
}


class NetworkGrant(FrozenRecord):
    schema_version: Literal["network-consent/v1"] = CONSENT_VERSION
    purpose: NetworkPurpose
    destination: str
    disclosure: str
    disclosure_sha256: str
    granted_at: datetime
    granted_by: str = "LOCAL_OPERATOR"


class NetworkConsent(FrozenRecord):
    schema_version: Literal["network-consent/v1"] = CONSENT_VERSION
    grants: tuple[NetworkGrant, ...] = Field(default=())


def _loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalize_destination(url: str, purpose: NetworkPurpose) -> str:
    parsed = urlsplit(url)
    allowed = {"postgresql"} if purpose is NetworkPurpose.DATABASE else {"http", "https"}
    if parsed.scheme not in allowed or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("consent needs an explicit endpoint without credentials or fragment")
    host = parsed.hostname.casefold()
    port = parsed.port or (5432 if parsed.scheme == "postgresql" else 443 if parsed.scheme == "https" else 80)
    if purpose is NetworkPurpose.CONNECTIVITY and parsed.scheme != "https":
        raise ValueError("active internet checks require HTTPS")
    host = f"[{host}]" if ":" in host else host
    origin = f"{parsed.scheme}://{host}:{port}"
    return origin + ((parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
                     if purpose is NetworkPurpose.CONNECTIVITY else "")


def consent_path(root: Path | None = None) -> Path:
    return (root or artifact_root()) / "operator" / "network-consent.json"


def load_consent(root: Path | None = None) -> NetworkConsent:
    path = consent_path(root)
    if not path.exists():
        return NetworkConsent()
    result = NetworkConsent.model_validate_json(path.read_text(encoding="utf-8"))
    if result.schema_version != CONSENT_VERSION:
        raise ValueError("unsupported network consent version")
    return result


def consent_proposal(url: str, purpose: NetworkPurpose) -> dict:
    return {"schema_version": CONSENT_VERSION, "purpose": purpose.value,
            "destination": normalize_destination(url, purpose), "disclosure": DISCLOSURES[purpose],
            "duration": "until explicitly revoked", "granted_by": "LOCAL_OPERATOR"}


def grant_consent(url: str, purpose: NetworkPurpose, *, accepted_digest: str, root: Path | None = None):
    proposal = consent_proposal(url, purpose)
    if accepted_digest != content_digest(proposal):
        raise ValueError("consent requires acceptance of the exact destination and disclosure")
    grant = NetworkGrant(purpose=purpose, destination=proposal["destination"],
        disclosure=proposal["disclosure"], disclosure_sha256=accepted_digest, granted_at=datetime.now(timezone.utc))
    with policy_lock(consent_path(root)):
        prior = load_consent(root)
        grants = [g for g in prior.grants if (g.destination, g.purpose) != (grant.destination, purpose)]
        grants.append(grant)
        write_private_policy(consent_path(root), NetworkConsent(grants=tuple(grants)).model_dump(mode="json"))
    return grant


def revoke_consent(url: str, purpose: NetworkPurpose, *, root: Path | None = None):
    destination = normalize_destination(url, purpose)
    with policy_lock(consent_path(root)):
        prior = load_consent(root)
        kept = tuple(g for g in prior.grants if (g.destination, g.purpose) != (destination, purpose))
        write_private_policy(consent_path(root), NetworkConsent(grants=kept).model_dump(mode="json"))


def require_destination(url: str, purpose: NetworkPurpose, *, root: Path | None = None):
    destination = normalize_destination(url, purpose)
    # Active checks always require informed consent, including a selected local
    # endpoint; a generic "online" setting cannot silently authorize probing.
    if purpose in (NetworkPurpose.MODEL, NetworkPurpose.DATABASE) and _loopback(urlsplit(destination).hostname):
        return
    proposal = consent_proposal(url, purpose)
    for grant in load_consent(root).grants:
        if (grant.schema_version == CONSENT_VERSION and grant.granted_by == "LOCAL_OPERATOR"
                and grant.purpose is purpose and grant.destination == destination
                and grant.disclosure_sha256 == content_digest(proposal)
                and grant.disclosure == DISCLOSURES[purpose]):
            return
    raise PermissionError(f"No informed outbound consent for {purpose.value} to {destination}")


def require_database_destination(conninfo: str, *, root: Path | None = None):
    from psycopg.conninfo import conninfo_to_dict
    params = conninfo_to_dict(conninfo)
    # Service files can hide endpoints; require an explicit host/hostaddr instead.
    if params.get("service") or os.environ.get("PGSERVICE"):
        raise PermissionError("database service indirection is not an explicit consent destination")
    hosts = params.get("host", os.environ.get("PGHOST", ""))
    addresses = params.get("hostaddr", os.environ.get("PGHOSTADDR", ""))
    ports = params.get("port", os.environ.get("PGPORT", "5432")).split(",")
    for field in (hosts, addresses):
        for index, host in enumerate(field.split(",")):
            if not host or host.startswith(("/", "@")):
                continue  # local Unix-domain socket
            port = ports[index] if len(ports) > 1 else ports[0]
            formatted = f"[{host}]" if ":" in host else host
            require_destination(f"postgresql://{formatted}:{port}", NetworkPurpose.DATABASE, root=root)


def check_connectivity(url: str):
    import httpx
    require_destination(url, NetworkPurpose.CONNECTIVITY)
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=CONNECTIVITY_TIMEOUT_SECONDS) as client:
        response = client.head(url)
    return {"evidence": "CONSENTED_ENDPOINT_REACHED", "destination": normalize_destination(url, NetworkPurpose.CONNECTIVITY),
            "http_status": response.status_code, "observed_at": datetime.now(timezone.utc).isoformat(),
            "meaning": "This endpoint was reached; it does not establish access to every internet service."}
