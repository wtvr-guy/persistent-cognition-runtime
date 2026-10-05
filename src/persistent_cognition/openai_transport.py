"""Stateless Responses transport with the same validation and evidence bounds."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import time

import httpx

from persistent_cognition.model_parameters import openai_reasoning
from persistent_cognition.model_evidence_budget import validate_model_input
from persistent_cognition.network_consent import NetworkPurpose, require_destination

OPENAI_ORIGIN = "https://api.openai.com"
OPENAI_BASE_URL = OPENAI_ORIGIN + "/v1"
DEFAULT_REASONING_BUDGET = 8192


def strict_schema(schema: dict) -> dict:
    """Translate typed optional/default fields, never relax the output contract."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                if isinstance(node.get("additionalProperties"), dict):
                    raise ValueError("OpenAI strict outputs cannot represent an open mapping")
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


def response_request(selection, *, kind, system, user, evidence, schema, max_tokens):
    parameters = selection.parameters
    final = kind == "FINAL_RESPONSE_V2"
    reasoning = openai_reasoning(selection.model)
    inputs = []
    if evidence:
        # A separate user data item; historical material never becomes developer authority.
        inputs.append({"role": "user", "content": evidence})
    inputs.append({"role": "user", "content": user})
    output_budget = DEFAULT_REASONING_BUDGET if reasoning else max_tokens
    if final:
        output_budget = parameters.get("max_output_tokens", output_budget)
    request = {
        "model": selection.model, "instructions": system, "input": inputs,
        "store": False, "stream": False,
        "max_output_tokens": output_budget,
        "text": {"format": {"type": "json_schema", "name": "persistent_cognition_output",
                            "strict": True, "schema": strict_schema(schema)}},
    }
    if reasoning:
        if final and "reasoning_effort" in parameters:
            request["reasoning"] = {"effort": parameters["reasoning_effort"]}
    else:
        request["temperature"] = parameters.get("temperature", 0.65) if final else 0
        if final and "top_p" in parameters:
            request["top_p"] = parameters["top_p"]
    if "verbosity" in parameters and final:
        request["text"]["verbosity"] = parameters["verbosity"]
    return request


def perform_response(owner, *, kind, system, user, evidence, schema, max_tokens):
    request = response_request(owner.selection, kind=kind, system=system, user=user,
                               evidence=evidence, schema=schema, max_tokens=max_tokens)
    diagnostics = {"transport": "openai-responses", "request_path": "/v1/responses",
                   "request_body": request, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        require_destination(OPENAI_ORIGIN, NetworkPurpose.MODEL)
        validate_model_input(request)
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("Connect an OpenAI API key in Models or set OPENAI_API_KEY")
        # Fixed origin; proxies, redirects and user-supplied endpoints cannot receive credentials.
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=300.0) as client:
            response = client.post(OPENAI_BASE_URL + "/responses", json=request,
                                   headers={"Authorization": "Bearer " + key})
        diagnostics["http_status_code"] = response.status_code
        diagnostics["response_body_sha256"] = hashlib.sha256(response.content).hexdigest()
        if response.is_error:
            # Provider error text can repeat private input. Keep a bounded type/code only.
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            code = str(error.get("code") or error.get("type") or "request_failed")[:120]
            raise ValueError(f"OpenAI HTTP {response.status_code}: {code}. Check model access, parameters, billing and output budget.")
        body = response.json()
        diagnostics["response_envelope"] = {key: body.get(key) for key in
                                            ("id", "model", "status", "usage", "incomplete_details")}
        if body.get("status") != "completed":
            raise ValueError("OpenAI response was incomplete; increase output budget or inspect Activity")
        texts = []
        for item in body.get("output", []):
            if item.get("type") != "message":
                continue  # Never retain provider reasoning content.
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise ValueError("OpenAI declined this request")
                if part.get("type") == "output_text":
                    texts.append(part["text"])
        result = "".join(texts).strip()
        if not result:
            raise ValueError("OpenAI returned no structured answer")
        json.loads(result)
        return result
    except Exception as exc:
        diagnostics["transport_error_type"] = type(exc).__name__
        # Authentication material is never attached to diagnostic artifacts.
        raise
    finally:
        diagnostics["completed_at"] = datetime.now(timezone.utc).isoformat()
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        owner._last_invocation_diagnostics = diagnostics
