"""Closed stage, prompt and schema identities for guarded cognitive execution.

Executable external capabilities remain owned by CapabilityRegistry. Internal
stage capabilities are deliberately unavailable to semantic work selection.
"""
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
import json
from types import MappingProxyType

from persistent_cognition.prompt_registry import PROMPTS, prompt_digest

CONTRACT_REGISTRY_VERSION = "cognitive-contracts/v1"


@dataclass(frozen=True)
class SemanticContract:
    prompt: str
    schema: str | None

    def output_schema(self):
        if self.schema is None:
            return {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"], "additionalProperties": False}
        module, name = self.schema.rsplit(":", 1)
        return getattr(import_module("persistent_cognition." + module), name).model_json_schema()


SEMANTIC_CONTRACTS = MappingProxyType({
    "V2_RESPONSE_POLICY": SemanticContract("_RESPONSE_POLICY_PROMPT", "response_policy:ResponsePolicy"),
    "PRECOGNITIVE_USER_PROMPT_WORK": SemanticContract("_USER_PROMPT_WORK_SELECTION", "percept_response_worker:UserPromptWorkSelection"),
    "V2_CURRENT_FALLBACK_SELECTION": SemanticContract("_CURRENT_FALLBACK_SELECTION_PROMPT", "response_policy:CurrentFallbackSelection"),
    "V2_EXACT_SOURCE_SELECTION": SemanticContract("_EXACT_SOURCE_SELECTION_PROMPT", "response_policy:ExactSourceSelection"),
    "V2_EXACT_SOURCE_COMPOSITION": SemanticContract("_EXACT_SOURCE_COMPOSITION_PROMPT", "response_policy:ExactSourceComposition"),
    "FINAL_RESPONSE_V2": SemanticContract("_FINAL_RESPONSE_PROMPT", "llm:_TextAnswer"),
    "PERCEPT_TRIAGE": SemanticContract("TRIAGE_PROMPT", "percept_triage:TriageDecision"),
})


@dataclass(frozen=True)
class StageContract:
    capability: str
    role: str
    kinds: frozenset[str] = frozenset()


STAGE_CONTRACTS = MappingProxyType({
    "V2_RESOLVE_REFERENCES": StageContract("interaction.resolve_references", "deterministic reference resolver"),
    "V2_EVIDENCE_POLICY": StageContract("interaction.plan_evidence", "evidence policy specialist", frozenset({"V2_RESPONSE_POLICY"})),
    "V2_PRECOGNITIVE": StageContract("interaction.precognitive_disposition", "work triage specialist", frozenset({"PRECOGNITIVE_USER_PROMPT_WORK"})),
    "V2_EXECUTE_WORK": StageContract("capability.execute", "deterministic capability executor"),
    "V3_RETRIEVE_MEMORY": StageContract("interaction.retrieve_memory", "deterministic evidence retriever"),
    "V2_RESPOND": StageContract("interaction.respond", "final response specialist", frozenset({"V2_CURRENT_FALLBACK_SELECTION", "V2_EXACT_SOURCE_SELECTION", "V2_EXACT_SOURCE_COMPOSITION", "FINAL_RESPONSE_V2"})),
    "V2_PERSIST_RESULT": StageContract("interaction.persist_result", "deterministic result persister"),
    "SITUATION_MEMORY": StageContract("situation.memory", "deterministic memory activation"),
    "SITUATION_TRIAGE": StageContract("situation.triage", "percept triage specialist", frozenset({"PERCEPT_TRIAGE"})),
    "SITUATION_EXECUTE": StageContract("situation.execute", "deterministic action executor"),
    "SITUATION_RETRIEVE_MEMORY": StageContract("situation.retrieve", "deterministic evidence retriever"),
    "SITUATION_RESPOND": StageContract("situation.respond", "final response specialist", frozenset({"FINAL_RESPONSE_V2", "V2_CURRENT_FALLBACK_SELECTION"})),
    "SITUATION_PERSIST": StageContract("situation.persist", "deterministic result persister"),
})


SCHEMA_CONTRACTS = MappingProxyType({
    "app-settings/v1": "runtime_settings:AppSettings",
    "model-selection/v1": "runtime_settings:ModelSelection",
    "task-routing-settings/v1": "runtime_settings:TaskRoutingSettings",
    "model-eligibility/v1": "model_admission:ModelEligibility",
    "task-plan/v1": "model_admission:TaskPlan",
    "network-consent/v1": "network_consent:NetworkConsent",
    "precognitive-disposition/v1": "percept_response_runtime:PreCognitiveDisposition",
    "response-memory/v3": "percept_response_runtime:ResponseMemoryPackage",
    "memory-packet/v1": "models:MemoryPacket",
    "percept/v1": "perception:Percept",
    "source-policy/v1": "percept_triage:SourcePolicy",
})


def contract_manifest():
    """Resolve and validate every active reference; usable without DB or model."""
    for stage in STAGE_CONTRACTS.values():
        if not stage.kinds <= SEMANTIC_CONTRACTS.keys():
            raise ValueError("stage references an unregistered semantic contract")
    result = {}
    for kind, contract in SEMANTIC_CONTRACTS.items():
        if contract.prompt not in PROMPTS:
            raise ValueError("missing registered prompt")
        schema = contract.output_schema()
        result[kind] = {"prompt_id": contract.prompt, "prompt_sha256": prompt_digest(contract.prompt),
                        "schema_id": contract.schema or "response-text/v1",
                        "schema_sha256": sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()}
    from persistent_cognition.capability_registry import DEFAULT_REGISTRY, CAPABILITY_REGISTRY_VERSION
    schemas = {}
    for key, target in SCHEMA_CONTRACTS.items():
        module, name = target.split(":")
        schema = getattr(import_module("persistent_cognition." + module), name).model_json_schema()
        schemas[key] = {"model": target, "sha256": sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()}
    from persistent_cognition.model_parameters import parameter_manifest
    from persistent_cognition.model_admission import admission_manifest
    return {"version": CONTRACT_REGISTRY_VERSION, "model_parameters": parameter_manifest(), "model_routing": admission_manifest(), "semantic_contracts": result,
            "schemas": schemas, "external_capability_registry": CAPABILITY_REGISTRY_VERSION,
            "external_capabilities": [d.model_dump(mode="json") for d in DEFAULT_REGISTRY.capability_catalog()],
            "stages": {key: {"capability": value.capability, "role": value.role,
                              "kinds": sorted(value.kinds)} for key, value in STAGE_CONTRACTS.items()}}
