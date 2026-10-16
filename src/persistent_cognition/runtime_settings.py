"""Typed engine settings and immutable worker configuration snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from persistent_cognition.model_parameters import validate_parameters
from persistent_cognition.ollama_runtime import configured_ollama_model
from persistent_cognition.operator_state import policy_lock, write_private_policy

GUI_CONFIG_ENV = "PCR_GUI_JOB_CONFIG"
GUI_CONFIG_FILE_ENV = "PCR_GUI_JOB_CONFIG_FILE"


class SettingsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSelection(SettingsRecord):
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = Field(default_factory=configured_ollama_model, min_length=1, max_length=200)
    parameters: dict = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def model_reference(cls, value):
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+){0,3}(?::[A-Za-z0-9_.-]+)?", value):
            raise ValueError("use a model ID or Ollama namespace/name:tag, not a URL")
        if value.casefold().endswith(("-cloud", ":cloud")):
            raise ValueError("Ollama cloud aliases are not supported; use an explicit cloud provider")
        return value

    @model_validator(mode="after")
    def valid_options(self):
        validate_parameters(self.provider, self.model, self.parameters)
        return self


class MemorySettings(SettingsRecord):
    max_item_bytes: int = Field(default=16384, ge=1024, le=1048576)
    max_total_bytes: int = Field(default=65536, ge=1024, le=4194304)
    max_input_bytes: int = Field(default=98304, ge=1024, le=8388608)

    @model_validator(mode="after")
    def ordered_budgets(self):
        if not self.max_item_bytes <= self.max_total_bytes <= self.max_input_bytes:
            raise ValueError("evidence item ≤ evidence total ≤ complete model input is required")
        return self


class ResourceSettings(SettingsRecord):
    cpu_system_headroom_percent: int = Field(default=10, ge=1, le=90)
    # Allow a fixed MiB system reserve without a proportional RAM deduction.
    memory_system_headroom_percent: int = Field(default=0, ge=0, le=90)
    memory_system_headroom_min_mib: int = Field(default=1024, ge=256, le=1048576)
    uncertainty_headroom_percent: int = Field(default=5, ge=1, le=90)
    max_cpu_pressure_percent: int = Field(default=85, ge=10, le=99)
    default_llm_process_memory_mib: int = Field(default=3072, ge=512, le=1048576)
    default_process_memory_mib: int = Field(default=512, ge=128, le=1048576)
    model_context_tokens: int = Field(default=4096, ge=512, le=1048576)
    model_runtime_headroom_percent: int = Field(default=25, ge=10, le=200)
    model_runtime_headroom_min_mib: int = Field(default=512, ge=256, le=1048576)

    @model_validator(mode="after")
    def enough_worker_memory(self):
        if self.default_llm_process_memory_mib < self.default_process_memory_mib:
            raise ValueError("local model memory estimate must include worker memory")
        return self


class SpecialistModel(SettingsRecord):
    """Operator-declared expertise, distinct from provider-reported modalities."""
    selection: ModelSelection
    tasks: list[Literal["general", "coding", "vision"]] = Field(min_length=1, max_length=3)
    priority: int = Field(default=100, ge=0, le=1000)

    @model_validator(mode="after")
    def local_specialist(self):
        if self.selection.provider != "ollama":
            raise ValueError("Automatic specialists must be local Ollama models; pin cloud models explicitly by stage")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Specialist tasks must be unique")
        return self


class TaskRoutingSettings(SettingsRecord):
    enabled: bool = True
    allow_default_fallback: bool = True
    specialists: list[SpecialistModel] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def unique_models(self):
        from persistent_cognition.ollama_runtime import _canonical_model_name
        names = [_canonical_model_name(item.selection.model) for item in self.specialists]
        if len(set(names)) != len(names):
            raise ValueError("Register each specialist model once")
        return self


class AppSettings(SettingsRecord):
    schema_version: Literal["app-settings/v1"] = "app-settings/v1"
    selection: ModelSelection = Field(default_factory=ModelSelection)
    routes: dict[str, ModelSelection] = Field(default_factory=dict)
    ollama_url: str = "http://localhost:11434"
    memory: MemorySettings = Field(default_factory=MemorySettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
    task_routing: TaskRoutingSettings = Field(default_factory=TaskRoutingSettings)
    worker_timeout_seconds: int = Field(default=600, ge=30, le=3600)

    @field_validator("ollama_url")
    @classmethod
    def explicit_endpoint(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Ollama URL must be an HTTP(S) origin without credentials or a path")
        # Parsing the port also validates its range.
        _ = parsed.port
        return value.rstrip("/")

    @field_validator("routes")
    @classmethod
    def registered_stages(cls, value):
        from persistent_cognition.contract_registry import STAGE_CONTRACTS
        for stage in value:
            if stage not in STAGE_CONTRACTS or not STAGE_CONTRACTS[stage].kinds:
                raise ValueError(f"no model role registered for stage {stage}")
        return value

    def selection_for(self, stage: str | None) -> ModelSelection:
        return self.routes.get(stage, self.selection)

    def worker_environment(self, config_path: Path | None = None) -> dict[str, str]:
        return {
            GUI_CONFIG_ENV: "" if config_path else self.model_dump_json(),
            GUI_CONFIG_FILE_ENV: str(config_path) if config_path else "",
            "OLLAMA_BASE_URL": self.ollama_url,
            "OLLAMA_MODEL": self.selection.model,
            "PCR_WORKER_TIMEOUT_SECONDS": str(self.worker_timeout_seconds),
            "PCR_MAX_MODEL_EVIDENCE_ITEM_BYTES": str(self.memory.max_item_bytes),
            "PCR_MAX_MODEL_EVIDENCE_TOTAL_BYTES": str(self.memory.max_total_bytes),
            "PCR_MAX_MODEL_INPUT_BYTES": str(self.memory.max_input_bytes),
        }


def job_settings() -> AppSettings | None:
    path = os.environ.get(GUI_CONFIG_FILE_ENV)
    from persistent_cognition.resource_limits import read_json_file, validate_text_intake
    if path:
        return AppSettings.model_validate(read_json_file(Path(path)))
    raw = os.environ.get(GUI_CONFIG_ENV)
    if raw:
        validate_text_intake(raw)
    return AppSettings.model_validate_json(raw) if raw else None


def load_settings(root: Path) -> AppSettings:
    path = root / "operator" / "app-settings.json"
    from persistent_cognition.resource_limits import read_json_file
    return AppSettings.model_validate(read_json_file(path)) if path.exists() else AppSettings()


def save_settings(root: Path, settings: AppSettings):
    path = root / "operator" / "app-settings.json"
    with policy_lock(path):
        write_private_policy(path, settings.model_dump(mode="json"))


def settings_revision(settings: AppSettings) -> str:
    from hashlib import sha256
    return sha256(json.dumps(settings.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def local_ollama(settings: AppSettings) -> bool:
    from persistent_cognition.network_consent import _loopback
    return _loopback(urlsplit(settings.ollama_url).hostname)
