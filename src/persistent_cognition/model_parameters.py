"""Versioned, inspectable generation controls. An absent value means inherit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType

PARAMETER_REGISTRY_VERSION = "generation-parameters/v1"


@dataclass(frozen=True)
class Parameter:
    label: str
    kind: str
    group: str
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    scope: str = "response"


OLLAMA_PARAMETERS = MappingProxyType({
    "temperature": Parameter("Temperature", "number", "Sampling", "Randomness in the final response. Control workers remain at zero.", 0, 2),
    "top_p": Parameter("Top P", "number", "Sampling", "Cumulative probability cutoff.", 0, 1),
    "top_k": Parameter("Top K", "integer", "Sampling", "Limit sampling to the most likely tokens; zero disables.", 0, 1000),
    "min_p": Parameter("Min P", "number", "Sampling", "Discard tokens unlikely relative to the best candidate.", 0, 1),
    "seed": Parameter("Seed", "integer", "Sampling", "Reproducibility input; not a guarantee across runtimes or hardware.", -1, 2147483647),
    "num_predict": Parameter("Output tokens", "integer", "Context", "Finite final-response output budget. Infinite generation is not supported.", 1, 131072),
    "num_ctx": Parameter("Context window", "integer", "Context", "Allocated context; bounded by reported model metadata when available. Larger values need more memory.", 256, 2097152, scope="all"),
    "num_keep": Parameter("Keep tokens", "integer", "Context", "Tokens retained when context is shifted.", -1, 2097152, scope="all"),
    "repeat_last_n": Parameter("Repetition window", "integer", "Sampling", "Zero disables; -1 uses the context window.", -1, 2097152),
    "repeat_penalty": Parameter("Repeat penalty", "number", "Sampling", "Penalty applied to repeated tokens.", 0, 5),
    "presence_penalty": Parameter("Presence penalty", "number", "Sampling", "Penalty for previously present tokens.", -2, 2),
    "frequency_penalty": Parameter("Frequency penalty", "number", "Sampling", "Penalty proportional to token frequency.", -2, 2),
    "stop": Parameter("Stop sequences", "strings", "Sampling", "One stop sequence per line. Early stopping can invalidate a structured answer."),
    "num_thread": Parameter("CPU threads", "integer", "Hardware", "Ollama worker threads; automatic when inherited.", 1, 1024, scope="all"),
    "num_batch": Parameter("Batch size", "integer", "Hardware", "Prompt-processing batch size.", 1, 65536, scope="all"),
    "num_gpu": Parameter("GPU layers", "integer", "Hardware", "-1 lets Ollama decide; zero requests CPU only.", -1, 1024, scope="all"),
    "main_gpu": Parameter("Main GPU", "integer", "Hardware", "Primary GPU index.", 0, 128, scope="all"),
    "use_mmap": Parameter("Memory mapping", "boolean", "Hardware", "Map model weights into memory.", scope="all"),
    "draft_num_predict": Parameter("Draft tokens", "integer", "Hardware", "Speculative decoding draft length; requires runtime and model support.", 0, 256, scope="all"),
    "think": Parameter("Thinking", "boolean", "Model", "Enable final-response thinking only when the model reports this capability."),
    "keep_alive": Parameter("Keep loaded (seconds)", "integer", "Hardware", "The local app enforces zero and verifies unloading between model calls. Remote services may honor this override.", -1, 86400, scope="all"),
})
OPENAI_PARAMETERS = MappingProxyType({
    "max_output_tokens": Parameter("Output budget", "integer", "Context", "Includes reasoning tokens. Low budgets may leave no final answer.", 256, 262144),
    "temperature": Parameter("Temperature", "number", "Sampling", "For standard models that accept sampling controls.", 0, 2),
    "top_p": Parameter("Top P", "number", "Sampling", "For standard models that accept sampling controls.", 0, 1),
    "reasoning_effort": Parameter("Reasoning effort", "choice", "Model", "Allowed efforts vary by model; inherit uses the model default.", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max")),
    "verbosity": Parameter("Verbosity", "choice", "Model", "Supported by GPT-5 and newer text models.", choices=("low", "medium", "high")),
})


def openai_reasoning(model: str) -> bool:
    return model.casefold().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def parameter_catalog(provider: str, model: str = "", metadata: dict | None = None) -> dict:
    registry = OLLAMA_PARAMETERS if provider == "ollama" else OPENAI_PARAMETERS
    result = {key: asdict(value) for key, value in registry.items()}
    if provider == "openai":
        for key in ("temperature", "top_p") if openai_reasoning(model) else ("reasoning_effort",):
            result.pop(key, None)
        if not model.startswith(("gpt-5", "gpt-6")):
            result.pop("verbosity", None)
    elif metadata is not None:
        if "thinking" not in metadata.get("capabilities", []):
            result.pop("think", None)
        lengths = [v for k, v in metadata.get("model_info", {}).items()
                   if k.endswith(".context_length") and isinstance(v, int) and v > 0]
        if lengths:
            result["num_ctx"]["maximum"] = min(lengths)
    return result


def validate_parameters(provider: str, model: str, values: dict) -> dict:
    catalog = parameter_catalog(provider, model)
    for name, value in values.items():
        if name not in catalog:
            raise ValueError(f"unsupported {provider} parameter for {model}: {name}")
        spec = catalog[name]
        kind = spec["kind"]
        if kind in ("integer", "number"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} requires a finite number")
            if kind == "integer" and not isinstance(value, int):
                raise ValueError(f"{name} requires an integer")
            if not spec["minimum"] <= value <= spec["maximum"]:
                raise ValueError(f"{name} is outside {spec['minimum']}..{spec['maximum']}")
        elif kind == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{name} requires a boolean")
        elif kind == "choice" and value not in spec["choices"]:
            raise ValueError(f"{name} is not a supported choice")
        elif kind == "strings" and (not isinstance(value, list) or len(value) > 16
                or any(not isinstance(v, str) or not 1 <= len(v) <= 256 for v in value)):
            raise ValueError(f"{name} needs up to 16 nonempty strings of at most 256 characters")
    return values


def parameter_manifest():
    return {"version": PARAMETER_REGISTRY_VERSION,
            "ollama": {name: asdict(value) for name, value in OLLAMA_PARAMETERS.items()},
            "openai": {name: asdict(value) for name, value in OPENAI_PARAMETERS.items()},
            "scope": "Sampling overrides apply to final response; hardware overrides apply to all local calls",
            "metadata_limits": "Ollama capabilities/context where reported; OpenAI conservative family profiles"}
