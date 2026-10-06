"""Persistent JSON-lines inference subprocess (Python 3.12, separate from PCR).

Never downloads on load. prepare_specialist_models.py freezes/caches weights first.
stdout is protocol only; diagnostics belong on stderr. No remote-code execution.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_function_selection(text: str) -> str:
    """Accept only the declared non-executable selection function and integer list.

FunctionGemma's native tool syntax is not JSON. Do not use eval or repair a
malformed response into a valid answer; failures must count against the model.
"""
    match = re.fullmatch(
        r"\s*<start_function_call>call:select_work\{capability_indices:\s*"
        r"(\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\])\s*\}<end_function_call>\s*",
        text,
    )
    if not match:
        raise ValueError("not a valid select_work FunctionGemma call")
    return json.dumps({"capability_indices": json.loads(match.group(1))})


def parse_hammer_selection(text: str) -> str:
    """Accept one native tool call, never arbitrary surrounding prose or code."""
    text = text.strip()
    if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
        text = text.removeprefix("<tool_call>").removesuffix("</tool_call>").strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.startswith("json\n"):
            text = text[5:]
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"name", "arguments"} or value["name"] != "select_work":
        raise ValueError("not a valid select_work Hammer call")
    if not isinstance(value["arguments"], dict):
        raise ValueError("Hammer arguments must be an object")
    return json.dumps(value["arguments"])


SCOPE_HYPOTHESES = {
    "USER_AUTHORED": "The user asks for a specific personal fact or statement from their own past messages.",
    "MODEL_OUTPUT": "The user asks for the content of an earlier assistant answer.",
    "EXTERNAL_TOOL": "The user asks for facts supplied by an external tool or observation.",
    "SYSTEM_RECORD": "The user asks for facts in the runtime's system logs or error records.",
    "DERIVED_INTERNAL": "The user asks for internal derived decisions or retrieval records.",
    "SELF_MODEL": "The user asks about their typical preferences, tendencies, values or identity patterns.",
    "MIXED_CONVERSATION": "The user asks to summarize or compare both sides of an earlier conversation.",
    "GENERAL_OR_CURRENT": "The user asks a general question or a task answerable from the current prompt.",
}
SURFACE_HYPOTHESES = {
    "NATURAL_LANGUAGE": "The user wants a natural language answer or explanation.",
    "EXACT_SOURCE_SUBSTRING": "The user explicitly wants only one exact verbatim extract from a source.",
    "EXACT_SOURCE_COMPOSITION": "The user explicitly wants several exact source extracts joined in a specified format.",
}


def memory_reader():
    """Telemetry must not break inference in containers with unusual PID mounts."""
    import psutil
    try:
        if sys.platform.startswith("linux") and os.readlink("/proc/self") != str(os.getpid()):
            raise OSError("PID namespace does not match procfs; cannot trust psutil PID telemetry")
        process = psutil.Process()
        process.memory_info()
        return lambda: process.memory_info().rss, "sampled_process_rss"
    except (psutil.Error, OSError):
        try:
            import resource
            multiplier = 1 if sys.platform == "darwin" else 1024
            return lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier, "process_lifetime_high_water_rss"
        except ImportError:
            return lambda: 0, "unavailable"


class Inference:
    def __init__(self, profile):
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        self.torch, self.profile = torch, profile
        revision = profile.get("revision", "")
        if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError("freeze an immutable 40-character HF revision before inference")
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(profile.get("threads", 4))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = profile.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback")
        if self.device == "cuda" and profile["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported by requested GPU")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[profile["precision"]]
        snapshot = Path(snapshot_download(
            profile["model"], revision=revision, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "merges.txt", "*.txt"],
        ))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False)
        model_class = (transformers.AutoModelForCausalLM if profile["adapter"] in ("functiongemma-selection", "hammer-selection")
                       else transformers.AutoModelForSequenceClassification)
        self.model = model_class.from_pretrained(
            snapshot, dtype=dtype, local_files_only=True, trust_remote_code=False,
            attn_implementation="eager",
        ).to(self.device).eval()
        count = sum(p.numel() for p in self.model.parameters())
        if count >= 1_000_000_000:
            raise ValueError("precision specialist must have fewer than 1B parameters")
        dtypes = Counter(str(p.dtype) for p in self.model.parameters())
        if set(dtypes) != {str(dtype)}:
            raise ValueError(f"loaded dtype mismatch: {dtypes}")
        files = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and (path.suffix in (".json", ".safetensors", ".jinja", ".model") or path.name == "merges.txt"):
                h = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
                files[str(path.relative_to(snapshot))] = h.hexdigest()
        self.identity = {"model": profile["model"], "revision": revision, "snapshot": str(snapshot),
                         "parameters": count, "requested_precision": profile["precision"],
                         "parameter_dtypes": dict(dtypes), "device": self.device,
                         "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                         "threads": torch.get_num_threads(), "attention": "eager",
                         "deterministic_algorithms": True, "weight_and_config_sha256": files,
                         "adapter": profile["adapter"],
                         "inference_source_sha256": SOURCE_SHA256,
                         "precision_note": "Casts one released checkpoint; FP32 cannot recover precision absent in its original weights."}
        self.raw = None

    def nli(self, premises, hypotheses):
        encoded = self.tokenizer(premises, hypotheses, padding=True, truncation=False,
                                 return_tensors="pt").to(self.device)
        maximum = getattr(self.model.config, "max_position_embeddings", 8192)
        if encoded["input_ids"].shape[-1] > maximum:
            raise ValueError("encoder input exceeds context; no silent truncation")
        logits = self.model(**encoded).logits.float()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if not {"entailment", "contradiction", "neutral"} <= labels.keys():
            raise ValueError("checkpoint must supply named NLI label mapping; cannot guess indices")
        return logits, labels, int(encoded["attention_mask"].sum().item())

    def infer(self, request, inputs):
        torch = self.torch
        self.raw = None
        rss, rss_metric = memory_reader()
        peak = [rss()]
        stop = threading.Event()

        def sample():
            while not stop.wait(.01):
                try:
                    peak[0] = max(peak[0], rss())
                except OSError:
                    break

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                if self.profile["adapter"] in ("functiongemma-selection", "hammer-selection"):
                    if request["kind"] != "PRECOGNITIVE_USER_PROMPT_WORK":
                        raise ValueError("FunctionGemma adapter supports only capability selection")
                    tools = [{"type": "function", "function": {
                        "name": "select_work", "description": "Select indices from PCR's admitted catalog; use an empty list when no work applies. This records a decision and executes no action.",
                        "parameters": request["schema"],
                    }}]
                    functiongemma = self.profile["adapter"] == "functiongemma-selection"
                    messages = [{"role": "developer" if functiongemma else "system", "content":
                                 ("You are a model that can do function calling with the following functions\n" if functiongemma else "") + request["system_prompt"]},
                                {"role": "user", "content": request["evidence_prompt"] + "\n\n" + request["user_prompt"]}]
                    if not functiongemma:
                        tools = [tool["function"] for tool in tools]
                    rendered = self.tokenizer.apply_chat_template(messages, tools=tools,
                                                                  add_generation_prompt=True, tokenize=False)
                    tokens = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(self.device)
                    prompt_tokens = tokens["input_ids"].shape[-1]
                    if prompt_tokens + request["max_tokens"] > self.model.config.max_position_embeddings:
                        raise ValueError("function-selection context overflow")
                    generated = self.model.generate(**tokens, do_sample=False,
                                                    max_new_tokens=request["max_tokens"],
                                                    pad_token_id=self.tokenizer.eos_token_id,
                                                    use_cache=True)
                    answer_tokens = generated[0][prompt_tokens:]
                    # Remove turn delimiters only; native function delimiters remain.
                    native = self.tokenizer.decode(answer_tokens, skip_special_tokens=False)
                    native = re.sub(r"(?:<end_of_turn>|<eos>|<pad>|<\|im_end\|>|<\|endoftext\|>)+$", "", native).strip()
                    self.raw = {"native_output": native, "rendered_prompt": rendered,
                                "tools": tools, "prompt_tokens": prompt_tokens,
                                "generated_tokens": len(answer_tokens), "decoding": "greedy"}
                    output = parse_function_selection(native) if functiongemma else parse_hammer_selection(native)
                elif request["kind"] == "V2_RESPONSE_POLICY":
                    scores = {}
                    chosen = {}
                    for field, hypotheses in (("evidence_scope", SCOPE_HYPOTHESES),
                                              ("surface_mode", SURFACE_HYPOTHESES)):
                        names = list(hypotheses)
                        logits, labels, token_count = self.nli([inputs["percept"]] * len(names),
                                                               list(hypotheses.values()))
                        # Explicit zero-shot adapter, not an imagined PCR-trained head.
                        entailment = logits[:, labels["entailment"]]
                        chosen[field] = names[int(entailment.argmax().item())]
                        scores[field] = {"hypotheses": hypotheses, "logits": logits.cpu().tolist(),
                                         "labels": labels, "input_tokens": token_count}
                    chosen["insufficient_literal"] = None
                    self.raw = {"zero_shot_nli": scores, "calibrated": False}
                    output = json.dumps(chosen)
                elif request["kind"] == "SELF_SCHEMA_REVIEW":
                    items = inputs["related_items"]
                    if items:
                        logits, labels, token_count = self.nli(items, [inputs["candidate"]] * len(items))
                        winners = logits.argmax(dim=-1).cpu().tolist()
                        opposition = [i for i, winner in enumerate(winners) if winner == labels["contradiction"]]
                        self.raw = {"logits": logits.cpu().tolist(), "labels": labels,
                                    "input_tokens": token_count, "calibrated": False,
                                    "adapter_limit": "Pairwise NLI does not jointly reason over support roots."}
                    else:
                        opposition = []
                        self.raw = {"logits": [], "input_tokens": 0}
                    output = json.dumps({"opposition_indices": opposition,
                                         "rationale": "Pairwise NLI contradiction indices: " + str(opposition)})
                else:
                    raise ValueError("NLI adapter does not implement this worker contract")
            if self.device == "cuda":
                torch.cuda.synchronize()
            return {"output": output, "raw": self.raw,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_bytes": max(peak[0], rss()) if rss_metric != "unavailable" else None,
                    "rss_metric": rss_metric,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None}
        finally:
            stop.set()
            sampler.join()


def main():
    engine = None
    for line in sys.stdin:
        try:
            value = json.loads(line)
            if value["op"] == "load":
                engine = Inference(value["profile"])
                result = engine.identity
            elif value["op"] == "infer" and engine is not None:
                result = engine.infer(value["request"], value["inputs"])
            else:
                raise ValueError("load one profile before inference")
        except Exception as exc:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)},
                      "raw": engine.raw if engine is not None else None}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
