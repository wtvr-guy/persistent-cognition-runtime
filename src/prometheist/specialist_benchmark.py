"""Offline experiments through actual PCR worker methods; never production routing.

Inference is injected only at the transport boundary. Existing worker prompts,
deterministic bypasses, token caps, retries, and validators remain authoritative.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Literal
from uuid import UUID, uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field

from prometheist.capability_registry import CapabilityDescriptor
from prometheist.contract_registry import contract_manifest
from prometheist.llm import OllamaClient
from prometheist.models import MemoryNeed, MemoryPacket
from prometheist.percept_response_runtime import PerceptStage
from prometheist.percept_response_worker import UserPromptLLM
from prometheist.percept_triage import SourcePolicy
from prometheist.situation_runtime import SituationStage
from prometheist.situation_worker import SituationLLM

BENCHMARK_VERSION = "specialist-worker-benchmark/v1"
KINDS = (
    "V2_RESPONSE_POLICY", "PRECOGNITIVE_USER_PROMPT_WORK",
    "PERCEPT_TRIAGE", "SELF_SCHEMA_REVIEW",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class WorkerCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    # All paraphrases/related events share one group and therefore one split.
    group_id: str = Field(min_length=1)
    split: Literal["train", "dev", "test"]
    kind: Literal[
        "V2_RESPONSE_POLICY", "PRECOGNITIVE_USER_PROMPT_WORK",
        "PERCEPT_TRIAGE", "SELF_SCHEMA_REVIEW",
    ]
    inputs: dict[str, Any]
    expected: dict[str, Any] | None = None
    score_fields: list[str] = Field(min_length=1)
    label_source: str = "unreviewed"


def load_cases(path: Path) -> list[WorkerCase]:
    cases = [WorkerCase.model_validate_json(line) for line in path.read_text().splitlines()
             if line.strip()]
    groups: dict[str, str] = {}
    ids = set()
    input_splits = {}
    for case in cases:
        if case.case_id in ids:
            raise ValueError(f"duplicate case ID: {case.case_id}")
        ids.add(case.case_id)
        if groups.setdefault(case.group_id, case.split) != case.split:
            raise ValueError(f"group crosses train/test boundary: {case.group_id}")
        key = digest({"kind": case.kind, "inputs": case.inputs})
        if input_splits.setdefault(key, case.split) != case.split:
            raise ValueError("identical worker input crosses splits")
        if case.expected is not None and not set(case.score_fields) <= case.expected.keys():
            raise ValueError(f"missing expected score fields: {case.case_id}")
    return cases


def invoke_worker(case: WorkerCase, worker: UserPromptLLM | SituationLLM) -> dict:
    """No DB, scheduler, capabilities, or durable self updates are executed."""
    data = case.inputs
    if case.kind == "V2_RESPONSE_POLICY":
        result = worker._response_policy(data["percept"])
    elif case.kind == "PRECOGNITIVE_USER_PROMPT_WORK":
        packet = MemoryPacket.model_validate(data["memory_packet"]) if "memory_packet" in data else (
            MemoryPacket(memory_request_id=UUID(int=1), need=MemoryNeed(),
                         supported=False, items=[])
        )
        result = worker.decide_disposition(
            data["percept"], packet,
            tuple(CapabilityDescriptor.model_validate(x) for x in data["catalog"]),
        )
    elif case.kind == "PERCEPT_TRIAGE":
        result = worker.triage(SourcePolicy.model_validate(data["policy"]), data["evidence"])
    elif case.kind == "SELF_SCHEMA_REVIEW":
        related = "\n\n".join(f"evidence_index: {i}\ncontent: {text}"
                                for i, text in enumerate(data["related_items"]))
        result = worker.review_self_schema(candidate=data["candidate"],
                                           support_evidence=data["support_evidence"],
                                           related_evidence=related)
        # Downstream index resolution is application-owned; score it as well.
        if any(i < 0 or i >= len(data["related_items"]) for i in result.opposition_indices):
            raise ValueError("review selected an unavailable related evidence index")
    else:
        raise ValueError(f"unsupported worker: {case.kind}")
    return result.model_dump(mode="json")


class Sidecar:
    """Keep one checkpoint resident, isolated from PCR's Python 3.14 environment."""
    def __init__(self, python: str, script: Path, profile: dict, timeout: float = 600):
        self.timeout = timeout
        self.proc = subprocess.Popen([python, str(script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.responses: queue.Queue = queue.Queue()
        self.stderr: list[str] = []

        def read_stdout():
            for line in self.proc.stdout:
                self.responses.put(line)
            self.responses.put(None)

        def read_stderr():
            for line in self.proc.stderr:
                self.stderr.append(line)
                self.stderr[:] = self.stderr[-100:]

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        try:
            self.identity = self.request({"op": "load", "profile": profile})
        except BaseException:
            self.close()
            raise

    def request(self, value: dict) -> dict:
        self.proc.stdin.write(json.dumps(value) + "\n")
        self.proc.stdin.flush()
        try:
            line = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("specialist inference sidecar timed out") from None
        if line is None:
            raise RuntimeError("sidecar exited: " + "".join(self.stderr)[-4000:])
        result = json.loads(line)
        if result.get("error"):
            raise InferenceError(result)
        return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()


class InferenceError(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__(json.dumps(result))


class RecordingTransport:
    def __init__(self, backend, profile: dict, case: WorkerCase):
        self.backend, self.profile, self.case = backend, profile, case
        self.calls: list[dict] = []

    def structured(self, kind, system, current_user, evidence, schema, max_tokens):
        request = {"kind": kind, "system_prompt": system, "user_prompt": current_user,
                   "evidence_prompt": evidence, "schema": schema, "max_tokens": max_tokens}
        call = {"request": request, "request_sha256": digest(request)}
        self.calls.append(call)
        start = time.perf_counter()
        try:
            if isinstance(self.backend, OllamaClient):
                output = self.backend._structured_with_evidence(
                    kind, system, current_user, evidence, schema, max_tokens)
            else:
                result = self.backend.request({"op": "infer", "request": request,
                                               "inputs": self.case.inputs})
                call["inference"] = result
                output = result["output"]
            call["output"] = output
            return output
        except Exception as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, InferenceError):
                call["inference"] = exc.result
            raise
        finally:
            call["elapsed_seconds"] = time.perf_counter() - start
            if isinstance(self.backend, OllamaClient):
                call["transport_diagnostics"] = self.backend._consume_invocation_diagnostics()

    def validation(self, **kwargs):
        if self.calls:
            self.calls[-1]["validation"] = {
                "status": kwargs["status"],
                "error": str(kwargs["error"]) if kwargs["error"] else None,
            }


def execute_case(case: WorkerCase, backend, profile: dict) -> dict:
    worker_type = SituationLLM if case.kind in ("PERCEPT_TRIAGE", "SELF_SCHEMA_REVIEW") else UserPromptLLM
    # Close the unused HTTP client immediately: only injected inference can run.
    stages = {"V2_RESPONSE_POLICY": PerceptStage.EVIDENCE_POLICY,
              "PRECOGNITIVE_USER_PROMPT_WORK": PerceptStage.PRECOGNITIVE,
              "PERCEPT_TRIAGE": SituationStage.TRIAGE,
              "SELF_SCHEMA_REVIEW": SituationStage.SELF_REVIEW}
    worker = worker_type(model=profile["model"], stage=stages[case.kind])
    worker._client.close()
    worker.selection = None
    recording = RecordingTransport(backend, profile, case)
    def guarded_inference(kind, *args):
        worker._require_stage_specialization(kind)
        return recording.structured(kind, *args)
    worker._structured_with_evidence = guarded_inference
    worker._record_validation_outcome = recording.validation
    record = {"case_id": case.case_id, "case_sha256": digest(case.model_dump(mode="json")),
              "group_id": case.group_id, "split": case.split, "kind": case.kind,
              "profile_id": profile["id"], "expected": case.expected,
              "label_source": case.label_source, "score_fields": case.score_fields,
              "calls": recording.calls}
    start = time.perf_counter()
    try:
        actual = invoke_worker(case, worker)
        record.update(status="VALID", actual=actual)
        record["correct"] = None if case.expected is None else all(
            actual.get(key) == case.expected[key] for key in case.score_fields)
        record["field_correct"] = {} if case.expected is None else {
            key: actual.get(key) == case.expected[key] for key in case.score_fields}
    except Exception as exc:
        record.update(status="ERROR", actual=None, correct=False if case.expected is not None else None,
                      error={"type": type(exc).__name__, "message": str(exc)})
    record["elapsed_seconds"] = time.perf_counter() - start
    record["deterministic_bypass"] = record["status"] == "VALID" and not recording.calls
    record["record_sha256"] = digest(record)
    return record


def summarize(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["profile_id"], record["kind"], record["split"])].append(record)
    summary = {}
    for (profile, kind, split), rows in grouped.items():
        semantic = [r for r in rows if not r["deterministic_bypass"]]
        labelled = [r for r in semantic if r["correct"] is not None]
        times = sorted(r["elapsed_seconds"] for r in semantic)
        first_calls = [r["calls"][0] for r in semantic if r["calls"]]
        confusions = defaultdict(Counter)
        selection_fp = selection_fn = 0
        for row in labelled:
            for field in row["score_fields"]:
                actual = row["actual"].get(field) if row["status"] == "VALID" else "<ERROR>"
                confusions[field][json.dumps(row["expected"][field]) + " -> " + json.dumps(actual)] += 1
            index_field = "capability_indices" if kind == "PRECOGNITIVE_USER_PROMPT_WORK" else "opposition_indices" if kind == "SELF_SCHEMA_REVIEW" else None
            if index_field and row["status"] == "VALID":
                wanted = set(row["expected"][index_field])
                selected = set(row["actual"][index_field])
                selection_fp += len(selected - wanted)
                selection_fn += len(wanted - selected)
        summary[f"{profile}/{kind}/{split}"] = {
            "cases": len(rows), "semantic_cases": len(semantic),
            "deterministic_bypasses": len(rows) - len(semantic),
            "labelled_semantic_cases": len(labelled),
            "contract_accuracy": sum(r["correct"] for r in labelled) / len(labelled) if labelled else None,
            "worker_validity_rate": sum(r["status"] == "VALID" for r in semantic) / len(semantic) if semantic else None,
            "first_attempt_validity_rate": sum(c.get("validation", {}).get("status") == "VALID" for c in first_calls) / len(first_calls) if first_calls else None,
            "retries": sum(max(0, len(r["calls"]) - 1) for r in semantic),
            "field_confusions": {k: dict(v) for k, v in confusions.items()},
            "selection_false_positives_valid_outputs": selection_fp,
            "selection_false_negatives_valid_outputs": selection_fn,
            "latency_median_seconds": statistics.median(times) if times else None,
            "latency_p95_seconds": times[min(len(times) - 1, int(.95 * len(times)))] if times else None,
            "errors": dict(Counter(r.get("error", {}).get("type", "none") for r in rows if r["status"] == "ERROR")),
        }
    # Compare paired decisions, not just aggregate accuracy, across precisions.
    paired = defaultdict(dict)
    for r in records:
        paired[(r["case_sha256"], r.get("repeat", 0))][r["profile_id"]] = r
    disagreements = []
    for (case_hash, repeat), rows in paired.items():
        for name, reference in rows.items():
            if not name.endswith("-fp32"):
                continue
            for suffix in ("bf16", "fp16"):
                other_name = name.removesuffix("fp32") + suffix
                if other_name not in rows:
                    continue
                other = rows[other_name]
                # Free-form rationale is deliberately not a decision metric.
                a = reference.get("actual") or {}
                b = other.get("actual") or {}
                a = {k: v for k, v in a.items() if k != "rationale"}
                b = {k: v for k, v in b.items() if k != "rationale"}
                disagreements.append({"case_sha256": case_hash, "repeat": repeat,
                                      "fp32": name, "other": other_name,
                                      "different": a != b or reference["status"] != other["status"]})
    return {"by_worker": summary, "precision_pairs": disagreements,
            "note": "Seed labels are provisional; schema validity is not semantic accuracy."}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args):
    cases = load_cases(args.corpus)
    matrix = json.loads(args.matrix.read_text())
    selected = [p for p in matrix["profiles"] if not args.profile or p["id"] in args.profile]
    if args.profile and set(args.profile) - {p["id"] for p in selected}:
        raise ValueError("unknown profile")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    directory = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid4())[:8])
    directory.mkdir(parents=True)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        git, dirty = None, None
    manifest = {"version": BENCHMARK_VERSION, "started_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": git, "git_dirty": dirty, "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(),
                "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                "corpus_sha256": digest([c.model_dump(mode="json") for c in cases]),
                "contracts": contract_manifest(), "matrix": matrix, "repeats": args.repeats,
                "mode": "preflight" if args.preflight else "live", "profiles": []}
    manifest["benchmark_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), args.sidecar) if path.is_file()
    }
    write_json(directory / "manifest.json", manifest)
    with (directory / "cases.jsonl").open("x", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    records = []
    for profile in selected:
        backend = None
        status = {"id": profile["id"], "status": "STARTING"}
        manifest["profiles"].append(status)
        try:
            if profile.get("blocked_reason"):
                raise ValueError(profile["blocked_reason"])
            started = time.perf_counter()
            if profile["backend"] == "ollama":
                backend = OllamaClient(base_url=args.ollama_url, model=profile["model"])
                identity = backend.runtime_snapshot()
                if identity["status"] != "COMPLETE":
                    raise ValueError(json.dumps(identity))
            else:
                backend = Sidecar(args.inference_python, args.sidecar, profile, args.timeout)
                identity = backend.identity
            status.update(status="READY", identity=identity, load_seconds=time.perf_counter() - started)
            write_json(directory / "manifest.json", manifest)
            if args.preflight:
                continue
            with (directory / "results.jsonl").open("a", encoding="utf-8") as handle:
                for repeat in range(args.repeats):
                    for case in cases:
                        if case.kind not in profile["kinds"]:
                            continue
                        record = execute_case(case, backend, profile)
                        record.pop("record_sha256")
                        record["repeat"] = repeat
                        record["record_sha256"] = digest(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(record)
                        print(f"{profile['id']} {case.case_id}: {record['status']} correct={record['correct']}", flush=True)
            if isinstance(backend, OllamaClient):
                after = backend.runtime_snapshot()
                status["identity_after"] = after
                if after.get("model", {}).get("digest") != identity["model"]["digest"]:
                    raise ValueError("Ollama model digest changed during benchmark")
            status["status"] = "COMPLETE"
        except Exception as exc:
            status.update(status="UNAVAILABLE", error_type=type(exc).__name__, error=str(exc))
            print(f"{profile['id']}: UNAVAILABLE: {exc}", flush=True)
        finally:
            if isinstance(backend, OllamaClient):
                backend._client.close()
            elif backend is not None:
                backend.close()
            write_json(directory / "manifest.json", manifest)
    write_json(directory / "summary.json", summarize(records))
    print(directory)
    return 2 if any(p["status"] == "UNAVAILABLE" for p in manifest["profiles"]) else 0


def export_reviewed(args):
    """Only explicit labels bound to frozen case hashes can enter a train export."""
    cases = load_cases(args.run / "cases.jsonl")
    reviews = [json.loads(x) for x in args.reviews.read_text().splitlines() if x.strip()]
    by_hash = {digest(c.model_dump(mode="json")): c for c in cases}
    records = [json.loads(x) for x in (args.run / "results.jsonl").read_text().splitlines() if x.strip()]
    manifest = json.loads((args.run / "manifest.json").read_text())
    if digest([c.model_dump(mode="json") for c in cases]) != manifest["corpus_sha256"]:
        raise ValueError("frozen corpus hash mismatch")
    for record in records:
        hashed = {k: v for k, v in record.items() if k != "record_sha256"}
        if digest(hashed) != record["record_sha256"]:
            raise ValueError("captured result hash mismatch")
    rows = []
    seen = set()
    for review in reviews:
        if review.get("decision") != "APPROVE" or not review.get("reviewer"):
            continue
        key = review["case_sha256"]
        case = by_hash[key]
        if case.split != "train":
            raise ValueError("held-out dev/test cases must never enter a training export")
        if key in seen:
            raise ValueError("duplicate review for case")
        seen.add(key)
        label = review["label"]
        # Pass the reviewed semantic answer through the actual worker validators.
        class LabelBackend:
            def request(self, value):
                return {"output": json.dumps(label)}
        checked = execute_case(case, LabelBackend(), {"id": "review", "model": "review"})
        if checked["status"] != "VALID":
            raise ValueError(f"invalid reviewed label: {checked.get('error')}")
        if checked["deterministic_bypass"]:
            continue  # No semantic task exists to train here.
        invocations = [r for r in records if r["case_sha256"] == key and r["calls"]]
        if not invocations:
            raise ValueError("no captured worker invocation for reviewed label")
        request = invocations[0]["calls"][0]["request"]
        rows.append({"kind": case.kind, "case_sha256": key, "group_id": case.group_id,
                     "split": case.split, "inputs": case.inputs, "worker_request": request,
                     "label": label, "review": review,
                     "source_run": str(args.run), "source_manifest_sha256": digest(manifest),
                     "training_format": "contract-neutral/v1"})
    # Exclusive creation prevents accidentally replacing a prior reviewed dataset.
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} reviewed training cases to {args.output}")


def capture_journal(args):
    """Preserve real teacher invocations; never infer correctness from acceptance."""
    paths = sorted(args.source.rglob("*.json*")) if args.source.is_dir() else [args.source]
    seen = set()
    count = 0
    with args.output.open("x", encoding="utf-8") as output:
        for path in paths:
            text = path.read_text(encoding="utf-8")
            entries = ([json.loads(line) for line in text.splitlines() if line.strip()]
                       if path.suffix == ".jsonl" else [json.loads(text)])
            for entry in entries:
                if entry.get("artifact_type") != "LLM_INVOCATION":
                    continue
                payload = entry["payload"]
                if payload.get("kind") not in KINDS:
                    continue
                key = digest(entry)
                if key in seen:
                    continue
                seen.add(key)
                output.write(json.dumps({"capture_version": BENCHMARK_VERSION,
                                         "source_path": str(path), "source_sha256": key,
                                         "label_status": "UNREVIEWED_TEACHER_OUTPUT",
                                         "artifact": entry}, ensure_ascii=False) + "\n")
                count += 1
    print(f"Preserved {count} real worker invocations to {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--corpus", type=Path, default=Path("benchmarks/specialists/worker_cases.jsonl"))
    runner.add_argument("--matrix", type=Path, default=Path("benchmarks/specialists/profiles.json"))
    runner.add_argument("--profile", action="append")
    runner.add_argument("--output", type=Path, default=Path("benchmarks/generated/specialists"))
    runner.add_argument("--ollama-url", default="http://localhost:11434")
    runner.add_argument("--inference-python", default=sys.executable)
    runner.add_argument("--sidecar", type=Path, default=Path("scripts/specialist_inference.py"))
    runner.add_argument("--timeout", type=float, default=600)
    runner.add_argument("--repeats", type=int, default=1)
    runner.add_argument("--preflight", action="store_true")
    exporter = sub.add_parser("export-reviewed")
    exporter.add_argument("--run", type=Path, required=True)
    exporter.add_argument("--reviews", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture-journal")
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    return export_reviewed(args) if args.command == "export-reviewed" else capture_journal(args)


if __name__ == "__main__":
    raise SystemExit(main())
