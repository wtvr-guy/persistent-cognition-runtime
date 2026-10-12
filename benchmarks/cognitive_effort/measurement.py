"""Crash-inspectable audit chain, per-trial samples and exact failure artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
from typing import Any, Callable

import psutil

AUDIT_VERSION = "cognitive-effort-audit/v1"
ZERO_HASH = "0" * 64
SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,96}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_artifact(path: Path, value: Any) -> str:
    data = canonical(value) + b"\n"
    if path.exists():
        raise FileExistsError(f"benchmark artifact already exists: {path}")
    temporary = path.with_name(path.name + ".pending")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(data)


class AuditTrail:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        (root / "cases").mkdir()
        self.stream = (root / "events.jsonl").open("x", encoding="utf-8", newline="\n")
        self.seq = 0
        self.previous_hash = ZERO_HASH

    def append(self, kind: str, **fields: Any) -> None:
        self.seq += 1
        entry = {
            "seq": self.seq,
            "time_utc": utc_now(),
            "kind": kind,
            "previous_sha256": self.previous_hash,
            "fields": fields,
        }
        entry["sha256"] = sha256(canonical(entry))
        self.stream.write(canonical(entry).decode("utf-8") + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous_hash = entry["sha256"]

    def artifact(self, relative: str, value: Any) -> None:
        path = self.root / relative
        digest = write_artifact(path, value)
        self.append("artifact", path=relative, sha256=digest)

    def close(self) -> None:
        self.stream.close()


class Sampler:
    """Samples entire-host pressure, benchmark process tree and named Ollama processes.

    Process RSS can double-count shared memory. Samples are observational, not
    measurements of electrical power; psutil itself has an overhead.
    """

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_signal = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="cognitive-effort-sampler", daemon=True)

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self.thread.start()

    def stop(self) -> None:
        self.stop_signal.set()
        self.thread.join(timeout=max(5.0, self.interval * 3))
        if self.thread.is_alive():
            self.errors.append("sampler thread did not terminate before deadline")

    def _loop(self) -> None:
        while True:
            try:
                self.samples.append(self._capture())
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            if len(self.samples) >= 10000:
                self.errors.append("sample bound reached (10000); later samples omitted")
                return
            if self.stop_signal.wait(self.interval):
                try:
                    self.samples.append(self._capture())
                except Exception as exc:
                    self.errors.append(f"final: {type(exc).__name__}: {exc}")
                return

    @staticmethod
    def _capture() -> dict[str, Any]:
        own = psutil.Process(os.getpid())
        owned = {own.pid}
        try:
            owned.update(item.pid for item in own.children(recursive=True))
        except psutil.Error:
            pass
        own_cpu = own_rss = ollama_cpu = ollama_rss = 0.0
        own_count = ollama_count = 0
        for process in psutil.process_iter(["pid", "name", "memory_info", "cpu_times"]):
            try:
                info = process.info
                name = str(info["name"] or "").casefold()
                is_own = info["pid"] in owned
                is_ollama = not is_own and "ollama" in name
                if not is_own and not is_ollama:
                    continue
                mem = info["memory_info"]
                cpu = info["cpu_times"]
                cpu_total = float(cpu.user + cpu.system) if cpu else 0.0
                rss = float(mem.rss) if mem else 0.0
                if is_own:
                    own_cpu += cpu_total
                    own_rss += rss
                    own_count += 1
                else:
                    ollama_cpu += cpu_total
                    ollama_rss += rss
                    ollama_count += 1
            except (psutil.Error, TypeError, AttributeError):
                continue
        vm = psutil.virtual_memory()
        return {
            "epoch_seconds": time.time(),
            "host_cpu_percent": psutil.cpu_percent(interval=None),
            "host_available_bytes": vm.available,
            "host_used_bytes": vm.used,
            "own_rss_bytes": int(own_rss),
            "own_cpu_cumulative_seconds": round(own_cpu, 6),
            "own_process_count": own_count,
            "ollama_rss_bytes": int(ollama_rss),
            "ollama_cpu_cumulative_seconds": round(ollama_cpu, 6),
            "ollama_named_process_count": ollama_count,
        }


def measurement_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def delta(key: str) -> float | None:
        if len(samples) < 2:
            return None
        return round(max(0.0, samples[-1][key] - samples[0][key]), 6)

    return {
        "sample_count": len(samples),
        "host_cpu_average_percent": (
            round(sum(s["host_cpu_percent"] for s in samples) / len(samples), 3)
            if samples else None
        ),
        "minimum_host_available_bytes": (
            min(s["host_available_bytes"] for s in samples) if samples else None
        ),
        "peak_own_rss_bytes": max((s["own_rss_bytes"] for s in samples), default=None),
        "peak_ollama_named_rss_bytes": max(
            (s["ollama_rss_bytes"] for s in samples), default=None
        ),
        "own_cpu_delta_seconds": delta("own_cpu_cumulative_seconds"),
        "ollama_named_cpu_delta_seconds": delta("ollama_cpu_cumulative_seconds"),
        "caution": (
            "Named Ollama process identification is best-effort; RSS may count "
            "shared pages more than once; CPU percentage is not energy."
        ),
    }


def run_trial(
    audit: AuditTrail,
    trial_id: str,
    phase: str,
    case: dict[str, Any],
    action: Callable[[], dict[str, Any]],
    *,
    sample_seconds: float,
) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(trial_id):
        raise ValueError("unsafe trial identifier")
    audit.append("trial_start", trial_id=trial_id, phase=phase)
    sampler = Sampler(sample_seconds)
    started_utc = utc_now()
    started_epoch = time.time()
    sampler.start()
    tick = time.perf_counter_ns()
    result: dict[str, Any] = {}
    failure: dict[str, Any] | None = None
    try:
        result = action()
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    elapsed_ns = time.perf_counter_ns() - tick
    ended_epoch = time.time()
    ended_utc = utc_now()
    sampler.stop()
    status = "ERROR" if failure else ("PASS" if result.get("passed") else "FAIL")
    payload = {
        "trial_id": trial_id,
        "phase": phase,
        "status": status,
        "case": case,
        "result": result,
        "failure": failure,
        "start_utc": started_utc,
        "end_utc": ended_utc,
        "start_epoch_seconds": started_epoch,
        "end_epoch_seconds": ended_epoch,
        "elapsed_ns": elapsed_ns,
        "samples": sampler.samples,
        "sample_errors": sampler.errors,
        "measurements": measurement_summary(sampler.samples),
    }
    audit.artifact(f"cases/{trial_id}.json", payload)
    audit.append(
        "trial_end", trial_id=trial_id, phase=phase, status=status,
        elapsed_ns=elapsed_ns, artifact_path=f"cases/{trial_id}.json",
    )
    return payload


def verify_audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    previous = ZERO_HASH
    starts: set[str] = set()
    ends: set[str] = set()
    sequence = 0
    run_started = False
    run_ended = False
    run_error = False
    referenced_artifacts: set[str] = set()
    source = root / "events.jsonl"
    if not source.is_file():
        return {"valid": False, "errors": ["events.jsonl is missing"], "events": 0}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
            digest = event.pop("sha256")
            if event["seq"] != sequence + 1 or event["previous_sha256"] != previous:
                raise ValueError("audit sequence/chain discontinuity")
            if digest != sha256(canonical(event)):
                raise ValueError("audit entry hash mismatch")
            previous = digest
            sequence += 1
            if event["kind"] == "run_start":
                run_started = True
            if event["kind"] == "run_end":
                run_ended = True
            if event["kind"] == "run_error":
                run_error = True
            if event["kind"] == "artifact":
                relative = event["fields"]["path"]
                referenced_artifacts.add(relative)
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe artifact path")
                blob = root / path
                if not blob.is_file() or sha256(blob.read_bytes()) != event["fields"]["sha256"]:
                    raise ValueError(f"artifact missing or digest mismatch: {relative}")
            if event["kind"] == "trial_start":
                starts.add(event["fields"]["trial_id"])
            if event["kind"] == "trial_end":
                ends.add(event["fields"]["trial_id"])
        except (KeyError, ValueError, TypeError, OSError) as exc:
            errors.append(f"events.jsonl:{line_number}: {type(exc).__name__}: {exc}")
            break
    if starts != ends:
        errors.append(f"incomplete trials: {sorted(starts - ends)}")
    if not run_started or not run_ended or run_error:
        errors.append("run incomplete or failed: inspect run_error and any pending trials")
    for candidate in (root / "cases").glob("*.json"):
        if f"cases/{candidate.name}" not in referenced_artifacts:
            errors.append(f"unreferenced trial artifact: {candidate.name}")
    return {
        "valid": not errors, "errors": errors, "events": sequence,
        "completed_trials": len(ends), "chain_head_sha256": previous,
    }
