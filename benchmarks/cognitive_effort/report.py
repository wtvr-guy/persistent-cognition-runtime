"""Replayable aggregate report; failures retain trial IDs and exact artifact paths."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.cognitive_effort.energy import energy_joules, read_power_samples
from benchmarks.cognitive_effort.measurement import verify_audit, utc_now


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 6)


def report(
    root: Path, *, power_csv: Path | None = None, idle_watts: float | None = None,
    max_power_gap_seconds: float = 5.0, meter_scope: str = "wall",
) -> dict[str, Any]:
    verification = verify_audit(root)
    if not verification["valid"]:
        raise ValueError(f"audit verification failed: {verification['errors']}")
    if idle_watts is not None and (not math.isfinite(idle_watts) or idle_watts < 0):
        raise ValueError("idle_watts must be finite and nonnegative")
    if idle_watts is not None and power_csv is None:
        raise ValueError("idle_watts requires power_csv")
    if power_csv is not None and meter_scope not in {"wall", "cpu_package", "device"}:
        raise ValueError("meter_scope must be wall, cpu_package, or device")
    power = read_power_samples(power_csv) if power_csv else None
    cases = sorted((root / "cases").glob("*.json"))
    if verification["completed_trials"] != len(cases):
        raise ValueError("case artifact count disagrees with completed trial count")
    groups: dict[str, list[dict[str, Any]]] = {}
    rows = []
    for case_path in cases:
        trial = json.loads(case_path.read_text(encoding="utf-8"))
        phase = trial["phase"]
        metric = trial["measurements"]
        duration_s = trial["elapsed_ns"] / 1_000_000_000
        energy: dict[str, Any] = {
            "joules": None, "incremental_joules": None,
            "status": "NOT_MEASURED",
        }
        if power is not None:
            try:
                gross = energy_joules(
                    power, trial["start_epoch_seconds"], trial["end_epoch_seconds"],
                    max_gap_seconds=max_power_gap_seconds,
                )
                energy = {
                    "joules": round(gross, 6),
                    "incremental_joules": (
                        round(gross - idle_watts * (trial["end_epoch_seconds"] -
                                                   trial["start_epoch_seconds"]), 6)
                        if idle_watts is not None else None
                    ),
                    "status": "MEASURED",
                }
            except ValueError as exc:
                energy["status"] = f"UNAVAILABLE: {exc}"
        groups.setdefault(phase, []).append({
            "trial": trial, "energy": energy, "path": f"cases/{case_path.name}",
        })
        rows.append({
            "trial_id": trial["trial_id"], "phase": phase, "status": trial["status"],
            "elapsed_ms": round(duration_s * 1000, 6),
            "own_cpu_seconds": metric["own_cpu_delta_seconds"],
            "ollama_cpu_seconds": metric["ollama_named_cpu_delta_seconds"],
            "peak_own_rss_bytes": metric["peak_own_rss_bytes"],
            "peak_ollama_named_rss_bytes": metric["peak_ollama_named_rss_bytes"],
            "host_cpu_average_percent": metric["host_cpu_average_percent"],
            "model_eval_count": trial["result"].get("model_metrics", {}).get("eval_count"),
            "gross_joules": energy["joules"],
            "incremental_joules": energy["incremental_joules"],
            "power_status": energy["status"],
            "artifact": f"cases/{case_path.name}",
        })
    phases: dict[str, Any] = {}
    for phase, items in sorted(groups.items()):
        all_times = [v["trial"]["elapsed_ns"] / 1_000_000 for v in items]
        measured = [v["energy"]["joules"] for v in items
                    if v["energy"]["status"] == "MEASURED"]
        phases[phase] = {
            "trial_count": len(items),
            "passes": sum(v["trial"]["status"] == "PASS" for v in items),
            "failures": [
                {"trial_id": v["trial"]["trial_id"],
                 "status": v["trial"]["status"], "artifact": v["path"],
                 "error": (v["trial"]["failure"] or {}).get("message")}
                for v in items if v["trial"]["status"] != "PASS"
            ],
            "duration_ms_median": _percentile(all_times, .5),
            "duration_ms_p95": _percentile(all_times, .95),
            "sample_warnings": [
                {"trial_id": v["trial"]["trial_id"],
                 "messages": v["trial"]["sample_errors"]}
                for v in items if v["trial"]["sample_errors"]
            ],
            "meter_covered_trials": len(measured),
            "gross_joules_total_of_covered_trials": (
                round(sum(measured), 6) if measured else None
            ),
            "gross_joules_median_of_covered_trials": _percentile(measured, .5),
        }

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "cognitive-effort-summary/v1",
        "generated_utc": utc_now(),
        "manifest": manifest,
        "audit": verification,
        "electric_energy": {
            "status": "EXTERNAL_METER" if power is not None else "NOT_MEASURED",
            "source_csv": str(power_csv) if power_csv else None,
            "source_sha256": hashlib.sha256(power_csv.read_bytes()).hexdigest() if power_csv else None,
            "scope": meter_scope if power_csv else None,
            "idle_watts": idle_watts,
            "note": (
                "Meter energy is time-integrated watts and includes background load. "
                "Incremental energy is an idle-baseline estimate, not direct attribution. "
                "CPU seconds, host CPU percent, tokens and RSS are NOT electrical energy."
            ),
        },
        "phases": phases,
        "overall_pass": (bool(rows) and all(row["status"] == "PASS" for row in rows)
                         and (power is None or all(row["power_status"] == "MEASURED"
                                                   for row in rows))),
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result
