"""Join external timestamped power-meter samples to immutable trial time windows.

No psutil metric is silently converted to joules. The meter must measure real
watts and cover the entire measured interval. Time alignment is an assumption
which operators should verify on their hardware.
"""
from __future__ import annotations

import bisect
import csv
from datetime import datetime, timezone
import math
from pathlib import Path


def read_power_samples(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp_utc", "power_watts"}.issubset(reader.fieldnames or ()):
            raise ValueError("power CSV requires timestamp_utc,power_watts columns")
        result: list[tuple[float, float]] = []
        for line, row in enumerate(reader, 2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timestamp requires UTC offset")
                epoch = timestamp.astimezone(timezone.utc).timestamp()
                watts = float(row["power_watts"])
                if not (math.isfinite(epoch) and math.isfinite(watts) and watts >= 0):
                    raise ValueError("invalid timestamp or nonnegative watts")
                if result and epoch <= result[-1][0]:
                    raise ValueError("timestamps must strictly increase")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
            result.append((epoch, watts))
    if len(result) < 2:
        raise ValueError("power CSV requires at least two samples")
    return result


def energy_joules(
    samples: list[tuple[float, float]],
    start: float,
    end: float,
    *,
    max_gap_seconds: float = 5.0,
) -> float:
    if not (math.isfinite(start) and math.isfinite(end) and 0 < max_gap_seconds):
        raise ValueError("invalid window or gap")
    if end <= start:
        raise ValueError("trial interval must have positive duration")
    times = [sample[0] for sample in samples]
    if start < times[0] or end > times[-1]:
        raise ValueError("power series does not cover the full trial")
    lower = max(0, bisect.bisect_right(times, start) - 1)
    upper = min(len(times) - 1, bisect.bisect_left(times, end))
    if any(
        times[i + 1] - times[i] > max_gap_seconds
        for i in range(lower, upper)
    ):
        raise ValueError("power series has a sampling gap exceeding the allowed maximum")

    def interpolated(at: float) -> float:
        index = bisect.bisect_right(times, at) - 1
        if index < 0:
            raise ValueError("missing preceding power sample")
        if index >= len(samples) - 1:
            return samples[-1][1]
        t0, p0 = samples[index]
        t1, p1 = samples[index + 1]
        return p0 + (p1 - p0) * ((at - t0) / (t1 - t0))

    interior = [(t, watts) for t, watts in samples if start < t < end]
    values = [(start, interpolated(start)), *interior, (end, interpolated(end))]
    return sum(
        (right_t - left_t) * (left_w + right_w) / 2
        for (left_t, left_w), (right_t, right_w) in zip(values, values[1:])
    )
