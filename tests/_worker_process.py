"""Fresh-process helper for Increment F abandoned-lease recovery tests."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from uuid import UUID

from prometheist import db
from prometheist.attention_observation import HostResourceMetrics
from prometheist.worker_store import guarded_claim_worker_step


CAPTURED_AT = datetime(2026, 8, 26, 16, 0, tzinfo=timezone.utc)


class FixedProbe:
    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=10,
            load_1m=0,
            memory_total_mib=16_384,
            memory_available_mib=12_000,
        )


def claim_and_disappear(step_id: UUID) -> None:
    conn = db.get_connection()
    try:
        attempt = guarded_claim_worker_step(
            conn,
            step_id=step_id,
            worker_id="destroyed-process",
            probe=FixedProbe(),
            clock=lambda: CAPTURED_AT,
            lease_seconds=1,
        )
        if attempt.envelope is None:
            raise RuntimeError(attempt.observation.reason)
        print(
            json.dumps(
                {
                    "claim_id": str(attempt.envelope.claim.claim_id),
                    "idempotency_key": attempt.envelope.step.idempotency_key,
                }
            ),
            flush=True,
        )
        # Deliberately bypass connection cleanup and every graceful worker
        # lifecycle method. PostgreSQL is the only surviving state.
        os._exit(0)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests._worker_process <step-id>")
    claim_and_disappear(UUID(sys.argv[1]))
