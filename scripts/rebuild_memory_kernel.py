"""Rebuild and verify all v0.2 derived memory state from authoritative events."""
from __future__ import annotations

from prometheist import db
from prometheist.postgres_memory_kernel import rebuild, verify


def main() -> None:
    conn = db.get_connection()
    try:
        summary = rebuild(conn)
        verification = verify(conn)
    finally:
        conn.close()

    print("Memory Kernel rebuild complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"integrity_valid: {verification.valid}")
    print(f"integrity_events_checked: {verification.checked_events}")
    if not verification.valid:
        print(f"integrity_failure: {verification.reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
