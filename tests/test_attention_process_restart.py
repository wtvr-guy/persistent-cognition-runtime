from __future__ import annotations

import json
import subprocess
import sys
import uuid


def test_jit_attention_survives_forced_process_kill_and_resumes():
    run_key = uuid.uuid4().hex
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests._attention_process",
            "checkpoint-wait",
            run_key,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert process.stdout is not None
        checkpoint_line = process.stdout.readline().strip()
        assert checkpoint_line, (
            process.stderr.read() if process.stderr is not None else "checkpoint process produced no output"
        )
        checkpoint = json.loads(checkpoint_line)

        # Destroy the process without giving it a chance to serialize any
        # additional Python state. Only the committed PostgreSQL checkpoint may
        # survive this point.
        process.kill()
        process.wait(timeout=10)
        assert process.returncode != 0

        resumed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests._attention_process",
                "resume",
                run_key,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout.strip())

        assert result["recovered_active_task_id"] == checkpoint["active_task_id"]
        assert result["recovered_state"] == {
            "step": 1,
            "artifact": "persisted-before-kill",
        }
        assert result["next_active_task_id"] == checkpoint["queued_task_id"]
        assert result["completed_status"] == "COMPLETED"
        assert result["cycle"] > checkpoint["cycle"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
