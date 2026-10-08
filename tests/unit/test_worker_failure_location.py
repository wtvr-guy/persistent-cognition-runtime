"""Worker failures expose code location without exporting rejected input."""
from types import SimpleNamespace

import pytest

from persistent_cognition.diagnostics import exception_location, run_command
from persistent_cognition.private_storage import seek_lock_byte


def test_worker_failure_reports_runtime_location_without_private_error_text(capsys):
    def rejected_seek(*args):
        raise RuntimeError("private autobiography and secret database password")

    with pytest.raises(SystemExit) as exited:
        run_command(lambda: seek_lock_byte(SimpleNamespace(seek=rejected_seek)))
    assert exited.value.code == 1
    output = capsys.readouterr().err
    assert output.startswith("RuntimeError at persistent_cognition.private_storage.seek_lock_byte:")
    assert "autobiography" not in output and "password" not in output
    assert "Traceback" not in output


def test_errors_without_runtime_frames_do_not_invent_a_location():
    assert exception_location(RuntimeError("private data")) is None
