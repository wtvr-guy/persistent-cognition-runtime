from __future__ import annotations

import subprocess
import sys
import textwrap

from prometheist import cli


class _StreamProbe:
    def __init__(self) -> None:
        self.configuration = None

    def reconfigure(self, **kwargs) -> None:
        self.configuration = kwargs


def test_cli_configures_utf8_output_for_windows_subprocess_boundaries(monkeypatch):
    stdout = _StreamProbe()
    stderr = _StreamProbe()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    cli._configure_utf8_streams()

    assert stdout.configuration == {"encoding": "utf-8", "errors": "replace"}
    assert stderr.configuration == {"encoding": "utf-8", "errors": "replace"}


def test_module_entrypoint_configures_utf8_before_argument_parsing():
    probe = textwrap.dedent(
        """
        import runpy
        import sys

        class Probe:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = []

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def write(self, value):
                return self.wrapped.write(value)

            def flush(self):
                return self.wrapped.flush()

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs.copy())
                return self.wrapped.reconfigure(**kwargs)

        stdout = Probe(sys.stdout)
        stderr = Probe(sys.stderr)
        sys.stdout = stdout
        sys.stderr = stderr
        sys.argv = ["prometheist.cli", "--help"]
        try:
            runpy.run_module("prometheist.cli", run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        print("__STDOUT_CALLS__=" + repr(stdout.calls))
        print("__STDERR_CALLS__=" + repr(stderr.calls))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    expected = "[{'encoding': 'utf-8', 'errors': 'replace'}]"
    assert f"__STDOUT_CALLS__={expected}" in result.stdout
    assert f"__STDERR_CALLS__={expected}" in result.stdout
