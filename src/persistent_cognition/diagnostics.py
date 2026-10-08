"""Private exact evidence and exportable exception summaries have distinct roles."""
from __future__ import annotations

import os
import re

from pydantic import ValidationError

DEFAULT_MAX_DIAGNOSTIC_CHARS = 2000


def sanitize_message(message: str) -> str:
    from persistent_cognition.resource_limits import limit
    maximum = limit("PCR_MAX_DIAGNOSTIC_CHARS", DEFAULT_MAX_DIAGNOSTIC_CHARS)
    # Redact before bounding: a secret must not leak as a truncated prefix.
    for name, value in os.environ.items():
        if value and (name in {"DATABASE_URL", "TEST_DATABASE_URL"}
                      or re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I)):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)(https?|postgres(?:ql)?)://[^\s/@]+:[^\s/@]+@",
                     r"\1://[redacted]@", message)
    return message[:maximum]


def exception_summary(error: BaseException) -> str:
    """Never serialize arbitrary exception text or rejected private input."""
    if isinstance(error, ValidationError):
        kinds = sorted({entry["type"] for entry in error.errors(
            include_input=False, include_context=False, include_url=False)})
        return sanitize_message("ValidationError: " + ", ".join(kinds))
    return sanitize_message(type(error).__name__)


def exception_location(error: BaseException) -> str | None:
    """Locate the last runtime frame without exposing messages, paths or locals."""
    location = None
    frame = error.__traceback__
    while frame is not None:
        module = frame.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("persistent_cognition."):
            location = f"{module}.{frame.tb_frame.f_code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return sanitize_message(location) if location is not None else None


def run_command(command) -> None:
    """Console entry boundary: report a safe summary without a raw traceback."""
    import sys
    try:
        command()
    except Exception as error:
        summary = exception_summary(error)
        location = exception_location(error)
        print(f"{summary} at {location}" if location else summary, file=sys.stderr)
        raise SystemExit(1) from None
