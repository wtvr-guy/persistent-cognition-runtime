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


def run_command(command) -> None:
    """Console entry boundary: report a safe summary without a raw traceback."""
    import sys
    try:
        command()
    except Exception as error:
        print(exception_summary(error), file=sys.stderr)
        raise SystemExit(1) from None
