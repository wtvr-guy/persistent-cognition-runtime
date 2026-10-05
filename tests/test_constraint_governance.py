from pathlib import Path
import runpy


def _load_auditor():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "scripts" / "audit_constraints.py"))


def test_every_behavioral_numeric_constraint_is_registered_and_classified():
    auditor = _load_auditor()
    findings = auditor["discover_python_constraints"]()
    registry = auditor["load_registry"]()
    uncovered = auditor["uncovered_findings"](findings, registry)
    stale = auditor["stale_registry_keys"](findings, registry)
    mismatches = auditor["value_mismatches"](findings, registry)
    errors = auditor["registry_errors"](registry)

    messages = []
    messages.extend(
        f"UNREGISTERED {item.key} value={item.value!r} line={item.line} "
        f"role={item.role} :: {item.source}"
        for item in uncovered
    )
    messages.extend(f"STALE {key}" for key in stale)
    messages.extend(f"MISMATCH {message}" for message in mismatches)
    messages.extend(f"INVALID {error}" for error in errors)
    assert not messages, "Constraint governance violations:\n" + "\n".join(
        f"- {message}" for message in messages
    )
