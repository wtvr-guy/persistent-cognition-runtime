# Validation

`scripts/verify_source_baseline.py` verifies the extracted file digests and reports unchanged versus adapted paths. This checks source fidelity; behavior requires the tests too.

## PostgreSQL regression gate

Set `TEST_DATABASE_URL` to a dedicated database containing `test` or `benchmark` in its name, then run:

```sh
uv sync --frozen
uv run ruff check .
uv run python scripts/verify_source_baseline.py
uv run python scripts/audit_registries.py
uv run python scripts/audit_constraints.py --fail-unregistered
uv run python benchmarks/run_deterministic_constraints.py
uv run pytest -q -ra
```

This exercises real persistence, process restart, bounded memory, temporal/source authority, compact journals, recovery, and scripted specialist contracts. Hosted CI provisions PostgreSQL 16 and retains the JUnit report. The pure policy subset can also run without PostgreSQL:

```sh
uv run pytest -q tests/unit --confcutdir=tests/unit
```

Frozen retrieval fixtures originally discovered during upstream application benchmarks remain under `tests/fixtures`. They test delivery of required canonical sources; application fidelity benchmark runners and subjective scoring are excluded.

## Live-model gate

No real Ollama server is provisioned by ordinary CI. Tests marked `ollama` report that dependency separately. The inherited Windows acceptance script requires a clean named branch at an exact SHA, PostgreSQL, and the configured Ollama model:

```powershell
.\scripts\run_v07_acceptance.ps1 -ExpectedCommit (git rev-parse HEAD)
```

It requires the actual stack and prints native response artifacts for human review. A passing deterministic suite does not certify natural-answer quality or native Windows behavior. Its historical filename is retained; it executes this checkout's tests.

Current extraction results are recorded in [EXTRACTION.md](EXTRACTION.md). Historical benchmark files retain their original date and are not new measurements.
