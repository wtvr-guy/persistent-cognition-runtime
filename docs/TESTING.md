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

Current extraction results and the planned laptop run are recorded in [EXTRACTION.md](EXTRACTION.md). Historical benchmark files retain their original date and are not new measurements.

## Historical evidence in Prometheist

[Prometheist](https://github.com/wtvr-guy/prometheist) is the upstream record of the engine's development and prior validation, including successful local tests with real PostgreSQL and live Ollama. The links below pin the evidence archive at extraction source `106bb22be4ad60f2455ece8bc8c4e2806225d0fe`; each result applies to the revision and environment identified in its own record, rather than automatically validating that archive commit or this extraction.

| Evidence | What the record establishes |
| --- | --- |
| [V2 continuity and epistemic-memory validation, 2026-09-11](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/audits/V07_EPISTEMIC_RETRIEVAL_SCOPING_2026-09-11.md#3-verification--results) | Reports passing native four-turn stateless continuity and epistemic-memory red-team tests after local Windows/PostgreSQL/Ollama remediation, alongside 26 worker/response-policy tests. |
| [V0.7 closure and frozen baseline](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/CLOSURE_STATUS.md) | Records maintainer acceptance and closure on 2026-09-12, with baseline `39a3223c38f1b1f8fae7f9e667c6cd7460774ffe` and the historical gate/branch disposition. |
| [Earlier native source-authority acceptance, 2026-08-28](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/audits/history/pr19/RT04_REMEDIATION_2026-08-28.md#acceptance-evidence) | Records 5 passing focused Windows/PostgreSQL/Ollama tests in 69.02 seconds on the historical red-team branch. This is evidence for that branch's mechanism. |
| [Native failures and remediation, 2026-09-03](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/audits/V07_NATIVE_ACCEPTANCE_2026-09-03.md) | Preserves the earlier 8-pass/6-fail and 11-pass/3-fail runs, their provenance limits, and the correction to artifact-first acceptance with human response review. |
| [Historical benchmark results](https://github.com/wtvr-guy/prometheist/tree/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/benchmarks/results) and [experiment records](https://github.com/wtvr-guy/prometheist/tree/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/experiments) | Retain native constraint calibration, memory/retrieval measurements, experiment methods, successes, and negative results. Consult each record's model, revision, scope, and result; these files are not a single universal pass claim. |
| [Upstream tests](https://github.com/wtvr-guy/prometheist/tree/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/tests) and [testing/acceptance contract](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/TESTING_AND_ACCEPTANCE.md) | Provide the test definitions and explain statelessness, restart, evidence provenance, structural acceptance, and human review of natural model answers. |

The successful upstream native runs are historical evidence for the architecture. The current repository has its own [extraction regression results](EXTRACTION.md#verification) and still needs live-model acceptance on its own exact checkout. Save the new run's full console output and response artifacts with its commit, model, and environment so that evidence can be added here.
