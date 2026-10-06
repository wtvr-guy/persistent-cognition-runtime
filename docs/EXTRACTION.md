# Engine extraction — 2026-10-05

## Source

Copied from upstream [Prometheist](https://github.com/wtvr-guy/prometheist) commit `106bb22be4ad60f2455ece8bc8c4e2806225d0fe`, the sequential model-residency fix based on main `d56517bf8eaf548b7d6e3b39e3d79549ad46c721`.

The original v0.7 closure tag was inspected as historical context. This extraction uses the later engine because the requested scope includes subsequent improvements: v2 workers, fixed deterministic retrieval, canonical neighbors, learned-memory source links, semantic facts, compact journals, model parameter/routing contracts, and sequential model unloading with peak-memory admission.

The source's later fixed retrieval update replaces Composer control. Restoring the old Composer loop would discard an engine improvement rather than preserve v2 compatibility.

## Boundaries and adaptations

The source application is unchanged. This repository has its own history and the same exact LICENSE file.

- Uses the `persistent_cognition` import namespace, `PCR_*` environment variables, durable IDs, prompts, and protocols for the current runtime.
- Renamed the installed distribution to `persistent-cognition-runtime` and exposed `pcr` and `pcr-percept` commands.
- Extracted model settings/catalog/probe/locking into headless modules. The headless chat job retains model admission and verified sequential residency.
- Removed CLI application-profile commands, automatic sensor startup, GUI/file/firewall job actions, and deployment-only SQL binding.
- Removed Android/node/tunnel services, sensor discovery, OS-security controls, web assets, GUI servers, deployment-only setup, and application mission documents.
- Retained source-grounded learned-memory structures because they supply canonical retrieval evidence; application-fidelity runners and subjective scoring are excluded.
- Preserved frozen synthetic retrieval fixtures and their assertions. Test-only fixture loaders are separated from production code.
- Removed only schema/constraint registrations belonging to excluded modules; all remaining numerical bounds are still audited.
- Kept current technical specs and engineering rules, with absent historical references linked to the pinned upstream source.
- Retired operational identity/personality/imprinting and self-model/reflection contracts, not generic semantic evidence or counterevidence. Retired identity-specific tests remain retired.
- Completed namespace-owned continuous perception, arbitrary structured source inputs, deterministic scheduled/sensor/error/tool reactions, observable failures, graceful worker-boundary stop/restart, and trusted application callback startup in fresh workers.
- Isolated action feedback from pending entity reactions, quarantined failed situations with explicit reconciled recovery while healthy siblings continue, and based model admission only on actual model-using stages.
- Corrected historical upstream links to the real `wtvr-guy/prometheist` repository; current imports/commands remain neutral.

The current `SOURCE_BASELINE.json` records **44 byte-identical source files, 197 adapted files, 5 explicit additions, and 7 retirements**. Original `source_path`/`source_sha256` are preserved independently of current `extraction_path`/`extracted_sha256`; earlier extraction digests are retained when adapted or retired. The verifier also rejects unrecorded Python code. Runtime behavior is tested with retained and strengthened regressions; source parity alone is not a functional acceptance result.

## Verification

Current completion checks (2026-10-06 working tree based on `360ce54`; Python 3.14.8, PostgreSQL 16, Linux):

- `uv sync --frozen` restored the existing lockfile without adding project dependencies.
- `uv run ruff check .` and `git diff --check` passed.
- Source verifier passed: 44 unchanged, 197 adapted, 5 added, 7 retired; all 248 original source paths and source hashes were preserved.
- `uv run python scripts/audit_registries.py` resolved all active prompts, schemas, and stages; no operational self-model contract remains.
- `uv run python scripts/audit_constraints.py --fail-unregistered`: 306 discovered/registered; zero uncovered, stale, mismatched, or invalid entries.
- `uv run python benchmarks/run_deterministic_constraints.py` completed all four retained families with `INSUFFICIENT_DISCRIMINATION`, not claimed optimality.
- `uv run pytest -q tests/unit --confcutdir=tests/unit`: 97 passed.
- Full `TEST_DATABASE_URL=postgresql:///pcr_review_test uv run pytest -q -ra --basetemp=.pcr/review-full --junitxml=.pcr/review-full.xml` after the reviewed fixes: **586 passed, 13 skipped**. All skips report unreachable Ollama; none skips PostgreSQL integration. Regressions cover sensor intake during a trusted callback, unhealthy/healthy siblings, bounded explicit retries, ambiguous action reconciliation, retained stage/assignment history, live-claim capacity, and 512-MiB model-free consolidation admission.
- `uv build` produced the source distribution and wheel; `pcr --help`, `pcr-percept --help`, `serve --help`, and `tick --help` loaded.
- Post-fix CodeQL remains pending. No automated security/code-review pass is claimed for these changes.

Live Ollama and native Windows acceptance remain unverified. These working-tree results are not an assertion about an untested future commit or live-model quality.

Historical local extraction checks (2026-10-05, before the identity removal/service completion):

- Locked Python 3.14.7 environment installed successfully.
- Ruff passed.
- Registry resolution passed.
- Constraint audit: 317 discovered, 317 registered; zero uncovered, stale, mismatched, or invalid entries.
- Pure engine tests: 97 passed.
- 589 tests collected.
- Deterministic constraint calibration completed; provisional/insufficient-discrimination results remain provisional.
- Source and wheel distributions built; both command help paths loaded successfully.

Hosted PostgreSQL 16 CI passed on extraction commit [`3d7e845`](https://github.com/wtvr-guy/persistent-cognition-runtime/commit/3d7e8458f9eadac81f232b8270d616a684294ea8): **576 passed, 13 skipped in 115.43 seconds**. The [complete run](https://github.com/wtvr-guy/persistent-cognition-runtime/actions/runs/37344783424) also passed static checks, source verification, both registry audits, and deterministic calibration. Its JUnit report is attached as `engine-test-results`.

All 13 skips require a reachable Ollama instance. Those live-model checks, including cross-process recall and memory-authority red-team tests, remain unverified here. Native Windows acceptance was not run. The passing suite verifies deterministic engine behavior and PostgreSQL integration; no model-quality improvement is claimed by this extraction.

Successful upstream local/live-Ollama tests and historical benchmark records are linked in the [historical evidence index](TESTING.md#historical-upstream-evidence). The maintainer plans a fresh laptop live-model run of this repository on **2026-10-06**; its result is pending and will be recorded against the exact tested commit.
