# Persistent Cognition Runtime

**Persistent memory. Stateless inference. Durable execution.**

The runnable cognitive engine extracted from the pinned upstream source revision, including the v2 worker architecture and subsequent engine improvements through `106bb22be4ad60f2455ece8bc8c4e2806225d0fe`.

Exact history lives outside the model. Fresh specialist workers receive bounded evidence just in time; deterministic software owns scheduling, resources, execution authority, and recovery. The goal is useful continuous cognition on modest local hardware.

## Included

- Append-only PostgreSQL events and compact independent event/percept artifact journals.
- JIT lexical, entity, temporal, and associative retrieval, canonical source links, and bounded neighborhood expansion.
- v2 specialist workers and the later fixed deterministic retrieval stage that superseded the Composer control loop.
- Semantic and learned-memory representations with provenance, corrections, counterevidence, and canonical-source navigation.
- Bounded working state, generic percept intake, situation formation, expectations, consolidation, and action receipts.
- Deterministic attention, resource admission, leases, checkpoints, retries, and restart recovery.
- Stateless Ollama transport, optional explicit remote-provider routing, governed model parameters, and sequential local-model residency.
- Terminal and headless job interfaces, SQL schema, synthetic retrieval fixtures, and regression tests.

Android, device nodes, tunnel/sync services, sensor discovery, OS administration, desktop/web UI, deployment-only application setup, and application philosophy are excluded.

## Install

Requirements: Python 3.14+, uv, PostgreSQL, and Ollama for local model-backed chat. Docker is not required.

```sh
git clone https://github.com/wtvr-guy/persistent-cognition-runtime.git
cd persistent-cognition-runtime
uv sync --frozen
```

Create a PostgreSQL role and separate `pcr` and `pcr_test` databases. Copy `.env.example` to `.env`, set your own connection credentials, and apply the schema:

```sh
psql "postgresql://USER:PASSWORD@localhost:5432/pcr" -f schema.sql
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv run pcr
```

Ollama must be running. Full Windows and Linux/macOS instructions are in [SETUP.md](docs/SETUP.md).

Two separate invocations can use the same persistent memory:

```sh
uv run pcr --once "My Project Kestrel budget is 400 dollars."
uv run pcr --once "What budget did I set for Project Kestrel?"
```

This exercises fresh command processes, with historical evidence reconstructed by the engine. Model answer quality still depends on the selected model and retrieved evidence.

## Inspect and recover

```sh
uv run pcr inspect --latest
uv run pcr verify --latest
uv run pcr audit --latest
uv run pcr recover --latest
uv run pcr restore-events
```

Inspection and verification do not require a live model. Recovery requires the relevant database and any model work that remains incomplete. `pcr-percept` exposes generic intake, bounded scheduler ticks, and `pcr-percept serve` for a continuous governed percept loop without collecting device sensors.

## Validate

Set `TEST_DATABASE_URL` in your shell to a disposable database whose name includes `test` or `benchmark`. Tests clear that database.

```sh
uv run ruff check .
uv run python scripts/verify_source_baseline.py
uv run python scripts/audit_registries.py
uv run python scripts/audit_constraints.py --fail-unregistered
uv run pytest -q -ra
```

See [TESTING.md](docs/TESTING.md) for PostgreSQL CI, live-model acceptance, and interpretation of skipped tests. [EXTRACTION.md](docs/EXTRACTION.md) records what was copied, adapted, and verified.

### Historical upstream validation

The pinned upstream evidence archive preserves this architecture's development, benchmarks, test definitions, and local acceptance history. Its [2026-09-11 v2 validation record](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/audits/V07_EPISTEMIC_RETRIEVAL_SCOPING_2026-09-11.md#3-verification--results) reports successful local Windows/PostgreSQL/live-Ollama tests for four-turn stateless continuity and epistemic memory. The [v0.7 closure record](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/CLOSURE_STATUS.md) records maintainer acceptance and closure on 2026-09-12.

See the [historical evidence index](docs/TESTING.md#historical-upstream-evidence) for benchmark results, native test records, and earlier failures and repairs. Those results establish behavior of the recorded upstream revisions and environments; live Ollama acceptance of this extracted repository remains pending a fresh laptop run.

## Compatibility

The installed distribution is `persistent-cognition-runtime`. The source package is `persistent_cognition`; console commands remain `pcr` and `pcr-percept`. Use a separate virtual environment, database, and artifact directory from any upstream source application.

This is a copy with explicit provenance, not a GitHub fork carrying the source application's full history. [SOURCE_BASELINE.json](SOURCE_BASELINE.json) distinguishes byte-identical source files from the small extraction adaptations.

## Documentation and license

[Architecture](docs/ARCHITECTURE.md) · [Setup](docs/SETUP.md) · [Testing](docs/TESTING.md) · [Model routing](docs/MODEL_ROUTING.md) · [Engineering constitution](CONSTITUTION.md)

[PolyForm Noncommercial License 1.0.0](LICENSE), copied exactly from the upstream source project, including its project-specific commercial-licensing notice.
