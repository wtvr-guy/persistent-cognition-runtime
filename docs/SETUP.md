# Setup

Install Python 3.14+, uv, PostgreSQL, and Ollama. Clone the repo and run `uv sync --frozen`. The reference backend is PostgreSQL; no Docker deployment is required.

## Database

In an administrative psql session, create a dedicated local role and databases. Set your own password interactively:

```sql
CREATE ROLE pcr_app LOGIN;
\password pcr_app
CREATE DATABASE pcr OWNER pcr_app;
CREATE DATABASE pcr_test OWNER pcr_app;
```

On Linux an installed server may expose that session through `sudo -u postgres psql`; on Windows use SQL Shell (psql) or your configured administrative client.

PowerShell: `Copy-Item .env.example .env`. Linux/macOS: `cp .env.example .env`.

Edit the connection URLs with your role and password. URL-encode special characters in credentials. Apply `schema.sql` to the runtime database using its actual connection URL:

```sh
psql "postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr" -f schema.sql
```

The test harness applies the schema automatically to its disposable database.

## Model and storage

Start Ollama, pull `qwen3:4b-instruct-2507-q4_K_M`, and run `uv run pcr`. Keep the same model when comparing behavior with the source project. Resource admission may reject a model that does not fit available host resources.

Use a dedicated artifact root. The inherited environment name is `PCR_ARTIFACT_ROOT`; `.env.example` uses the checkout's ignored `.pcr/artifacts` folder. Do not point this extraction at the live application's database or artifact root.

Local loopback model/database endpoints work without remote-service enrollment. Nonlocal destinations retain the source engine's explicit outbound-consent policy. No hosted model service is required.

## Tests

Set the test URL in the process environment; the test harness chooses it before application `.env` loading.

```powershell
$env:TEST_DATABASE_URL = "postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr_test"
uv run pytest -q -ra
```

```sh
export TEST_DATABASE_URL='postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr_test'
uv run pytest -q -ra
```

Only use a disposable database: its tables are cleared before tests. See [TESTING.md](TESTING.md).

## Continuous percept service and embedding

Ollama is optional for deterministic structured reactions. It is required when a source policy enables semantic triage or natural-language responses. The runtime installs no persona, imprint, or self-model.

Application startup installs a `SourcePolicy` with its source ID, kind, modalities, evidence scope, and allowed work. Adapters call `ingest_percept` with a stable `delivery_id`, timezone-aware observation time, and arbitrary JSON-compatible payload. Supply `PerceptContext.observations`/expectation references for measured discrepancies; install an explicit `deterministic_task` for scheduled work, sensor redflags, exceptions, and asynchronous tool results that must trigger a reaction. Payload text cannot install policy or select an executor.

Source classes include `SCHEDULED_EVENT`, `ANOMALY_ALERT`, `EXTERNAL_OBSERVATION`, `SYSTEM_OBSERVATION`, and `ACTION_OUTCOME`. Chat uses the separate mandatory-response user pipeline. Text, scalar numeric metrics, structured JSON, and bounded object event streams are supported. For image/audio/video/file/document inputs, call `preserve_media` and submit its immutable local content-hash reference. Media descriptions are untrusted metadata, not an assertion that the runtime decoded the media.

```sh
uv run pcr-percept install-source SOURCE_POLICY.json
uv run pcr-percept ingest PERCEPT.json
uv run pcr-percept serve --scheduler-key automation
# Alternatively, run a single bounded page; do not run alongside the service:
uv run pcr-percept tick --scheduler-key automation
```

The ingestion JSON contains `source`, `observation`, `observed_at`, `delivery_id`, and optional `context`, `conversation_id`, `correlation_id`. Sources/candidates are shared durable evidence; each scheduler namespace maintains separate cursors, task progress, and worker claims. Different namespaces are independent consumers, not input filters. Using two namespaces for the same source may run its configured reaction twice.

### Trusted application callbacks

An installed policy may set `trusted_executor` only alongside an explicit non-consolidation `deterministic_task`. The named callback must be registered by trusted startup **in each fresh worker process**. A parent-process registration alone does not survive process launch. CLI `serve` uses the stock worker, with no application callbacks. Embedding applications supply an explicit `worker_command` argument sequence; the runtime never constructs it from observation data, loads data-named modules, or invokes a shell.

For example, the application's existing `my_app` module can own both startup branches:

```python
import json
import signal
import sys
from datetime import datetime, timezone

from persistent_cognition import db, situation_worker
from persistent_cognition.percept_intake import ingest_percept, install_source_policy
from persistent_cognition.percept_service import PerceptService
from persistent_cognition.perception import PerceptKind, PerceptModality, PerceptSource
from persistent_cognition.percept_triage import SourcePolicy, TaskClass
from persistent_cognition.trusted_executors import register_trusted_executor

def react(request):
    # Treat this payload as data; never execute code/commands embedded in it.
    observation = json.loads(request.task_text)
    return {"received": observation, "execution_id": str(request.execution_id)}

if "--worker" in sys.argv:
    register_trusted_executor("app_reaction", react)
    situation_worker.main()
else:
    source = PerceptSource(
        source_id="app:tool-results", kind=PerceptKind.EXTERNAL_OBSERVATION,
        modality=PerceptModality.STRUCTURED,
    )
    with db.get_connection() as conn:
        install_source_policy(conn, SourcePolicy(
            source_id=source.source_id, kind=source.kind, modalities=(source.modality,),
            deterministic_task=TaskClass.RECONCILE, trusted_executor="app_reaction",
        ))
        ingest_percept(
            conn, source=source, observation={"tool_id": "job-42", "exit_code": 0},
            observed_at=datetime.now(timezone.utc), delivery_id="job-42:completed",
        )
    service = PerceptService(
        scheduler_key="automation",
        worker_command=[sys.executable, "-m", "my_app", "--worker"],
    )
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: service.request_stop())
    service.run()
```

Callbacks return JSON objects. A successful return means the callback completed, not that an external fault healed. Raised callback errors produce a durable failed action receipt and governed outcome feedback. Handlers performing external effects **must deduplicate using `request.execution_id`**: a crash between an external effect and receipt persistence can cause a retry. The runtime does not promise exactly-once external effects. A missing bootstrap binding fails closed before execution.

Action feedback forms an independent action situation. The original intention retains its entity links, but feedback cannot overwrite a newer pending sensor/source reaction. Deterministic consolidation without a natural-language response requires no model admission.

A retryable resource-admission denial starts no worker and leaves the same task pending for a later bounded tick. A structurally failed situation stage is durably quarantined in `situation_failure` (key `SCHEDULER:TASK_ID`), so healthy siblings continue and unchanged poison work is not automatically retried on restart. Its task, stage artifacts, receipts and assignment history remain available. Reservations are released through normal scheduler epochs; a surviving active claim retains capacity until release or lease expiry. Inspect the error and repair the application binding or storage before calling `retry_situation_task(conn, task_id, scheduler_key=...)` under `scheduler_ownership`. Each explicit retry permits one further attempt. An issued action without a completion receipt requires application reconciliation first; only then pass `effects_reconciled=True`. This assertion does not establish exactly-once effects: the handler must still honor the same execution ID. Completed tasks with interrupted bookkeeping can be finalized with `run_situation_task` after releasing their quarantine, without repeating work.

Inspect `action_execution`, `situation_completion` (including structured `work`), and `service_error` via `cognitive_store.get_record`/`list_records`. These remain available when `response_required` is false. Service errors also emit bounded exception summaries; exact private input is not copied into ordinary logs.

Idle polling uses finite positive delays, geometric backoff capped at 30 seconds by default, and a five-second error cooldown. `run(max_iterations=N)` bounds a run; `request_stop()` interrupts sleeping and stops at the next durable worker boundary without draining the remaining page. An already executing worker may finish or reach its declared timeout; stopping does not interrupt an ambiguous external effect. Pending stages remain durable for restart. The CLI handles SIGINT/SIGTERM. A lost database session ends the run, rather than silently reconnecting without its ownership lock; restart with a new connection to reacquire ownership and resume durable progress. Do not share one connection across concurrent service owners.

Supported embedding and wheel provisioning are documented in [EMBEDDING.md](EMBEDDING.md). Private history, permission remediation, TLS, quotas and retention are documented in [SECURITY_AND_STORAGE.md](SECURITY_AND_STORAGE.md).
