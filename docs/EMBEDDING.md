# Supported embedding API

Import the supported surface from `persistent_cognition.api`. `API_VERSION` is
`persistent-cognition-api/v1`. Within v1, signatures and documented receipt
semantics remain compatible; incompatible changes require a new API version and
migration notes. Internal modules and underscore-prefixed helpers are not covered.
Pydantic settings and receipts retain their independently versioned schemas.

| Surface | Contract |
|---|---|
| `connect(database_url)` | PostgreSQL connection after destination consent and remote TLS checks. Caller owns lifetime and transactions. |
| `initialize_schema(conn)` | Explicit, idempotent provisioning from the schema bundled in the wheel. No implicit initialization during connection. |
| `worker_environment(settings, database_url=..., artifact_root=...)` | Returns environment for trusted application startup and inherited workers; includes immutable typed settings. |
| `respond(conn, text, conversation_id, ...)` | Owns a scheduler namespace and runs guarded fresh specialist processes. Conversation IDs are evidence cues. |
| `install_source_policy`, `ingest_percept` | Application installs authority; adapters deliver exact admitted observations under stable delivery IDs. |
| `tick`, `PerceptService` | Bounded tick or continuous supervisor with namespace ownership. |
| `CapabilityRegistry`, `register_trusted_executor` | Trusted bootstrap owns descriptors and callable bindings. Observed data cannot register code. |

Configuration and executor bindings are **process-scoped**. Establish them once
before concurrent work; do not swap environment or registries between requests.
Use separate connections per thread. Applications with distinct settings or
security principals need separate processes, databases and artifact roots.

## Installed-wheel provisioning and startup

```python
import os
from pathlib import Path
from persistent_cognition import api

settings = api.AppSettings()
database_url = os.environ["DATABASE_URL"]
os.environ.update(api.worker_environment(
    settings, database_url=database_url, artifact_root=Path("private-pcr-store"),
))
with api.connect(database_url) as conn:
    api.initialize_schema(conn)  # Operator setup, once; not each request.
```

Schema provisioning grants no authority to drop or recreate an existing database.
Back up before upgrading; migrations must be explicit. The packaged schema is
checked against the checkout schema by the source-baseline verifier.

## Versioned tools in fresh workers

Register handlers in **every** worker's trusted startup. Parent-process Python
objects are not inherited across fresh interpreters. An application may invoke
`percept_response_worker.main(registry=registry)` or `situation_worker.main()` in
its own worker entry point. Supply that entry point as `worker_command`, an argument
sequence, to `respond` or `PerceptService`. It never comes from observations.

```python
from persistent_cognition.capability_registry import (
    CapabilityDescriptor, CapabilityKind, RegisteredCapability,
)
from persistent_cognition import api, percept_response_worker

def my_tool(request):
    # External effects must deduplicate using request.execution_id.
    return {"result": "application result"}

registry = api.CapabilityRegistry((RegisteredCapability(
    descriptor=CapabilityDescriptor(capability_id="app.lookup", kind=CapabilityKind.TOOL,
                                    description="Look up application-owned records"),
    routing_terms=("lookup",), executor="app_lookup", executor_revision="2026-10-06",
),))
api.register_trusted_executor("app_lookup", my_tool, revision="2026-10-06")
# In the trusted worker branch:
# percept_response_worker.main(registry=registry)
# In the parent:
# api.respond(conn, text, conversation_id, registry=registry,
#             worker_command=[sys.executable, "-m", "my_app", "--worker"])
```

Capability snapshots contain versioned metadata and a deterministic hash. The
parent supplies the expected hash to workers; the first durable stage also records
it. Later stages reject registration changes. Executor revisions are checked before
effects. Revisions identify application releases; they do not independently prove
code integrity. Pin and protect the installed application release separately.

For deterministic non-user reactions, set `SourcePolicy.trusted_executor_revision`
to the same release string. Its default is `"1"`, matching legacy registrations.
Results must be bounded finite JSON objects with string keys at every depth.
Callback errors become failed action receipts; persistence failures remain runtime
failures. Retryable worker admission denials defer work to later bounded ticks.
Structural/ambiguous failures remain quarantined and require explicit reconciliation.

## Provider and storage boundaries

The operational storage contract currently requires `psycopg.Connection`, the
packaged PostgreSQL schema, transactions, advisory locks, and indexed event/worker
tables. SQLite and arbitrary database adapters are not interchangeable backends.

The supported model settings select Ollama or OpenAI Responses, with registered
per-stage routes. Ollama uses stateless `/api/chat` or `/api/generate`; OpenAI uses
stateless `/v1/responses` with `store=false`, fixed HTTPS origin, consent, strict
structured output and a response-byte cap with identity encoding. Provider additions require
transport/schema/authority acceptance and a settings-contract update; v1 does not
promise arbitrary transport injection. Applications can use model-free reactions
without installing a model service.

Headless job settings and a complete payload example are in [MODEL_ROUTING.md](MODEL_ROUTING.md).
Ownership, isolation, TLS, diagnostic export, and retention requirements are in
[SECURITY_AND_STORAGE.md](SECURITY_AND_STORAGE.md).

The retained PolyForm Noncommercial license permits noncommercial reuse within
its terms. Commercial embedding requires a separate license; API stability does
not change that restriction.
