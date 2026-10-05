# Immutable Artifact Journal

**Status:** constitutional architecture deep dive  
**Constitutional authority:** implements Article 36 of [`../../CONSTITUTION.md`](../../CONSTITUTION.md).  
**Applies to:** canonical events, percept-to-response stage boundaries, stateless LLM invocations, content-addressed blob storage, inspection and human auditing, signed journal heads, interruption recovery, and database reconstruction

Persistent Cognition maintains an independent append-only artifact journal in addition to PostgreSQL. Each logical record is immutable and hash-checked. PostgreSQL remains the indexed operational store used for efficient retrieval, scheduling, and execution. It is not the only surviving representation of Persistent Cognition's memory, cognition, or completed work.

The artifact journal exists for four reasons:

1. **Exact observability.** A developer must be able to inspect what each disposable worker received and produced without reconstructing it from logs or model behavior.
2. **Crash recovery.** A process may disappear after computing a result but before downstream operational state is committed. A durable stage artifact allows a replacement worker to rehydrate that result instead of repeating cognition or side effects.
3. **Disaster recovery.** Canonical event history must remain reconstructable when PostgreSQL is corrupted, lost, or deliberately recreated.
4. **Independent auditability.** Persistent Cognition's causal history must remain inspectable even when the primary database or scheduler tables are unavailable.

This design extends the system-continuity, append-only evidence, disposable-worker recovery, causal-provenance, local-first, and operator-stewardship rules in the Constitution.

## 1. Persistence domains

Persistent Cognition currently has two durable persistence domains.

### 1.1 PostgreSQL

PostgreSQL is the active indexed representation. It provides:

- canonical `events` rows;
- conversations and total event ordering;
- memory indexes and derived projections;
- attention/scheduler state;
- worker assignments, claims, checkpoints, and results;
- interaction bindings and working state.

PostgreSQL is optimized for querying and coordinated operational mutation. Its loss must not imply the loss of all cognitive history.

### 1.2 Filesystem artifact journal

The independent artifact root defaults to:

```text
.pcr/artifacts/
```

and may be moved with:

```text
PCR_ARTIFACT_ROOT=<path>
```

The `.pcr/` directory is local and Git-ignored. It may eventually contain
personal cognitive records, so its contents must not be checked into a public
repository. The previously tracked synthetic examples remain recoverable in earlier
commits, but new runtime artifacts stay in the user-controlled local root. A
deployment may also set `PCR_ARTIFACT_ROOT` outside the checkout.

Public fictional benchmark journals under `benchmarks/generated/` are retained
locally and excluded from new Git commits. A native benchmark run writes a
content-addressed manifest over every raw event and interaction artifact so the
compact result under `benchmarks/results/` remains connected to the exact causal
record. A verified Git-visible ZIP in `.tmp/` makes the latest run portable for review; see
[`BENCHMARK_SHARING_BUNDLE.md`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/BENCHMARK_SHARING_BUNDLE.md). Historical Git commits
still contain the previously checked-in raw evidence.

Current mechanism-run manifests also bind the host/runtime evidence used to interpret
model behavior: operating-system and Python identity, CPU/RAM facts, discoverable
NVIDIA GPU/driver/VRAM facts, Ollama version, configured model, immutable model
digest, and the loaded-model record returned by Ollama. New artifact-backed runs fail
closed if the Ollama version or configured model digest cannot be captured.

The journal contains two classes of records:

```text
artifacts/
  events/
    <event-id>.jsonl                    # new, one append-only file per event
    <event-id>.json                    # historical two-file format
    <event-id>.commit.json             # historical two-file format

  interactions/
    <interaction-id>/
      000001-<artifact-id>.json
      000002-<artifact-id>.json
      000003-<artifact-id>.json
      ...
      00000N-<artifact-id>.json
```

The exact filenames are presentation details. Artifact IDs, semantic artifact keys, hashes, types, interaction IDs, and sequence numbers are authoritative identifiers inside each document.

Interaction filenames deliberately use the bounded deterministic artifact UUID rather than concatenating the full semantic artifact key into the filesystem name. The full semantic key remains inside the JSON envelope and remains the idempotency identity. This keeps ordinary paths portable on Windows and other filesystems without discarding semantic identity or provenance.

## 2. Canonical event mirroring

Every canonical event written through `event_store.record_event` is independently mirrored. New events use a two-line JSON Lines file. Each line is a complete JSON object with its own hash; the first line is atomically written and fsynced, and the second is appended and fsynced after PostgreSQL commits. Existing `.json` and `.commit.json` pairs stay in their original form and remain readable and repairable by retry.

A new event follows this durability order:

```text
allocate deterministic/explicit event identity
        ↓
lock conversation sequence in PostgreSQL transaction
        ↓
atomically write + fsync first EVENT_RECORD JSONL entry
        ↓
insert event row and advance conversation sequence
        ↓
COMMIT PostgreSQL transaction
        ↓
append + fsync second EVENT_DATABASE_COMMIT JSONL entry
```

The first artifact contains the semantic event record:

- event ID;
- conversation ID;
- correlation ID;
- conversation sequence;
- event type;
- source;
- exact JSON payload;
- payload text when present;
- content hash.

The second artifact records database-commit metadata:

- event ID;
- semantic record hash;
- global sequence;
- conversation sequence;
- database-created timestamp;
- schema version;
- commit hash.

This two-entry protocol distinguishes an event that was durably journaled before a crash from an event known to have committed to PostgreSQL. The two immutable logical artifacts now share one physical file. Benchmark inventories call that file `EVENT_JOURNAL` while recording both entry hashes; old manifests and their separate file counts remain valid.

A crash may produce a complete first entry without a commit entry. This means database commit status is unknown, not necessarily rollback. A partial second entry is rejected by readers; a deterministic retry with authoritative database metadata can finish it only if the existing bytes are an exact prefix of the expected entry. Conflicting bytes fail closed. A committed new PostgreSQL event should not exist without its semantic record because that entry is written first.

Deterministic retries repair missing commit artifacts and fail closed if an existing JSON artifact conflicts with the retry's semantic identity.

## 3. Interaction artifact chain

Every user-prompt interaction and non-user situation task receives its own
append-only artifact chain.

Each artifact envelope contains at least:

```json
{
  "artifact_schema_version": 1,
  "artifact_id": "...",
  "artifact_key": "...",
  "artifact_type": "...",
  "interaction_id": "...",
  "conversation_id": "...",
  "correlation_id": "...",
  "task_id": "...",
  "assignment_id": "...",
  "stage": "...",
  "producer": "...",
  "journal_sequence": 1,
  "created_at": "...",
  "previous_artifact_id": null,
  "previous_artifact_hash": null,
  "payload_hash": "...",
  "artifact_hash": "...",
  "payload": {}
}
```

Artifacts after the first point to the immediately preceding artifact ID and hash. `pcr verify` recomputes payload and envelope hashes and checks chain continuity.

The interaction chain currently records:

- the original percept;
- each completed architectural stage result;
- stage errors;
- each exact stateless LLM invocation envelope;
- one parse/schema validation artifact linked to every LLM invocation;
- the exact triage policy and closed source allowlist as its own stage artifact;
- exact work/tool results as part of the EXECUTE stage artifact;
- the exact fixed-retrieval memory package and route receipts as part of the RETRIEVE stage artifact;
- the exact final responder output;
- PERSIST state;
- a final-disposition manifest for completed interactions.

Because the stage result is the same structured output used to complete the durable worker claim, the artifact is not a later summary of what the worker probably saw. It is the checkpointed boundary object itself.

The six-stage situation pipeline writes the same final-disposition artifact,
with `SITUATION_PERSIST` as its terminal stage. Its supervisor verifies all stage
results and their independent copies before publishing the manifest and marking
the task completed. Silent completion also requires this manifest.

Final-disposition retries use the original chain prefix preceding the manifest.
They never include the manifest itself or later diagnostics in its evidence list.
An identical retry returns the existing artifact; a different disposition remains
an immutable-content conflict. Existing v2 user-prompt manifests retain their
`V2_PERSIST_RESULT` terminal-stage label.

### 3.1 Exact stateless LLM invocation artifacts

Every LLM call made by the live v2 user-prompt worker is independently journaled. The `LLM_INVOCATION` payload includes:

- architectural stage;
- worker claim ID and invocation index;
- semantic call kind;
- configured model;
- backend base URL;
- exact system prompt;
- exact current user/context prompt;
- exact quarantined evidence payload and transport layout when evidence is isolated;
- canonical source-event references for memory evidence admitted to that invocation;
- exact structured-output JSON schema;
- generation token cap;
- effective generation temperature;
- normalized constrained model output when the call succeeds;
- exception type/message when the call fails;
- exact backend request path and JSON body;
- the HTTP status, elapsed wall time, transport error, and exact response-body byte
  count/SHA-256 when present; and
- the complete backend JSON response envelope, including token counts, durations,
  completion reason, and model metadata returned by Ollama.

Hidden-reasoning fields are never stored as plaintext. The response envelope keeps
their location plus byte length and SHA-256 so truncation, unexpected thinking-mode
activation, and cross-attempt identity remain diagnosable without turning private
chain-of-thought into an artifact. User-visible constrained output remains exact.

When a model attempt fails parsing, schema validation, or transport, a linked
`LLM_VALIDATION` artifact records:

- the invocation artifact ID and hash;
- invocation index and semantic kind;
- `INVALID` or `TRANSPORT_ERROR` status;
- the SHA-256 of the raw normalized output when present;
- the parser/schema/transport exception type and message.

Retries remain separate causal attempts. A malformed first response and a valid
second response produce two invocation artifacts, one linked failure artifact, and
the committed stage result. Routine validation success creates no separate record:
the stage result establishes acceptance. If the worker crashes before that result,
the surviving invocation is an incomplete attempt, not proof of successful validation.
Historical journals with `VALID` records remain readable and verifiable.

For natural response this includes the resolved Persistent Cognition evidence
system prompt. For response policy it proves that only current authority was
presented. For exact-source selection it preserves the admitted candidate payload.
Canonical event references allow native acceptance to establish which stored facts
reached response realization even when opaque literals are represented by
model-facing placeholders and restored by application code. For triage and retrieval it records memory separately from the later current
prompt/catalog. The layout field states whether the backend used chat
system/tool/user roles or raw Qwen system/evidence/current-user blocks.

This means debugging does not need to infer what a model saw from its answer. The actual stateless invocation contract is part of the hash-linked interaction history.

Invocation artifacts are causal evidence, not recovery substitutes for a completed architectural stage. Stage recovery uses a completed `STAGE_RESULT` artifact because downstream validation/transformation may occur after an individual raw model call. Failed or superseded model attempts remain visible in history.

## 4. Artifact-before-terminal ordering

For a disposable worker stage, the required order is:

```text
load fresh durable inputs
        ↓
compute stage result
        ↓
atomically write + fsync STAGE_RESULT artifact
        ↓
complete durable worker claim/result in PostgreSQL
        ↓
allow downstream stage
```

LLM invocations and exceptional validation artifacts are written during stage
computation, before the stage-result artifact. A successful invocation cannot be
followed by another model call in the same worker until its validation has resolved;
routine success is proven by the eventual stage result.

A stage is therefore recoverable if the process disappears after the artifact write but before the PostgreSQL worker result commits.

When a replacement worker claims that stage, it checks for a valid stage-result artifact first. If one exists, it uses the exact stored output and output references to complete the database worker result. It does **not** repeat the LLM invocation, retrieval decision, tool call, or other stage computation.

This ordering is especially important for expensive or externally consequential work. Existing worker idempotency/effect policy remains authoritative for side-effect safety; the artifact journal complements rather than replaces it.

## 5. Final disposition

A completed interaction ends with a `FINAL_DISPOSITION` artifact.

The final disposition is a compact manifest, not a duplicate transcript. It records:

- completed status;
- last completed architectural stage;
- whether a response was required;
- the resulting response text when applicable;
- references, types, hashes, sequence numbers, and stages for all prior interaction artifacts.

The manifest makes the entire percept-to-outcome chain discoverable from one terminal artifact while preserving detailed data in the preceding immutable records, including every v2 LLM invocation used to produce that outcome.

An interrupted interaction has no final disposition. Its partial chain is intentional durable state, not garbage.

## 6. Interruption recovery

The CLI exposes artifact-backed recovery:

```powershell
uv run pcr recover --latest
```

or:

```powershell
uv run pcr recover --interaction-id <uuid>
```

Recovery verifies the artifact chain first.

If operational interaction state still exists in PostgreSQL, Persistent Cognition continues that interaction. For each architectural stage:

1. an existing PostgreSQL worker result is accepted;
2. otherwise an existing stage artifact is rehydrated into the durable worker result;
3. otherwise the first genuinely unfinished stage is executed by a fresh worker.

If the final database stage exists but the final-disposition write itself was interrupted, recovery creates the missing final disposition after completing normal task finalization.

## 7. Database-loss recovery

The artifact journal is designed to survive recreation of PostgreSQL.

After creating an empty database and applying the current schema, canonical event history can be restored with:

```powershell
uv run pcr restore-events
```

This operation:

- verifies event artifact hashes;
- inserts missing conversations and events;
- preserves committed global/conversation sequence numbers where commit artifacts exist;
- assigns a safe new global sequence to a semantic artifact that survived before its original DB commit;
- repairs conversation next-sequence counters;
- advances the PostgreSQL global event sequence;
- fails closed on conflicting existing rows.

After event restoration, an interrupted interaction can be resumed from its filesystem artifacts. If its operational interaction/scheduler rows were lost, Persistent Cognition reconstructs the deterministic interaction identity from:

- the original conversation ID;
- the original correlation ID;
- the original user percept.

It then creates fresh operational execution state and walks the artifact chain. Previously completed stage results are rehydrated rather than regenerated.

The intended disaster sequence is therefore:

```text
artifact journal survives
        ↓
recreate PostgreSQL database
        ↓
apply schema.sql and normal permissions
        ↓
pcr restore-events
        ↓
pcr recover --latest   # if an interaction was incomplete
        ↓
rebuild disposable/derived projections as required
```

The artifact journal does not make every derived index automatically reconstruct itself yet. Canonical events are sufficient source evidence for those projections to be rebuilt by their own deterministic/replay mechanisms.

## 8. Inspection without PostgreSQL

Artifact inspection and integrity verification do not require a working database.

Latest interaction:

```powershell
uv run pcr inspect --latest
```

Specific interaction:

```powershell
uv run pcr inspect --interaction-id <uuid>
```

Verify hash-chain integrity:

```powershell
uv run pcr verify --latest
```

or:

```powershell
uv run pcr verify --interaction-id <uuid>
```

This allows response diagnosis even when PostgreSQL or scheduler state is offline. In particular, `inspect` exposes the exact memory packet and exact final-responder LLM invocation that produced a suspicious answer.

`inspect`/`verify` print the raw artifact chain and machine-readable verification result. For a human-readable forensic timeline over the same data, use:

```powershell
uv run pcr audit --latest
```

or:

```powershell
uv run pcr audit --interaction-id <uuid>
```

See [Section 9](#9-content-addressed-blob-store) and [Section 12](#12-current-scope-and-future-hardening) for what `audit` renders and why it is not itself authoritative.

## 9. Content-addressed blob store

`blob_store.py` stores exact bytes once, addressed by their SHA-256 digest, beside the artifact journal at `<artifact root>/blobs/sha256/<first two hex chars>/<hex digest>`. A caller gets back a small OCI-style descriptor:

```json
{"mediaType": "application/json", "digest": "sha256:8f47c1...", "size": 18442}
```

The digest identifies the bytes; nothing about storage location or filename participates in identity. `put_blob` is idempotent — writing identical bytes twice is a no-op the second time — and fails closed (`BlobIntegrityError`) if a path already exists with different byte length than the content it is being asked to store, since two distinct byte strings must not legitimately share one SHA-256 digest. `get_blob` recomputes the digest on every read and raises `BlobIntegrityError` if stored bytes no longer match it.

A journal artifact does not have to register its use of the blob store anywhere. Any artifact payload — present or future — can embed a descriptor of this exact shape anywhere in its structure, and `blob_store.iter_blob_descriptors` will discover it generically by walking the payload. The audit renderer (Section 8) uses this to report how many referenced blobs still verify, without every artifact producer needing a separate registration mechanism.

Verify one blob independently of any interaction:

```powershell
uv run pcr blob-verify --digest sha256:<hex>
```

Deduplication is scoped to one local artifact root (one installation), not globally across unrelated people/security domains. A shared global blob store would let two otherwise-isolated domains learn that they hold identical bytes merely by comparing digests. A future multi-tenant deployment should partition blob storage per security domain rather than widen this store, matching the per-root scoping the rest of the artifact journal already uses.

As of this writing, no existing artifact producer (percept, stage result, LLM invocation, final disposition) externalizes its payload to the blob store; every field they write remains inline JSON, matching their existing hash-chained contract and historical fixtures. The blob store is available for new/large payloads — model generations, documents, audio, and similar exact byte-for-byte evidence — to adopt incrementally as they become live, consistent with the conservative migration posture in Section 12.

## 10. Signed journal heads

Hash chaining alone (Sections 2-3) makes ordinary local corruption or truncation *detectable*, but it cannot defend against one actor with full filesystem access rewriting an interaction's artifacts **and** recomputing every downstream hash so the rewritten chain stays perfectly self-consistent. That is exactly the failure mode a purely local hash-chain-plus-sidecar design (e.g. SuperLocalMemory's audit chain) cannot rule out: the sidecar anchor lives in the same trust domain as the data it attests.

`journal_signing.py` adds one additional, independent, and strictly optional trust primitive: an Ed25519 digital signature over a *finalized* interaction's current journal head.

```powershell
uv run pcr sign --interaction-id <uuid>
uv run pcr verify-signature --interaction-id <uuid>
```

`sign` fails closed unless the interaction's artifact chain is both internally valid and already carries a `FINAL_DISPOSITION` artifact — a signature must never assert integrity or completeness the journal itself cannot already demonstrate. It generates a local Ed25519 keypair on first use (`<artifact root>/keys/<key_id>.private` / `.public`; the key id is a truncated SHA-256 fingerprint of the public key, used only as a lookup label) and writes a small signed anchor to `<artifact root>/anchors/<interaction_id>.json` containing the interaction id, current journal head hash, record count, timestamp, signing key id, and the signature itself.

`verify-signature` recomputes the same canonical bytes and checks the signature using **only the public key** — it never reads or needs the private key, so a copy of the anchor plus the public key file can be handed to a separate machine, an independent remote store, or a human reviewer, and verified there without any trust in the current state of the local artifact root. It also re-verifies the interaction's live artifact chain and re-derives its current head hash, so a chain that was rewritten (even a "sophisticated" rewrite that keeps every internal hash link self-consistent) after signing is caught as soon as its recomputed head no longer matches what was originally signed.

Signing is deliberately **not** wired into the live percept-response or situation pipelines. It is an explicit, opt-in operation over an already-finalized interaction, consistent with the principle that "the signature/remote anchor is not necessary for basic crash recovery; it is an audit-strengthening layer." An operator or external script may run `sign` on a schedule (e.g. after each finalized interaction, or in a nightly sweep) without any change to the pipeline that produced the interaction.

This remains **tamper-evident, not tamper-proof**: a local actor who controls the signing key at the moment of signing can still sign a false statement. What it defends against is a *later* rewrite of already-signed history being passed off as the original — the same distinction the original research draws between local hash chaining and an externally verifiable anchor.

## 11. Storage policy

Artifact redundancy is intentional. Storage efficiency must not silently erase exact causal evidence or eliminate the independent recovery copy.

Current policy:

- artifacts are immutable;
- ordinary retention/compaction does not delete them;
- the artifact root is user-controlled;
- `.pcr/` is Git-ignored. Runtime journals remain in their user-controlled
  local root; deliberately selected synthetic evidence may be shared separately;
- public fictional benchmark journals and generated corpora are permanent local
  evidence. They are ignored by Git for new runs and shared as verified run ZIPs;
  they must not be deleted or rewritten after a failed run;
- every retained training candidate keeps its source role, model/runtime, tested
  revision, artifact hashes, and human-review status; preservation alone never marks a
  model output as a positive training example;
- future cold-storage compression may transform old `.json` records to a content-preserving representation such as `.json.zst`, provided hashes/identity remain verifiable and the transformation is reversible;
- large binary objects should use the content-addressed blob store (Section 9), with JSON artifacts referring to their OCI-style descriptor rather than embedding arbitrary binary payloads;
- a locally generated Ed25519 signing key (Section 10) is not itself backed up or escrowed by this implementation; losing it does not lose any journal data, only the ability to produce *new* signatures under that key id, and previously-published anchors remain independently verifiable with the already-exported public key.

Explicit policy-governed erasure remains a separate stewardship/sovereignty operation and must not be confused with automatic compaction.

## 12. Current scope and future hardening

The current implementation establishes independent durability for:

- canonical append-only events;
- the live v2 user-prompt percept-to-response stage chain;
- exact stateless LLM invocation envelopes for the v2 semantic roles;
- stage result rehydration;
- final disposition manifests;
- event-store reconstruction;
- interaction continuation after operational-state loss;
- content-addressed storage for large exact payloads (Section 9);
- a deterministic human-readable audit renderer over the artifact chain (Section 8);
- opt-in Ed25519 signing and independent public-key verification of finalized journal heads (Section 10).

Further hardening should extend the same boundary rule to additional long-running/non-conversational task families as they become live, journal exact external-effect request envelopes alongside their results, adopt blob-referenced payloads at the LLM invocation and event boundaries where they are currently large inline strings, add bulk backup/restore commands, replicate signed anchors (and ideally the blobs/journals they cover) to a remote object store in a genuinely separate trust/IAM domain, and test deliberate power-loss/fsync fault cases on the native deployment environment.

The principle is broader than the current implementation: **if a meaningful cognitive or operational boundary would matter for explanation, replay, recovery, or reconstruction, its exact durable artifact must not exist only inside a disposable process or a single database.**
