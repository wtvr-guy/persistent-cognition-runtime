# Deployment and private history

One installation represents one security principal. Conversation IDs intentionally
allow cross-conversation retrieval; they are **not authorization walls**. Isolate
databases, artifact roots, credentials, processes and retrieval scope between people
or applications that must not share evidence. Worker processes inherit host access;
they are not security sandboxes. Append-only behavior is an application contract,
not database-enforced resistance to an administrator rewriting records.

## Files and diagnostics

New private directories use POSIX mode `0700`; keys, journals, blobs, media and
policies use `0600`. POSIX regular-file opens walk directory descriptors with
`O_NOFOLLOW` and reject nonregular files. Blob digests require exactly 64 lowercase
hexadecimal characters. Reads are capped before hash verification.

Stop processes using an existing store, back it up privately, then remediate old
permissions:

```sh
uv run pcr storage-permissions
```

This uses `PCR_ARTIFACT_ROOT`, rejects symlinks, changes directories/files to
`0700`/`0600`, and does not alter unrelated existing parent directories. A failure
can leave a partially remediated tree; correct the reported condition and rerun.

On Windows, POSIX modes cannot establish an ACL. Create a dedicated directory
whose ACL allows only the intended account and necessary system administrators;
disable inherited access for unrelated accounts before writing private history.
Verify the ACL with `icacls PATH`. The remediation command rejects Windows rather
than claiming chmod protects the store. Native Windows ACL/rename acceptance must
be run on the deployment machine.

Exact prompts, admitted evidence and outputs remain protected forensic history.
Do not publish an artifact root as ordinary logs. Error diagnostics use bounded
exception summaries, omit rejected Pydantic input/context, and redact configured
credential values. Console worker failures and service logs use the same summaries.
Diagnostic summaries are suitable for narrower exports; whole artifacts still
require content review. Digital signatures provide tamper evidence, not encryption;
protect private keys and keep independent copies of public anchors.

## Resource admission and retention

These defaults are provisional deployment limits, **not optimal measured values**.
Set positive integer environment overrides before worker startup.

| Environment variable | Default | Boundary |
|---|---:|---|
| `PCR_MAX_INTAKE_BYTES` | 1 MiB | Text UTF-8, structured intake, CLI/job/settings JSON; checked before canonical persistence. |
| `PCR_MAX_JSON_DEPTH` | 32 | Structured nesting; cyclic objects are rejected. |
| `PCR_MAX_JSON_NODES` | 65,536 | Structured values and object keys. |
| `PCR_MAX_HTTP_RESPONSE_BYTES` | 4 MiB | Model responses; identity encoding required, compression rejected before decoding. |
| `PCR_MAX_BLOB_BYTES` | 64 MiB | Blob writes and exact reads. |
| `PCR_MAX_JOURNAL_BYTES` | 64 MiB | Each journal's append/inspection ceiling. |
| `PCR_MAX_JOURNAL_ENTRY_BYTES` | 8 MiB | Each complete JSON line. |
| `PCR_MAX_JOURNAL_RECORDS` | 65,536 | Materialized inspection and page range. |
| `PCR_ARTIFACT_QUOTA_BYTES` | 1 GiB | Serialized filesystem-byte quota across runtime writers. |
| `PCR_MIN_FREE_DISK_BYTES` | 64 MiB | Free space protected by write admission. |
| `PCR_MAX_DIAGNOSTIC_CHARS` | 2,000 | Exportable exception messages. |

Limits reject operations explicitly; they never silently truncate canonical
evidence. They do not replace OS memory limits or PostgreSQL/model storage quotas.
The filesystem quota counts root contents under a process lock, including temporary
files. This costs a metadata scan and is not a constant-time total-storage claim.
Uncooperative external writers and filesystem owners require OS quota protection.

Use `percept_journal.read_page(path, offset=..., page_size=...)` for paged inspection.
Full materialization remains available under explicit file/entry/record caps.
Large media use the existing bounded adapter instead of inline JSON or text.

The archive policy is explicit: stop intake before reaching the quota, finish or
reconcile in-flight work, back up complete finalized chains and referenced blobs/
media/anchors, verify them, then select a new private root and record its location
in application administration. Old canonical chains remain intact in the archive.
The runtime does not automatically delete, prune or partially archive evidence;
retrieval/recovery of an archived root requires selecting that root explicitly.
Provision a separate PostgreSQL disk quota and retention plan before continuous
production intake; the filesystem quota does not bound database growth.

## Concurrency and remote services

Chat, one-shot chat, recovery, service, ticks and supported API calls share a
session-scoped advisory lock for their scheduler namespace. Execution reset also
takes ownership and refuses unexpired active worker claims, including workers that
outlive their parent's database session. Restart only after workers release their
claims or their leases expire. Namespaces are independent consumers of shared
evidence; using two namespaces can deliberately run a source reaction twice.

Local loopback services require no remote consent. Remote models require verified
HTTPS; remote PostgreSQL requires `sslmode=verify-full` with an appropriate trusted
CA. Enrollment is still required separately: an operator obtains
`consent_proposal(endpoint, purpose)`, reviews it, then calls `grant_consent` with
`accepted_digest=content_digest(proposal)`. A model or observation cannot enroll
itself. `revoke_consent` removes the grant for subsequent requests. Consent does
not encrypt data or prevent the destination from retaining it.

For PostgreSQL grants, use the credential-free endpoint such as
`postgresql://database.example:5432`, purpose `DATABASE`; put credentials and TLS
parameters only in the actual connection string. For models use their exact
origin and purpose `MODEL`. Proxy/environment trust and redirects remain disabled.

Destructive PostgreSQL scale benchmarks now require
`--confirm-database EXACT_TEST_DATABASE_NAME`, destination consent/TLS, and the
existing test/benchmark-name guard. Never target a live application database.
