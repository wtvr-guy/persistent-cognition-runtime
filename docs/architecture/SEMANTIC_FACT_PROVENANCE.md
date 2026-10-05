# Semantic Memory Provenance

**Status:** v0.8 architecture.
**Constitutional authority:** Articles 9, 10, 11, 23, and 24 of
[CONSTITUTION.md](../../CONSTITUTION.md).
**Applies to:** durable derived claims, their supporting/opposing evidence,
current semantic conclusions, and historical belief reconstruction.

Persistent Cognition does not store one mutable fact object and silently rewrite it as
new information arrives. Semantic memory is split into three immutable record
families:

1. **SemanticAssertion** — a claim that may be true.
2. **SemanticEvidence** — one exact provenance-bearing observation that bears
   on an assertion.
3. **SemanticResolution** — Persistent Cognition's current derived conclusion for one
   subject/property.

This separation is required for lifelong memory. A historical observation can
be learned years after the event it describes; repeated corroboration should
strengthen provenance without creating duplicate claims; contradictory evidence
must remain inspectable; and storage pagination must never determine what the
system believes.

## Why the first prototype was replaced

The first semantic-fact prototype combined assertion, evidence, temporal
history, contradiction, confidence, and current-head semantics into one
SemanticFact record. That created several incorrect behaviors:

- a historical backfill could become cognitive_heads current merely because
  its event was inserted later;
- CONTRADICTS records incorrectly used supersedes;
- asserted_at existed but historical lookup ignored it;
- observed_at was treated as the time a claim became true;
- corroborating evidence was not represented in semantic memory;
- confidence was collapsed with a page-local min operation;
- the 64-record situation window became an accidental provenance limit;
- equal-time conflicts were broken by UUID ordering;
- consolidation page boundaries affected whether a fact was created;
- read/classify/write reconciliation was not serialized.

The v2 representation removes those conflations rather than adding special
cases around them.

## Record model

### SemanticAssertion

A SemanticAssertion contains the semantic content of one claim:

    assertion_id
    subject
    property
    value
    unit
    claim_valid_from   optional real-world validity
    claim_valid_until  optional real-world validity
    created_at         when this assertion object first entered derived memory

Assertion identity is based on semantic content and explicit validity, not on
confidence, source, or derivation method. Repeated evidence for the same claim
therefore points to the same assertion.

Observation time is deliberately not treated as validity start. If Persistent Cognition
learns today that a person has been vegetarian since 2012, the evidence may
have observed_at=today, while the assertion may have claim_valid_from=2012.

When no explicit validity interval is known, the assertion does not invent one.
Its evidence observation times remain available for temporal resolution.

### SemanticEvidence

A SemanticEvidence record captures one exact support/opposition relationship:

    evidence_id
    assertion_id
    subject / property
    source_kind          PERCEPT | ACTION | EVENT
    source_id
    relation             SUPPORTS | OPPOSES
    observed_at           when the source evidence describes/was observed
    known_at              when Persistent Cognition admitted/learned this evidence
    confidence            preserved source/derivation confidence
    derivation_method

Evidence is append-only and individually provenance-linked. The source can be
a percept, action, or canonical event, so later reflection/consolidation workers
can use the same representation without pretending their derived work was a
direct percept. Corroboration does not duplicate the assertion; it adds another
evidence record.

There is no fixed-size derived_from tuple. A claim may accumulate arbitrary
evidence over a lifetime. Normal processing remains bounded because evidence is
stored/indexed individually and historical scans page through bounded reads.

Confidence is preserved per evidence item. v2 intentionally does not invent a
universal confidence aggregation formula. Source independence, reliability,
self-report authority, inference confidence, and contextual relevance are
different concepts and should not be collapsed into an unvalidated average,
minimum, or vote.

### SemanticResolution

A SemanticResolution is Persistent Cognition's current derived conclusion for exactly
one subject/property:

    resolution_id
    subject / property
    status                 ACCEPTED | AMBIGUOUS | UNKNOWN
    candidate_assertion_ids
    selected_assertion_id  only for ACCEPTED
    support_observed_at
    resolution_policy
    resolved_at
    supersedes             prior resolution, never a conflicting assertion

The cognitive_heads index points to the latest resolution, not to the last
assertion written. Therefore inserting historical evidence cannot silently
replace the current belief.

Resolution history is itself append-only. supersedes means only that one
resolution replaced a prior resolution as Persistent Cognition's current derived
conclusion. It is not used to label contradictory assertions.

## Temporal coordinates

Semantic memory keeps distinct temporal roles instead of collapsing them:

- **claim validity** — optional claim_valid_from / claim_valid_until interval
  describing when an assertion applies in the represented world;
- **observation time** — evidence observed_at, used to order competing
  observations without pretending that observation automatically defines the
  assertion's full validity interval;
- **knowledge time** — evidence known_at, when Persistent Cognition actually admitted or
  learned that evidence; and
- **resolution time** — resolution resolved_at, when the semantic resolver
  formed that derived conclusion.

semantic_resolution_as_of(valid_at=..., known_at=...) answers the bi-temporal
historical query over represented-world validity and system knowledge:

> Given only evidence Persistent Cognition had learned by known_at, what could it
> conclude about the subject/property at real-world time valid_at?

Example:

    2026-01-01  historical reality: role = manager
    2026-06-01  historical reality: role = engineer
    2026-09-01  Persistent Cognition learns the old manager evidence

A query with known_at=2026-08-01 cannot use the manager evidence even when
asking about January. A later query may.

This is different from ordinary current-memory lookup. Current cognition reads
the incrementally maintained resolution head and does not scan lifetime
history.

## Current resolution policy

The v2 policy is versioned as:

    latest-supported-observation/v2

For ordinary point observations:

- later supported observations can represent temporal change;
- older backfilled observations remain evidence but do not roll the current
  resolution backward;
- different assertions at the same effective time produce AMBIGUOUS;
- opposition at or after the latest accepted support produces AMBIGUOUS, while
  stale opposition remains visible without rolling the conclusion backward;
- repeated support for an already accepted assertion adds evidence and advances
  the resolution's effective observation time when the support is newer. That
  temporal advancement is a meaningful resolution revision because it prevents
  later-arriving stale evidence from displacing newer corroborated state.

Assertions with an explicit validity interval that does not include the current
resolution time are retained for historical queries but do not become current.

The policy deliberately does not use confidence as an automatic winner between
conflicting values. A later specialist can add source-aware belief revision
without destroying v2 evidence.

## Consolidation semantics

consolidation.py no longer asks whether all observations inside one storage
page agree before semantic memory may be written.

Every unique structured observation admitted to consolidation becomes its own
semantic evidence relationship. Stable evidence IDs deduplicate overlapping
situation snapshots and replay.

The page is therefore only a bounded execution/storage unit:

    page boundary != evidence boundary
    page boundary != belief boundary
    page boundary != contradiction boundary

Two runs over the same canonical observations must converge on the same
assertions/evidence/current resolution regardless of how those observations are
partitioned into pages.

Before derivation begins, the consolidation action durably freezes its exact
input record keys and one derived_at timestamp in a consolidation_input record.
Each source event's durable created_at supplies evidence known_at, while
derived_at supplies resolution time. Retries therefore process the same evidence
page under the same resolution timestamp even if newer observations arrive
later.

## Concurrency and replay

Semantic updates acquire the existing PostgreSQL advisory record_lock for the
exact subject/property key before they read and advance the current resolution.
Two workers cannot independently read the same head and race to create
incompatible current branches.

Assertion and evidence identities are deterministic. Replaying identical work
is idempotent. Reusing the same evidence identity with different immutable
content fails closed.

## Inspection

The existing command remains:

    uv run pcr memory-fact --subject "person:mike" --property "preferred_drink"

It now prints:

- current resolution;
- selected assertion, if any;
- all assertions for that subject/property;
- all evidence records;
- append-only resolution history.

The command name is retained for operator familiarity even though the internal
model no longer uses a monolithic SemanticFact.

## Storage and migration

The first prototype used record_kind=semantic_fact. v2 uses:

    semantic_assertion
    semantic_evidence
    semantic_resolution

Old v1 derived records are not canonical evidence and are not read by v2.
Canonical percept/event history remains authoritative, so v2 semantic state can
be regenerated through consolidation rather than mutating or deleting old
history.

This follows the project's rule that derived cognitive structures are
replaceable while admitted source evidence is lossless.

## Deliberate remaining boundaries

- Free-text conversation is not automatically converted into ordinary structured
  semantic assertions.
- Confidence remains evidence metadata, not a universal belief score.
- Historical bi-temporal semantic queries may scan all evidence for one exact
  subject/property through bounded pages; ordinary cognition uses indexed heads.
