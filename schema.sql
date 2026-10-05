-- Append-only event log schema for the JIT persistent-memory agent prototype.
-- Application code only ever INSERTs into `events`; never UPDATE/DELETE.
-- This is currently an application-level invariant, not a database-level
-- tamper boundary. Database permissions/immutability enforcement belong to the
-- explicit v0.9 hardening/threat-model work and must not be inferred from this
-- schema alone.

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    next_event_seq INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id),
    correlation_id UUID NOT NULL,
    global_seq BIGSERIAL,
    conversation_seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB NOT NULL,
    payload_text TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_events_conversation_seq
    ON events (conversation_id, conversation_seq);

CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);

CREATE INDEX IF NOT EXISTS idx_events_created_at ON events (created_at);

-- Deterministic keyword search stand-in for future embedding-based semantic retrieval.
CREATE INDEX IF NOT EXISTS idx_events_payload_text_fts
    ON events USING GIN (to_tsvector('english', coalesce(payload_text, '')));

-- ---------------------------------------------------------------------------
-- Prometheist Memory Kernel derived state.
--
-- `events` remains the sole authoritative history. Everything below is
-- disposable: it can be deleted and regenerated solely from `events`.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS event_integrity (
    event_id UUID PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    global_seq BIGINT NOT NULL UNIQUE,
    previous_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    hash_algorithm TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_event_integrity_global_seq
    ON event_integrity (global_seq);

CREATE TABLE IF NOT EXISTS memory_projection_runs (
    run_id UUID PRIMARY KEY,
    projection_name TEXT NOT NULL,
    projection_version TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    source_event_count BIGINT NOT NULL,
    entry_count BIGINT NOT NULL,
    projection_digest TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_projection_runs_name_version
    ON memory_projection_runs (projection_name, projection_version, started_at DESC);

CREATE TABLE IF NOT EXISTS memory_projection_entries (
    projection_name TEXT NOT NULL,
    projection_version TEXT NOT NULL,
    source_event_id UUID NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    source_global_seq BIGINT NOT NULL,
    source_hash TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (projection_name, projection_version, source_event_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_projection_entries_source_seq
    ON memory_projection_entries (projection_name, projection_version, source_global_seq DESC);

CREATE INDEX IF NOT EXISTS idx_memory_projection_entries_data
    ON memory_projection_entries USING GIN (data);

-- Memory Kernel v0.4 deterministic association projection. Association rows
-- are routing hints only; provenance points back to authoritative events.
CREATE TABLE IF NOT EXISTS memory_association_entries (
    association_id TEXT PRIMARY KEY,
    projection_version TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target TEXT NOT NULL,
    relationship TEXT NOT NULL,
    strength DOUBLE PRECISION NOT NULL,
    provenance_event_ids JSONB NOT NULL,
    required_cue_terms JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_association_entries_source
    ON memory_association_entries (projection_version, source_kind, source);

CREATE INDEX IF NOT EXISTS idx_memory_association_entries_target
    ON memory_association_entries (projection_version, target_kind, target);

-- ---------------------------------------------------------------------------
-- JIT Attention durable execution state.
--
-- Unlike derived memory projections, this is authoritative operational state:
-- it records what Prometheist is doing and what can be resumed after process
-- destruction. The transition journal is append-only at the application
-- layer; the task table is the current-state snapshot for fast restart.
-- ---------------------------------------------------------------------------

CREATE SEQUENCE IF NOT EXISTS attention_task_created_seq START WITH 1;

CREATE TABLE IF NOT EXISTS attention_execution_resources (
    resource_id TEXT PRIMARY KEY,
    resource_class TEXT NOT NULL,
    capacity INTEGER NOT NULL CHECK (capacity >= 1),
    system_headroom INTEGER NOT NULL DEFAULT 0 CHECK (system_headroom >= 0),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT attention_execution_resources_headroom_check
        CHECK (system_headroom <= capacity)
);

ALTER TABLE attention_execution_resources
    ADD COLUMN IF NOT EXISTS system_headroom INTEGER NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_execution_resources_headroom_check'
          AND conrelid = 'attention_execution_resources'::regclass
    ) THEN
        ALTER TABLE attention_execution_resources
            ADD CONSTRAINT attention_execution_resources_headroom_check
            CHECK (system_headroom >= 0 AND system_headroom <= capacity);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_attention_execution_resources_class_enabled
    ON attention_execution_resources (resource_class, enabled, resource_id);

CREATE TABLE IF NOT EXISTS attention_tasks (
    scheduler_key TEXT NOT NULL DEFAULT 'default',
    task_id UUID PRIMARY KEY,
    task_key TEXT NOT NULL,
    created_seq BIGINT NOT NULL UNIQUE,
    parent_task_id UUID REFERENCES attention_tasks(task_id),
    criticality TEXT NOT NULL,
    service_class TEXT NOT NULL,
    interruption_policy TEXT NOT NULL,
    deadline TIMESTAMPTZ,
    required_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    required_resource_classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    resource_requirements JSONB NOT NULL DEFAULT '[]'::jsonb,
    process_resource_estimate JSONB,
    dependency_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    enqueued_cycle BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 0,
    resumable_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Keep schema.sql idempotent for existing local v0.7 databases created before
-- execution-resource requirements were added.
ALTER TABLE attention_tasks
    ADD COLUMN IF NOT EXISTS required_resource_classes JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE attention_tasks
    ADD COLUMN IF NOT EXISTS resource_requirements JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE attention_tasks
    ADD COLUMN IF NOT EXISTS process_resource_estimate JSONB;

ALTER TABLE attention_tasks
    ADD COLUMN IF NOT EXISTS scheduler_key TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_attention_tasks_status_priority
    ON attention_tasks (status, criticality, created_seq);

CREATE INDEX IF NOT EXISTS idx_attention_tasks_scheduler_status_priority
    ON attention_tasks (scheduler_key, status, criticality, created_seq);

CREATE TABLE IF NOT EXISTS attention_task_transitions (
    scheduler_key TEXT NOT NULL DEFAULT 'default',
    transition_id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES attention_tasks(task_id) ON DELETE CASCADE,
    revision BIGINT NOT NULL,
    scheduler_cycle BIGINT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, revision)
);

ALTER TABLE attention_task_transitions
    ADD COLUMN IF NOT EXISTS scheduler_key TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_attention_transitions_task_revision
    ON attention_task_transitions (task_id, revision);

CREATE INDEX IF NOT EXISTS idx_attention_transitions_scheduler_task_revision
    ON attention_task_transitions (scheduler_key, task_id, revision);

CREATE TABLE IF NOT EXISTS attention_scheduler_state (
    scheduler_key TEXT PRIMARY KEY,
    cycle BIGINT NOT NULL DEFAULT 0,
    active_task_id UUID REFERENCES attention_tasks(task_id),
    pending_preemption_task_id UUID REFERENCES attention_tasks(task_id),
    admission_policy_version TEXT NOT NULL DEFAULT 'v0.7-c-greedy-v1',
    admitted_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    epoch_sequence BIGINT NOT NULL DEFAULT 0 CHECK (epoch_sequence >= 0),
    current_epoch_id UUID,
    assignment_policy_version TEXT NOT NULL DEFAULT 'v0.7-d-epoch-v1',
    preemption_policy_version TEXT NOT NULL DEFAULT 'v0.7-e-contention-v1',
    preemption_state_revision BIGINT NOT NULL DEFAULT 0
        CHECK (preemption_state_revision >= 0),
    pending_preemptions JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(pending_preemptions) = 'array'),
    resource_safety_required BOOLEAN NOT NULL DEFAULT FALSE,
    resource_safety_policy_version TEXT NOT NULL
        DEFAULT 'v0.7-resource-observation-v1',
    current_resource_observation_id UUID,
    CONSTRAINT attention_scheduler_epoch_pointer_check CHECK (
        (epoch_sequence = 0 AND current_epoch_id IS NULL)
        OR (epoch_sequence >= 1 AND current_epoch_id IS NOT NULL)
    ),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS admission_policy_version TEXT NOT NULL
        DEFAULT 'v0.7-c-greedy-v1';

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS admitted_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS epoch_sequence BIGINT NOT NULL DEFAULT 0;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS current_epoch_id UUID;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS assignment_policy_version TEXT NOT NULL
        DEFAULT 'v0.7-d-epoch-v1';

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS preemption_policy_version TEXT NOT NULL
        DEFAULT 'v0.7-e-contention-v1';

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS preemption_state_revision BIGINT NOT NULL
        DEFAULT 0;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS pending_preemptions JSONB NOT NULL
        DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS resource_safety_required BOOLEAN NOT NULL
        DEFAULT FALSE;

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS resource_safety_policy_version TEXT NOT NULL
        DEFAULT 'v0.7-resource-observation-v1';

ALTER TABLE attention_scheduler_state
    ADD COLUMN IF NOT EXISTS current_resource_observation_id UUID;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduler_epoch_pointer_check'
          AND conrelid = 'attention_scheduler_state'::regclass
    ) THEN
        ALTER TABLE attention_scheduler_state
            ADD CONSTRAINT attention_scheduler_epoch_pointer_check CHECK (
                (epoch_sequence = 0 AND current_epoch_id IS NULL)
                OR (epoch_sequence >= 1 AND current_epoch_id IS NOT NULL)
            );
    END IF;
END
$$;

-- Host measurements are immutable scheduler inputs rather than unrecorded
-- reads inside policy evaluation. Failed probes are persisted too, with a
-- fail-closed zero-new-work capacity envelope.
CREATE TABLE IF NOT EXISTS attention_resource_observations (
    scheduler_key TEXT NOT NULL,
    observation_id UUID NOT NULL,
    scheduler_cycle BIGINT NOT NULL CHECK (scheduler_cycle >= 1),
    captured_at TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    safety_policy_version TEXT NOT NULL,
    healthy BOOLEAN NOT NULL,
    snapshot JSONB NOT NULL CHECK (jsonb_typeof(snapshot) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, observation_id),
    CHECK (valid_until > captured_at)
);

CREATE INDEX IF NOT EXISTS idx_attention_resource_observations_cycle
    ON attention_resource_observations (scheduler_key, scheduler_cycle DESC);

-- Assignment and epoch rows are append-only authoritative scheduling history.
-- PLANNED epochs exist only in memory; a database row is worker-visible only
-- when the scheduler pointer and its complete reservation set commit with it.
CREATE TABLE IF NOT EXISTS attention_assignments (
    scheduler_key TEXT NOT NULL,
    assignment_id UUID NOT NULL,
    task_id UUID NOT NULL REFERENCES attention_tasks(task_id),
    task_revision BIGINT NOT NULL CHECK (task_revision >= 0),
    created_epoch_sequence BIGINT NOT NULL CHECK (created_epoch_sequence >= 1),
    reservation_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(reservation_ids) = 'array'),
    status TEXT NOT NULL CHECK (status = 'READY'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, assignment_id)
);

CREATE INDEX IF NOT EXISTS idx_attention_assignments_task
    ON attention_assignments (scheduler_key, task_id, created_epoch_sequence);

CREATE TABLE IF NOT EXISTS attention_scheduling_epochs (
    scheduler_key TEXT NOT NULL,
    epoch_id UUID NOT NULL,
    epoch_sequence BIGINT NOT NULL CHECK (epoch_sequence >= 1),
    previous_epoch_id UUID,
    scheduler_cycle BIGINT NOT NULL CHECK (scheduler_cycle >= 0),
    admission_policy_version TEXT NOT NULL,
    assignment_policy_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'COMMITTED'),
    assignment_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(assignment_ids) = 'array'),
    reservations JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(reservations) = 'array'),
    preemption_policy_version TEXT,
    pending_preemptions JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(pending_preemptions) = 'array'),
    preemption_event_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(preemption_event_ids) = 'array'),
    executed_preemption_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(executed_preemption_ids) = 'array'),
    cancelled_preemption_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(cancelled_preemption_ids) = 'array'),
    resource_observation_id UUID,
    resource_safety_policy_version TEXT,
    CONSTRAINT attention_scheduling_epochs_observation_pair_check CHECK (
        (resource_observation_id IS NULL AND resource_safety_policy_version IS NULL)
        OR (
            resource_observation_id IS NOT NULL
            AND resource_safety_policy_version IS NOT NULL
        )
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, epoch_id),
    UNIQUE (scheduler_key, epoch_sequence)
);

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS preemption_policy_version TEXT;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS pending_preemptions JSONB NOT NULL
        DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS preemption_event_ids JSONB NOT NULL
        DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS executed_preemption_ids JSONB NOT NULL
        DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS cancelled_preemption_ids JSONB NOT NULL
        DEFAULT '[]'::jsonb;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS resource_observation_id UUID;

ALTER TABLE attention_scheduling_epochs
    ADD COLUMN IF NOT EXISTS resource_safety_policy_version TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduling_epochs_observation_pair_check'
          AND conrelid = 'attention_scheduling_epochs'::regclass
    ) THEN
        ALTER TABLE attention_scheduling_epochs
            ADD CONSTRAINT attention_scheduling_epochs_observation_pair_check
            CHECK (
                (
                    resource_observation_id IS NULL
                    AND resource_safety_policy_version IS NULL
                )
                OR (
                    resource_observation_id IS NOT NULL
                    AND resource_safety_policy_version IS NOT NULL
                )
            );
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_attention_scheduling_epochs_current
    ON attention_scheduling_epochs (scheduler_key, epoch_sequence DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduling_epochs_previous_fk'
          AND conrelid = 'attention_scheduling_epochs'::regclass
    ) THEN
        ALTER TABLE attention_scheduling_epochs
            ADD CONSTRAINT attention_scheduling_epochs_previous_fk
            FOREIGN KEY (scheduler_key, previous_epoch_id)
            REFERENCES attention_scheduling_epochs(scheduler_key, epoch_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduler_current_epoch_fk'
          AND conrelid = 'attention_scheduler_state'::regclass
    ) THEN
        ALTER TABLE attention_scheduler_state
            ADD CONSTRAINT attention_scheduler_current_epoch_fk
            FOREIGN KEY (scheduler_key, current_epoch_id)
            REFERENCES attention_scheduling_epochs(scheduler_key, epoch_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduling_epochs_observation_fk'
          AND conrelid = 'attention_scheduling_epochs'::regclass
    ) THEN
        ALTER TABLE attention_scheduling_epochs
            ADD CONSTRAINT attention_scheduling_epochs_observation_fk
            FOREIGN KEY (scheduler_key, resource_observation_id)
            REFERENCES attention_resource_observations(
                scheduler_key, observation_id
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'attention_scheduler_observation_fk'
          AND conrelid = 'attention_scheduler_state'::regclass
    ) THEN
        ALTER TABLE attention_scheduler_state
            ADD CONSTRAINT attention_scheduler_observation_fk
            FOREIGN KEY (scheduler_key, current_resource_observation_id)
            REFERENCES attention_resource_observations(
                scheduler_key, observation_id
            );
    END IF;
END
$$;

-- Preemption explanations are append-only even though pending intent state is
-- replaceable. This preserves why a victim was selected, checkpointed,
-- executed, or spared after conditions changed.
CREATE TABLE IF NOT EXISTS attention_preemption_events (
    scheduler_key TEXT NOT NULL
        REFERENCES attention_scheduler_state(scheduler_key) ON DELETE CASCADE,
    event_id UUID NOT NULL,
    preemption_id UUID NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'REQUESTED',
            'CHECKPOINT_ACKNOWLEDGED',
            'EXECUTED',
            'CANCELLED'
        )
    ),
    scheduler_cycle BIGINT NOT NULL CHECK (scheduler_cycle >= 0),
    epoch_sequence BIGINT NOT NULL CHECK (epoch_sequence >= 0),
    target_task_id UUID NOT NULL REFERENCES attention_tasks(task_id),
    victim_task_ids JSONB NOT NULL CHECK (jsonb_typeof(victim_task_ids) = 'array'),
    victim_assignment_ids JSONB NOT NULL
        CHECK (jsonb_typeof(victim_assignment_ids) = 'array'),
    checkpoint_task_id UUID REFERENCES attention_tasks(task_id),
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, event_id)
);

CREATE INDEX IF NOT EXISTS idx_attention_preemption_events_intent
    ON attention_preemption_events (scheduler_key, preemption_id, created_at);

CREATE TABLE IF NOT EXISTS attention_resource_reservations (
    reservation_id UUID PRIMARY KEY,
    scheduler_key TEXT NOT NULL
        REFERENCES attention_scheduler_state(scheduler_key) ON DELETE CASCADE,
    task_id UUID NOT NULL REFERENCES attention_tasks(task_id) ON DELETE CASCADE,
    resource_id TEXT NOT NULL
        REFERENCES attention_execution_resources(resource_id),
    resource_class TEXT NOT NULL,
    units INTEGER NOT NULL CHECK (units >= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scheduler_key, task_id, resource_id)
);

CREATE INDEX IF NOT EXISTS idx_attention_resource_reservations_resource
    ON attention_resource_reservations (scheduler_key, resource_class, resource_id);

-- ---------------------------------------------------------------------------
-- Increment F: durable disposable-worker protocol.
--
-- A READY assignment is only an entitlement.  It becomes executable through
-- an expiring claim created after a fresh resource observation.  Step,
-- observation, checkpoint, and result rows are immutable; claim rows are the
-- replaceable lease state.  Exactly one terminal result may exist per step.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS attention_worker_steps (
    scheduler_key TEXT NOT NULL,
    step_id UUID NOT NULL,
    assignment_id UUID NOT NULL,
    task_id UUID NOT NULL REFERENCES attention_tasks(task_id),
    task_revision BIGINT NOT NULL CHECK (task_revision >= 0),
    created_epoch_sequence BIGINT NOT NULL CHECK (created_epoch_sequence >= 1),
    protocol_version TEXT NOT NULL,
    step_key TEXT NOT NULL,
    capability TEXT NOT NULL,
    reservation_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(reservation_ids) = 'array'),
    input_refs JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(input_refs) = 'array'),
    idempotency_key TEXT NOT NULL,
    effect_policy TEXT NOT NULL CHECK (
        effect_policy IN (
            'NO_EXTERNAL_EFFECT',
            'IDEMPOTENT_WITH_KEY',
            'AT_MOST_ONCE'
        )
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, step_id),
    UNIQUE (scheduler_key, assignment_id, step_key),
    UNIQUE (idempotency_key),
    FOREIGN KEY (scheduler_key, assignment_id)
        REFERENCES attention_assignments(scheduler_key, assignment_id)
);

CREATE INDEX IF NOT EXISTS idx_attention_worker_steps_assignment
    ON attention_worker_steps (scheduler_key, assignment_id, step_id);

CREATE TABLE IF NOT EXISTS attention_worker_claim_observations (
    scheduler_key TEXT NOT NULL,
    observation_id UUID NOT NULL,
    step_id UUID NOT NULL,
    assignment_id UUID NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    claim_policy_version TEXT NOT NULL,
    resource_safety_policy_version TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('GRANTED', 'DENIED')),
    reason TEXT NOT NULL,
    snapshot JSONB NOT NULL CHECK (jsonb_typeof(snapshot) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, observation_id),
    FOREIGN KEY (scheduler_key, step_id)
        REFERENCES attention_worker_steps(scheduler_key, step_id),
    FOREIGN KEY (scheduler_key, assignment_id)
        REFERENCES attention_assignments(scheduler_key, assignment_id),
    CHECK (valid_until > captured_at)
);

CREATE INDEX IF NOT EXISTS idx_attention_worker_claim_observations_step
    ON attention_worker_claim_observations (
        scheduler_key, step_id, captured_at DESC
    );

CREATE TABLE IF NOT EXISTS attention_worker_claims (
    scheduler_key TEXT NOT NULL,
    claim_id UUID NOT NULL,
    step_id UUID NOT NULL,
    assignment_id UUID NOT NULL,
    task_id UUID NOT NULL REFERENCES attention_tasks(task_id),
    worker_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    checkpoint_revision BIGINT NOT NULL DEFAULT 0
        CHECK (checkpoint_revision >= 0),
    observation_id UUID NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    last_heartbeat_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'ACTIVE',
            'CHECKPOINTED',
            'COMPLETED',
            'ABANDONED',
            'RELEASED'
        )
    ),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, claim_id),
    UNIQUE (scheduler_key, step_id, attempt),
    FOREIGN KEY (scheduler_key, step_id)
        REFERENCES attention_worker_steps(scheduler_key, step_id),
    FOREIGN KEY (scheduler_key, observation_id)
        REFERENCES attention_worker_claim_observations(
            scheduler_key, observation_id
        ),
    CHECK (lease_expires_at > claimed_at),
    CHECK (
        last_heartbeat_at >= claimed_at
        AND last_heartbeat_at < lease_expires_at
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_attention_worker_one_active_step
    ON attention_worker_claims (scheduler_key, step_id)
    WHERE status = 'ACTIVE';

CREATE UNIQUE INDEX IF NOT EXISTS idx_attention_worker_one_active_assignment
    ON attention_worker_claims (scheduler_key, assignment_id)
    WHERE status = 'ACTIVE';

CREATE INDEX IF NOT EXISTS idx_attention_worker_claims_lease
    ON attention_worker_claims (status, lease_expires_at, scheduler_key);

CREATE TABLE IF NOT EXISTS attention_worker_checkpoints (
    scheduler_key TEXT NOT NULL,
    checkpoint_id UUID NOT NULL,
    step_id UUID NOT NULL,
    claim_id UUID NOT NULL,
    revision BIGINT NOT NULL CHECK (revision >= 1),
    state JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(state) = 'object'),
    output_refs JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(output_refs) = 'array'),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scheduler_key, checkpoint_id),
    UNIQUE (scheduler_key, step_id, revision),
    FOREIGN KEY (scheduler_key, step_id)
        REFERENCES attention_worker_steps(scheduler_key, step_id),
    FOREIGN KEY (scheduler_key, claim_id)
        REFERENCES attention_worker_claims(scheduler_key, claim_id)
);

CREATE INDEX IF NOT EXISTS idx_attention_worker_checkpoints_latest
    ON attention_worker_checkpoints (scheduler_key, step_id, revision DESC);

CREATE TABLE IF NOT EXISTS attention_worker_results (
    scheduler_key TEXT NOT NULL,
    result_id UUID NOT NULL,
    step_id UUID NOT NULL,
    claim_id UUID NOT NULL,
    checkpoint_revision BIGINT NOT NULL CHECK (checkpoint_revision >= 0),
    idempotency_key TEXT NOT NULL,
    output JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(output) = 'object'),
    output_refs JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(output_refs) = 'array'),
    completed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scheduler_key, result_id),
    UNIQUE (scheduler_key, step_id),
    UNIQUE (idempotency_key),
    FOREIGN KEY (scheduler_key, step_id)
        REFERENCES attention_worker_steps(scheduler_key, step_id),
    FOREIGN KEY (scheduler_key, claim_id)
        REFERENCES attention_worker_claims(scheduler_key, claim_id)
);

-- Increment G interaction intake. Worker-step results are the authoritative
-- stage journal; this row binds one external percept to its durable task.
CREATE TABLE IF NOT EXISTS attention_interactions (
    scheduler_key TEXT NOT NULL,
    interaction_id UUID NOT NULL,
    protocol_version TEXT NOT NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id),
    correlation_id UUID NOT NULL,
    user_prompt_event_id UUID NOT NULL UNIQUE REFERENCES events(event_id),
    before_global_seq BIGINT NOT NULL CHECK (before_global_seq >= 1),
    task_id UUID NOT NULL UNIQUE REFERENCES attention_tasks(task_id),
    assignment_id UUID NOT NULL,
    user_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scheduler_key, interaction_id),
    FOREIGN KEY (scheduler_key, assignment_id)
        REFERENCES attention_assignments(scheduler_key, assignment_id)
);

ALTER TABLE attention_interactions
    ADD COLUMN IF NOT EXISTS percept_payload JSONB;

ALTER TABLE attention_interactions
    ADD COLUMN IF NOT EXISTS salience_assessment_payload JSONB;

CREATE INDEX IF NOT EXISTS idx_attention_interactions_task
    ON attention_interactions (scheduler_key, task_id);

-- v0.8 cognitive heads are a disposable index over immutable canonical events.
CREATE TABLE IF NOT EXISTS cognitive_heads (
    record_kind TEXT NOT NULL,
    record_key TEXT NOT NULL,
    event_id UUID NOT NULL REFERENCES events(event_id),
    global_seq BIGINT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (record_kind, record_key)
);
CREATE INDEX IF NOT EXISTS idx_cognitive_heads_sequence ON cognitive_heads(record_kind, global_seq);
-- Learned descriptions are routing hints; exact source records retain authority.
CREATE INDEX IF NOT EXISTS idx_self_memory_navigation_terms
    ON cognitive_heads USING GIN (to_tsvector('english',
        coalesce(payload->>'statement', '') || ' ' ||
        replace(coalesce((payload->'context_tags')::text, ''), '_', ' ')))
    WHERE record_kind = 'self_representation';
CREATE INDEX IF NOT EXISTS idx_self_evidence_representation
    ON cognitive_heads ((payload->>'representation_id'), global_seq)
    WHERE record_kind = 'self_evidence';
CREATE OR REPLACE FUNCTION project_cognitive_event() RETURNS trigger AS $$
BEGIN
    IF NEW.source = 'cognitive_runtime' AND NEW.payload->>'kind' = 'COGNITIVE_RECORD' THEN
        INSERT INTO cognitive_heads(record_kind, record_key, event_id, global_seq, payload)
        VALUES (NEW.payload->>'record_kind', NEW.payload->>'record_key', NEW.event_id,
                NEW.global_seq, NEW.payload->'data')
        ON CONFLICT (record_kind, record_key) DO UPDATE
        SET event_id = EXCLUDED.event_id, global_seq = EXCLUDED.global_seq, payload = EXCLUDED.payload
        WHERE cognitive_heads.global_seq < EXCLUDED.global_seq;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS cognitive_event_head ON events;
CREATE TRIGGER cognitive_event_head AFTER INSERT ON events
FOR EACH ROW EXECUTE FUNCTION project_cognitive_event();
