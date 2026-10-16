from types import SimpleNamespace
from uuid import uuid4

import pytest

from persistent_cognition import capability_runtime, situation_runtime, trusted_executors, worker_runtime
from persistent_cognition.capability_registry import CapabilityDescriptor, CapabilityKind, CapabilityRegistry, RegisteredCapability
from persistent_cognition.trusted_executors import TrustedExecutorError


@pytest.fixture(autouse=True)
def clear_bindings():
    trusted_executors.clear_trusted_executors()
    yield
    trusted_executors.clear_trusted_executors()


def registration(*, revision="1"):
    return RegisteredCapability(descriptor=CapabilityDescriptor(capability_id="app.tool",
        kind=CapabilityKind.TOOL, description="Application callback"),
        routing_terms=("reaction",), executor="app_tool", executor_revision=revision)


def execute(registration):
    return capability_runtime._execute_application_capability(None, registration=registration,
        capability_execution_id=uuid4(), requester_task_id=uuid4(), requester_step_id=uuid4(),
        plan_position=0, conversation_id=uuid4(), correlation_id=uuid4(), task_text="data",
        before_global_seq=1)


@pytest.mark.parametrize("result", [{1: "value"}, {"nested": {2: "value"}}, {"bad": float("nan")}, ["not an object"]])
def test_invalid_callback_data_has_callback_failure_semantics(result, monkeypatch):
    trusted_executors.register_trusted_executor("app_tool", lambda _: result)
    monkeypatch.setattr(capability_runtime.event_store, "record_event", lambda *a, **k: pytest.fail("invalid result persisted"))
    with pytest.raises(TrustedExecutorError):
        execute(registration())


def test_storage_failure_is_never_disguised_as_a_callback_failure(monkeypatch):
    trusted_executors.register_trusted_executor("app_tool", lambda _: {"completed": True})
    def storage_failure(*args, **kwargs):
        raise OSError("storage failed")
    monkeypatch.setattr(capability_runtime.event_store, "record_event", storage_failure)
    with pytest.raises(OSError):
        execute(registration())


def test_executor_revision_mismatch_fails_before_any_effect():
    trusted_executors.register_trusted_executor("app_tool", lambda _: pytest.fail("stale handler ran"), revision="2")
    with pytest.raises(RuntimeError, match="revision"):
        execute(registration(revision="1"))


def test_registration_snapshots_survive_json_and_detect_changed_bindings(monkeypatch):
    import json
    from persistent_cognition import percept_response_worker
    registry = CapabilityRegistry((registration(),))
    assert registry.snapshot() == json.loads(json.dumps(registry.snapshot()))
    other = CapabilityRegistry((registration(revision="2"),))
    assert registry.snapshot()["sha256"] != other.snapshot()["sha256"]
    monkeypatch.setenv("PCR_CAPABILITY_SNAPSHOT_SHA256", registry.snapshot()["sha256"])
    monkeypatch.setattr(percept_response_worker.db, "get_connection", lambda: pytest.fail("database opened"))
    with pytest.raises(RuntimeError, match="bootstrap"):
        percept_response_worker.main(registry=other)


def test_durable_stage_rejects_changed_bindings_before_model_or_tool_work(monkeypatch):
    from persistent_cognition import percept_response_runtime as runtime
    old = CapabilityRegistry((registration(revision="1"),))
    new = CapabilityRegistry((registration(revision="2"),))

    def bootstrap(_conn, _interaction, stage, scheduler_key):
        assert stage is runtime.PerceptStage.RESOLVE_REFERENCES
        return {"capability_snapshot": old.snapshot()}

    monkeypatch.setattr(runtime, "_stage_result", bootstrap)
    llm = SimpleNamespace(_response_policy=lambda _: pytest.fail("changed bindings reached model"))
    envelope = SimpleNamespace(step=SimpleNamespace(step_key=runtime.PerceptStage.EVIDENCE_POLICY.value))
    with pytest.raises(RuntimeError, match="bindings changed"):
        runtime._execute_stage(None, llm, envelope, SimpleNamespace(user_text="data"),
                               scheduler_key="test", registry=new)


def test_exhausted_transient_admission_defers_the_same_task_to_a_later_tick(monkeypatch):
    task_id = uuid4()
    task = SimpleNamespace(resumable_state={"situation_task_id": str(task_id)})
    scheduler = SimpleNamespace(tasks={task_id: task}, worker_visible_assignments=lambda: [SimpleNamespace(task_id=task_id)])
    monkeypatch.setattr(situation_runtime, "submit_situation_page", lambda *a, **kw: [])
    monkeypatch.setattr(situation_runtime, "load_scheduler", lambda *a, **kw: scheduler)
    monkeypatch.setattr(situation_runtime, "_quarantine_situation_task", lambda *a, **kw: pytest.fail("retryable work quarantined"))
    class Observation:
        reason = "irrelevant text"
        def model_copy(self, **kwargs):
            return self
    calls = []
    def run(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 1:
            raise worker_runtime.WorkerLaunchDenied(Observation(), retryable=True)
        return {"task_id": str(task_id)}
    monkeypatch.setattr(situation_runtime, "run_situation_task", run)
    assert situation_runtime.drain_situations(None) == []
    assert situation_runtime.drain_situations(None) == [{"task_id": str(task_id)}]
    assert calls == [task_id, task_id]


@pytest.mark.parametrize("retryable", [False, True])
def test_launcher_preserves_structured_retryability_at_the_exception_boundary(monkeypatch, retryable):
    class Observation:
        reason = "structural" if retryable else "resource pressure"
        def model_copy(self, **kwargs):
            return self
    observation = Observation()
    attempt = SimpleNamespace(envelope=None, observation=observation)
    class Connection:
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(worker_runtime, "guarded_claim_worker_step", lambda *a, **kw: attempt)
    monkeypatch.setattr(worker_runtime, "_is_transient_resource_denial", lambda *a, **kw: retryable)
    monkeypatch.setattr("persistent_cognition.reflexes.record_admission_denial", lambda *a, **kw: None)
    launcher = worker_runtime.GuardedWorkerLauncher(Connection, claim_retry_attempts=1)
    with pytest.raises(worker_runtime.WorkerLaunchDenied) as failure:
        launcher.claim(step_id=uuid4(), worker_id="worker")
    assert failure.value.retryable is retryable


def test_nonretryable_admission_keeps_quarantine_semantics(monkeypatch):
    task_id = uuid4()
    scheduler = SimpleNamespace(tasks={task_id: SimpleNamespace(resumable_state={"situation_task_id": "task"})},
                                worker_visible_assignments=lambda: [SimpleNamespace(task_id=task_id)])
    monkeypatch.setattr(situation_runtime, "submit_situation_page", lambda *a, **kw: [])
    monkeypatch.setattr(situation_runtime, "load_scheduler", lambda *a, **kw: scheduler)
    quarantined = []
    monkeypatch.setattr(situation_runtime, "_quarantine_situation_task", lambda *a, **kw: quarantined.append(a[1]))
    def fail(*args, **kwargs):
        raise RuntimeError("ambiguous external effect")
    monkeypatch.setattr(situation_runtime, "run_situation_task", fail)
    assert situation_runtime.drain_situations(None) == []
    assert quarantined == [task_id]
