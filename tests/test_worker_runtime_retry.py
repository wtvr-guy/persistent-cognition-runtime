from types import SimpleNamespace
from uuid import uuid4

from prometheist import worker_runtime
from prometheist.worker_protocol import WorkerClaimDecision


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _query, _params):
        return None

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.closed = False

    def cursor(self):
        return FakeCursor(self.rows)

    def close(self):
        self.closed = True


class Copyable:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_copy(self, deep=False):
        del deep
        return self


def test_structured_capacity_shortage_is_retryable_without_reason_parsing():
    reservation_id = uuid4()
    snapshot = SimpleNamespace(
        healthy=True,
        capacity_by_resource_id=lambda: {
            "host-ram-mib": SimpleNamespace(available_for_new_work=4092)
        },
    )
    observation = SimpleNamespace(
        decision=WorkerClaimDecision.DENIED,
        resource_snapshot=snapshot,
        target_reservation_ids=[reservation_id],
        reason="this text is deliberately irrelevant to retry classification",
    )
    conn = FakeConnection([(reservation_id, "host-ram-mib", 4096)])

    assert worker_runtime._is_transient_resource_denial(
        conn,
        observation=observation,
        scheduler_key="default",
    )


def test_structural_denial_is_not_retryable_when_reserved_capacity_is_safe():
    reservation_id = uuid4()
    snapshot = SimpleNamespace(
        healthy=True,
        capacity_by_resource_id=lambda: {
            "host-ram-mib": SimpleNamespace(available_for_new_work=8192)
        },
    )
    observation = SimpleNamespace(
        decision=WorkerClaimDecision.DENIED,
        resource_snapshot=snapshot,
        target_reservation_ids=[reservation_id],
        reason="worker step already has a live claim",
    )
    conn = FakeConnection([(reservation_id, "host-ram-mib", 4096)])

    assert not worker_runtime._is_transient_resource_denial(
        conn,
        observation=observation,
        scheduler_key="default",
    )


def test_guarded_launcher_reobserves_transient_denial_before_spawning(monkeypatch):
    step_id = uuid4()
    denial = Copyable(reason="pressure", decision=WorkerClaimDecision.DENIED)
    envelope = Copyable()
    attempts = iter(
        [
            SimpleNamespace(envelope=None, observation=denial),
            SimpleNamespace(envelope=envelope, observation=Copyable(reason="granted")),
        ]
    )
    connections = []
    sleeps = []

    def connection_factory():
        conn = FakeConnection()
        connections.append(conn)
        return conn

    monkeypatch.setattr(
        worker_runtime,
        "guarded_claim_worker_step",
        lambda *_args, **_kwargs: next(attempts),
    )
    monkeypatch.setattr(
        worker_runtime,
        "_is_transient_resource_denial",
        lambda *_args, **_kwargs: True,
    )
    launcher = worker_runtime.GuardedWorkerLauncher(
        connection_factory,
        claim_retry_attempts=2,
        claim_retry_delay_seconds=0.25,
        sleep=sleeps.append,
    )

    assert launcher.claim(step_id=step_id, worker_id="worker-a") is envelope
    assert sleeps == [0.25]
    assert len(connections) == 2
    assert all(conn.closed for conn in connections)
