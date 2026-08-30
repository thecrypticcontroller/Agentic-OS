from datetime import datetime, timedelta, timezone

from tools.worker_registry import WorkerRegistry


def test_register_worker(tmp_path):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_id="worker-1",
        hostname="test-host",
        pid=1234,
        concurrency=4,
    )

    assert worker.worker_id == "worker-1"
    assert worker.hostname == "test-host"
    assert worker.pid == 1234
    assert worker.concurrency == 4
    assert worker.active_runs == 0
    assert worker.status == "running"


def test_heartbeat_updates_active_runs(tmp_path):
    db = tmp_path / "test.db"
    registry = WorkerRegistry(db)

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    assert registry.heartbeat(
        "worker-1",
        active_runs=3,
    ) is True

    worker = registry.get("worker-1")

    assert worker is not None
    assert worker.active_runs == 3
    assert worker.status == "running"


def test_stop_worker(tmp_path):
    db = tmp_path / "test.db"
    registry = WorkerRegistry(db)

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    registry.heartbeat(
        "worker-1",
        active_runs=2,
    )

    assert registry.stop(
        "worker-1"
    ) is True

    worker = registry.get("worker-1")

    assert worker is not None
    assert worker.status == "stopped"
    assert worker.active_runs == 0


def test_snapshot_marks_stale_worker(tmp_path):
    db = tmp_path / "test.db"
    registry = WorkerRegistry(db)

    worker = registry.register(
        worker_id="worker-1",
    )

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=5)
    ).isoformat()

    with registry._connect() as connection:
        connection.execute(
            """
            UPDATE workers
            SET last_heartbeat = ?
            WHERE worker_id = ?
            """,
            (
                stale_time,
                worker.worker_id,
            ),
        )
        connection.commit()

    snapshot = registry.snapshot(
        stale_after_seconds=30
    )

    assert snapshot["count"] == 1
    assert snapshot["healthy"] == 0
    assert snapshot["workers"][0]["status"] == "stale"


def test_snapshot_counts_healthy_workers_and_active_runs(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )
    registry.register(
        worker_id="worker-2",
        concurrency=2,
    )

    registry.heartbeat(
        "worker-1",
        active_runs=2,
    )
    registry.heartbeat(
        "worker-2",
        active_runs=1,
    )

    snapshot = registry.snapshot()

    assert snapshot["count"] == 2
    assert snapshot["healthy"] == 2
    assert snapshot["active_runs"] == 3


def test_drain_worker(tmp_path):
    db = tmp_path / "test.db"

    registry = WorkerRegistry(db)

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    assert registry.drain(
        "worker-1"
    ) is True

    worker = registry.get("worker-1")

    assert worker is not None
    assert worker.status == "draining"


def test_heartbeat_preserves_draining_state(
    tmp_path,
):
    db = tmp_path / "test.db"

    registry = WorkerRegistry(db)

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    registry.drain(
        "worker-1"
    )

    assert registry.heartbeat(
        "worker-1",
        active_runs=2,
    ) is True

    worker = registry.get("worker-1")

    assert worker is not None
    assert worker.status == "draining"
    assert worker.active_runs == 2


def test_draining_worker_is_healthy_until_stopped(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    registry.drain(
        "worker-1"
    )

    snapshot = registry.snapshot()

    assert snapshot["count"] == 1
    assert snapshot["healthy"] == 1
    assert snapshot["workers"][0]["status"] == "draining"


def test_is_stale_detects_expired_heartbeat(
    tmp_path,
):
    db = tmp_path / "test.db"

    registry = WorkerRegistry(db)

    worker = registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=5)
    ).isoformat()

    with registry._connect() as connection:
        connection.execute(
            """
            UPDATE workers
            SET last_heartbeat = ?
            WHERE worker_id = ?
            """,
            (
                stale_time,
                worker.worker_id,
            ),
        )
        connection.commit()

    assert registry.is_stale(
        worker.worker_id,
        stale_after_seconds=30,
    ) is True


def test_is_stale_returns_false_for_healthy_worker(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_id="worker-1",
        concurrency=4,
    )

    assert registry.is_stale(
        worker.worker_id,
        stale_after_seconds=30,
    ) is False
