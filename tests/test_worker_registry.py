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


def test_rejoin_same_logical_worker_gets_new_instance(
    tmp_path,
):
    db = tmp_path / "test.db"

    registry = WorkerRegistry(db)

    first = registry.register(
        worker_name="researcher-01",
        hostname="test-host",
        pid=100,
        concurrency=4,
    )

    registry.stop(
        first.worker_id
    )

    second = registry.register(
        worker_name="researcher-01",
        hostname="test-host",
        pid=200,
        concurrency=4,
    )

    assert first.worker_name == "researcher-01"
    assert second.worker_name == "researcher-01"

    assert first.instance_id
    assert second.instance_id
    assert first.instance_id != second.instance_id

    assert first.worker_id != second.worker_id

    assert first.pid == 100
    assert second.pid == 200

    first_loaded = registry.get(
        first.worker_id
    )
    second_loaded = registry.get(
        second.worker_id
    )

    assert first_loaded is not None
    assert second_loaded is not None

    assert first_loaded.status == "stopped"
    assert second_loaded.status == "running"


def test_explicit_instance_id_is_persisted(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_name="researcher-01",
        instance_id="instance-abc",
        worker_id="worker-abc",
    )

    loaded = registry.get(
        worker.worker_id
    )

    assert loaded is not None
    assert loaded.worker_name == "researcher-01"
    assert loaded.instance_id == "instance-abc"


def test_runtime_snapshot_exposes_worker_identity(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_name="researcher-01",
        hostname="test-host",
        pid=321,
        concurrency=4,
    )

    snapshot = registry.snapshot()

    assert snapshot["count"] == 1

    item = snapshot["workers"][0]

    assert item["worker_id"] == worker.worker_id
    assert item["worker_name"] == "researcher-01"
    assert item["instance_id"] == worker.instance_id
    assert item["hostname"] == "test-host"
    assert item["pid"] == 321
    assert item["concurrency"] == 4
    assert item["status"] == "running"


def test_register_rejects_invalid_concurrency(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    try:
        registry.register(
            worker_name="worker-1",
            concurrency=0,
        )
    except ValueError as exc:
        assert "concurrency" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid concurrency"
        )


def test_same_instance_id_is_rejected(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    registry.register(
        worker_name="worker-1",
        instance_id="instance-1",
    )

    try:
        registry.register(
            worker_name="worker-2",
            instance_id="instance-1",
        )
    except Exception as exc:
        assert "unique" in str(exc).lower() or "constraint" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected duplicate instance_id to be rejected"
        )


def test_snapshot_reports_capacity_breakdown(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    running_one = registry.register(
        worker_name="worker-a",
        concurrency=4,
    )

    running_two = registry.register(
        worker_name="worker-b",
        concurrency=2,
    )

    draining = registry.register(
        worker_name="worker-c",
        concurrency=4,
    )

    registry.heartbeat(
        running_one.worker_id,
        active_runs=2,
    )

    registry.heartbeat(
        running_two.worker_id,
        active_runs=1,
    )

    registry.drain(
        draining.worker_id
    )

    registry.heartbeat(
        draining.worker_id,
        active_runs=3,
    )

    snapshot = registry.snapshot()

    assert snapshot["count"] == 3
    assert snapshot["healthy"] == 3
    assert snapshot["active_runs"] == 6

    assert snapshot["total_capacity"] == 10
    assert snapshot["healthy_capacity"] == 10
    assert snapshot["running_capacity"] == 6
    assert snapshot["draining_capacity"] == 4
    assert snapshot["available_capacity"] == 3


def test_snapshot_never_reports_negative_available_capacity(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_name="worker-a",
        concurrency=2,
    )

    registry.heartbeat(
        worker.worker_id,
        active_runs=5,
    )

    snapshot = registry.snapshot()

    assert snapshot["total_capacity"] == 2
    assert snapshot["running_capacity"] == 2
    assert snapshot["available_capacity"] == 0


def test_stopped_worker_does_not_contribute_healthy_capacity(
    tmp_path,
):
    registry = WorkerRegistry(
        tmp_path / "test.db"
    )

    worker = registry.register(
        worker_name="worker-a",
        concurrency=4,
    )

    registry.stop(
        worker.worker_id
    )

    snapshot = registry.snapshot()

    assert snapshot["count"] == 1
    assert snapshot["healthy"] == 0
    assert snapshot["total_capacity"] == 4
    assert snapshot["healthy_capacity"] == 0
    assert snapshot["running_capacity"] == 0
    assert snapshot["draining_capacity"] == 0
    assert snapshot["available_capacity"] == 0
