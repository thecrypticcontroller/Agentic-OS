from datetime import datetime, timedelta, timezone


def test_worker_module_imports():
    from workers.worker import (
        execute_claimed_run,
        worker_loop,
    )

    assert callable(
        execute_claimed_run
    )

    assert callable(
        worker_loop
    )


def test_registry_claim_sets_lease(
    tmp_path,
):
    from tools.run_registry import (
        RunRecord,
        RunRegistry,
    )

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    registry.save(
        RunRecord(
            run_id="run-1",
            objective="Research AI",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="queued",
            started_at=None,
            completed_at=None,
            duration_ms=None,
            result=None,
            error=None,
        )
    )

    record = registry.claim_next_queued(
        lease_seconds=120
    )

    assert record is not None
    assert record.status == "running"
    assert record.lease_until is not None

    lease = datetime.fromisoformat(
        record.lease_until
    )

    assert lease > datetime.now(
        timezone.utc
    )


def test_renew_lease(
    tmp_path,
):
    from tools.run_registry import (
        RunRecord,
        RunRegistry,
    )

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    registry.save(
        RunRecord(
            run_id="run-1",
            objective="Research AI",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="queued",
            started_at=None,
            completed_at=None,
            duration_ms=None,
            result=None,
            error=None,
        )
    )

    claimed = registry.claim_next_queued(
        lease_seconds=1
    )

    assert claimed is not None

    before = claimed.lease_until

    assert registry.renew_lease(
        "run-1",
        lease_seconds=120,
    ) is True

    renewed = registry.get(
        "run-1"
    )

    assert renewed is not None
    assert renewed.lease_until is not None
    assert renewed.lease_until != before


def test_recover_stale_run(
    tmp_path,
):
    from tools.run_registry import (
        RunRecord,
        RunRegistry,
    )

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    stale_time = (
        datetime.now(
            timezone.utc
        ) - timedelta(
            minutes=5
        )
    ).isoformat()

    registry.save(
        RunRecord(
            run_id="stale-1",
            objective="Stale job",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="running",
            started_at=stale_time,
            completed_at=None,
            duration_ms=None,
            result=None,
            error=None,
            lease_until=stale_time,
        )
    )

    recovered = registry.recover_stale_runs()

    assert recovered == 1

    record = registry.get(
        "stale-1"
    )

    assert record is not None
    assert record.status == "queued"
    assert record.lease_until is None
    assert record.error == (
        "Recovered stale running job."
    )


def test_execute_claimed_run_processes_running_job(
    tmp_path,
    monkeypatch,
):
    from agents.manager import Manager
    from tools.run_registry import (
        RunRecord,
        RunRegistry,
    )
    from workers.worker import (
        execute_claimed_run,
    )

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    record = RunRecord(
        run_id="run-1",
        objective="Research AI agent platforms",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="firecrawl.search+scrape",
        status="running",
        started_at=datetime.now(
            timezone.utc
        ).isoformat(),
        completed_at=None,
        duration_ms=None,
        result=None,
        error=None,
        parent_run_id=None,
        attempt=1,
        lease_until=(
            datetime.now(
                timezone.utc
            ) + timedelta(
                minutes=1
            )
        ).isoformat(),
    )

    registry.save(
        record
    )

    calls = []

    def fake_execute(job):
        calls.append(
            job.id
        )

        job.status = "completed"

        registry.save(
            RunRecord(
                run_id=job.id,
                objective=job.objective,
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=job.target_url,
                tool="test",
                status="completed",
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_ms=1,
                result={"ok": True},
                error=None,
                parent_run_id=job.parent_run_id,
                attempt=job.attempt,
                lease_until=None,
            )
        )

        return job

    monkeypatch.setattr(
        manager,
        "execute",
        fake_execute,
    )

    assert execute_claimed_run(
        manager,
        record,
    ) is True

    assert calls == ["run-1"]

    loaded = registry.get(
        "run-1"
    )

    assert loaded is not None
    assert loaded.status == "completed"


def test_execute_claimed_run_rejects_non_running():
    from agents.manager import Manager
    from tools.run_registry import RunRecord
    from workers.worker import (
        execute_claimed_run,
    )

    manager = Manager()

    record = RunRecord(
        run_id="queued-1",
        objective="Research AI",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="test",
        status="queued",
        started_at=None,
        completed_at=None,
        duration_ms=None,
        result=None,
        error=None,
        parent_run_id=None,
        attempt=1,
    )

    assert execute_claimed_run(
        manager,
        record,
    ) is False


def test_concurrent_executor_can_run_multiple_jobs(
    tmp_path,
):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from agents.manager import Manager
    from tools.run_registry import (
        RunRecord,
        RunRegistry,
    )
    from workers.worker import (
        execute_claimed_run,
    )

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    records = []

    for index in range(4):
        record = RunRecord(
            run_id=f"parallel-{index}",
            objective=f"Research topic {index}",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="running",
            started_at=datetime.now(
                timezone.utc
            ).isoformat(),
            completed_at=None,
            duration_ms=None,
            result=None,
            error=None,
            parent_run_id=None,
            attempt=1,
            lease_until=(
                datetime.now(
                    timezone.utc
                ) + timedelta(
                    minutes=1
                )
            ).isoformat(),
        )

        registry.save(record)
        records.append(record)

    lock = threading.Lock()
    started = []
    finished = []

    def fake_execute(job):
        with lock:
            started.append(job.id)

        time.sleep(0.1)

        with lock:
            finished.append(job.id)

        return job

    manager.execute = fake_execute

    started_at = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:
        futures = [
            executor.submit(
                execute_claimed_run,
                manager,
                record,
            )
            for record in records
        ]

        for future in futures:
            assert future.result() is True

    elapsed = (
        time.perf_counter()
        - started_at
    )

    assert len(started) == 4
    assert len(finished) == 4

    # Four 100ms jobs should overlap substantially.
    assert elapsed < 0.35


def test_worker_claim_slots_zero_when_queue_paused(
    tmp_path,
):
    from tools.runtime_control import RuntimeControl
    from workers.worker import available_claim_slots

    control = RuntimeControl(
        tmp_path / "runtime.db"
    )

    control.pause_queue()

    assert available_claim_slots(
        4,
        control,
    ) == 0


def test_worker_claim_slots_follow_available_capacity(
    tmp_path,
):
    from tools.runtime_control import RuntimeControl
    from workers.worker import available_claim_slots

    control = RuntimeControl(
        tmp_path / "runtime.db"
    )

    assert available_claim_slots(
        4,
        control,
    ) == 4

    assert available_claim_slots(
        -1,
        control,
    ) == 0
