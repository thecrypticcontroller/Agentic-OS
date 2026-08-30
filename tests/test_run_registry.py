from tools.run_registry import RunRecord, RunRegistry


def test_registry_save_get(tmp_path):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    record = RunRecord(
        run_id="run-1",
        objective="Test objective",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="firecrawl.search+scrape",
        status="completed",
        started_at="2026-08-28T00:00:00+00:00",
        completed_at="2026-08-28T00:00:01+00:00",
        duration_ms=1000,
        result={
            "type": "research",
            "answer": "Test answer",
        },
        error=None,
    )

    registry.save(record)

    loaded = registry.get("run-1")

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.status == "completed"
    assert loaded.duration_ms == 1000
    assert loaded.result["answer"] == "Test answer"


def test_registry_list_recent(tmp_path):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    for index in range(3):
        registry.save(
            RunRecord(
                run_id=f"run-{index}",
                objective=f"Objective {index}",
                worker="researcher",
                research_mode="normal",
                target_url=None,
                tool="firecrawl.search+scrape",
                status="completed",
                started_at=None,
                completed_at=None,
                duration_ms=index,
                result={"index": index},
                error=None,
            )
        )

    records = registry.list_recent(2)

    assert len(records) == 2
    assert records[0].run_id == "run-2"
    assert records[1].run_id == "run-1"
    assert registry.count() == 3


def test_registry_dump(tmp_path):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    registry.save(
        RunRecord(
            run_id="run-42",
            objective="Dump test",
            worker="browser_worker",
            research_mode=None,
            target_url="https://example.com",
            tool="browser_harness",
            status="completed",
            started_at=None,
            completed_at=None,
            duration_ms=250,
            result={"title": "Example Domain"},
            error=None,
        )
    )

    dumped = registry.dump("run-42")

    assert dumped is not None
    assert dumped["run_id"] == "run-42"
    assert dumped["result"]["title"] == "Example Domain"

def test_registry_filters(tmp_path):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    records = [
        RunRecord(
            run_id="browser-1",
            objective="Open site",
            worker="browser_worker",
            research_mode=None,
            target_url="https://example.com",
            tool="browser_harness",
            status="completed",
            started_at=None,
            completed_at=None,
            duration_ms=100,
            result={"ok": True},
            error=None,
        ),
        RunRecord(
            run_id="research-1",
            objective="Research AI",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="completed",
            started_at=None,
            completed_at=None,
            duration_ms=200,
            result={"ok": True},
            error=None,
        ),
        RunRecord(
            run_id="deep-1",
            objective="Deep research AI",
            worker="researcher",
            research_mode="deep",
            target_url=None,
            tool="firecrawl.agent",
            status="failed",
            started_at=None,
            completed_at=None,
            duration_ms=300,
            result=None,
            error="RuntimeError: test",
        ),
    ]

    for record in records:
        registry.save(record)

    browser_runs = registry.list_runs(
        worker="browser_worker"
    )

    assert len(browser_runs) == 1
    assert browser_runs[0].run_id == "browser-1"

    deep_runs = registry.list_runs(
        research_mode="deep"
    )

    assert len(deep_runs) == 1
    assert deep_runs[0].run_id == "deep-1"

    failed_runs = registry.list_runs(
        status="failed"
    )

    assert len(failed_runs) == 1
    assert failed_runs[0].run_id == "deep-1"

    researcher_runs = registry.list_runs(
        worker="researcher",
        status="completed",
    )

    assert len(researcher_runs) == 1
    assert researcher_runs[0].run_id == "research-1"

def test_registry_migrates_and_tracks_attempts(tmp_path):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    first = RunRecord(
        run_id="run-1",
        parent_run_id=None,
        attempt=1,
        objective="Retry test",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="firecrawl.search+scrape",
        status="failed",
        started_at=None,
        completed_at=None,
        duration_ms=100,
        result=None,
        error="test failure",
    )

    second = RunRecord(
        run_id="run-2",
        parent_run_id="run-1",
        attempt=2,
        objective="Retry test",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="firecrawl.search+scrape",
        status="completed",
        started_at=None,
        completed_at=None,
        duration_ms=200,
        result={"ok": True},
        error=None,
    )

    registry.save(first)
    registry.save(second)

    assert registry.next_attempt("run-1") == 3

    lineage = registry.list_runs(
        parent_run_id="run-1"
    )

    assert len(lineage) == 2

    assert registry.get("run-1").attempt == 1
    assert registry.get("run-2").attempt == 2


def test_claim_next_queued_for_worker_reserves_capacity(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-1",
        concurrency=2,
    )

    for run_id in ("run-1", "run-2", "run-3"):
        registry.save(
            RunRecord(
                run_id=run_id,
                objective=f"Objective {run_id}",
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

    first = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert first is not None
    assert first.status == "running"

    loaded_worker = workers.get(
        worker.worker_id
    )

    assert loaded_worker is not None
    assert loaded_worker.active_runs == 1

    second = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert second is not None
    assert second.status == "running"

    loaded_worker = workers.get(
        worker.worker_id
    )

    assert loaded_worker is not None
    assert loaded_worker.active_runs == 2

    third = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert third is None

    queued = registry.list_runs(
        status="queued",
        limit=10,
    )

    assert len(queued) == 1


def test_claim_next_queued_for_worker_rejects_draining_worker(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-draining",
        concurrency=2,
    )

    workers.drain(
        worker.worker_id
    )

    registry.save(
        RunRecord(
            run_id="run-1",
            objective="Objective",
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

    claimed = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert claimed is None

    loaded = registry.get(
        "run-1"
    )

    assert loaded is not None
    assert loaded.status == "queued"


def test_claim_next_queued_for_worker_is_atomic_at_capacity(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-full",
        concurrency=1,
    )

    registry.save(
        RunRecord(
            run_id="run-1",
            objective="Objective 1",
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

    registry.save(
        RunRecord(
            run_id="run-2",
            objective="Objective 2",
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

    first = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert first is not None

    second = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert second is None

    running = registry.list_runs(
        status="running",
        limit=10,
    )

    queued = registry.list_runs(
        status="queued",
        limit=10,
    )

    assert len(running) == 1
    assert len(queued) == 1


def test_concurrent_workers_cannot_overclaim_capacity(
    tmp_path,
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker_one = workers.register(
        worker_id="worker-one",
        concurrency=1,
    )

    worker_two = workers.register(
        worker_id="worker-two",
        concurrency=1,
    )

    for run_id in ("run-1", "run-2"):
        registry.save(
            RunRecord(
                run_id=run_id,
                objective=f"Objective {run_id}",
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

    barrier = threading.Barrier(2)

    def claim(worker_id):
        barrier.wait()

        return registry.claim_next_queued_for_worker(
            worker_id,
            lease_seconds=120,
        )

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        futures = [
            executor.submit(
                claim,
                worker_one.worker_id,
            ),
            executor.submit(
                claim,
                worker_two.worker_id,
            ),
        ]

        results = [
            future.result()
            for future in futures
        ]

    successful = [
        result
        for result in results
        if result is not None
    ]

    assert len(successful) == 2

    assert {
        result.run_id
        for result in successful
    } == {
        "run-1",
        "run-2",
    }

    assert workers.get(
        worker_one.worker_id
    ).active_runs == 1

    assert workers.get(
        worker_two.worker_id
    ).active_runs == 1

    assert len(
        registry.list_runs(
            status="running",
            limit=10,
        )
    ) == 2

    assert len(
        registry.list_runs(
            status="queued",
            limit=10,
        )
    ) == 0


def test_atomic_claim_records_worker_owner(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-owner",
        concurrency=2,
    )

    registry.save(
        RunRecord(
            run_id="run-owner",
            objective="Owned run",
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

    claimed = registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    assert claimed is not None
    assert claimed.worker_id == worker.worker_id

    loaded = registry.get(
        "run-owner"
    )

    assert loaded is not None
    assert loaded.worker_id == worker.worker_id


def test_reconcile_worker_active_runs_repairs_counter(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-reconcile",
        concurrency=4,
    )

    for run_id in ("run-a", "run-b"):
        registry.save(
            RunRecord(
                run_id=run_id,
                objective="Running job",
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

    registry.claim_next_queued_for_worker(
        worker.worker_id
    )
    registry.claim_next_queued_for_worker(
        worker.worker_id
    )

    with workers._connect() as connection:
        connection.execute(
            """
            UPDATE workers
            SET active_runs = 0
            WHERE worker_id = ?
            """,
            (worker.worker_id,),
        )
        connection.commit()

    assert workers.get(
        worker.worker_id
    ).active_runs == 0

    reconciled = registry.reconcile_worker_active_runs(
        worker.worker_id
    )

    assert reconciled == 2

    loaded = workers.get(
        worker.worker_id
    )

    assert loaded is not None
    assert loaded.active_runs == 2


def test_reconcile_ignores_completed_runs(
    tmp_path,
):
    from tools.worker_registry import WorkerRegistry

    db = tmp_path / "test.db"

    registry = RunRegistry(db)
    workers = WorkerRegistry(db)

    worker = workers.register(
        worker_id="worker-completed",
        concurrency=4,
    )

    registry.save(
        RunRecord(
            run_id="run-completed",
            objective="Completed job",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="completed",
            started_at=None,
            completed_at=None,
            duration_ms=10,
            result={"ok": True},
            error=None,
            worker_id=worker.worker_id,
        )
    )

    reconciled = registry.reconcile_worker_active_runs(
        worker.worker_id
    )

    assert reconciled == 0

    loaded = workers.get(
        worker.worker_id
    )

    assert loaded is not None
    assert loaded.active_runs == 0


def test_reconcile_missing_worker_returns_none(
    tmp_path,
):
    registry = RunRegistry(
        tmp_path / "test.db"
    )

    assert registry.reconcile_worker_active_runs(
        "missing-worker"
    ) is None
