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
