from tools.job_queue import JobQueue
from tools.run_registry import RunRecord, RunRegistry


def test_claim_next_marks_job_running(tmp_path):
    db = tmp_path / "test.db"

    registry = RunRegistry(db)

    registry.save(
        RunRecord(
            run_id="run-1",
            objective="Research AI agents",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="queued",
            started_at=None,
            completed_at=None,
            duration_ms=None,
            result=None,
            error=None,
        )
    )

    queue = JobQueue(db)

    item = queue.claim_next()

    assert item is not None
    assert item.run_id == "run-1"

    record = registry.get("run-1")

    assert record is not None
    assert record.status == "running"


def test_claim_next_empty_queue(tmp_path):
    db = tmp_path / "test.db"

    RunRegistry(db)

    queue = JobQueue(db)

    assert queue.claim_next() is None
