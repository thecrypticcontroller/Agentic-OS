from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def test_claimed_worker_run_preserves_run_context_for_provider_telemetry(
    tmp_path,
    monkeypatch,
):
    from agents.manager import Manager
    from tools.provider_observability import ProviderObservability
    from tools.provider_router import ProviderRouter
    from tools.run_context import current_run_id
    from tools.run_registry import RunRecord, RunRegistry
    from workers.worker import execute_claimed_run

    db_path = tmp_path / "worker-observability.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)

    monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")

    monkeypatch.setattr(
        "agents.manager.research",
        lambda _objective: SimpleNamespace(
            query="worker observability",
            source_count=1,
            sources=[],
            summary="ok",
        ),
    )
    monkeypatch.setattr(
        "agents.manager.synthesize",
        lambda _report: SimpleNamespace(
            query="worker observability",
            answer="ok",
            key_findings=[],
            citations=[],
            caveats=[],
        ),
    )
    monkeypatch.setattr(
        "tools.provider_router._brave_search",
        lambda _query, _limit, _key: [
            SimpleNamespace(
                url="https://example.com",
                title="Example",
                description="example",
                position=1,
            )
        ],
    )

    record = RunRecord(
        run_id="worker-run-1",
        objective="Research worker observability",
        worker="researcher",
        research_mode="normal",
        target_url=None,
        tool="test",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        duration_ms=None,
        result=None,
        error=None,
        parent_run_id=None,
        attempt=1,
        lease_until=(
            datetime.now(timezone.utc) + timedelta(minutes=1)
        ).isoformat(),
    )
    registry.save(record)

    manager = Manager(registry=registry)

    original_research = __import__(
        "agents.manager",
        fromlist=["research"],
    ).research

    def wrapped_research(objective):
        router = ProviderRouter(observer=observer)
        result = router.search("worker observability", limit=1)
        assert result
        assert current_run_id() == "worker-run-1"
        return original_research(objective)

    monkeypatch.setattr("agents.manager.research", wrapped_research)

    assert execute_claimed_run(manager, record) is True
    assert current_run_id() is None

    events = observer.recent(run_id="worker-run-1", limit=10)
    assert len(events) == 1
    assert events[0].provider == "brave"
    assert events[0].status == "success"
