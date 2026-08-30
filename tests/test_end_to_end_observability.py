from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from agents.manager import Manager
from tools.provider_observability import ProviderObservability
from tools.provider_router import ProviderRouter
from tools.run_registry import RunRegistry
from tools.run_context import current_run_id


def test_manager_run_produces_correlated_provider_trace(monkeypatch, tmp_path):
    db_path = tmp_path / "e2e-observability.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)

    monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")

    def fake_research(_objective):
        router = ProviderRouter(observer=observer)
        results = router.search("observability integration", limit=1)
        assert results
        assert current_run_id() is not None
        return SimpleNamespace()

    def fake_synthesize(_report):
        return SimpleNamespace(
            query="observability integration",
            answer="ok",
            key_findings=[],
            citations=[],
            caveats=[],
        )

    monkeypatch.setattr("agents.manager.research", fake_research)
    monkeypatch.setattr("agents.manager.synthesize", fake_synthesize)

    with mock.patch(
        "tools.provider_router._brave_search",
        return_value=[
            SimpleNamespace(
                url="https://example.com",
                title="Example",
                description="example",
                position=1,
            )
        ],
    ):
        manager = Manager(registry=registry)
        job = manager.create_job("research observability integration")
        result = manager.execute(job)

    assert result.status == "completed"
    assert result.id == job.id
    assert current_run_id() is None

    events = observer.recent(run_id=job.id, limit=10)
    assert len(events) == 1
    assert events[0].provider == "brave"
    assert events[0].operation == "search"
    assert events[0].status == "success"
    assert events[0].run_id == job.id

    summary = observer.summary(run_id=job.id)
    assert summary["calls"] == 1
    assert summary["successes"] == 1
