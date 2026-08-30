from __future__ import annotations

from dataclasses import dataclass


def test_manager_binds_provider_events_to_job_run(monkeypatch, tmp_path):
    from agents.manager import Manager
    from agents.researcher import ResearchResult
    from agents.synthesizer import SynthesisResult
    from tools.provider_observability import ProviderObservability
    from tools.run_context import current_run_id
    from tools.run_registry import RunRegistry

    db_path = tmp_path / "context.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)
    manager = Manager(registry=registry)

    def fake_research(objective):
        observer.record(
            provider="brave",
            operation="search",
            status="success",
            latency_ms=12,
        )
        return ResearchResult(
            query=objective,
            source_count=1,
            sources=[],
            summary="test",
        )

    @dataclass
    class FakeCitation:
        title: str = "Example"
        url: str = "https://example.com"
        domain: str = "example.com"

    def fake_synthesize(report):
        return SynthesisResult(
            query=report.query,
            answer="ok",
            key_findings=[],
            citations=[FakeCitation()],
            caveats=[],
        )

    monkeypatch.setattr("agents.manager.research", fake_research)
    monkeypatch.setattr("agents.manager.synthesize", fake_synthesize)

    job = manager.create_job("test context correlation")
    result = manager.execute(job)

    assert result.status == "completed"
    assert current_run_id() is None

    events = observer.recent(run_id=job.id)
    assert len(events) == 1
    assert events[0].provider == "brave"
    assert events[0].run_id == job.id
