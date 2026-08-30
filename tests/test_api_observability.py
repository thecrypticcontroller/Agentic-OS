from __future__ import annotations

from dataclasses import asdict

from fastapi.testclient import TestClient


def test_run_observability_endpoint(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.provider_observability import ProviderObservability
    from tools.run_registry import RunRecord, RunRegistry

    db_path = tmp_path / "observability-api.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)

    run_id = "run-api-observability"
    registry.save(
        RunRecord(
            run_id=run_id,
            objective="test observability",
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="test",
            status="completed",
            started_at="2026-08-30T00:00:00+00:00",
            completed_at="2026-08-30T00:00:01+00:00",
            duration_ms=1000,
            result={"ok": True},
            error=None,
        )
    )
    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=120,
        run_id=run_id,
    )

    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "observability", observer)

    response = TestClient(app_module.app).get(
        f"/v1/runs/{run_id}/observability"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["summary"]["calls"] == 1
    assert payload["summary"]["successes"] == 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["provider"] == "brave"


def test_run_observability_missing_run_returns_404(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.provider_observability import ProviderObservability
    from tools.run_registry import RunRegistry

    db_path = tmp_path / "missing-run.db"
    monkeypatch.setattr(app_module, "registry", RunRegistry(db_path))
    monkeypatch.setattr(
        app_module,
        "observability",
        ProviderObservability(db_path),
    )

    response = TestClient(app_module.app).get(
        "/v1/runs/does-not-exist/observability"
    )

    assert response.status_code == 404


def test_provider_observability_endpoint(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(
        tmp_path / "provider-health.db"
    )
    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=90,
    )
    observer.record(
        provider="tavily",
        operation="search",
        status="failed",
        latency_ms=200,
        error_type="TimeoutError",
        error="request timed out",
    )

    monkeypatch.setattr(
        app_module,
        "observability",
        observer,
    )

    response = TestClient(app_module.app).get(
        "/v1/observability/providers"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0

    providers = {
        item["provider"]: item
        for item in payload["providers"]
    }
    assert providers["brave"]["summary"]["successes"] == 1
    assert providers["tavily"]["summary"]["failures"] == 1
    assert "capabilities" in providers["brave"]
    assert "reserve_only" in providers["brave"]
