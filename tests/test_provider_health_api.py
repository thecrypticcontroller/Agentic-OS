from __future__ import annotations


def test_provider_health_is_exposed_by_api(monkeypatch, tmp_path):
    import api.app as app_module
    from fastapi.testclient import TestClient
    from tools.provider_observability import ProviderObservability
    from tools.provider_health import ProviderHealthService

    observer = ProviderObservability(tmp_path / "api-health.db")
    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=150,
    )
    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=250,
    )

    monkeypatch.setattr(app_module, "observability", observer)
    monkeypatch.setattr(
        app_module,
        "health_service",
        ProviderHealthService(observer),
    )

    response = TestClient(app_module.app).get(
        "/v1/observability/providers"
    )

    assert response.status_code == 200
    providers = {
        item["provider"]: item
        for item in response.json()["providers"]
    }

    brave_health = providers["brave"]["health"]
    assert brave_health["score"] == 100.0
    assert brave_health["state"] == "healthy"
    assert brave_health["recent_failures"] == 0
