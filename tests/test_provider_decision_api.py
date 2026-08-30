from fastapi.testclient import TestClient


def test_provider_decision_endpoint(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.cost_control import CostController
    from tools.cost_intelligence import CostIntelligence
    from tools.provider_decision import ProviderDecisionEngine
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    db = tmp_path / "decision-api.db"
    observer = ProviderObservability(db)
    controller = CostController(db)
    health = ProviderHealthService(observer)
    costs = CostIntelligence(observer, controller)
    engine = ProviderDecisionEngine(health, costs)

    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=100,
    )

    monkeypatch.setattr(app_module, "decision_engine", engine)

    response = TestClient(app_module.app).get(
        "/v1/observability/decisions/web_search"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability"] == "web_search"
    assert payload["allow_reserve"] is False
    assert payload["decisions"]
    assert all(item["reserve_only"] is False for item in payload["decisions"])


def test_provider_decision_endpoint_can_include_reserve(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.cost_control import CostController
    from tools.cost_intelligence import CostIntelligence
    from tools.provider_decision import ProviderDecisionEngine
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    db = tmp_path / "decision-reserve-api.db"
    observer = ProviderObservability(db)
    controller = CostController(db)
    engine = ProviderDecisionEngine(
        ProviderHealthService(observer),
        CostIntelligence(observer, controller),
    )

    monkeypatch.setattr(app_module, "decision_engine", engine)

    response = TestClient(app_module.app).get(
        "/v1/observability/decisions/web_search",
        params={"allow_reserve": "true"},
    )

    assert response.status_code == 200
    providers = [item["provider"] for item in response.json()["decisions"]]
    assert "firecrawl" in providers
