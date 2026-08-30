from fastapi.testclient import TestClient


def test_run_cost_endpoint(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.cost_control import CostController
    from tools.cost_intelligence import CostIntelligence
    from tools.provider_observability import ProviderObservability
    from tools.run_registry import RunRecord, RunRegistry

    db_path = tmp_path / "cost-api.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)

    run_id = "run-cost-api"
    registry.save(
        RunRecord(
            run_id=run_id,
            objective="test cost api",
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
    controller.create_budget(run_id, max_usd=2.0)
    controller.record_usage(
        run_id,
        model="gemini",
        task="test",
        input_tokens=100,
        output_tokens=25,
        thinking_tokens=5,
        estimated_cost_usd=0.5,
    )

    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "cost_controller", controller)
    monkeypatch.setattr(
        app_module,
        "cost_intelligence",
        CostIntelligence(observer, controller),
    )

    response = TestClient(app_module.app).get(
        f"/v1/runs/{run_id}/cost"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["spent_usd"] == 0.5
    assert payload["remaining_usd"] == 1.5
    assert payload["total_tokens"] == 130


def test_run_cost_missing_run_returns_404(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.cost_control import CostController
    from tools.cost_intelligence import CostIntelligence
    from tools.provider_observability import ProviderObservability
    from tools.run_registry import RunRegistry

    db_path = tmp_path / "missing-cost-api.db"
    registry = RunRegistry(db_path)
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)

    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "cost_controller", controller)
    monkeypatch.setattr(
        app_module,
        "cost_intelligence",
        CostIntelligence(observer, controller),
    )

    response = TestClient(app_module.app).get(
        "/v1/runs/does-not-exist/cost"
    )

    assert response.status_code == 404


def test_provider_observability_includes_cost(monkeypatch, tmp_path):
    import api.app as app_module
    from tools.cost_control import CostController
    from tools.cost_intelligence import CostIntelligence
    from tools.provider_observability import ProviderObservability

    db_path = tmp_path / "provider-cost-api.db"
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)
    observer.record(
        provider="gemini",
        operation="generate",
        status="success",
        latency_ms=120,
        input_tokens=100,
        output_tokens=50,
        thinking_tokens=0,
        total_tokens=150,
        estimated_cost_usd=0.03,
    )

    monkeypatch.setattr(app_module, "observability", observer)
    monkeypatch.setattr(
        app_module,
        "cost_intelligence",
        CostIntelligence(observer, controller),
    )

    response = TestClient(app_module.app).get(
        "/v1/observability/providers"
    )

    assert response.status_code == 200
    providers = {
        item["provider"]: item
        for item in response.json()["providers"]
    }
    assert "cost" in providers["gemini"]
    assert providers["gemini"]["cost"]["calls"] == 1
    assert providers["gemini"]["cost"]["estimated_cost_usd"] == 0.03
