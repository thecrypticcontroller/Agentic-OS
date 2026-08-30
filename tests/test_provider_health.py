from __future__ import annotations


def test_provider_health_unknown_without_observations(tmp_path):
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(tmp_path / "health.db")
    health = ProviderHealthService(observer).snapshot("brave")

    assert health.score is None
    assert health.state == "unknown"
    assert health.calls == 0
    assert health.success_rate == 0.0


def test_provider_health_healthy_when_successful_and_fast(tmp_path):
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(tmp_path / "health.db")
    for latency in (100, 200, 300):
        observer.record(
            provider="brave",
            operation="search",
            status="success",
            latency_ms=latency,
        )

    health = ProviderHealthService(observer).snapshot("brave")

    assert health.score == 100.0
    assert health.state == "healthy"
    assert health.calls == 3
    assert health.failures == 0
    assert health.recent_failures == 0


def test_provider_health_degrades_after_failures(tmp_path):
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(tmp_path / "health.db")
    observer.record(
        provider="tavily",
        operation="search",
        status="success",
        latency_ms=400,
    )
    observer.record(
        provider="tavily",
        operation="search",
        status="failed",
        latency_ms=800,
        error_type="TimeoutError",
        error="timed out",
    )

    health = ProviderHealthService(observer).snapshot("tavily")

    assert 30.0 <= health.score < 85.0
    assert health.state == "degraded"
    assert health.failures == 1
    assert health.recent_failures == 1


def test_provider_health_can_be_unhealthy(tmp_path):
    from tools.provider_health import ProviderHealthService
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(tmp_path / "health.db")
    for _ in range(3):
        observer.record(
            provider="exa",
            operation="search",
            status="failed",
            latency_ms=5000,
            error_type="RuntimeError",
            error="upstream failure",
        )

    health = ProviderHealthService(observer).snapshot("exa")

    assert health.score == 0.0
    assert health.state == "unhealthy"
    assert health.failures == 3
    assert health.recent_failures == 3
