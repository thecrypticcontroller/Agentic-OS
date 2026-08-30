from tools.cost_control import CostController
from tools.cost_intelligence import CostIntelligence
from tools.provider_decision import ProviderDecisionEngine
from tools.provider_health import ProviderHealthService
from tools.provider_observability import ProviderObservability


def _engine(tmp_path):
    db = tmp_path / "decision.db"
    observer = ProviderObservability(db)
    health = ProviderHealthService(observer)
    costs = CostIntelligence(observer, CostController(db))
    return observer, ProviderDecisionEngine(health, costs)


def test_choose_prefers_observed_healthy_provider(tmp_path):
    observer, engine = _engine(tmp_path)

    for _ in range(4):
        observer.record(
            provider="tavily",
            operation="search",
            status="success",
            latency_ms=100,
        )
    observer.record(
        provider="brave",
        operation="search",
        status="failed",
        latency_ms=2000,
        error_type="TimeoutError",
        error="timeout",
    )

    choice = engine.choose("web_search")

    assert choice is not None
    assert choice.provider == "tavily"
    assert choice.health_score is not None


def test_unknown_provider_is_conservative_and_static_is_deterministic(tmp_path):
    _observer, engine = _engine(tmp_path)

    ranked = engine.rank("web_search")
    assert ranked
    assert all(item.health_score is None for item in ranked)

    static = engine.rank("web_search", mode="static")
    assert static[0].provider == "brave"


def test_reserve_provider_excluded_by_default(tmp_path):
    _observer, engine = _engine(tmp_path)

    names = [item.provider for item in engine.rank("web_search")]
    assert "firecrawl" not in names

    names_with_reserve = [
        item.provider
        for item in engine.rank("web_search", allow_reserve=True)
    ]
    assert "firecrawl" in names_with_reserve
