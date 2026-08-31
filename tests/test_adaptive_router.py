from unittest import mock

from tools.provider_router import ProviderRouter
from tools.provider_observability import ProviderObservability
from tools.web_research import WebSearchResult


def _results():
    return [WebSearchResult(url="https://example.com", title="Example", position=1)]


def test_adaptive_routing_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_OS_ADAPTIVE_ROUTING", raising=False)
    observer = ProviderObservability(tmp_path / "router.db")
    router = ProviderRouter(observer=observer)

    specs = router._ordered_specs("web_search", allow_reserve=False)
    assert [item.name for item in specs] == ["brave", "tavily", "exa"]


def test_adaptive_routing_promotes_configured_ranked_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_OS_ADAPTIVE_ROUTING", "true")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "")

    observer = ProviderObservability(tmp_path / "router-adaptive.db")
    router = ProviderRouter(observer=observer)

    with mock.patch("tools.provider_router.ProviderDecisionEngine.rank") as rank:
        rank.return_value = [
            mock.Mock(provider="tavily"),
            mock.Mock(provider="brave"),
        ]
        specs = router._ordered_specs("web_search", allow_reserve=False)

    assert [item.name for item in specs] == ["tavily", "brave", "exa"]


def test_adaptive_routing_keeps_reserve_out_without_explicit_permission(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_OS_ADAPTIVE_ROUTING", "true")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-key")

    observer = ProviderObservability(tmp_path / "reserve.db")
    router = ProviderRouter(observer=observer)

    specs = router._ordered_specs("web_search", allow_reserve=False)
    assert "firecrawl" not in [item.name for item in specs]

    with mock.patch("tools.provider_router._tavily_search", return_value=_results()):
        results = router.search("test", limit=1, allow_reserve=False)

    assert results[0].url == "https://example.com"

def test_adaptive_routing_keeps_reserve_after_normal_providers_when_allowed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AGENT_OS_ADAPTIVE_ROUTING", "true")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-key")

    observer = ProviderObservability(tmp_path / "reserve-order.db")
    router = ProviderRouter(observer=observer)

    with mock.patch(
        "tools.provider_router.ProviderDecisionEngine.rank"
    ) as rank:
        rank.return_value = [
            mock.Mock(provider="firecrawl"),
            mock.Mock(provider="tavily"),
            mock.Mock(provider="brave"),
            mock.Mock(provider="exa"),
        ]

        specs = router._ordered_specs(
            "web_search",
            allow_reserve=True,
        )

    names = [item.name for item in specs]

    assert names[-1] == "firecrawl"
    assert set(names[:-1]) == {"brave", "tavily", "exa"}
