from __future__ import annotations

from dataclasses import dataclass, field


def test_provider_observability_records_and_summarizes(tmp_path):
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(
        tmp_path / "observability.db"
    )

    observer.record(
        provider="brave",
        operation="search",
        status="success",
        latency_ms=120,
        run_id="run-1",
    )
    observer.record(
        provider="tavily",
        operation="search",
        status="failed",
        latency_ms=80,
        run_id="run-1",
        error_type="TimeoutError",
        error="request timed out",
    )
    observer.record(
        provider="exa",
        operation="search",
        status="skipped",
        latency_ms=0,
        run_id="run-1",
    )

    summary = observer.summary(run_id="run-1")

    assert summary["calls"] == 3
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["skipped"] == 1
    assert summary["success_rate"] == 0.3333
    assert summary["total_latency_ms"] == 200
    assert summary["avg_latency_ms"] == 66.67

    events = observer.recent(
        provider="tavily",
        run_id="run-1",
    )

    assert len(events) == 1
    assert events[0].error_type == "TimeoutError"


def test_provider_router_persists_attempts(tmp_path, monkeypatch):
    from tools.provider_observability import ProviderObservability
    from tools.provider_router import (
        ProviderRouter,
        _SEARCH_ADAPTERS,
    )
    from tools.web_research import WebSearchResult

    observer = ProviderObservability(
        tmp_path / "router.db"
    )
    router = ProviderRouter(
        observer,
        run_id="run-router",
    )

    monkeypatch.setenv(
        "BRAVE_API_KEY",
        "test-key",
    )
    original = _SEARCH_ADAPTERS["brave"]
    monkeypatch.setitem(
        _SEARCH_ADAPTERS,
        "brave",
        lambda query, limit, key: [
            WebSearchResult(
                url="https://example.com",
                title="Example",
                position=1,
            )
        ],
    )

    try:
        results = router.search(
            "example",
            limit=1,
        )
    finally:
        _SEARCH_ADAPTERS["brave"] = original

    assert results[0].title == "Example"

    summary = observer.summary(
        run_id="run-router"
    )
    assert summary["calls"] == 1
    assert summary["successes"] == 1

    event = observer.recent(
        run_id="run-router"
    )[0]
    assert event.provider == "brave"
    assert event.operation == "search"
    assert event.status == "success"


def test_gemini_observability_records_usage(tmp_path, monkeypatch):
    from tools.gemini_client import GeminiClient
    from tools.provider_observability import ProviderObservability

    @dataclass
    class FakeUsage:
        prompt_token_count: int = 100
        candidates_token_count: int = 40
        thoughts_token_count: int = 10
        total_token_count: int = 150

    @dataclass
    class FakeResponse:
        text: str = "fake response"
        usage_metadata: FakeUsage = field(
            default_factory=FakeUsage
        )

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    observer = ProviderObservability(
        tmp_path / "gemini.db"
    )
    client = GeminiClient(
        observer=observer,
        run_id="run-gemini",
    )
    client.client = FakeClient()

    result = client.generate(
        "Extract the title from a page"
    )

    assert result.total_tokens == 150

    event = observer.recent(
        run_id="run-gemini"
    )[0]
    assert event.provider == "gemini"
    assert event.status == "success"
    assert event.model == "gemini-3.1-flash-lite"
    assert event.input_tokens == 100
    assert event.output_tokens == 40
    assert event.thinking_tokens == 10
    assert event.total_tokens == 150
    assert event.estimated_cost_usd == result.estimated_cost_usd
