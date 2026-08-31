"""
Tests for tools.provider_router — covers:
  A. Registry ordering
  B. Missing-key skip behaviour
  C. Provider fallback
  D. Reserve protection (Firecrawl)
  E. Zero-key raises ProviderUnavailableError (not [])
  F. No API-key leakage in exception messages
  G. web_research public API compatibility
  H. Jina extraction (no API key required)
  I. research_url() free-first before direct HTTP
  J. Manager -> Researcher -> ProviderRouter integration
"""
from __future__ import annotations

import os
import unittest.mock as mock

import pytest

from tools.provider_registry import free_first_chain, reserve_provider
from tools.provider_router import (
    ProviderAttempt,
    ProviderRouter,
    ProviderUnavailableError,
)
from tools.web_research import WebPageResult, WebSearchResult, search_web


# ---------------------------------------------------------------------------
# A. Registry ordering
# ---------------------------------------------------------------------------

def test_search_chain_order():
    assert free_first_chain("web_search") == ["brave", "tavily", "exa"]


def test_extract_chain_order():
    assert free_first_chain("web_extract") == ["jina", "exa"]


def test_deep_chain_has_no_normal_providers():
    assert free_first_chain("deep_research") == []


def test_reserve_search_is_firecrawl():
    assert reserve_provider("web_search") == "firecrawl"


def test_reserve_deep_is_firecrawl():
    assert reserve_provider("deep_research") == "firecrawl"


# ---------------------------------------------------------------------------
# B. Missing-key skip behaviour
# ---------------------------------------------------------------------------

def _cleared(**keys):
    """Return env patch dict with given keys cleared."""
    return {k: "" for k in keys}


def _fake_search_results():
    return [WebSearchResult(url="https://example.com", title="T", position=1)]


def test_brave_skipped_when_key_missing(monkeypatch):
    """Brave skipped; Tavily succeeds."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")

    with mock.patch("tools.provider_router._tavily_search", return_value=_fake_search_results()):
        results = ProviderRouter().search("test query")

    assert results[0].url == "https://example.com"


def test_tavily_skipped_when_key_missing(monkeypatch):
    """Tavily skipped; Exa succeeds."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")

    with mock.patch("tools.provider_router._exa_search", return_value=_fake_search_results()):
        results = ProviderRouter().search("test query")

    assert results[0].url == "https://example.com"


def test_exa_skipped_when_key_missing(monkeypatch):
    """All keys missing -> ProviderUnavailableError."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")

    with pytest.raises(ProviderUnavailableError):
        ProviderRouter().search("test query", allow_reserve=False)


# ---------------------------------------------------------------------------
# C. Fallback
# ---------------------------------------------------------------------------

def test_brave_fails_tavily_succeeds(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")

    with mock.patch("tools.provider_router._brave_search", side_effect=RuntimeError("HTTP 500")):
        with mock.patch("tools.provider_router._tavily_search", return_value=_fake_search_results()):
            results = ProviderRouter().search("test query")

    assert results[0].url == "https://example.com"


def test_tavily_fails_exa_succeeds(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")

    with mock.patch("tools.provider_router._tavily_search", side_effect=RuntimeError("HTTP 401")):
        with mock.patch("tools.provider_router._exa_search", return_value=_fake_search_results()):
            results = ProviderRouter().search("test query")

    assert results[0].url == "https://example.com"


# ---------------------------------------------------------------------------
# D. Reserve protection
# ---------------------------------------------------------------------------

def test_firecrawl_not_called_when_allow_reserve_false(monkeypatch):
    """Firecrawl must NOT be called when allow_reserve=False."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-fc-key")

    with mock.patch("tools.provider_router._firecrawl_search") as fc_mock:
        with pytest.raises(ProviderUnavailableError):
            ProviderRouter().search("test query", allow_reserve=False)
        fc_mock.assert_not_called()


def test_firecrawl_called_when_allow_reserve_true(monkeypatch):
    """Firecrawl MAY be called when allow_reserve=True."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-fc-key")

    with mock.patch(
        "tools.provider_router._firecrawl_search",
        return_value=_fake_search_results(),
    ) as fc_mock:
        results = ProviderRouter().search("test query", allow_reserve=True)

    fc_mock.assert_called_once()
    assert results[0].url == "https://example.com"


# ---------------------------------------------------------------------------
# E. Zero-key behaviour
# ---------------------------------------------------------------------------

def test_zero_keys_raises_not_returns_empty(monkeypatch):
    """With all keys absent, search_web must raise, NOT return []."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")

    with pytest.raises(ProviderUnavailableError) as exc_info:
        search_web("latest AI agent frameworks", limit=3)

    err = str(exc_info.value)
    assert "web_search" in err
    assert "skipped" in err


def test_zero_keys_does_not_call_firecrawl(monkeypatch):
    """Even if FIRECRAWL_API_KEY is set, allow_reserve=False must block it."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "some-key")

    with mock.patch("tools.provider_router._firecrawl_search") as fc_mock:
        with pytest.raises(ProviderUnavailableError):
            search_web("test", limit=1)
        fc_mock.assert_not_called()


# ---------------------------------------------------------------------------
# F. No secret leakage
# ---------------------------------------------------------------------------

def test_no_secret_in_exception(monkeypatch):
    secret = "SUPER_SECRET_API_KEY_1234567890abcd"
    monkeypatch.setenv("BRAVE_API_KEY", secret)
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")

    with mock.patch(
        "tools.provider_router._brave_search",
        side_effect=RuntimeError(f"Auth failed: {secret}"),
    ):
        with pytest.raises(ProviderUnavailableError) as exc_info:
            ProviderRouter().search("test", allow_reserve=False)

    assert secret not in str(exc_info.value)


def test_no_secret_in_attempt_reason(monkeypatch):
    secret = "SUPER_SECRET_API_KEY_1234567890abcd"
    monkeypatch.setenv("BRAVE_API_KEY", secret)
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")

    with mock.patch(
        "tools.provider_router._brave_search",
        side_effect=RuntimeError(f"token={secret}"),
    ):
        try:
            ProviderRouter().search("test", allow_reserve=False)
        except ProviderUnavailableError as exc:
            for attempt in exc.attempts:
                if attempt.reason:
                    assert secret not in attempt.reason


# ---------------------------------------------------------------------------
# G. Existing public API compatibility
# ---------------------------------------------------------------------------

def test_search_web_raises_on_empty_query():
    with pytest.raises(ValueError, match="empty"):
        search_web("")


def test_provider_attempt_dataclass():
    a = ProviderAttempt(provider="brave", status="skipped", reason="BRAVE_API_KEY is not configured")
    assert a.provider == "brave"
    assert a.status == "skipped"
    assert "BRAVE_API_KEY" in a.reason


def test_provider_unavailable_error_contains_capability():
    attempts = [
        ProviderAttempt(provider="brave", status="skipped", reason="no key"),
    ]
    err = ProviderUnavailableError("web_search", attempts)
    assert "web_search" in str(err)
    assert "brave" in str(err)


# ---------------------------------------------------------------------------
# H. Jina extraction — no API key required
# ---------------------------------------------------------------------------

def test_jina_extract_no_api_key_required(monkeypatch):
    """Jina must be attempted even when no API key is configured."""
    fake_page = WebPageResult(
        url="https://example.com",
        title="Example Domain",
        markdown="This domain is for use in illustrative examples in documents.",
        source="jina",
    )
    with mock.patch("tools.provider_router._jina_extract", return_value=fake_page) as jina_mock:
        result = ProviderRouter().extract("https://example.com", allow_reserve=False)

    jina_mock.assert_called_once()
    assert result.source == "jina"
    assert result.title == "Example Domain"


def test_jina_skipped_status_never_recorded_for_key_absence():
    """Jina requires_api_key=False in registry — router must not skip it for a missing key."""
    from tools.provider_registry import get_provider
    spec = get_provider("jina")
    assert spec.requires_api_key is False


def test_missing_key_recorded_as_skipped_not_failed(monkeypatch):
    """Skipped-due-to-missing-key attempt has status='skipped', not 'failed'."""
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")

    try:
        ProviderRouter().search("test", allow_reserve=False)
    except ProviderUnavailableError as exc:
        for attempt in exc.attempts:
            # All should be skipped (missing key), none failed (runtime error)
            assert attempt.status == "skipped", (
                f"{attempt.provider} has status={attempt.status!r}, expected 'skipped'"
            )


# ---------------------------------------------------------------------------
# I. research_url() uses free-first extraction before direct HTTP
# ---------------------------------------------------------------------------

def test_research_url_tries_router_first(monkeypatch):
    """research_url() must invoke the router (Jina path) before direct HTTP."""
    from tools.web_research import research_url

    fake_page = WebPageResult(
        url="https://example.com",
        title="Router Title",
        markdown="Content from router extraction path via jina.",
        source="jina",
    )
    with mock.patch("tools.provider_router._jina_extract", return_value=fake_page):
        result = research_url("https://example.com")

    assert result.source == "jina"
    assert result.title == "Router Title"


def test_research_url_falls_back_to_direct_http_when_all_providers_unavailable(monkeypatch):
    """When router raises ProviderUnavailableError, research_url() falls back to direct HTTP."""
    from tools.web_research import research_url
    from tools.provider_router import ProviderUnavailableError

    fake_page = WebPageResult(
        url="https://example.com",
        title="Direct Title",
        markdown="Direct HTTP content.",
        source="direct-http",
    )

    # Make Jina fail (no other extract providers w/o keys)
    with mock.patch(
        "tools.provider_router._jina_extract",
        side_effect=RuntimeError("jina down"),
    ):
        with mock.patch(
            "tools.web_research.requests.get"
        ) as http_mock:
            # Build a minimal fake HTTP response
            fake_resp = mock.MagicMock()
            fake_resp.url = "https://example.com"
            fake_resp.text = (
                "<html><head><title>Direct Title</title></head>"
                "<body><p>Direct HTTP content.</p></body></html>"
            )
            fake_resp.raise_for_status = mock.MagicMock()
            http_mock.return_value = fake_resp

            result = research_url("https://example.com")

    assert result.source == "direct-http"
    assert result.title == "Direct Title"


def test_research_url_does_not_call_firecrawl_without_reserve(monkeypatch):
    """Firecrawl extract must not be called from research_url (allow_reserve=False)."""
    from tools.web_research import research_url

    # Jina succeeds — Firecrawl should never be reached
    fake_page = WebPageResult(
        url="https://example.com",
        title="T",
        markdown="x" * 200,
        source="jina",
    )
    with mock.patch("tools.provider_router._jina_extract", return_value=fake_page):
        with mock.patch("tools.provider_router._firecrawl_extract") as fc_mock:
            research_url("https://example.com")
        fc_mock.assert_not_called()


# ---------------------------------------------------------------------------
# J. Manager -> Researcher -> ProviderRouter integration
# ---------------------------------------------------------------------------

def test_manager_routes_to_researcher():
    """Manager.route() for a research job returns 'researcher'."""
    from agents.manager import Manager
    manager = Manager()
    job = manager.create_job("Research Python async patterns")
    assert manager.route(job) == "researcher"
    assert job.research_mode == "normal"


def test_researcher_search_web_raises_on_all_keys_missing(monkeypatch):
    """Researcher.research() propagates ProviderUnavailableError when all keys absent."""
    from agents.researcher import research
    from tools.provider_router import ProviderUnavailableError

    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")

    with pytest.raises((ProviderUnavailableError, RuntimeError)):
        research("test query", limit=1)


def test_firecrawl_not_invoked_in_normal_search_path(monkeypatch):
    """End-to-end: researcher.research() with Jina succeeding must not touch Firecrawl."""
    from agents.researcher import research

    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")

    fake_results = [
        WebSearchResult(url="https://example.com", title="Ex", position=1),
    ]
    fake_page = WebPageResult(
        url="https://example.com",
        title="Example Domain",
        markdown=(
            "This domain is for use in illustrative examples in documents. "
            "You may use this domain in literature without prior coordination."
        ),
        source="jina",
    )

    with mock.patch("tools.provider_router._tavily_search", return_value=fake_results):
        with mock.patch("tools.provider_router._jina_extract", return_value=fake_page):
            with mock.patch("tools.provider_router._firecrawl_search") as fc_search:
                with mock.patch("tools.provider_router._firecrawl_extract") as fc_extract:
                    report = research("test query", limit=1)

    fc_search.assert_not_called()
    fc_extract.assert_not_called()
    assert report.source_count >= 1

# ---------------------------------------------------------------------------
# K. Deep-research orchestration
# ---------------------------------------------------------------------------

def test_deep_research_uses_free_first_orchestrator(monkeypatch):
    fake_result = mock.Mock(
        success=True,
        status="completed",
        model="test-model",
        credits_used=None,
        data={"provider_mode": "free-first"},
        error=None,
    )

    monkeypatch.setattr(
        "tools.deep_research.run_deep_research",
        lambda prompt, **kwargs: fake_result,
    )

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-fc-key")

    with mock.patch("tools.provider_router._firecrawl") as fc_mock:
        result = ProviderRouter().deep_research(
            "test deep research",
            allow_reserve=False,
        )

    assert result is fake_result
    fc_mock.assert_not_called()


def test_deep_research_uses_free_first_before_firecrawl_reserve(monkeypatch):
    free_first_result = mock.Mock(
        success=False,
        status="failed",
        model=None,
        credits_used=None,
        data={"provider_mode": "free-first"},
        error="No usable evidence was collected.",
    )

    reserve_result = mock.Mock(
        data={"result": "firecrawl"},
        success=True,
        status="completed",
        model="firecrawl-agent",
        credits_used=1,
        error=None,
    )

    monkeypatch.setattr(
        "tools.deep_research.run_deep_research",
        lambda prompt, **kwargs: free_first_result,
    )
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-fc-key")

    fake_app = mock.Mock()
    fake_app.agent.return_value = reserve_result

    with mock.patch(
        "tools.provider_router._firecrawl",
        return_value=fake_app,
    ) as fc_factory:
        result = ProviderRouter().deep_research(
            "test deep research",
            allow_reserve=True,
        )

    fc_factory.assert_called_once()
    fake_app.agent.assert_called_once_with(
        prompt="test deep research"
    )
    assert result is reserve_result


def test_deep_research_does_not_use_reserve_when_free_first_succeeds(monkeypatch):
    free_first_result = mock.Mock(
        success=True,
        status="completed",
        model="test-model",
        credits_used=None,
        data={"provider_mode": "free-first"},
        error=None,
    )

    monkeypatch.setattr(
        "tools.deep_research.run_deep_research",
        lambda prompt, **kwargs: free_first_result,
    )
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-fc-key")

    with mock.patch("tools.provider_router._firecrawl") as fc_mock:
        result = ProviderRouter().deep_research(
            "test deep research",
            allow_reserve=True,
        )

    assert result is free_first_result
    fc_mock.assert_not_called()

def test_deep_research_reuses_router_observability_context(
    monkeypatch,
    tmp_path,
):
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(
        tmp_path / "deep-observability.db"
    )
    router = ProviderRouter(
        observer=observer,
        run_id="deep-run-1",
    )

    captured = {}

    def fake_run_deep_research(prompt, *, router):
        captured["prompt"] = prompt
        captured["router"] = router
        return mock.Mock(
            success=True,
            status="completed",
            model="test-model",
            credits_used=None,
            data={"provider_mode": "free-first"},
            error=None,
        )

    monkeypatch.setattr(
        "tools.deep_research.run_deep_research",
        fake_run_deep_research,
    )

    result = router.deep_research("trace this")

    assert result.success is True
    assert captured["prompt"] == "trace this"
    assert captured["router"] is router
    assert captured["router"].observer is observer
    assert captured["router"].run_id == "deep-run-1"

def test_deep_research_nested_search_uses_same_observability_context(
    monkeypatch,
    tmp_path,
):
    from tools.provider_observability import ProviderObservability

    observer = ProviderObservability(
        tmp_path / "deep-e2e.db"
    )

    router = ProviderRouter(
        observer=observer,
        run_id="deep-run-1",
    )

    monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")

    fake_results = [
        WebSearchResult(
            url="https://example.com",
            title="Example",
            position=1,
        )
    ]

    fake_page = WebPageResult(
        url="https://example.com",
        title="Example",
        markdown=(
            "This is sufficiently long example content for the deep "
            "research observability integration test. It contains "
            "enough factual-looking material to pass the content "
            "quality threshold and exercise the nested provider path."
        ),
        source="jina",
    )

    with mock.patch(
        "tools.provider_router._brave_search",
        return_value=fake_results,
    ):
        with mock.patch(
            "tools.provider_router._jina_extract",
            return_value=fake_page,
        ):
            with mock.patch(
                "tools.deep_research.synthesize",
                return_value=mock.Mock(
                    query="deep observability test",
                    answer="ok",
                    key_findings=[],
                    citations=[],
                    caveats=[],
                ),
            ):
                with mock.patch(
                    "tools.deep_research.get_model",
                    return_value=mock.Mock(
                        name="test-model",
                    ),
                ):
                    result = router.deep_research(
                        "deep observability test"
                    )

    assert result.success is True

    events = observer.recent(
        run_id="deep-run-1",
        limit=10,
    )

    assert any(
        event.provider == "brave"
        and event.operation == "search"
        and event.status == "success"
        and event.run_id == "deep-run-1"
        for event in events
    )
