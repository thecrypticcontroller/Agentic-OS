"""
Provider Router — free-first, reserve-only Firecrawl.

Routing policy (from provider_registry):
  web_search:    brave -> tavily -> exa -> firecrawl(reserve)
  web_extract:  jina -> exa -> firecrawl(reserve)
  deep_research: tavily -> exa -> firecrawl(reserve)

Missing-key skip: a provider that requires an API key not present
in the environment is skipped silently and recorded in ProviderAttempt.
Runtime fallback: if a provider raises during execution the router
catches the exception, records it, and continues to the next provider.

Adaptive routing: when AGENT_OS_ADAPTIVE_ROUTING is explicitly enabled,
observed provider health/cost rankings reorder only configured providers.
Reserve-only providers remain excluded unless allow_reserve=True.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import requests

from tools.cost_control import CostController
from tools.cost_intelligence import CostIntelligence
from tools.provider_decision import ProviderDecisionEngine
from tools.provider_health import ProviderHealthService
from tools.provider_observability import ProviderObservability
from tools.provider_registry import ProviderSpec, providers_for
from tools.web_research import (
    WebPageResult,
    WebSearchResult,
    _clean_markdown,
    _is_quality_content,
    _firecrawl,
)


class ProviderUnavailableError(RuntimeError):
    """All providers were skipped or failed; includes structured diagnostics."""

    def __init__(
        self,
        capability: str,
        attempts: list[ProviderAttempt],
    ) -> None:
        self.capability = capability
        self.attempts = attempts
        summary = "; ".join(
            f"{a.provider}={a.status}({a.reason})"
            for a in attempts
        )
        super().__init__(
            f"No provider available for '{capability}'. "
            f"Attempts: [{summary}]"
        )


class ProviderExecutionError(RuntimeError):
    """A provider was attempted but raised an exception."""


@dataclass
class ProviderAttempt:
    provider: str
    status: str
    reason: str | None


_TIMEOUT = 15


def _env(key: str) -> str | None:
    """Return env var value or None. Never leaks the value in exceptions."""
    return os.getenv(key) or None


def _brave_search(query: str, limit: int, api_key: str) -> list[WebSearchResult]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": query, "count": limit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    items = (resp.json().get("web") or {}).get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("description"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _tavily_search(query: str, limit: int, api_key: str) -> list[WebSearchResult]:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": limit,
            "include_answer": False,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("content"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _exa_search(query: str, limit: int, api_key: str) -> list[WebSearchResult]:
    resp = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        json={"query": query, "numResults": limit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("text") or item.get("snippet"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _firecrawl_search(query: str, limit: int) -> list[WebSearchResult]:
    """Delegate to existing Firecrawl integration (reserve-only)."""
    app = _firecrawl()
    if app is None:
        raise ProviderExecutionError("FIRECRAWL_API_KEY not configured")
    result = app.search(query, limit=limit)
    return [
        WebSearchResult(
            url=getattr(item, "url", ""),
            title=getattr(item, "title", "") or "",
            description=getattr(item, "description", None),
            position=getattr(item, "position", None),
        )
        for item in (getattr(result, "web", None) or [])
    ]


def _jina_extract(url: str, max_chars: int) -> WebPageResult:
    """Jina Reader — no API key required."""
    encoded = urllib.parse.quote(url, safe="")
    resp = requests.get(
        f"https://r.jina.ai/{encoded}",
        headers={"Accept": "text/plain"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.text or ""
    title = url
    lines = raw.splitlines()
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0][6:].strip() or url
    markdown = _clean_markdown(raw, max_chars)
    if not _is_quality_content(title, markdown):
        raise ProviderExecutionError(f"Jina returned low-quality content for {url}")
    return WebPageResult(url=url, title=title, markdown=markdown, source="jina")


def _firecrawl_extract(url: str, max_chars: int) -> WebPageResult:
    from tools.web_research import _scrape_firecrawl
    try:
        return _scrape_firecrawl(url, max_chars)
    except (RuntimeError, ValueError) as exc:
        raise ProviderExecutionError(str(exc)) from exc


_SEARCH_ADAPTERS: dict[str, Any] = {
    "brave": lambda q, n, key: _brave_search(q, n, key),
    "tavily": lambda q, n, key: _tavily_search(q, n, key),
    "exa": lambda q, n, key: _exa_search(q, n, key),
    "firecrawl": lambda q, n, _key: _firecrawl_search(q, n),
}

_EXTRACT_ADAPTERS: dict[str, Any] = {
    "jina": lambda url, chars, _key: _jina_extract(url, chars),
    "firecrawl": lambda url, chars, _key: _firecrawl_extract(url, chars),
}


class ProviderRouter:
    """Free-first router with optional health/cost-aware provider ordering."""

    def __init__(
        self,
        observer: ProviderObservability | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self.observer = observer or ProviderObservability()
        self.run_id = run_id
        self._adaptive_enabled = (
            os.getenv("AGENT_OS_ADAPTIVE_ROUTING", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )

    def _ordered_specs(self, capability: str, allow_reserve: bool) -> list[ProviderSpec]:
        specs = providers_for(capability, include_reserve=allow_reserve)
        if not self._adaptive_enabled or not specs:
            return specs

        try:
            health = ProviderHealthService(self.observer)
            cost = CostIntelligence(self.observer, CostController(self.observer.db_path))
            decisions = ProviderDecisionEngine(health, cost).rank(
                capability,
                allow_reserve=allow_reserve,
            )
            rank = {decision.provider: index for index, decision in enumerate(decisions)}

            configured = [spec for spec in specs if self._key_for(spec) is not None]
            unavailable = [spec for spec in specs if self._key_for(spec) is None]
            configured.sort(
                key=lambda spec: (
                    rank.get(spec.name, len(rank)),
                    spec.priority,
                    spec.name,
                )
            )
            return configured + unavailable
        except Exception:
            return specs

    def _key_for(self, spec: ProviderSpec) -> str | None:
        if not spec.requires_api_key:
            return ""
        if spec.env_key is None:
            return None
        return _env(spec.env_key)

    def _record(
        self,
        *,
        provider: str,
        operation: str,
        status: str,
        started: float,
        reason: str | None = None,
    ) -> None:
        try:
            self.observer.record(
                provider=provider,
                operation=operation,
                status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
                run_id=self.run_id,
                error_type="ProviderExecutionError" if status == "failed" else None,
                error=reason if status == "failed" else None,
            )
        except Exception:
            pass

    def search(self, query: str, limit: int = 5, allow_reserve: bool = False) -> list[WebSearchResult]:
        attempts: list[ProviderAttempt] = []
        for spec in self._ordered_specs("web_search", allow_reserve):
            started = time.perf_counter()
            adapter = _SEARCH_ADAPTERS.get(spec.name)
            if adapter is None:
                reason = "no adapter implemented"
                attempts.append(ProviderAttempt(spec.name, "skipped", reason))
                self._record(provider=spec.name, operation="search", status="skipped", started=started, reason=reason)
                continue
            key = self._key_for(spec)
            if key is None:
                reason = f"{spec.env_key} is not configured"
                attempts.append(ProviderAttempt(spec.name, "skipped", reason))
                self._record(provider=spec.name, operation="search", status="skipped", started=started)
                continue
            try:
                results = adapter(query, limit, key)
                attempts.append(ProviderAttempt(spec.name, "success", None))
                self._record(provider=spec.name, operation="search", status="success", started=started)
                return results
            except Exception as exc:
                reason = _safe_reason(exc)
                attempts.append(ProviderAttempt(spec.name, "failed", reason))
                self._record(provider=spec.name, operation="search", status="failed", started=started, reason=reason)
        raise ProviderUnavailableError("web_search", attempts)

    def extract(self, url: str, max_chars: int = 4000, allow_reserve: bool = False) -> WebPageResult:
        attempts: list[ProviderAttempt] = []
        for spec in self._ordered_specs("web_extract", allow_reserve):
            started = time.perf_counter()
            adapter = _EXTRACT_ADAPTERS.get(spec.name)
            if adapter is None:
                reason = "no adapter implemented"
                attempts.append(ProviderAttempt(spec.name, "skipped", reason))
                self._record(provider=spec.name, operation="extract", status="skipped", started=started, reason=reason)
                continue
            key = self._key_for(spec)
            if key is None:
                reason = f"{spec.env_key} is not configured"
                attempts.append(ProviderAttempt(spec.name, "skipped", reason))
                self._record(provider=spec.name, operation="extract", status="skipped", started=started)
                continue
            try:
                result = adapter(url, max_chars, key)
                attempts.append(ProviderAttempt(spec.name, "success", None))
                self._record(provider=spec.name, operation="extract", status="success", started=started)
                return result
            except Exception as exc:
                reason = _safe_reason(exc)
                attempts.append(ProviderAttempt(spec.name, "failed", reason))
                self._record(provider=spec.name, operation="extract", status="failed", started=started, reason=reason)
        raise ProviderUnavailableError("web_extract", attempts)

    def deep_research(self, prompt: str, allow_reserve: bool = False) -> Any:
        if not prompt.strip():
            raise ValueError("Deep research prompt cannot be empty.")

        from tools.deep_research import run_deep_research

        # The canonical normal path is the existing free-first deep-research
        # orchestration. It composes this router's search/extract capabilities.
        result = run_deep_research(
            prompt,
            router=self,
        )

        if result.success or not allow_reserve:
            return result

        # Reserve use is explicit: only after free-first orchestration fails.
        spec = next(
            (
                item
                for item in self._ordered_specs(
                    "deep_research",
                    allow_reserve=True,
                )
                if item.name == "firecrawl"
            ),
            None,
        )

        started = time.perf_counter()

        if spec is None:
            reason = "no reserve provider configured"
            self._record(
                provider="firecrawl",
                operation="deep_research",
                status="skipped",
                started=started,
                reason=reason,
            )
            raise ProviderUnavailableError(
                "deep_research",
                [ProviderAttempt("firecrawl", "skipped", reason)],
            )

        key = self._key_for(spec)

        if key is None:
            reason = "FIRECRAWL_API_KEY is not configured"
            self._record(
                provider="firecrawl",
                operation="deep_research",
                status="skipped",
                started=started,
                reason=reason,
            )
            raise ProviderUnavailableError(
                "deep_research",
                [ProviderAttempt("firecrawl", "skipped", reason)],
            )

        try:
            app = _firecrawl()

            if app is None:
                raise ProviderExecutionError(
                    "key present but client failed"
                )

            reserve_result = app.agent(prompt=prompt)

            self._record(
                provider="firecrawl",
                operation="deep_research",
                status="success",
                started=started,
            )
            return reserve_result

        except Exception as exc:
            reason = _safe_reason(exc)
            self._record(
                provider="firecrawl",
                operation="deep_research",
                status="failed",
                started=started,
                reason=reason,
            )
            raise ProviderUnavailableError(
                "deep_research",
                [ProviderAttempt("firecrawl", "failed", reason)],
            ) from exc


def _safe_reason(exc: Exception) -> str:
    """Convert an exception to a non-secret diagnostic string."""
    reason = str(exc)
    for key in (
        os.getenv("BRAVE_API_KEY"),
        os.getenv("TAVILY_API_KEY"),
        os.getenv("EXA_API_KEY"),
        os.getenv("FIRECRAWL_API_KEY"),
    ):
        if key:
            reason = reason.replace(key, "[REDACTED]")
    return reason[:200]
