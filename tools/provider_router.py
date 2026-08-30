"""
Provider Router — free-first, reserve-only Firecrawl.

Routing policy (from provider_registry):
  web_search:    brave -> tavily -> exa -> firecrawl(reserve)
  web_extract:   jina  -> exa   -> firecrawl(reserve)
  deep_research: tavily -> exa  -> firecrawl(reserve)

Missing-key skip: a provider that requires an API key not present
in the environment is skipped silently and recorded in ProviderAttempt.

Runtime fallback: if a provider raises during execution the router
catches the exception, records it, and continues to the next provider.

Reserve protection: Firecrawl (or any reserve_only provider) is NEVER
attempted unless allow_reserve=True is explicitly passed.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import requests

from tools.provider_registry import ProviderSpec, providers_for
from tools.web_research import (
    WebPageResult,
    WebSearchResult,
    _clean_markdown,
    _is_quality_content,
    _firecrawl,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ProviderAttempt:
    provider: str
    status: str          # "skipped" | "failed" | "success"
    reason: str | None   # human-readable, NEVER contains key values


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 15  # seconds


def _env(key: str) -> str | None:
    """Return env var value or None. Never leaks the value in exceptions."""
    return os.getenv(key) or None


def _brave_search(
    query: str,
    limit: int,
    api_key: str,
) -> list[WebSearchResult]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
        params={"q": query, "count": limit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    items = (data.get("web") or {}).get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("description"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _tavily_search(
    query: str,
    limit: int,
    api_key: str,
) -> list[WebSearchResult]:
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
    data = resp.json()
    items = data.get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("content"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _exa_search(
    query: str,
    limit: int,
    api_key: str,
) -> list[WebSearchResult]:
    resp = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        json={"query": query, "numResults": limit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results") or []
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("text") or item.get("snippet"),
            position=i + 1,
        )
        for i, item in enumerate(items[:limit])
    ]


def _firecrawl_search(
    query: str,
    limit: int,
) -> list[WebSearchResult]:
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


def _jina_extract(
    url: str,
    max_chars: int,
) -> WebPageResult:
    """Jina Reader — no API key required."""
    encoded = urllib.parse.quote(url, safe="")
    resp = requests.get(
        f"https://r.jina.ai/{encoded}",
        headers={"Accept": "text/plain"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.text or ""
    # Parse title from first line if formatted as "Title: ..."
    title = url
    lines = raw.splitlines()
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0][6:].strip() or url
    markdown = _clean_markdown(raw, max_chars)
    if not _is_quality_content(title, markdown):
        raise ProviderExecutionError(
            f"Jina returned low-quality content for {url}"
        )
    return WebPageResult(url=url, title=title, markdown=markdown, source="jina")


def _firecrawl_extract(
    url: str,
    max_chars: int,
) -> WebPageResult:
    """Delegate to low-level Firecrawl scrape (reserve-only).
    Calls _scrape_firecrawl directly to avoid routing back through scrape_web
    (which would re-enter the router with allow_reserve=False and never reach here)."""
    from tools.web_research import _scrape_firecrawl
    try:
        return _scrape_firecrawl(url, max_chars)
    except (RuntimeError, ValueError) as exc:
        raise ProviderExecutionError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Provider-level dispatch tables
# ---------------------------------------------------------------------------

# Maps provider name -> callable(query, limit, api_key?) -> results
_SEARCH_ADAPTERS: dict[str, Any] = {
    "brave": lambda q, n, key: _brave_search(q, n, key),
    "tavily": lambda q, n, key: _tavily_search(q, n, key),
    "exa": lambda q, n, key: _exa_search(q, n, key),
    "firecrawl": lambda q, n, _key: _firecrawl_search(q, n),
}

_EXTRACT_ADAPTERS: dict[str, Any] = {
    "jina": lambda url, chars, _key: _jina_extract(url, chars),
    "firecrawl": lambda url, chars, _key: _firecrawl_extract(url, chars),
    # exa extraction not implemented via direct REST; skip cleanly
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ProviderRouter:
    """
    Routes requests through the free-first provider chain.

    Firecrawl (reserve_only=True) is NEVER attempted unless
    allow_reserve=True is explicitly passed.
    """

    def _ordered_specs(
        self,
        capability: str,
        allow_reserve: bool,
    ) -> list[ProviderSpec]:
        return providers_for(capability, include_reserve=allow_reserve)

    def _key_for(self, spec: ProviderSpec) -> str | None:
        if not spec.requires_api_key:
            return ""          # sentinel: key not needed
        if spec.env_key is None:
            return None
        return _env(spec.env_key)  # None means missing

    def search(
        self,
        query: str,
        limit: int = 5,
        allow_reserve: bool = False,
    ) -> list[WebSearchResult]:
        attempts: list[ProviderAttempt] = []

        for spec in self._ordered_specs("web_search", allow_reserve):
            adapter = _SEARCH_ADAPTERS.get(spec.name)
            if adapter is None:
                # No adapter implemented; skip cleanly
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="skipped",
                    reason="no adapter implemented",
                ))
                continue

            key = self._key_for(spec)
            if key is None:
                # Missing env key — skip, do not attempt
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="skipped",
                    reason=f"{spec.env_key} is not configured",
                ))
                continue

            try:
                results = adapter(query, limit, key)
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="success",
                    reason=None,
                ))
                return results
            except Exception as exc:
                # Runtime failure — record and fall through to next provider
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="failed",
                    reason=_safe_reason(exc),
                ))

        raise ProviderUnavailableError("web_search", attempts)

    def extract(
        self,
        url: str,
        max_chars: int = 4000,
        allow_reserve: bool = False,
    ) -> WebPageResult:
        attempts: list[ProviderAttempt] = []

        for spec in self._ordered_specs("web_extract", allow_reserve):
            adapter = _EXTRACT_ADAPTERS.get(spec.name)
            if adapter is None:
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="skipped",
                    reason="no adapter implemented",
                ))
                continue

            key = self._key_for(spec)
            if key is None:
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="skipped",
                    reason=f"{spec.env_key} is not configured",
                ))
                continue

            try:
                result = adapter(url, max_chars, key)
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="success",
                    reason=None,
                ))
                return result
            except Exception as exc:
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="failed",
                    reason=_safe_reason(exc),
                ))

        raise ProviderUnavailableError("web_extract", attempts)

    def deep_research(
        self,
        prompt: str,
        allow_reserve: bool = False,
    ) -> Any:
        """
        Deep research routing.  Currently Firecrawl is the only adapter
        with full agent-research capability.  Free-first providers (tavily,
        exa) may support basic search; reserve is Firecrawl agent.
        """
        attempts: list[ProviderAttempt] = []

        for spec in self._ordered_specs("deep_research", allow_reserve):
            if spec.name == "firecrawl":
                # Firecrawl agent — reserve only
                key = self._key_for(spec)
                if key is None:
                    attempts.append(ProviderAttempt(
                        provider="firecrawl",
                        status="skipped",
                        reason="FIRECRAWL_API_KEY is not configured",
                    ))
                    continue
                try:
                    app = _firecrawl()
                    if app is None:
                        raise ProviderExecutionError("key present but client failed")
                    result = app.agent(prompt=prompt)
                    attempts.append(ProviderAttempt(
                        provider="firecrawl",
                        status="success",
                        reason=None,
                    ))
                    return result
                except Exception as exc:
                    attempts.append(ProviderAttempt(
                        provider="firecrawl",
                        status="failed",
                        reason=_safe_reason(exc),
                    ))
            else:
                # tavily/exa do not have a full agent endpoint; skip cleanly
                attempts.append(ProviderAttempt(
                    provider=spec.name,
                    status="skipped",
                    reason="no deep-research agent adapter implemented",
                ))

        raise ProviderUnavailableError("deep_research", attempts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_reason(exc: Exception) -> str:
    """
    Convert exception to a diagnostic string that NEVER leaks API key values.
    We redact anything that looks like a 32+ character alphanumeric token.
    """
    import re
    raw = str(exc)
    # Redact long hex/alphanum tokens (typical API key shapes)
    sanitized = re.sub(r"[A-Za-z0-9_\-]{32,}", "[REDACTED]", raw)
    return sanitized[:200]
