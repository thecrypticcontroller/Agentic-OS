from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


@dataclass
class WebSearchResult:
    url: str
    title: str
    description: str | None = None
    position: int | None = None


@dataclass
class WebPageResult:
    url: str
    title: str
    markdown: str
    source: str


def _firecrawl():
    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        return None

    from firecrawl import Firecrawl

    return Firecrawl(api_key=api_key)


def search_web(
    query: str,
    limit: int = 5,
    *,
    router=None,
) -> list[WebSearchResult]:
    """Free-first provider chain: brave -> tavily -> exa -> firecrawl(reserve).
    Firecrawl is reserve-only and not attempted here (allow_reserve=False)."""
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if router is None:
        from tools.provider_router import ProviderRouter
        router = ProviderRouter()

    return router.search(query, limit=limit, allow_reserve=False)


def _clean_markdown(
    markdown: str,
    max_chars: int,
) -> str:
    text = markdown or ""

    # Remove images.
    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        "",
        text,
    )

    # Remove standalone image/file URLs.
    text = re.sub(
        r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|svg)(?:\?\S+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove common browser/UI noise.
    noise_patterns = (
        r"Skip to content",
        r"Open more actions menu",
        r"Dismiss alert",
        r"\{\{ message \}\}",
        r"You must be signed in.*",
        r"You signed out.*",
        r"You switched accounts.*",
    )

    for pattern in noise_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    # Remove obviously corrupted/demo payloads.
    text = re.sub(
        r"(?s)```json.*?```",
        lambda match: (
            ""
            if any(
                marker in match.group(0)
                for marker in (
                    "A0-0",
                    "ha-Z",
                    "h?*ps",
                    "G=tZA",
                    "zZ:",
                    "?9a",
                )
            )
            else match.group(0)
        ),
        text,
    )

    # Remove lines made mostly from symbols/gibberish.
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        alnum = sum(char.isalnum() for char in stripped)

        if len(stripped) >= 12 and alnum / max(len(stripped), 1) < 0.35:
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Normalize whitespace.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()[:max_chars]


def _is_quality_content(
    title: str,
    markdown: str,
) -> bool:
    if not markdown:
        return False

    normalized = markdown.lower()

    # Too little useful content.
    if len(markdown) < 120:
        return False

    # Obvious scraper/UI failures.
    garbage_markers = (
        "uh oh!",
        "{{ message }}",
        "you must be signed in",
        "scraping...",
        "sign in to change notification settings",
    )

    if sum(
        marker in normalized
        for marker in garbage_markers
    ) >= 2:
        return False

    # Extremely low alphabetic density usually indicates corrupted output.
    letters = sum(char.isalpha() for char in markdown)

    if letters / max(len(markdown), 1) < 0.45:
        return False

    return bool(title.strip())


def _scrape_firecrawl(url: str, max_chars: int) -> WebPageResult:
    """Direct Firecrawl scrape — called only when explicitly allowed as reserve."""
    app = _firecrawl()

    if app is None:
        raise RuntimeError("FIRECRAWL_API_KEY is not configured.")

    result = app.scrape(url)
    metadata = getattr(result, "metadata", None)
    title = (
        getattr(metadata, "title", None) if metadata else None
    ) or url
    markdown = _clean_markdown(
        getattr(result, "markdown", None) or "",
        max_chars,
    )
    if not _is_quality_content(title, markdown):
        raise ValueError(f"Low-quality content returned for {url}")
    return WebPageResult(url=url, title=title, markdown=markdown, source="firecrawl")


def scrape_web(
    url: str,
    max_chars: int = 4000,
) -> WebPageResult:
    """Free-first extraction: jina -> exa -> firecrawl(reserve).
    Firecrawl is reserve-only and not attempted here (allow_reserve=False)."""
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    from tools.provider_router import ProviderRouter
    router = ProviderRouter()
    return router.extract(url, max_chars=max_chars, allow_reserve=False)


def research_url(
    url: str,
    max_chars: int = 4000,
    *,
    router=None,
) -> WebPageResult:
    """Extract URL content via free-first router; fall back to direct HTTP.
    Firecrawl is NOT automatically preferred even if its key exists."""
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    from tools.provider_router import ProviderRouter, ProviderUnavailableError

    if router is None:
        router = ProviderRouter()

    try:
        return router.extract(url, max_chars=max_chars, allow_reserve=False)
    except ProviderUnavailableError:
        pass  # fall through to direct HTTP

    # Local fallback: direct HTTP without any paid provider
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Agent-OS/0.1"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = (
        soup.title.get_text(" ", strip=True) if soup.title else url
    )
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(" ", strip=True)
    ]
    markdown = " ".join(paragraphs[:10])

    return WebPageResult(
        url=response.url,
        title=title,
        markdown=markdown[:max_chars],
        source="direct-http",
    )
def agent_research(
    prompt: str,
    allow_reserve: bool = False,
):
    """
    Autonomous web research via the deep_research router.
    Firecrawl agent is reserve-only; only attempted when allow_reserve=True.
    """
    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Agent research prompt cannot be empty.")

    from tools.provider_router import ProviderRouter
    router = ProviderRouter()
    return router.deep_research(prompt, allow_reserve=allow_reserve)
