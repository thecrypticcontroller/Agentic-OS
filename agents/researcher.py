from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.web_research import (
    agent_research,
    research_url as _research_url,
    scrape_web,
    search_web,
)


@dataclass
class ResearchSource:
    title: str
    url: str
    domain: str
    content: str
    rank: int


@dataclass
class ResearchResult:
    query: str
    source_count: int
    sources: list[ResearchSource]
    summary: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "source_count": self.source_count,
            "sources": [asdict(source) for source in self.sources],
            "summary": self.summary,
        }


@dataclass
class ResearchURLResult:
    url: str
    title: str
    summary: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeepResearchResult:
    prompt: str
    success: bool
    status: str
    model: str | None
    credits_used: int | float | None
    data: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def research_url(
    url: str,
    max_chars: int = 4000,
    *,
    router=None,
) -> ResearchURLResult:
    result = _research_url(
        url,
        max_chars=max_chars,
        router=router,
    )

    return ResearchURLResult(
        url=result.url,
        title=result.title,
        summary=result.markdown,
        source=result.source,
    )


def normalize_query(query: str) -> str:
    query = query.strip()

    query = re.sub(
        r"^(please\s+)?"
        r"(research|research\s+on|find|investigate|look\s+into)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )

    return query.strip()


def _domain(url: str) -> str:
    return (
        urlparse(url)
        .netloc
        .lower()
        .removeprefix("www.")
    )


def _source_key(url: str) -> str:
    parsed = urlparse(url.lower())

    return (
        f"{parsed.netloc}"
        f"{parsed.path.rstrip('/')}"
    )


def _clean_content(text: str) -> str:
    text = text.strip()

    # Remove image markdown.
    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        "",
        text,
    )

    # Remove image/file URLs.
    text = re.sub(
        r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|svg)(?:\?\S+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    noise = (
        r"Skip to content",
        r"Open more actions menu",
        r"Dismiss alert",
        r"\{\{ message \}\}",
        r"You must be signed in.*",
        r"You signed out.*",
        r"You switched accounts.*",
    )

    for pattern in noise:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(
        r"(?m)^#{1,6}\s*$",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def research(
    query: str,
    limit: int = 3,
    max_chars_per_page: int = 2500,
) -> ResearchResult:
    query = normalize_query(query)

    if not query:
        raise ValueError(
            "Research query cannot be empty."
        )

    search_results = search_web(
        query,
        limit=limit,
    )

    if not search_results:
        raise RuntimeError(
            "No web results returned."
        )

    sources: list[ResearchSource] = []
    seen_sources: set[str] = set()

    for result in search_results:
        url = result.url.strip()

        if not url:
            continue

        key = _source_key(url)

        if key in seen_sources:
            continue

        seen_sources.add(key)

        try:
            page = scrape_web(
                url,
                max_chars=max_chars_per_page,
            )

            content = _clean_content(
                page.markdown
            )

            if not content:
                continue

            sources.append(
                ResearchSource(
                    title=page.title,
                    url=page.url,
                    domain=_domain(page.url),
                    content=content,
                    rank=(
                        result.position
                        or len(sources) + 1
                    ),
                )
            )

        except Exception:
            continue

    if not sources:
        raise RuntimeError(
            "Search results were returned, "
            "but none could be scraped."
        )

    sources.sort(
        key=lambda item: item.rank
    )

    summary_parts = [
        "# Research Report",
        f"Query: {query}",
        f"Sources analyzed: {len(sources)}",
    ]

    for index, source in enumerate(
        sources,
        start=1,
    ):
        summary_parts.append(
            f"## Source {index}: {source.title}\n"
            f"Domain: {source.domain}\n"
            f"URL: {source.url}\n\n"
            f"{source.content}"
        )

    return ResearchResult(
        query=query,
        source_count=len(sources),
        sources=sources,
        summary="\n\n".join(summary_parts),
    )


def deep_research(
    prompt: str,
) -> DeepResearchResult:
    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "Deep research prompt cannot be empty."
        )

    response = agent_research(prompt)

    raw_data = getattr(
        response,
        "data",
        None,
    )

    if isinstance(raw_data, dict):
        data = raw_data
    else:
        data = {
            "result": raw_data
        }

    return DeepResearchResult(
        prompt=prompt,
        success=bool(
            getattr(response, "success", False)
        ),
        status=str(
            getattr(response, "status", "unknown")
        ),
        model=getattr(
            response,
            "model",
            None,
        ),
        credits_used=getattr(
            response,
            "credits_used",
            None,
        ),
        data=data,
        error=getattr(
            response,
            "error",
            None,
        ),
    )


if __name__ == "__main__":
    print("=== NORMAL RESEARCH ===")

    report = research(
        "Firecrawl capabilities for AI agents",
        limit=3,
    )

    print("Query:", report.query)
    print("Sources:", report.source_count)

    print()
    print("=== DEEP RESEARCH ===")

    deep = deep_research(
        "Find the main capabilities of Firecrawl "
        "for AI agents and summarize them."
    )

    print("Success:", deep.success)
    print("Status:", deep.status)
    print("Model:", deep.model)
    print("Credits:", deep.credits_used)
    print()

    print("Executive summary:")
    print(
        deep.data.get(
            "executive_summary",
            "No executive summary returned.",
        )
    )

    print()
    print("Capabilities:")

    for capability in deep.data.get(
        "capabilities",
        [],
    ):
        print(
            f"- {capability.get('name')}: "
            f"{capability.get('summary')}"
        )

    print()
    print("Selection guidance:")

    for key, value in deep.data.get(
        "selection_guidance",
        {},
    ).items():
        print(f"- {key}: {value}")
