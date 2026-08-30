from __future__ import annotations

from agents.synthesizer import SynthesisResult
from agents.researcher import ResearchURLResult
from tools.deep_research import run_deep_research
from tools.web_research import WebSearchResult


def test_deep_research_is_free_first_and_deduplicates(monkeypatch):
    calls: list[str] = []

    def fake_search(query: str, limit: int = 5):
        calls.append(query)
        return [
            WebSearchResult(
                url="https://example.com/article",
                title="Example Article",
                description="Example",
                position=1,
            ),
            WebSearchResult(
                url="https://example.org/report",
                title="Example Report",
                description="Report",
                position=2,
            ),
        ]

    def fake_research_url(url: str, max_chars: int = 4000):
        return ResearchURLResult(
            url=url,
            title="Source title",
            summary=(
                "This is sufficiently long evidence content for the deterministic "
                "deep research test. It provides useful facts and context for synthesis."
            ),
            source="direct-http",
        )

    def fake_synthesize(report):
        assert report.source_count == 2
        return SynthesisResult(
            query=report.query,
            answer="Synthesized from free-first evidence.",
            key_findings=["Finding one"],
            citations=[],
            caveats=[],
        )

    monkeypatch.setattr("tools.deep_research.search_web", fake_search)
    monkeypatch.setattr("tools.deep_research.research_url", fake_research_url)
    monkeypatch.setattr("tools.deep_research.synthesize", fake_synthesize)

    result = run_deep_research(
        "Deep research AI agent platforms",
        queries=2,
        results_per_query=5,
        max_sources=4,
    )

    assert result.success is True
    assert result.status == "completed"
    assert result.data["provider_mode"] == "free-first"
    assert result.data["reserve_provider_used"] is False
    assert result.data["evidence"]["final_count"] == 2
    assert len(calls) == 2


def test_deep_research_reports_no_evidence(monkeypatch):
    monkeypatch.setattr(
        "tools.deep_research.search_web",
        lambda query, limit=5: [],
    )

    result = run_deep_research(
        "Deep research unavailable topic"
    )

    assert result.success is False
    assert result.status == "no_evidence"
    assert result.error == "No usable evidence was collected."
    assert result.data["reserve_provider_used"] is False


def test_deep_research_rejects_empty_prompt():
    try:
        run_deep_research("  ")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "cannot be empty" in str(exc).lower()
