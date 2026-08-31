from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from agents.model import get_model
from agents.researcher import (
    ResearchResult,
    ResearchSource,
    normalize_query,
    research_url,
)
from agents.synthesizer import SynthesisResult, synthesize
from tools.evidence_engine import (
    EvidenceItem,
    build_evidence_report,
    make_evidence_item,
)
from tools.web_research import search_web


@dataclass(frozen=True)
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


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _query_variants(query: str) -> list[str]:
    base = normalize_query(query)
    candidates = [
        base,
        f"{base} official documentation",
        f"{base} independent analysis",
    ]

    seen: set[str] = set()
    variants: list[str] = []

    for candidate in candidates:
        normalized = candidate.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            variants.append(normalized)

    return variants


def _collect_evidence(
    query: str,
    *,
    router=None,
    queries: int = 3,
    results_per_query: int = 5,
    max_sources: int = 8,
    max_chars_per_item: int = 1400,
) -> tuple[list[EvidenceItem], list[str]]:
    evidence: list[EvidenceItem] = []
    providers: list[str] = []
    seen_urls: set[str] = set()

    for search_query in _query_variants(query)[:queries]:
        results = search_web(
            search_query,
            limit=results_per_query,
            router=router,
        )

        for result in results:
            url = result.url.strip()
            if not url:
                continue

            normalized_url = url.lower().rstrip("/")
            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            try:
                page = research_url(
                    url,
                    max_chars=max_chars_per_item,
                    router=router,
                )
            except Exception:
                continue

            item = make_evidence_item(
                title=page.title,
                url=page.url,
                content=page.summary,
            )
            evidence.append(item)

            if len(evidence) >= max_sources * 2:
                return evidence, providers

    return evidence, providers


def _research_result(
    query: str,
    evidence: list[EvidenceItem],
) -> ResearchResult:
    sources = [
        ResearchSource(
            title=item.title,
            url=item.url,
            domain=item.domain,
            content=item.content,
            rank=index,
        )
        for index, item in enumerate(evidence, start=1)
    ]

    return ResearchResult(
        query=query,
        source_count=len(sources),
        sources=sources,
        summary="\n\n".join(
            item.compact_text
            for item in evidence
        ),
    )


def _synthesis_data(
    synthesis: SynthesisResult,
    evidence_report,
    query: str,
    model_name: str | None,
) -> dict:
    return {
        "query": query,
        "executive_summary": synthesis.answer,
        "key_findings": synthesis.key_findings,
        "citations": [
            {
                "title": item.title,
                "url": item.url,
                "domain": item.domain,
            }
            for item in synthesis.citations
        ],
        "caveats": synthesis.caveats,
        "evidence": {
            "original_count": evidence_report.original_count,
            "final_count": evidence_report.final_count,
            "duplicates_removed": evidence_report.duplicates_removed,
            "compressed_characters": evidence_report.compressed_characters,
        },
        "provider_mode": "free-first",
        "reserve_provider_used": False,
        "model": model_name,
    }


def run_deep_research(
    prompt: str,
    *,
    router=None,
    queries: int = 3,
    results_per_query: int = 5,
    max_sources: int = 8,
    max_chars_per_item: int = 1400,
) -> DeepResearchResult:
    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Deep research prompt cannot be empty.")

    query = normalize_query(prompt)

    try:
        raw_evidence, _providers = _collect_evidence(
            query,
            router=router,
            queries=queries,
            results_per_query=results_per_query,
            max_sources=max_sources,
            max_chars_per_item=max_chars_per_item,
        )

        evidence_report = build_evidence_report(
            raw_evidence,
            max_chars_per_item=max_chars_per_item,
            max_total_chars=max_chars_per_item * max_sources,
        )

        if not evidence_report.items:
            return DeepResearchResult(
                prompt=prompt,
                success=False,
                status="no_evidence",
                model=None,
                credits_used=None,
                data={
                    "query": query,
                    "provider_mode": "free-first",
                    "reserve_provider_used": False,
                },
                error="No usable evidence was collected.",
            )

        report = _research_result(
            query,
            evidence_report.items,
        )
        synthesis = synthesize(report)
        model_name = getattr(
            get_model(),
            "name",
            None,
        )

        return DeepResearchResult(
            prompt=prompt,
            success=True,
            status="completed",
            model=model_name,
            credits_used=None,
            data=_synthesis_data(
                synthesis,
                evidence_report,
                query,
                model_name,
            ),
            error=None,
        )

    except Exception as exc:
        return DeepResearchResult(
            prompt=prompt,
            success=False,
            status="failed",
            model=None,
            credits_used=None,
            data={
                "query": query,
                "provider_mode": "free-first",
                "reserve_provider_used": False,
            },
            error=f"{type(exc).__name__}: {exc}",
        )
