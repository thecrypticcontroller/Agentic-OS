from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.model import get_model
from agents.researcher import ResearchResult


@dataclass
class Citation:
    title: str
    url: str
    domain: str


@dataclass
class SynthesisResult:
    query: str
    answer: str
    key_findings: list[str]
    citations: list[Citation]
    caveats: list[str]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "key_findings": self.key_findings,
            "citations": [asdict(item) for item in self.citations],
            "caveats": self.caveats,
        }


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if len(part.strip()) >= 40
    ]


def _extract_findings(
    report: ResearchResult,
    limit: int = 6,
) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    for source in report.sources:
        for sentence in _sentences(source.content):
            normalized = sentence.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            findings.append(sentence)

            if len(findings) >= limit:
                return findings

    return findings


def _build_prompt(report: ResearchResult) -> str:
    sections = []

    for source in report.sources:
        sections.append(
            f"TITLE: {source.title}\n"
            f"URL: {source.url}\n"
            f"CONTENT:\n{source.content}"
        )

    return (
        "You are a research synthesis agent.\n\n"
        f"QUESTION:\n{report.query}\n\n"
        "SOURCES:\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n\n"
        "Return a concise answer grounded only in these sources."
    )


def synthesize(report: ResearchResult) -> SynthesisResult:
    if not report.sources:
        raise ValueError(
            "Cannot synthesize an empty research report."
        )

    findings = _extract_findings(report)

    model = get_model()
    llm_available = False
    model_note = ""

    try:
        raw = model.generate(_build_prompt(report))
        # DeterministicModel returns a fixed "no LLM" string; treat as unavailable
        _no_llm_marker = "No external LLM is configured"
        if _no_llm_marker in raw:
            model_note = raw
        else:
            model_note = raw
            llm_available = True
    except Exception:
        model_note = "LLM synthesis failed."

    if findings:
        answer = (
            f"Research on '{report.query}' used "
            f"{report.source_count} source(s). "
            f"{model_note}"
        )
    else:
        findings = [
            f"Research returned {report.source_count} usable source(s)."
        ]
        answer = model_note

    citations = [
        Citation(
            title=source.title,
            url=source.url,
            domain=source.domain,
        )
        for source in report.sources
    ]

    caveats: list[str] = []
    if not llm_available:
        caveats.append(
            "LLM synthesis is not configured in this environment."
        )
    caveats.append(
        "Findings are extracted from the retrieved source content."
    )

    return SynthesisResult(
        query=report.query,
        answer=answer,
        key_findings=findings,
        citations=citations,
        caveats=caveats,
    )


if __name__ == "__main__":
    from agents.researcher import research

    report = research(
        "Research Firecrawl capabilities for AI agents",
        limit=3,
    )

    result = synthesize(report)

    print("=== SYNTHESIS ===")
    print("Query:", result.query)
    print()
    print("Answer:")
    print(result.answer)
    print()
    print("Key findings:")

    for finding in result.key_findings:
        print("-", finding)

    print()
    print("Citations:")

    for citation in result.citations:
        print("-", citation.title)
        print(" ", citation.url)

    print()
    print("Caveats:")

    for caveat in result.caveats:
        print("-", caveat)
