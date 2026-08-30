from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "ref",
    "referrer",
}


@dataclass(frozen=True)
class EvidenceItem:
    title: str
    url: str
    domain: str
    content: str
    source_type: str
    quality_score: float
    content_hash: str

    @property
    def compact_text(self) -> str:
        return (
            f"[{self.domain}] {self.title}\n"
            f"{self.content}"
        )


@dataclass(frozen=True)
class EvidenceReport:
    items: list[EvidenceItem]
    duplicates_removed: int
    original_count: int
    final_count: int
    compressed_characters: int


def normalize_url(url: str) -> str:
    url = url.strip()

    if not url:
        return ""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url.lower()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        if key.lower() not in TRACKING_PARAMS
    ]

    query = urlencode(
        sorted(
            query_pairs,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )
    )

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            query,
            "",
        )
    )


def normalize_content(content: str) -> str:
    text = content.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def content_fingerprint(content: str) -> str:
    normalized = normalize_content(
        content
    ).lower()

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def domain_from_url(url: str) -> str:
    try:
        domain = urlsplit(
            url
        ).netloc.lower()
    except ValueError:
        return ""

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def classify_source_type(
    url: str,
) -> str:
    domain = domain_from_url(url)

    if not domain:
        return "unknown"

    academic_markers = (
        ".edu",
        ".ac.",
        "arxiv.org",
        "pubmed",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
    )

    official_markers = (
        "openai.com",
        "anthropic.com",
        "google.com",
        "microsoft.com",
        "github.com",
        "aws.amazon.com",
        "cloud.google.com",
        "vercel.com",
    )

    news_markers = (
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "techcrunch.com",
    )

    if any(
        marker in domain
        for marker in academic_markers
    ):
        return "academic"

    if any(
        marker in domain
        for marker in official_markers
    ):
        return "official"

    if any(
        marker in domain
        for marker in news_markers
    ):
        return "news"

    return "web"


def source_quality_score(
    url: str,
    title: str = "",
    content: str = "",
) -> float:
    domain = domain_from_url(url)
    source_type = classify_source_type(url)

    score = 0.50

    if source_type == "official":
        score += 0.35

    elif source_type == "academic":
        score += 0.35

    elif source_type == "news":
        score += 0.20

    if domain.endswith(
        (
            ".gov",
            ".gov.uk",
            ".nic.in",
        )
    ):
        score += 0.15

    if domain.endswith(
        ".org"
    ):
        score += 0.05

    if title.strip():
        score += 0.03

    if len(
        normalize_content(content)
    ) >= 500:
        score += 0.05

    return round(
        min(score, 1.0),
        3,
    )


def make_evidence_item(
    *,
    title: str,
    url: str,
    content: str,
) -> EvidenceItem:
    normalized_url = normalize_url(
        url
    )

    normalized_content = normalize_content(
        content
    )

    domain = domain_from_url(
        normalized_url
    )

    return EvidenceItem(
        title=title.strip(),
        url=normalized_url,
        domain=domain,
        content=normalized_content,
        source_type=classify_source_type(
            normalized_url
        ),
        quality_score=source_quality_score(
            normalized_url,
            title,
            normalized_content,
        ),
        content_hash=content_fingerprint(
            normalized_content
        ),
    )


def deduplicate_evidence(
    items: list[EvidenceItem],
) -> tuple[list[EvidenceItem], int]:
    selected: list[EvidenceItem] = []

    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    duplicates = 0

    ranked = sorted(
        items,
        key=lambda item: (
            item.quality_score,
            len(item.content),
        ),
        reverse=True,
    )

    for item in ranked:
        if not item.url:
            duplicates += 1
            continue

        if item.url in seen_urls:
            duplicates += 1
            continue

        if item.content_hash in seen_hashes:
            duplicates += 1
            continue

        seen_urls.add(item.url)
        seen_hashes.add(
            item.content_hash
        )
        selected.append(item)

    selected.sort(
        key=lambda item: (
            item.quality_score,
            len(item.content),
        ),
        reverse=True,
    )

    return selected, duplicates


def compress_evidence(
    items: list[EvidenceItem],
    *,
    max_chars_per_item: int = 1200,
    max_total_chars: int = 12_000,
) -> list[EvidenceItem]:
    if max_chars_per_item < 1:
        raise ValueError(
            "max_chars_per_item must be at least 1."
        )

    if max_total_chars < 1:
        raise ValueError(
            "max_total_chars must be at least 1."
        )

    compressed: list[EvidenceItem] = []
    total = 0

    for item in items:
        content = item.content[:max_chars_per_item]

        if len(item.content) > max_chars_per_item:
            content = content.rstrip() + "..."

        projected = (
            total
            + len(content)
            + len(item.title)
            + len(item.domain)
            + 16
        )

        if projected > max_total_chars:
            break

        compressed.append(
            EvidenceItem(
                title=item.title,
                url=item.url,
                domain=item.domain,
                content=content,
                source_type=item.source_type,
                quality_score=item.quality_score,
                content_hash=item.content_hash,
            )
        )

        total = projected

    return compressed


def build_evidence_report(
    items: list[EvidenceItem],
    *,
    max_chars_per_item: int = 1200,
    max_total_chars: int = 12_000,
) -> EvidenceReport:
    original_count = len(items)

    unique_items, duplicates_removed = (
        deduplicate_evidence(items)
    )

    compressed = compress_evidence(
        unique_items,
        max_chars_per_item=max_chars_per_item,
        max_total_chars=max_total_chars,
    )

    compressed_characters = sum(
        len(item.content)
        for item in compressed
    )

    return EvidenceReport(
        items=compressed,
        duplicates_removed=duplicates_removed,
        original_count=original_count,
        final_count=len(compressed),
        compressed_characters=compressed_characters,
    )


def format_for_model(
    items: list[EvidenceItem],
) -> str:
    blocks: list[str] = []

    for index, item in enumerate(
        items,
        start=1,
    ):
        blocks.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"Title: {item.title}",
                    f"Domain: {item.domain}",
                    f"Type: {item.source_type}",
                    f"Quality: {item.quality_score}",
                    f"URL: {item.url}",
                    f"Evidence: {item.content}",
                ]
            )
        )

    return "\n\n".join(
        blocks
    )
