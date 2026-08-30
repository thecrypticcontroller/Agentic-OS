from tools.evidence_engine import (
    build_evidence_report,
    content_fingerprint,
    deduplicate_evidence,
    format_for_model,
    make_evidence_item,
    normalize_content,
    normalize_url,
)


def test_normalize_url_removes_tracking():
    url = (
        "https://www.Example.com/page/"
        "?utm_source=test&id=42#section"
    )

    assert normalize_url(url) == (
        "https://example.com/page?id=42"
    )


def test_normalize_content_collapses_whitespace():
    assert normalize_content(
        "  Hello\n\n   world   "
    ) == "Hello world"


def test_content_fingerprint_is_stable():
    first = content_fingerprint(
        "Hello   world"
    )

    second = content_fingerprint(
        "Hello world"
    )

    assert first == second


def test_official_source_scores_high():
    item = make_evidence_item(
        title="Gemini API",
        url="https://ai.google.com/docs",
        content="A" * 800,
    )

    assert item.source_type == "official"
    assert item.quality_score >= 0.85


def test_academic_source_scores_high():
    item = make_evidence_item(
        title="Research paper",
        url="https://arxiv.org/abs/1234",
        content="A" * 800,
    )

    assert item.source_type == "academic"
    assert item.quality_score >= 0.85


def test_deduplicate_same_url():
    first = make_evidence_item(
        title="Example",
        url="https://example.com/a?utm_source=x",
        content="Unique content A",
    )

    second = make_evidence_item(
        title="Example copy",
        url="https://example.com/a",
        content="Different content B",
    )

    unique, duplicates = deduplicate_evidence(
        [first, second]
    )

    assert len(unique) == 1
    assert duplicates == 1


def test_deduplicate_same_content():
    first = make_evidence_item(
        title="Source A",
        url="https://example.com/a",
        content="Same evidence",
    )

    second = make_evidence_item(
        title="Source B",
        url="https://other.com/b",
        content="Same evidence",
    )

    unique, duplicates = deduplicate_evidence(
        [first, second]
    )

    assert len(unique) == 1
    assert duplicates == 1


def test_higher_quality_source_wins():
    official = make_evidence_item(
        title="Official",
        url="https://google.com/docs",
        content="Same evidence",
    )

    web = make_evidence_item(
        title="Mirror",
        url="https://randomsite.com/article",
        content="Same evidence",
    )

    unique, duplicates = deduplicate_evidence(
        [web, official]
    )

    assert len(unique) == 1
    assert unique[0].url == official.url
    assert duplicates == 1


def test_compression_limits_item_size():
    item = make_evidence_item(
        title="Long",
        url="https://example.com",
        content="A" * 5000,
    )

    report = build_evidence_report(
        [item],
        max_chars_per_item=100,
        max_total_chars=200,
    )

    assert report.final_count == 1
    assert len(
        report.items[0].content
    ) <= 103


def test_report_tracks_deduplication():
    first = make_evidence_item(
        title="A",
        url="https://example.com/a",
        content="Same",
    )

    second = make_evidence_item(
        title="B",
        url="https://example.com/b",
        content="Same",
    )

    report = build_evidence_report(
        [first, second]
    )

    assert report.original_count == 2
    assert report.duplicates_removed == 1
    assert report.final_count == 1


def test_format_for_model_contains_sources():
    item = make_evidence_item(
        title="Example",
        url="https://example.com",
        content="Important evidence",
    )

    formatted = format_for_model(
        [item]
    )

    assert "SOURCE 1" in formatted
    assert "Example" in formatted
    assert "Important evidence" in formatted
    assert "https://example.com" in formatted
