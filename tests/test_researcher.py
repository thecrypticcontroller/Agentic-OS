from agents.researcher import normalize_query
from tools.web_research import _is_quality_content


def test_normalize_query():
    assert (
        normalize_query(
            "Research Firecrawl capabilities for AI agents"
        )
        == "Firecrawl capabilities for AI agents"
    )


def test_normalize_query_plain_query():
    assert normalize_query(
        "latest AI agent frameworks"
    ) == "latest AI agent frameworks"


def test_normalize_query_please_research():
    assert normalize_query(
        "Please research autonomous AI agents"
    ) == "autonomous AI agents"


def test_normalize_query_find():
    assert normalize_query(
        "Find modern Python agent frameworks"
    ) == "modern Python agent frameworks"


def test_quality_gate_accepts_real_content():
    content = (
        "Firecrawl provides search and scraping capabilities for "
        "applications that need reliable, structured web information "
        "for AI agents and automated research workflows."
    )

    assert _is_quality_content(
        "Firecrawl",
        content,
    )


def test_quality_gate_rejects_empty_content():
    assert not _is_quality_content(
        "Example",
        "",
    )


def test_quality_gate_rejects_garbage():
    content = (
        "Uh oh! {{ message }} "
        "You must be signed in "
        "You must be signed in"
    )

    assert not _is_quality_content(
        "GitHub",
        content,
    )

def test_synthesis_result_contract():
    from agents.researcher import ResearchResult, ResearchSource
    from agents.synthesizer import synthesize

    report = ResearchResult(
        query="test query",
        source_count=1,
        sources=[
            ResearchSource(
                title="Test Source",
                url="https://example.com",
                domain="example.com",
                content=(
                    "This is a sufficiently long factual sentence "
                    "that should be selected as a research finding."
                ),
                rank=1,
            )
        ],
        summary="test",
    )

    result = synthesize(report)

    assert result.query == "test query"
    assert result.citations[0].url == "https://example.com"
    assert result.key_findings
    assert result.caveats

def test_deep_research_contract(monkeypatch):
    from agents.researcher import deep_research

    class FakeResponse:
        success = True
        status = "completed"
        model = "test-model"
        credits_used = 0
        error = None

        data = {
            "title": "Test research",
            "executive_summary": "Test summary",
            "capabilities": [],
        }

    def fake_agent(prompt):
        assert prompt == "test prompt"
        return FakeResponse()

    monkeypatch.setattr(
        "agents.researcher.agent_research",
        fake_agent,
    )

    result = deep_research("test prompt")

    assert result.success is True
    assert result.status == "completed"
    assert result.model == "test-model"
    assert result.data["title"] == "Test research"
