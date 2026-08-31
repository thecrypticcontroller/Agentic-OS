from tools.provider_policy import (
    choose_provider,
    provider_available,
)
from tools.provider_registry import (
    free_first_chain,
    get_provider,
    reserve_provider,
)


def test_jina_is_free_first_without_key():
    assert provider_available("jina") is True


def test_firecrawl_is_not_in_normal_chain():
    chain = free_first_chain(
        "web_search"
    )

    assert "firecrawl" not in chain


def test_firecrawl_is_reserve_for_web_search():
    assert (
        reserve_provider(
            "web_search"
        )
        == "firecrawl"
    )


def test_provider_metadata():
    provider = get_provider(
        "firecrawl"
    )

    assert provider.reserve_only is True
    assert provider.env_key == (
        "FIRECRAWL_API_KEY"
    )


def test_image_chain_is_local_first():
    chain = free_first_chain(
        "image_generation"
    )

    assert chain[0] in (
        "comfyui",
        "gemini",
    )


def test_video_editing_prefers_local_tools():
    chain = free_first_chain(
        "video_editing"
    )

    assert chain
    assert "kdenlive" in chain
    assert "ffmpeg" in chain


def test_choose_jina_keeps_firecrawl_as_reserve():
    decision = choose_provider(
        "web_extract"
    )

    assert decision.provider == "jina"
    assert decision.reserve == "firecrawl"

def test_policy_reserve_is_metadata_only_when_normal_provider_missing(monkeypatch):
    from tools.provider_policy import choose_provider

    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("EXA_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")

    try:
        choose_provider("web_search")
        assert False, "Expected normal-provider failure"
    except RuntimeError as exc:
        assert "No normal provider available" in str(exc)


def test_policy_decision_is_compatibility_type():
    from tools.provider_policy import (
        ProviderDecision,
        ProviderPolicyDecision,
    )

    assert ProviderDecision is ProviderPolicyDecision

def test_research_plan_contains_only_provider_capabilities(monkeypatch):
    from tools.provider_policy import research_plan

    monkeypatch.setattr(
        "tools.provider_policy.provider_available",
        lambda name: True,
    )

    capabilities = [
        decision.capability
        for decision in research_plan()
    ]

    assert capabilities == [
        "web_search",
        "web_extract",
        "reasoning",
    ]
