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
