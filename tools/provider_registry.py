from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Capability = Literal[
    "reasoning",
    "web_search",
    "web_extract",
    "deep_research",
    "image_generation",
    "image_editing",
    "video_generation",
    "video_editing",
    "cloud_compute",
    "storage",
    "knowledge",
]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    capabilities: tuple[Capability, ...]
    free_first: bool
    priority: int
    reserve_only: bool = False
    requires_api_key: bool = False
    env_key: str | None = None
    notes: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="gemini",
        capabilities=(
            "reasoning",
            "image_generation",
            "image_editing",
        ),
        free_first=True,
        priority=1,
        requires_api_key=True,
        env_key="GEMINI_API_KEY",
        notes="Primary intelligence and multimodal provider.",
    ),
    ProviderSpec(
        name="brave",
        capabilities=("web_search",),
        free_first=True,
        priority=1,
        requires_api_key=True,
        env_key="BRAVE_API_KEY",
        notes="Primary independent web-search fallback.",
    ),
    ProviderSpec(
        name="jina",
        capabilities=("web_extract",),
        free_first=True,
        priority=1,
        requires_api_key=False,
        notes="Basic Reader mode is available without an API key.",
    ),
    ProviderSpec(
        name="tavily",
        capabilities=("web_search",),
        free_first=True,
        priority=2,
        requires_api_key=True,
        env_key="TAVILY_API_KEY",
        notes="Free-tier research provider; quota protected.",
    ),
    ProviderSpec(
        name="exa",
        capabilities=("web_search", "web_extract"),
        free_first=True,
        priority=3,
        requires_api_key=True,
        env_key="EXA_API_KEY",
        notes="Use signup/free credit selectively.",
    ),
    ProviderSpec(
        name="firecrawl",
        capabilities=("web_search", "web_extract", "deep_research"),
        free_first=True,
        priority=99,
        reserve_only=True,
        requires_api_key=True,
        env_key="FIRECRAWL_API_KEY",
        notes="Protected reserve. Do not use as the default provider.",
    ),
    ProviderSpec(
        name="comfyui",
        capabilities=(
            "image_generation",
            "image_editing",
            "video_generation",
        ),
        free_first=True,
        priority=1,
        requires_api_key=False,
        notes="Local-first creative engine.",
    ),
    ProviderSpec(
        name="kdenlive",
        capabilities=("video_editing",),
        free_first=True,
        priority=1,
        notes="Local video editor.",
    ),
    ProviderSpec(
        name="ffmpeg",
        capabilities=("video_editing",),
        free_first=True,
        priority=1,
        notes="Local media processing and automation.",
    ),
    ProviderSpec(
        name="huggingface",
        capabilities=("video_generation", "image_generation"),
        free_first=True,
        priority=2,
        requires_api_key=True,
        env_key="HF_TOKEN",
        notes="Use free/shared inference selectively.",
    ),
    ProviderSpec(
        name="cloudflare",
        capabilities=("cloud_compute", "storage"),
        free_first=True,
        priority=1,
        requires_api_key=True,
        env_key="CLOUDFLARE_API_TOKEN",
        notes="Cloud control plane and queue infrastructure.",
    ),
    ProviderSpec(
        name="supabase",
        capabilities=("storage", "cloud_compute"),
        free_first=True,
        priority=2,
        requires_api_key=True,
        env_key="SUPABASE_KEY",
        notes="Database/auth/storage layer.",
    ),
    ProviderSpec(
        name="vercel",
        capabilities=("cloud_compute",),
        free_first=True,
        priority=2,
        requires_api_key=True,
        env_key="VERCEL_TOKEN",
        notes="Frontend/API hosting where appropriate.",
    ),
    ProviderSpec(
        name="obsidian",
        capabilities=("knowledge",),
        free_first=True,
        priority=1,
        notes="Local knowledge vault on D:.",
    ),
)


def get_provider(name: str) -> ProviderSpec:
    name = name.strip().lower()

    for provider in PROVIDERS:
        if provider.name == name:
            return provider

    raise ValueError(
        f"Unknown provider: {name}"
    )


def providers_for(
    capability: Capability,
    *,
    include_reserve: bool = False,
) -> list[ProviderSpec]:
    providers = [
        provider
        for provider in PROVIDERS
        if capability in provider.capabilities
        and (
            include_reserve
            or not provider.reserve_only
        )
    ]

    return sorted(
        providers,
        key=lambda provider: provider.priority,
    )


def free_first_chain(
    capability: Capability,
) -> list[str]:
    return [
        provider.name
        for provider in providers_for(
            capability,
            include_reserve=False,
        )
    ]


def reserve_provider(
    capability: Capability,
) -> str | None:
    for provider in providers_for(
        capability,
        include_reserve=True,
    ):
        if provider.reserve_only:
            return provider.name

    return None
