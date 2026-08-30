from __future__ import annotations

import os
from dataclasses import dataclass

from tools.provider_registry import (
    Capability,
    free_first_chain,
    get_provider,
    reserve_provider,
)


@dataclass(frozen=True)
class ProviderDecision:
    capability: Capability
    provider: str
    fallbacks: tuple[str, ...]
    reserve: str | None
    reason: str


def provider_available(name: str) -> bool:
    provider = get_provider(name)

    if not provider.requires_api_key:
        return True

    if not provider.env_key:
        return False

    return bool(
        os.getenv(provider.env_key)
    )


def choose_provider(
    capability: Capability,
) -> ProviderDecision:
    chain = [
        name
        for name in free_first_chain(
            capability
        )
        if provider_available(name)
    ]

    reserve = reserve_provider(
        capability
    )

    if not chain:
        if reserve and provider_available(
            reserve
        ):
            return ProviderDecision(
                capability=capability,
                provider=reserve,
                fallbacks=(),
                reserve=reserve,
                reason=(
                    "No normal free-first provider is "
                    "available; using protected reserve."
                ),
            )

        raise RuntimeError(
            f"No available provider for capability: "
            f"{capability}"
        )

    return ProviderDecision(
        capability=capability,
        provider=chain[0],
        fallbacks=tuple(
            chain[1:]
        ),
        reserve=reserve,
        reason=(
            "Selected the highest-priority available "
            "free-first provider."
        ),
    )


def research_plan() -> list[ProviderDecision]:
    return [
        choose_provider("web_search"),
        choose_provider("web_extract"),
        choose_provider("deep_research"),
        choose_provider("reasoning"),
    ]


def creative_plan() -> list[ProviderDecision]:
    return [
        choose_provider("image_generation"),
        choose_provider("image_editing"),
        choose_provider("video_generation"),
        choose_provider("video_editing"),
    ]
