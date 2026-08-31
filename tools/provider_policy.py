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
class ProviderPolicyDecision:
    """
    Backward-compatible static provider-policy decision.

    Canonical provider capability/order metadata lives in
    tools.provider_registry. Runtime execution belongs to
    tools.provider_router; adaptive ranking belongs to
    tools.provider_decision.
    """

    capability: Capability
    provider: str
    fallbacks: tuple[str, ...]
    reserve: str | None
    reason: str


# Backward compatibility for existing callers that imported
# ProviderDecision from this module.
ProviderDecision = ProviderPolicyDecision


def provider_available(name: str) -> bool:
    """Return whether a registry-defined provider is configured."""
    provider = get_provider(name)

    if not provider.requires_api_key:
        return True

    if not provider.env_key:
        return False

    return bool(os.getenv(provider.env_key))


def choose_provider(
    capability: Capability,
) -> ProviderPolicyDecision:
    """
    Choose the highest-priority configured provider.

    This is the static compatibility policy. It intentionally does
    not perform runtime execution, health-based ranking, or automatic
    reserve-provider activation.
    """
    chain = [
        name
        for name in free_first_chain(capability)
        if provider_available(name)
    ]

    reserve = reserve_provider(capability)

    if not chain:
        raise RuntimeError(
            f"No normal provider available for capability: "
            f"{capability}"
        )

    return ProviderPolicyDecision(
        capability=capability,
        provider=chain[0],
        fallbacks=tuple(chain[1:]),
        reserve=reserve,
        reason=(
            "Selected the highest-priority available "
            "free-first provider. Reserve providers are "
            "not activated automatically."
        ),
    )


def research_plan() -> list[ProviderPolicyDecision]:
    return [
        choose_provider("web_search"),
        choose_provider("web_extract"),
        choose_provider("deep_research"),
        choose_provider("reasoning"),
    ]


def creative_plan() -> list[ProviderPolicyDecision]:
    return [
        choose_provider("image_generation"),
        choose_provider("image_editing"),
        choose_provider("video_generation"),
        choose_provider("video_editing"),
    ]
