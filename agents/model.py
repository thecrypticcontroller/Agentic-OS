from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class Model(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class DeterministicModel:
    name: str = "deterministic"

    def generate(self, prompt: str) -> str:
        return (
            "No external LLM is configured. "
            "Use the structured research data directly."
        )


@dataclass
class GeminiModel:
    name: str = "gemini"

    def generate(self, prompt: str) -> str:
        from tools.gemini_client import GeminiClient

        result = GeminiClient().generate(prompt)
        return result.text


def get_model() -> Model:
    """Return the configured synthesis model.

    External LLM use is opt-in so local tests and free-first research
    never trigger an unexpected Gemini request or API charge.
    """
    enabled = os.getenv(
        "AGENT_OS_LLM_ENABLED",
        "false",
    ).strip().lower()

    if enabled in {"1", "true", "yes", "on"} and os.getenv("GEMINI_API_KEY"):
        return GeminiModel()

    return DeterministicModel()
