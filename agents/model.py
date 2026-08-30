from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Model(Protocol):
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


def get_model() -> Model:
    # Future providers can be added here without changing
    # the researcher or manager.
    return DeterministicModel()
