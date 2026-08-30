from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskClass = Literal[
    "fast",
    "balanced",
    "reasoning",
]

Provider = Literal[
    "gemini",
]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    name: str
    task_class: TaskClass
    input_usd_per_million: float
    output_usd_per_million: float
    max_context_tokens: int


@dataclass(frozen=True)
class RouteDecision:
    provider: Provider
    model: str
    task_class: TaskClass
    reason: str


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        provider="gemini",
        name="gemini-3.1-flash-lite",
        task_class="fast",
        input_usd_per_million=0.25,
        output_usd_per_million=1.50,
        max_context_tokens=1_048_576,
    ),
    ModelSpec(
        provider="gemini",
        name="gemini-3.5-flash-lite",
        task_class="fast",
        input_usd_per_million=0.30,
        output_usd_per_million=2.50,
        max_context_tokens=1_048_576,
    ),
    ModelSpec(
        provider="gemini",
        name="gemini-3.7-flash",
        task_class="balanced",
        input_usd_per_million=0.75,
        output_usd_per_million=3.75,
        max_context_tokens=1_048_576,
    ),
    ModelSpec(
        provider="gemini",
        name="gemini-3.1-pro-preview",
        task_class="reasoning",
        input_usd_per_million=2.00,
        output_usd_per_million=12.00,
        max_context_tokens=1_048_576,
    ),
)


def _spec(task_class: TaskClass) -> ModelSpec:
    for model in MODELS:
        if model.task_class == task_class:
            return model

    raise ValueError(
        f"No model configured for task class: {task_class}"
    )


def classify_task(prompt: str) -> TaskClass:
    text = prompt.strip().lower()

    reasoning_markers = (
        "deep research",
        "comprehensive",
        "compare",
        "comparison",
        "competitive analysis",
        "architecture",
        "root cause",
        "evaluate",
        "critique",
        "strategy",
        "tradeoff",
        "multi-step",
        "hard problem",
        "final decision",
        "recommendation",
        "judge",
        "synthesize",
        "synthesis",
    )

    fast_markers = (
        "classify",
        "extract",
        "rewrite",
        "format",
        "summarize",
        "title",
        "short answer",
        "translate",
        "normalize",
        "deduplicate",
        "parse",
        "label",
        "clean",
        "filter",
        "collect",
        "list",
    )

    if any(
        marker in text
        for marker in reasoning_markers
    ):
        return "reasoning"

    if any(
        marker in text
        for marker in fast_markers
    ):
        return "fast"

    return "balanced"


def route_task(prompt: str) -> RouteDecision:
    task_class = classify_task(prompt)

    if task_class == "fast":
        model = _spec("fast")

        return RouteDecision(
            provider=model.provider,
            model=model.name,
            task_class=task_class,
            reason=(
                "Selected Gemini 3.1 Flash-Lite for "
                "high-volume low-cost execution."
            ),
        )

    if task_class == "reasoning":
        model = _spec("reasoning")

        return RouteDecision(
            provider=model.provider,
            model=model.name,
            task_class=task_class,
            reason=(
                "Escalated to Gemini 3.1 Pro for "
                "difficult reasoning or final decisions."
            ),
        )

    model = _spec("balanced")

    return RouteDecision(
        provider=model.provider,
        model=model.name,
        task_class="balanced",
        reason=(
            "Selected Gemini 3.7 Flash as the "
            "default agent workhorse."
        ),
    )


def estimate_cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            "Token counts cannot be negative."
        )

    model = next(
        (
            item
            for item in MODELS
            if item.name == model_name
        ),
        None,
    )

    if model is None:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    input_cost = (
        input_tokens
        / 1_000_000
        * model.input_usd_per_million
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * model.output_usd_per_million
    )

    return round(
        input_cost + output_cost,
        8,
    )
