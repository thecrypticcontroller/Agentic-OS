from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ErrorClass = Literal[
    "retryable",
    "permanent",
    "unknown",
]


@dataclass(frozen=True)
class ClassifiedError:
    kind: ErrorClass
    reason: str


RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "temporary failure",
    "connection reset",
    "connection refused",
    "502",
    "503",
    "504",
    "rate limit",
    "429",
    "too many requests",
)

PERMANENT_MARKERS = (
    "invalid url",
    "url cannot be empty",
    "objective cannot be empty",
    "browser task requires a url",
    "authentication required",
    "permission denied",
    "not found",
    "404",
)


def classify_error(
    error: str | None,
) -> ClassifiedError:
    if not error:
        return ClassifiedError(
            kind="unknown",
            reason="No error message was supplied.",
        )

    text = error.lower()

    for marker in RETRYABLE_MARKERS:
        if marker in text:
            return ClassifiedError(
                kind="retryable",
                reason=f"Matched retryable marker: {marker}",
            )

    for marker in PERMANENT_MARKERS:
        if marker in text:
            return ClassifiedError(
                kind="permanent",
                reason=f"Matched permanent marker: {marker}",
            )

    return ClassifiedError(
        kind="unknown",
        reason="No known failure pattern matched.",
    )
