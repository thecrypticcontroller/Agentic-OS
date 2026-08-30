from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if self.base_delay_seconds < 0:
            raise ValueError(
                "base_delay_seconds cannot be negative."
            )

        if self.max_delay_seconds < 0:
            raise ValueError(
                "max_delay_seconds cannot be negative."
            )

    def can_retry(
        self,
        attempt: int,
    ) -> bool:
        return attempt < self.max_attempts

    def delay_for(
        self,
        attempt: int,
    ) -> float:
        if attempt < 1:
            raise ValueError(
                "attempt must be at least 1."
            )

        delay = self.base_delay_seconds * (
            2 ** (attempt - 1)
        )

        return min(
            delay,
            self.max_delay_seconds,
        )
