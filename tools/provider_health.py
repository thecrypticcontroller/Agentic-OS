from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.provider_observability import ProviderObservability


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    score: float | None
    state: str
    success_rate: float
    avg_latency_ms: float
    calls: int
    failures: int
    skipped: int
    recent_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "score": self.score,
            "state": self.state,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "calls": self.calls,
            "failures": self.failures,
            "skipped": self.skipped,
            "recent_failures": self.recent_failures,
        }


class ProviderHealthService:
    """Turn observed provider telemetry into a transparent health snapshot.

    Health is descriptive only. Routing policy remains unchanged until a
    separate policy layer explicitly consumes these measurements.
    """

    def __init__(
        self,
        observability: ProviderObservability,
        *,
        latency_target_ms: float = 1000.0,
        latency_ceiling_ms: float = 5000.0,
        recent_limit: int = 20,
    ) -> None:
        if latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        if latency_ceiling_ms <= latency_target_ms:
            raise ValueError(
                "latency_ceiling_ms must exceed latency_target_ms"
            )
        if recent_limit < 1:
            raise ValueError("recent_limit must be at least 1")

        self.observability = observability
        self.latency_target_ms = latency_target_ms
        self.latency_ceiling_ms = latency_ceiling_ms
        self.recent_limit = recent_limit

    def snapshot(self, provider: str) -> ProviderHealth:
        summary = self.observability.summary(provider=provider)
        calls = int(summary["calls"])
        failures = int(summary["failures"])
        skipped = int(summary["skipped"])
        success_rate = float(summary["success_rate"])
        avg_latency_ms = float(summary["avg_latency_ms"])

        recent = self.observability.recent(
            limit=self.recent_limit,
            provider=provider,
        )
        recent_failures = sum(
            event.status == "failed"
            for event in recent
        )

        if calls == 0:
            score = None
            state = "unknown"
        else:
            latency_factor = self._latency_factor(avg_latency_ms)
            score = round(
                (success_rate * 70.0)
                + (latency_factor * 30.0),
                2,
            )
            state = self._state_for_score(score)

        return ProviderHealth(
            provider=provider,
            score=score,
            state=state,
            success_rate=success_rate,
            avg_latency_ms=avg_latency_ms,
            calls=calls,
            failures=failures,
            skipped=skipped,
            recent_failures=recent_failures,
        )

    def _latency_factor(self, latency_ms: float) -> float:
        if latency_ms <= self.latency_target_ms:
            return 1.0
        if latency_ms >= self.latency_ceiling_ms:
            return 0.0

        span = self.latency_ceiling_ms - self.latency_target_ms
        return 1.0 - (
            (latency_ms - self.latency_target_ms) / span
        )

    @staticmethod
    def _state_for_score(score: float) -> str:
        if score >= 85.0:
            return "healthy"
        if score >= 65.0:
            return "degraded"
        return "unhealthy"
