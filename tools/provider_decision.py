from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.provider_health import ProviderHealthService
from tools.provider_registry import ProviderSpec, providers_for
from tools.cost_intelligence import CostIntelligence


DecisionMode = Literal["health_cost", "static"]


@dataclass(frozen=True)
class ProviderDecision:
    provider: str
    capability: str
    score: float
    reason: str
    health_score: float | None
    cost_state: str
    priority: int
    reserve_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "capability": self.capability,
            "score": self.score,
            "reason": self.reason,
            "health_score": self.health_score,
            "cost_state": self.cost_state,
            "priority": self.priority,
            "reserve_only": self.reserve_only,
        }


class ProviderDecisionEngine:
    """Rank providers using observed health and cost without changing router policy."""

    def __init__(
        self,
        health_service: ProviderHealthService,
        cost_intelligence: CostIntelligence,
    ) -> None:
        self.health_service = health_service
        self.cost_intelligence = cost_intelligence

    def rank(
        self,
        capability: str,
        *,
        allow_reserve: bool = False,
        mode: DecisionMode = "health_cost",
    ) -> list[ProviderDecision]:
        specs = providers_for(capability, include_reserve=allow_reserve)
        decisions: list[ProviderDecision] = []

        for spec in specs:
            decisions.append(self._score(spec, capability, mode))

        return sorted(
            decisions,
            key=lambda item: (-item.score, item.priority, item.provider),
        )

    def choose(
        self,
        capability: str,
        *,
        allow_reserve: bool = False,
        mode: DecisionMode = "health_cost",
    ) -> ProviderDecision | None:
        ranked = self.rank(
            capability,
            allow_reserve=allow_reserve,
            mode=mode,
        )
        return ranked[0] if ranked else None

    def _score(
        self,
        spec: ProviderSpec,
        capability: str,
        mode: DecisionMode,
    ) -> ProviderDecision:
        health = self.health_service.snapshot(spec.name)
        cost = self.cost_intelligence.provider_snapshot(spec.name)

        if mode == "static":
            score = 100.0 - float(spec.priority)
            reason = f"static priority={spec.priority}"
        elif health.score is None:
            score = max(0.0, 50.0 - float(spec.priority))
            reason = "no observations; conservative baseline"
        else:
            cost_penalty = {
                "free": 0.0,
                "low-cost": 2.0,
                "moderate-cost": 8.0,
                "high-cost": 20.0,
                "unknown": 4.0,
            }.get(cost.state, 4.0)
            priority_bonus = max(0.0, 5.0 - float(spec.priority))
            score = max(
                0.0,
                round(
                    float(health.score) - cost_penalty + priority_bonus,
                    2,
                ),
            )
            reason = (
                f"health={health.score:.2f}, "
                f"cost_state={cost.state}, "
                f"priority={spec.priority}"
            )

        return ProviderDecision(
            provider=spec.name,
            capability=capability,
            score=score,
            reason=reason,
            health_score=health.score,
            cost_state=cost.state,
            priority=spec.priority,
            reserve_only=spec.reserve_only,
        )
