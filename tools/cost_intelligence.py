from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.cost_control import CostController
from tools.provider_observability import ProviderObservability


@dataclass(frozen=True)
class ProviderCostSnapshot:
    provider: str
    calls: int
    total_tokens: int
    estimated_cost_usd: float
    avg_cost_per_call_usd: float
    avg_cost_per_1k_tokens_usd: float | None
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "calls": self.calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "avg_cost_per_call_usd": self.avg_cost_per_call_usd,
            "avg_cost_per_1k_tokens_usd": self.avg_cost_per_1k_tokens_usd,
            "state": self.state,
        }


@dataclass(frozen=True)
class RunCostSnapshot:
    run_id: str
    budget_usd: float | None
    spent_usd: float
    reserved_usd: float
    remaining_usd: float | None
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int
    usage_calls: int
    budget_utilization: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "budget_usd": self.budget_usd,
            "spent_usd": self.spent_usd,
            "reserved_usd": self.reserved_usd,
            "remaining_usd": self.remaining_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "usage_calls": self.usage_calls,
            "budget_utilization": self.budget_utilization,
        }


class CostIntelligence:
    """Expose transparent spend/quota measurements without changing routing."""

    def __init__(
        self,
        observability: ProviderObservability,
        cost_controller: CostController,
    ) -> None:
        self.observability = observability
        self.cost_controller = cost_controller

    def provider_snapshot(self, provider: str) -> ProviderCostSnapshot:
        summary = self.observability.summary(provider=provider)
        calls = int(summary["calls"])
        total_tokens = int(summary["total_tokens"])
        cost = float(summary["estimated_cost_usd"])

        avg_call = round(cost / calls, 8) if calls else 0.0
        per_1k = (
            round(cost / total_tokens * 1000.0, 8)
            if total_tokens
            else None
        )

        if calls == 0:
            state = "unknown"
        elif cost == 0.0:
            state = "free"
        elif per_1k is not None and per_1k <= 0.01:
            state = "low-cost"
        elif per_1k is not None and per_1k <= 0.10:
            state = "moderate-cost"
        else:
            state = "high-cost"

        return ProviderCostSnapshot(
            provider=provider,
            calls=calls,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 8),
            avg_cost_per_call_usd=avg_call,
            avg_cost_per_1k_tokens_usd=per_1k,
            state=state,
        )

    def run_snapshot(self, run_id: str) -> RunCostSnapshot:
        budget = self.cost_controller.get_budget(run_id)
        usage = self.cost_controller.usage_summary(run_id)

        budget_usd = budget.max_usd if budget else None
        spent_usd = budget.spent_usd if budget else float(usage["estimated_cost_usd"])
        reserved_usd = budget.reserved_usd if budget else 0.0
        remaining_usd = (
            max(0.0, budget_usd - spent_usd - reserved_usd)
            if budget_usd is not None
            else None
        )
        utilization = (
            round(spent_usd / budget_usd, 4)
            if budget_usd
            else None
        )

        return RunCostSnapshot(
            run_id=run_id,
            budget_usd=budget_usd,
            spent_usd=round(spent_usd, 8),
            reserved_usd=round(reserved_usd, 8),
            remaining_usd=(round(remaining_usd, 8) if remaining_usd is not None else None),
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            thinking_tokens=int(usage["thinking_tokens"]),
            total_tokens=int(usage["total_tokens"]),
            usage_calls=int(usage["calls"]),
            budget_utilization=utilization,
        )
