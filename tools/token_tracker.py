from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    task: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class TokenTracker:
    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> UsageRecord:
        record = UsageRecord(
            provider=provider,
            model=model,
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        self._records.append(record)

        return record

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(
            record.input_tokens
            for record in self._records
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            record.output_tokens
            for record in self._records
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.total_input_tokens
            + self.total_output_tokens
        )

    @property
    def total_estimated_cost_usd(self) -> float:
        return round(
            sum(
                record.estimated_cost_usd
                for record in self._records
            ),
            8,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "calls": len(self._records),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_estimated_cost_usd,
        }
