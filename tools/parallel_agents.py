from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass

from tools.cost_control import (
    CostController,
)
from tools.gemini_client import (
    GenerationResult,
    GeminiClient,
)
from tools.model_router import (
    estimate_cost_usd,
    route_task,
)


@dataclass(frozen=True)
class PlannedTask:
    prompt: str
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_reservation_usd: float


@dataclass(frozen=True)
class FanoutResult:
    results: list[GenerationResult]
    errors: list[str]
    skipped: list[str]

    @property
    def successful(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.errors)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def total_tokens(self) -> int:
        return sum(
            result.total_tokens
            for result in self.results
        )

    @property
    def estimated_cost_usd(self) -> float:
        return round(
            sum(
                result.estimated_cost_usd
                for result in self.results
            ),
            8,
        )


def _estimate_input_tokens(
    prompt: str,
) -> int:
    return max(
        1,
        (len(prompt) + 3) // 4,
    )


def _plan_task(
    prompt: str,
    *,
    max_output_tokens: int,
) -> PlannedTask:
    decision = route_task(prompt)

    input_tokens = _estimate_input_tokens(
        prompt
    )

    reservation = estimate_cost_usd(
        decision.model,
        input_tokens,
        max_output_tokens,
    )

    # Same safety margin used by GeminiClient.
    reservation *= 1.25

    reservation = max(
        reservation,
        0.000001,
    )

    return PlannedTask(
        prompt=prompt,
        model=decision.model,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=max_output_tokens,
        estimated_reservation_usd=reservation,
    )


def plan_budgeted_fanout(
    prompts: list[str],
    *,
    budget_usd: float,
    max_output_tokens: int = 512,
) -> tuple[list[PlannedTask], list[str]]:
    if budget_usd <= 0:
        raise ValueError(
            "budget_usd must be greater than zero."
        )

    normalized = [
        prompt.strip()
        for prompt in prompts
        if prompt and prompt.strip()
    ]

    if not normalized:
        raise ValueError(
            "At least one prompt is required."
        )

    accepted: list[PlannedTask] = []
    skipped: list[str] = []
    remaining = budget_usd

    for prompt in normalized:
        task = _plan_task(
            prompt,
            max_output_tokens=max_output_tokens,
        )

        if (
            task.estimated_reservation_usd
            <= remaining
        ):
            accepted.append(task)
            remaining -= (
                task.estimated_reservation_usd
            )
        else:
            skipped.append(prompt)

    return accepted, skipped


def run_parallel(
    prompts: list[str],
    *,
    max_workers: int = 4,
    client: GeminiClient | None = None,
    run_id: str | None = None,
    cost_controller: CostController | None = None,
    budget_usd: float | None = None,
    max_output_tokens: int = 512,
) -> FanoutResult:
    prompts = [
        prompt.strip()
        for prompt in prompts
        if prompt and prompt.strip()
    ]

    if not prompts:
        raise ValueError(
            "At least one prompt is required."
        )

    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    skipped: list[str] = []

    if budget_usd is not None:
        planned, skipped = plan_budgeted_fanout(
            prompts,
            budget_usd=budget_usd,
            max_output_tokens=max_output_tokens,
        )

        prompts = [
            item.prompt
            for item in planned
        ]

    if not prompts:
        return FanoutResult(
            results=[],
            errors=[],
            skipped=skipped,
        )

    worker_count = min(
        max_workers,
        len(prompts),
    )

    gemini = (
        client
        or GeminiClient(
            cost_controller=cost_controller,
            run_id=run_id,
        )
    )

    results: list[GenerationResult] = []
    errors: list[str] = []

    def execute(
        prompt: str,
    ) -> GenerationResult:
        return gemini.generate(
            prompt,
            max_output_tokens=max_output_tokens,
            run_id=run_id,
        )

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="agent-os-gemini",
    ) as executor:
        futures = {
            executor.submit(
                execute,
                prompt,
            ): prompt
            for prompt in prompts
        }

        for future in as_completed(futures):
            prompt = futures[future]

            try:
                results.append(
                    future.result()
                )
            except Exception as exc:
                errors.append(
                    f"{type(exc).__name__}: "
                    f"{exc} | prompt={prompt}"
                )

    return FanoutResult(
        results=results,
        errors=errors,
        skipped=skipped,
    )
