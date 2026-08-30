from __future__ import annotations

import os
from dataclasses import dataclass

import google.genai as genai
from google.genai import types

from tools.cost_control import (
    CostController,
)
from tools.model_router import (
    estimate_cost_usd,
    route_task,
)


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cost_controller: CostController | None = None,
        run_id: str | None = None,
    ) -> None:
        key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=key
        )

        self.cost_controller = (
            cost_controller
        )
        self.run_id = run_id

    @staticmethod
    def _estimate_input_tokens(
        prompt: str,
    ) -> int:
        # Conservative local estimate used only
        # for reservation. Actual billing is taken
        # from Gemini usage metadata afterward.
        return max(
            1,
            (len(prompt) + 3) // 4,
        )

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int = 2048,
        run_id: str | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be at least 1."
            )

        decision = route_task(prompt)

        selected_model = (
            model
            if model is not None
            else decision.model
        )

        effective_run_id = (
            run_id
            or self.run_id
        )

        reservation_usd = 0.0

        if (
            self.cost_controller is not None
            and effective_run_id is not None
        ):
            estimated_input_tokens = (
                self._estimate_input_tokens(
                    prompt
                )
            )

            reservation_usd = estimate_cost_usd(
                selected_model,
                estimated_input_tokens,
                max_output_tokens,
            )

            # Keep a tiny safety margin for
            # model-side thinking tokens and
            # estimation error.
            reservation_usd *= 1.25

            reservation_usd = max(
                reservation_usd,
                0.000001,
            )

            self.cost_controller.reserve(
                effective_run_id,
                reservation_usd,
            )

        try:
            response = (
                self.client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=(
                            max_output_tokens
                        ),
                    ),
                )
            )

        except Exception:
            if (
                self.cost_controller is not None
                and effective_run_id is not None
                and reservation_usd > 0
            ):
                self.cost_controller.release_reservation(
                    effective_run_id,
                    reservation_usd,
                )

            raise

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        input_tokens = int(
            getattr(
                usage,
                "prompt_token_count",
                0,
            )
            or 0
        )

        output_tokens = int(
            getattr(
                usage,
                "candidates_token_count",
                0,
            )
            or 0
        )

        thinking_tokens = int(
            getattr(
                usage,
                "thoughts_token_count",
                0,
            )
            or 0
        )

        billed_output_tokens = (
            output_tokens
            + thinking_tokens
        )

        total_tokens = int(
            getattr(
                usage,
                "total_token_count",
                input_tokens
                + billed_output_tokens,
            )
            or (
                input_tokens
                + billed_output_tokens
            )
        )

        text = (
            response.text
            or ""
        )

        actual_cost = estimate_cost_usd(
            selected_model,
            input_tokens,
            billed_output_tokens,
        )

        if (
            self.cost_controller is not None
            and effective_run_id is not None
        ):
            self.cost_controller.record_usage(
                effective_run_id,
                model=selected_model,
                task=prompt[:120],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                estimated_cost_usd=actual_cost,
                reservation_usd=reservation_usd,
            )

        return GenerationResult(
            prompt=prompt,
            model=selected_model,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=actual_cost,
        )


def generate(
    prompt: str,
    *,
    model: str | None = None,
    max_output_tokens: int = 2048,
    run_id: str | None = None,
    cost_controller: CostController | None = None,
) -> GenerationResult:
    client = GeminiClient(
        cost_controller=cost_controller,
        run_id=run_id,
    )

    return client.generate(
        prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        run_id=run_id,
    )
