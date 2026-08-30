from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeUsage:
    prompt_token_count: int = 100
    candidates_token_count: int = 40
    thoughts_token_count: int = 10
    total_token_count: int = 150


@dataclass
class FakeResponse:
    text: str = "fake response"
    usage_metadata: FakeUsage = field(
        default_factory=FakeUsage
    )


class FakeModels:
    def generate_content(
        self,
        *,
        model,
        contents,
        config,
    ):
        return FakeResponse(
            text=f"{model}: {contents}"
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_client_records_usage(monkeypatch):
    from tools.gemini_client import GeminiClient

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    client = GeminiClient()
    client.client = FakeClient()

    result = client.generate(
        "Extract the title from a page"
    )

    assert result.model == (
        "gemini-3.1-flash-lite"
    )
    assert result.input_tokens == 100
    assert result.output_tokens == 40
    assert result.thinking_tokens == 10
    assert result.total_tokens == 150
    assert result.estimated_cost_usd > 0
    assert result.text.startswith(
        "gemini-3.1-flash-lite:"
    )


def test_parallel_runner_handles_multiple_tasks():
    from tools.gemini_client import GenerationResult
    from tools.parallel_agents import run_parallel

    class FakeGemini:
        def generate(
            self,
            prompt,
            **kwargs,
        ):
            return GenerationResult(
                prompt=prompt,
                model="gemini-3.1-flash-lite",
                text=f"done: {prompt}",
                input_tokens=10,
                output_tokens=5,
                thinking_tokens=0,
                total_tokens=15,
                estimated_cost_usd=0.0001,
            )

    result = run_parallel(
        [
            "Extract title A",
            "Extract title B",
            "Extract title C",
        ],
        max_workers=3,
        client=FakeGemini(),
    )

    assert result.successful == 3
    assert result.failed == 0
    assert result.skipped_count == 0
    assert len(result.results) == 3


def test_parallel_runner_rejects_empty_prompts():
    from tools.parallel_agents import run_parallel

    try:
        run_parallel([])
        assert False
    except ValueError:
        pass


def test_budgeted_fanout_skips_work_when_budget_is_small():
    from tools.parallel_agents import (
        plan_budgeted_fanout,
    )

    accepted, skipped = plan_budgeted_fanout(
        [
            "Extract title A",
            "Extract title B",
            "Extract title C",
        ],
        budget_usd=0.0004,
        max_output_tokens=128,
    )

    assert accepted
    assert skipped
    assert len(accepted) < 3


def test_budgeted_fanout_can_fit_single_task():
    from tools.parallel_agents import (
        plan_budgeted_fanout,
    )

    accepted, skipped = plan_budgeted_fanout(
        [
            "Extract title A",
        ],
        budget_usd=0.001,
        max_output_tokens=128,
    )

    assert len(accepted) == 1
    assert skipped == []


def test_gemini_client_respects_budget(
    monkeypatch,
    tmp_path,
):
    from tools.cost_control import (
        BudgetExceededError,
        CostController,
    )
    from tools.gemini_client import GeminiClient

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    controller = CostController(
        tmp_path / "budget.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=0.000001,
        max_input_tokens=100_000,
        max_output_tokens=100_000,
    )

    client = GeminiClient(
        cost_controller=controller,
        run_id="run-1",
    )

    client.client = FakeClient()

    try:
        client.generate(
            "This request should exceed the tiny budget.",
            max_output_tokens=2048,
        )
        assert False, "Expected budget rejection"
    except BudgetExceededError:
        pass


def test_gemini_client_settles_reservation(
    monkeypatch,
    tmp_path,
):
    from tools.cost_control import (
        CostController,
    )
    from tools.gemini_client import GeminiClient

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    controller = CostController(
        tmp_path / "budget.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
        max_input_tokens=100_000,
        max_output_tokens=100_000,
    )

    client = GeminiClient(
        cost_controller=controller,
        run_id="run-1",
    )

    client.client = FakeClient()

    result = client.generate(
        "Extract the title.",
        max_output_tokens=100,
    )

    budget = controller.get_budget(
        "run-1"
    )

    assert budget is not None
    assert budget.reserved_usd == 0.0
    assert budget.spent_usd == (
        result.estimated_cost_usd
    )

    summary = controller.usage_summary(
        "run-1"
    )

    assert summary["calls"] == 1
    assert summary["total_tokens"] == 150
