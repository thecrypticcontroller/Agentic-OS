from tools.model_router import (
    classify_task,
    estimate_cost_usd,
    route_task,
)
from tools.token_tracker import TokenTracker


def test_fast_task_uses_31_flash_lite():
    decision = route_task(
        "Extract the title from this webpage"
    )

    assert decision.provider == "gemini"
    assert decision.model == (
        "gemini-3.1-flash-lite"
    )
    assert decision.task_class == "fast"


def test_balanced_task_uses_37_flash():
    decision = route_task(
        "Research the pricing of this company"
    )

    assert decision.provider == "gemini"
    assert decision.model == (
        "gemini-3.7-flash"
    )
    assert decision.task_class == "balanced"


def test_deep_task_uses_reasoning_model():
    decision = route_task(
        "Deep research and compare the major AI "
        "web research platforms"
    )

    assert decision.provider == "gemini"
    assert decision.model == (
        "gemini-3.1-pro-preview"
    )
    assert decision.task_class == "reasoning"


def test_flash_lite_31_cost_estimation():
    cost = estimate_cost_usd(
        "gemini-3.1-flash-lite",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 1.75


def test_flash_lite_35_cost_estimation():
    cost = estimate_cost_usd(
        "gemini-3.5-flash-lite",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 2.80


def test_flash_cost_estimation():
    cost = estimate_cost_usd(
        "gemini-3.7-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 4.50


def test_pro_cost_estimation():
    cost = estimate_cost_usd(
        "gemini-3.1-pro-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 14.00


def test_negative_tokens_rejected():
    try:
        estimate_cost_usd(
            "gemini-3.7-flash",
            input_tokens=-1,
            output_tokens=10,
        )
        assert False
    except ValueError:
        pass


def test_token_tracker():
    tracker = TokenTracker()

    tracker.record(
        provider="gemini",
        model="gemini-3.1-flash-lite",
        task="summary",
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.001,
    )

    tracker.record(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        task="deep research",
        input_tokens=200,
        output_tokens=100,
        estimated_cost_usd=0.002,
    )

    snapshot = tracker.snapshot()

    assert snapshot["calls"] == 2
    assert snapshot["input_tokens"] == 300
    assert snapshot["output_tokens"] == 150
    assert snapshot["total_tokens"] == 450
    assert snapshot["estimated_cost_usd"] == 0.003
