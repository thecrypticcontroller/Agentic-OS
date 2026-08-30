from tools.cost_control import (
    BudgetExceededError,
    CostController,
)


def test_create_budget(tmp_path):
    controller = CostController(
        tmp_path / "cost.db"
    )

    budget = controller.create_budget(
        "run-1",
        max_usd=2.0,
        max_input_tokens=1000,
        max_output_tokens=500,
    )

    assert budget.run_id == "run-1"
    assert budget.max_usd == 2.0
    assert budget.max_input_tokens == 1000
    assert budget.max_output_tokens == 500


def test_reserve_budget(tmp_path):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
    )

    budget = controller.reserve(
        "run-1",
        0.25,
    )

    assert budget.reserved_usd == 0.25
    assert budget.spent_usd == 0.0


def test_release_reservation(tmp_path):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
    )

    controller.reserve(
        "run-1",
        0.25,
    )

    assert controller.release_reservation(
        "run-1",
        0.10,
    ) is True

    budget = controller.get_budget(
        "run-1"
    )

    assert budget is not None
    assert budget.reserved_usd == 0.15


def test_reserve_rejects_excess(tmp_path):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
    )

    try:
        controller.reserve(
            "run-1",
            1.01,
        )
        assert False
    except BudgetExceededError:
        pass


def test_record_usage_settles_reservation(
    tmp_path,
):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
        max_input_tokens=10_000,
        max_output_tokens=5_000,
    )

    controller.reserve(
        "run-1",
        0.10,
    )

    event = controller.record_usage(
        "run-1",
        model="gemini-3.7-flash",
        task="research",
        input_tokens=1000,
        output_tokens=500,
        thinking_tokens=200,
        estimated_cost_usd=0.04,
        reservation_usd=0.10,
    )

    assert event.run_id == "run-1"
    assert event.input_tokens == 1000
    assert event.output_tokens == 500
    assert event.thinking_tokens == 200
    assert event.estimated_cost_usd == 0.04

    budget = controller.get_budget(
        "run-1"
    )

    assert budget is not None
    assert budget.reserved_usd == 0.0
    assert budget.spent_usd == 0.04

    summary = controller.usage_summary(
        "run-1"
    )

    assert summary["calls"] == 1
    assert summary["input_tokens"] == 1000
    assert summary["output_tokens"] == 500
    assert summary["thinking_tokens"] == 200
    assert summary["total_tokens"] == 1700
    assert summary["estimated_cost_usd"] == 0.04


def test_output_token_budget_is_enforced(
    tmp_path,
):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=10.0,
        max_input_tokens=10_000,
        max_output_tokens=100,
    )

    try:
        controller.record_usage(
            "run-1",
            model="gemini-3.5-flash-lite",
            task="bulk",
            input_tokens=10,
            output_tokens=101,
            thinking_tokens=0,
            estimated_cost_usd=0.01,
        )
        assert False
    except BudgetExceededError:
        pass


def test_existing_budget_is_reused(tmp_path):
    controller = CostController(
        tmp_path / "cost.db"
    )

    first = controller.create_budget(
        "run-1",
        max_usd=2.0,
    )

    second = controller.create_budget(
        "run-1",
        max_usd=99.0,
    )

    assert first.budget_id == second.budget_id
    assert second.max_usd == 2.0


def test_multiple_reservations_are_isolated(
    tmp_path,
):
    controller = CostController(
        tmp_path / "cost.db"
    )

    controller.create_budget(
        "run-1",
        max_usd=1.0,
    )

    controller.reserve(
        "run-1",
        0.30,
    )

    controller.reserve(
        "run-1",
        0.20,
    )

    budget = controller.get_budget(
        "run-1"
    )

    assert budget is not None
    assert budget.reserved_usd == 0.50
