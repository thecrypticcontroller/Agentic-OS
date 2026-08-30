from tools.cost_control import CostController
from tools.cost_intelligence import CostIntelligence
from tools.provider_observability import ProviderObservability


def test_provider_cost_snapshot_classifies_free_usage(tmp_path):
    db_path = tmp_path / "cost.db"
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)
    intelligence = CostIntelligence(observer, controller)

    observer.record(
        provider="jina",
        operation="extract",
        status="success",
        latency_ms=100,
        input_tokens=100,
        output_tokens=0,
        thinking_tokens=0,
        total_tokens=100,
        estimated_cost_usd=0.0,
    )

    snapshot = intelligence.provider_snapshot("jina")

    assert snapshot.calls == 1
    assert snapshot.total_tokens == 100
    assert snapshot.estimated_cost_usd == 0.0
    assert snapshot.state == "free"
    assert snapshot.avg_cost_per_call_usd == 0.0
    assert snapshot.avg_cost_per_1k_tokens_usd == 0.0


def test_provider_cost_snapshot_computes_cost_density(tmp_path):
    db_path = tmp_path / "cost-density.db"
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)
    intelligence = CostIntelligence(observer, controller)

    observer.record(
        provider="gemini",
        operation="generate",
        status="success",
        latency_ms=300,
        input_tokens=900,
        output_tokens=100,
        thinking_tokens=0,
        total_tokens=1000,
        estimated_cost_usd=0.02,
    )

    snapshot = intelligence.provider_snapshot("gemini")

    assert snapshot.avg_cost_per_call_usd == 0.02
    assert snapshot.avg_cost_per_1k_tokens_usd == 0.02
    assert snapshot.state == "moderate-cost"


def test_run_cost_snapshot_reads_budget_and_usage(tmp_path):
    db_path = tmp_path / "run-cost.db"
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)
    intelligence = CostIntelligence(observer, controller)

    run_id = "cost-run"
    controller.create_budget(
        run_id,
        max_usd=1.0,
        max_input_tokens=10_000,
        max_output_tokens=2_000,
    )
    controller.record_usage(
        run_id,
        model="gemini",
        task="test",
        input_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
        estimated_cost_usd=0.25,
    )

    snapshot = intelligence.run_snapshot(run_id)

    assert snapshot.run_id == run_id
    assert snapshot.budget_usd == 1.0
    assert snapshot.spent_usd == 0.25
    assert snapshot.remaining_usd == 0.75
    assert snapshot.input_tokens == 100
    assert snapshot.output_tokens == 50
    assert snapshot.thinking_tokens == 10
    assert snapshot.total_tokens == 160
    assert snapshot.usage_calls == 1
    assert snapshot.budget_utilization == 0.25


def test_run_cost_snapshot_without_budget_is_observational(tmp_path):
    db_path = tmp_path / "no-budget.db"
    observer = ProviderObservability(db_path)
    controller = CostController(db_path)
    intelligence = CostIntelligence(observer, controller)

    snapshot = intelligence.run_snapshot("missing-budget")

    assert snapshot.budget_usd is None
    assert snapshot.remaining_usd is None
    assert snapshot.budget_utilization is None
    assert snapshot.spent_usd == 0.0
