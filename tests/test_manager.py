from agents.manager import Manager
from tools.run_registry import RunRecord


def test_normal_research_mode():
    manager = Manager()

    assert (
        manager.research_mode(
            "Research Firecrawl capabilities"
        )
        == "normal"
    )


def test_deep_research_mode():
    manager = Manager()

    assert (
        manager.research_mode(
            "Compare all major AI web research platforms "
            "comprehensively"
        )
        == "deep"
    )


def test_browser_routing():
    manager = Manager()

    job = manager.create_job(
        "Open https://example.com and inspect the page"
    )

    assert manager.route(job) == "browser_worker"
    assert job.research_mode is None


def test_research_routing():
    manager = Manager()

    job = manager.create_job(
        "Research Firecrawl capabilities"
    )

    assert manager.route(job) == "researcher"
    assert job.research_mode == "normal"


def test_deep_research_routing():
    manager = Manager()

    job = manager.create_job(
        "Investigate the AI agent ecosystem "
        "and compare the major platforms"
    )

    assert manager.route(job) == "researcher"
    assert job.research_mode == "deep"


def test_research_precedence_over_browser_keyword():
    manager = Manager()

    job = manager.create_job(
        "Investigate browser automation platforms "
        "and compare their capabilities"
    )

    assert manager.route(job) == "researcher"
    assert job.research_mode == "deep"


def test_plain_web_question_is_research():
    manager = Manager()

    job = manager.create_job(
        "What are the major AI web research platforms?"
    )

    assert manager.route(job) == "researcher"
    assert job.research_mode == "normal"

def test_deep_research_execution(monkeypatch):
    from agents.researcher import DeepResearchResult

    manager = Manager()

    fake_result = DeepResearchResult(
        prompt="Deep research AI agent platforms",
        success=True,
        status="completed",
        model="spark-2",
        credits_used=0,
        data={
            "title": "AI agent platforms",
            "summary": "Comparison completed.",
            "capabilities": [],
            "sources_consulted": [],
        },
        error=None,
    )

    monkeypatch.setattr(
        "agents.manager.deep_research",
        lambda prompt: fake_result,
    )

    job = manager.create_job(
        "Deep research and compare AI agent platforms"
    )

    result = manager.execute(job)

    assert result.status == "completed"
    assert result.worker == "researcher"
    assert result.research_mode == "deep"

    payload = __import__("json").loads(result.result)

    assert payload["type"] == "deep_research"
    assert payload["success"] is True
    assert payload["model"] == "spark-2"
    assert payload["data"]["title"] == "AI agent platforms"


def test_normal_research_execution_is_separate(monkeypatch):
    from agents.researcher import ResearchResult, ResearchSource

    manager = Manager()

    fake_report = ResearchResult(
        query="AI agent platforms",
        source_count=1,
        sources=[
            ResearchSource(
                title="Example Source",
                url="https://example.com",
                domain="example.com",
                content=(
                    "AI agent platforms provide tools for "
                    "building automated workflows."
                ),
                rank=1,
            )
        ],
        summary="Research report.",
    )

    monkeypatch.setattr(
        "agents.manager.research",
        lambda objective: fake_report,
    )

    class FakeAnswer:
        query = "AI agent platforms"
        answer = "Test answer"
        key_findings = ["Finding one"]
        citations = []
        caveats = []

    monkeypatch.setattr(
        "agents.manager.synthesize",
        lambda report: FakeAnswer(),
    )

    job = manager.create_job(
        "Research AI agent platforms"
    )

    result = manager.execute(job)

    assert result.status == "completed"
    assert result.worker == "researcher"
    assert result.research_mode == "normal"

    payload = __import__("json").loads(result.result)

    assert payload["type"] == "research"
    assert payload["query"] == "AI agent platforms"
    assert payload["answer"] == "Test answer"

def test_plan_browser():
    manager = Manager()

    plan = manager.plan(
        "Open https://example.com and inspect the page"
    )

    assert plan.worker == "browser_worker"
    assert plan.research_mode is None
    assert plan.target_url == "https://example.com"
    assert plan.tool == "browser_harness"


def test_plan_normal_research():
    manager = Manager()

    plan = manager.plan(
        "Research Firecrawl capabilities"
    )

    assert plan.worker == "researcher"
    assert plan.research_mode == "normal"
    assert plan.target_url is None
    assert plan.tool == "firecrawl.search+scrape"


def test_plan_deep_research():
    manager = Manager()

    plan = manager.plan(
        "Deep research and compare AI agent platforms"
    )

    assert plan.worker == "researcher"
    assert plan.research_mode == "deep"
    assert plan.target_url is None
    assert plan.tool == "free-first-deep-research"

def test_deep_research_with_url_stays_deep():
    manager = Manager()

    job = manager.create_job(
        "Deep research https://example.com and analyze the site"
    )

    assert manager.route(job) == "researcher"
    assert job.research_mode == "deep"
    assert manager.extract_url(job.objective) == "https://example.com"

def test_retry_creates_new_attempt(monkeypatch, tmp_path):
    from tools.run_registry import RunRegistry

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    failed = manager.create_job(
        "Research something"
    )

    failed.status = "failed"
    failed.error = "temporary failure"
    failed.started_at = "2026-08-29T00:00:00+00:00"
    failed.completed_at = "2026-08-29T00:00:01+00:00"

    registry.save(
        RunRecord(
            run_id=failed.id,
            parent_run_id=None,
            attempt=1,
            objective=failed.objective,
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="failed",
            started_at=failed.started_at,
            completed_at=failed.completed_at,
            duration_ms=100,
            result=None,
            error=failed.error,
        )
    )

    retried = manager.retry(
        failed.id
    )

    assert retried.id != failed.id
    assert retried.parent_run_id == failed.id
    assert retried.attempt == 2
    assert retried.objective == failed.objective
def test_retry_rejects_permanent_failure(tmp_path):
    from tools.run_registry import RunRecord, RunRegistry

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    job = manager.create_job(
        "Research something"
    )

    registry.save(
        RunRecord(
            run_id=job.id,
            objective=job.objective,
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="failed",
            started_at=None,
            completed_at=None,
            duration_ms=10,
            result=None,
            error="ValueError: invalid url",
            parent_run_id=None,
            attempt=1,
        )
    )

    try:
        manager.retry(job.id)
        assert False, "Expected permanent failure rejection"
    except ValueError as exc:
        assert "permanent" in str(exc).lower()


def test_retry_rejects_exhausted_attempts(tmp_path):
    from tools.run_registry import RunRecord, RunRegistry
    from tools.retry_policy import RetryPolicy

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    job = manager.create_job(
        "Research something"
    )

    registry.save(
        RunRecord(
            run_id=job.id,
            objective=job.objective,
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="failed",
            started_at=None,
            completed_at=None,
            duration_ms=10,
            result=None,
            error="TimeoutError: request timed out",
            parent_run_id=None,
            attempt=3,
        )
    )

    try:
        manager.retry(
            job.id,
            policy=RetryPolicy(max_attempts=3),
        )
        assert False, "Expected retry exhaustion"
    except ValueError as exc:
        assert "exhausted" in str(exc).lower()


def test_retry_allows_retryable_failure(tmp_path, monkeypatch):
    from agents.manager import Job
    from tools.run_registry import RunRecord, RunRegistry
    from tools.retry_policy import RetryPolicy

    registry = RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    original = manager.create_job(
        "Research something"
    )

    registry.save(
        RunRecord(
            run_id=original.id,
            objective=original.objective,
            worker="researcher",
            research_mode="normal",
            target_url=None,
            tool="firecrawl.search+scrape",
            status="failed",
            started_at=None,
            completed_at=None,
            duration_ms=10,
            result=None,
            error="TimeoutError: request timed out",
            parent_run_id=None,
            attempt=1,
        )
    )

    def fake_execute(job: Job) -> Job:
        job.status = "completed"
        return job

    monkeypatch.setattr(
        manager,
        "execute",
        fake_execute,
    )

    retried = manager.retry(
        original.id,
        policy=RetryPolicy(max_attempts=3),
    )

    assert retried.id != original.id
    assert retried.parent_run_id == original.id
    assert retried.attempt == 2
