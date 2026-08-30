from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from agents.browser_worker import open_and_inspect
from agents.researcher import (
    research,
    research_url,
)
from agents.synthesizer import synthesize
from tools.deep_research import run_deep_research
from tools.error_classifier import classify_error
from tools.retry_policy import RetryPolicy
from tools.run_registry import RunRecord, RunRegistry


# Compatibility seam retained for the existing manager tests and callers.
deep_research = run_deep_research


WorkerName = Literal["researcher", "browser_worker"]
ResearchMode = Literal["normal", "deep"]
JobStatus = Literal["queued", "running", "completed", "failed"]

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+",
    re.IGNORECASE,
)

BROWSER_TERMS = (
    "browser",
    "click",
    "form",
    "navigate",
    "open",
    "inspect page",
)

DEEP_RESEARCH_TERMS = (
    "deep research",
    "comprehensive",
    "comprehensively",
    "investigate",
    "in-depth",
    "in depth",
    "thorough",
    "all major",
    "all the",
    "compare",
    "comparison",
    "competitive analysis",
    "landscape",
    "across multiple",
    "multiple sources",
    "find all",
    "identify all",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contains_term(text: str, term: str) -> bool:
    if " " in term:
        return term in text

    return re.search(
        rf"\b{re.escape(term)}\b",
        text,
        re.IGNORECASE,
    ) is not None


@dataclass
class Job:
    id: str
    objective: str
    worker: WorkerName | None = None
    research_mode: ResearchMode | None = None
    target_url: str | None = None
    status: JobStatus = "queued"
    started_at: str | None = None
    completed_at: str | None = None
    result: str | None = None
    error: str | None = None
    parent_run_id: str | None = None
    attempt: int = 1


@dataclass
class ExecutionPlan:
    worker: WorkerName
    research_mode: ResearchMode | None
    target_url: str | None
    tool: str


class Manager:
    def __init__(
        self,
        registry: RunRegistry | None = None,
    ) -> None:
        self.name = "Agent OS Manager"
        self.registry = registry or RunRegistry(
            Path("agent_os.db")
        )

    def create_job(
        self,
        objective: str,
        *,
        parent_run_id: str | None = None,
        attempt: int = 1,
    ) -> Job:
        objective = objective.strip()

        if not objective:
            raise ValueError(
                "Objective cannot be empty."
            )

        if attempt < 1:
            raise ValueError(
                "attempt must be at least 1."
            )

        return Job(
            id=str(uuid.uuid4()),
            objective=objective,
            parent_run_id=parent_run_id,
            attempt=attempt,
        )

    def research_mode(
        self,
        objective: str,
    ) -> ResearchMode:
        text = objective.lower()

        if any(
            contains_term(text, term)
            for term in DEEP_RESEARCH_TERMS
        ):
            return "deep"

        return "normal"

    def route(
        self,
        job: Job,
    ) -> WorkerName:
        text = job.objective.lower()

        research_mode = self.research_mode(
            job.objective
        )

        explicit_browser_intent = any(
            contains_term(text, term)
            for term in BROWSER_TERMS
        )

        # Deep research takes precedence over
        # generic browser-related words.
        if research_mode == "deep":
            worker: WorkerName = "researcher"
        elif explicit_browser_intent:
            worker = "browser_worker"
        else:
            worker = "researcher"

        job.worker = worker

        if worker == "researcher":
            job.research_mode = research_mode
        else:
            job.research_mode = None

        return worker

    def extract_url(
        self,
        objective: str,
    ) -> str | None:
        match = URL_PATTERN.search(objective)

        if match:
            return match.group(0).rstrip(
                ".,);\""
            )

        return None

    def plan(
        self,
        objective: str,
    ) -> ExecutionPlan:
        job = self.create_job(objective)

        worker = self.route(job)

        target_url = self.extract_url(
            objective
        )

        if worker == "browser_worker":
            tool = "browser_harness"
        elif job.research_mode == "deep":
            tool = "free-first-deep-research"
        elif target_url:
            tool = "firecrawl.scrape"
        else:
            tool = "firecrawl.search+scrape"

        return ExecutionPlan(
            worker=worker,
            research_mode=job.research_mode,
            target_url=target_url,
            tool=tool,
        )

    def queue(
        self,
        job: Job,
    ) -> Job:
        plan = self.plan(
            job.objective
        )

        job.worker = plan.worker
        job.research_mode = plan.research_mode
        job.target_url = plan.target_url
        job.status = "queued"

        self.registry.save(
            RunRecord(
                run_id=job.id,
                parent_run_id=job.parent_run_id,
                attempt=job.attempt,
                objective=job.objective,
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=job.target_url,
                tool=plan.tool,
                status=job.status,
                started_at=None,
                completed_at=None,
                duration_ms=None,
                result=None,
                error=None,
            )
        )

        return job

    def retry(
        self,
        run_id: str,
        *,
        policy: RetryPolicy | None = None,
    ) -> Job:
        record = self.registry.get(run_id)

        if record is None:
            raise ValueError(
                f"Run not found: {run_id}"
            )

        if record.status != "failed":
            raise ValueError(
                "Only failed runs can be retried."
            )

        classified = classify_error(
            record.error
        )

        if classified.kind == "permanent":
            raise ValueError(
                f"Run failure is permanent: "
                f"{classified.reason}"
            )

        retry_policy = policy or RetryPolicy()

        root_run_id = (
            record.parent_run_id
            if record.parent_run_id
            else record.run_id
        )

        attempt = self.registry.next_attempt(
            root_run_id
        )

        if not retry_policy.can_retry(
            record.attempt
        ):
            raise ValueError(
                f"Retry limit exhausted at attempt "
                f"{record.attempt}."
            )

        job = self.create_job(
            record.objective,
            parent_run_id=root_run_id,
            attempt=attempt,
        )

        return self.execute(job)

    def execute(
        self,
        job: Job,
    ) -> Job:
        plan = ExecutionPlan(
            worker=job.worker or self.route(job),
            research_mode=job.research_mode,
            target_url=job.target_url,
            tool=(
                "browser_harness"
                if job.worker == "browser_worker"
                else (
                    "free-first-deep-research"
                    if job.research_mode == "deep"
                    else (
                        "firecrawl.scrape"
                        if job.target_url
                        else "firecrawl.search+scrape"
                    )
                )
            ),
        )

        job.worker = plan.worker

        if job.worker == "researcher" and job.research_mode is None:
            job.research_mode = self.research_mode(
                job.objective
            )

        if job.target_url is None:
            job.target_url = self.extract_url(
                job.objective
            )

        # Keep the plan synchronized after any inferred fields.
        if job.worker == "browser_worker":
            plan = ExecutionPlan(
                worker=job.worker,
                research_mode=None,
                target_url=job.target_url,
                tool="browser_harness",
            )
        elif job.research_mode == "deep":
            plan = ExecutionPlan(
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=job.target_url,
                tool="free-first-deep-research",
            )
        elif job.target_url:
            plan = ExecutionPlan(
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=job.target_url,
                tool="firecrawl.scrape",
            )
        else:
            plan = ExecutionPlan(
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=None,
                tool="firecrawl.search+scrape",
            )

        job.status = "running"
        job.started_at = utc_now()

        # Preserve an existing worker lease when this job was
        # claimed by the queue worker. API/local execution may
        # legitimately have no lease.
        existing = self.registry.get(job.id)
        lease_until = (
            existing.lease_until
            if existing is not None
            else None
        )

        self.registry.save(
            RunRecord(
                run_id=job.id,
                parent_run_id=job.parent_run_id,
                attempt=job.attempt,
                objective=job.objective,
                worker=job.worker,
                research_mode=job.research_mode,
                target_url=job.target_url,
                tool=plan.tool,
                status=job.status,
                started_at=job.started_at,
                completed_at=None,
                duration_ms=None,
                result=None,
                error=None,
                lease_until=lease_until,
            )
        )

        started_perf = perf_counter()

        try:
            if plan.worker == "browser_worker":
                if not job.target_url:
                    raise ValueError(
                        "Browser task requires a URL."
                    )

                job.result = open_and_inspect(
                    job.target_url
                )

            elif plan.research_mode == "deep":
                deep = deep_research(
                    job.objective
                )

                if not deep.success:
                    raise RuntimeError(
                        deep.error
                        or "Deep research failed."
                    )

                job.result = json.dumps(
                    {
                        "type": "deep_research",
                        "prompt": deep.prompt,
                        "success": deep.success,
                        "status": deep.status,
                        "model": deep.model,
                        "credits_used": deep.credits_used,
                        "data": deep.data,
                        "error": deep.error,
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            elif job.target_url:
                page = research_url(
                    job.target_url
                )

                job.result = json.dumps(
                    {
                        "type": "url_research",
                        "url": page.url,
                        "title": page.title,
                        "source": page.source,
                        "summary": page.summary,
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            else:
                report = research(
                    job.objective
                )

                answer = synthesize(
                    report
                )

                job.result = json.dumps(
                    {
                        "type": "research",
                        "query": answer.query,
                        "answer": answer.answer,
                        "key_findings": answer.key_findings,
                        "citations": [
                            {
                                "title": item.title,
                                "url": item.url,
                                "domain": item.domain,
                            }
                            for item in answer.citations
                        ],
                        "caveats": answer.caveats,
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            job.status = "completed"

        except Exception as exc:
            job.status = "failed"
            job.error = (
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            job.completed_at = utc_now()

            duration_ms = int(
                (perf_counter() - started_perf)
                * 1000
            )

            self.registry.save(
                RunRecord(
                    run_id=job.id,
                    parent_run_id=job.parent_run_id,
                    attempt=job.attempt,
                    objective=job.objective,
                    worker=job.worker,
                    research_mode=job.research_mode,
                    target_url=job.target_url,
                    tool=plan.tool,
                    status=job.status,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    duration_ms=duration_ms,
                    result=job.result,
                    error=job.error,
                    lease_until=None,
                )
            )

        return job
