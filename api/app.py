from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from agents.manager import Manager
from tools.provider_observability import ProviderObservability
from tools.provider_registry import PROVIDERS
from tools.run_registry import RunRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env"
)


app = FastAPI(
    title="Agent OS",
    version="0.1.0",
    description="Control plane for research and browser agents.",
)


registry = RunRegistry(
    PROJECT_ROOT / "agent_os.db"
)

observability = ProviderObservability(
    PROJECT_ROOT / "agent_os.db"
)

manager = Manager(
    registry=registry
)


class CreateJobRequest(BaseModel):
    objective: str = Field(
        min_length=1,
        description="Natural-language objective for Agent OS.",
    )


def job_payload(
    job: Any,
) -> dict[str, Any]:
    return {
        "id": job.id,
        "objective": job.objective,
        "worker": job.worker,
        "research_mode": job.research_mode,
        "target_url": job.target_url,
        "status": job.status,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "error": job.error,
        "parent_run_id": job.parent_run_id,
        "attempt": job.attempt,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "agent-os",
        "version": "0.1.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/v1/health",
        "worker": "workers.worker",
    }


@app.get("/v1/health")
def health() -> dict[str, Any]:
    queued = len(
        registry.list_runs(
            status="queued",
            limit=100,
        )
    )

    running = len(
        registry.list_runs(
            status="running",
            limit=100,
        )
    )

    return {
        "status": "ok",
        "service": "agent-os",
        "runs": registry.count(),
        "queued": queued,
        "running": running,
    }


@app.post(
    "/v1/jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_job(
    request: CreateJobRequest,
) -> dict[str, Any]:
    job = manager.create_job(
        request.objective
    )

    queued = manager.queue(
        job
    )

    return job_payload(
        queued
    )


@app.get("/v1/jobs/{run_id}")
def get_job(
    run_id: str,
) -> dict[str, Any]:
    record = registry.get(
        run_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    return asdict(
        record
    )


@app.get("/v1/runs")
def list_runs(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    status: str | None = None,
    worker: str | None = None,
    research_mode: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    records = registry.list_runs(
        limit=limit,
        status=status,
        worker=worker,
        research_mode=research_mode,
        parent_run_id=parent_run_id,
    )

    return {
        "count": len(records),
        "runs": [
            asdict(record)
            for record in records
        ],
    }


@app.get("/v1/runs/{run_id}")
def get_run(
    run_id: str,
) -> dict[str, Any]:
    record = registry.get(
        run_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    return asdict(
        record
    )


@app.get("/v1/runs/{run_id}/observability")
def get_run_observability(
    run_id: str,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> dict[str, Any]:
    if registry.get(run_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    events = observability.recent(
        limit=limit,
        run_id=run_id,
    )

    return {
        "run_id": run_id,
        "summary": observability.summary(
            run_id=run_id
        ),
        "events": [
            observability.to_dict(event)
            for event in reversed(events)
        ],
    }


@app.get("/v1/observability/providers")
def provider_observability() -> dict[str, Any]:
    providers = []

    for spec in PROVIDERS:
        summary = observability.summary(
            provider=spec.name
        )
        providers.append(
            {
                "provider": spec.name,
                "capabilities": list(spec.capabilities),
                "free_first": spec.free_first,
                "priority": spec.priority,
                "reserve_only": spec.reserve_only,
                "requires_api_key": spec.requires_api_key,
                "summary": summary,
            }
        )

    return {
        "count": len(providers),
        "providers": providers,
    }


@app.post("/v1/runs/{run_id}/retry")
def retry_run(
    run_id: str,
) -> dict[str, Any]:
    try:
        job = manager.retry(
            run_id
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return job_payload(
        job
    )
