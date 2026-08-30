from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agents.manager import Manager
from tools.run_registry import RunRegistry


PROJECT_ROOT = Path(
    os.environ.get(
        "CLAUDE_PROJECT_DIR",
        Path(__file__).resolve().parent,
    )
).resolve()

load_dotenv(
    PROJECT_ROOT / ".env"
)


mcp = FastMCP(
    "agent-os"
)

registry = RunRegistry(
    PROJECT_ROOT / "agent_os.db"
)

manager = Manager(
    registry=registry
)


@mcp.tool()
def agent_os_health() -> dict[str, Any]:
    """
    Check Agent OS health and queue state.
    """
    queued = registry.list_runs(
        status="queued",
        limit=100,
    )

    running = registry.list_runs(
        status="running",
        limit=100,
    )

    return {
        "service": "agent-os",
        "status": "ok",
        "runs": registry.count(),
        "queued": len(queued),
        "running": len(running),
        "project_root": str(PROJECT_ROOT),
    }


@mcp.tool()
def submit_job(
    objective: str,
) -> dict[str, Any]:
    """
    Submit a research or browser task to Agent OS.
    """
    job = manager.create_job(
        objective
    )

    queued = manager.queue(
        job
    )

    return {
        "id": queued.id,
        "objective": queued.objective,
        "worker": queued.worker,
        "research_mode": queued.research_mode,
        "target_url": queued.target_url,
        "status": queued.status,
        "attempt": queued.attempt,
        "parent_run_id": queued.parent_run_id,
    }


@mcp.tool()
def get_job(
    run_id: str,
) -> dict[str, Any]:
    """
    Get the complete current state of an Agent OS run.
    """
    record = registry.get(
        run_id
    )

    if record is None:
        return {
            "error": f"Run not found: {run_id}"
        }

    return {
        "run_id": record.run_id,
        "objective": record.objective,
        "worker": record.worker,
        "research_mode": record.research_mode,
        "target_url": record.target_url,
        "tool": record.tool,
        "status": record.status,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "duration_ms": record.duration_ms,
        "result": record.result,
        "error": record.error,
        "parent_run_id": record.parent_run_id,
        "attempt": record.attempt,
        "lease_until": record.lease_until,
    }


@mcp.tool()
def list_runs(
    limit: int = 10,
    status: str | None = None,
    worker: str | None = None,
    research_mode: str | None = None,
) -> dict[str, Any]:
    """
    List recent Agent OS runs.
    """
    if limit < 1:
        limit = 1

    if limit > 100:
        limit = 100

    records = registry.list_runs(
        limit=limit,
        status=status,
        worker=worker,
        research_mode=research_mode,
    )

    return {
        "count": len(records),
        "runs": [
            {
                "run_id": record.run_id,
                "objective": record.objective,
                "worker": record.worker,
                "research_mode": record.research_mode,
                "status": record.status,
                "attempt": record.attempt,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "duration_ms": record.duration_ms,
                "error": record.error,
            }
            for record in records
        ],
    }


@mcp.tool()
def retry_job(
    run_id: str,
) -> dict[str, Any]:
    """
    Retry a failed Agent OS run.
    """
    try:
        job = manager.retry(
            run_id
        )

    except ValueError as exc:
        return {
            "error": str(exc)
        }

    return {
        "id": job.id,
        "objective": job.objective,
        "worker": job.worker,
        "research_mode": job.research_mode,
        "target_url": job.target_url,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "parent_run_id": job.parent_run_id,
        "attempt": job.attempt,
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )
