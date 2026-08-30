from __future__ import annotations

import json

from agents.manager import Manager


def main() -> None:
    manager = Manager()

    jobs = [
        "Open https://example.com and inspect the page",
        "Research https://example.com",
        "Research Firecrawl capabilities for AI agents",
        "Deep research and compare the major AI web research platforms",
    ]

    for objective in jobs:
        plan = manager.plan(objective)

        print("=" * 70)
        print("PLAN")
        print(json.dumps(
            {
                "worker": plan.worker,
                "research_mode": plan.research_mode,
                "target_url": plan.target_url,
                "tool": plan.tool,
            },
            indent=2,
            ensure_ascii=False,
        ))

        job = manager.create_job(objective)
        result = manager.execute(job)

        print()
        print("EXECUTION")
        print(json.dumps(
            {
                "id": result.id,
                "objective": result.objective,
                "worker": result.worker,
                "research_mode": result.research_mode,
                "target_url": result.target_url,
                "status": result.status,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "result": result.result,
                "error": result.error,
            },
            indent=2,
            ensure_ascii=False,
        ))
        print("=" * 70)


if __name__ == "__main__":
    main()
