from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from agents.manager import Manager
from tools.run_registry import RunRegistry
from tools.runtime_control import RuntimeControl
from tools.worker_registry import WorkerRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env"
)

POLL_INTERVAL_SECONDS = float(
    os.environ.get(
        "AGENT_OS_POLL_INTERVAL",
        "0.25",
    )
)

LEASE_SECONDS = int(
    os.environ.get(
        "AGENT_OS_LEASE_SECONDS",
        "180",
    )
)

HEARTBEAT_INTERVAL_SECONDS = int(
    os.environ.get(
        "AGENT_OS_HEARTBEAT_INTERVAL",
        "30",
    )
)

RECOVERY_INTERVAL_SECONDS = int(
    os.environ.get(
        "AGENT_OS_RECOVERY_INTERVAL",
        "15",
    )
)

WORKER_CONCURRENCY = max(
    1,
    int(
        os.environ.get(
            "AGENT_OS_WORKER_CONCURRENCY",
            "4",
        )
    ),
)


def available_claim_slots(
    available_slots: int,
    runtime_control: RuntimeControl,
    worker_registry: WorkerRegistry | None = None,
    worker_id: str | None = None,
) -> int:
    """Return the number of queued jobs the worker may claim."""
    if runtime_control.is_queue_paused():
        return 0

    if worker_registry is not None or worker_id is not None:
        if worker_registry is None or worker_id is None:
            return 0

        worker = worker_registry.get(worker_id)

        if worker is None:
            return 0

        if worker.status != "running":
            return 0

    return max(
        0,
        available_slots,
    )


def _heartbeat_loop(
    registry: RunRegistry,
    run_id: str,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(
        HEARTBEAT_INTERVAL_SECONDS
    ):
        try:
            renewed = registry.renew_lease(
                run_id,
                lease_seconds=LEASE_SECONDS,
            )

            if not renewed:
                print(
                    f"Heartbeat lost lease for {run_id}",
                    flush=True,
                )
                return

        except Exception as exc:
            print(
                f"Heartbeat error for {run_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def execute_claimed_run(
    manager: Manager,
    record,
) -> bool:
    if record.status != "running":
        return False

    renewed = manager.registry.renew_lease(
        record.run_id,
        lease_seconds=LEASE_SECONDS,
    )

    if not renewed:
        return False

    job = manager.create_job(
        record.objective,
        parent_run_id=record.parent_run_id,
        attempt=record.attempt,
    )

    job.id = record.run_id
    job.worker = record.worker
    job.research_mode = record.research_mode
    job.target_url = record.target_url
    job.status = "running"

    stop_event = threading.Event()

    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(
            manager.registry,
            record.run_id,
            stop_event,
        ),
        daemon=True,
        name=f"agent-os-heartbeat-{record.run_id}",
    )

    heartbeat.start()

    started = time.monotonic()

    try:
        manager.execute(
            job
        )

        elapsed = time.monotonic() - started

        print(
            f"Completed run {record.run_id} "
            f"in {elapsed:.1f}s "
            f"(attempt {record.attempt})",
            flush=True,
        )

        return True

    except Exception as exc:
        print(
            f"Execution error for {record.run_id}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    finally:
        stop_event.set()

        heartbeat.join(
            timeout=2
        )

        try:
            manager.registry.renew_lease(
                record.run_id,
                lease_seconds=1,
            )
        except Exception:
            pass


def _future_done(
    future: Future,
    run_id: str,
) -> None:
    try:
        future.result()
    except Exception as exc:
        print(
            f"Worker future failed for {run_id}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def worker_loop() -> None:
    registry = RunRegistry(
        PROJECT_ROOT / "agent_os.db"
    )

    manager = Manager(
        registry=registry
    )

    runtime_control = RuntimeControl(
        PROJECT_ROOT / "agent_os.db"
    )

    worker_registry = WorkerRegistry(
        PROJECT_ROOT / "agent_os.db"
    )

    worker_record = worker_registry.register(
        concurrency=WORKER_CONCURRENCY,
    )

    worker_id = worker_record.worker_id

    print(
        "Agent OS concurrent worker started.",
        flush=True,
    )

    print(
        f"Concurrency: {WORKER_CONCURRENCY}",
        flush=True,
    )

    print(
        f"Poll interval: {POLL_INTERVAL_SECONDS}s",
        flush=True,
    )

    executor = ThreadPoolExecutor(
        max_workers=WORKER_CONCURRENCY,
        thread_name_prefix="agent-os-worker",
    )

    futures: dict[Future, str] = {}

    last_recovery = 0.0
    last_worker_heartbeat = 0.0

    try:
        while True:
            try:
                now = time.monotonic()

                if (
                    now - last_worker_heartbeat
                    >= HEARTBEAT_INTERVAL_SECONDS
                ):
                    worker_registry.heartbeat(
                        worker_id,
                        active_runs=len(futures),
                    )
                    last_worker_heartbeat = now

                if (
                    now - last_recovery
                    >= RECOVERY_INTERVAL_SECONDS
                ):
                    recovered = (
                        registry.recover_stale_runs()
                    )

                    if recovered:
                        print(
                            f"Recovered {recovered} "
                            "stale run(s).",
                            flush=True,
                        )

                    last_recovery = now

                completed = []

                for future, run_id in list(
                    futures.items()
                ):
                    if future.done():
                        completed.append(
                            (future, run_id)
                        )

                for future, run_id in completed:
                    futures.pop(
                        future,
                        None,
                    )

                    _future_done(
                        future,
                        run_id,
                    )

                available_slots = (
                    WORKER_CONCURRENCY
                    - len(futures)
                )

                claim_slots = available_claim_slots(
                    available_slots,
                    runtime_control,
                    worker_registry,
                    worker_id,
                )

                for _ in range(
                    claim_slots
                ):
                    record = (
                        registry.claim_next_queued(
                            lease_seconds=LEASE_SECONDS
                        )
                    )

                    if record is None:
                        break

                    print(
                        f"Claimed run {record.run_id} "
                        f"(attempt {record.attempt}) "
                        f"[active={len(futures) + 1}/"
                        f"{WORKER_CONCURRENCY}]",
                        flush=True,
                    )

                    future = executor.submit(
                        execute_claimed_run,
                        manager,
                        record,
                    )

                    futures[future] = (
                        record.run_id
                    )

                current_worker = worker_registry.get(
                    worker_id
                )

                if (
                    current_worker is not None
                    and current_worker.status == "draining"
                    and not futures
                ):
                    worker_registry.stop(
                        worker_id
                    )
                    break

                if not futures:
                    time.sleep(
                        POLL_INTERVAL_SECONDS
                    )
                else:
                    time.sleep(
                        min(
                            POLL_INTERVAL_SECONDS,
                            0.25,
                        )
                    )

            except KeyboardInterrupt:
                print(
                    "Worker stopping...",
                    flush=True,
                )
                break

            except Exception as exc:
                print(
                    "Worker loop error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                time.sleep(
                    POLL_INTERVAL_SECONDS
                )

    finally:
        print(
            "Waiting for active jobs...",
            flush=True,
        )

        executor.shutdown(
            wait=True
        )

        worker_registry.stop(
            worker_id
        )

        print(
            "Agent OS worker stopped.",
            flush=True,
        )


def main() -> None:
    worker_loop()


if __name__ == "__main__":
    main()
