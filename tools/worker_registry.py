from __future__ import annotations

import os
import socket
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    hostname: str
    pid: int
    started_at: str
    last_heartbeat: str
    concurrency: int
    active_runs: int
    status: str


class WorkerRegistry:
    """Persistent worker presence and heartbeat registry."""

    def __init__(
        self,
        db_path: str | Path = "agent_os.db",
    ) -> None:
        self.db_path = Path(db_path)

        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    concurrency INTEGER NOT NULL,
                    active_runs INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def register(
        self,
        *,
        worker_id: str | None = None,
        hostname: str | None = None,
        pid: int | None = None,
        concurrency: int = 1,
    ) -> WorkerRecord:
        if concurrency < 1:
            raise ValueError(
                "concurrency must be at least 1"
            )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        record = WorkerRecord(
            worker_id=worker_id or str(uuid.uuid4()),
            hostname=hostname or socket.gethostname(),
            pid=pid if pid is not None else os.getpid(),
            started_at=now,
            last_heartbeat=now,
            concurrency=concurrency,
            active_runs=0,
            status="running",
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers (
                    worker_id,
                    hostname,
                    pid,
                    started_at,
                    last_heartbeat,
                    concurrency,
                    active_runs,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.worker_id,
                    record.hostname,
                    record.pid,
                    record.started_at,
                    record.last_heartbeat,
                    record.concurrency,
                    record.active_runs,
                    record.status,
                ),
            )
            connection.commit()

        return record

    def heartbeat(
        self,
        worker_id: str,
        *,
        active_runs: int,
    ) -> bool:
        if active_runs < 0:
            raise ValueError(
                "active_runs must be non-negative"
            )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE workers
                SET last_heartbeat = ?,
                    active_runs = ?,
                    status = 'running'
                WHERE worker_id = ?
                """,
                (
                    now,
                    active_runs,
                    worker_id,
                ),
            )

            connection.commit()

        return updated.rowcount == 1

    def stop(
        self,
        worker_id: str,
    ) -> bool:
        now = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE workers
                SET last_heartbeat = ?,
                    status = 'stopped',
                    active_runs = 0
                WHERE worker_id = ?
                """,
                (
                    now,
                    worker_id,
                ),
            )

            connection.commit()

        return updated.rowcount == 1

    def get(
        self,
        worker_id: str,
    ) -> WorkerRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM workers
                WHERE worker_id = ?
                """,
                (worker_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_workers(
        self,
    ) -> list[WorkerRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM workers
                ORDER BY started_at DESC
                """
            ).fetchall()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def snapshot(
        self,
        *,
        stale_after_seconds: int = 90,
    ) -> dict[str, object]:
        if stale_after_seconds < 1:
            raise ValueError(
                "stale_after_seconds must be at least 1"
            )

        now = datetime.now(
            timezone.utc
        )

        workers = self.list_workers()
        normalized: list[dict[str, object]] = []

        for worker in workers:
            try:
                last_heartbeat = datetime.fromisoformat(
                    worker.last_heartbeat
                )
                stale = (
                    now - last_heartbeat
                    > timedelta(
                        seconds=stale_after_seconds
                    )
                )
            except ValueError:
                stale = True

            status = worker.status

            if (
                status == "running"
                and stale
            ):
                status = "stale"

            normalized.append(
                {
                    "worker_id": worker.worker_id,
                    "hostname": worker.hostname,
                    "pid": worker.pid,
                    "started_at": worker.started_at,
                    "last_heartbeat": worker.last_heartbeat,
                    "concurrency": worker.concurrency,
                    "active_runs": worker.active_runs,
                    "status": status,
                }
            )

        healthy = sum(
            1
            for worker in normalized
            if worker["status"] == "running"
        )

        active_runs = sum(
            int(worker["active_runs"])
            for worker in normalized
            if worker["status"] != "stopped"
        )

        return {
            "count": len(normalized),
            "healthy": healthy,
            "active_runs": active_runs,
            "workers": normalized,
        }

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> WorkerRecord:
        return WorkerRecord(
            worker_id=row["worker_id"],
            hostname=row["hostname"],
            pid=int(row["pid"]),
            started_at=row["started_at"],
            last_heartbeat=row["last_heartbeat"],
            concurrency=int(row["concurrency"]),
            active_runs=int(row["active_runs"]),
            status=row["status"],
        )
