from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    run_id: str
    objective: str
    worker: str | None
    research_mode: str | None
    target_url: str | None
    tool: str | None
    status: str
    started_at: str | None
    completed_at: str | None
    duration_ms: int | None
    result: Any | None
    error: str | None
    parent_run_id: str | None = None
    attempt: int = 1
    lease_until: str | None = None
    worker_id: str | None = None


class RunRegistry:
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
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    objective TEXT NOT NULL,
                    worker TEXT,
                    research_mode TEXT,
                    target_url TEXT,
                    tool TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_ms INTEGER,
                    result_json TEXT,
                    error TEXT,
                    lease_until TEXT,
                    worker_id TEXT
                )
                """
            )

            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(runs)"
                ).fetchall()
            }

            if "parent_run_id" not in columns:
                connection.execute(
                    """
                    ALTER TABLE runs
                    ADD COLUMN parent_run_id TEXT
                    """
                )

            if "attempt" not in columns:
                connection.execute(
                    """
                    ALTER TABLE runs
                    ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1
                    """
                )

            if "lease_until" not in columns:
                connection.execute(
                    """
                    ALTER TABLE runs
                    ADD COLUMN lease_until TEXT
                    """
                )

            if "worker_id" not in columns:
                connection.execute(
                    """
                    ALTER TABLE runs
                    ADD COLUMN worker_id TEXT
                    """
                )

            connection.commit()

    def save(
        self,
        record: RunRecord,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    parent_run_id,
                    attempt,
                    objective,
                    worker,
                    research_mode,
                    target_url,
                    tool,
                    status,
                    started_at,
                    completed_at,
                    duration_ms,
                    result_json,
                    error,
                    lease_until,
                    worker_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.parent_run_id,
                    record.attempt,
                    record.objective,
                    record.worker,
                    record.research_mode,
                    record.target_url,
                    record.tool,
                    record.status,
                    record.started_at,
                    record.completed_at,
                    record.duration_ms,
                    json.dumps(
                        record.result,
                        ensure_ascii=False,
                    )
                    if record.result is not None
                    else None,
                    record.error,
                    record.lease_until,
                    record.worker_id,
                ),
            )

            connection.commit()

    def get(
        self,
        run_id: str,
    ) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_runs(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
        worker: str | None = None,
        research_mode: str | None = None,
        parent_run_id: str | None = None,
    ) -> list[RunRecord]:
        if limit < 1:
            raise ValueError(
                "limit must be at least 1"
            )

        conditions: list[str] = []
        parameters: list[Any] = []

        if status is not None:
            conditions.append(
                "status = ?"
            )
            parameters.append(status)

        if worker is not None:
            conditions.append(
                "worker = ?"
            )
            parameters.append(worker)

        if research_mode is not None:
            conditions.append(
                "research_mode = ?"
            )
            parameters.append(research_mode)

        if parent_run_id is not None:
            conditions.append(
                "(run_id = ? OR parent_run_id = ?)"
            )
            parameters.extend(
                [parent_run_id, parent_run_id]
            )

        where = (
            "WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        query = f"""
            SELECT *
            FROM runs
            {where}
            ORDER BY rowid DESC
            LIMIT ?
        """

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def claim_next_queued(
        self,
        lease_seconds: int = 120,
    ) -> RunRecord | None:
        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be at least 1"
            )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT *
                FROM runs
                WHERE status = 'queued'
                ORDER BY rowid ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                connection.rollback()
                return None

            now = datetime.now(timezone.utc)

            lease_until = (
                now.timestamp()
                + lease_seconds
            )

            lease_text = datetime.fromtimestamp(
                lease_until,
                timezone.utc,
            ).isoformat()

            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'running',
                    started_at = COALESCE(
                        started_at,
                        ?
                    ),
                    lease_until = ?
                WHERE run_id = ?
                  AND status = 'queued'
                """,
                (
                    now.isoformat(),
                    lease_text,
                    row["run_id"],
                ),
            )

            if updated.rowcount != 1:
                connection.rollback()
                return None

            claimed = connection.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (row["run_id"],),
            ).fetchone()

            connection.commit()

            if claimed is None:
                return None

            return self._row_to_record(
                claimed
            )

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()


    def claim_next_queued_for_worker(
        self,
        worker_id: str,
        lease_seconds: int = 120,
    ) -> RunRecord | None:
        """Atomically claim one queued run against worker capacity."""
        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be at least 1"
            )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            worker = connection.execute(
                """
                SELECT *
                FROM workers
                WHERE worker_id = ?
                """,
                (worker_id,),
            ).fetchone()

            if worker is None:
                connection.rollback()
                return None

            if worker["status"] != "running":
                connection.rollback()
                return None

            available = max(
                0,
                int(worker["concurrency"])
                - int(worker["active_runs"]),
            )

            if available < 1:
                connection.rollback()
                return None

            row = connection.execute(
                """
                SELECT *
                FROM runs
                WHERE status = 'queued'
                ORDER BY rowid ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                connection.rollback()
                return None

            now = datetime.now(
                timezone.utc
            )

            lease_text = datetime.fromtimestamp(
                now.timestamp() + lease_seconds,
                timezone.utc,
            ).isoformat()

            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'running',
                    started_at = COALESCE(
                        started_at,
                        ?
                    ),
                    lease_until = ?,
                    worker_id = ?
                WHERE run_id = ?
                  AND status = 'queued'
                """,
                (
                    now.isoformat(),
                    lease_text,
                    worker_id,
                    row["run_id"],
                ),
            )

            if updated.rowcount != 1:
                connection.rollback()
                return None

            worker_updated = connection.execute(
                """
                UPDATE workers
                SET active_runs = active_runs + 1,
                    last_heartbeat = ?
                WHERE worker_id = ?
                  AND status = 'running'
                  AND active_runs < concurrency
                """,
                (
                    now.isoformat(),
                    worker_id,
                ),
            )

            if worker_updated.rowcount != 1:
                connection.rollback()
                return None

            claimed = connection.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (row["run_id"],),
            ).fetchone()

            if claimed is None:
                connection.rollback()
                return None

            connection.commit()

            return self._row_to_record(
                claimed
            )

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()


    def reconcile_worker_active_runs(
        self,
        worker_id: str,
    ) -> int | None:
        """Repair workers.active_runs from authoritative running runs."""
        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                worker = connection.execute(
                    """
                    SELECT worker_id
                    FROM workers
                    WHERE worker_id = ?
                    """,
                    (worker_id,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if "no such table: workers" in str(exc).lower():
                    connection.rollback()
                    return None
                raise

            if worker is None:
                connection.rollback()
                return None

            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM runs
                WHERE status = 'running'
                  AND worker_id = ?
                """,
                (worker_id,),
            ).fetchone()

            active_runs = int(
                row["count"]
            )

            connection.execute(
                """
                UPDATE workers
                SET active_runs = ?
                WHERE worker_id = ?
                """,
                (
                    active_runs,
                    worker_id,
                ),
            )

            connection.commit()

            return active_runs

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def renew_lease(
        self,
        run_id: str,
        lease_seconds: int = 120,
    ) -> bool:
        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be at least 1"
            )

        now = datetime.now(timezone.utc)

        lease_until = datetime.fromtimestamp(
            now.timestamp() + lease_seconds,
            timezone.utc,
        ).isoformat()

        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE runs
                SET lease_until = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (
                    lease_until,
                    run_id,
                ),
            )

            connection.commit()

        return updated.rowcount == 1

    def recover_stale_runs(
        self,
    ) -> int:
        now = utc_now()

        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'queued',
                    lease_until = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    duration_ms = NULL,
                    result_json = NULL,
                    error = 'Recovered stale running job.',
                    worker_id = NULL
                WHERE status = 'running'
                  AND (
                      lease_until IS NULL
                      OR lease_until < ?
                  )
                """,
                (now,),
            )

            connection.commit()

        return int(
            updated.rowcount
        )

    def list_recent(
        self,
        limit: int = 20,
    ) -> list[RunRecord]:
        return self.list_runs(
            limit=limit
        )

    def next_attempt(
        self,
        root_run_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(
                    MAX(attempt),
                    0
                ) AS max_attempt
                FROM runs
                WHERE run_id = ?
                   OR parent_run_id = ?
                """,
                (
                    root_run_id,
                    root_run_id,
                ),
            ).fetchone()

        return int(
            row["max_attempt"]
        ) + 1

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM runs
                """
            ).fetchone()

        return int(
            row["count"]
        )

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> RunRecord:
        result_json = row["result_json"]

        try:
            result = (
                json.loads(result_json)
                if result_json is not None
                else None
            )
        except json.JSONDecodeError:
            result = result_json

        return RunRecord(
            run_id=row["run_id"],
            parent_run_id=row["parent_run_id"],
            attempt=int(
                row["attempt"] or 1
            ),
            objective=row["objective"],
            worker=row["worker"],
            research_mode=row["research_mode"],
            target_url=row["target_url"],
            tool=row["tool"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
            result=result,
            error=row["error"],
            lease_until=row["lease_until"],
            worker_id=(
                row["worker_id"]
                if "worker_id" in row.keys()
                else None
            ),
        )

    def dump(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        record = self.get(
            run_id
        )

        if record is None:
            return None

        return asdict(record)
