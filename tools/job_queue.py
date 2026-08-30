from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueueItem:
    run_id: str


class JobQueue:
    def __init__(self, db_path: str | Path = "agent_os.db") -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def enqueue(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'queued',
                    started_at = NULL,
                    completed_at = NULL,
                    duration_ms = NULL,
                    result_json = NULL,
                    error = NULL
                WHERE run_id = ?
                  AND status IN ('queued', 'failed')
                """,
                (run_id,),
            )
            connection.commit()

    def claim_next(self) -> QueueItem | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            row = connection.execute(
                """
                SELECT run_id
                FROM runs
                WHERE status = 'queued'
                ORDER BY rowid ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                connection.commit()
                return None

            connection.execute(
                """
                UPDATE runs
                SET status = 'running'
                WHERE run_id = ?
                  AND status = 'queued'
                """,
                (row["run_id"],),
            )

            connection.commit()

            return QueueItem(
                run_id=row["run_id"]
            )
