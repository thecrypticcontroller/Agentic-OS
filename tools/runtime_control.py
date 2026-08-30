from __future__ import annotations

import sqlite3
from pathlib import Path


class RuntimeControl:
    """Persistent operational controls shared by API and workers."""

    def __init__(self, db_path: str | Path = "agent_os.db") -> None:
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
                CREATE TABLE IF NOT EXISTS runtime_control (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    queue_paused INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO runtime_control (
                    id,
                    queue_paused
                )
                VALUES (1, 0)
                """
            )

            connection.commit()

    def is_queue_paused(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT queue_paused
                FROM runtime_control
                WHERE id = 1
                """
            ).fetchone()

        if row is None:
            return False

        return bool(row["queue_paused"])

    def set_queue_paused(self, paused: bool) -> bool:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE runtime_control
                SET queue_paused = ?
                WHERE id = 1
                """,
                (1 if paused else 0,),
            )

            connection.commit()

        return updated.rowcount == 1

    def pause_queue(self) -> bool:
        return self.set_queue_paused(True)

    def resume_queue(self) -> bool:
        return self.set_queue_paused(False)

    def snapshot(self) -> dict[str, bool]:
        return {
            "paused": self.is_queue_paused(),
        }
