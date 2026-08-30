from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProviderEvent:
    event_id: str
    run_id: str | None
    provider: str
    operation: str
    status: str
    latency_ms: int
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_type: str | None = None
    error: str | None = None
    created_at: str = ""


class ProviderObservability:
    """Persist provider execution telemetry without storing prompts or secrets."""

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:
        configured = db_path or os.getenv(
            "AGENT_OS_OBSERVABILITY_DB",
            "agent_os.db",
        )
        self.db_path = Path(configured)

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
                CREATE TABLE IF NOT EXISTS provider_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    provider TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    model TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    thinking_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_events_provider
                ON provider_events(provider)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_events_run
                ON provider_events(run_id)
                """
            )
            connection.commit()

    def record(
        self,
        *,
        provider: str,
        operation: str,
        status: str,
        latency_ms: int = 0,
        run_id: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        total_tokens: int | None = None,
        estimated_cost_usd: float = 0.0,
        error_type: str | None = None,
        error: str | None = None,
    ) -> ProviderEvent:
        if not provider.strip():
            raise ValueError("provider cannot be empty")
        if not operation.strip():
            raise ValueError("operation cannot be empty")
        if status not in {"success", "failed", "skipped"}:
            raise ValueError("status must be success, failed, or skipped")
        if latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if min(input_tokens, output_tokens, thinking_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative")

        if total_tokens is None:
            total_tokens = (
                input_tokens
                + output_tokens
                + thinking_tokens
            )

        event = ProviderEvent(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            provider=provider,
            operation=operation,
            status=status,
            latency_ms=int(latency_ms),
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            thinking_tokens=int(thinking_tokens),
            total_tokens=int(total_tokens),
            estimated_cost_usd=float(estimated_cost_usd),
            error_type=error_type,
            error=error,
            created_at=utc_now(),
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO provider_events (
                    event_id,
                    run_id,
                    provider,
                    operation,
                    status,
                    latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    thinking_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    error_type,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.provider,
                    event.operation,
                    event.status,
                    event.latency_ms,
                    event.model,
                    event.input_tokens,
                    event.output_tokens,
                    event.thinking_tokens,
                    event.total_tokens,
                    event.estimated_cost_usd,
                    event.error_type,
                    event.error,
                    event.created_at,
                ),
            )
            connection.commit()

        return event

    def recent(
        self,
        limit: int = 20,
        *,
        provider: str | None = None,
        run_id: str | None = None,
    ) -> list[ProviderEvent]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        conditions: list[str] = []
        parameters: list[Any] = []

        if provider is not None:
            conditions.append("provider = ?")
            parameters.append(provider)

        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)

        where = (
            "WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM provider_events
                {where}
                ORDER BY rowid DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [self._row_to_event(row) for row in rows]

    def summary(
        self,
        *,
        provider: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, int | float]:
        conditions: list[str] = []
        parameters: list[Any] = []

        if provider is not None:
            conditions.append("provider = ?")
            parameters.append(provider)

        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)

        where = (
            "WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS calls,
                    COALESCE(SUM(status = 'success'), 0) AS successes,
                    COALESCE(SUM(status = 'failed'), 0) AS failures,
                    COALESCE(SUM(status = 'skipped'), 0) AS skipped,
                    COALESCE(SUM(latency_ms), 0) AS latency_ms,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM provider_events
                {where}
                """,
                parameters,
            ).fetchone()

        calls = int(row["calls"])
        successes = int(row["successes"])

        return {
            "calls": calls,
            "successes": successes,
            "failures": int(row["failures"]),
            "skipped": int(row["skipped"]),
            "success_rate": round(
                successes / calls,
                4,
            ) if calls else 0.0,
            "total_latency_ms": int(row["latency_ms"]),
            "avg_latency_ms": round(
                int(row["latency_ms"]) / calls,
                2,
            ) if calls else 0.0,
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "thinking_tokens": int(row["thinking_tokens"]),
            "total_tokens": int(row["total_tokens"]),
            "estimated_cost_usd": round(
                float(row["estimated_cost_usd"]),
                8,
            ),
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ProviderEvent:
        return ProviderEvent(
            event_id=row["event_id"],
            run_id=row["run_id"],
            provider=row["provider"],
            operation=row["operation"],
            status=row["status"],
            latency_ms=int(row["latency_ms"]),
            model=row["model"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            thinking_tokens=int(row["thinking_tokens"]),
            total_tokens=int(row["total_tokens"]),
            estimated_cost_usd=float(row["estimated_cost_usd"]),
            error_type=row["error_type"],
            error=row["error"],
            created_at=row["created_at"],
        )

    @staticmethod
    def to_dict(event: ProviderEvent) -> dict[str, Any]:
        return asdict(event)
