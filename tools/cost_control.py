from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Budget:
    budget_id: str
    run_id: str
    max_usd: float
    max_input_tokens: int
    max_output_tokens: int
    reserved_usd: float = 0.0
    spent_usd: float = 0.0


@dataclass(frozen=True)
class UsageEvent:
    event_id: str
    run_id: str
    model: str
    task: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    estimated_cost_usd: float
    created_at: str


class BudgetExceededError(RuntimeError):
    pass


class CostController:
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
                CREATE TABLE IF NOT EXISTS budgets (
                    budget_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    max_usd REAL NOT NULL,
                    max_input_tokens INTEGER NOT NULL,
                    max_output_tokens INTEGER NOT NULL,
                    reserved_usd REAL NOT NULL DEFAULT 0,
                    spent_usd REAL NOT NULL DEFAULT 0
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    thinking_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def create_budget(
        self,
        run_id: str,
        *,
        max_usd: float = 1.00,
        max_input_tokens: int = 500_000,
        max_output_tokens: int = 100_000,
    ) -> Budget:
        if max_usd <= 0:
            raise ValueError(
                "max_usd must be greater than zero."
            )

        if max_input_tokens < 1:
            raise ValueError(
                "max_input_tokens must be at least 1."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be at least 1."
            )

        existing = self.get_budget(run_id)

        if existing is not None:
            return existing

        budget = Budget(
            budget_id=str(uuid.uuid4()),
            run_id=run_id,
            max_usd=max_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO budgets (
                    budget_id,
                    run_id,
                    max_usd,
                    max_input_tokens,
                    max_output_tokens,
                    reserved_usd,
                    spent_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    budget.budget_id,
                    budget.run_id,
                    budget.max_usd,
                    budget.max_input_tokens,
                    budget.max_output_tokens,
                    0.0,
                    0.0,
                ),
            )

            connection.commit()

        return budget

    def get_budget(
        self,
        run_id: str,
    ) -> Budget | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM budgets
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_budget(row)

    def reserve(
        self,
        run_id: str,
        amount_usd: float,
    ) -> Budget:
        if amount_usd <= 0:
            raise ValueError(
                "amount_usd must be greater than zero."
            )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT *
                FROM budgets
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                raise ValueError(
                    f"No budget configured for run: {run_id}"
                )

            available = (
                row["max_usd"]
                - row["spent_usd"]
                - row["reserved_usd"]
            )

            if amount_usd > available:
                raise BudgetExceededError(
                    f"Budget exceeded for {run_id}: "
                    f"requested ${amount_usd:.6f}, "
                    f"available ${available:.6f}."
                )

            connection.execute(
                """
                UPDATE budgets
                SET reserved_usd = reserved_usd + ?
                WHERE run_id = ?
                """,
                (
                    amount_usd,
                    run_id,
                ),
            )

            updated = connection.execute(
                """
                SELECT *
                FROM budgets
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            connection.commit()

            if updated is None:
                raise RuntimeError(
                    "Budget disappeared after reservation."
                )

            return self._row_to_budget(updated)

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def release_reservation(
        self,
        run_id: str,
        amount_usd: float,
    ) -> bool:
        if amount_usd < 0:
            raise ValueError(
                "amount_usd cannot be negative."
            )

        if amount_usd == 0:
            return True

        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE budgets
                SET reserved_usd = MAX(
                    0,
                    reserved_usd - ?
                )
                WHERE run_id = ?
                """,
                (
                    amount_usd,
                    run_id,
                ),
            )

            connection.commit()

        return updated.rowcount == 1

    def record_usage(
        self,
        run_id: str,
        *,
        model: str,
        task: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        estimated_cost_usd: float,
        reservation_usd: float = 0.0,
    ) -> UsageEvent:
        if input_tokens < 0:
            raise ValueError(
                "input_tokens cannot be negative."
            )

        if output_tokens < 0:
            raise ValueError(
                "output_tokens cannot be negative."
            )

        if thinking_tokens < 0:
            raise ValueError(
                "thinking_tokens cannot be negative."
            )

        if estimated_cost_usd < 0:
            raise ValueError(
                "estimated_cost_usd cannot be negative."
            )

        if reservation_usd < 0:
            raise ValueError(
                "reservation_usd cannot be negative."
            )

        event = UsageEvent(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            model=model,
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=estimated_cost_usd,
            created_at=utc_now(),
        )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            budget = connection.execute(
                """
                SELECT *
                FROM budgets
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if budget is None:
                raise ValueError(
                    f"No budget configured for run: {run_id}"
                )

            total_input = connection.execute(
                """
                SELECT COALESCE(
                    SUM(input_tokens),
                    0
                )
                FROM usage_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()[0]

            total_output = connection.execute(
                """
                SELECT COALESCE(
                    SUM(output_tokens),
                    0
                )
                FROM usage_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()[0]

            new_input = (
                int(total_input)
                + input_tokens
            )

            new_output = (
                int(total_output)
                + output_tokens
            )

            if new_input > budget[
                "max_input_tokens"
            ]:
                raise BudgetExceededError(
                    f"Input-token budget exceeded for "
                    f"{run_id}: "
                    f"{new_input} > "
                    f"{budget['max_input_tokens']}."
                )

            if new_output > budget[
                "max_output_tokens"
            ]:
                raise BudgetExceededError(
                    f"Output-token budget exceeded for "
                    f"{run_id}: "
                    f"{new_output} > "
                    f"{budget['max_output_tokens']}."
                )

            projected_spend = (
                budget["spent_usd"]
                + estimated_cost_usd
            )

            if projected_spend > budget["max_usd"]:
                raise BudgetExceededError(
                    f"USD budget exceeded for {run_id}: "
                    f"projected ${projected_spend:.6f}, "
                    f"limit ${budget['max_usd']:.6f}."
                )

            connection.execute(
                """
                INSERT INTO usage_events (
                    event_id,
                    run_id,
                    model,
                    task,
                    input_tokens,
                    output_tokens,
                    thinking_tokens,
                    estimated_cost_usd,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.model,
                    event.task,
                    event.input_tokens,
                    event.output_tokens,
                    event.thinking_tokens,
                    event.estimated_cost_usd,
                    event.created_at,
                ),
            )

            remaining_reserved = max(
                0.0,
                budget["reserved_usd"]
                - reservation_usd,
            )

            connection.execute(
                """
                UPDATE budgets
                SET reserved_usd = ?,
                    spent_usd = spent_usd + ?
                WHERE run_id = ?
                """,
                (
                    remaining_reserved,
                    estimated_cost_usd,
                    run_id,
                ),
            )

            connection.commit()

            return event

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def usage_summary(
        self,
        run_id: str,
    ) -> dict[str, int | float]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS calls,
                    COALESCE(
                        SUM(input_tokens),
                        0
                    ) AS input_tokens,
                    COALESCE(
                        SUM(output_tokens),
                        0
                    ) AS output_tokens,
                    COALESCE(
                        SUM(thinking_tokens),
                        0
                    ) AS thinking_tokens,
                    COALESCE(
                        SUM(estimated_cost_usd),
                        0
                    ) AS estimated_cost_usd
                FROM usage_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        return {
            "calls": int(row["calls"]),
            "input_tokens": int(
                row["input_tokens"]
            ),
            "output_tokens": int(
                row["output_tokens"]
            ),
            "thinking_tokens": int(
                row["thinking_tokens"]
            ),
            "total_tokens": (
                int(row["input_tokens"])
                + int(row["output_tokens"])
                + int(row["thinking_tokens"])
            ),
            "estimated_cost_usd": round(
                float(
                    row["estimated_cost_usd"]
                ),
                8,
            ),
        }

    @staticmethod
    def _row_to_budget(
        row: sqlite3.Row,
    ) -> Budget:
        return Budget(
            budget_id=row["budget_id"],
            run_id=row["run_id"],
            max_usd=float(row["max_usd"]),
            max_input_tokens=int(
                row["max_input_tokens"]
            ),
            max_output_tokens=int(
                row["max_output_tokens"]
            ),
            reserved_usd=float(
                row["reserved_usd"]
            ),
            spent_usd=float(
                row["spent_usd"]
            ),
        )
