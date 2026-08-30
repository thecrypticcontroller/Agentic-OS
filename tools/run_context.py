from __future__ import annotations

from contextvars import ContextVar, Token


_RUN_ID: ContextVar[str | None] = ContextVar(
    "agent_os_run_id",
    default=None,
)


def current_run_id() -> str | None:
    """Return the run ID bound to the current execution context."""
    return _RUN_ID.get()


def set_run_id(run_id: str) -> Token[str | None]:
    """Bind a run ID to the current execution context."""
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("run_id cannot be empty")
    return _RUN_ID.set(run_id)


def reset_run_id(token: Token[str | None]) -> None:
    """Restore the execution context to its previous run ID."""
    _RUN_ID.reset(token)
