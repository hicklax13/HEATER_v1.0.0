"""Liveness + readiness endpoints with a dependency-health summary.

`/healthz` stays as the existing liveness probe (back-compat); `/livez` is its
alias. `/readyz` checks that the primary database is reachable and returns a
per-dependency status, responding 503 when a required dependency is down so an
orchestrator can keep a not-ready replica out of rotation. Closes the Codex
finding that `/healthz` was a static liveness response with no readiness or
dependency-health surface.

Ops endpoints use `include_in_schema=False` so they do not appear in the public
OpenAPI contract (no snapshot/TS churn)."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _check_database() -> bool:
    """Return True if the primary database accepts a trivial query."""
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            conn.execute("SELECT 1")
            return True
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - a readiness probe must never raise
        logger.exception("readiness: database check failed")
        return False


def install_health_routes(app: FastAPI) -> None:
    @app.get("/livez", include_in_schema=False)
    def livez() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        checks = {"database": _check_database()}
        ready = all(checks.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready", "checks": checks},
        )
