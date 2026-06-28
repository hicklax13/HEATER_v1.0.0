"""In-memory job store for async lineup optimization.

The Enhanced optimize (FA composition) takes up to ~3 minutes — far past Vercel's
~30s serverless gateway timeout — so the client can't wait on one synchronous
request. Instead the start endpoint creates a job here, runs the optimize in a
FastAPI BackgroundTask, and the client polls the result endpoint until the job
finishes. Each request is fast → no gateway timeout.

Single Railway replica (the SQLite sole-writer invariant), so an in-process dict
guarded by a Lock is sufficient — no cross-process store needed. Finished jobs are
pruned after a TTL so the dict can't grow unbounded across a long-lived process.

The stored `result` is a LineupOptimizeResponse (or None until done); `error` is a
string when the background runner failed. status ∈ {"running","done","error"}.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Prune jobs this many seconds after they were created (covers the ~3min worst-case
# Enhanced run plus client poll latency, with comfortable headroom).
_TTL_SECONDS = 600.0

_lock = threading.Lock()
_jobs: dict[str, _Job] = {}


@dataclass
class _Job:
    status: str = "running"  # "running" | "done" | "error"
    result: Any = None  # LineupOptimizeResponse | None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


def _prune_locked(now: float) -> None:
    """Drop jobs older than the TTL. Caller MUST hold _lock."""
    stale = [jid for jid, j in _jobs.items() if now - j.created_at > _TTL_SECONDS]
    for jid in stale:
        _jobs.pop(jid, None)


def create() -> str:
    """Register a new running job and return its id."""
    job_id = uuid.uuid4().hex
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        _jobs[job_id] = _Job(created_at=now)
    return job_id


def finish(job_id: str, result: Any = None, error: str | None = None) -> None:
    """Mark a job complete. error set → status 'error'; otherwise 'done'.

    Idempotent + safe on an unknown/pruned id (no-op) so the background runner can
    always call it without a guard."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        if error is not None:
            job.status = "error"
            job.error = error
            job.result = None
        else:
            job.status = "done"
            job.result = result
            job.error = None


def get(job_id: str) -> dict | None:
    """Return {status, result, error} for a job, or None if unknown/pruned."""
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {"status": job.status, "result": job.result, "error": job.error}


def reset() -> None:
    """Clear all jobs — test isolation only."""
    with _lock:
        _jobs.clear()


def run_optimize_job(service, job_id: str, team_name: str, date, scope: str, depth: str) -> None:
    """Background-task body: run the (possibly slow, Enhanced) optimize and store the
    result on the job. NEVER raises — any failure is captured as the job's error so a
    crash in the worker can't take down the request thread or leave the job hung in
    'running'. This is the orchestration the THIN router hands to BackgroundTasks."""
    import logging

    try:
        result = service.optimize(team_name, date, scope, depth=depth)
        finish(job_id, result=result)
    except Exception as exc:  # noqa: BLE001 — the runner must swallow to record the error
        logging.getLogger(__name__).warning("optimize job %s failed: %s", job_id, exc)
        finish(job_id, error=str(exc) or "optimize failed")
