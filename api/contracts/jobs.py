"""Async job resource contract (shape only). The /api/v1/jobs router and the
Redis/Arq worker execution land in Phase 9; this fixes the public shape now so
later phases and the frontend can build against it."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    expired = "expired"


class JobResource(BaseModel):
    job_id: str
    status: JobStatus
    job_type: str
    progress: float = 0.0
    result_url: str | None = None
    error_code: str | None = None
