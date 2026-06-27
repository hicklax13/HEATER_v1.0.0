"""Standard error envelope + a raisable HeaterError, so operational failures
return a structured, documented body instead of masquerading as valid data.

ADDITIVE: existing endpoints keep their current behavior until migrated to
raise HeaterError. The catch-all handler upgrades unstructured 500s into the
envelope (and never leaks the raw exception). The live frontend reads only the
HTTP status, so changing error *bodies* is backward-compatible."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.request_context import get_request_id

logger = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str = ""
    retryable: bool = False
    degraded: bool = False
    dependency: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HeaterError(Exception):
    """Raise to return a structured error envelope with an explicit status."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        degraded: bool = False,
        dependency: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.degraded = degraded
        self.dependency = dependency
        self.details = details or {}


def _envelope(body: ErrorBody, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": body.model_dump()})


async def _heater_error_handler(request: Request, exc: HeaterError) -> JSONResponse:
    return _envelope(
        ErrorBody(
            code=exc.code,
            message=exc.message,
            request_id=get_request_id(),
            retryable=exc.retryable,
            degraded=exc.degraded,
            dependency=exc.dependency,
            details=exc.details,
        ),
        exc.status_code,
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the real exception server-side; return a safe, generic envelope.
    logger.exception("unhandled exception on %s", request.url.path)
    return _envelope(
        ErrorBody(
            code="internal_error",
            message="An unexpected error occurred.",
            request_id=get_request_id(),
            retryable=True,
        ),
        500,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the HeaterError + catch-all handlers on the app."""
    app.add_exception_handler(HeaterError, _heater_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
