"""Request-scoped correlation ID, propagated via a contextvar so logs and
error handlers can stamp the current request without threading it through
every call. Set by the correlation-ID middleware in api/main.py."""

from __future__ import annotations

import contextvars
import uuid

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("heater_request_id", default="")


def set_request_id(value: str | None) -> str:
    """Set the current request id (generating one if absent) and return it."""
    rid = value or uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """Return the current request id, or '' outside a request."""
    return _request_id.get()
