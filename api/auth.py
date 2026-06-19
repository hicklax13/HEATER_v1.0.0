"""API authentication seam — the ONE place write-endpoint auth lives.

Interim gate (slice 2): a static bearer token compared against the
HEATER_API_WRITE_TOKEN env var, DENY-BY-DEFAULT when that var is unset/empty.
At B4 this is replaced by Clerk JWT verification by swapping the AuthVerifier
returned from get_auth_verifier(); require_principal and the routers do NOT
change. Kept self-contained (no `api.deps` import) so there is no import cycle:
the router imports require_principal from here, and api.deps imports nothing
from here."""

from __future__ import annotations

import hmac
import os
from typing import Protocol

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

_TOKEN_ENV = "HEATER_API_WRITE_TOKEN"


class Principal(BaseModel):
    """The authenticated caller. Minimal today (an opaque subject); gains
    tenant/team/tier fields at B4 when multi-tenancy + billing land."""

    subject: str


class AuthVerifier(Protocol):
    """Strategy for turning an Authorization header into a Principal (or
    raising HTTPException 401). Swapped for a Clerk JWT verifier at B4."""

    def verify(self, authorization: str | None) -> Principal: ...


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an 'Authorization: Bearer <token>' header.
    Returns None for any missing/malformed/empty value or non-Bearer scheme."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


class EnvTokenVerifier:
    """Deny-by-default static-token verifier. Reads the expected secret from
    HEATER_API_WRITE_TOKEN at verify time (not import time) so test env-vars
    and rotations take effect without a reload. Constant-time comparison."""

    def verify(self, authorization: str | None) -> Principal:
        expected = os.environ.get(_TOKEN_ENV, "")
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Write API auth not configured.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = _bearer(authorization)
        if token is None or not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal(subject="api-token")


def get_auth_verifier() -> AuthVerifier:
    """DI provider for the write-auth verifier. Tests override this; B4 swaps
    the returned verifier for a Clerk implementation."""
    return EnvTokenVerifier()


def require_principal(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
) -> Principal:
    """FastAPI dependency that enforces write-endpoint auth. Attach it via the
    route's dependencies=[...] list. Returns the Principal so a future B4
    handler can inject it for tenant/team resolution."""
    return verifier.verify(authorization)
