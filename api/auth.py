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
import logging
import os
from collections.abc import Callable
from typing import Protocol

import jwt
from fastapi import Depends, Header, HTTPException, status
from jwt import PyJWKClient
from pydantic import BaseModel

_TOKEN_ENV = "HEATER_API_WRITE_TOKEN"

logger = logging.getLogger(__name__)


def _unauthorized(detail: str) -> HTTPException:
    """The single 401 shape (deny-by-default, advertises Bearer). Shared by the
    Clerk verifier; EnvTokenVerifier keeps its own inline 401s unchanged."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


class Principal(BaseModel):
    """The authenticated caller. `subject` is opaque (the Clerk `sub` for Clerk
    callers, "api-token" for the env path). `clerk_user_id` is set only for Clerk
    callers — it ties the request to a local AppUser at provisioning time. Gains
    tenant/team/tier fields at B4 when multi-tenancy + billing land."""

    subject: str
    clerk_user_id: str | None = None


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
        # Compare bytes, not str: hmac.compare_digest raises TypeError on the
        # str/str path when either operand is non-ASCII, which would turn a
        # hostile non-ASCII token into a 500 instead of a deny. Bytes never raise.
        if token is None or not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal(subject="api-token")


SigningKeyResolver = Callable[[str], object]


class ClerkVerifier:
    """Verifies a Clerk RS256 session JWT against Clerk's JWKS, locally — no
    per-request network call, no Clerk SDK. Fail-closed: ANY error → 401 (never
    fall open). Tests inject `signing_key_resolver` so no network is touched."""

    def __init__(
        self,
        issuer: str,
        audience: str | None = None,
        jwks_url: str | None = None,
        *,
        signing_key_resolver: SigningKeyResolver | None = None,
        leeway: int = 30,
    ) -> None:
        self._issuer = issuer
        self._audience = audience or None
        self._jwks_url = jwks_url or (issuer.rstrip("/") + "/.well-known/jwks.json")
        self._leeway = leeway
        self._resolver = signing_key_resolver
        self._client: PyJWKClient | None = None

    def _signing_key(self, token: str) -> object:
        if self._resolver is not None:
            return self._resolver(token)
        if self._client is None:
            # PyJWKClient caches keys and refetches on an unknown kid (rotation).
            self._client = PyJWKClient(self._jwks_url)
        return self._client.get_signing_key_from_jwt(token).key

    def verify(self, authorization: str | None) -> Principal:
        token = _bearer(authorization)
        if token is None:
            raise _unauthorized("Invalid or missing bearer token.")
        try:
            key = self._signing_key(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "sub"], "verify_aud": self._audience is not None},
            )
        except Exception as exc:
            # Fail closed on ANY error (bad sig, expired, wrong iss/aud, unreachable
            # JWKS, malformed). Log the failure TYPE only (never the token/claims).
            logger.debug("Clerk token verification failed: %s", type(exc).__name__)
            raise _unauthorized("Invalid or expired token.")
        subject = claims.get("sub")
        if not subject:
            raise _unauthorized("Token missing subject.")
        return Principal(subject=subject, clerk_user_id=subject)


def get_auth_verifier() -> AuthVerifier:
    """DI provider for the write-auth verifier. Returns ClerkVerifier when Clerk
    is configured (CLERK_ISSUER set), else the interim EnvTokenVerifier (the
    server-to-server/CI + dormant default). Read at call time so env changes take
    effect without reload. Tests override this."""
    issuer = os.environ.get("CLERK_ISSUER", "").strip()
    if issuer:
        return ClerkVerifier(
            issuer=issuer,
            audience=os.environ.get("CLERK_AUDIENCE", "").strip() or None,
            jwks_url=os.environ.get("CLERK_JWKS_URL", "").strip() or None,
        )
    return EnvTokenVerifier()


def require_principal(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
) -> Principal:
    """FastAPI dependency that enforces write-endpoint auth. Attach it via the
    route's dependencies=[...] list. Returns the Principal so a future B4
    handler can inject it for tenant/team resolution."""
    return verifier.verify(authorization)
