"""Defense-in-depth security response headers on every API response.

The browser consumes the app through the Next.js server-side proxy, so the
browser-facing headers (CSP, HSTS) live on the frontend; these protect *direct*
API access and are standard hardening. Additive — `setdefault` never overwrites
a header a route already set, and payloads are untouched."""

from __future__ import annotations

from fastapi import FastAPI

_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def install_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        for key, value in _HEADERS.items():
            response.headers.setdefault(key, value)
        return response
