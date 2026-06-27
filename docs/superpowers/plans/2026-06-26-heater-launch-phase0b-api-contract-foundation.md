# Phase 0b — API Contract Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay the cross-cutting API contracts every later phase builds on — a standard error envelope, request-correlation IDs, an OpenAPI bearer-security scheme, an API-versioning policy, and the idempotency + async-job contract shapes — all **additive and backward-compatible with the live beta**.

**Architecture:** Additive plumbing on the FastAPI app. A correlation-ID middleware and two exception handlers (a raisable `HeaterError` and a catch-all that upgrades unstructured 500s) are wired in `api/main.py`. New contract modules (`api/errors.py`, `api/request_context.py`, `api/idempotency.py`, `api/contracts/jobs.py`) define shapes that later phases enforce. The OpenAPI security scheme is added via a custom `app.openapi`, which changes `api/openapi.json` (regenerated + the frontend types resynced under the Phase-0a guard).

**Tech Stack:** FastAPI 0.137.1, Pydantic 2, Starlette middleware, pytest + `fastapi.testclient.TestClient`.

**Master spec:** `docs/superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md` (Phase 0). **Backend audit source:** `codex-backend-critiques.md` §8 (Phase 1: API Contract and Failure Architecture).

**Backward-compat contract (load-bearing):** The live frontend (`web/src/lib/api/client.ts`) throws `ApiError(res.status, path)` on any non-OK response and **never parses the error body**; `errors.ts` branches only on status (401/402/409). Therefore: (a) error-body shape may change freely; (b) status codes 401/402/409 and the 200 success path MUST be preserved; (c) FastAPI's default 422 validation shape is left untouched (it is in `generated.ts`). This plan only adds a 500 envelope (previously an unstructured `Internal Server Error`), an `X-Request-ID` response header, and OpenAPI metadata — all additive.

**Scope boundary:** Idempotency and async-jobs are delivered as **contracts only** (types + seams + tests), not enforced on live routes — enforcement needs the durable store/workers from Phases 2/9. The `/api/v1` prefix migration is **documented but deferred** (a coordinated FE+BE change for the cutover).

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `api/request_context.py` | Request-scoped correlation ID (contextvar) | Create |
| `api/errors.py` | Error envelope models + `HeaterError` + handlers | Create |
| `api/idempotency.py` | Idempotency contract: header dep + store Protocol + in-memory impl | Create |
| `api/contracts/jobs.py` | Async-job resource + status enum (shape only) | Create |
| `api/main.py` | Wire middleware + handlers + custom OpenAPI | Modify |
| `api/openapi.json` | Regenerated with the BearerAuth scheme | Modify |
| `web/src/lib/api/generated.ts` | Resynced to the new openapi.json | Modify |
| `docs/launch/api-versioning-policy.md` | Versioning + compatibility policy | Create |
| `docs/launch/evidence_registry.yaml` | Flip P0B row(s) to passing | Modify |
| `tests/api/test_request_context.py` | Correlation-ID middleware tests | Create |
| `tests/api/test_error_envelope.py` | Envelope + handler tests | Create |
| `tests/api/test_openapi_security_scheme.py` | BearerAuth scheme present | Create |
| `tests/api/test_idempotency_contract.py` | Idempotency store/dep tests | Create |
| `tests/api/test_job_contract.py` | Job contract model tests | Create |

---

## Task 1: Correlation-ID middleware (TDD)

**Files:**
- Create: `api/request_context.py`, `tests/api/test_request_context.py`
- Modify: `api/main.py`

- [ ] **Step 1: Write the failing test**

`tests/api/test_request_context.py`:
```python
from fastapi.testclient import TestClient

from api.main import create_app


def test_response_has_request_id_header():
    client = TestClient(create_app())
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID")  # non-empty


def test_provided_request_id_is_echoed():
    client = TestClient(create_app())
    res = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert res.headers.get("X-Request-ID") == "abc-123"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_request_context.py -v`
Expected: FAIL — no `X-Request-ID` header on the response.

- [ ] **Step 3: Create the request-context module**

`api/request_context.py`:
```python
"""Request-scoped correlation ID, propagated via a contextvar so logs and
error handlers can stamp the current request without threading it through
every call. Set by the correlation-ID middleware in api/main.py."""

from __future__ import annotations

import contextvars
import uuid

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "heater_request_id", default=""
)


def set_request_id(value: str | None) -> str:
    """Set the current request id (generating one if absent) and return it."""
    rid = value or uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """Return the current request id, or '' outside a request."""
    return _request_id.get()
```

- [ ] **Step 4: Wire the middleware into `create_app`**

In `api/main.py`, add the import near the top (after the existing imports):
```python
from api.request_context import REQUEST_ID_HEADER, set_request_id
```
Then, inside `create_app()` immediately AFTER `app.add_middleware(CORSMiddleware, ...)` block and BEFORE the `@app.get("/healthz")` route, add:
```python
    @app.middleware("http")
    async def _correlation_id(request, call_next):
        rid = set_request_id(request.headers.get(REQUEST_ID_HEADER))
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/api/test_request_context.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Confirm no api regression**

Run: `python -m pytest tests/api -q`
Expected: PASS (the existing api suite still green; only the new header is added). Note: `tests/api/test_openapi_contract.py` should still pass — middleware does not change the schema.

- [ ] **Step 7: Commit**

```bash
git add api/request_context.py tests/api/test_request_context.py api/main.py
git commit -m "feat(api): request-correlation-ID middleware + contextvar (Phase 0b)"
```

---

## Task 2: Error envelope + HeaterError + handlers (TDD)

**Files:**
- Create: `api/errors.py`, `tests/api/test_error_envelope.py`
- Modify: `api/main.py`

- [ ] **Step 1: Write the failing test**

`tests/api/test_error_envelope.py`:
```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import HeaterError, install_error_handlers


@pytest.fixture
def app_with_routes():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom-heater")
    def boom_heater():
        raise HeaterError(
            "provider_unavailable",
            "League data could not be refreshed.",
            status_code=503,
            retryable=True,
            dependency="yahoo",
        )

    @app.get("/boom-unhandled")
    def boom_unhandled():
        raise ValueError("kaboom")

    return app


def test_heater_error_renders_envelope(app_with_routes):
    client = TestClient(app_with_routes)
    res = client.get("/boom-heater")
    assert res.status_code == 503
    body = res.json()
    assert body["error"]["code"] == "provider_unavailable"
    assert body["error"]["message"] == "League data could not be refreshed."
    assert body["error"]["retryable"] is True
    assert body["error"]["dependency"] == "yahoo"
    assert "request_id" in body["error"]


def test_unhandled_exception_is_enveloped_not_leaked(app_with_routes):
    # raise_server_exceptions=False so the handler runs instead of TestClient re-raising.
    client = TestClient(app_with_routes, raise_server_exceptions=False)
    res = client.get("/boom-unhandled")
    assert res.status_code == 500
    body = res.json()
    assert body["error"]["code"] == "internal_error"
    assert "kaboom" not in res.text  # the raw exception message is not leaked
    assert body["error"]["retryable"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_error_envelope.py -v`
Expected: FAIL — `ImportError` (`api.errors` not defined).

- [ ] **Step 3: Create the errors module**

`api/errors.py`:
```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_error_envelope.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire the handlers into `create_app`**

In `api/main.py`, add the import:
```python
from api.errors import install_error_handlers
```
Then inside `create_app()`, AFTER the `_correlation_id` middleware definition and BEFORE the `/healthz` route, add:
```python
    install_error_handlers(app)
```

- [ ] **Step 6: Confirm no api regression**

Run: `python -m pytest tests/api -q`
Expected: PASS. The catch-all `Exception` handler only changes the body of previously-unstructured 500s; existing endpoints and their status codes are unchanged. If any existing test asserts the literal `"Internal Server Error"` body, that is the one acceptable update — adjust it to assert `status_code == 500` (the status is the contract, not the body).

- [ ] **Step 7: Commit**

```bash
git add api/errors.py tests/api/test_error_envelope.py api/main.py
git commit -m "feat(api): standard error envelope + HeaterError + catch-all 500 handler (Phase 0b)"
```

---

## Task 3: OpenAPI bearer-security scheme (TDD)

**Files:**
- Create: `tests/api/test_openapi_security_scheme.py`
- Modify: `api/main.py`, `api/openapi.json`, `web/src/lib/api/generated.ts`

- [ ] **Step 1: Write the failing test**

`tests/api/test_openapi_security_scheme.py`:
```python
from api.main import create_app


def test_openapi_documents_bearer_auth_scheme():
    schema = create_app().openapi()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes
    assert schemes["BearerAuth"]["type"] == "http"
    assert schemes["BearerAuth"]["scheme"] == "bearer"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_openapi_security_scheme.py -v`
Expected: FAIL — no `BearerAuth` scheme.

- [ ] **Step 3: Add a custom OpenAPI that documents the scheme**

In `api/main.py`, add the import:
```python
from fastapi.openapi.utils import get_openapi
```
Then inside `create_app()`, AFTER all `app.include_router(...)` calls and BEFORE `return app`, add:
```python
    def _custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": (
                "Clerk session JWT. Required on write + Pro/admin endpoints; "
                "harmless on open reads. Per-operation security requirements are "
                "annotated progressively as each phase hardens its routes."
            ),
        }
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi
```

- [ ] **Step 4: Run to verify the scheme test passes**

Run: `python -m pytest tests/api/test_openapi_security_scheme.py -v`
Expected: PASS.

- [ ] **Step 5: Regenerate the committed snapshot + frontend types**

The schema changed, so the snapshot is now intentionally stale. Regenerate both:
```bash
python scripts/export_openapi.py
cd web && pnpm gen:api && cd ..
```

- [ ] **Step 6: Verify both contract guards pass**

Run:
```bash
python -m pytest tests/api/test_openapi_contract.py -q
python scripts/launch/check_ts_sync.py
```
Expected: the snapshot test PASSES (committed == live) and the TS checker prints `generated.ts is in sync`.

- [ ] **Step 7: Commit**

```bash
git add api/main.py api/openapi.json web/src/lib/api/generated.ts tests/api/test_openapi_security_scheme.py
git commit -m "feat(api): document BearerAuth security scheme in OpenAPI (Phase 0b)"
```

---

## Task 4: Idempotency contract (TDD)

**Files:**
- Create: `api/idempotency.py`, `tests/api/test_idempotency_contract.py`

- [ ] **Step 1: Write the failing test**

`tests/api/test_idempotency_contract.py`:
```python
from api.idempotency import (
    IDEMPOTENCY_HEADER,
    InMemoryIdempotencyStore,
)


def test_header_name_is_standard():
    assert IDEMPOTENCY_HEADER == "Idempotency-Key"


def test_in_memory_store_roundtrip():
    store = InMemoryIdempotencyStore()
    assert store.get("k1") is None
    store.put("k1", {"ok": True})
    assert store.get("k1") == {"ok": True}


def test_in_memory_store_does_not_overwrite_on_replay():
    store = InMemoryIdempotencyStore()
    store.put("k1", {"v": 1})
    # Replays return the first stored result; put() with the same key is a no-op.
    store.put("k1", {"v": 2})
    assert store.get("k1") == {"v": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_idempotency_contract.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Create the idempotency contract**

`api/idempotency.py`:
```python
"""Mutation idempotency contract (seam + types). The in-memory store is a
single-replica placeholder; a durable Redis/Postgres store lands in Phase 9.
NOT yet enforced on live write routes — wiring happens when the durable store
and provider writes are built (Phases 5/9)."""

from __future__ import annotations

from typing import Any, Protocol

from fastapi import Header

IDEMPOTENCY_HEADER = "Idempotency-Key"


class IdempotencyStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, result: Any) -> None: ...


class InMemoryIdempotencyStore:
    """Single-process store. First write per key wins (replays are stable)."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._d.get(key)

    def put(self, key: str, result: Any) -> None:
        # First-write-wins so a replay never overwrites the original result.
        self._d.setdefault(key, result)


async def idempotency_key(
    idempotency_key: str | None = Header(default=None),
) -> str | None:
    """FastAPI dependency that surfaces the optional Idempotency-Key header."""
    return idempotency_key
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_idempotency_contract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/idempotency.py tests/api/test_idempotency_contract.py
git commit -m "feat(api): idempotency contract (header dep + store seam, in-memory) (Phase 0b)"
```

---

## Task 5: Async-job contract (TDD)

**Files:**
- Create: `api/contracts/jobs.py`, `tests/api/test_job_contract.py`

- [ ] **Step 1: Write the failing test**

`tests/api/test_job_contract.py`:
```python
from api.contracts.jobs import JobResource, JobStatus


def test_job_status_values():
    assert {s.value for s in JobStatus} == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "canceled",
        "expired",
    }


def test_job_resource_minimal():
    job = JobResource(job_id="job_1", status=JobStatus.queued, job_type="playoff_sim")
    assert job.progress == 0.0
    assert job.result_url is None
    assert job.status is JobStatus.queued


def test_job_resource_serializes_status_as_string():
    job = JobResource(job_id="job_1", status=JobStatus.running, job_type="trade_mc")
    assert job.model_dump()["status"] == "running"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_job_contract.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Create the job contract**

`api/contracts/jobs.py`:
```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_job_contract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/contracts/jobs.py tests/api/test_job_contract.py
git commit -m "feat(api): async-job resource contract (shape only; Phase 9 builds workers) (Phase 0b)"
```

---

## Task 6: Versioning policy + registry update

**Files:**
- Create: `docs/launch/api-versioning-policy.md`
- Modify: `docs/launch/evidence_registry.yaml`

- [ ] **Step 1: Write the versioning policy**

`docs/launch/api-versioning-policy.md`:
```markdown
# HEATER API versioning & compatibility policy

**Status:** adopted 2026-06-26 (Phase 0b). The `/api/v1` prefix migration is
DEFERRED to the strangler-fig cutover (a coordinated frontend+backend change);
this policy governs change management until then.

## Surface

- Current surface: `/api/*` (unversioned), proxied server-side by the Next.js
  frontend (`web/next.config.ts`), so there are no browser cross-origin calls.
- The committed contract is `api/openapi.json`, snapshot-guarded by
  `tests/api/test_openapi_contract.py`; the frontend types are generated from it
  and drift-guarded by the `openapi-ts-sync` CI job.

## Compatibility rules

- **Additive changes** (new endpoint, new OPTIONAL field, new enum value behind a
  flag) do not require a version bump. Regenerate the snapshot + frontend types.
- **Breaking changes** (removing/renaming a field, tightening a type, changing a
  status code the frontend branches on — 401/402/409, or changing success shape)
  require: a migration window, a deprecation note in the operation description,
  and a coordinated frontend update in the same release train.
- **Errors** use the standard envelope (`api/errors.py`); error *bodies* may evolve
  freely because the frontend reads only the status code. Status codes are part of
  the contract and follow the breaking-change rule.

## Authentication

- Documented in OpenAPI as the `BearerAuth` scheme (Clerk session JWT). Per-operation
  security requirements are annotated progressively as each phase hardens its routes.

## When `/api/v1` lands (cutover)

- Introduce `/api/v1` as an alias, keep `/api/*` serving for one migration window,
  add `Deprecation`/`Sunset` headers to the unversioned surface, switch the frontend
  proxy, then remove the unversioned surface after the window.
```

- [ ] **Step 2: Flip the P0B registry row to passing**

In `docs/launch/evidence_registry.yaml`, find the `P0B-API-CONTRACT-FOUNDATION` row and update these fields (leave the rest):
```yaml
    status: passing
    verify: "python -m pytest tests/api/test_request_context.py tests/api/test_error_envelope.py tests/api/test_openapi_security_scheme.py tests/api/test_idempotency_contract.py tests/api/test_job_contract.py -q"
    evidence: "api/errors.py, api/request_context.py, api/idempotency.py, api/contracts/jobs.py, docs/launch/api-versioning-policy.md"
    last_verified: "2026-06-26"
```

- [ ] **Step 3: Validate the registry still passes**

Run:
```bash
python -m scripts.launch.evidence_registry --summary
python -m pytest tests/launch/test_evidence_registry.py -q
```
Expected: summary shows `passing 5` (was 4); registry tests green.

- [ ] **Step 4: Commit**

```bash
git add docs/launch/api-versioning-policy.md docs/launch/evidence_registry.yaml
git commit -m "docs(launch): API versioning policy + mark P0B contract foundation passing (Phase 0b)"
```

---

## Task 7: Final verification + push

**Files:** none (verification + integration)

- [ ] **Step 1: Run the full api suite + the new contract tests**

Run: `python -m pytest tests/api -q`
Expected: PASS (the prior ~281+ api tests plus the new Phase-0b tests; no failures). If a pre-existing test asserted a raw 500 body, it was updated in Task 2 Step 6.

- [ ] **Step 2: Refresh the baseline (operation count may have changed)**

Run:
```bash
python -m scripts.launch.freeze_baseline
git add docs/launch/baseline/baseline.md
git diff --cached --quiet || git commit -m "docs(launch): refresh baseline after Phase 0b (Phase 0b)"
```

- [ ] **Step 3: Run the structural-guard subset (no regression)**

Run: `python -m pytest tests/ -k "no_ or guard or launch" --ignore=tests/test_cheat_sheet.py -q`
Expected: PASS.

- [ ] **Step 4: Push (reconcile first, pre-push runs the structural suite)**

Run:
```bash
git pull --no-rebase --no-edit origin master
git push origin master
```
Expected: pre-push structural suite PASSES; push succeeds. Never `--no-verify`.

---

## Self-review notes

- **Spec coverage (Phase 0 API-contract-foundation):** error envelope → Task 2; correlation ID → Task 1; OpenAPI bearer scheme → Task 3; versioning policy → Task 6; idempotency framework contract → Task 4; async-job contract shape → Task 5. The internal gate ("error envelope + correlation ID + OpenAPI security scheme live + test-enforced") is met by Tasks 1–3.
- **Backward-compat:** verified against `web/src/lib/api/client.ts` (status-only error handling). Only additive live changes: 500-body envelope, `X-Request-ID` header, OpenAPI `BearerAuth` metadata. Status codes 401/402/409 and the 422 validation shape are untouched.
- **No placeholders:** every step has exact code or commands. Idempotency/async-jobs are explicitly contracts-only with enforcement assigned to later phases (not a vague deferral).
- **Type/name consistency:** `set_request_id`/`get_request_id`/`REQUEST_ID_HEADER` (request_context) used identically in main.py + handlers; `HeaterError`/`install_error_handlers`/`ErrorBody` (errors) consistent across the module + tests; `JobStatus`/`JobResource` consistent; `InMemoryIdempotencyStore`/`IDEMPOTENCY_HEADER` consistent.
```
