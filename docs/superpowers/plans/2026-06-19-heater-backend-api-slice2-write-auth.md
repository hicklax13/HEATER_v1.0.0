# Backend API Slice 2 — Auth-Gate the Write Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach a real, testable authentication gate to the two dormant write endpoints (`POST /api/lineup/set`, `POST /api/transactions/add-drop`) via a self-contained auth seam, structured so Clerk drops in at B4 by swapping one provider — no router change.

**Architecture:** A new `api/auth.py` holds the entire seam: a `Principal` model, an `AuthVerifier` Protocol, a deny-by-default `EnvTokenVerifier` (static bearer token vs the `HEATER_API_WRITE_TOKEN` env var), a `get_auth_verifier()` DI provider, and a `require_principal()` FastAPI dependency. The mutation router attaches `dependencies=[Depends(require_principal)]` to both write routes (handlers unchanged → stays logic-free, AST-guarded). Reads and all other routers are untouched. At B4, Clerk replaces `EnvTokenVerifier` by swapping what `get_auth_verifier()` returns; `require_principal` and the routers never change. The API write endpoints are dormant (live Streamlit writes do not go through the API), so this touches no live data path.

**Tech Stack:** FastAPI, Pydantic v2, `hmac.compare_digest`, pytest + `monkeypatch`, Starlette `TestClient`.

---

## Conventions for every command below

- Run all commands from the **worktree root**: `C:\Users\conno\Code\HEATER_v1.0.1\.claude\worktrees\api-slice2-write-auth`
- The worktree has no `.venv`; use the **main checkout's** interpreter. `PY` below means:
  `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`
- Lint after touching any `.py`: `$PY -m ruff format <files>` then `$PY -m ruff check <files>`.

## File Structure (what each file is responsible for)

- **Create** `api/auth.py` — the entire write-auth seam (Principal, AuthVerifier Protocol, `_bearer`, EnvTokenVerifier, get_auth_verifier, require_principal). Self-contained: imports nothing from `api.deps` (avoids an import cycle). The ONE place write auth lives.
- **Modify** `api/routers/roster_write.py` — import `require_principal`; add `dependencies=[Depends(require_principal)]` to both routes. Handler bodies unchanged.
- **Modify** `tests/api/test_api_roster_write.py` — the 3 endpoint tests' shared `_client_with_fake()` helper must also override `get_auth_verifier` (else the new gate 401s them).
- **Create** `tests/api/test_api_roster_write_auth.py` — verifier unit tests + endpoint 401/200 tests + verifier-override (Clerk-seam) test + reads-not-gated guard.
- **Modify** `api/openapi.json` — regenerated (the `authorization` header param now appears on the two write routes).

No `api/deps.py` change (the provider lives in `api/auth.py`). No `api/main.py` change (the router is already mounted). No service change (gate is at the router/dependency layer).

---

### Task 1: The auth seam (`api/auth.py`)

**Files:**
- Create: `api/auth.py`
- Test: `tests/api/test_api_roster_write_auth.py`

- [ ] **Step 1: Write the failing verifier unit tests**

Create `tests/api/test_api_roster_write_auth.py`:

```python
import pytest
from fastapi import HTTPException

from api.auth import EnvTokenVerifier, Principal, _bearer

_ENV = "HEATER_API_WRITE_TOKEN"


def test_bearer_parsing():
    assert _bearer(None) is None
    assert _bearer("") is None
    assert _bearer("Bearer abc") == "abc"
    assert _bearer("bearer abc") == "abc"  # scheme is case-insensitive
    assert _bearer("Bearer   ") is None  # empty token after scheme
    assert _bearer("Token abc") is None  # wrong scheme
    assert _bearer("abc") is None  # no scheme


def test_verify_denies_by_default_when_secret_unset(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        EnvTokenVerifier().verify("Bearer anything")
    assert ei.value.status_code == 401
    assert "not configured" in ei.value.detail.lower()


def test_verify_rejects_missing_and_bad_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    for header in (None, "", "Bearer", "Bearer wrong", "Token s3cret"):
        with pytest.raises(HTTPException) as ei:
            EnvTokenVerifier().verify(header)
        assert ei.value.status_code == 401


def test_verify_accepts_matching_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    principal = EnvTokenVerifier().verify("Bearer s3cret")
    assert isinstance(principal, Principal)
    assert principal.subject == "api-token"
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/api/test_api_roster_write_auth.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.auth'`.

- [ ] **Step 3: Implement `api/auth.py`**

Create `api/auth.py`:

```python
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
```

- [ ] **Step 4: Run to verify the unit tests pass**

Run: `$PY -m pytest tests/api/test_api_roster_write_auth.py -q`
Expected: 4 passed.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format api/auth.py tests/api/test_api_roster_write_auth.py
$PY -m ruff check api/auth.py tests/api/test_api_roster_write_auth.py
git add api/auth.py tests/api/test_api_roster_write_auth.py
git commit -m "feat(api): add write-auth seam (Principal + EnvTokenVerifier + require_principal) (slice 2)"
```

---

### Task 2: Gate the write routes + endpoint tests + fix the existing write tests

**Files:**
- Modify: `api/routers/roster_write.py`
- Modify: `tests/api/test_api_roster_write.py` (the `_client_with_fake()` helper)
- Test: `tests/api/test_api_roster_write_auth.py` (append endpoint tests)

- [ ] **Step 1: Write the failing endpoint auth tests**

Append to `tests/api/test_api_roster_write_auth.py`:

```python
from starlette.testclient import TestClient

from api.auth import get_auth_verifier
from api.contracts.roster_write import MutationResult
from api.deps import get_roster_write_service
from api.main import create_app


class _RecordingWriteService:
    def __init__(self):
        self.called = False

    def set_lineup(self, req) -> MutationResult:
        self.called = True
        return MutationResult(ok=True, applied=len(req.assignments))

    def add_drop(self, req) -> MutationResult:
        self.called = True
        return MutationResult(ok=True)


class _AlwaysVerifier:
    def verify(self, authorization):
        return Principal(subject="test")


_LINEUP_BODY = {
    "team_name": "Team Hickey",
    "date": "2027-04-05",
    "assignments": [{"yahoo_player_key": "469.p.1", "slot": "SS"}],
}


def test_write_denied_without_token_and_service_not_called(monkeypatch):
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    app = create_app()
    fake = _RecordingWriteService()
    app.dependency_overrides[get_roster_write_service] = lambda: fake
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY)
    assert resp.status_code == 401
    assert fake.called is False  # the gate runs before the handler


def test_write_denied_with_bad_token(monkeypatch):
    monkeypatch.setenv("HEATER_API_WRITE_TOKEN", "s3cret")
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_write_allowed_with_env_token(monkeypatch):
    monkeypatch.setenv("HEATER_API_WRITE_TOKEN", "s3cret")
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY, headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_write_allowed_via_verifier_override(monkeypatch):
    # The Clerk-at-B4 seam: overriding the verifier authenticates without env.
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    app.dependency_overrides[get_auth_verifier] = lambda: _AlwaysVerifier()
    client = TestClient(app)
    resp = client.post("/api/transactions/add-drop", json={"add_player_key": "469.p.9"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_reads_are_not_auth_gated(monkeypatch):
    # The gate is writes-only: a read endpoint answers without a token
    # (any non-401 status proves it is not behind require_principal).
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    client = TestClient(create_app())
    resp = client.get("/api/standings")
    assert resp.status_code != 401
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/api/test_api_roster_write_auth.py -q`
Expected: FAIL — `test_write_denied_without_token_and_service_not_called` and `test_write_denied_with_bad_token` get 200 (no gate yet) instead of 401. (The other new tests may pass incidentally.)

- [ ] **Step 3: Gate the routes (`api/routers/roster_write.py`)**

Replace the file body so both routes carry the dependency (note the new import + `dependencies=[...]`; handler bodies are unchanged):

```python
"""Roster write-back router — the SINGLE mutation entry point.

THIN: validate → delegate to the service → return its MutationResult. ALL
mutation endpoints live here so auth + Pro-tier gating + audit attach in one
place. Slice 2 attaches auth (require_principal); Pro-tier + audit follow at B4.
No engine imports, no logic (guarded by tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_principal
from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult
from api.deps import get_roster_write_service

router = APIRouter(prefix="/api", tags=["roster-write"])


@router.post("/lineup/set", response_model=MutationResult, dependencies=[Depends(require_principal)])
def set_lineup(req: LineupSetRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.set_lineup(req)


@router.post("/transactions/add-drop", response_model=MutationResult, dependencies=[Depends(require_principal)])
def add_drop(req: AddDropRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.add_drop(req)
```

- [ ] **Step 4: Fix the existing endpoint tests (`tests/api/test_api_roster_write.py`)**

The 3 endpoint tests there POST without auth and will now 401. Update their shared helper to also override the verifier. Add this import near the other `api.` imports at the top of the file:

```python
from api.auth import Principal, get_auth_verifier
```

Add this class just above the `_client_with_fake` definition:

```python
class _AlwaysVerifier:
    def verify(self, authorization):
        return Principal(subject="test")
```

Then change `_client_with_fake()` from:

```python
def _client_with_fake():
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _FakeWriteService()
    return TestClient(app)
```

to:

```python
def _client_with_fake():
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _FakeWriteService()
    app.dependency_overrides[get_auth_verifier] = lambda: _AlwaysVerifier()
    return TestClient(app)
```

- [ ] **Step 5: Run both write-test files + the no-logic guard to verify all pass**

Run: `$PY -m pytest tests/api/test_api_roster_write_auth.py tests/api/test_api_roster_write.py tests/api/test_no_logic_in_routers.py -q`
Expected: all green — the auth file's ~9 tests, the existing file's tests (3 endpoint tests now authed), and the 2 no-logic guards.

- [ ] **Step 6: Lint + commit**

```bash
$PY -m ruff format api/routers/roster_write.py tests/api/test_api_roster_write.py tests/api/test_api_roster_write_auth.py
$PY -m ruff check api/routers/roster_write.py tests/api/test_api_roster_write.py tests/api/test_api_roster_write_auth.py
git add api/routers/roster_write.py tests/api/test_api_roster_write.py tests/api/test_api_roster_write_auth.py
git commit -m "feat(api): gate write endpoints with require_principal (slice 2)"
```

---

### Task 3: Regenerate the OpenAPI snapshot

**Files:**
- Modify: `api/openapi.json`

- [ ] **Step 1: Confirm the snapshot is now stale (failing)**

Run: `$PY -m pytest tests/api/test_openapi_contract.py -q`
Expected: FAIL — "api/openapi.json is stale" (the two write routes gained an `authorization` header parameter).

- [ ] **Step 2: Regenerate**

Run: `$PY scripts/export_openapi.py`
Expected: prints `wrote .../api/openapi.json`.

- [ ] **Step 3: Verify the snapshot test passes**

Run: `$PY -m pytest tests/api/test_openapi_contract.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regen openapi.json for write-endpoint auth header (slice 2)"
```

---

### Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole API suite**

Run: `$PY -m pytest tests/api/ -q`
Expected: all green — 66 prior + the new auth tests, 0 failures.

- [ ] **Step 2: Lint the full changeset**

Run:
```bash
$PY -m ruff format --check api/auth.py api/routers/roster_write.py tests/api/test_api_roster_write.py tests/api/test_api_roster_write_auth.py
$PY -m ruff check api/auth.py api/routers/roster_write.py tests/api/test_api_roster_write.py tests/api/test_api_roster_write_auth.py
```
Expected: "All checks passed!" / nothing to reformat.

- [ ] **Step 3: Confirm clean tree + commits**

Run: `git status --short && git log --oneline 2f66c87..HEAD`
Expected: clean working tree; 3 slice-2 commits (auth seam, gate, openapi) on `worktree-api-slice2-write-auth`.

Hand back to the orchestrator for the 2-stage review.

---

## Self-Review

**1. Spec coverage:**
- Self-contained auth seam (`api/auth.py`): Principal, AuthVerifier, EnvTokenVerifier, get_auth_verifier, require_principal → Task 1. ✅
- Deny-by-default + env bearer token + constant-time compare → `EnvTokenVerifier.verify` (Task 1). ✅
- Gate ONLY the two write routes; reads untouched → Task 2 router change + `test_reads_are_not_auth_gated`. ✅
- Router stays logic-free → Task 2 step 5 runs `test_no_logic_in_routers.py`. ✅
- Existing slice-7 endpoint tests stay green → Task 2 step 4. ✅
- Clerk-at-B4 swap proven by `test_write_allowed_via_verifier_override` (override the provider, no env). ✅
- OpenAPI regen → Task 3. ✅
- No `api/deps.py`/`api/main.py`/service change (seam self-contained, router mounted, gate at dependency layer). ✅

**2. Placeholder scan:** none — every code/test step shows complete code and exact commands.

**3. Type consistency:** `require_principal(authorization, verifier) -> Principal`; `AuthVerifier.verify(authorization) -> Principal`; `EnvTokenVerifier.verify` and the test `_AlwaysVerifier.verify` share the exact `verify(self, authorization)` signature so the `get_auth_verifier` override is type-compatible. `Principal(subject=...)` constructed identically in impl and tests. `get_auth_verifier` is the single override key used in both test files.
