# Backend API — Slice 7: Write-back endpoints (set-lineup + add/drop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two dormant write-back endpoints — `POST /api/lineup/set` and `POST /api/transactions/add-drop` — over the existing graceful `YahooFantasyClient.set_lineup` / `add_drop` methods, following the same contract-first pattern as read slices 1–6.

**Architecture:** Contract (`api/contracts/`) → service seam (`api/services/roster_write_service.py`, the ONE place that touches `src/`) → thin logic-free router (`api/routers/roster_write.py`, the SINGLE mutation entry point so auth/gating attaches in one place at B4) → DI provider (`api/deps.py`) → fake-service dependency-override tests → mount in `api/main.py` → regenerate `api/openapi.json`. The `src/` engines and the live Streamlit app are UNCHANGED; the endpoints are mounted but dormant (the FastAPI app is not deployed).

**Tech Stack:** FastAPI, Pydantic v2, Starlette TestClient, pytest.

---

## Environment (READ FIRST)

- **Working dir:** this worktree (`.claude/worktrees/feat+api-slice7-writeback`). It shares `.git` with the main checkout but has NO `.venv`.
- **Python / test runner:** use the main checkout's venv:
  `PY="/c/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe"`
  Run tests as `$PY -m pytest …`, ruff as `$PY -m ruff …`, the export script as `$PY scripts/export_openapi.py`.
- **Baseline (already verified):** `$PY -m pytest tests/api/ -q` → 31 pass, 1 pre-existing fail (`test_openapi_contract.py::test_openapi_snapshot_is_current`). That failure is benign fastapi-0.137.1 version drift (the auto-generated `ValidationError` schema gained `ctx`/`input`); **Task 4 regenerates `openapi.json` and clears it.** Do not be alarmed by it before Task 4.
- **New api test files MUST use the `test_api_*` basename** (coordination rule — a non-`test_api_` name can collide with a `tests/` file of the same basename and only the full suite surfaces it).

---

## Design decisions (locked with the owner — do not deviate)

1. **Player identity = Yahoo `player_key`** is authoritative in the write contract (thin pass-through; no fragile server-side id→key mapping). An optional `yahoo_player_key` field is added to the shared `PlayerRef` so read responses can carry it later. **Populating that field on read responses is OUT OF SCOPE here** (the engine output shapes the read services map from don't carry Yahoo keys; reliable population needs the raw Yahoo feeds + live verification — deferred to the frontend write-seam slice).
2. **Confirm gate lives in the UI**, not the API. The endpoints are thin pass-throughs to the already-graceful client methods.
3. **All mutation outcomes return HTTP 200**; success/failure is carried in `MutationResult.ok`. The underlying client methods never raise — they return `{"ok": bool, "error"?, "status"?, "applied"?}`. (A both-null add/drop is `ok=False`, NOT an HTTP 422 — keep the router/contract logic-free.)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `api/contracts/common.py` | modify | add optional `yahoo_player_key` to `PlayerRef` |
| `api/contracts/roster_write.py` | create | `LineupAssignment`, `LineupSetRequest`, `AddDropRequest`, `MutationResult` |
| `api/services/roster_write_service.py` | create | the ONE write-path module touching `src/`; thin pass-through + dict→`MutationResult` mapping |
| `api/routers/roster_write.py` | create | single mutation router; 2 thin POST routes |
| `api/deps.py` | modify | `get_roster_write_service()` provider |
| `api/main.py` | modify | mount the roster-write router |
| `tests/api/test_api_roster_write.py` | create | contract-shape + service-unit (fake client) + endpoint (fake-service override) tests |
| `api/openapi.json` | regenerate | frontend's type source — regenerated in Task 4 |

---

## Task 1: Contracts

**Files:**
- Modify: `api/contracts/common.py`
- Create: `api/contracts/roster_write.py`
- Test: `tests/api/test_api_roster_write.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_api_roster_write.py` with ONLY the contract tests for now:

```python
from api.contracts.common import PlayerRef
from api.contracts.roster_write import (
    AddDropRequest,
    LineupAssignment,
    LineupSetRequest,
    MutationResult,
)


def test_player_ref_yahoo_key_is_optional_and_defaults_none():
    p = PlayerRef(id=1, name="A. Player", positions="OF")
    assert p.yahoo_player_key is None
    p2 = PlayerRef(id=1, name="A. Player", positions="OF", yahoo_player_key="469.p.1")
    assert p2.model_dump()["yahoo_player_key"] == "469.p.1"


def test_lineup_set_request_shape():
    req = LineupSetRequest(
        team_name="Team Hickey",
        date="2027-04-05",
        assignments=[
            LineupAssignment(yahoo_player_key="469.p.1", slot="SS", player_id=1),
            LineupAssignment(yahoo_player_key="469.p.2", slot="BN"),
        ],
    )
    dumped = req.model_dump()
    assert dumped["assignments"][0]["yahoo_player_key"] == "469.p.1"
    assert dumped["assignments"][1]["player_id"] is None


def test_add_drop_request_allows_partial():
    assert AddDropRequest(add_player_key="469.p.9").drop_player_key is None
    assert AddDropRequest(drop_player_key="469.p.3").add_player_key is None
    assert AddDropRequest().add_player_key is None  # both-null is a valid shape; service returns ok=False


def test_mutation_result_shape():
    ok = MutationResult(ok=True, applied=2)
    assert ok.model_dump() == {"ok": True, "applied": 2, "error": None, "status": None}
    fail = MutationResult(ok=False, error="denied", status=403)
    assert fail.ok is False and fail.status == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.contracts.roster_write'` (and `PlayerRef` has no `yahoo_player_key`).

- [ ] **Step 3a: Add the field to `PlayerRef`**

In `api/contracts/common.py`, change the `PlayerRef` class to:

```python
class PlayerRef(BaseModel):
    id: int
    name: str
    positions: str
    yahoo_player_key: str | None = None
```

- [ ] **Step 3b: Create `api/contracts/roster_write.py`**

```python
"""Contract models for roster write-back (set-lineup + add/drop).

All mutation outcomes return HTTP 200; success/failure is carried in the `ok`
field of MutationResult — the underlying YahooFantasyClient methods never raise,
they return {"ok": ...}. The frontend owns the confirm dialog."""

from __future__ import annotations

from pydantic import BaseModel


class LineupAssignment(BaseModel):
    yahoo_player_key: str          # authoritative Yahoo id, e.g. "469.p.11111"
    slot: str                      # Yahoo slot token: "SS","OF","BN","SP","Util","IL"…
    player_id: int | None = None   # optional HEATER id, echoed for display/audit


class LineupSetRequest(BaseModel):
    team_name: str                 # echoed; today the write targets the authenticated team
    date: str                      # "YYYY-MM-DD"
    assignments: list[LineupAssignment]


class AddDropRequest(BaseModel):
    add_player_key: str | None = None    # Yahoo player_key to add, or None
    drop_player_key: str | None = None   # Yahoo player_key to drop, or None
    # ≥1 expected; the client returns ok=False if both are None (no validator — keep it thin)


class MutationResult(BaseModel):
    ok: bool
    applied: int | None = None     # set-lineup: number of assignments applied
    error: str | None = None       # graceful Yahoo message (incl. write-scope re-auth)
    status: int | None = None      # Yahoo HTTP status on failure (e.g. 401/403)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add api/contracts/common.py api/contracts/roster_write.py tests/api/test_api_roster_write.py
git commit -m "feat(api): write-back contracts (set-lineup + add/drop) — B-slice7 task 1"
```

---

## Task 2: RosterWriteService (the src/ seam)

**Files:**
- Create: `api/services/roster_write_service.py`
- Test: `tests/api/test_api_roster_write.py` (append)

- [ ] **Step 1: Write the failing test**

Add this import to the TOP import block of `tests/api/test_api_roster_write.py` (keep all imports at the top — do NOT place imports below the test functions, or ruff E402/F811 will fail Task 4):

```python
from api.services.roster_write_service import RosterWriteService
```

Then append the following helpers + tests BELOW the existing tests (the contract names `AddDropRequest`/`LineupAssignment`/`LineupSetRequest` are already imported from Task 1 — do not re-import them):

```python
class _FakeClient:
    """Stand-in for YahooFantasyClient — records calls, returns canned dicts."""

    def __init__(self, lineup_ret=None, addrop_ret=None):
        self._lineup_ret = lineup_ret if lineup_ret is not None else {"ok": True, "applied": 2}
        self._addrop_ret = addrop_ret if addrop_ret is not None else {"ok": True}
        self.last_assignments = None
        self.last_date = None
        self.last_add = None
        self.last_drop = None

    def set_lineup(self, assignments, coverage_date):
        self.last_assignments = assignments
        self.last_date = coverage_date
        return self._lineup_ret

    def add_drop(self, add_player_key, drop_player_key):
        self.last_add = add_player_key
        self.last_drop = drop_player_key
        return self._addrop_ret


def _lineup_req():
    return LineupSetRequest(
        team_name="Team Hickey",
        date="2027-04-05",
        assignments=[
            LineupAssignment(yahoo_player_key="469.p.1", slot="SS"),
            LineupAssignment(yahoo_player_key="469.p.2", slot="BN"),
        ],
    )


def test_service_set_lineup_maps_and_passes_through():
    fake = _FakeClient()
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is True and result.applied == 2
    # the service translates {yahoo_player_key, slot} -> {player_key, position}
    assert fake.last_assignments == [
        {"player_key": "469.p.1", "position": "SS"},
        {"player_key": "469.p.2", "position": "BN"},
    ]
    assert fake.last_date == "2027-04-05"


def test_service_add_drop_passes_keys_through():
    fake = _FakeClient(addrop_ret={"ok": True})
    result = RosterWriteService().add_drop(
        AddDropRequest(add_player_key="469.p.9", drop_player_key="469.p.3"), client=fake
    )
    assert result.ok is True
    assert fake.last_add == "469.p.9" and fake.last_drop == "469.p.3"


def test_service_passes_through_write_scope_denial():
    fake = _FakeClient(
        lineup_ret={"ok": False, "error": "Yahoo write access denied — re-authorize.", "status": 403}
    )
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is False and result.status == 403
    assert "re-authorize" in (result.error or "")


def test_service_no_client_is_graceful(monkeypatch):
    svc = RosterWriteService()
    monkeypatch.setattr(svc, "_client", lambda: None)
    result = svc.set_lineup(_lineup_req())
    assert result.ok is False and result.status is None
    assert "Not connected" in (result.error or "")


def test_service_non_dict_response_is_graceful():
    fake = _FakeClient(lineup_ret=None)  # client returned something unexpected
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is False and "Unexpected" in (result.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.services.roster_write_service'`.

- [ ] **Step 3: Create `api/services/roster_write_service.py`**

```python
"""Roster write service — the ONE place in the write path that touches src/.

Thin pass-through to YahooFantasyClient.set_lineup / add_drop (both already
graceful: they return {"ok": bool, "error"?, "status"?, "applied"?} and never
raise). Maps that dict → MutationResult. Reaches the write-capable client via
the YahooDataService singleton's ._client; in a cold/dormant environment that
client is None → returns a graceful "Not connected" result.

NOTE: today the write always targets the authenticated user's team (the client
discovers the team key itself). At B4, tenant/auth resolution selects the team —
this single service is the seam where that lands."""

from __future__ import annotations

from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult

_NOT_CONNECTED = "Not connected to Yahoo."


class RosterWriteService:
    def set_lineup(self, req: LineupSetRequest, client=None) -> MutationResult:
        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        assignments = [{"player_key": a.yahoo_player_key, "position": a.slot} for a in req.assignments]
        return self._to_result(c.set_lineup(assignments, req.date))

    def add_drop(self, req: AddDropRequest, client=None) -> MutationResult:
        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        return self._to_result(c.add_drop(req.add_player_key, req.drop_player_key))

    @staticmethod
    def _client():
        # The write-capable client is the YahooDataService singleton's wrapped
        # YahooFantasyClient. None in a cold/dormant env (no live session).
        try:
            from src.yahoo_data_service import get_yahoo_data_service

            return get_yahoo_data_service()._client
        except Exception:
            return None

    @staticmethod
    def _to_result(raw) -> MutationResult:
        if not isinstance(raw, dict):
            return MutationResult(ok=False, error="Unexpected response from Yahoo client.", status=None)
        return MutationResult(
            ok=bool(raw.get("ok", False)),
            applied=raw.get("applied"),
            error=raw.get("error"),
            status=raw.get("status"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -q`
Expected: PASS (9 tests total in the file now).

- [ ] **Step 5: Commit**

```bash
git add api/services/roster_write_service.py tests/api/test_api_roster_write.py
git commit -m "feat(api): RosterWriteService seam over Yahoo write methods — B-slice7 task 2"
```

---

## Task 3: Router + DI provider + mount

**Files:**
- Create: `api/routers/roster_write.py`
- Modify: `api/deps.py`
- Modify: `api/main.py`
- Test: `tests/api/test_api_roster_write.py` (append)

- [ ] **Step 1: Write the failing test**

Add these imports to the TOP import block of `tests/api/test_api_roster_write.py` (`MutationResult` is already imported from Task 1 — do not re-import it):

```python
from starlette.testclient import TestClient

from api.deps import get_roster_write_service
from api.main import create_app
```

Then append the following helper + tests BELOW the existing tests:

```python
class _FakeWriteService:
    def set_lineup(self, req) -> MutationResult:
        return MutationResult(ok=True, applied=len(req.assignments))

    def add_drop(self, req) -> MutationResult:
        if not req.add_player_key and not req.drop_player_key:
            return MutationResult(
                ok=False,
                error="Must provide at least one of add_player_key or drop_player_key.",
                status=None,
            )
        return MutationResult(ok=True)


def _client_with_fake():
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _FakeWriteService()
    return TestClient(app)


def test_post_lineup_set_returns_contract():
    client = _client_with_fake()
    resp = client.post(
        "/api/lineup/set",
        json={
            "team_name": "Team Hickey",
            "date": "2027-04-05",
            "assignments": [
                {"yahoo_player_key": "469.p.1", "slot": "SS"},
                {"yahoo_player_key": "469.p.2", "slot": "BN"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["applied"] == 2


def test_post_add_drop_returns_contract():
    client = _client_with_fake()
    resp = client.post(
        "/api/transactions/add-drop",
        json={"add_player_key": "469.p.9", "drop_player_key": "469.p.3"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_post_add_drop_both_null_is_ok_false_not_http_error():
    client = _client_with_fake()
    resp = client.post("/api/transactions/add-drop", json={})
    assert resp.status_code == 200  # graceful: failure is in the body, not the HTTP status
    assert resp.json()["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -k "post_" -q`
Expected: FAIL — `ImportError: cannot import name 'get_roster_write_service'` (and the routes 404 once that's fixed, until the router is mounted).

- [ ] **Step 3a: Create `api/routers/roster_write.py`**

```python
"""Roster write-back router — the SINGLE mutation entry point.

THIN: validate → delegate to the service → return its MutationResult. ALL
mutation endpoints live here so auth + Pro-tier gating + audit can attach in one
place at B4. No engine imports, no logic (guarded by
tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult
from api.deps import get_roster_write_service

router = APIRouter(prefix="/api", tags=["roster-write"])


@router.post("/lineup/set", response_model=MutationResult)
def set_lineup(req: LineupSetRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.set_lineup(req)


@router.post("/transactions/add-drop", response_model=MutationResult)
def add_drop(req: AddDropRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.add_drop(req)
```

- [ ] **Step 3b: Add the DI provider to `api/deps.py`**

Add the import alongside the other service imports (keep alphabetical with the existing block):

```python
from api.services.roster_write_service import RosterWriteService
```

Add the provider at the end of the file:

```python
def get_roster_write_service() -> RosterWriteService:
    return RosterWriteService()
```

- [ ] **Step 3c: Mount the router in `api/main.py`**

Inside `create_app()`, add the import alongside the other router imports:

```python
    from api.routers.roster_write import router as roster_write_router
```

And add the include alongside the other `app.include_router(...)` calls:

```python
    app.include_router(roster_write_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/api/test_api_roster_write.py -q`
Expected: PASS (12 tests total).

- [ ] **Step 5: Commit**

```bash
git add api/routers/roster_write.py api/deps.py api/main.py tests/api/test_api_roster_write.py
git commit -m "feat(api): mount write-back router (POST /lineup/set, /transactions/add-drop) — B-slice7 task 3"
```

---

## Task 4: Regenerate OpenAPI + verify all guards green

**Files:**
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Regenerate the OpenAPI snapshot**

Run: `$PY scripts/export_openapi.py`
Expected: `wrote …/api/openapi.json`.

- [ ] **Step 2: Sanity-check the new endpoints + the version-drift are present**

Run: `git --no-pager diff --stat api/openapi.json` then `grep -E "lineup/set|transactions/add-drop|MutationResult|LineupSetRequest|AddDropRequest" api/openapi.json`
Expected: the diff shows the two new paths + new schemas (`LineupAssignment`, `LineupSetRequest`, `AddDropRequest`, `MutationResult`) AND the small fastapi-0.137.1 `ValidationError` `ctx`/`input` additions noted in Environment. Both are expected.

- [ ] **Step 3: Run the full api suite + structural guards**

Run: `$PY -m pytest tests/api/ -q`
Expected: PASS — all green, including `test_openapi_contract.py::test_openapi_snapshot_is_current` (now current) and `test_no_logic_in_routers.py` (the new router has no `src.` import and no arithmetic).

- [ ] **Step 4: Lint**

Run: `$PY -m ruff check api/ tests/api/ && $PY -m ruff format --check api/ tests/api/`
Expected: no errors. If `ruff format --check` reports diffs, run `$PY -m ruff format api/ tests/api/` and re-run the check.

- [ ] **Step 5: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regenerate OpenAPI for write-back endpoints — B-slice7 task 4

Includes the fastapi 0.137.1 ValidationError ctx/input schema drift
(framework error-model versioning, unrelated to the new endpoints)."
```

---

## Self-Review checklist (run before handing off for review)

1. **Spec coverage** — both endpoints (`/api/lineup/set`, `/api/transactions/add-drop`) built; `PlayerRef` field reserved; read-side population correctly OUT of scope per the locked decision. ✓
2. **Placeholder scan** — every step has concrete code/commands. ✓
3. **Type consistency** — `MutationResult` fields (`ok`/`applied`/`error`/`status`), `LineupAssignment` fields (`yahoo_player_key`/`slot`/`player_id`), and the service→client key mapping (`{player_key, position}`) are identical across contract, service, router, and tests. ✓
4. **Thin-router guard** — router has no `src.` import and no assignments (the list-comprehension mapping lives in the SERVICE). ✓
5. **Final full-suite check (optional but recommended):** `$PY -m pytest tests/api/ -q` green; if time permits, `$PY -m pytest tests/ -k "openapi or roster or no_logic" -q`.
