# HEATER Backend API — Slice 1 (B0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a runnable FastAPI service that serves HEATER's "My Team" data as a typed JSON endpoint by calling the existing Python engines — establishing the contract-first, logic-free API pattern the rest of Sub-project B reuses.

**Architecture:** A new `api/` package (FastAPI) sits *beside* the untouched Streamlit app and Python engines. Routers are thin (auth/validate/serialize only); all data work happens in a `services/` layer that calls the existing engines (`src/...`) and maps their output to Pydantic **contract** models. Tests inject a fake service via FastAPI `dependency_overrides`, so they never touch the DB. Still reads the current SQLite via existing code — Postgres is Slice B2.

**Tech Stack:** FastAPI · Pydantic v2 · Starlette `TestClient` · pytest · the existing `src/` engines. (No Postgres, workers, or auth yet — later slices.)

**Reconciles with:** `docs/superpowers/specs/2026-06-16-heater-backend-api-foundation-design.md` (B) and the CMO's `...-frontend-foundation-design.md` (A). The Pydantic models here are the canonical contract; A's `web/lib/data/types.ts` is generated from this API's OpenAPI (Task 6), resolving the chicken-and-egg (A's types don't exist yet).

---

## File Structure

| File | Responsibility |
|---|---|
| `api/__init__.py` | Marks the package. |
| `api/main.py` | FastAPI app factory + `/healthz`; mounts routers. |
| `api/contracts/__init__.py` | Package marker. |
| `api/contracts/my_team.py` | Pydantic response models for the My Team page (the contract). |
| `api/services/__init__.py` | Package marker. |
| `api/services/team_service.py` | `get_my_team(team_name)` — calls existing engines, returns a `MyTeamResponse`. The ONE place engine code is touched. |
| `api/deps.py` | DI providers (`get_team_service`) so tests can override. |
| `api/routers/team.py` | `GET /api/me/team` — thin router; no analytics logic. |
| `scripts/export_openapi.py` | Dumps the OpenAPI schema to `api/openapi.json` (A's contract source). |
| `tests/api/conftest.py` | `TestClient` + a `FakeTeamService` fixture. |
| `tests/api/test_healthz.py` | Health endpoint test. |
| `tests/api/test_me_team.py` | Contract + endpoint behavior tests. |
| `tests/api/test_no_logic_in_routers.py` | Structural guard: no analytics math in `api/routers/`. |
| `tests/api/test_openapi_contract.py` | Snapshot guard: `api/openapi.json` matches the live schema. |
| `requirements.txt` | Add `fastapi`, `httpx` (TestClient dep). |

---

### Task 1: FastAPI app skeleton + health check

**Files:**
- Create: `api/__init__.py`, `api/main.py`, `tests/api/conftest.py`, `tests/api/test_healthz.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add deps**

Append to `requirements.txt`:
```
fastapi>=0.115
httpx>=0.27        # required by starlette.testclient.TestClient
```
Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

Create `tests/api/conftest.py`:
```python
import pytest
from starlette.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
```

Create `tests/api/test_healthz.py`:
```python
def test_healthz_returns_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/api/test_healthz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 4: Write minimal implementation**

Create `api/__init__.py` (empty).

Create `api/main.py`:
```python
"""FastAPI app factory for the HEATER backend API (Sub-project B, Slice 1).

Thin transport layer over the existing Python engines. Routers do auth/
validate/serialize only — all data work lives in api/services/.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="HEATER API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    from api.routers.team import router as team_router

    app.include_router(team_router)
    return app
```

> Note: the `team_router` import will fail until Task 3 creates it. Temporarily comment the last two lines of `create_app()` to run Task 1 green, then uncomment in Task 3, Step 4.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/api/test_healthz.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add api/__init__.py api/main.py tests/api/conftest.py tests/api/test_healthz.py requirements.txt
git commit -m "feat(api): FastAPI app skeleton + /healthz (B-slice1 task 1)"
```

---

### Task 2: The My Team contract (Pydantic models)

**Files:**
- Create: `api/contracts/__init__.py`, `api/contracts/my_team.py`, add to `tests/api/test_me_team.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_me_team.py`:
```python
from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse


def test_my_team_contract_shape():
    resp = MyTeamResponse(
        team_name="Team Hickey",
        record="4-7-1",
        rank=10,
        matchup=MatchupHero(
            opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35
        ),
        categories=[
            CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18, inverse=False)
        ],
    )
    # win/tie/loss must sum to ~1
    m = resp.matchup
    assert abs((m.win_prob + m.tie_prob + m.loss_prob) - 1.0) < 1e-6
    # round-trips to the JSON shape the frontend consumes
    dumped = resp.model_dump()
    assert dumped["matchup"]["win_prob"] == 0.46
    assert dumped["categories"][0]["cat"] == "SB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_me_team.py::test_my_team_contract_shape -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.contracts'`

- [ ] **Step 3: Write minimal implementation**

Create `api/contracts/__init__.py` (empty).

Create `api/contracts/my_team.py`:
```python
"""Contract models for the My Team page. This is the canonical API contract;
the frontend's web/lib/data/types.ts is generated from the OpenAPI schema
these produce (see scripts/export_openapi.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchupHero(BaseModel):
    opponent: str
    week: int
    win_prob: float = Field(ge=0.0, le=1.0)
    tie_prob: float = Field(ge=0.0, le=1.0)
    loss_prob: float = Field(ge=0.0, le=1.0)


class CategoryLine(BaseModel):
    cat: str
    you: float
    opp: float
    edge: float
    win_prob: float = Field(ge=0.0, le=1.0)
    inverse: bool = False


class MyTeamResponse(BaseModel):
    team_name: str
    record: str
    rank: int
    matchup: MatchupHero | None
    categories: list[CategoryLine]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_me_team.py::test_my_team_contract_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/contracts/__init__.py api/contracts/my_team.py tests/api/test_me_team.py
git commit -m "feat(api): My Team response contract models (B-slice1 task 2)"
```

---

### Task 3: `GET /api/me/team` endpoint (thin router + service + DI override)

**Files:**
- Create: `api/services/__init__.py`, `api/services/team_service.py`, `api/deps.py`, `api/routers/__init__.py`, `api/routers/team.py`
- Modify: `api/main.py` (uncomment router include), `tests/api/test_me_team.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_me_team.py`:
```python
from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse
from api.deps import get_team_service
from api.main import create_app
from starlette.testclient import TestClient


class _FakeTeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        return MyTeamResponse(
            team_name=team_name,
            record="4-7-1",
            rank=10,
            matchup=MatchupHero(
                opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35
            ),
            categories=[CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18)],
        )


def test_get_me_team_returns_contract():
    app = create_app()
    app.dependency_overrides[get_team_service] = lambda: _FakeTeamService()
    client = TestClient(app)
    resp = client.get("/api/me/team", params={"team_name": "Team Hickey"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["rank"] == 10
    assert body["matchup"]["opponent"] == "Baty Babies"
    assert body["categories"][0]["cat"] == "SB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_me_team.py::test_get_me_team_returns_contract -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.deps'`

- [ ] **Step 3: Write minimal implementation**

Create `api/services/__init__.py` (empty) and `api/routers/__init__.py` (empty).

Create `api/services/team_service.py`:
```python
"""The ONE module in the API package allowed to call the engines.

Maps existing engine output → the My Team contract. Engine calls mirror the
canonical signatures in CLAUDE.md (get_yahoo_data_service, load_player_pool,
resolve_viewer_team_name, MatchupContextService). Kept resilient: any missing
live data degrades to an empty/None field rather than raising."""

from __future__ import annotations

from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse


class TeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        from src.valuation import LeagueConfig
        from src.yahoo_data_service import get_yahoo_data_service

        yds = get_yahoo_data_service()
        standings = yds.get_standings()  # pd.DataFrame
        rank, record = self._rank_and_record(standings, team_name)
        matchup = self._matchup(yds.get_matchup(), LeagueConfig())
        categories = self._categories(yds.get_matchup(), LeagueConfig())
        return MyTeamResponse(
            team_name=team_name,
            record=record,
            rank=rank,
            matchup=matchup,
            categories=categories,
        )

    @staticmethod
    def _rank_and_record(standings, team_name: str) -> tuple[int, str]:
        # standings carries per-category rows; the WINS category holds the W-L.
        try:
            wins = standings[(standings["team_name"] == team_name) & (standings["category"] == "WINS")]
            rank = int(wins["rank"].iloc[0]) if not wins.empty else 0
            record = str(wins["total"].iloc[0]) if not wins.empty else "0-0-0"
        except Exception:
            rank, record = 0, "0-0-0"
        return rank, record

    @staticmethod
    def _matchup(raw_matchup: dict | None, cfg) -> MatchupHero | None:
        if not raw_matchup:
            return None
        return MatchupHero(
            opponent=str(raw_matchup.get("opponent", "")),
            week=int(raw_matchup.get("week", 0)),
            win_prob=float(raw_matchup.get("win_prob", 0.0)),
            tie_prob=float(raw_matchup.get("tie_prob", 0.0)),
            loss_prob=float(raw_matchup.get("loss_prob", 0.0)),
        )

    @staticmethod
    def _categories(raw_matchup: dict | None, cfg) -> list[CategoryLine]:
        if not raw_matchup or "categories" not in raw_matchup:
            return []
        inverse = set(cfg.inverse_stats)
        out: list[CategoryLine] = []
        for c in raw_matchup["categories"]:
            cat = str(c.get("cat", ""))
            you = float(c.get("you", 0.0))
            opp = float(c.get("opp", 0.0))
            out.append(
                CategoryLine(
                    cat=cat,
                    you=you,
                    opp=opp,
                    edge=you - opp,
                    win_prob=float(c.get("win_prob", 0.0)),
                    inverse=cat in inverse,
                )
            )
        return out
```

> The exact keys on `get_matchup()` are adapted here; if a key differs at integration time, fix the mapping in this service only — the contract and router stay fixed. This is the intended seam.

Create `api/deps.py`:
```python
"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

from api.services.team_service import TeamService


def get_team_service() -> TeamService:
    return TeamService()
```

Create `api/routers/team.py`:
```python
"""My Team router. THIN: depends on the service, returns its contract output.
No analytics/category math here (guarded by test_no_logic_in_routers)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.my_team import MyTeamResponse
from api.deps import get_team_service
from api.services.team_service import TeamService

router = APIRouter(prefix="/api", tags=["team"])


@router.get("/me/team", response_model=MyTeamResponse)
def get_my_team(team_name: str, service: TeamService = Depends(get_team_service)) -> MyTeamResponse:
    return service.get_my_team(team_name)
```

In `api/main.py`, ensure the last two lines of `create_app()` (the `team_router` import + `include_router`) are uncommented.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/api/ -v`
Expected: PASS (healthz + contract + endpoint)

- [ ] **Step 5: Commit**

```bash
git add api/services api/deps.py api/routers api/main.py tests/api/test_me_team.py
git commit -m "feat(api): GET /api/me/team thin router + service seam (B-slice1 task 3)"
```

---

### Task 4: Structural guard — no analytics logic in routers

**Files:**
- Create: `tests/api/test_no_logic_in_routers.py`

This mirrors the existing `pages/` discipline (logic lives in engines/services, never the presentation/transport layer).

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_no_logic_in_routers.py`:
```python
"""Guard: api/routers/* must stay thin — no analytics math, no engine imports.
All data work belongs in api/services/. Mirrors the pages/ logic-free rule."""

import ast
import pathlib

ROUTERS = pathlib.Path(__file__).resolve().parents[2] / "api" / "routers"


def test_routers_do_not_import_src_engines():
    offenders = []
    for py in ROUTERS.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                offenders.append(f"{py.name}: from {node.module}")
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.startswith("src."):
                        offenders.append(f"{py.name}: import {n.name}")
    assert not offenders, "routers must not import src.* engines directly — use api/services/: " + "; ".join(offenders)


def test_routers_have_no_arithmetic_assignments():
    # crude but effective: no BinOp on the RHS of an assignment in a router
    offenders = []
    for py in ROUTERS.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, "no arithmetic in routers (push math into services/engines): " + "; ".join(offenders)
```

- [ ] **Step 2: Run test to verify it passes (router is already clean)**

Run: `python -m pytest tests/api/test_no_logic_in_routers.py -v`
Expected: PASS — `api/routers/team.py` imports only `api.*` + fastapi, and has no arithmetic. (If it fails, the router has logic that belongs in the service — move it.)

- [ ] **Step 3: Commit**

```bash
git add tests/api/test_no_logic_in_routers.py
git commit -m "test(api): guard routers stay logic-free (B-slice1 task 4)"
```

---

### Task 5: OpenAPI export — the frontend's contract source

**Files:**
- Create: `scripts/export_openapi.py`, `tests/api/test_openapi_contract.py`
- Generates: `api/openapi.json`

- [ ] **Step 1: Write the export script**

Create `scripts/export_openapi.py`:
```python
"""Dump the FastAPI OpenAPI schema to api/openapi.json. The frontend
(Sub-project A) generates web/lib/data/types.ts from this file, so the
API contract has exactly one source of truth. Run after any contract change."""

from __future__ import annotations

import json
import pathlib

from api.main import create_app

OUT = pathlib.Path(__file__).resolve().parents[1] / "api" / "openapi.json"


def main() -> None:
    schema = create_app().openapi()
    OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the schema**

Run: `python scripts/export_openapi.py`
Expected: writes `api/openapi.json`, prints the path.

- [ ] **Step 3: Write the drift-guard test**

Create `tests/api/test_openapi_contract.py`:
```python
"""Guard: the committed api/openapi.json matches the live schema. If this
fails, a contract changed without regenerating — run scripts/export_openapi.py
and commit, then tell the frontend to regenerate types.ts."""

import json
import pathlib

from api.main import create_app

SCHEMA = pathlib.Path(__file__).resolve().parents[2] / "api" / "openapi.json"


def test_openapi_snapshot_is_current():
    live = json.loads(json.dumps(create_app().openapi(), sort_keys=True))
    committed = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert committed == live, "api/openapi.json is stale — run `python scripts/export_openapi.py` and commit"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_openapi_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/export_openapi.py api/openapi.json tests/api/test_openapi_contract.py
git commit -m "feat(api): OpenAPI export as the frontend contract source (B-slice1 task 5)"
```

---

### Task 6: Full-suite sanity + lint

**Files:** none (verification task)

- [ ] **Step 1: Run the new API suite + ruff**

Run: `python -m pytest tests/api/ -v`
Expected: PASS (all tasks' tests)

Run: `python -m ruff format api/ scripts/export_openapi.py tests/api/ && python -m ruff check api/ scripts/export_openapi.py tests/api/`
Expected: formatted; all checks pass.

- [ ] **Step 2: Confirm the engines are untouched**

Run: `git diff --name-only origin/master -- src/ | wc -l`
Expected: `0` — Slice 1 adds `api/` only; no `src/` engine file changed. (The service *calls* engines; it doesn't modify them.)

- [ ] **Step 3: Commit any formatting**

```bash
git add -A api/ tests/api/ scripts/export_openapi.py
git commit -m "chore(api): ruff format B-slice1" || echo "nothing to format"
```

---

## Self-Review

**1. Spec coverage (against `2026-06-16-heater-backend-api-foundation-design.md`):**
- §4.1 thin API / "no logic in routers" → Task 4 guard. ✓
- §2 contract seam (== A's `types.ts`) → Tasks 2 + 5 (OpenAPI is the single source). ✓
- §5 B0 (contract + FastAPI returning the foundation-page data) → Tasks 1–3 (My Team). ✓
- Slice boundary: B1 (Free Agents + Lineup-optimize endpoints), B2 (Postgres), B3 (workers), B4 (auth/multi-tenancy), B5 (strangler proof) are explicitly **out of this plan** — each is its own spec→plan. Noted at top. ✓
- "engines unchanged" → Task 6 Step 2 asserts zero `src/` diff. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. ✓

**3. Type consistency:** `MyTeamResponse` / `MatchupHero` / `CategoryLine` and `get_team_service` / `TeamService.get_my_team` are used identically across Tasks 2, 3, 5. ✓

> Known integration risk (flagged, not a placeholder): the exact key names on `get_matchup()` are adapted in `team_service.py`; if they differ at wiring time, the fix is confined to that service method — the contract, router, and tests are stable. This is the intended seam, not a gap.

---

## What this slice deliberately leaves for later
- **B1:** `GET /api/free-agents` + `POST /api/lineup/optimize` (same pattern; the optimize endpoint becomes a job in B3).
- **B2:** Postgres + repository abstraction + Alembic (replaces SQLite).
- **B3:** Redis/Arq workers (refresh + heavy compute as jobs).
- **B4:** Clerk auth + `tenant_id` scoping + `LeagueConnector` interface.
- **B5:** point the live Streamlit app at this API (strangler-fig proof).
