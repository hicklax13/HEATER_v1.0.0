# HEATER Backend API — Slice 2 (B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add two more read/compute endpoints — `GET /api/free-agents` and `POST /api/lineup/optimize` — over the existing engines, using the exact contract-first, logic-free-router pattern Slice 1 established.

**Architecture:** Same as Slice 1. Each endpoint = a Pydantic contract (`api/contracts/`) + a service that calls the engine and maps to the contract (`api/services/`) + a thin router (`api/routers/`) + a DI provider (`api/deps.py`) + a test that injects a fake service via `app.dependency_overrides`. The existing `api/me/team` files are the canonical template — read them first and mirror them.

**Tech Stack:** FastAPI · Pydantic v2 · pytest · the existing `src/optimizer` (FA recommender, lineup pipeline) engines. Still SQLite; the optimize endpoint is synchronous here (it becomes an Arq job in B3).

**Pattern templates (read these first — mirror their structure exactly):**
- `api/contracts/my_team.py` · `api/services/team_service.py` · `api/routers/team.py` · `api/deps.py` · `tests/api/test_me_team.py`

---

## File Structure

| File | Responsibility |
|---|---|
| `api/contracts/common.py` | Shared `PlayerRef` model (id, name, positions). |
| `api/contracts/free_agents.py` | `FreeAgentRec`, `FreeAgentsResponse`. |
| `api/contracts/lineup.py` | `LineupSlot`, `LineupOptimizeResponse`, `LineupOptimizeRequest`. |
| `api/services/fa_service.py` | `FreeAgentService.get_free_agents(team_name, limit)` — calls the FA recommender, maps to contract. |
| `api/services/lineup_service.py` | `LineupService.optimize(team_name, date, scope)` — calls the optimizer pipeline, maps to contract. |
| `api/routers/free_agents.py` | `GET /api/free-agents` — thin. |
| `api/routers/lineup.py` | `POST /api/lineup/optimize` — thin. |
| `api/deps.py` | ADD `get_fa_service`, `get_lineup_service`. |
| `api/main.py` | ADD the two routers to `create_app()`. |
| `tests/api/test_free_agents.py` | Contract + endpoint test (fake service). |
| `tests/api/test_lineup.py` | Contract + endpoint test (fake service). |
| `api/openapi.json` | Regenerated (new endpoints). |

---

### Task 1: Shared PlayerRef + Free Agents contract

**Files:** Create `api/contracts/common.py`, `api/contracts/free_agents.py`, `tests/api/test_free_agents.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_free_agents.py`:
```python
from api.contracts.common import PlayerRef
from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse


def test_free_agents_contract_shape():
    resp = FreeAgentsResponse(
        team_name="Team Hickey",
        recommendations=[
            FreeAgentRec(
                add=PlayerRef(id=1, name="A. Player", positions="OF"),
                drop=PlayerRef(id=2, name="B. Bench", positions="OF"),
                marginal_value=2.31,
                categories_helped=["SB", "R"],
                ownership_pct=44.0,
                rationale="Adds steals you're behind in.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["recommendations"][0]["add"]["name"] == "A. Player"
    assert dumped["recommendations"][0]["categories_helped"] == ["SB", "R"]
    # drop is optional (roster-grow adds have none)
    assert FreeAgentRec(add=PlayerRef(id=3, name="C", positions="SP"), marginal_value=1.0).drop is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_free_agents.py::test_free_agents_contract_shape -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.contracts.common'`

- [ ] **Step 3: Implement**

Create `api/contracts/common.py`:
```python
"""Contract models shared across pages."""

from __future__ import annotations

from pydantic import BaseModel


class PlayerRef(BaseModel):
    id: int
    name: str
    positions: str
```

Create `api/contracts/free_agents.py`:
```python
"""Contract models for the Free Agents page. The frontend's
web/lib/data/types.ts is generated from the OpenAPI these produce."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class FreeAgentRec(BaseModel):
    add: PlayerRef
    drop: PlayerRef | None = None
    marginal_value: float
    categories_helped: list[str] = []
    ownership_pct: float | None = None
    rationale: str = ""


class FreeAgentsResponse(BaseModel):
    team_name: str
    recommendations: list[FreeAgentRec]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_free_agents.py::test_free_agents_contract_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/contracts/common.py api/contracts/free_agents.py tests/api/test_free_agents.py
git commit -m "feat(api): PlayerRef + Free Agents contract models (B-slice2 task 1)"
```

---

### Task 2: Free Agents service + thin router + endpoint test

**Files:** Create `api/services/fa_service.py`, `api/routers/free_agents.py`; Modify `api/deps.py`, `api/main.py`, `tests/api/test_free_agents.py`

- [ ] **Step 1: Write the failing endpoint test**

Append to `tests/api/test_free_agents.py`:
```python
from api.contracts.common import PlayerRef
from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse
from api.deps import get_fa_service
from api.main import create_app
from starlette.testclient import TestClient


class _FakeFAService:
    def get_free_agents(self, team_name: str, limit: int = 5) -> FreeAgentsResponse:
        return FreeAgentsResponse(
            team_name=team_name,
            recommendations=[
                FreeAgentRec(
                    add=PlayerRef(id=1, name="A. Player", positions="OF"),
                    drop=PlayerRef(id=2, name="B. Bench", positions="OF"),
                    marginal_value=2.31,
                    categories_helped=["SB"],
                    ownership_pct=44.0,
                    rationale="steals",
                )
            ],
        )


def test_get_free_agents_returns_contract():
    app = create_app()
    app.dependency_overrides[get_fa_service] = lambda: _FakeFAService()
    client = TestClient(app)
    resp = client.get("/api/free-agents", params={"team_name": "Team Hickey", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["recommendations"][0]["add"]["name"] == "A. Player"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_free_agents.py::test_get_free_agents_returns_contract -v`
Expected: FAIL — `ImportError: cannot import name 'get_fa_service' from 'api.deps'`

- [ ] **Step 3: Implement (mirror `team_service.py` / `routers/team.py` / `deps.py`)**

Create `api/services/fa_service.py`. Mirror `api/services/team_service.py`'s structure (lazy `src` imports inside the method; resilient mapping; best-effort). Call the engine per CLAUDE.md:
```python
"""Free Agents service — the ONE place that calls the FA recommender engine.
Maps engine output → the Free Agents contract. Resilient: missing live data
degrades to an empty recommendation list rather than raising."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse


class FreeAgentService:
    def get_free_agents(self, team_name: str, limit: int = 5) -> FreeAgentsResponse:
        from src.optimizer.fa_recommender import recommend_fa_moves
        from src.optimizer.shared_data_layer import build_optimizer_context
        from src.valuation import LeagueConfig
        from src.yahoo_data_service import get_yahoo_data_service

        recs: list[FreeAgentRec] = []
        try:
            yds = get_yahoo_data_service()
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=yds,
                config=LeagueConfig(),
                user_team_name=team_name,
                level_filter="MLB only",
            )
            for move in recommend_fa_moves(ctx, max_moves=limit) or []:
                recs.append(self._to_rec(move))
        except Exception:
            recs = []  # cold env / no data → empty list (page shows EmptyState)
        return FreeAgentsResponse(team_name=team_name, recommendations=recs)

    @staticmethod
    def _to_rec(move) -> FreeAgentRec:
        # `move` is the recommender's per-swap object/dict; map defensively.
        g = move.get if isinstance(move, dict) else lambda k, d=None: getattr(move, k, d)
        add = g("add") or {}
        drop = g("drop")
        add_ref = PlayerRef(
            id=int((add.get("player_id") if isinstance(add, dict) else getattr(add, "player_id", 0)) or 0),
            name=str((add.get("player_name") if isinstance(add, dict) else getattr(add, "player_name", "")) or ""),
            positions=str((add.get("positions") if isinstance(add, dict) else getattr(add, "positions", "")) or ""),
        )
        drop_ref = None
        if drop:
            drop_ref = PlayerRef(
                id=int((drop.get("player_id") if isinstance(drop, dict) else getattr(drop, "player_id", 0)) or 0),
                name=str((drop.get("player_name") if isinstance(drop, dict) else getattr(drop, "player_name", "")) or ""),
                positions=str((drop.get("positions") if isinstance(drop, dict) else getattr(drop, "positions", "")) or ""),
            )
        return FreeAgentRec(
            add=add_ref,
            drop=drop_ref,
            marginal_value=float(g("marginal_value", 0.0) or 0.0),
            categories_helped=list(g("categories_helped", []) or []),
            ownership_pct=g("ownership_pct"),
            rationale=str(g("rationale", "") or ""),
        )
```

> The exact shape of a `recommend_fa_moves` result is the integration seam (like `team_service`); the `_to_rec` mapping is best-effort and confined to this service. Tests use the fake, so they don't depend on it. Flag mapping uncertainty in your report.

Create `api/routers/free_agents.py` (mirror `routers/team.py` — thin, no `src` import, no arithmetic):
```python
"""Free Agents router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.free_agents import FreeAgentsResponse
from api.deps import get_fa_service

router = APIRouter(prefix="/api", tags=["free-agents"])


@router.get("/free-agents", response_model=FreeAgentsResponse)
def get_free_agents(team_name: str, limit: int = 5, service=Depends(get_fa_service)) -> FreeAgentsResponse:
    return service.get_free_agents(team_name, limit)
```

In `api/deps.py`, ADD:
```python
from api.services.fa_service import FreeAgentService


def get_fa_service() -> FreeAgentService:
    return FreeAgentService()
```

In `api/main.py` `create_app()`, ADD alongside the existing team router include:
```python
    from api.routers.free_agents import router as fa_router

    app.include_router(fa_router)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_free_agents.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add api/services/fa_service.py api/routers/free_agents.py api/deps.py api/main.py tests/api/test_free_agents.py
git commit -m "feat(api): GET /api/free-agents thin router + service seam (B-slice2 task 2)"
```

---

### Task 3: Lineup-optimize contract

**Files:** Create `api/contracts/lineup.py`, `tests/api/test_lineup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_lineup.py`:
```python
from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot


def test_lineup_contract_shape():
    resp = LineupOptimizeResponse(
        team_name="Team Hickey",
        date="2027-04-05",
        slots=[
            LineupSlot(
                slot="OF",
                player=PlayerRef(id=1, name="A. Player", positions="OF"),
                action="START",
                projected=4.2,
                forced_start=False,
                reason=None,
            )
        ],
        summary="9 starters set; 0 forced.",
    )
    dumped = resp.model_dump()
    assert dumped["slots"][0]["action"] == "START"
    assert dumped["slots"][0]["player"]["name"] == "A. Player"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_lineup.py::test_lineup_contract_shape -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.contracts.lineup'`

- [ ] **Step 3: Implement**

Create `api/contracts/lineup.py`:
```python
"""Contract models for the Lineup Optimizer page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class LineupOptimizeRequest(BaseModel):
    team_name: str
    date: str | None = None
    scope: str = "rest_of_season"


class LineupSlot(BaseModel):
    slot: str
    player: PlayerRef
    action: str  # "START" | "SIT"
    projected: float
    forced_start: bool = False
    reason: str | None = None


class LineupOptimizeResponse(BaseModel):
    team_name: str
    date: str
    slots: list[LineupSlot]
    summary: str = ""
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_lineup.py::test_lineup_contract_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/contracts/lineup.py tests/api/test_lineup.py
git commit -m "feat(api): Lineup-optimize contract models (B-slice2 task 3)"
```

---

### Task 4: Lineup service + thin router + endpoint test

**Files:** Create `api/services/lineup_service.py`, `api/routers/lineup.py`; Modify `api/deps.py`, `api/main.py`, `tests/api/test_lineup.py`

- [ ] **Step 1: Write the failing endpoint test**

Append to `tests/api/test_lineup.py`:
```python
from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app
from starlette.testclient import TestClient


class _FakeLineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="A. Player", positions="OF"),
                    action="START",
                    projected=4.2,
                )
            ],
            summary="1 starter",
        )


def test_post_lineup_optimize_returns_contract():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    client = TestClient(app)
    resp = client.post("/api/lineup/optimize", json={"team_name": "Team Hickey", "date": "2027-04-05"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["slots"][0]["action"] == "START"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_lineup.py::test_post_lineup_optimize_returns_contract -v`
Expected: FAIL — `ImportError: cannot import name 'get_lineup_service' from 'api.deps'`

- [ ] **Step 3: Implement (mirror Task 2's structure)**

Create `api/services/lineup_service.py`. Lazy `src` imports; resilient. Call the optimizer per CLAUDE.md (`LineupOptimizerPipeline`). Map the pipeline output → `LineupOptimizeResponse`; on cold env, return an empty slot list with a summary noting no data:
```python
"""Lineup service — the ONE place that calls the optimizer pipeline. Maps
its output → the Lineup contract. Resilient: missing data → empty slots.
NOTE: synchronous for B1; becomes an Arq background job in B3."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot


class LineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        slots: list[LineupSlot] = []
        summary = ""
        resolved_date = date or ""
        try:
            from src.game_day import get_target_game_date
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            resolved_date = date or str(get_target_game_date())
            yds = get_yahoo_data_service()
            roster = yds.get_rosters()
            pipeline = LineupOptimizerPipeline(roster, mode="standard", config=LeagueConfig())
            result = pipeline.run() if hasattr(pipeline, "run") else None
            slots = self._to_slots(result)
            summary = f"{sum(1 for s in slots if s.action == 'START')} starters set."
        except Exception:
            summary = "Lineup unavailable (no live data in this environment)."
        return LineupOptimizeResponse(team_name=team_name, date=resolved_date, slots=slots, summary=summary)

    @staticmethod
    def _to_slots(result) -> list[LineupSlot]:
        # `result` shape is the integration seam; map defensively. Return [] if absent.
        rows = []
        if result is None:
            return rows
        lineup = getattr(result, "lineup", None) or (result.get("lineup") if isinstance(result, dict) else None) or []
        for r in lineup:
            g = r.get if isinstance(r, dict) else lambda k, d=None: getattr(r, k, d)
            rows.append(
                LineupSlot(
                    slot=str(g("slot", "") or ""),
                    player=PlayerRef(
                        id=int(g("player_id", 0) or 0),
                        name=str(g("player_name", "") or ""),
                        positions=str(g("positions", "") or ""),
                    ),
                    action="START" if g("action", "START") in ("START", "start", True) else "SIT",
                    projected=float(g("projected", 0.0) or 0.0),
                    forced_start=bool(g("forced_start", False)),
                    reason=g("reason"),
                )
            )
        return rows
```

> The pipeline's exact run-method + output shape is the integration seam; `_to_slots` is best-effort and confined here. Tests use the fake. Flag uncertainty in your report; do NOT block on it.

Create `api/routers/lineup.py` (thin; POST with the request body model):
```python
"""Lineup router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse
from api.deps import get_lineup_service

router = APIRouter(prefix="/api", tags=["lineup"])


@router.post("/lineup/optimize", response_model=LineupOptimizeResponse)
def optimize_lineup(req: LineupOptimizeRequest, service=Depends(get_lineup_service)) -> LineupOptimizeResponse:
    return service.optimize(req.team_name, req.date, req.scope)
```

In `api/deps.py`, ADD:
```python
from api.services.lineup_service import LineupService


def get_lineup_service() -> LineupService:
    return LineupService()
```

In `api/main.py` `create_app()`, ADD:
```python
    from api.routers.lineup import router as lineup_router

    app.include_router(lineup_router)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_lineup.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add api/services/lineup_service.py api/routers/lineup.py api/deps.py api/main.py tests/api/test_lineup.py
git commit -m "feat(api): POST /api/lineup/optimize thin router + service seam (B-slice2 task 4)"
```

---

### Task 5: Regenerate OpenAPI + full verification

**Files:** Modify `api/openapi.json`

- [ ] **Step 1: Regenerate the OpenAPI snapshot**

The two new endpoints changed the schema, so `tests/api/test_openapi_contract.py` now fails. Regenerate:

Run: `python scripts/export_openapi.py`
Expected: rewrites `api/openapi.json` (now includes `/api/free-agents` + `/api/lineup/optimize`).

- [ ] **Step 2: Full verification**

Run: `python -m pytest tests/api/ -v`
Expected: PASS — all tests (healthz, my_team, free_agents, lineup, the two structural guards, openapi snapshot).

Run: `python -m ruff format api/ scripts/ tests/api/ && python -m ruff check api/ scripts/ tests/api/`
Expected: formatted; all checks pass.

Run: `git diff --name-only master -- src/`
Expected: empty — `src/` engines untouched.

- [ ] **Step 3: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regenerate OpenAPI for B-slice2 endpoints (task 5)"
```

---

## Self-Review

**Spec coverage** (`2026-06-16-heater-backend-api-foundation-design.md`): §4.1 lists `GET /free-agents` and `POST /lineup/optimize` → Tasks 1–4. §5 B1 = "Free Agents + Lineup-optimize endpoints" → this whole plan. The optimize endpoint is synchronous (B1); B3 turns it into an Arq job (noted in `lineup_service.py`). ✓

**Placeholder scan:** No TBD/TODO. Every code step has complete code; commands have expected output. The two service `_to_*` mappers are explicitly best-effort integration seams (the same pattern Slice 1's `team_service` uses), confined to the service, and bypassed by the dependency-override tests — not placeholders. ✓

**Type consistency:** `PlayerRef` (shared), `FreeAgentRec`/`FreeAgentsResponse`, `LineupSlot`/`LineupOptimizeResponse`/`LineupOptimizeRequest`, `get_fa_service`/`get_lineup_service`, `FreeAgentService.get_free_agents`, `LineupService.optimize` are used identically across tasks. ✓

**Invariants preserved:** new routers stay logic-free (no `src` imports, no arithmetic) → the existing `tests/api/test_no_logic_in_routers.py` guard auto-covers them; `src/` untouched (Task 5 Step 2 asserts it); OpenAPI snapshot regenerated + guarded.

## Leaves for later
- **B2:** Postgres + repository abstraction + Alembic (replaces SQLite under these services).
- **B3:** the optimize endpoint becomes an Arq background job (enqueue → poll/SSE).
- **B4:** Clerk auth + `tenant_id` scoping + the `LeagueConnector` interface.
- **B5:** point the live Streamlit app at these endpoints (strangler-fig proof).
