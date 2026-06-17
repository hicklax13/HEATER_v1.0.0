# HEATER Backend API — Slice 5 (Trade Analyzer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Add `POST /api/trade/evaluate` — HEATER's marquee feature — over the existing 6-phase trade engine (`evaluate_trade`), using the contract-first, logic-free-router pattern Slices 1–4 established.

**Architecture:** Same as prior slices. Contract (`api/contracts/trade.py`) + service (`api/services/trade_service.py`, the engine seam) + thin POST router (`api/routers/trade.py`) + DI provider + fake-service test. Mount in `api/main.py`.

**Tech Stack:** FastAPI · Pydantic v2 · pytest · `src/engine/output/trade_evaluator.evaluate_trade`. SQLite (unchanged).

> **⚠ Two load-bearing rules:**
> 1. Implement the contract models **EXACTLY as written below** (the frontend generates its types from the OpenAPI). Reuse `api/contracts/common.py::PlayerRef`.
> 2. **`enable_mc` defaults to `False`.** Monte Carlo is the slow path (the 44.8s synchronous-freeze the 2026-06-13 audit fixed by making MC opt-in). Default fast/deterministic; MC only when the caller asks. (MC becomes an async job in B3 — note it, don't build it here.)

**Pattern templates (read first, mirror exactly):** `api/contracts/lineup.py` (has a POST request model), `api/services/lineup_service.py`, `api/routers/lineup.py` (POST), `api/deps.py`, `api/main.py`, `tests/api/test_lineup.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `api/contracts/trade.py` | `TradeEvaluateRequest`, `GradeRange`, `CategoryImpact`, `TradeEvaluationResponse`. |
| `api/services/trade_service.py` | `TradeService.evaluate(...)` — the ONE place that calls `evaluate_trade`. |
| `api/routers/trade.py` | `POST /api/trade/evaluate` — thin. |
| `api/deps.py` | ADD `get_trade_service`. |
| `api/main.py` | ADD the trade router to `create_app()`. |
| `tests/api/test_trade.py` | contract + endpoint test (fake service). |
| `api/openapi.json` | regenerated. |

---

### Task 1: Trade contract models

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/trade.py
"""Contract models for the Trade Analyzer page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class TradeEvaluateRequest(BaseModel):
    team_name: str
    giving_ids: list[int]
    receiving_ids: list[int]
    enable_mc: bool = False     # Monte Carlo is the slow opt-in path (async in B3)


class GradeRange(BaseModel):
    grade: str
    low: float
    center: float
    high: float


class CategoryImpact(BaseModel):
    cat: str
    delta: float                # SGP delta for this category from the trade


class TradeEvaluationResponse(BaseModel):
    grade: str = ""             # Phase-1 weighted-SGP grade (the authority)
    verdict: str = ""           # e.g. "Accept", "Reject", "Fair"
    confidence_pct: float = 0.0
    surplus_sgp: float = 0.0    # headline net SGP surplus
    grade_range: GradeRange | None = None
    giving: list[PlayerRef] = []
    receiving: list[PlayerRef] = []
    category_impacts: list[CategoryImpact] = []
    delta_playoff_prob: float | None = None
    delta_champ_prob: float | None = None
    mc_enabled: bool = False
    summary: str = ""
    warnings: list[str] = []    # reshuffle / IP-floor / ghost-team flags
```

- [ ] Step 1: write a contract-shape test (`tests/api/test_trade.py`) building a `TradeEvaluationResponse` with a `GradeRange`, a `CategoryImpact`, and giving/receiving `PlayerRef`s; assert `model_dump()` shape. Run → confirm fail (module missing).
- [ ] Step 2: create `api/contracts/trade.py` verbatim. Run → pass.
- [ ] Step 3: `git commit -m "feat(api): Trade Analyzer contract models (B-slice5 task 1)"`

---

### Task 2: Trade service + thin POST router + endpoint test

**Service** (`TradeService.evaluate(team_name, giving_ids, receiving_ids, enable_mc=False)`): mirror `lineup_service.py` (lazy `src` imports, try/except → resilient default on cold env). Steps inside the method:
1. `pool = load_player_pool()` (from `src.database`).
2. Resolve `user_roster_ids`: `get_yahoo_data_service().get_rosters()` filtered to `team_name` → that team's `player_id` list. (INSPECT the rosters frame columns at build time.)
3. `result = evaluate_trade(giving_ids, receiving_ids, user_roster_ids, pool, config=LeagueConfig(), enable_mc=enable_mc, enable_context=True, enable_game_theory=True)` (from `src.engine.output.trade_evaluator`).
4. Map the `result` dict → `TradeEvaluationResponse`. INSPECT `evaluate_trade`'s output keys at build time (per CLAUDE.md it returns at least: `grade`, `verdict`, `confidence_pct`, `surplus_sgp`, `grade_range` = `{grade, low, center, high}`; and when the playoff sim ran: `delta_playoff_prob`, `delta_champ_prob`; plus reshuffle/IP-floor risk text you can fold into `warnings`). Build `giving`/`receiving` `PlayerRef`s by looking up the ids in `pool` (player_name/positions). Set `mc_enabled=enable_mc`.
5. On any failure (cold env / invalid ids): return `TradeEvaluationResponse(verdict="Could not evaluate", summary="<reason>")` — never raise a 500.

Flag any mapping uncertainty in your report (this is the integration seam, same as prior slices).

**Router** (`api/routers/trade.py`, thin — mirror `routers/lineup.py`'s POST):
```python
"""Trade Analyzer router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.trade import TradeEvaluateRequest, TradeEvaluationResponse
from api.deps import get_trade_service

router = APIRouter(prefix="/api", tags=["trade"])


@router.post("/trade/evaluate", response_model=TradeEvaluationResponse)
def evaluate_trade_endpoint(req: TradeEvaluateRequest, service=Depends(get_trade_service)) -> TradeEvaluationResponse:
    return service.evaluate(req.team_name, req.giving_ids, req.receiving_ids, req.enable_mc)
```

**DI:** add `get_trade_service` to `api/deps.py`. **main.py:** import + `app.include_router(trade_router)`.

**Endpoint test** (`tests/api/test_trade.py`, mirror `test_lineup.py`): a fake `TradeService` via `app.dependency_overrides` returning a populated `TradeEvaluationResponse`; `client.post("/api/trade/evaluate", json={"team_name": "Team Hickey", "giving_ids": [1], "receiving_ids": [2]})`; assert `body["grade"]` and `body["verdict"]`.

- [ ] Step 1: write the failing endpoint test → confirm fail
- [ ] Step 2: implement service, router, deps, main → confirm pass
- [ ] Step 3: `git commit -m "feat(api): POST /api/trade/evaluate thin router + service seam (B-slice5 task 2)"`

---

### Task 3: Regenerate OpenAPI + full verification

- [ ] Step 1: `python scripts/export_openapi.py` (adds `/api/trade/evaluate` to `api/openapi.json`)
- [ ] Step 2 — verify:
  - `python -m pytest tests/api/ -v` → ALL pass (Slices 1–5 + structural guards + openapi snapshot)
  - `python -m ruff format api/ scripts/ tests/api/ && python -m ruff check api/ scripts/ tests/api/` → clean
  - `git diff --name-only origin/master -- src/` → empty (engines untouched)
- [ ] Step 3: `git commit -m "chore(api): regenerate OpenAPI for trade endpoint (B-slice5 task 3)"`

---

## Self-Review
- **Spec coverage:** `POST /api/trade/evaluate` over the unchanged 6-phase `evaluate_trade` engine (the report's primary objective). MC opt-in (`enable_mc=False` default), to become an async job in B3. ✓
- **Placeholders:** the contract is given in full; service/router/test mirror the concrete on-disk `lineup` template; the `evaluate_trade` output mapping is a best-effort seam confined to the service (proven pattern) and bypassed by the dependency-override test. ✓
- **Type consistency:** reuses `PlayerRef`; `TradeEvaluateRequest`/`GradeRange`/`CategoryImpact`/`TradeEvaluationResponse` + `get_trade_service` + `TradeService.evaluate` referenced identically across tasks. ✓
- **Invariants:** router stays logic-free (existing guard auto-covers it); `src/` untouched (Task 3 asserts); OpenAPI regenerated + snapshot-guarded; MC default-off preserves the no-freeze guarantee.

## Leaves for later
- **B3:** `POST /api/trade/evaluate` (MC path) becomes an Arq background job (enqueue → poll/SSE).
- Trade Finder (`find_trade_opportunities`), Player Compare, Databank, Draft Sim — future slices.
- B2 Postgres / B4 auth+multi-tenancy — deferred per owner.
