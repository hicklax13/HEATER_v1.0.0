# Backend API — Slice 8: Draft Simulator `recommend` endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one dormant, **stateless** endpoint — `POST /api/draft/recommend` — that wraps the existing `DraftRecommendationEngine.recommend(...)`, following the slice 1–7 pattern.

**Architecture:** The client holds the draft (`config` + `pick_log`) and sends it on each call; the service rebuilds `DraftState` by replaying picks (exactly how `DraftState.load` already works — `src/draft_state.py:434-463`), loads the player pool, runs the engine, and maps the result → the contract. **No server-side draft storage** (we have none pre-Postgres/Redis). The snake-draft clock is always computed from the rebuilt state (pure Python); only the pool+MC compute degrades gracefully in a cold env. `src/` engines stay UNCHANGED; the endpoint is mounted but dormant.

**Tech Stack:** FastAPI, Pydantic v2, pandas (engine output), Starlette TestClient, pytest.

---

## Environment (READ FIRST)

- **Working dir:** this worktree (`.claude/worktrees/feat+api-slice8-draft-recommend`), branch `worktree-feat+api-slice8-draft-recommend`, based on `origin/master` (`a512ab0`, which already contains slice 7). NO `.venv` here.
- **Python / runner:** use the main checkout's venv: `PY="/c/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe"`. Tests: `"$PY" -m pytest …`; ruff: `"$PY" -m ruff …`; openapi export: `"$PY" scripts/export_openapi.py`.
- **Baseline (already verified):** `"$PY" -m pytest tests/api/ -q` → **46 passed, 0 failed**. Keep it green; the openapi snapshot test must stay green (Task 4 regenerates the snapshot after the new endpoint is added).
- **New api test file MUST be `tests/api/test_api_draft.py`** (the `test_api_*` basename rule — a plain `test_draft.py` could collide with an existing `tests/` basename and only the full suite would surface it).

---

## Locked decisions (owner-approved — do not deviate)

1. **Scope = `recommend` ONLY.** Do NOT add an AI-opponent simulation endpoint, and do NOT touch `pages/20_Draft_Simulator.py` or `src/`. (The page's `auto_pick_opponents` extraction is a separate future slice.)
2. **Stateless.** No server-side session/state. The request carries the whole draft; the server is a pure function.
3. **HEATER `player_id` (int) is the identity** throughout (no Yahoo keys — this is a simulator).
4. **All outcomes return HTTP 200**; the endpoint must never 500. The clock is best-effort-always; recommendations degrade to `[]` on any compute failure.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `api/contracts/draft.py` | create | `DraftConfig`, `DraftPick`, `DraftRecommendRequest`, `DraftClock`, `DraftRecommendation`, `DraftRecommendResponse` |
| `api/services/draft_service.py` | create | the ONE `src/`-touching unit; rebuild state + clock + run engine + map output (+ NaN-safe numeric helpers) |
| `api/routers/draft.py` | create | thin logic-free router; `POST /api/draft/recommend` |
| `api/deps.py` | modify | `get_draft_service()` provider |
| `api/main.py` | modify | mount the draft router |
| `tests/api/test_api_draft.py` | create | contract-shape + service-unit (clock/rebuild/map/graceful, via injected fake engine) + endpoint (fake-service override) tests |
| `api/openapi.json` | regenerate | frontend's type source — regenerated in Task 4 |

---

## Task 1: Contracts

**Files:**
- Create: `api/contracts/draft.py`
- Test: `tests/api/test_api_draft.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_api_draft.py` with ONLY these contract tests:

```python
from api.contracts.common import PlayerRef
from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
)


def test_request_defaults():
    req = DraftRecommendRequest()
    assert req.config.num_teams == 12
    assert req.config.num_rounds == 23
    assert req.config.user_team_index == 0
    assert req.config.roster_config is None
    assert req.pick_log == []
    assert req.top_n == 8
    assert req.n_simulations == 300


def test_request_with_picks():
    req = DraftRecommendRequest(
        config=DraftConfig(num_teams=12, user_team_index=3),
        pick_log=[DraftPick(pick=0, team_index=0, player_id=1, player_name="A", positions="SS")],
        top_n=5,
    )
    dumped = req.model_dump()
    assert dumped["config"]["user_team_index"] == 3
    assert dumped["pick_log"][0]["player_id"] == 1
    assert dumped["top_n"] == 5


def test_response_shape():
    resp = DraftRecommendResponse(
        clock=DraftClock(current_pick=0, round=1, pick_in_round=1, picking_team_index=0, is_user_turn=True),
        recommendations=[
            DraftRecommendation(
                player=PlayerRef(id=1, name="A. Player", positions="SS"),
                rank=1,
                score=87.5,
                projected_sgp=4.2,
                confidence=0.8,
                tag="BUY",
            )
        ],
        summary="1 recommendation",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["recommendations"][0]["player"]["name"] == "A. Player"
    assert dumped["recommendations"][0]["reason"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.contracts.draft'`.

- [ ] **Step 3: Create `api/contracts/draft.py`**

```python
"""Contract models for the Draft Simulator 'recommend' endpoint.

Stateless: the client holds the draft (config + pick_log) and sends it on each
call; the server rebuilds DraftState by replaying picks (exactly like
DraftState.load) and answers as a pure function — no server-side draft storage."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class DraftConfig(BaseModel):
    num_teams: int = 12
    num_rounds: int = 23
    user_team_index: int = 0                       # 0-based seat of the user
    roster_config: dict[str, int] | None = None    # None → engine default


class DraftPick(BaseModel):
    pick: int                                       # 0-indexed overall pick number
    team_index: int
    player_id: int
    player_name: str
    positions: str                                  # comma-separated, e.g. "SS,3B"


class DraftRecommendRequest(BaseModel):
    config: DraftConfig = DraftConfig()
    pick_log: list[DraftPick] = []                  # picks so far, in order
    top_n: int = 8                                  # capped server-side
    n_simulations: int = 300                        # MC sims; capped server-side


class DraftClock(BaseModel):
    current_pick: int
    round: int
    pick_in_round: int
    picking_team_index: int
    is_user_turn: bool


class DraftRecommendation(BaseModel):
    player: PlayerRef
    rank: int
    score: float                                    # composite value (0-100)
    projected_sgp: float                            # MC mean roster SGP (best-effort)
    confidence: float | None = None
    tag: str | None = None                          # e.g. buy/fair/avoid
    reason: str = ""


class DraftRecommendResponse(BaseModel):
    clock: DraftClock
    recommendations: list[DraftRecommendation]
    summary: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/contracts/draft.py tests/api/test_api_draft.py
git commit -m "feat(api): Draft Simulator recommend contracts — B-slice8 task 1"
```

---

## Task 2: DraftService (the src/ seam)

**Files:**
- Create: `api/services/draft_service.py`
- Test: `tests/api/test_api_draft.py` (append)

- [ ] **Step 1: Write the failing test**

Add these imports to the TOP import block of `tests/api/test_api_draft.py` (keep ALL imports at the top — never below the tests, or ruff E402/F811 fails Task 4):

```python
import pandas as pd

from api.services.draft_service import DraftService
```

Then append these helpers + tests BELOW the existing tests:

```python
def _req(picks=None, top_n_req=8):
    return DraftRecommendRequest(pick_log=picks or [], top_n=top_n_req)


class _FakeEngine:
    """Stand-in for DraftRecommendationEngine — returns a canned result frame."""

    def __init__(self, frame):
        self._frame = frame
        self.called_with = None

    def recommend(self, player_pool, draft_state, top_n=8, n_simulations=300):
        self.called_with = {"top_n": top_n, "n_simulations": n_simulations}
        return self._frame


def test_rebuild_and_clock_empty_draft():
    # 12 teams, user seat 0, no picks yet → pick 0, round 1, user on the clock.
    svc = DraftService()
    ds = svc._rebuild_state(_req())
    clock = svc._clock(ds)
    assert clock.current_pick == 0
    assert clock.round == 1
    assert clock.pick_in_round == 1
    assert clock.picking_team_index == 0
    assert clock.is_user_turn is True


def test_clock_snake_order_after_replays():
    # After 1 pick → team 1 on the clock (forward round 1).
    svc = DraftService()
    one = [DraftPick(pick=0, team_index=0, player_id=1, player_name="A", positions="SS")]
    clock1 = svc._clock(svc._rebuild_state(_req(picks=one)))
    assert clock1.current_pick == 1 and clock1.picking_team_index == 1 and clock1.is_user_turn is False
    # After a full round of 12 picks → round 2, snake reverses → team 11 on the clock.
    twelve = [
        DraftPick(pick=i, team_index=i, player_id=100 + i, player_name=f"P{i}", positions="OF")
        for i in range(12)
    ]
    clock2 = svc._clock(svc._rebuild_state(_req(picks=twelve)))
    assert clock2.current_pick == 12 and clock2.round == 2 and clock2.picking_team_index == 11


def test_to_recs_maps_and_is_nan_safe():
    frame = pd.DataFrame(
        [
            {
                "player_id": 7,
                "player_name": "Star Hitter",
                "positions": "OF",
                "overall_rank": 1,
                "composite_value": 92.3,
                "mean_sgp": 5.1,
                "confidence": 0.77,
                "buy_fair_avoid": "BUY",
            },
            {
                "player_id": 9,
                "player_name": "Other Guy",
                "positions": "2B",
                "overall_rank": 2,
                "composite_value": float("nan"),  # missing → must degrade to 0.0, not NaN
                "mean_sgp": 3.0,
                "confidence": float("nan"),        # missing → None
                "buy_fair_avoid": None,
            },
        ]
    )
    recs = DraftService._to_recs(frame)
    assert len(recs) == 2
    assert recs[0].player.id == 7 and recs[0].rank == 1 and recs[0].score == 92.3
    assert recs[0].tag == "BUY" and recs[0].confidence == 0.77
    assert recs[1].score == 0.0  # NaN composite_value degraded
    assert recs[1].confidence is None  # NaN confidence → None
    assert recs[1].tag is None


def test_to_recs_empty_frame():
    assert DraftService._to_recs(pd.DataFrame()) == []
    assert DraftService._to_recs(None) == []


def test_recommend_full_path_with_injected_engine():
    frame = pd.DataFrame(
        [{"player_id": 7, "player_name": "Star", "positions": "OF", "overall_rank": 1, "composite_value": 90.0, "mean_sgp": 5.0}]
    )
    fake = _FakeEngine(frame)
    resp = DraftService().recommend(_req(top_n_req=5), engine=fake, pool=object())
    assert resp.clock.is_user_turn is True
    assert len(resp.recommendations) == 1 and resp.recommendations[0].player.id == 7
    assert "1 recommendation" in resp.summary
    assert fake.called_with["top_n"] == 5  # request top_n forwarded (within cap)


def test_recommend_is_graceful_when_engine_raises():
    class _BoomEngine:
        def recommend(self, player_pool, draft_state, top_n=8, n_simulations=300):
            raise RuntimeError("engine exploded")

    resp = DraftService().recommend(_req(), engine=_BoomEngine(), pool=object())
    assert resp.recommendations == []
    assert resp.clock.round == 1 and resp.clock.is_user_turn is True  # clock still computed
    assert "unavailable" in resp.summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.services.draft_service'`.

- [ ] **Step 3: Create `api/services/draft_service.py`**

```python
"""Draft Simulator service — the ONE place that calls the draft engine.

Stateless: rebuilds DraftState from (config + pick_log) by replaying picks
(mirrors DraftState.load), loads the player pool, runs the recommendation
engine, and maps the result → the Draft contract. Resilient: the snake-draft
clock is always computed from the rebuilt state (pure Python, no DB); only the
pool+MC compute degrades to an empty list. The endpoint never 500s.

NOTE: synchronous for now; becomes an Arq background job in B3 (like
POST /lineup/optimize). `engine`/`pool` params exist for tests to inject fakes;
production passes neither (the real engine + pool are loaded lazily)."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.draft import (
    DraftClock,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
)

_TOP_N_CAP = 50
_SIM_CAP = 1000
_ZERO_CLOCK = DraftClock(
    current_pick=0, round=0, pick_in_round=0, picking_team_index=0, is_user_turn=False
)


class DraftService:
    def recommend(self, req: DraftRecommendRequest, engine=None, pool=None) -> DraftRecommendResponse:
        try:
            ds = self._rebuild_state(req)
        except Exception:
            return DraftRecommendResponse(clock=_ZERO_CLOCK, recommendations=[], summary="Invalid draft state.")
        clock = self._clock(ds)
        try:
            results = self._run_engine(req, ds, engine=engine, pool=pool)
            recs = self._to_recs(results)
            return DraftRecommendResponse(
                clock=clock,
                recommendations=recs,
                summary=f"{len(recs)} recommendation{'s' if len(recs) != 1 else ''} for pick {ds.current_pick + 1}.",
            )
        except Exception:
            return DraftRecommendResponse(
                clock=clock,
                recommendations=[],
                summary="Draft recommendations unavailable (no pool data in this environment).",
            )

    @staticmethod
    def _rebuild_state(req: DraftRecommendRequest):
        from src.draft_state import DraftState

        cfg = req.config
        ds = DraftState(
            num_teams=cfg.num_teams,
            num_rounds=cfg.num_rounds,
            user_team_index=cfg.user_team_index,
            roster_config=cfg.roster_config,
        )
        for p in req.pick_log:
            ds.make_pick(p.player_id, p.player_name, p.positions, team_index=p.team_index)
        return ds

    @staticmethod
    def _clock(ds) -> DraftClock:
        return DraftClock(
            current_pick=ds.current_pick,
            round=ds.current_round,
            pick_in_round=ds.pick_in_round,
            picking_team_index=ds.picking_team_index(),
            is_user_turn=ds.is_user_turn,
        )

    @staticmethod
    def _run_engine(req: DraftRecommendRequest, ds, engine=None, pool=None):
        top_n = max(1, min(req.top_n, _TOP_N_CAP))
        n_sims = max(1, min(req.n_simulations, _SIM_CAP))
        if pool is None:
            from src.database import load_player_pool

            pool = load_player_pool()
        if engine is None:
            from src.draft_engine import DraftRecommendationEngine
            from src.valuation import LeagueConfig

            engine = DraftRecommendationEngine(LeagueConfig(), mode="standard")
        return engine.recommend(pool, ds, top_n=top_n, n_simulations=n_sims)

    @staticmethod
    def _to_recs(results) -> list[DraftRecommendation]:
        out: list[DraftRecommendation] = []
        if results is None or getattr(results, "empty", True):
            return out
        for row in results.to_dict("records"):
            g = row.get
            pid = _i(g("player_id"))
            if pid == 0:
                continue
            rank = g("overall_rank")
            score = g("composite_value")
            psgp = g("mean_sgp")
            out.append(
                DraftRecommendation(
                    player=PlayerRef(
                        id=pid,
                        name=str(g("player_name") or g("name") or ""),
                        positions=str(g("positions") or ""),
                    ),
                    rank=_i(rank if rank is not None else g("rank")),
                    score=_f(score if score is not None else g("combined_score")),
                    projected_sgp=_f(psgp if psgp is not None else g("risk_adjusted_sgp")),
                    confidence=_opt_f(g("confidence")),
                    tag=_opt_s(g("buy_fair_avoid")),
                    reason="",
                )
            )
        return out


def _f(v, d: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return d if x != x else x  # x != x is True only for NaN


def _i(v, d: int = 0) -> int:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return d if x != x else int(x)


def _opt_f(v):
    if v is None:
        return None
    x = _f(v, d=float("nan"))
    return None if x != x else x


def _opt_s(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -q`
Expected: PASS (9 tests in the file now).

- [ ] **Step 5: Commit**

```bash
git add api/services/draft_service.py tests/api/test_api_draft.py
git commit -m "feat(api): DraftService — stateless rebuild + engine recommend + NaN-safe map — B-slice8 task 2"
```

---

## Task 3: Router + DI provider + mount

**Files:**
- Create: `api/routers/draft.py`
- Modify: `api/deps.py`
- Modify: `api/main.py`
- Test: `tests/api/test_api_draft.py` (append)

- [ ] **Step 1: Write the failing test**

Add these imports to the TOP import block of `tests/api/test_api_draft.py`:

```python
from starlette.testclient import TestClient

from api.deps import get_draft_service
from api.main import create_app
```

Then append BELOW the existing tests:

```python
class _FakeDraftService:
    def recommend(self, req) -> DraftRecommendResponse:
        return DraftRecommendResponse(
            clock=DraftClock(current_pick=0, round=1, pick_in_round=1, picking_team_index=0, is_user_turn=True),
            recommendations=[
                DraftRecommendation(
                    player=PlayerRef(id=1, name="A. Player", positions="SS"),
                    rank=1,
                    score=88.0,
                    projected_sgp=4.0,
                )
            ],
            summary="1 recommendation for pick 1.",
        )


def test_post_draft_recommend_returns_contract():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/recommend", json={"config": {"num_teams": 12, "user_team_index": 0}, "pick_log": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clock"]["is_user_turn"] is True
    assert body["recommendations"][0]["player"]["name"] == "A. Player"


def test_post_draft_recommend_accepts_empty_body_defaults():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/recommend", json={})  # all fields default
    assert resp.status_code == 200
    assert resp.json()["recommendations"][0]["rank"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -k "post_draft" -q`
Expected: FAIL — `ImportError: cannot import name 'get_draft_service'` (then 404 until the router is mounted).

- [ ] **Step 3a: Create `api/routers/draft.py`**

```python
"""Draft Simulator router. THIN — delegates to the service. No engine imports,
no logic (guarded by tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.draft import DraftRecommendRequest, DraftRecommendResponse
from api.deps import get_draft_service

router = APIRouter(prefix="/api", tags=["draft"])


@router.post("/draft/recommend", response_model=DraftRecommendResponse)
def draft_recommend(req: DraftRecommendRequest, service=Depends(get_draft_service)) -> DraftRecommendResponse:
    return service.recommend(req)
```

- [ ] **Step 3b: Add the DI provider to `api/deps.py`**

Add the import alongside the other service imports:

```python
from api.services.draft_service import DraftService
```

Add the provider at the end of the file:

```python
def get_draft_service() -> DraftService:
    return DraftService()
```

- [ ] **Step 3c: Mount the router in `api/main.py`**

Inside `create_app()`, add the import alongside the other router imports:

```python
    from api.routers.draft import router as draft_router
```

And add the include alongside the other `app.include_router(...)` calls:

```python
    app.include_router(draft_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$PY" -m pytest tests/api/test_api_draft.py -q`
Expected: PASS (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add api/routers/draft.py api/deps.py api/main.py tests/api/test_api_draft.py
git commit -m "feat(api): mount draft router (POST /draft/recommend) — B-slice8 task 3"
```

---

## Task 4: Regenerate OpenAPI + verify all guards green

**Files:**
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Regenerate the OpenAPI snapshot**

Run: `"$PY" scripts/export_openapi.py`
Expected: `wrote …/api/openapi.json`.

- [ ] **Step 2: Sanity-check the new path + schemas are present**

Run: `grep -E "draft/recommend|DraftRecommendRequest|DraftRecommendResponse|DraftClock|DraftRecommendation|DraftConfig|DraftPick" api/openapi.json`
Expected: the new path `/api/draft/recommend` and all six Draft schemas appear.

- [ ] **Step 3: Run the full api suite + structural guards**

Run: `"$PY" -m pytest tests/api/ -q`
Expected: PASS — all green, including `test_openapi_contract.py::test_openapi_snapshot_is_current` (now current) and `test_no_logic_in_routers.py` (the new router has no `src.` import and no arithmetic).

- [ ] **Step 4: Lint**

Run: `"$PY" -m ruff check api/ tests/api/ && "$PY" -m ruff format --check api/ tests/api/`
Expected: clean. If `format --check` reports diffs, run `"$PY" -m ruff format api/ tests/api/` then re-check.

- [ ] **Step 5: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regenerate OpenAPI for draft recommend endpoint — B-slice8 task 4"
```

---

## Self-Review checklist (run before handing off for review)

1. **Scope** — exactly one endpoint (`POST /api/draft/recommend`); NO AI-opponent endpoint; `src/` and `pages/` untouched (`git diff a512ab0 -- src/ pages/` is empty). ✓
2. **Stateless** — the service rebuilds `DraftState` from the request every call; no module-level/global draft state. ✓
3. **Never-500** — `recommend` wraps both the rebuild and the engine compute; the clock is always returned. ✓
4. **NaN-safe mapping** — `_f`/`_i`/`_opt_f` guard against pandas NaN (the FA-engine lesson); the `_to_recs` NaN test proves it. ✓
5. **Thin-router guard** — router has no `src.` import and no arithmetic. ✓
6. **Type consistency** — `DraftClock`/`DraftRecommendation`/`DraftRecommendResponse` field names match across contract, service, router, and tests; the engine call is `engine.recommend(pool, ds, top_n=…, n_simulations=…)` everywhere. ✓
7. **Final check:** `"$PY" -m pytest tests/api/ -q` green; `git diff a512ab0 -- src/ pages/` empty.
