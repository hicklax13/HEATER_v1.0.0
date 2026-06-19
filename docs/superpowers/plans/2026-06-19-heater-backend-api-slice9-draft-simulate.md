# Backend API Slice 9 — `POST /api/draft/simulate-picks` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stateless AI-opponent draft-simulation endpoint that auto-picks every opponent until it is the user's turn again, by extracting the live Draft Simulator page's opponent logic into the engine (`src/`) behind a behavior-identical guard test.

**Architecture:** Mirror slice 8 (`POST /api/draft/recommend`) exactly. Stateless: client sends `config + pick_log`; the service replays them into a `DraftState`, runs the (newly-extracted) pure `src.simulation.auto_pick_opponents`, and returns the new picks + updated clock. The ONLY live-code touch is moving ~15 lines out of `pages/20_Draft_Simulator.py` into `src/simulation.py`; the page becomes a thin delegate so its runtime behavior is byte-identical (guarded by a frozen-reference equivalence test + an AST structural test). Engines in `src/` keep their public behavior; routers stay logic-free (AST-guarded by `tests/api/test_no_logic_in_routers.py`).

**Tech Stack:** FastAPI, Pydantic v2, pandas, numpy, pytest, Starlette `TestClient`.

---

## Conventions for every command below

- Run all commands from the **worktree root**: `C:\Users\conno\Code\HEATER_v1.0.1\.claude\worktrees\api-slice9-draft-simulate`
- The worktree has no `.venv`; use the **main checkout's** interpreter. In these steps `PY` means:
  `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`
- Lint after touching any `.py`: `$PY -m ruff format <files>` then `$PY -m ruff check <files>`.

## File Structure (what each file is responsible for)

- **Create** `src/simulation.py` (append fn) — pure `auto_pick_opponents(ds, pool, rng=None) -> list[dict]`. The engine's home for AI-opponent auto-picking. No Streamlit, no DB.
- **Modify** `pages/20_Draft_Simulator.py:15` (import) and `:132-149` (fn body) — thin delegate to the engine fn. Signature `auto_pick_opponents(pool)` unchanged (3 call sites stay valid).
- **Modify** `api/contracts/draft.py` — add `DraftSimulatePicksRequest` + `DraftSimulatePicksResponse` (reuse `DraftConfig`/`DraftPick`/`DraftClock`).
- **Modify** `api/services/draft_service.py` — add `DraftService.simulate_picks(req, pool=None)`. Reuses `_rebuild_state`/`_clock`. The ONE place importing `src/` for this path.
- **Modify** `api/routers/draft.py` — add the thin `POST /api/draft/simulate-picks` route. No logic.
- **Modify** `api/openapi.json` — regenerated snapshot (script-generated, never hand-edited).
- **Create** `tests/test_draft_auto_pick_extraction.py` — behavior-identical guard (frozen-reference equivalence + determinism + AST page-delegation).
- **Create** `tests/api/test_api_draft_simulate.py` — contract defaults, service unit tests (real small pool + seed), fake-service endpoint test.

No `api/main.py` change: the draft router is already mounted (`api/main.py:49`); we add a second route to it. No `api/deps.py` change: reuse `get_draft_service` (`api/deps.py:79`).

---

### Task 1: Extract `auto_pick_opponents` into the engine (`src/simulation.py`)

**Files:**
- Create (append): `src/simulation.py` (end of file)
- Test: `tests/test_draft_auto_pick_extraction.py`

- [ ] **Step 1: Write the failing equivalence + determinism tests**

Create `tests/test_draft_auto_pick_extraction.py`:

```python
"""Behavior-identical guard for the slice-9 extraction of the Draft Simulator's
AI-opponent picking out of pages/20_Draft_Simulator.py into src/simulation.py.

The frozen reference below is a verbatim copy of the OLD inline page logic
(pre-extraction). The extracted engine fn must produce the identical pick
sequence under the same RNG state, so the live page (now a thin delegate)
keeps byte-identical behavior."""

import ast
import pathlib

import numpy as np
import pandas as pd

from src.draft_state import DraftState
from src.simulation import auto_pick_opponents

_PAGE = pathlib.Path(__file__).resolve().parents[1] / "pages" / "20_Draft_Simulator.py"


def _pool() -> pd.DataFrame:
    positions = ["SS", "OF", "2B", "3B", "1B", "SP", "RP", "C"]
    rows = [
        {
            "player_id": 100 + i,
            "name": f"Player{i}",
            "player_name": f"Player{i}",
            "positions": positions[i % len(positions)],
            "adp": float(i + 1),
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def _state(user_seat: int = 2, num_teams: int = 12, num_rounds: int = 23) -> DraftState:
    return DraftState(num_teams=num_teams, num_rounds=num_rounds, user_team_index=user_seat)


def _reference_auto_pick(ds: DraftState, pool: pd.DataFrame) -> list[int]:
    """Frozen copy of the OLD pages/20_Draft_Simulator.py inline logic."""
    made: list[int] = []
    while not ds.is_user_turn and ds.current_pick < ds.total_picks:
        available = ds.available_players(pool)
        if available.empty:
            break
        candidates = available.nsmallest(min(15, len(available)), "adp")
        size = len(candidates)
        weights = np.arange(size, 0, -1, dtype=float)
        weights /= weights.sum()
        pick_idx = int(np.random.choice(size, p=weights))
        player = candidates.iloc[pick_idx]
        pname = str(player.get("player_name", player.get("name", "Unknown")))
        made.append(int(player["player_id"]))
        ds.make_pick(
            player_id=int(player["player_id"]),
            player_name=pname,
            positions=str(player.get("positions", "Util")),
        )
    return made


def test_extracted_matches_frozen_reference():
    pool = _pool()
    np.random.seed(20260619)
    ref_ids = _reference_auto_pick(_state(), pool)
    np.random.seed(20260619)
    new_ids = [p["player_id"] for p in auto_pick_opponents(_state(), pool)]
    assert new_ids == ref_ids
    assert len(ref_ids) == 2  # user at seat 2 → opponents fill seats 0,1 then stop


def test_seeded_rng_is_deterministic():
    pool = _pool()
    a = [p["player_id"] for p in auto_pick_opponents(_state(), pool, rng=np.random.default_rng(123))]
    b = [p["player_id"] for p in auto_pick_opponents(_state(), pool, rng=np.random.default_rng(123))]
    assert a == b and len(a) == 2


def test_returns_pick_metadata():
    made = auto_pick_opponents(_state(user_seat=2), _pool(), rng=np.random.default_rng(1))
    assert [m["team_index"] for m in made] == [0, 1]
    assert [m["pick"] for m in made] == [0, 1]
    for m in made:
        assert set(m) == {"pick", "team_index", "player_id", "player_name", "positions"}


def test_no_picks_when_user_already_on_clock():
    made = auto_pick_opponents(_state(user_seat=0), _pool(), rng=np.random.default_rng(1))
    assert made == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_draft_auto_pick_extraction.py -q`
Expected: FAIL — `ImportError: cannot import name 'auto_pick_opponents' from 'src.simulation'`.

- [ ] **Step 3: Implement the pure engine fn (append to `src/simulation.py`)**

Append at the END of `src/simulation.py` (module-level; `numpy as np` is already imported at the top):

```python
def auto_pick_opponents(ds, pool, rng=None) -> list[dict]:
    """Advance a mock draft by auto-picking AI opponents until it is the
    user's turn (or the draft ends). Mutates *ds* in place; returns the list
    of picks made, each a dict with keys pick/team_index/player_id/
    player_name/positions (matches the api DraftPick contract).

    Opponent model (unchanged from the Draft Simulator page): take the top-15
    available players by ADP and sample one with linearly-decreasing weights.

    Args:
        ds: A DraftState. Read for is_user_turn/current_pick/total_picks and
            mutated via make_pick().
        pool: Player pool DataFrame (needs player_id, name/player_name,
            positions, adp columns).
        rng: Optional numpy Generator. None (default) → module-level
            ``np.random`` (byte-identical to the live page). When set, picks
            are reproducible — used by the stateless API + tests.

    Returns:
        list[dict]: one entry per AI pick made this call, in pick order.
    """
    choice = np.random.choice if rng is None else rng.choice
    made: list[dict] = []
    while not ds.is_user_turn and ds.current_pick < ds.total_picks:
        available = ds.available_players(pool)
        if available.empty:
            break
        candidates = available.nsmallest(min(15, len(available)), "adp")
        size = len(candidates)
        weights = np.arange(size, 0, -1, dtype=float)
        weights /= weights.sum()
        pick_idx = int(choice(size, p=weights))
        player = candidates.iloc[pick_idx]
        pname = str(player.get("player_name", player.get("name", "Unknown")))
        positions = str(player.get("positions", "Util"))
        pick_no = ds.current_pick
        team_index = ds.picking_team_index()
        ds.make_pick(player_id=int(player["player_id"]), player_name=pname, positions=positions)
        made.append(
            {
                "pick": pick_no,
                "team_index": team_index,
                "player_id": int(player["player_id"]),
                "player_name": pname,
                "positions": positions,
            }
        )
    return made
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_draft_auto_pick_extraction.py -q`
Expected: 4 passed (the AST page-delegation test is added in Task 2).

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format src/simulation.py tests/test_draft_auto_pick_extraction.py
$PY -m ruff check src/simulation.py tests/test_draft_auto_pick_extraction.py
git add src/simulation.py tests/test_draft_auto_pick_extraction.py
git commit -m "feat(simulation): extract auto_pick_opponents engine fn (slice 9)"
```

---

### Task 2: Make the live page delegate to the engine

**Files:**
- Modify: `pages/20_Draft_Simulator.py:15` (import), `:132-149` (fn body)
- Test: `tests/test_draft_auto_pick_extraction.py` (add the AST guard)

- [ ] **Step 1: Add the failing AST page-delegation test**

Append to `tests/test_draft_auto_pick_extraction.py`:

```python
def test_page_delegates_to_engine_no_inline_loop():
    """The page's auto_pick_opponents must import + call the engine fn and
    contain no inline RNG pick (so it can never drift from the engine)."""
    tree = ast.parse(_PAGE.read_text(encoding="utf-8"))

    asname = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.simulation":
            for alias in node.names:
                if alias.name == "auto_pick_opponents":
                    asname = alias.asname or alias.name
    assert asname, "page must import auto_pick_opponents from src.simulation"

    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "auto_pick_opponents"
    )
    # inline MC pick is gone — no `.choice` call remains in the page fn
    assert not any(isinstance(n, ast.Attribute) and n.attr == "choice" for n in ast.walk(func)), (
        "page auto_pick_opponents still contains an inline .choice call"
    )
    # it delegates to the imported engine fn
    called = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert asname in called, "page auto_pick_opponents must call the src.simulation delegate"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_draft_auto_pick_extraction.py::test_page_delegates_to_engine_no_inline_loop -q`
Expected: FAIL — page does not yet import `auto_pick_opponents` from `src.simulation` (asname is None).

- [ ] **Step 3: Edit the page — import alias + delegate body**

In `pages/20_Draft_Simulator.py`, change line 15 from:

```python
from src.simulation import DraftSimulator
```

to:

```python
from src.simulation import DraftSimulator, auto_pick_opponents as _auto_pick_opponents
```

Then replace the function body at lines 132-149 (keep the `# ── AI Opponent Logic` header comment above it):

```python
def auto_pick_opponents(pool: pd.DataFrame) -> None:
    ds: DraftState = st.session_state.mock_ds
    _auto_pick_opponents(ds, pool)
```

(Do NOT touch the 3 call sites — `auto_pick_opponents(pool)` at lines ~434, ~827, ~937 stay valid. Leave `import numpy as np` — still used at line ~549.)

- [ ] **Step 4: Run the full guard file to verify it passes**

Run: `$PY -m pytest tests/test_draft_auto_pick_extraction.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format pages/20_Draft_Simulator.py tests/test_draft_auto_pick_extraction.py
$PY -m ruff check pages/20_Draft_Simulator.py tests/test_draft_auto_pick_extraction.py
git add pages/20_Draft_Simulator.py tests/test_draft_auto_pick_extraction.py
git commit -m "refactor(draft-sim page): delegate auto_pick_opponents to engine (slice 9)"
```

---

### Task 3: Add the contract models

**Files:**
- Modify: `api/contracts/draft.py` (append after `DraftRecommendResponse`)
- Test: `tests/api/test_api_draft_simulate.py`

- [ ] **Step 1: Write the failing contract test**

Create `tests/api/test_api_draft_simulate.py`:

```python
from starlette.testclient import TestClient

from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
from api.deps import get_draft_service
from api.main import create_app
from api.services.draft_service import DraftService


def test_simulate_request_defaults():
    req = DraftSimulatePicksRequest()
    assert req.config.num_teams == 12
    assert req.config.user_team_index == 0
    assert req.pick_log == []
    assert req.seed is None


def test_simulate_response_shape():
    resp = DraftSimulatePicksResponse(
        clock=DraftClock(current_pick=2, round=1, pick_in_round=3, picking_team_index=2, is_user_turn=True),
        picks=[DraftPick(pick=0, team_index=0, player_id=101, player_name="A", positions="SS")],
        summary="1 opponent pick simulated.",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["picks"][0]["player_id"] == 101
    assert dumped["summary"].startswith("1 opponent")
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py -q`
Expected: FAIL — `ImportError: cannot import name 'DraftSimulatePicksRequest'`.

- [ ] **Step 3: Add the models (append to `api/contracts/draft.py`)**

```python
class DraftSimulatePicksRequest(BaseModel):
    config: DraftConfig = DraftConfig()
    pick_log: list[DraftPick] = []  # picks so far, in order
    seed: int | None = None  # None → non-deterministic (matches the live page)


class DraftSimulatePicksResponse(BaseModel):
    clock: DraftClock  # updated clock after the AI picks (whose turn now)
    picks: list[DraftPick] = []  # the NEW AI opponent picks made this call
    summary: str = ""
```

- [ ] **Step 4: Run to verify the two contract tests pass**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py -q`
Expected: 2 passed.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format api/contracts/draft.py tests/api/test_api_draft_simulate.py
$PY -m ruff check api/contracts/draft.py tests/api/test_api_draft_simulate.py
git add api/contracts/draft.py tests/api/test_api_draft_simulate.py
git commit -m "feat(api): add draft simulate-picks contract models (slice 9)"
```

---

### Task 4: Add `DraftService.simulate_picks`

**Files:**
- Modify: `api/services/draft_service.py` (add method + import)
- Test: `tests/api/test_api_draft_simulate.py` (add service unit tests)

- [ ] **Step 1: Write the failing service unit tests**

Append to `tests/api/test_api_draft_simulate.py`:

```python
import pandas as pd


def _sim_pool() -> pd.DataFrame:
    positions = ["SS", "OF", "2B", "3B", "1B", "SP", "RP", "C"]
    rows = [
        {
            "player_id": 100 + i,
            "name": f"Player{i}",
            "player_name": f"Player{i}",
            "positions": positions[i % len(positions)],
            "adp": float(i + 1),
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def test_simulate_picks_advances_to_user_turn_with_seed():
    # user at seat 2, fresh draft → exactly seats 0 and 1 auto-pick, then user's turn.
    req = DraftSimulatePicksRequest(config=DraftConfig(num_teams=12, user_team_index=2), pick_log=[], seed=42)
    resp = DraftService().simulate_picks(req, pool=_sim_pool())
    assert resp.clock.is_user_turn is True
    assert resp.clock.picking_team_index == 2
    assert [p.team_index for p in resp.picks] == [0, 1]
    assert "2 opponent picks" in resp.summary


def test_simulate_picks_seed_is_reproducible():
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=2), seed=7)
    pool = _sim_pool()
    a = [p.player_id for p in DraftService().simulate_picks(req, pool=pool).picks]
    b = [p.player_id for p in DraftService().simulate_picks(req, pool=pool).picks]
    assert a == b


def test_simulate_picks_no_picks_when_user_on_clock():
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=0), seed=1)
    resp = DraftService().simulate_picks(req, pool=_sim_pool())
    assert resp.picks == []
    assert resp.clock.is_user_turn is True


def test_simulate_picks_graceful_when_pool_load_fails():
    # pool=None forces the real load_player_pool(); in a DB-less env it raises and
    # the service must still return a valid clock with no picks (never 500).
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=3), seed=1)
    resp = DraftService().simulate_picks(req, pool=None)
    assert isinstance(resp, DraftSimulatePicksResponse)
    assert resp.clock.round >= 1  # clock always computed from the rebuilt state
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py -q`
Expected: FAIL — `AttributeError: 'DraftService' object has no attribute 'simulate_picks'`.

- [ ] **Step 3: Implement the method**

In `api/services/draft_service.py`, extend the contract import block:

```python
from api.contracts.draft import (
    DraftClock,
    DraftPick,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
```

Then add this method to the `DraftService` class (e.g. directly after `recommend`):

```python
    def simulate_picks(self, req: DraftSimulatePicksRequest, pool=None) -> DraftSimulatePicksResponse:
        try:
            ds = self._rebuild_state(req)
        except Exception:
            return DraftSimulatePicksResponse(clock=_ZERO_CLOCK, picks=[], summary="Invalid draft state.")
        try:
            import numpy as np

            from src.simulation import auto_pick_opponents

            if pool is None:
                from src.database import load_player_pool

                pool = load_player_pool()
            rng = np.random.default_rng(req.seed) if req.seed is not None else None
            made = auto_pick_opponents(ds, pool, rng=rng)
            picks = [DraftPick(**m) for m in made]
            n = len(picks)
            return DraftSimulatePicksResponse(
                clock=self._clock(ds),
                picks=picks,
                summary=f"{n} opponent pick{'s' if n != 1 else ''} simulated.",
            )
        except Exception:
            try:
                clock = self._clock(ds)
            except Exception:
                clock = _ZERO_CLOCK
            return DraftSimulatePicksResponse(
                clock=clock,
                picks=[],
                summary="Draft simulation unavailable (no pool data in this environment).",
            )
```

Note: `_rebuild_state` already reads only `req.config`/`req.pick_log`, which both request models share — no change needed there.

- [ ] **Step 4: Run to verify the service tests pass**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py -q`
Expected: 6 passed (2 contract + 4 service).

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format api/services/draft_service.py tests/api/test_api_draft_simulate.py
$PY -m ruff check api/services/draft_service.py tests/api/test_api_draft_simulate.py
git add api/services/draft_service.py tests/api/test_api_draft_simulate.py
git commit -m "feat(api): DraftService.simulate_picks over the extracted engine fn (slice 9)"
```

---

### Task 5: Add the thin router route + fake-service endpoint test

**Files:**
- Modify: `api/routers/draft.py` (import + route)
- Test: `tests/api/test_api_draft_simulate.py` (add the endpoint test)

- [ ] **Step 1: Write the failing endpoint test**

Append to `tests/api/test_api_draft_simulate.py`:

```python
class _FakeDraftService:
    def simulate_picks(self, req) -> DraftSimulatePicksResponse:
        return DraftSimulatePicksResponse(
            clock=DraftClock(current_pick=2, round=1, pick_in_round=3, picking_team_index=2, is_user_turn=True),
            picks=[
                DraftPick(pick=0, team_index=0, player_id=101, player_name="A", positions="SS"),
                DraftPick(pick=1, team_index=1, player_id=102, player_name="B", positions="OF"),
            ],
            summary="2 opponent picks simulated.",
        )


def test_post_draft_simulate_returns_contract():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post(
        "/api/draft/simulate-picks",
        json={"config": {"num_teams": 12, "user_team_index": 2}, "pick_log": [], "seed": 7},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["clock"]["is_user_turn"] is True
    assert len(body["picks"]) == 2
    assert body["picks"][0]["player_id"] == 101


def test_post_draft_simulate_accepts_empty_body_defaults():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/simulate-picks", json={})  # all fields default
    assert resp.status_code == 200
    assert resp.json()["picks"][0]["team_index"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py -q`
Expected: FAIL — POST returns 404 (route not yet registered).

- [ ] **Step 3: Add the route (`api/routers/draft.py`)**

Extend the contract import:

```python
from api.contracts.draft import (
    DraftRecommendRequest,
    DraftRecommendResponse,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
```

Append the route (keep it logic-free — just delegate):

```python
@router.post("/draft/simulate-picks", response_model=DraftSimulatePicksResponse)
def draft_simulate_picks(
    req: DraftSimulatePicksRequest, service=Depends(get_draft_service)
) -> DraftSimulatePicksResponse:
    return service.simulate_picks(req)
```

- [ ] **Step 4: Run to verify the endpoint tests pass + router stays logic-free**

Run: `$PY -m pytest tests/api/test_api_draft_simulate.py tests/api/test_no_logic_in_routers.py -q`
Expected: 8 passed in the simulate file + the no-logic guard green.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format api/routers/draft.py tests/api/test_api_draft_simulate.py
$PY -m ruff check api/routers/draft.py tests/api/test_api_draft_simulate.py
git add api/routers/draft.py tests/api/test_api_draft_simulate.py
git commit -m "feat(api): POST /api/draft/simulate-picks route (slice 9)"
```

---

### Task 6: Regenerate the OpenAPI snapshot

**Files:**
- Modify: `api/openapi.json` (script-generated)

- [ ] **Step 1: Confirm the snapshot is now stale (failing)**

Run: `$PY -m pytest tests/api/test_openapi_contract.py -q`
Expected: FAIL — "api/openapi.json is stale" (the new endpoint isn't in the committed schema yet).

- [ ] **Step 2: Regenerate**

Run: `$PY scripts/export_openapi.py`
Expected: prints `wrote .../api/openapi.json`.

- [ ] **Step 3: Verify the snapshot test passes**

Run: `$PY -m pytest tests/api/test_openapi_contract.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regen openapi.json for draft simulate-picks (slice 9)"
```

---

### Task 7: Full verification + branch wrap

**Files:** none (verification only)

- [ ] **Step 1: Run the whole API suite + the guard test**

Run: `$PY -m pytest tests/api/ tests/test_draft_auto_pick_extraction.py -q`
Expected: all green — 58 prior + 8 new simulate tests + 5 extraction-guard tests, 0 failures.

- [ ] **Step 2: Lint the full changeset**

Run:
```bash
$PY -m ruff format --check api/ src/simulation.py pages/20_Draft_Simulator.py tests/test_draft_auto_pick_extraction.py tests/api/test_api_draft_simulate.py
$PY -m ruff check api/ src/simulation.py pages/20_Draft_Simulator.py tests/test_draft_auto_pick_extraction.py tests/api/test_api_draft_simulate.py
```
Expected: "All checks passed!" / no reformat needed.

- [ ] **Step 3: Sanity — the live page still parses (no syntax break)**

Run: `$PY -c "import ast, pathlib; ast.parse(pathlib.Path('pages/20_Draft_Simulator.py').read_text(encoding='utf-8')); print('page parses OK')"`
Expected: `page parses OK`.

- [ ] **Step 4: Confirm clean tree**

Run: `git status --short && git log --oneline -6`
Expected: clean working tree; 6 slice-9 commits on `worktree-api-slice9-draft-simulate`.

Hand back to the orchestrator for the 2-stage review.

---

## Self-Review

**1. Spec coverage:**
- Stateless endpoint mirroring slice 8 → Tasks 3-6. ✅
- Advance-to-user-turn semantics → engine `while not ds.is_user_turn` (Task 1) + service returns updated clock (Task 4). ✅
- Live-page extraction with behavior-identical guard → Tasks 1-2 (frozen-reference equivalence + determinism + AST delegation). ✅
- Engines in `src/` keep public behavior; routers logic-free → Task 5 runs `test_no_logic_in_routers.py`. ✅
- OpenAPI snapshot regen → Task 6. ✅
- Reuse `DraftConfig`/`DraftPick`/`DraftClock`/`get_draft_service`/mounted router → Tasks 3-5 (no `main.py`/`deps.py` change). ✅

**2. Placeholder scan:** none — every code/test step shows complete code and exact commands.

**3. Type consistency:** `auto_pick_opponents(ds, pool, rng=None) -> list[dict]` with keys `pick/team_index/player_id/player_name/positions` is produced in Task 1 and consumed via `DraftPick(**m)` in Task 4 — keys match `DraftPick`'s fields exactly. `DraftSimulatePicksRequest`/`DraftSimulatePicksResponse` names are identical across Tasks 3-6. `simulate_picks(req, pool=None)` signature matches the fake-service `simulate_picks(req)` (the override only ever gets `req`).
