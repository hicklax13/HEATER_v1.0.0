# M0 — Free-Agent Pool endpoint (`GET /api/free-agents/pool`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new read endpoint `GET /api/free-agents/pool` that returns ALL available free agents ranked by marginal value (a browseable pool for the Players page), distinct from the existing `/api/free-agents` add/drop recommendations.

**Architecture:** Mirror the proven per-endpoint pattern. New contracts (`StatItem`, `FreeAgentPoolItem`, `FreeAgentPoolResponse`) in `api/contracts/free_agents.py`. New service `api/services/fa_pool_service.py` with pure mapping helpers (`_top_need`, `_tag_from`, `_key_stats`, `_to_pool_item`) + `FreeAgentPoolService.get_free_agents_pool` which composes `build_optimizer_context → src.in_season.rank_free_agents → map`. The keystone `rank_free_agents` already scores every FA by marginal SGP, sorts, and tags `best_category`. The service normalizes SGP→0-100, reuses the existing `player_ref_from_pool` for player identity (mlb_id/team), and derives `top_need` from `ctx.category_gaps`. New route added to the EXISTING `api/routers/free_agents.py` (already mounted — no `api/main.py` change). `src/` is NOT modified.

**Contract convention:** snake_case (the frontend adapter maps to its camelCase mock, exactly as it does for `PlayerRef`). Player identity is the nested `PlayerRef` (already carries name/positions/mlb_id/team_abbr/team_id).

**Deferred/defaulted display fields (documented, not bugs):** `own_delta` → `0.0` (ownership-trend delta needs date-diffing `ownership_trends`; the gap spec lists it as "not in API"); `tag` → only "Buy Low"/"Sell High" from `regression_flag` (the "Rising"/"Streamer"/"Closer Role" tags are deferred); `value` → marginal SGP normalized relative to the top FA in the pool (top = 100), clipped to [0,100]; `stats` → a fixed 3-stat triple per player type (hitters: HR/SB/AVG; pitchers: K/ERA/WHIP).

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = the worktree root.

**Tech Stack:** FastAPI + Pydantic; pandas; pytest. Routers stay logic-free (AST-guarded). `fastapi==0.137.1`/`httpx==0.28.1` are pinned (the OpenAPI snapshot is version-sensitive).

---

## File Structure

- `api/contracts/free_agents.py` — **modify**: add `StatItem`, `FreeAgentPoolItem`, `FreeAgentPoolResponse`.
- `api/services/fa_pool_service.py` — **create**: helpers + `FreeAgentPoolService`.
- `api/deps.py` — **modify**: add `get_fa_pool_service()` + its import.
- `api/routers/free_agents.py` — **modify**: add the `/free-agents/pool` route.
- `api/openapi.json` — **regenerate** (new models + route).
- `tests/api/test_api_free_agent_pool.py` — **create**: helper unit tests + the fake-service router/contract test.

---

### Task 1: Contracts + pure mapping helpers

**Files:**
- Modify: `api/contracts/free_agents.py`
- Create: `api/services/fa_pool_service.py`
- Test: `tests/api/test_api_free_agent_pool.py`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/api/test_api_free_agent_pool.py`:

```python
"""FA-pool endpoint: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.fa_pool_service import _key_stats, _tag_from, _to_pool_item, _top_need


def test_top_need_picks_most_negative_gap():
    # gap < 0 means the user is behind in that category (a need)
    assert _top_need({"R": 1.2, "SB": -3.4, "HR": -0.5}) == "SB"
    assert _top_need({}) == ""


def test_tag_from_regression_flag():
    assert _tag_from("BUY_LOW") == "Buy Low"
    assert _tag_from("SELL_HIGH") == "Sell High"
    assert _tag_from(None) is None
    assert _tag_from("") is None


def test_key_stats_hitter_and_pitcher():
    hitter = {"ytd_hr": 24, "ytd_sb": 11, "ytd_avg": 0.285}
    s = _key_stats(hitter, True)
    assert [x.label for x in s] == ["HR", "SB", "AVG"]
    assert s[0].value == "24"
    assert s[2].value == "0.285"
    pitcher = {"ytd_k": 180, "ytd_era": 3.21, "ytd_whip": 1.05}
    p = _key_stats(pitcher, False)
    assert [x.label for x in p] == ["K", "ERA", "WHIP"]
    assert p[1].value == "3.21"


def test_to_pool_item_normalizes_value_and_enriches():
    pool = pd.DataFrame([{"player_id": 1, "name": "Judge", "positions": "OF", "mlb_id": 592450, "team": "NYY"}])
    row = {
        "player_id": 1, "player_name": "Judge", "positions": "OF", "is_hitter": True,
        "marginal_value": 5.0, "best_category": "HR", "regression_flag": "BUY_LOW",
        "percent_owned": 60.0, "ytd_hr": 30, "ytd_sb": 5, "ytd_avg": 0.31,
    }
    item = _to_pool_item(2, row, pool, max_value=10.0)
    assert item.rank == 2
    assert item.value == 50.0  # 5/10*100
    assert item.player.mlb_id == 592450
    assert item.player.team_id == 147
    assert item.player.name == "Judge"
    assert item.own_pct == 60.0
    assert item.own_delta == 0.0
    assert item.hitter is True
    assert item.fit == "HR"
    assert item.tag == "Buy Low"
    assert item.stats[0].value == "30"


def test_to_pool_item_zero_max_value_is_safe():
    pool = pd.DataFrame([{"player_id": 1, "name": "X", "positions": "OF", "mlb_id": 1, "team": "NYY"}])
    item = _to_pool_item(1, {"player_id": 1, "marginal_value": 0.0, "is_hitter": True}, pool, max_value=0.0)
    assert item.value == 0.0  # no divide-by-zero
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_free_agent_pool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.services.fa_pool_service'`.

- [ ] **Step 3: Add the contracts**

Append to `api/contracts/free_agents.py` (after the existing classes):

```python
class StatItem(BaseModel):
    label: str
    value: str


class FreeAgentPoolItem(BaseModel):
    player: PlayerRef
    rank: int
    value: float  # marginal value normalized to 0-100 (top FA in pool = 100)
    own_pct: float = 0.0
    own_delta: float = 0.0
    hitter: bool = True
    stats: list[StatItem] = []
    fit: str = ""  # category key this FA most helps (e.g. "SB")
    tag: str | None = None


class FreeAgentPoolResponse(BaseModel):
    top_need: str = ""  # the user's biggest category gap
    free_agents: list[FreeAgentPoolItem] = []
```

- [ ] **Step 4: Create the service + helpers**

Create `api/services/fa_pool_service.py`:

```python
"""Free-agent POOL service — ALL available FAs ranked by marginal value
(distinct from the add/drop recommendations in fa_service). Powers the
Players page. Resilient: cold env / no data → empty response.

Composes the existing engine: build_optimizer_context → rank_free_agents
(src.in_season) → map. Player identity reuses player_ref_from_pool; value is
the marginal SGP normalized to 0-100 (top FA = 100)."""

from __future__ import annotations

import logging

from api.contracts.free_agents import FreeAgentPoolItem, FreeAgentPoolResponse, StatItem
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# (display label, source column, format kind). Presentation choice; rate stats
# go through format_stat, counting stats render as integers.
_HITTER_STATS = (("HR", "ytd_hr", "int"), ("SB", "ytd_sb", "int"), ("AVG", "ytd_avg", "AVG"))
_PITCHER_STATS = (("K", "ytd_k", "int"), ("ERA", "ytd_era", "ERA"), ("WHIP", "ytd_whip", "WHIP"))


def _tag_from(regression_flag) -> str | None:
    flag = str(regression_flag or "").upper()
    if flag == "BUY_LOW":
        return "Buy Low"
    if flag == "SELL_HIGH":
        return "Sell High"
    return None


def _top_need(category_gaps: dict) -> str:
    """User's biggest need = the most-negative gap (gap < 0 ⇒ behind). '' if none."""
    if not category_gaps:
        return ""
    return min(category_gaps, key=category_gaps.get)


def _key_stats(row, hitter: bool) -> list[StatItem]:
    from src.ui_shared import format_stat

    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    spec = _HITTER_STATS if hitter else _PITCHER_STATS
    out: list[StatItem] = []
    for label, col, kind in spec:
        raw = g(col)
        try:
            fval = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            fval = 0.0
        value = format_stat(fval, kind) if kind != "int" else str(int(round(fval)))
        out.append(StatItem(label=label, value=value))
    return out


def _to_pool_item(rank: int, row, full_pool, max_value: float) -> FreeAgentPoolItem:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    pid = int(g("player_id", 0) or 0)
    hitter = bool(g("is_hitter", True))
    mval = float(g("marginal_value", 0.0) or 0.0)
    value = round(max(0.0, min(100.0, mval / max_value * 100.0)), 1) if max_value and max_value > 0 else 0.0
    return FreeAgentPoolItem(
        player=player_ref_from_pool(pid, full_pool, name=g("player_name"), positions=g("positions")),
        rank=rank,
        value=value,
        own_pct=float(g("percent_owned", 0.0) or 0.0),
        own_delta=0.0,  # ownership-trend delta deferred (gap spec: not in API)
        hitter=hitter,
        stats=_key_stats(row, hitter),
        fit=str(g("best_category", "") or ""),
        tag=_tag_from(g("regression_flag")),
    )


class FreeAgentPoolService:
    def get_free_agents_pool(self, team_name: str, limit: int = 100) -> FreeAgentPoolResponse:
        try:
            from src.in_season import rank_free_agents
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            config = LeagueConfig()
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=config,
                user_team_name=team_name,
                level_filter="MLB only",
            )
            top_need = _top_need(ctx.category_gaps)
            if ctx.free_agents is None or ctx.free_agents.empty or ctx.player_pool.empty:
                return FreeAgentPoolResponse(top_need=top_need, free_agents=[])

            ranked = rank_free_agents(ctx.user_roster_ids, ctx.free_agents, ctx.player_pool, config)
            if ranked is None or ranked.empty:
                return FreeAgentPoolResponse(top_need=top_need, free_agents=[])

            # Bring Yahoo ownership % onto the ranked rows (rank_free_agents doesn't carry it).
            if "percent_owned" in ctx.free_agents.columns and "percent_owned" not in ranked.columns:
                ranked = ranked.merge(
                    ctx.free_agents[["player_id", "percent_owned"]].drop_duplicates("player_id"),
                    on="player_id",
                    how="left",
                )
            max_value = float(ranked["marginal_value"].max() or 0.0)
            items = [
                _to_pool_item(i + 1, row, ctx.player_pool, max_value)
                for i, row in enumerate(ranked.head(limit).to_dict("records"))
            ]
            return FreeAgentPoolResponse(top_need=top_need, free_agents=items)
        except Exception as exc:
            logger.warning("FreeAgentPoolService.get_free_agents_pool failed: %s", exc)
            return FreeAgentPoolResponse(top_need="", free_agents=[])
```

- [ ] **Step 5: Run the helper tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_free_agent_pool.py -q`
Expected: PASS (5 helper tests). (The fake-service test is added in Task 2.)

NOTE: `format_stat(0.285, "AVG")` returns `"0.285"` and `format_stat(3.21, "ERA")` returns `"3.21"` (canonical formatter). Confirm by reading `src/ui_shared.py::format_stat` if a test value mismatches.

- [ ] **Step 6: Commit**

```bash
git add api/contracts/free_agents.py api/services/fa_pool_service.py tests/api/test_api_free_agent_pool.py
git commit -m "feat(api): FA-pool contracts + ranking/mapping helpers (FreeAgentPoolService)"
```

---

### Task 2: Wire the endpoint (route + DI + fake-service test + openapi)

**Files:**
- Modify: `api/deps.py`
- Modify: `api/routers/free_agents.py`
- Test: append to `tests/api/test_api_free_agent_pool.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing fake-service contract test**

Append to `tests/api/test_api_free_agent_pool.py`:

```python
def test_free_agents_pool_endpoint_returns_contract():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.free_agents import FreeAgentPoolItem, FreeAgentPoolResponse, StatItem
    from api.deps import get_fa_pool_service
    from api.main import app

    class _FakePoolService:
        def get_free_agents_pool(self, team_name, limit=100):
            return FreeAgentPoolResponse(
                top_need="SB",
                free_agents=[
                    FreeAgentPoolItem(
                        player=PlayerRef(
                            id=1, mlb_id=592450, name="Aaron Judge", positions="OF",
                            team_abbr="NYY", team_id=147,
                        ),
                        rank=1, value=100.0, own_pct=45.0, own_delta=0.0, hitter=True,
                        stats=[StatItem(label="HR", value="24")], fit="HR", tag="Buy Low",
                    )
                ],
            )

    app.dependency_overrides[get_fa_pool_service] = lambda: _FakePoolService()
    try:
        resp = TestClient(app).get("/api/free-agents/pool?team_name=Team+Hickey")
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_need"] == "SB"
        fa = body["free_agents"][0]
        assert fa["player"]["mlb_id"] == 592450
        assert fa["player"]["team_id"] == 147
        assert fa["value"] == 100.0
        assert fa["fit"] == "HR"
        assert fa["stats"][0] == {"label": "HR", "value": "24"}
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_free_agent_pool.py::test_free_agents_pool_endpoint_returns_contract -q`
Expected: FAIL — `ImportError: cannot import name 'get_fa_pool_service'` (and the route 404s).

- [ ] **Step 3: Add the DI provider**

In `api/deps.py`: add the import alongside the others:
```python
from api.services.fa_pool_service import FreeAgentPoolService
```
and the provider (near `get_fa_service`):
```python
def get_fa_pool_service() -> FreeAgentPoolService:
    return FreeAgentPoolService()
```

- [ ] **Step 4: Add the route**

In `api/routers/free_agents.py`: extend the imports and add the route (the router already has `prefix="/api"` and is already mounted in `api/main.py` — NO main.py change):

```python
from api.contracts.free_agents import FreeAgentPoolResponse, FreeAgentsResponse
from api.deps import get_fa_pool_service, get_fa_service
```

```python
@router.get("/free-agents/pool", response_model=FreeAgentPoolResponse)
def get_free_agents_pool(
    team_name: str, limit: int = 100, service=Depends(get_fa_pool_service)
) -> FreeAgentPoolResponse:
    return service.get_free_agents_pool(team_name, limit)
```

- [ ] **Step 5: Run the fake-service test to verify it passes**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_free_agent_pool.py -q`
Expected: PASS (all 6 tests).

- [ ] **Step 6: Regenerate the OpenAPI snapshot**

The new route + models change the schema.
Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe scripts/export_openapi.py`
Then: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS. Confirm `api/openapi.json` now contains the `/api/free-agents/pool` path and the `FreeAgentPoolItem`/`FreeAgentPoolResponse`/`StatItem` schemas.

- [ ] **Step 7: Commit**

```bash
git add api/deps.py api/routers/free_agents.py api/openapi.json tests/api/test_api_free_agent_pool.py
git commit -m "feat(api): mount GET /api/free-agents/pool (ranked FA pool for the Players page)"
```

---

### Task 3: Full api suite + lint

**Files:** none (verification).

- [ ] **Step 1: Run the full api suite**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS (prior 109 + the new tests). The `test_no_logic_in_routers` guard must stay green (the new route only calls the service — no logic).

- [ ] **Step 2: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: both clean.

- [ ] **Step 3: Commit (only if Step 2 reformatted anything)**

```bash
git add -A && git commit -m "chore(api): ruff format FA-pool files"
```

---

## Self-Review (completed by plan author)

**Spec coverage** (gap spec §5 Players): a ranked-pool endpoint returning all FAs by value ✔; per-FA `rank`/`value`(0-100)/`own_pct`/`hitter`/`stats`(3)/`fit`/`tag` ✔; `mlb_id`/`team_abbr`/`team_id` via nested `PlayerRef` ✔; `top_need` ✔. `own_delta` defaulted to 0.0 + `tag` limited to regression flags — documented as deferred (matches the gap spec's "not in API" notes).

**Pattern fidelity:** contract → service (the one place importing `src/`) → thin route on the existing router → DI provider → fake-service `test_api_*` override test → openapi regen. Routers stay logic-free. Reuses `player_ref_from_pool` + `rank_free_agents` (no new engine logic; `src/` untouched).

**Placeholder scan:** every code step shows full code. The two "verify" notes (format_stat outputs; router prefix) are confirmations, not gaps. ✔

**Type consistency:** `_to_pool_item(rank, row, full_pool, max_value)`, `_key_stats(row, hitter)`, `_top_need(category_gaps)`, `_tag_from(flag)` are used identically across tasks. Contract field names (`top_need`, `free_agents`, `own_pct`, `own_delta`, `fit`) match between the models, the service, and the tests. Team-id 147 (NYY) matches `_STATSAPI_TEAM_IDS`. ✔
