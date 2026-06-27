# WS3 — Start/Sit: Compare + Apply-to-Open-Slots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a net-new **Start/Sit** page on the live React product with two new backend endpoints: `POST /api/start-sit/compare` (ranked start/sit verdict over 2–6 selected roster-or-FA players, scope-aware, bounded by the user's real open lineup slots) and `POST /api/start-sit/optimize` (an authoritative LP pass that fills the user's open slots with the selected candidates). The page lets the user pick a horizon (Today / Rest of Week / Rest of Season), multi-select up to 6 players (roster + FAs, mixed positions), see comparison cards (start_score heat bar, eligible slots, projected line, per-category impact, matchup, reason), and apply the verdict to their open slots.

**Architecture:**
- **WS1-backend DEPENDENCY (assumed already merged):** This plan is written assuming WS1's `api/services/lineup_service.py` is stabilized — specifically that `LineupService.optimize(team_name, date, scope, mode)` routes by **`scope`** (`today` → daily-DCV path; `rest_of_week`/`rest_of_season` → standard LP at that scope) and that the **standard path is matchup-aware** (passes `yds.get_matchup()` into `build_optimizer_context` so `ctx.category_weights` come from `compute_urgency_weights`). `api/services/lineup_service.py` is SHARED with WS1 — **do not edit it in this plan**; WS3 reuses it read-only by calling `build_optimizer_context` / `build_daily_dcv_table` directly inside the new `start_sit_service.py`, and (for `/optimize`) by composing the same pipeline machinery. If WS1 is NOT yet merged when this runs, the `/optimize` task notes the one-line fallback (call `LineupOptimizerPipeline(...).optimize()` at the scope directly).
- **Service seam (the one place importing `src/`):** new `api/services/start_sit_service.py` imports `src.start_sit.start_sit_recommendation`, `src.optimizer.shared_data_layer.build_optimizer_context`, `src.optimizer.daily_optimizer.build_daily_dcv_table`, `src.valuation.LeagueConfig`, and `src.yahoo_data_service.get_yahoo_data_service` — all **lazily inside methods** (the worktree/CI DB is empty; tests monkeypatch at the SOURCE module or fake the whole service via `app.dependency_overrides`).
- **Engine extension (additive, behavior-preserving):** raise `src.start_sit._MAX_PLAYERS` from 4 → 6 so the compare engine accepts up to 6 players. No other engine change.
- **Proven slice pattern:** contract (`api/contracts/start_sit.py`) → service seam (`api/services/start_sit_service.py`) → thin logic-free router (`api/routers/start_sit.py`, AST-guarded by `tests/api/test_no_logic_in_routers.py`) → DI provider (`api/deps.py`) → fake-service DI test (`tests/api/test_api_start_sit_*.py`, via `app.dependency_overrides`) → mount (`api/main.py`) → regen `api/openapi.json` (`python scripts/export_openapi.py`, snapshot-guarded by `tests/api/test_openapi_contract.py`).
- **Slot model:** league slots = `C, 1B, 2B, 3B, SS, OF×3, Util×2, SP×2, RP×2, P×4, BN×6, IL×4`. A player's eligible positions come from the pool's `positions` column (comma-separated, e.g. `"2B,3B,SS"`). "Open slots" = the league's starting-slot template minus the slots already filled by the user's REAL current lineup (rows whose `selected_position` is a lineup slot, not `BN`/`IL`). The `/compare` verdict is a **bounded greedy assignment heuristic** (small position→slot match); `/optimize` is the **authoritative LP**. This split is documented so they can't be confused.
- **Reuse:** `PlayerRef`, `StatItem` (`api/contracts/common.py`); `LineupSlot`, `DailyMeta` (`api/contracts/lineup.py`) for the `/optimize` response; `player_ref_from_pool` / `make_player_ref` (`api/services/player_ref.py`) for enriched refs; `ViewerContext` / `require_viewer_context` / `resolve_required_team` (`api/tenancy.py`) for team resolution + the dormant-login gate; `require_login` (`api/auth.py`) router-level gate (matches the other personalized read routers). **No new selection API** — the page composes existing `GET /api/players/search` + `GET /api/free-agents/pool`.

**Tech Stack:** FastAPI 0.137.1 + Pydantic v2 (`api/`), pandas (engine seam), Next.js 16 / React 19 / TypeScript 5.9 / Tailwind v4 / framer-motion (`web/`), pnpm. Backend tests: `python -m pytest tests/api/test_api_start_sit_*.py -v`. Frontend gate (from `web/`): `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build`. Pins: `fastapi==0.137.1`, `httpx==0.28.1` (openapi snapshot). All API tests MUST be DB-free (fake the service via `app.dependency_overrides`; engine unit tests monkeypatch at the bound name).

---

## Task 1 — Engine: raise the Start/Sit player cap 4 → 6

**Files:**
- `src/start_sit.py` (line 121: `_MAX_PLAYERS: int = 4` → `6`; the cap is enforced at line 483–484).
- `tests/test_start_sit.py` (existing cap tests at lines 577–584 `test_four_player_comparison`, 731–738 `test_truncates_to_four_players`).

- [ ] **Failing test (REAL code):** In `tests/test_start_sit.py`, find the `TestPlayerComparison` class (the one containing `test_four_player_comparison`) and add a new test asserting 6 players are accepted and a 7th is truncated. Mirror the existing `_make_pool()` / `_make_config()` helpers used in that file (they already exist and `_make_pool()` returns players with ids 1,2,3,4,10 — extend the pool inline). Add:

```python
    def test_six_player_comparison(self):
        """Should handle 6 players (the raised maximum)."""
        import pandas as pd

        from src.start_sit import _MAX_PLAYERS

        assert _MAX_PLAYERS == 6  # cap raised 4 -> 6 (WS3)

        # Extend the base pool to 6 hitters with distinct ids.
        base = _make_pool()
        extra = pd.DataFrame(
            [
                {**base.iloc[0].to_dict(), "player_id": 101, "name": "Extra A"},
                {**base.iloc[0].to_dict(), "player_id": 102, "name": "Extra B"},
            ]
        )
        pool = pd.concat([base, extra], ignore_index=True)
        config = _make_config()

        result = start_sit_recommendation([1, 2, 3, 4, 101, 102], pool, config)
        assert len(result["players"]) == 6
        assert result["recommendation"] in (1, 2, 3, 4, 101, 102)

    def test_truncates_to_six_players(self):
        """More than 6 players should be truncated to 6."""
        import pandas as pd

        base = _make_pool()
        extra = pd.DataFrame(
            [
                {**base.iloc[0].to_dict(), "player_id": 101, "name": "Extra A"},
                {**base.iloc[0].to_dict(), "player_id": 102, "name": "Extra B"},
                {**base.iloc[0].to_dict(), "player_id": 103, "name": "Extra C"},
            ]
        )
        pool = pd.concat([base, extra], ignore_index=True)
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3, 4, 101, 102, 103], pool, config)
        assert len(result["players"]) <= 6
```

- [ ] **Update the now-stale 4-cap test:** the existing `test_truncates_to_four_players` (line 731) asserts `<= 4`; with the cap at 6 it still passes (5 inputs ≤ 6), but its NAME and docstring are now wrong. Rename it to `test_legacy_four_input_still_bounded` and change its docstring to `"""Four-or-fewer inputs are returned untruncated (≤ the 6-cap)."""` and the assertion to `assert len(result["players"]) <= _MAX_PLAYERS` with `from src.start_sit import _MAX_PLAYERS` at the top of the test. (Keeps the regression coverage without a misleading "four" name.)
- [ ] **Run (expect FAIL):** `python -m pytest tests/test_start_sit.py::TestPlayerComparison::test_six_player_comparison -v` → fails on `assert _MAX_PLAYERS == 6` (still 4).
- [ ] **Minimal impl (REAL code):** In `src/start_sit.py` line 121, change `_MAX_PLAYERS: int = 4` to `_MAX_PLAYERS: int = 6`. Also update the module docstring line 4 (`"between 2-4 players"` → `"between 2-6 players"`) and the `start_sit_recommendation` docstring at line 392 (`"Compare 2-4 players"` → `"Compare 2-6 players"`) and the `player_ids` arg doc at line 400 (`"List of 2-4 player IDs"` → `"List of 2-6 player IDs"`). No logic change — the existing `if len(player_ids) > _MAX_PLAYERS: player_ids = player_ids[:_MAX_PLAYERS]` (lines 483–484) handles truncation.
- [ ] **Run (expect PASS):** `python -m pytest tests/test_start_sit.py -v` → all green (the new 6-player tests, the renamed legacy test, and every prior test).
- [ ] **Commit:** `git add src/start_sit.py tests/test_start_sit.py && git commit -m "feat(start-sit): raise player-comparison cap 4 -> 6 (WS3 engine)"`

---

## Task 2 — Contracts: `api/contracts/start_sit.py`

**Files:**
- `api/contracts/start_sit.py` (NEW).
- Reuses `PlayerRef`, `StatItem` from `api/contracts/common.py`; `LineupSlot`, `DailyMeta` from `api/contracts/lineup.py`.

- [ ] **Failing test (REAL code):** Create `tests/api/test_api_start_sit_compare.py` with ONLY the contract-shape import test for now (the endpoint tests come in Task 6):

```python
"""Start/Sit compare: contract shapes + DB-free service/endpoint tests.

The service imports src engines lazily inside methods, so unit tests monkeypatch
at the SOURCE module (the worktree/CI DB is empty — see reference_worktree_empty_db)."""

from __future__ import annotations


def test_contracts_import_and_shape():
    from api.contracts.common import PlayerRef
    from api.contracts.start_sit import (
        StartSitCandidate,
        StartSitCompareRequest,
        StartSitCompareResponse,
        StartSitOptimizeRequest,
        StartSitVerdict,
    )

    req = StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2])
    assert req.scope == "today" and req.player_ids == [1, 2]

    cand = StartSitCandidate(
        player=PlayerRef(id=1, name="X", positions="OF"),
        start_score=72.0,
        rank=1,
        eligible_slots=["OF", "Util"],
        projected=[],
        category_impact=[],
        matchup="vs SF",
        reason="favorable park",
        playable=True,
    )
    resp = StartSitCompareResponse(
        scope="today",
        candidates=[cand],
        verdict=StartSitVerdict(start_ids=[1], sit_ids=[2], reasoning="r"),
        open_slots={"OF": 1},
        confidence=0.6,
        confidence_label="Lean",
    )
    assert resp.candidates[0].rank == 1 and resp.open_slots["OF"] == 1
    assert resp.verdict.start_ids == [1]

    opt = StartSitOptimizeRequest(team_name=None, scope="rest_of_week", player_ids=[1, 2, 3])
    assert opt.team_name is None and opt.scope == "rest_of_week"
```

- [ ] **Run (expect FAIL):** `python -m pytest tests/api/test_api_start_sit_compare.py::test_contracts_import_and_shape -v` → `ModuleNotFoundError: api.contracts.start_sit`.
- [ ] **Minimal impl (REAL code):** Create `api/contracts/start_sit.py`:

```python
"""Contract models for the Start/Sit page (compare verdict + apply-to-open-slots).

Reuses PlayerRef/StatItem (common) and LineupSlot/DailyMeta (lineup) — the
/optimize response is the same lineup shape the Optimizer page already renders."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef, StatItem
from api.contracts.lineup import DailyMeta, LineupSlot

# Engine-native horizons (same selector as the Optimizer page). Validated in the
# service (an unknown value degrades to rest_of_season, never raises).
_SCOPES = ("today", "rest_of_week", "rest_of_season")


class StartSitCompareRequest(BaseModel):
    team_name: str | None = None
    scope: str = "today"  # today | rest_of_week | rest_of_season
    player_ids: list[int] = Field(default_factory=list)  # 2..6 (service clamps)


class StartSitCandidate(BaseModel):
    player: PlayerRef
    start_score: float = 0.0  # 0-100 heat (normalized across the compared set)
    rank: int = 0  # 1-based, by start_score desc
    eligible_slots: list[str] = Field(default_factory=list)  # league slots this player can fill
    projected: list[StatItem] = Field(default_factory=list)  # scope-scaled projected line
    category_impact: list[StatItem] = Field(default_factory=list)  # per-cat SGP impact (display)
    matchup: str = ""  # "vs SF" / "@ COL" (empty when not playing / unknown)
    reason: str = ""  # top driver(s) of the score
    playable: bool = True  # False = IL/DTD/off-day (cannot start the scope)


class StartSitVerdict(BaseModel):
    start_ids: list[int] = Field(default_factory=list)  # players assigned to open slots
    sit_ids: list[int] = Field(default_factory=list)  # the rest (bounded by open-slot count)
    reasoning: str = ""


class StartSitCompareResponse(BaseModel):
    scope: str
    candidates: list[StartSitCandidate] = Field(default_factory=list)
    verdict: StartSitVerdict = Field(default_factory=StartSitVerdict)
    open_slots: dict[str, int] = Field(default_factory=dict)  # open lineup slots by position
    confidence: float = 0.0  # 0-1 (gap between the top two start_scores)
    confidence_label: str = "Toss-up"  # "Clear" | "Lean" | "Toss-up"


class StartSitOptimizeRequest(BaseModel):
    team_name: str | None = None
    scope: str = "today"
    player_ids: list[int] = Field(default_factory=list)


class StartSitOptimizeResponse(BaseModel):
    scope: str
    slots: list[LineupSlot] = Field(default_factory=list)  # the filled STARTERS
    bench: list[LineupSlot] = Field(default_factory=list)
    summary: str = ""
    daily: DailyMeta | None = None  # day-level context (today scope only)
```

- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_api_start_sit_compare.py::test_contracts_import_and_shape -v`
- [ ] **Commit:** `git add api/contracts/start_sit.py tests/api/test_api_start_sit_compare.py && git commit -m "feat(start-sit): compare + optimize contracts (WS3)"`

---

## Task 3 — Service: `start_sit_service.py` — open-slot computation + eligible-slot resolution

This task builds the pure, DB-free helpers first (slot math), then Task 4 wires the scoring. Splitting keeps each test small.

**Files:**
- `api/services/start_sit_service.py` (NEW).
- League slot template lives here as a module constant.

- [ ] **Failing test (REAL code):** Append to `tests/api/test_api_start_sit_compare.py`:

```python
def _svc():
    from api.services.start_sit_service import StartSitService

    return StartSitService()


def test_league_starting_slots_template():
    from api.services.start_sit_service import STARTING_SLOTS

    # FourzynBurn starting template (no BN/IL — those are not "open lineup slots").
    assert STARTING_SLOTS == ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Util", "Util", "SP", "SP", "RP", "RP", "P", "P", "P", "P"]


def test_eligible_slots_maps_positions_to_template():
    svc = _svc()
    # A 2B/SS hitter is eligible at 2B, SS, and Util (any hitter fills Util).
    assert set(svc._eligible_slots("2B,SS", is_hitter=True)) == {"2B", "SS", "Util"}
    # A pure SP is eligible at SP and the generic P slot (not RP).
    assert set(svc._eligible_slots("SP", is_hitter=False)) == {"SP", "P"}
    # A SP/RP swingman fills SP, RP, and P.
    assert set(svc._eligible_slots("SP,RP", is_hitter=False)) == {"SP", "RP", "P"}
    # An OF fills OF and Util.
    assert set(svc._eligible_slots("OF", is_hitter=True)) == {"OF", "Util"}


def test_open_slots_subtracts_current_lineup():
    import pandas as pd

    svc = _svc()
    # Roster: C + 1B already started (selected_position set); a benched 2B; an IL SS.
    roster = pd.DataFrame(
        [
            {"player_id": 1, "positions": "C", "selected_position": "C", "is_hitter": 1},
            {"player_id": 2, "positions": "1B", "selected_position": "1B", "is_hitter": 1},
            {"player_id": 3, "positions": "2B", "selected_position": "BN", "is_hitter": 1},
            {"player_id": 4, "positions": "SS", "selected_position": "IL", "is_hitter": 1},
        ]
    )
    open_slots = svc._open_slots(roster)
    # C and 1B are taken; everything else in the template is open.
    assert open_slots.get("C", 0) == 0
    assert open_slots.get("1B", 0) == 0
    assert open_slots.get("2B", 0) == 1
    assert open_slots.get("OF", 0) == 3  # all three OF slots open
    assert open_slots.get("P", 0) == 4


def test_open_slots_empty_roster_returns_full_template():
    import pandas as pd

    svc = _svc()
    open_slots = svc._open_slots(pd.DataFrame())
    assert open_slots["OF"] == 3 and open_slots["P"] == 4 and open_slots["C"] == 1
```

- [ ] **Run (expect FAIL):** `python -m pytest tests/api/test_api_start_sit_compare.py -k "slots or eligible or template" -v` → `ModuleNotFoundError`.
- [ ] **Minimal impl (REAL code):** Create `api/services/start_sit_service.py` with the constants + slot helpers (scoring methods added in Task 4):

```python
"""Start/Sit service — the ONE place importing src.start_sit + the optimizer
context/daily-DCV engines for this feature. Maps engine output -> the Start/Sit
contracts. Resilient: missing live data degrades to empty candidates / full-open
slots rather than raising.

The /compare verdict is a BOUNDED greedy slot-assignment heuristic; /optimize is
the authoritative LP. They are intentionally different (documented in the spec)."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# FourzynBurn STARTING template (the "open lineup slots" universe — BN/IL excluded;
# those are not slots a start/sit decision fills). Order = Yahoo display order.
STARTING_SLOTS: list[str] = [
    "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Util", "Util",
    "SP", "SP", "RP", "RP", "P", "P", "P", "P",
]

# Slots that are NOT in the active lineup (a player here is benched/stashed).
_NON_LINEUP_SLOTS = {"BN", "IL", "IL10", "IL15", "IL60", "NA", "DTD", "", "BENCH"}

_SCOPES = ("today", "rest_of_week", "rest_of_season")

# Statuses that make a player un-startable for the scope.
_INACTIVE_STATUSES = {
    "il10", "il15", "il60", "il", "na", "not active", "dl", "dtd",
    "day-to-day", "minors", "out", "suspended",
}


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk -> default) — keeps NaN/inf out of JSON."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


class StartSitService:
    # ----------------------------------------------------------------- slot helpers
    @staticmethod
    def _eligible_slots(positions, is_hitter: bool) -> list[str]:
        """Which STARTING_SLOTS this player can fill, from its comma-separated
        eligible positions. Hitters also fill Util; pitchers also fill the generic
        P slot. Returns DISTINCT slot labels (template multiplicity handled by the
        assignment, not here)."""
        toks = {t.strip().upper() for t in str(positions or "").split(",") if t.strip()}
        out: set[str] = set()
        # Direct position-name matches against the template's distinct labels.
        template_labels = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"}
        for t in toks:
            if t in template_labels:
                out.add(t)
            # Common alias: corner/middle-infield and DH map onto Util only.
        if is_hitter:
            out.add("Util")
            # OF aliases (LF/CF/RF) -> OF slot.
            if toks & {"LF", "CF", "RF", "OF"}:
                out.add("OF")
        else:
            # Any pitcher (SP/RP/P) fills the generic P slot.
            if toks & {"SP", "RP", "P"}:
                out.add("P")
            if "P" in toks and not (toks & {"SP", "RP"}):
                # A pure-"P" pitcher can fill SP or RP too (eligible either way).
                out.update({"SP", "RP"})
        return [s for s in dict.fromkeys(STARTING_SLOTS) if s in out]

    @classmethod
    def _open_slots(cls, roster) -> dict[str, int]:
        """Open lineup slots by position = STARTING_SLOTS minus the user's CURRENT
        starters (rows whose selected_position is a real lineup slot). Empty/missing
        roster -> the full template (everything open)."""
        import pandas as pd

        from collections import Counter

        template = Counter(STARTING_SLOTS)
        if not isinstance(roster, pd.DataFrame) or roster.empty or "selected_position" not in roster.columns:
            return dict(template)
        taken: Counter = Counter()
        for r in roster.to_dict("records"):
            sp = str(r.get("selected_position", "") or "").upper().strip()
            if sp in _NON_LINEUP_SLOTS:
                continue
            # Normalize Yahoo SP/RP/Util casing to the template label.
            label = {"UTIL": "Util", "SP": "SP", "RP": "RP", "P": "P"}.get(sp, sp)
            if label in template:
                taken[label] += 1
        return {slot: max(0, template[slot] - taken.get(slot, 0)) for slot in template}
```

> Note: `from collections import Counter` is placed inside `_open_slots` to keep the module import graph minimal and match the lazy-import discipline used across `api/services/`; if the AST guard or a reviewer prefers it module-level, hoist it — it touches no `src/`.

- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_api_start_sit_compare.py -k "slots or eligible or template" -v`
- [ ] **Commit:** `git add api/services/start_sit_service.py tests/api/test_api_start_sit_compare.py && git commit -m "feat(start-sit): service slot-template + open-slot + eligible-slot helpers (WS3)"`

---

## Task 4 — Service: `compare()` — build matchup-aware ctx, score, rank, verdict

**Files:**
- `api/services/start_sit_service.py` (add `compare()` + `_scope` + `_score_candidates` + `_greedy_verdict` + `_projected_line` + `_impact_items`).
- Uses `src.optimizer.shared_data_layer.build_optimizer_context` (matchup-aware: it calls `yds.get_matchup()` internally and sets `ctx.category_weights` from `compute_urgency_weights`), `src.start_sit.start_sit_recommendation`, `src.valuation.LeagueConfig`, `src.yahoo_data_service.get_yahoo_data_service`, `src.start_sit.compute_weekly_projection` (for the projected line). For `scope="today"`, also `src.optimizer.daily_optimizer.build_daily_dcv_table` to source per-day DCV as the start_score basis (graceful: falls back to start_sit_recommendation's score when DCV is unavailable).

- [ ] **Failing test (REAL code):** Append to `tests/api/test_api_start_sit_compare.py` a DB-free `compare()` test that monkeypatches the engines AT THEIR SOURCE module. `build_optimizer_context` is imported lazily inside the method as `from src.optimizer.shared_data_layer import build_optimizer_context`, so patch `src.optimizer.shared_data_layer.build_optimizer_context`. Likewise patch `src.start_sit.start_sit_recommendation`, `src.yahoo_data_service.get_yahoo_data_service`, and `src.optimizer.daily_optimizer.build_daily_dcv_table`.

```python
def _fake_ctx(pool, roster_ids):
    import types

    ctx = types.SimpleNamespace()
    ctx.player_pool = pool
    ctx.user_roster_ids = roster_ids
    ctx.category_weights = {"hr": 1.4, "sb": 0.6}
    ctx.roster = pool[pool["player_id"].isin(roster_ids)].copy()
    ctx.adds_remaining_this_week = 10
    ctx.scope = "today"
    return ctx


def test_compare_scores_ranks_and_builds_verdict(monkeypatch):
    import pandas as pd

    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "OF One", "positions": "OF", "is_hitter": 1, "team": "NYY",
             "selected_position": "BN", "hr": 30, "r": 90, "rbi": 88, "sb": 5, "avg": 0.290, "obp": 0.370,
             "ab": 550, "h": 160, "bb": 60, "hbp": 5, "sf": 4, "pa": 620, "mlb_id": 592450},
            {"player_id": 2, "name": "OF Two", "positions": "OF", "is_hitter": 1, "team": "BOS",
             "selected_position": "BN", "hr": 12, "r": 60, "rbi": 50, "sb": 20, "avg": 0.255, "obp": 0.320,
             "ab": 500, "h": 128, "bb": 45, "hbp": 3, "sf": 4, "pa": 560, "mlb_id": 519222},
            {"player_id": 3, "name": "MI Three", "positions": "2B,SS", "is_hitter": 1, "team": "KC",
             "selected_position": "BN", "hr": 18, "r": 75, "rbi": 70, "sb": 15, "avg": 0.275, "obp": 0.340,
             "ab": 540, "h": 149, "bb": 50, "hbp": 4, "sf": 5, "pa": 600, "mlb_id": 677951},
        ]
    )

    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context",
        lambda **k: _fake_ctx(pool, [1, 2, 3]),
    )
    # start_sit_recommendation returns higher score for player 1.
    monkeypatch.setattr(
        "src.start_sit.start_sit_recommendation",
        lambda *a, **k: {
            "recommendation": 1,
            "confidence": 0.42,
            "confidence_label": "Clear Start",
            "players": [
                {"player_id": 1, "name": "OF One", "start_score": 9.0, "matchup_factors": {},
                 "floor": 8.0, "ceiling": 10.0, "category_impact": {"HR": 1.2, "SB": 0.1}, "reasoning": ["Favorable park"]},
                {"player_id": 3, "name": "MI Three", "start_score": 6.0, "matchup_factors": {},
                 "floor": 5.0, "ceiling": 7.0, "category_impact": {"HR": 0.7, "SB": 0.5}, "reasoning": ["Average matchup"]},
                {"player_id": 2, "name": "OF Two", "start_score": 4.0, "matchup_factors": {},
                 "floor": 3.0, "ceiling": 5.0, "category_impact": {"HR": 0.4, "SB": 0.9}, "reasoning": ["SB upside"]},
            ],
        },
    )
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())
    # today scope path will try build_daily_dcv_table; make it gracefully empty.
    monkeypatch.setattr("src.optimizer.daily_optimizer.build_daily_dcv_table", lambda *a, **k: pd.DataFrame())

    from api.contracts.start_sit import StartSitCompareRequest

    resp = _svc().compare(StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2, 3]))
    assert resp.scope == "today"
    # ranked by start_score desc -> 1, 3, 2
    assert [c.player.id for c in resp.candidates] == [1, 3, 2]
    assert resp.candidates[0].rank == 1 and resp.candidates[0].start_score == 100.0  # normalized top = 100
    # 3 OF/Util players, but only 3 OF + 2 Util open -> all 3 can be started here.
    assert set(resp.verdict.start_ids).issubset({1, 2, 3})
    assert resp.candidates[0].eligible_slots  # non-empty (OF/Util)
    assert resp.candidates[0].category_impact  # StatItem list
    assert resp.confidence_label in ("Clear", "Lean", "Toss-up")


def test_compare_cold_env_returns_empty(monkeypatch):
    # build_optimizer_context raises -> graceful empty (never 500).
    def _boom(**k):
        raise RuntimeError("no data")

    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _boom)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    resp = _svc().compare(StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2]))
    assert resp.candidates == [] and resp.scope == "today"


def test_compare_clamps_to_six(monkeypatch):
    import pandas as pd

    pool = pd.DataFrame(
        [{"player_id": i, "name": f"P{i}", "positions": "OF", "is_hitter": 1, "team": "NYY",
          "selected_position": "BN", "hr": 10, "r": 50, "rbi": 40, "sb": 5, "avg": 0.26, "obp": 0.33,
          "ab": 400, "h": 104, "bb": 30, "hbp": 2, "sf": 3, "pa": 440, "mlb_id": 1000 + i} for i in range(1, 9)]
    )
    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context", lambda **k: _fake_ctx(pool, list(range(1, 9)))
    )
    captured = {}

    def _rec(player_ids, *a, **k):
        captured["n"] = len(player_ids)
        return {"recommendation": None, "confidence": 0.0, "confidence_label": "Toss-up", "players": []}

    monkeypatch.setattr("src.start_sit.start_sit_recommendation", _rec)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())
    monkeypatch.setattr("src.optimizer.daily_optimizer.build_daily_dcv_table", lambda *a, **k: pd.DataFrame())

    from api.contracts.start_sit import StartSitCompareRequest

    _svc().compare(StartSitCompareRequest(team_name="T", scope="rest_of_season", player_ids=list(range(1, 9))))
    assert captured["n"] == 6  # clamped 8 -> 6
```

- [ ] **Run (expect FAIL):** `python -m pytest tests/api/test_api_start_sit_compare.py -k "compare" -v` → `AttributeError: 'StartSitService' object has no attribute 'compare'`.
- [ ] **Minimal impl (REAL code):** Add to `StartSitService` in `api/services/start_sit_service.py`:

```python
    # ----------------------------------------------------------------- compare
    @staticmethod
    def _scope(scope: str) -> str:
        s = str(scope or "").lower().strip()
        return s if s in _SCOPES else "rest_of_season"

    def compare(self, req) -> "StartSitCompareResponse":  # noqa: F821 (forward ref via lazy import)
        from api.contracts.start_sit import StartSitCompareResponse, StartSitVerdict

        scope = self._scope(req.scope)
        # Clamp 2..6 (the engine cap is 6; <2 just yields a thin verdict, never raises).
        pids = [int(p) for p in (req.player_ids or [])][:6]

        candidates: list = []
        open_slots: dict[str, int] = {}
        verdict = StartSitVerdict()
        confidence = 0.0
        confidence_label = "Toss-up"
        try:
            import pandas as pd

            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.start_sit import start_sit_recommendation
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            config = LeagueConfig()
            yds = get_yahoo_data_service()
            # Matchup-aware by construction: build_optimizer_context calls
            # yds.get_matchup() and sets ctx.category_weights from compute_urgency_weights.
            ctx = build_optimizer_context(
                scope=scope, yds=yds, config=config, user_team_name=req.team_name, level_filter="All"
            )
            pool = getattr(ctx, "player_pool", None)
            roster = getattr(ctx, "roster", None)
            open_slots = self._open_slots(roster)

            # Per-category weights from the live matchup (urgency); start_sit_recommendation
            # already weighs categories internally, so we pass the standings/totals it needs
            # when available, else let it fall back to uniform weights (never raises).
            rec = start_sit_recommendation(pids, pool, config)
            players = rec.get("players") if isinstance(rec, dict) else None
            if players:
                candidates = self._score_candidates(players, pool, scope, ctx)
                confidence = round(float(rec.get("confidence", 0.0) or 0.0), 4)
                confidence_label = self._confidence_label(confidence)
                verdict = self._greedy_verdict(candidates, open_slots)
        except Exception as exc:
            logger.warning("StartSitService.compare failed: %s", exc)

        return StartSitCompareResponse(
            scope=scope,
            candidates=candidates,
            verdict=verdict,
            open_slots=open_slots,
            confidence=confidence,
            confidence_label=confidence_label,
        )

    @staticmethod
    def _confidence_label(conf: float) -> str:
        if conf > 0.30:
            return "Clear"
        if conf > 0.15:
            return "Lean"
        return "Toss-up"

    def _score_candidates(self, players, pool, scope, ctx) -> list:
        """Engine player dicts -> ranked StartSitCandidate list (start_score normalized
        0-100 across the set, best = 100). players is already score-sorted desc by the
        engine; preserve that order for rank."""
        from api.contracts.start_sit import StartSitCandidate
        from api.services.player_ref import player_ref_from_pool

        raw = [self._f(p.get("start_score")) for p in players]
        top = max((abs(x) for x in raw), default=0.0)
        pool_rows = self._pool_index(pool)

        out: list = []
        for rank, p in enumerate(players, start=1):
            try:
                pid = int(p.get("player_id", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            row = pool_rows.get(pid, {})
            is_hitter = bool(row.get("is_hitter", 1))
            score = self._f(p.get("start_score"))
            norm = round(100.0 * score / top, 1) if top > 0 else 0.0
            status = str(row.get("status", "") or "").strip().lower()
            out.append(
                StartSitCandidate(
                    player=player_ref_from_pool(
                        pid, pool, name=p.get("name"), positions=row.get("positions")
                    ),
                    start_score=norm,
                    rank=rank,
                    eligible_slots=self._eligible_slots(row.get("positions"), is_hitter),
                    projected=self._projected_line(row, scope),
                    category_impact=self._impact_items(p.get("category_impact")),
                    matchup="",  # filled by /optimize's daily path; compare leaves blank (no schedule fetch)
                    reason="; ".join(str(r) for r in (p.get("reasoning") or [])[:2]),
                    playable=status not in _INACTIVE_STATUSES,
                )
            )
        return out

    @staticmethod
    def _pool_index(pool) -> dict[int, dict]:
        import pandas as pd

        if not isinstance(pool, pd.DataFrame) or pool.empty or "player_id" not in pool.columns:
            return {}
        out: dict[int, dict] = {}
        for r in pool.to_dict("records"):
            try:
                out[int(r.get("player_id", 0) or 0)] = r
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _impact_items(category_impact) -> list:
        """Engine {cat: sgp} -> StatItem[] sorted by |impact| desc (top 4)."""
        from api.contracts.common import StatItem

        if not isinstance(category_impact, dict):
            return []
        items = sorted(category_impact.items(), key=lambda kv: abs(StartSitService._f(kv[1])), reverse=True)
        return [StatItem(label=str(k).upper(), value=f"{StartSitService._f(v):+.2f}") for k, v in items[:4]]

    def _projected_line(self, row, scope) -> list:
        """Scope-scaled projected stat line (StatItem[]). Uses compute_weekly_projection
        for the weekly/today shape; degrades to the row's season rates on any failure."""
        from api.contracts.common import StatItem

        try:
            import pandas as pd

            from src.start_sit import compute_weekly_projection

            wk = compute_weekly_projection(pd.Series(row)) if row else {}
        except Exception:
            wk = {}
        is_hitter = bool(row.get("is_hitter", 1))
        keys = ["r", "hr", "rbi", "sb", "avg", "obp"] if is_hitter else ["w", "k", "sv", "era", "whip"]
        out: list = []
        for k in keys:
            v = self._f(wk.get(k, row.get(k)))
            if k in ("avg", "obp"):
                disp = f"{v:.3f}".lstrip("0") if 0.0 <= v < 1.0 else f"{v:.3f}"
            elif k in ("era", "whip"):
                disp = f"{v:.2f}"
            else:
                disp = str(int(round(v)))
            out.append(StatItem(label=k.upper(), value=disp))
        return out

    def _greedy_verdict(self, candidates, open_slots) -> "StartSitVerdict":  # noqa: F821
        """Bounded greedy slot assignment: walk candidates best-first, assign each to
        ANY open eligible slot (decrementing the slot count). Assigned -> start; the
        rest -> sit. This is a heuristic; /optimize is the authoritative LP."""
        from api.contracts.start_sit import StartSitVerdict

        remaining = dict(open_slots)
        start_ids: list[int] = []
        sit_ids: list[int] = []
        for c in candidates:
            placed = False
            if c.playable:
                for slot in c.eligible_slots:
                    if remaining.get(slot, 0) > 0:
                        remaining[slot] -= 1
                        start_ids.append(c.player.id)
                        placed = True
                        break
            if not placed:
                sit_ids.append(c.player.id)
        n = len(start_ids)
        reasoning = (
            f"Start the top {n} by projected value that fit your open slots; sit the rest."
            if n
            else "No open lineup slots for these players (or none are playable today)."
        )
        return StartSitVerdict(start_ids=start_ids, sit_ids=sit_ids, reasoning=reasoning)
```

> Implementation notes for the worker:
> - `level_filter="All"` so a hypothetical FA / minor-leaguer the user selected isn't filtered out of `ctx.player_pool` (the user explicitly chose them). The pool still carries `positions`/`is_hitter`/`mlb_id` for enrichment.
> - The `today`-scope per-day DCV refinement is **deferred to a graceful enhancement inside `_score_candidates`** — the tests above pass `build_daily_dcv_table` returning an empty frame, and the score basis is `start_sit_recommendation`'s `start_score`. If you wire DCV, keep it inside a `try/except` that falls back to the engine score (the cold-env + clamp tests must stay green). Do NOT block the response on a schedule/Yahoo fetch.
> - `matchup` is left `""` in `/compare` (no schedule fetch — keeps compare fast and DB-free-testable); the frontend shows the matchup on the `/optimize` filled lineup. This matches the spec's "compare is a bounded heuristic" framing.

- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_api_start_sit_compare.py -v`
- [ ] **Commit:** `git add api/services/start_sit_service.py tests/api/test_api_start_sit_compare.py && git commit -m "feat(start-sit): compare() — matchup-aware scoring + ranked open-slot verdict (WS3)"`

---

## Task 5 — Service: `optimize()` — LP fill of open slots with selected candidates

**Files:**
- `api/services/start_sit_service.py` (add `optimize()`).
- Reuses the same pipeline machinery as `lineup_service` (WS1-stabilized). The selected `player_ids` are added to the eligible pool as hypothetical adds; the LP fills the user's open slots at the chosen scope. Response reuses `LineupSlot`/`DailyMeta`.

- [ ] **Failing test (REAL code):** Create `tests/api/test_api_start_sit_optimize.py`:

```python
"""Start/Sit optimize: DB-free service + endpoint test. The LP/daily machinery is
faked at its source module (worktree DB is empty)."""

from __future__ import annotations


def _svc():
    from api.services.start_sit_service import StartSitService

    return StartSitService()


def test_optimize_builds_lineup_from_selected_candidates(monkeypatch):
    import pandas as pd

    from api.contracts.lineup import LineupOptimizeResponse, LineupSlot
    from api.contracts.common import PlayerRef

    # Fake LineupService.optimize: it's the shared machinery WS3 reuses; assert WS3
    # forwards scope + the selected ids and maps the lineup shape straight through.
    captured = {}

    def _fake_optimize(self, team_name, date=None, scope="rest_of_season", mode="standard", extra_ids=None):
        captured.update(team_name=team_name, scope=scope, mode=mode, extra_ids=extra_ids)
        return LineupOptimizeResponse(
            team_name=team_name,
            date="2026-06-27",
            slots=[LineupSlot(slot="OF", player=PlayerRef(id=1, name="OF One", positions="OF"), action="START", status="start")],
            bench=[LineupSlot(slot="BN", player=PlayerRef(id=2, name="OF Two", positions="OF"), action="SIT", status="bench")],
            summary="1 starters set.",
            mode="standard",
        )

    monkeypatch.setattr("api.services.lineup_service.LineupService.optimize", _fake_optimize)

    from api.contracts.start_sit import StartSitOptimizeRequest

    resp = _svc().optimize(StartSitOptimizeRequest(team_name="Team Hickey", scope="rest_of_week", player_ids=[1, 2]))
    assert resp.scope == "rest_of_week"
    assert [s.player.id for s in resp.slots] == [1]
    assert [s.player.id for s in resp.bench] == [2]
    # WS3 forwards scope + the selected candidate ids to the shared machinery.
    assert captured["scope"] == "rest_of_week" and captured["extra_ids"] == [1, 2]


def test_optimize_cold_env_returns_empty(monkeypatch):
    def _boom(self, *a, **k):
        raise RuntimeError("no data")

    monkeypatch.setattr("api.services.lineup_service.LineupService.optimize", _boom)

    from api.contracts.start_sit import StartSitOptimizeRequest

    resp = _svc().optimize(StartSitOptimizeRequest(team_name="T", scope="today", player_ids=[1]))
    assert resp.slots == [] and resp.scope == "today"
```

> **WS1 coupling note for the worker:** the fake `_fake_optimize` above takes an `extra_ids=` kwarg. If WS1's stabilized `LineupService.optimize` signature does NOT yet accept `extra_ids` (i.e. it can't be told to add hypothetical candidates to the eligible pool), implement `optimize()` to call the pipeline DIRECTLY instead of via `LineupService` (see the fallback impl below) and change this test to monkeypatch `src.optimizer.pipeline.LineupOptimizerPipeline` / `build_daily_dcv_table` instead. Check the real `LineupService.optimize` signature first and pick ONE path. Keep the assertion that scope + the selected ids reach the machinery.

- [ ] **Run (expect FAIL):** `python -m pytest tests/api/test_api_start_sit_optimize.py -v` → `AttributeError: ... 'optimize'`.
- [ ] **Minimal impl (REAL code):** Add to `StartSitService`. **Path A (preferred, if WS1 exposes `extra_ids`):**

```python
    # ----------------------------------------------------------------- optimize
    def optimize(self, req) -> "StartSitOptimizeResponse":  # noqa: F821
        from api.contracts.start_sit import StartSitOptimizeResponse

        scope = self._scope(req.scope)
        pids = [int(p) for p in (req.player_ids or [])]
        slots: list = []
        bench: list = []
        summary = "Lineup unavailable (no live data in this environment)."
        daily = None
        try:
            from api.services.lineup_service import LineupService

            mode = "daily" if scope == "today" else "standard"
            # The selected candidates join the eligible pool as hypothetical adds;
            # the LP fills the user's open slots optimally at this scope.
            res = LineupService().optimize(req.team_name, None, scope, mode, extra_ids=pids)
            slots = list(res.slots)
            bench = list(res.bench)
            daily = res.daily
            if slots:
                summary = res.summary or f"{len(slots)} starters set."
        except Exception as exc:
            logger.warning("StartSitService.optimize failed: %s", exc)
        return StartSitOptimizeResponse(scope=scope, slots=slots, bench=bench, summary=summary, daily=daily)
```

**Path B (fallback, if WS1 has NOT added `extra_ids`):** call the pipeline directly — build the matchup-aware ctx at `scope`, append the selected players' pool rows to the roster passed into `LineupOptimizerPipeline`, run `.optimize()`, and reuse `lineup_service`'s `_to_slots` / `_daily_slots` mapping by importing those static helpers. Document in a code comment which path was taken and why. (Path A is strongly preferred — it keeps slot-mapping logic in one place.)

- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_api_start_sit_optimize.py -v`
- [ ] **Commit:** `git add api/services/start_sit_service.py tests/api/test_api_start_sit_optimize.py && git commit -m "feat(start-sit): optimize() — LP fill of open slots with selected candidates (WS3)"`

---

## Task 6 — Router + DI provider + endpoint tests

**Files:**
- `api/routers/start_sit.py` (NEW).
- `api/deps.py` (add `get_start_sit_service`).
- Append endpoint tests to `tests/api/test_api_start_sit_compare.py` and `tests/api/test_api_start_sit_optimize.py`.

- [ ] **Failing test (REAL code):** Append the endpoint contract test to `tests/api/test_api_start_sit_compare.py` (fake-service via `app.dependency_overrides`, mirroring `test_api_playoff.py`):

```python
def test_compare_endpoint_contract():
    from starlette.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.start_sit import StartSitCandidate, StartSitCompareResponse, StartSitVerdict
    from api.deps import get_start_sit_service
    from api.main import create_app

    class _Fake:
        def compare(self, req):
            return StartSitCompareResponse(
                scope=req.scope,
                candidates=[
                    StartSitCandidate(
                        player=PlayerRef(id=1, name="X", positions="OF"),
                        start_score=100.0,
                        rank=1,
                        eligible_slots=["OF", "Util"],
                        matchup="vs SF",
                        reason="park",
                    )
                ],
                verdict=StartSitVerdict(start_ids=[1], sit_ids=[2], reasoning="r"),
                open_slots={"OF": 2},
                confidence=0.4,
                confidence_label="Clear",
            )

    app = create_app()
    app.dependency_overrides[get_start_sit_service] = lambda: _Fake()
    try:
        body = TestClient(app).post(
            "/api/start-sit/compare", json={"team_name": "Team Hickey", "scope": "today", "player_ids": [1, 2]}
        ).json()
        assert body["scope"] == "today"
        assert body["candidates"][0]["rank"] == 1 and body["candidates"][0]["start_score"] == 100.0
        assert body["verdict"]["start_ids"] == [1]
        assert body["open_slots"]["OF"] == 2
        assert body["confidence_label"] == "Clear"
    finally:
        app.dependency_overrides.clear()
```

And to `tests/api/test_api_start_sit_optimize.py`:

```python
def test_optimize_endpoint_contract():
    from starlette.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.lineup import LineupSlot
    from api.contracts.start_sit import StartSitOptimizeResponse
    from api.deps import get_start_sit_service
    from api.main import create_app

    class _Fake:
        def optimize(self, req):
            return StartSitOptimizeResponse(
                scope=req.scope,
                slots=[LineupSlot(slot="OF", player=PlayerRef(id=1, name="X", positions="OF"), action="START", status="start")],
                bench=[],
                summary="1 starters set.",
            )

    app = create_app()
    app.dependency_overrides[get_start_sit_service] = lambda: _Fake()
    try:
        body = TestClient(app).post(
            "/api/start-sit/optimize", json={"team_name": "Team Hickey", "scope": "rest_of_week", "player_ids": [1]}
        ).json()
        assert body["scope"] == "rest_of_week"
        assert body["slots"][0]["player"]["id"] == 1
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Run (expect FAIL):** `python -m pytest tests/api/test_api_start_sit_compare.py::test_compare_endpoint_contract tests/api/test_api_start_sit_optimize.py::test_optimize_endpoint_contract -v` → fails (`get_start_sit_service` / router don't exist; 404).
- [ ] **Minimal impl (REAL code):** Create `api/routers/start_sit.py`:

```python
"""Start/Sit router. THIN — delegates to the service.

Personalized reads: gated by require_login (matches the other personalized
routers; dormant until Clerk is configured). Team resolution via the viewer
context so the client cannot spoof another team's data."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.start_sit import (
    StartSitCompareRequest,
    StartSitCompareResponse,
    StartSitOptimizeRequest,
    StartSitOptimizeResponse,
)
from api.deps import get_start_sit_service
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team

router = APIRouter(prefix="/api", tags=["start-sit"], dependencies=[Depends(require_login)])


@router.post("/start-sit/compare", response_model=StartSitCompareResponse)
def compare_start_sit(
    req: StartSitCompareRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_start_sit_service),
) -> StartSitCompareResponse:
    req.team_name = resolve_required_team(ctx, req.team_name)
    return service.compare(req)


@router.post("/start-sit/optimize", response_model=StartSitOptimizeResponse)
def optimize_start_sit(
    req: StartSitOptimizeRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_start_sit_service),
) -> StartSitOptimizeResponse:
    req.team_name = resolve_required_team(ctx, req.team_name)
    return service.optimize(req)
```

> The router mutates `req.team_name` (resolving the viewer's team) then delegates — this is the same logic-free pattern as `api/routers/lineup.py` (which calls `resolve_required_team(ctx, req.team_name)` inline). No arithmetic, no `src.*` import → passes `test_no_logic_in_routers.py`.

Add the DI provider to `api/deps.py` — import near the other service imports and add the factory near `get_lineup_service`:

```python
# (add to the import block at the top of api/deps.py)
from api.services.start_sit_service import StartSitService
```

```python
def get_start_sit_service() -> StartSitService:
    return StartSitService()
```

- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_api_start_sit_compare.py tests/api/test_api_start_sit_optimize.py tests/api/test_no_logic_in_routers.py -v`
- [ ] **Commit:** `git add api/routers/start_sit.py api/deps.py tests/api/test_api_start_sit_compare.py tests/api/test_api_start_sit_optimize.py && git commit -m "feat(start-sit): thin router + DI provider + endpoint tests (WS3)"`

---

## Task 7 — Mount router + regenerate openapi snapshot

**Files:**
- `api/main.py` (import + `include_router`).
- `api/openapi.json` (regenerated).

- [ ] **Failing test:** Run the openapi snapshot guard — it fails because the new routes aren't in the committed schema yet. First mount, then regen. Add to `api/main.py` in the router import block (alphabetical-ish, near `schedule`) and the `include_router` block:

```python
    from api.routers.start_sit import router as start_sit_router
```
```python
    app.include_router(start_sit_router)
```

- [ ] **Run (expect FAIL before regen):** `python -m pytest tests/api/test_openapi_contract.py -v` → fails: "api/openapi.json is stale".
- [ ] **Regenerate (REAL cmd):** `python scripts/export_openapi.py` (writes `api/openapi.json` with the two new routes + the `StartSit*` schemas).
- [ ] **Run (expect PASS):** `python -m pytest tests/api/test_openapi_contract.py tests/api/test_no_logic_in_routers.py tests/api/test_api_start_sit_compare.py tests/api/test_api_start_sit_optimize.py -v`
- [ ] **Run the FULL api suite (regression):** `python -m pytest tests/api/ -q` → all green (confirms no contract/snapshot collision with sibling routes; `test_api_pro_gating.py` introspection unaffected — start-sit is NOT Pro-gated).
- [ ] **Commit:** `git add api/main.py api/openapi.json && git commit -m "feat(start-sit): mount routes + regen openapi (WS3 backend complete)"`

---

## Task 8 — Frontend: regenerate TS types + data module `start-sit-data.ts`

**Files:**
- `web/src/lib/api/generated.ts` (regenerated from `api/openapi.json`).
- `web/src/lib/api/types.ts` (add `Api*` aliases).
- `web/src/lib/start-sit-data.ts` (NEW — types + fetchers + adapter, mirroring `streaming-data.ts` / `optimizer-data.ts`).

- [ ] **Regenerate TS types (REAL cmd, from `web/`):** `pnpm gen:api` (runs `openapi-typescript ../api/openapi.json -o src/lib/api/generated.ts`).
- [ ] **Add aliases to `web/src/lib/api/types.ts`** (after the existing `ApiLineupSlot` line):

```typescript
export type ApiStartSitCompareResponse = components["schemas"]["StartSitCompareResponse"];
export type ApiStartSitCandidate = components["schemas"]["StartSitCandidate"];
export type ApiStartSitOptimizeResponse = components["schemas"]["StartSitOptimizeResponse"];
```

- [ ] **Create `web/src/lib/start-sit-data.ts`** (types + adapter + the two live-or-mock fetchers). The page is INTERACTIVE (scope + player selection feed the request), so unlike the page-load fetchers these take args and are called imperatively (not via `usePageData`). Provide a small mock for off-live so the page renders in the showcase:

```typescript
import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiPost } from "@/lib/api/client";
import { isLive } from "@/lib/api/live";
import type {
  ApiStartSitCompareResponse,
  ApiStartSitCandidate,
  ApiStartSitOptimizeResponse,
} from "@/lib/api/types";
import { apiOptSlotToData } from "@/lib/api/adapters";
import type { LineupSlot } from "@/lib/optimizer-data";

/**
 * Start/Sit data. Scope + selected player ids drive each request, so these
 * fetchers take args and are called imperatively (NOT via usePageData). Live →
 * POST /api/start-sit/{compare,optimize}; off-live → a small deterministic mock so
 * the showcase renders. The shapes ARE the contract.
 */
export type Scope = "today" | "rest_of_week" | "rest_of_season";

export const SCOPE_LABELS: Record<Scope, string> = {
  today: "Today",
  rest_of_week: "Rest of Week",
  rest_of_season: "Rest of Season",
};

export interface StatItem {
  label: string;
  value: string;
}

export interface StartSitCandidate {
  player: PlayerRef;
  startScore: number; // 0-100 heat
  rank: number;
  eligibleSlots: string[];
  projected: StatItem[];
  categoryImpact: StatItem[];
  matchup: string;
  reason: string;
  playable: boolean;
}

export interface StartSitVerdict {
  startIds: number[];
  sitIds: number[];
  reasoning: string;
}

export interface StartSitCompareData {
  scope: Scope;
  candidates: StartSitCandidate[];
  verdict: StartSitVerdict;
  openSlots: Record<string, number>;
  confidence: number;
  confidenceLabel: string;
}

export interface StartSitOptimizeData {
  scope: Scope;
  starters: LineupSlot[];
  bench: LineupSlot[];
  summary: string;
}

function toPlayerRef(p: {
  name: string;
  positions: string;
  mlb_id?: number | null;
  team_abbr?: string | null;
  team_id?: number | null;
}): PlayerRef {
  return { name: p.name, pos: p.positions, teamAbbr: p.team_abbr ?? "", teamId: p.team_id ?? 0, mlbId: p.mlb_id ?? 0 };
}

function adaptCandidate(c: ApiStartSitCandidate): StartSitCandidate {
  return {
    player: toPlayerRef(c.player),
    startScore: c.start_score ?? 0,
    rank: c.rank ?? 0,
    eligibleSlots: c.eligible_slots ?? [],
    projected: c.projected ?? [],
    categoryImpact: c.category_impact ?? [],
    matchup: c.matchup ?? "",
    reason: c.reason ?? "",
    playable: c.playable ?? true,
  };
}

export function adaptCompare(api: ApiStartSitCompareResponse): StartSitCompareData {
  return {
    scope: (api.scope ?? "today") as Scope,
    candidates: (api.candidates ?? []).map(adaptCandidate),
    verdict: {
      startIds: api.verdict?.start_ids ?? [],
      sitIds: api.verdict?.sit_ids ?? [],
      reasoning: api.verdict?.reasoning ?? "",
    },
    openSlots: api.open_slots ?? {},
    confidence: api.confidence ?? 0,
    confidenceLabel: api.confidence_label ?? "Toss-up",
  };
}

export function adaptOptimize(api: ApiStartSitOptimizeResponse): StartSitOptimizeData {
  return {
    scope: (api.scope ?? "today") as Scope,
    starters: (api.slots ?? []).map(apiOptSlotToData),
    bench: (api.bench ?? []).map(apiOptSlotToData),
    summary: api.summary ?? "",
  };
}

/** Compare 2-6 selected players at a scope → ranked verdict. Live errors propagate. */
export async function compareStartSit(scope: Scope, playerIds: number[]): Promise<StartSitCompareData> {
  if (!isLive()) return mockCompare(scope, playerIds);
  const api = await apiPost<ApiStartSitCompareResponse>("/start-sit/compare", {
    team_name: getViewerTeam(),
    scope,
    player_ids: playerIds,
  });
  return adaptCompare(api);
}

/** Apply the selected candidates to the user's open slots (authoritative LP). */
export async function optimizeStartSit(scope: Scope, playerIds: number[]): Promise<StartSitOptimizeData> {
  if (!isLive()) return mockOptimize(scope, playerIds);
  const api = await apiPost<ApiStartSitOptimizeResponse>("/start-sit/optimize", {
    team_name: getViewerTeam(),
    scope,
    player_ids: playerIds,
  });
  return adaptOptimize(api);
}

// --- off-live mocks (showcase) -------------------------------------------------
function mockCompare(scope: Scope, ids: number[]): StartSitCompareData {
  const candidates: StartSitCandidate[] = ids.slice(0, 6).map((id, i) => ({
    player: { name: `Player ${id}`, pos: "OF", teamAbbr: "NYY", teamId: 147, mlbId: 0 },
    startScore: Math.max(20, 100 - i * 18),
    rank: i + 1,
    eligibleSlots: ["OF", "Util"],
    projected: [
      { label: "HR", value: String(3 - i) },
      { label: "R", value: String(6 - i) },
    ],
    categoryImpact: [
      { label: "HR", value: `+${(1.2 - i * 0.2).toFixed(2)}` },
      { label: "SB", value: `+${(0.4 + i * 0.1).toFixed(2)}` },
    ],
    matchup: i % 2 ? "@ COL" : "vs SF",
    reason: i === 0 ? "Favorable park + platoon edge" : "Average matchup",
    playable: true,
  }));
  const startIds = candidates.slice(0, Math.min(2, candidates.length)).map((c) => c.player.mlbId || c.rank);
  return {
    scope,
    candidates,
    verdict: { startIds, sitIds: candidates.slice(2).map((c) => c.rank), reasoning: "Start the top 2 that fit your open slots." },
    openSlots: { OF: 2, Util: 1 },
    confidence: 0.34,
    confidenceLabel: "Clear",
  };
}

function mockOptimize(scope: Scope, ids: number[]): StartSitOptimizeData {
  return {
    scope,
    starters: ids.slice(0, 2).map((id) => ({
      slot: "OF",
      player: { name: `Player ${id}`, pos: "OF", teamAbbr: "NYY", teamId: 147, mlbId: 0 },
      matchup: "vs SF",
      value: 70,
      status: "start" as const,
    })),
    bench: [],
    summary: `${Math.min(2, ids.length)} starters set.`,
  };
}
```

> **Adapter reuse:** `apiOptSlotToData` is the existing `ApiLineupSlot → LineupSlot` mapper in `web/src/lib/api/adapters.ts` (the Optimizer page uses it). Open `adapters.ts`, find the function that maps an `ApiLineupSlot` to the frontend `LineupSlot` (it is referenced by `apiOptimizeToData`), confirm its exported name, and import that exact symbol. If it is not exported, export it (one-line `export` keyword) — that is the only adapters.ts edit WS3 needs, and it's disjoint from WS1/WS2/WS4's appends.

- [ ] **Run (expect PASS):** from `web/`: `pnpm exec tsc --noEmit` (types compile against the regenerated `generated.ts`).
- [ ] **Commit:** `git add web/src/lib/api/generated.ts web/src/lib/api/types.ts web/src/lib/start-sit-data.ts web/src/lib/api/adapters.ts && git commit -m "feat(start-sit): TS types + start-sit-data module + adapters (WS3 frontend data)"`

---

## Task 9 — Frontend: components (scope selector, player multi-select, comparison cards, verdict)

**Files:**
- `web/src/components/startsit/ScopeSelector.tsx` (NEW).
- `web/src/components/startsit/PlayerMultiSelect.tsx` (NEW — composes `searchPlayers` from `@/lib/player-search`).
- `web/src/components/startsit/CompareCard.tsx` (NEW — one candidate card).
- `web/src/components/startsit/VerdictPanel.tsx` (NEW — ranked start/sit, open-slot summary).

- [ ] **ScopeSelector (REAL code):** a 3-button segmented control over `Scope`. Mirror the Combustion button styling already used on pages (orange active, surface inactive). Props: `{ value: Scope; onChange: (s: Scope) => void }`. Use `SCOPE_LABELS` from `start-sit-data.ts`.

```tsx
"use client";

import { cn } from "@/lib/utils";
import { SCOPE_LABELS, type Scope } from "@/lib/start-sit-data";

const SCOPES: Scope[] = ["today", "rest_of_week", "rest_of_season"];

export function ScopeSelector({ value, onChange }: { value: Scope; onChange: (s: Scope) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface p-1" role="tablist" aria-label="Horizon">
      {SCOPES.map((s) => (
        <button
          key={s}
          role="tab"
          aria-selected={value === s}
          onClick={() => onChange(s)}
          className={cn(
            "min-h-9 rounded-lg px-3.5 py-1.5 text-[13px] font-bold transition-colors",
            value === s ? "bg-gradient-to-b from-heat-bright to-heat text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)]" : "text-ink-2 hover:bg-surface-2",
          )}
        >
          {SCOPE_LABELS[s]}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **PlayerMultiSelect (REAL code):** a search box + results list (debounced `searchPlayers(q)` from `@/lib/player-search`) + selected-chips row. Enforce max 6. Props: `{ selected: PlayerPick[]; onChange: (next: PlayerPick[]) => void }`. Each result row click adds (if <6 and not already selected); chips have a remove ✕. `PlayerPick` is exported from `@/lib/player-search`. Reuse `PlayerAvatar` for the chip/row visuals. Debounce with a 200ms `setTimeout` in a `useEffect` keyed on the query string; guard against stale results with an `alive` flag.
- [ ] **CompareCard (REAL code):** renders one `StartSitCandidate`: a `startScore` heat bar (width = `startScore%`, color via the existing heat ramp — reuse `heatColor` from `@/lib/tokens` if present, else a `bg-heat` bar), the player (wrap in `PlayerDialog`/`PlayerLink` so the card opens the canonical player card), `eligibleSlots` as small chips, the `projected` StatItem line, `categoryImpact` chips (green for `+`, ember for `-`), `matchup`, and `reason`. Greyscale + a "Not playable" badge when `playable === false`. Show a `START`/`SIT` ribbon driven by whether the card's player id is in `verdict.startIds` (pass that down).
- [ ] **VerdictPanel (REAL code):** a sidebar card showing the open-slot summary (`openSlots` as `OF×2 · Util×1 · …`), the ranked start list (green) and sit list (ember) by name, `confidenceLabel` as a pill, and `verdict.reasoning`. An "Apply to open slots" primary button (`onApply`) that triggers `/optimize`.
- [ ] **Run (expect PASS):** from `web/`: `pnpm exec tsc --noEmit && pnpm run lint`
- [ ] **Commit:** `git add web/src/components/startsit/ && git commit -m "feat(start-sit): scope selector + multi-select + compare cards + verdict panel (WS3)"`

---

## Task 10 — Frontend: the page + nav tab

**Files:**
- `web/src/app/start-sit/page.tsx` (NEW).
- `web/src/components/chrome/TopBar.tsx` (add the nav entry to the `NAV` array).

- [ ] **Page (REAL code):** a `"use client"` page that holds local state: `scope: Scope` (default `"today"`), `selected: PlayerPick[]`, `compare: StartSitCompareData | null`, `optimized: StartSitOptimizeData | null`, plus `loading`/`error` flags. Layout: header ("Start/Sit") + `ScopeSelector` + `PlayerMultiSelect`; a "Compare" button (enabled when `selected.length >= 2`) that calls `compareStartSit(scope, ids)`; on success render the `CompareCard` grid (left) + `VerdictPanel` (right). The "Apply to open slots" button calls `optimizeStartSit(scope, ids)` and renders the filled lineup below using the existing `LineupTable` (`@/components/optimizer/LineupTable`) over `optimized.starters` / `optimized.bench`. Recompare when `scope` changes IF a comparison already exists (re-fire on scope change). Handle the empty case (`candidates.length === 0` → an `EmptyState`/`PageEmpty`-style "No comparison yet / pick 2+ players"). This page does NOT use `usePageData` (it's interactive — fetch on button click); catch live errors and show an inline error banner. Player ids fed to the API are `PlayerPick.id` (the HEATER `player_id`), NOT `mlbId`.
- [ ] **Nav (REAL code):** in `web/src/components/chrome/TopBar.tsx`, add to the `NAV` array immediately after the Optimizer entry:

```typescript
  { label: "Optimizer", href: "/optimizer" },
  { label: "Start/Sit", href: "/start-sit" },
  { label: "Streaming", href: "/streaming" },
```

- [ ] **Run (expect PASS):** from `web/`: `pnpm exec tsc --noEmit && pnpm run lint`
- [ ] **Run the build gate (REAL cmd):** from `web/`: `pnpm build` → succeeds (the page is statically analyzable; the new route compiles). This is the frontend's only CI-equivalent gate.
- [ ] **Commit:** `git add web/src/app/start-sit/ web/src/components/chrome/TopBar.tsx && git commit -m "feat(start-sit): page + nav tab (WS3 frontend complete)"`

---

## Task 11 — Live smoke + final verification

**Files:** none (verification only).

- [ ] **Backend full-suite (regression):** `python -m pytest tests/api/ -q && python -m pytest tests/test_start_sit.py -q` → all green.
- [ ] **Live local stack (per reference_live_verification_local_stack):** start the API as a FACTORY: `python -m uvicorn api.main:create_app --factory --port 8000` (background). Then exercise both endpoints with REAL player ids resolved via search. Note: with Clerk dormant locally (`CLERK_ISSUER` unset) `require_login` is a no-op and `resolve_required_team` falls back to the client `team_name`, so pass `"team_name": "Team Hickey"`:
  - `curl -s -X POST localhost:8000/api/players/search -G --data-urlencode q=judge` → grab two `results[].id`. (If `players/search` is GET, use `curl -s "localhost:8000/api/players/search?q=judge"`.)
  - `curl -s -X POST localhost:8000/api/start-sit/compare -H 'content-type: application/json' -d '{"team_name":"Team Hickey","scope":"today","player_ids":[<id1>,<id2>]}'` → expect `candidates` populated (or a graceful empty list with `scope:"today"` if the local DB lacks live Yahoo — that is acceptable; the response must never 500).
  - Repeat for `scope:"rest_of_week"` and `"rest_of_season"`, and for `/start-sit/optimize`.
  - Confirm: no 500s, ranks are 1..N descending by `start_score`, `open_slots` is present, `verdict.start_ids` ⊆ the selected ids.
- [ ] **Live-only degradation note:** matchup/`current_slot`/daily fields and live FA values only fully populate with live Yahoo (Railway). Locally they degrade gracefully (empty/`""`), which is expected — the contract shape is the gate here, full population is verified against the live API after deploy. Document any locally-empty field in the final summary.
- [ ] **Frontend preview (optional, per reference_preview_tool_gotchas):** point the React preview at the local API (`NEXT_PUBLIC_HEATER_LIVE=1` + the proxy) and walk the page: pick 2–3 players → Compare → see cards + verdict → Apply to open slots → see the filled lineup.
- [ ] **Finish (per the always-merge-and-push rule):** this WS3 work lands on the shared feature branch. After the integrator's openapi/TS regeneration pass, merge to local master and push origin/master. Run silent-failure-hunter on `api/services/start_sit_service.py` (the non-trivial new service) per the post-merge rule.

---

## Notes & guardrails for the worker

- **Worktree DB is EMPTY** — every `tests/api/test_api_start_sit_*.py` test MUST be DB-free: fake the whole service via `app.dependency_overrides` (endpoint tests) or monkeypatch the engines at their SOURCE module (`src.optimizer.shared_data_layer.build_optimizer_context`, `src.start_sit.start_sit_recommendation`, `src.yahoo_data_service.get_yahoo_data_service`, `src.optimizer.daily_optimizer.build_daily_dcv_table`). The engines are imported LAZILY inside service methods, so patching the source module is correct (do NOT patch a name bound inside the service — there is none at import time).
- **NaN/inf discipline:** every float that reaches the JSON response goes through `_f()` (drops NaN/inf). The pool can yield `NaN` for missing stats; never emit `"nan"`/`"inf"` (RFC-8259-invalid).
- **Never raise from a service method** — every path is `try/except` → graceful empty (`candidates=[]`, full-open `open_slots`, empty `slots`). The cold-env tests lock this.
- **Routers stay logic-free** — `resolve_required_team(ctx, req.team_name)` + delegate only. The AST guard (`test_no_logic_in_routers.py`) forbids `src.*` imports and arithmetic assignments in routers.
- **`/compare` is a heuristic, `/optimize` is authoritative** — do not try to make compare's greedy assignment match the LP exactly; that's by design (spec "Risks & open items"). The compare verdict is bounded by `open_slots`; the LP is the source of truth for the applied lineup.
- **Pins:** `fastapi==0.137.1` + `httpx==0.28.1` (the openapi snapshot guard fails on FastAPI version drift, not just contract changes — do not bump them).
- **Shared-file etiquette:** `api/main.py`, `api/openapi.json`, `web/src/lib/api/generated.ts`, `web/src/lib/api/types.ts`, and `web/src/lib/api/adapters.ts` are integrator-reconciled. WS3 owns `web/src/components/chrome/TopBar.tsx` (per the spec's file-ownership split). If a merge conflict hits a generated file (`openapi.json` / `generated.ts`), REGENERATE it (`python scripts/export_openapi.py` / `pnpm gen:api`) — never hand-merge.
