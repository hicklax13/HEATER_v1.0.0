# M0 — Matchup roster-comparison tables (`/api/matchup` extend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/api/matchup` with the **roster-comparison table** the frontend Matchup page needs: per-slot `hitters`/`pitchers` rows (you vs opponent), column headers, date-tabs, and a per-category winner. Player stat lines are **projected** + carry a game **state** (sched/live/final); LIVE in-game stats/status are explicitly DEFERRED (the follow-up plan `2026-06-19-heater-matchup-live-stats-followup.md` — no box-score feed exists). This is the centerpiece of the gap spec §4 Matchup. you/opp TeamSide metadata + totals + the league scoreboard are a SEPARATE follow-up slice (Matchup-B).

**Architecture:** Extend `api/contracts/matchup.py` (add `MatchPlayer`, `RosterRow`; extend `MatchupResponse`; add `win` to `MatchupCategory`) and `api/services/matchup_service.py` (build the roster comparison from `yds.get_rosters()` joined with the player pool, + game-state from `statsapi.schedule()` via `game_day`). NaN-safe. All new fields default empty (backward-compatible). `src/` untouched; router logic-free.

**Verified data sources (from grounding):**
- `yds.get_rosters()` → DataFrame cols: `team_name`, `player_id`, `roster_slot`, `selected_position`, `status`, `is_user_team`. Join the player pool (`load_player_pool()`) by `player_id` for `name`/`team`/`positions`/`is_hitter`/`mlb_id` + projected stats.
- Projected stat cols (pool): hitter `h, ab, r, hr, rbi, sb, avg, obp`; pitcher `ip, w, l, sv, k, era, whip`.
- Game state: `src.game_day.get_target_game_date()` + `statsapi.schedule(date=…)` → per-game `home_name`/`away_name`/`status`; `src.game_day.FINAL_GAME_STATUSES` / `LOCKED_GAME_STATUSES` classify. Map player `team` abbr → today's game → state. (NOTE: `schedule()` returns full team NAMES, not abbrs — match via the team-name↔abbr map or substring; the implementer verifies the matching.)

**Convention:** snake_case; nested `PlayerRef`. `status`/`state` carry the engine's basic game info (not live play-by-play). Columns are static. `pos` = the player's roster slot.

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = worktree root.

**Tech Stack:** FastAPI + Pydantic v2; pandas; pytest. `fastapi==0.137.1`/`httpx==0.28.1` pinned.

---

## File Structure

- `api/contracts/matchup.py` — **modify**: add `MatchPlayer`, `RosterRow`; extend `MatchupResponse` with `date_tabs`/`hitter_columns`/`pitcher_columns`/`hitters`/`pitchers`; add `win` to `MatchupCategory`.
- `api/services/matchup_service.py` — **modify**: pure helpers (`_f`, `_fmt_hitter_stats`, `_fmt_pitcher_stats`, `_game_state`, `_to_match_player`, `_pair_rows`, `_date_tabs`, `_cat_win`) + extend the response build.
- `api/openapi.json` — **regenerate**.
- `tests/api/test_api_matchup_rosters.py` — **create**: helper unit tests + fake-service contract test.

---

### Task 1: Contracts + pure helpers (stat formatting, game-state, pairing, date-tabs, cat-win)

**Files:**
- Modify: `api/contracts/matchup.py`
- Modify: `api/services/matchup_service.py`
- Test: `tests/api/test_api_matchup_rosters.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/api/test_api_matchup_rosters.py`:

```python
"""Matchup roster tables: helper unit tests + fake-service contract test."""

from __future__ import annotations

from api.services.matchup_service import (
    _cat_win,
    _date_tabs,
    _fmt_hitter_stats,
    _fmt_pitcher_stats,
    _game_state,
    _pair_rows,
    _to_match_player,
)


def test_fmt_hitter_stats():
    row = {"h": 120, "ab": 410, "r": 70, "hr": 24, "rbi": 80, "sb": 12, "avg": 0.293, "obp": 0.371}
    assert _fmt_hitter_stats(row) == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_fmt_pitcher_stats():
    row = {"ip": 180.0, "w": 14, "l": 7, "sv": 0, "k": 200, "era": 3.21, "whip": 1.05}
    assert _fmt_pitcher_stats(row) == ["180.0", "14", "7", "0", "200", "3.21", "1.05"]


def test_fmt_stats_nan_safe():
    assert _fmt_hitter_stats({"h": float("nan"), "ab": None})[0] == "0/0"


def test_game_state_maps_status():
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "Final"}]
    state, status = _game_state("NYY", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})
    assert state == "final"
    sched2 = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "In Progress"}]
    assert _game_state("BOS", sched2, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "live"
    # no game today → none
    assert _game_state("LAD", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "none"


def test_cat_win_respects_inverse():
    assert _cat_win(5.0, 3.0, inverse=False) == "you"  # higher wins
    assert _cat_win(5.0, 3.0, inverse=True) == "opp"   # lower wins (ERA/WHIP/L)
    assert _cat_win(3.0, 3.0, inverse=False) == ""      # tie


def test_date_tabs_starts_live_totals():
    tabs = _date_tabs(7)
    assert tabs[0] == "Live"
    assert tabs[1] == "Totals"
    assert len(tabs) >= 3


def test_to_match_player_builds_ref_and_stats():
    import pandas as pd

    pool = pd.DataFrame([{
        "player_id": 1, "name": "Aaron Judge", "positions": "OF", "mlb_id": 592450,
        "team": "NYY", "is_hitter": True, "h": 120, "ab": 410, "r": 70, "hr": 24,
        "rbi": 80, "sb": 12, "avg": 0.293, "obp": 0.371,
    }])
    mp = _to_match_player(1, "OF", pool, hitter=True, state="final", status="Final")
    assert mp.player.mlb_id == 592450
    assert mp.player.team_id == 147
    assert mp.pos == "OF"
    assert mp.state == "final"
    assert mp.stats == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_pair_rows_zips_sides_and_pads():
    import pandas as pd

    pool = pd.DataFrame([{"player_id": i, "name": f"P{i}", "positions": "OF", "mlb_id": i,
                          "team": "NYY", "is_hitter": True} for i in (1, 2, 3)])
    you = [_to_match_player(1, "OF", pool, True, "none", ""), _to_match_player(2, "OF", pool, True, "none", "")]
    opp = [_to_match_player(3, "OF", pool, True, "none", "")]
    rows = _pair_rows(you, opp, ["OF", "OF"])
    assert len(rows) == 2
    assert rows[0].you is not None and rows[0].opp is not None
    assert rows[1].you is not None and rows[1].opp is None  # opp padded
```

NOTE: `_fmt_hitter_stats` AVG/OBP use the leading-zero strip (".293"); confirm the exact format and align the assertion. `_game_state`'s team-name matching: the plan passes an abbr→full-name map; the implementer wires the real map (see `src/depth_charts._STATSAPI_TEAM_IDS` keys + a name map, or substring match on `team` in `home_name`/`away_name`).

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_matchup_rosters.py -q`
Expected: FAIL — ImportError on the new helpers.

- [ ] **Step 3: Add the contracts**

In `api/contracts/matchup.py`: add `win: str = ""` to `MatchupCategory`; add the new models + extend `MatchupResponse` (keep existing fields):

```python
class MatchPlayer(BaseModel):
    player: PlayerRef
    pos: str = ""
    status: str = ""  # basic game status (NOT live play-by-play — see live-stats follow-up)
    state: str = "none"  # sched/live/final/none
    stats: list[str] = Field(default_factory=list)  # 7-stat projected line
    badge: str | None = None  # IL / DTD


class RosterRow(BaseModel):
    slot: str = ""
    you: MatchPlayer | None = None
    opp: MatchPlayer | None = None
```

Extend `MatchupResponse` with:
```python
    date_tabs: list[str] = Field(default_factory=list)
    hitter_columns: list[str] = Field(default_factory=list)
    pitcher_columns: list[str] = Field(default_factory=list)
    hitters: list[RosterRow] = Field(default_factory=list)
    pitchers: list[RosterRow] = Field(default_factory=list)
```
(Add `from pydantic import Field` + ensure `PlayerRef` is imported.)

- [ ] **Step 4: Add the service helpers**

In `api/services/matchup_service.py`, add `import math` + the helpers (the data-source wiring comes in Task 2):

```python
from api.contracts.matchup import MatchPlayer, RosterRow
from api.services.player_ref import player_ref_from_pool

_HITTER_COLUMNS = ["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"]
_PITCHER_COLUMNS = ["IP", "W", "L", "SV", "K", "ERA", "WHIP"]


def _f(value, default: float = 0.0) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _avg(value) -> str:
    fval = _f(value)
    return f"{fval:.3f}"[1:] if 0.0 <= fval < 1.0 else f"{fval:.3f}"


def _fmt_hitter_stats(row) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    return [
        f"{int(_f(g('h')))}/{int(_f(g('ab')))}",
        str(int(_f(g("r")))), str(int(_f(g("hr")))), str(int(_f(g("rbi")))), str(int(_f(g("sb")))),
        _avg(g("avg")), _avg(g("obp")),
    ]


def _fmt_pitcher_stats(row) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    return [
        f"{_f(g('ip')):.1f}",
        str(int(_f(g("w")))), str(int(_f(g("l")))), str(int(_f(g("sv")))), str(int(_f(g("k")))),
        f"{_f(g('era')):.2f}", f"{_f(g('whip')):.2f}",
    ]


def _cat_win(you, opp, inverse: bool) -> str:
    y, o = _f(you), _f(opp)
    if y == o:
        return ""
    higher_you = y > o
    return ("you" if higher_you else "opp") if not inverse else ("opp" if higher_you else "you")


def _date_tabs(week: int) -> list[str]:
    # Live + Totals + the 7 weekday labels (Mon..Sun). Day dates are presentation;
    # the frontend may relabel — keep simple + deterministic.
    return ["Live", "Totals", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _game_state(team_abbr: str, schedule: list, abbr_to_name: dict) -> tuple[str, str]:
    """Map a team abbr → today's game state (sched/live/final/none) + basic status."""
    from src.game_day import FINAL_GAME_STATUSES, LOCKED_GAME_STATUSES

    name = abbr_to_name.get(str(team_abbr).upper(), "")
    for g in schedule or []:
        home, away = str(g.get("home_name", "")), str(g.get("away_name", ""))
        if name and (name == home or name == away):
            status = str(g.get("status", "")).strip()
            low = status.lower()
            if low in FINAL_GAME_STATUSES:
                state = "final"
            elif low in LOCKED_GAME_STATUSES:
                state = "live"
            else:
                state = "sched"
            opp = away if name == home else home
            vs = "vs" if name == home else "@"
            return state, f"{vs} {opp} · {status}" if status else f"{vs} {opp}"
    return "none", ""


def _to_match_player(player_id, slot: str, pool, hitter: bool, state: str, status: str) -> MatchPlayer:
    import pandas as pd

    prow = None
    try:
        if isinstance(pool, pd.DataFrame) and not pool.empty:
            m = pool[pool["player_id"] == player_id]
            if not m.empty:
                prow = m.iloc[0]
    except Exception:
        prow = None
    name = str(prow.get("name", "")) if prow is not None else ""
    stats = (
        (_fmt_hitter_stats(prow) if hitter else _fmt_pitcher_stats(prow)) if prow is not None else []
    )
    return MatchPlayer(
        player=player_ref_from_pool(player_id, pool, name=name, positions=slot),
        pos=slot,
        status=status,
        state=state,
        stats=stats,
        badge=None,
    )


def _pair_rows(you: list, opp: list, slots: list) -> list[RosterRow]:
    rows = []
    n = max(len(you), len(opp), len(slots))
    for i in range(n):
        rows.append(RosterRow(
            slot=slots[i] if i < len(slots) else "",
            you=you[i] if i < len(you) else None,
            opp=opp[i] if i < len(opp) else None,
        ))
    return rows
```

- [ ] **Step 5: Run the unit tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_matchup_rosters.py -q`
Expected: PASS (helper tests). (Fix the AVG-format assertion vs `_avg` output if needed.)

- [ ] **Step 6: Commit**

```bash
git add api/contracts/matchup.py api/services/matchup_service.py tests/api/test_api_matchup_rosters.py
git commit -m "feat(api): matchup roster-table contracts + pure helpers (stats/game-state/pairing)"
```

---

### Task 2: Wire the roster comparison into the matchup response + openapi

**Files:**
- Modify: `api/services/matchup_service.py` (the response-building method)
- Test: append to `tests/api/test_api_matchup_rosters.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing contract test**

Append a fake-service test asserting the route returns the new fields:

```python
def test_matchup_endpoint_includes_roster_tables():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.matchup import MatchPlayer, MatchupResponse, RosterRow
    from api.deps import get_matchup_service
    from api.main import create_app

    class _Fake:
        def get_matchup(self, team_name):
            mp = MatchPlayer(player=PlayerRef(id=1, mlb_id=592450, name="Judge", positions="OF",
                                              team_abbr="NYY", team_id=147),
                             pos="OF", state="final", stats=["1/4", "1", "0", "0", "0", ".250", ".300"])
            return MatchupResponse(
                team_name=team_name, opponent="Rivals", week=7,
                hitter_columns=["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"],
                pitcher_columns=["IP", "W", "L", "SV", "K", "ERA", "WHIP"],
                date_tabs=["Live", "Totals"],
                hitters=[RosterRow(slot="OF", you=mp, opp=None)],
            )

    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/matchup?team_name=Team+Hickey").json()
        assert body["hitter_columns"][0] == "H/AB"
        assert body["hitters"][0]["you"]["player"]["mlb_id"] == 592450
        assert body["hitters"][0]["you"]["stats"][0] == "1/4"
    finally:
        app.dependency_overrides.clear()
```

NOTE: confirm the matchup service method name (`get_matchup`?) + the DI provider (`get_matchup_service`) + that the router passes `team_name`. Mirror the real signature.

- [ ] **Step 2: Run to verify failure** (the route returns no roster fields yet from the REAL service; the fake test fails until the contract fields exist — they do after Task 1, so this mainly drives the wiring + confirms the shape).

- [ ] **Step 3: Wire the roster build into the service**

In the matchup service's response method, AFTER the existing `categories` build, add the roster comparison (read the file to place it correctly). Pseudocode the implementer makes concrete:
1. `you_team = team_name`, `opp_team = <the opponent name already resolved>`.
2. `rosters = yds.get_rosters()`; `pool = load_player_pool()`; merge rosters↔pool by `player_id` (defensive: empty → skip, leave roster fields empty).
3. `date = get_target_game_date()`; `schedule = statsapi.schedule(date=date)` (wrap in try/except → []). Build `abbr_to_name` (the team-name↔abbr map — verify the canonical source; e.g. a dict mirroring `_STATSAPI_TEAM_IDS` keys to full names, or substring matching).
4. For each side (you, opp): split into hitters (`is_hitter`) / pitchers, sort by a stable slot order, build `MatchPlayer` via `_to_match_player(player_id, slot, pool, hitter, *_game_state(team_abbr, schedule, abbr_to_name))`.
5. `hitters = _pair_rows(you_hitters, opp_hitters, you_hitter_slots)`; `pitchers = _pair_rows(you_pitchers, opp_pitchers, you_pitcher_slots)`.
6. Set `win` on each existing `MatchupCategory` via `_cat_win(cat.you, cat.opp, cat.inverse)`.
7. Set `hitter_columns=_HITTER_COLUMNS`, `pitcher_columns=_PITCHER_COLUMNS`, `date_tabs=_date_tabs(week)`.
8. ALL wrapped so a failure leaves the roster fields empty (the existing aggregate response still returns) — never raise.

- [ ] **Step 4: Run the contract test** — PASS.

- [ ] **Step 5: Regenerate OpenAPI** + confirm the snapshot test passes; confirm `MatchPlayer`/`RosterRow` + the new `MatchupResponse` fields appear.

- [ ] **Step 6: Commit**

```bash
git add api/services/matchup_service.py api/openapi.json tests/api/test_api_matchup_rosters.py
git commit -m "feat(api): build matchup roster-comparison tables (projected stats + game-state)"
```

---

### Task 3: Full api suite + lint

- [ ] Run `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q` (prior 134 + new; fix any exact-equality matchup test for the new fields). `test_no_logic_in_routers` stays green.
- [ ] `ruff format` + `ruff check` on `api/` + `tests/api/` — clean.
- [ ] Commit if anything changed.

---

## Self-Review (plan author)

**Spec coverage** (gap spec §4 roster centerpiece): `hitters`/`pitchers` RosterRows (you vs opp, per slot) ✔; `MatchPlayer` (PlayerRef + pos + projected stats + state + badge) ✔; `hitter_columns`/`pitcher_columns` ✔; `date_tabs` ✔; `cats[].win` ✔. **Deferred (documented):** live in-game stats/status (follow-up plan), IL/DTD badge (defaults None — needs player_tags), TeamSide you/opp + totals + league scoreboard (Matchup-B follow-up).

**Reuse/safety:** reuses `player_ref_from_pool` + `_f` NaN-safe; all new fields default empty (backward-compatible); service never raises (roster build wrapped → empty on failure). `src/` untouched; router logic-free.

**Placeholder note:** Task 2 Step 3 is the one pseudocode step (the multi-source wiring) — deliberately, because the exact roster↔pool merge + team-name matching must be verified against the live shapes; the implementer makes it concrete using the grounded calls (`yds.get_rosters()`, `load_player_pool()`, `statsapi.schedule()`, `get_target_game_date()`). All PURE helpers (the testable logic) are fully specified in Task 1.

**Type consistency:** `_fmt_hitter_stats`/`_fmt_pitcher_stats`/`_game_state`/`_to_match_player`/`_pair_rows`/`_date_tabs`/`_cat_win` used consistently; `MatchPlayer`/`RosterRow` fields match service + tests; NYY=147 matches `_STATSAPI_TEAM_IDS`.
