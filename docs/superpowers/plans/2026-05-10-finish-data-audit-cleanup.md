# Finish Data Audit Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out every outstanding item from the 2026-04-18 data audit (SF-1 → SF-14) plus the SP gate "unresolved contradiction" so the data pipeline is fully resilient and honestly reports its state.

**Architecture:** TDD per item. Each fix gets a failing test that would have caught the silent-fail, then the minimal change to make it pass, then verification against the live DB at `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db`. Where a fix is genuinely unsolvable (e.g. FanGraphs 403 with no alt source), the task downgrades to "document as known limitation in the Data Status panel and tests assert the limitation is surfaced honestly."

**Tech Stack:** pytest, sqlite3, pandas, MLB-StatsAPI, pybaseball, requests

---

## Survey of Current State (verified 2026-05-10 against live DB)

| Item | Status | Evidence |
|------|--------|----------|
| SF-1 game_logs | ✅ FIXED | 39615 rows in `game_logs` |
| SF-2 season_stats threshold | ⚠️ PARTIAL | Threshold logic exists but `1073/7478 = 14%` saved is logged as `'success'` despite `expected_min=5234` — bug in chain |
| SF-3 Two-Way Players | ✅ FIXED | Logic patched in `live_stats.py:311` |
| SF-4 team_strength race | ✅ FIXED | refresh_log fresh, `30/28` rows |
| SF-5 depth_charts | ❌ BROKEN | `depth_chart_role` 100% NULL across 9576 players, `lineup_slot` 100% NULL, `depth_charts` table doesn't exist |
| SF-6 stuff_plus + batting_stats | ❌ BROKEN | `stuff_plus` 100% NULL in `season_stats`, FG 403 |
| SF-7 catcher_framing + umpire_tendencies | ❌ BROKEN | Both tables 0 rows |
| SF-8 team canonicalization | ✅ FIXED | `_TEAM_EQUIVALENCES` updated |
| SF-9 opp_pitcher_stats recency | ✅ FIXED | Filter at `shared_data_layer.py:619` |
| SF-10 players row count | ✅ MOSTLY FIXED | 9576 rows (was 917). But only 12% have `mlb_id` |
| SF-11 weather UTC date | ❌ BROKEN | `daily_optimizer.py:547` still uses `_dt.now(_utc).strftime(...)` instead of `get_target_game_date()` |
| SF-12 matchup_mult sentinel | ✅ FIXED | None sentinel at line 653 |
| SF-13 pvb_splits status | ❌ BROKEN | Line 1956 unconditionally writes `'success'` regardless of how many were freshly fetched vs cache-hits |
| SF-14 persistent logging | ✅ FIXED | data/logs/bootstrap.log present |
| **Unresolved: SP gate** | ❓ UNVERIFIED | Need reproducer. Hypothesis: works for rostered players (Yahoo merge) but not for FA pool (DB-only positions) |

---

## Task 1: SP Gate Trace + Confirm/Fix

**Files:**
- Create: `tests/test_sp_gate_trace.py`
- Modify (only if test reveals bug): `src/optimizer/daily_optimizer.py:614`

**Goal:** Resolve the audit's unresolved contradiction by writing a reproducer that builds a roster with positions=`'P'` only AND a roster with positions=`'SP'`, runs `build_daily_dcv_table`, and asserts the gate fires correctly in BOTH cases.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp_gate_trace.py
"""SF gate trace: confirm the pure-SP probable-starter gate at
daily_optimizer.py:617 fires whether positions is 'SP' (Yahoo-merged) or
'P' (raw MLB players.positions)."""
import pandas as pd
import pytest
from datetime import date

from src.optimizer.daily_optimizer import build_daily_dcv_table
from src.valuation import LeagueConfig


@pytest.fixture
def cfg():
    return LeagueConfig()


def _make_roster(positions: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_id": 1,
        "name": "Test Pitcher",
        "positions": positions,
        "team": "NYY",
        "is_hitter": 0,
        "bats": "R",
        "throws": "R",
        "status": "active",
        "fp_proj": 5.0,
        "ip": 5.0,
        "k": 5.0,
        "w": 0.5,
        "sv": 0.0,
        "era": 4.0,
        "whip": 1.3,
        "hr": 0.0,
        "rbi": 0.0,
        "r": 0.0,
        "sb": 0.0,
        "avg": 0.0,
        "obp": 0.0,
        "l": 0.5,
    }])


def test_pure_sp_pos_marker_gates_when_not_probable(cfg):
    """Pitcher with positions='SP' on a team that plays today but who is NOT
    today's probable should get volume=0.0 (zeroed out). This is the canonical
    case the gate was designed for."""
    roster = _make_roster("SP")
    schedule_today = [{
        "home_name": "NEW YORK YANKEES", "away_name": "BOSTON RED SOX",
        "home_short": "NYY", "away_short": "BOS",
        "home_probable_pitcher": "Other Pitcher",
        "away_probable_pitcher": "Some Other",
    }]
    df = build_daily_dcv_table(
        roster, config=cfg, target_date=str(date.today()),
        schedule_today=schedule_today, teams_playing={"NYY", "BOS"},
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] == 0.0, "Pure SP not in probables must zero out"
    assert row["reason"] == "OFF_DAY", f"Expected OFF_DAY, got {row['reason']}"


def test_pure_p_pos_marker_also_gates_when_not_probable(cfg):
    """Pitcher with positions='P' (raw MLB marker, no SP/RP distinction).
    Currently the gate at line 615 looks for 'SP' substring — if positions
    is just 'P', the gate is dead code and the player will NOT be zeroed.
    This test PROVES whether the audit's silent-fail is real."""
    roster = _make_roster("P")
    schedule_today = [{
        "home_name": "NEW YORK YANKEES", "away_name": "BOSTON RED SOX",
        "home_short": "NYY", "away_short": "BOS",
        "home_probable_pitcher": "Other Pitcher",
        "away_probable_pitcher": "Some Other",
    }]
    df = build_daily_dcv_table(
        roster, config=cfg, target_date=str(date.today()),
        schedule_today=schedule_today, teams_playing={"NYY", "BOS"},
    )
    row = df[df["player_id"] == 1].iloc[0]
    # Expected behavior: 'P' should ALSO be treated as a starter marker
    # (since MLB's raw position type is 'Pitcher' for both starters and
    # relievers but we don't know which without more signal). The safest
    # rule: if positions == 'P' alone (no 'SP' or 'RP'), and team has a
    # probable, and player isn't it, treat as off-day.
    assert row["volume_factor"] == 0.0, (
        "positions='P' (no SP/RP qualifier) on team with named probable: "
        "player is provably not pitching, must zero out"
    )


def test_sp_rp_hybrid_not_zeroed_when_not_probable(cfg):
    """SP/RP hybrid not in today's probables can still relieve — should
    NOT be zeroed. This is the existing carve-out at line 627."""
    roster = _make_roster("SP,RP")
    schedule_today = [{
        "home_name": "NEW YORK YANKEES", "away_name": "BOSTON RED SOX",
        "home_short": "NYY", "away_short": "BOS",
        "home_probable_pitcher": "Other Pitcher",
        "away_probable_pitcher": "Some Other",
    }]
    df = build_daily_dcv_table(
        roster, config=cfg, target_date=str(date.today()),
        schedule_today=schedule_today, teams_playing={"NYY", "BOS"},
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] > 0.0, "SP/RP hybrid retains 0.9 baseline"


def test_sp_probable_today_keeps_volume(cfg):
    """The named probable starter with positions='SP' must not be zeroed."""
    roster = _make_roster("SP")
    roster.loc[0, "name"] = "Gerrit Cole"
    schedule_today = [{
        "home_name": "NEW YORK YANKEES", "away_name": "BOSTON RED SOX",
        "home_short": "NYY", "away_short": "BOS",
        "home_probable_pitcher": "Gerrit Cole",
        "away_probable_pitcher": "Some Other",
    }]
    df = build_daily_dcv_table(
        roster, config=cfg, target_date=str(date.today()),
        schedule_today=schedule_today, teams_playing={"NYY", "BOS"},
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] > 0.0, "Named probable must keep volume"
```

- [ ] **Step 2: Run test to verify which assertions actually fail**

Run: `python -m pytest tests/test_sp_gate_trace.py -v`
Expected: 3 of 4 pass; `test_pure_p_pos_marker_also_gates_when_not_probable` fails (pure-`P` is dead code per audit).

If all 4 pass → audit was wrong, no fix needed; close task by deleting the affected test or marking it as a regression check.
If `test_pure_p_pos_marker_also_gates_when_not_probable` fails → proceed to Step 3.

- [ ] **Step 3: Patch the gate to also catch pure-`P` positions**

In `src/optimizer/daily_optimizer.py` around line 612-628, the existing block:

```python
if not is_hitter and team_plays:
    pos_upper = positions.upper()
    _has_sp = "SP" in pos_upper
    _has_rp = "RP" in pos_upper
    if _has_sp and probable_starters:
        _norm_name = _normalize_pitcher_name(name)
        _norm_probable = {_normalize_pitcher_name(p) for p in probable_starters}
        _is_probable_today = _norm_name in _norm_probable
        if _is_probable_today:
            in_lineup = True
        elif not _has_rp:
            in_lineup = False
            _pitcher_volume_override = 0.0
```

Becomes:

```python
if not is_hitter and team_plays:
    pos_upper = positions.upper()
    _pos_tokens = {p.strip() for p in pos_upper.split(",")}
    _has_sp = "SP" in _pos_tokens
    _has_rp = "RP" in _pos_tokens
    # 'P' as a SOLO token (no SP/RP qualifier) means the source data
    # didn't tell us starter vs reliever. Treat as starter for gating
    # purposes — if a probable list exists for this team and this player
    # isn't on it, they're not pitching today.
    _has_p_only = _pos_tokens == {"P"}
    if (_has_sp or _has_p_only) and probable_starters:
        _norm_name = _normalize_pitcher_name(name)
        _norm_probable = {_normalize_pitcher_name(p) for p in probable_starters}
        _is_probable_today = _norm_name in _norm_probable
        if _is_probable_today:
            in_lineup = True
        elif not _has_rp:
            in_lineup = False
            _pitcher_volume_override = 0.0
```

- [ ] **Step 4: Run all 4 tests pass**

Run: `python -m pytest tests/test_sp_gate_trace.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Run full optimizer test suite to confirm no regression**

Run: `python -m pytest tests/test_daily_optimizer.py tests/test_optimizer_integration.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add tests/test_sp_gate_trace.py src/optimizer/daily_optimizer.py
git commit -m "fix(optimizer): SP gate handles pure-'P' positions marker (audit-unresolved trace)"
```

---

## Task 2: SF-11 Weather UTC Date Bug

**Files:**
- Modify: `src/optimizer/daily_optimizer.py:547`
- Test: `tests/test_sf11_weather_date.py` (new)

**Goal:** Replace `_dt.now(_utc).strftime("%Y-%m-%d")` with `get_target_game_date()` so the weather lookup uses the correct game date during the UTC 00:00–05:00 window when ET is still "today".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sf11_weather_date.py
"""SF-11: Weather lookup must use ET-anchored game date, not UTC."""
from unittest.mock import patch
import pandas as pd
from src.optimizer.daily_optimizer import build_daily_dcv_table
from src.valuation import LeagueConfig


def test_weather_lookup_uses_get_target_game_date():
    """The weather load_game_day_weather call must receive the date from
    get_target_game_date (ET-anchored, with auto-advance), not raw UTC."""
    cfg = LeagueConfig()
    roster = pd.DataFrame([{
        "player_id": 1, "name": "X", "positions": "OF", "team": "NYY",
        "is_hitter": 1, "fp_proj": 5.0,
        **{c: 0 for c in ["hr","rbi","r","sb","avg","obp","ip","k","w","l","sv","era","whip"]}
    }])
    with patch("src.optimizer.daily_optimizer.get_target_game_date",
               return_value="2099-01-01") as gtgd, \
         patch("src.game_day.load_game_day_weather", return_value=pd.DataFrame()) as lgdw:
        build_daily_dcv_table(roster, config=cfg, target_date="2099-01-01")
    assert gtgd.called, "get_target_game_date must be invoked"
    # The weather call must receive the same date
    if lgdw.call_args:
        call_date = lgdw.call_args[0][0] if lgdw.call_args[0] else lgdw.call_args.kwargs.get("date")
        assert call_date == "2099-01-01", f"Weather got {call_date}, expected 2099-01-01"
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_sf11_weather_date.py -v`
Expected: FAIL — `get_target_game_date` not called (current code uses raw UTC).

- [ ] **Step 3: Apply the fix**

In `src/optimizer/daily_optimizer.py` near line 547, replace:

```python
        today_date = _dt.now(_utc).strftime("%Y-%m-%d")
        weather_df = load_game_day_weather(today_date)
```

With:

```python
        from src.game_day import get_target_game_date
        today_date = get_target_game_date()
        weather_df = load_game_day_weather(today_date)
```

- [ ] **Step 4: Run test passes**

Run: `python -m pytest tests/test_sf11_weather_date.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_sf11_weather_date.py src/optimizer/daily_optimizer.py
git commit -m "fix(optimizer): SF-11 weather uses ET-anchored target_date, not UTC"
```

---

## Task 3: SF-13 pvb_splits Honest Status

**Files:**
- Modify: `src/data_bootstrap.py:1805-1960` (`_bootstrap_pvb_splits`)
- Test: `tests/test_sf13_pvb_status.py` (new)

**Goal:** When the function runs and 0 new matchups were freshly fetched (everything was cached), write `"cached"` not `"success"` to refresh_log so the Data Status UI distinguishes "we did real work" from "we hit cache".

- [ ] **Step 1: Read the current function**

Already read at survey time. Lines 1860-1956 show:
- `updated` counter tracks freshly-saved matchups
- `skipped` counter tracks cache hits
- Line 1956 unconditionally writes `update_refresh_log("pvb_splits", "success")` regardless of counts

- [ ] **Step 2: Write the failing test**

```python
# tests/test_sf13_pvb_status.py
"""SF-13: pvb_splits must distinguish 'cached' from 'success'."""
import sqlite3
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_bootstrap import _bootstrap_pvb_splits, BootstrapProgress


def test_pvb_status_cached_when_zero_new_fetched(tmp_path, monkeypatch):
    """When all rostered hitters x pitchers are already in pvb_splits cache,
    the refresh_log status must be 'cached', not 'success'."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, mlb_id INTEGER, name TEXT, is_hitter INTEGER);
        CREATE TABLE league_rosters (player_id INTEGER, team_key TEXT);
        CREATE TABLE opp_pitcher_stats (pitcher_id INTEGER, name TEXT, season INTEGER);
        CREATE TABLE pvb_splits (batter_id INTEGER, pitcher_id INTEGER, fetched_at TEXT,
                                 PRIMARY KEY(batter_id, pitcher_id));
        CREATE TABLE refresh_log (source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT,
                                  rows_written INTEGER, rows_expected_min INTEGER,
                                  message TEXT, tier TEXT);
        INSERT INTO players VALUES (1, 100, 'A', 1);
        INSERT INTO league_rosters VALUES (1, 'k1');
        INSERT INTO opp_pitcher_stats VALUES (200, 'P1', 2026);
        INSERT INTO pvb_splits VALUES (1, 200, '2026-05-01');
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.database.DB_PATH", str(db_path))
    progress = BootstrapProgress()
    result = _bootstrap_pvb_splits(progress)

    conn = sqlite3.connect(str(db_path))
    status = conn.execute("SELECT status FROM refresh_log WHERE source='pvb_splits'").fetchone()
    conn.close()
    assert status[0] == "cached", f"Expected 'cached' (0 new fetched), got {status[0]}"


def test_pvb_status_success_when_new_rows_fetched(tmp_path, monkeypatch):
    """When at least 1 new matchup was fetched, status must be 'success'."""
    # Setup similar to above but with empty cache so fetcher will be invoked.
    # Mock pybaseball.statcast_batter to return realistic data.
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, mlb_id INTEGER, name TEXT, is_hitter INTEGER);
        CREATE TABLE league_rosters (player_id INTEGER, team_key TEXT);
        CREATE TABLE opp_pitcher_stats (pitcher_id INTEGER, name TEXT, season INTEGER);
        CREATE TABLE pvb_splits (batter_id INTEGER, pitcher_id INTEGER,
                                 PA INTEGER, H INTEGER, AB INTEGER, fetched_at TEXT,
                                 PRIMARY KEY(batter_id, pitcher_id));
        CREATE TABLE refresh_log (source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT,
                                  rows_written INTEGER, rows_expected_min INTEGER,
                                  message TEXT, tier TEXT);
        INSERT INTO players VALUES (1, 100, 'A', 1);
        INSERT INTO league_rosters VALUES (1, 'k1');
        INSERT INTO opp_pitcher_stats VALUES (200, 'P1', 2026);
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.database.DB_PATH", str(db_path))
    fake_pvb = pd.DataFrame({
        "pitcher": [200] * 10,
        "events": ["single"] * 10,
        "type": ["X"] * 10,
    })
    with patch("pybaseball.statcast_batter", return_value=fake_pvb):
        progress = BootstrapProgress()
        _bootstrap_pvb_splits(progress)

    conn = sqlite3.connect(str(db_path))
    status = conn.execute("SELECT status FROM refresh_log WHERE source='pvb_splits'").fetchone()
    conn.close()
    assert status[0] == "success", f"Expected 'success' (new fetched), got {status[0]}"
```

- [ ] **Step 3: Run to confirm failure**

Run: `python -m pytest tests/test_sf13_pvb_status.py -v`
Expected: FAIL — current code unconditionally writes `'success'`.

- [ ] **Step 4: Apply fix**

In `src/data_bootstrap.py` around line 1956, replace:

```python
        update_refresh_log("pvb_splits", "success")
        return f"PvB splits: {updated} new, {skipped} cached"
```

With:

```python
        if updated == 0 and skipped > 0:
            update_refresh_log("pvb_splits", "cached", message=f"all {skipped} matchups already cached")
            return f"PvB splits: {skipped} already cached, 0 new fetched"
        elif updated > 0:
            update_refresh_log("pvb_splits", "success", message=f"{updated} new, {skipped} cached")
            return f"PvB splits: {updated} new, {skipped} cached"
        else:
            update_refresh_log("pvb_splits", "no_data", message="0 hitters or 0 matchups available")
            return "PvB splits: no data fetched"
```

Also need to make sure `updated` and `skipped` are defined when no work happens — verify by reading the function. Currently they're initialized in the loop scope (line 1861-1862 inside the conn block). Confirm they're in scope at the writeback.

- [ ] **Step 5: Run tests pass**

Run: `python -m pytest tests/test_sf13_pvb_status.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_sf13_pvb_status.py src/data_bootstrap.py
git commit -m "fix(bootstrap): SF-13 pvb_splits distinguishes cached/success/no_data"
```

---

## Task 4: SF-2 follow-up — season_stats threshold not actually firing

**Files:**
- Investigate: `src/live_stats.py:507-516` (`save_season_stats_to_db` writeback)
- Test: `tests/test_sf2_partial_status.py` (new)

**Goal:** The merge claimed SF-2 was DONE, but the live DB shows `season_stats: 'success'` with `'saved 1073/7478 rows'` — only 14% saved against a threshold of `max(500, 7478*0.70) = 5234`. Either the call site uses the wrong updater, or the threshold isn't being computed correctly. Find and fix.

- [ ] **Step 1: Read the call site**

Read `src/live_stats.py` lines 500-525 to see exactly how `update_refresh_log_auto` is called.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_sf2_partial_status.py
"""SF-2 follow-up: season_stats must report 'partial' when <70% saved."""
import sqlite3
import pandas as pd
from unittest.mock import patch
from src.live_stats import save_season_stats_to_db


def test_season_stats_partial_when_lots_dropped(tmp_path, monkeypatch):
    """If 7478 rows came in but only 1073 saved, status must be 'partial'."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT,
                              team TEXT, mlb_id INTEGER, is_hitter INTEGER);
        CREATE TABLE season_stats (player_id INTEGER, season INTEGER,
                                   PA INTEGER, H INTEGER,
                                   PRIMARY KEY(player_id, season));
        CREATE TABLE refresh_log (source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT,
                                  rows_written INTEGER, rows_expected_min INTEGER,
                                  message TEXT, tier TEXT);
        INSERT INTO players VALUES (1, 'A', 'NYY', 100, 1);
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.database.DB_PATH", str(db_path))

    # Fabricate 7478 rows where only 1 will match (rest have no mlb_id match)
    df = pd.DataFrame([
        {"name": "A", "team": "NYY", "mlb_id": 100, "is_hitter": 1, "PA": 50, "H": 10, "season": 2026}
    ] + [
        {"name": f"Unknown{i}", "team": "XX", "mlb_id": 99000+i, "is_hitter": 1,
         "PA": 50, "H": 10, "season": 2026}
        for i in range(7477)
    ])

    save_season_stats_to_db(df, season=2026)

    conn = sqlite3.connect(str(db_path))
    status = conn.execute("SELECT status, message FROM refresh_log WHERE source='season_stats'").fetchone()
    conn.close()
    assert status[0] == "partial", f"Expected 'partial' (only 1/7478 saved), got {status[0]}"
```

- [ ] **Step 3: Run test, observe behavior**

Run: `python -m pytest tests/test_sf2_partial_status.py -v`

If FAIL: the writeback chain is buggy. Fix: ensure `save_season_stats_to_db` returns `(saved, total_input)` and the writeback uses `update_refresh_log_auto` with `expected_min = max(500, int(total_input * 0.70))`.

If PASS: the test environment doesn't reproduce the bug (probably because the test path uses a fresh DB). Inspect why production logged `'success'` — check whether the bootstrap caller wraps `save_season_stats_to_db` with its own log writer that overrides.

- [ ] **Step 4: Apply fix (likely call site)**

If the bug is at the bootstrap call site, find it (probably in `src/data_bootstrap.py` `_bootstrap_season_stats`) and ensure it uses `update_refresh_log_auto` with the saved/total ratio.

- [ ] **Step 5: Re-run live bootstrap to verify**

```bash
python -c "from src.data_bootstrap import _bootstrap_season_stats, BootstrapProgress; _bootstrap_season_stats(BootstrapProgress())"
```

Then check refresh_log:
```bash
python -c "import sqlite3; c = sqlite3.connect('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db'); print(c.execute(\"SELECT status, message FROM refresh_log WHERE source='season_stats'\").fetchone())"
```

Expected output: `('partial', 'saved 1073/7478 rows')` (or similar low-percentage with `'partial'`).

- [ ] **Step 6: Commit**

```bash
git add tests/test_sf2_partial_status.py src/live_stats.py src/data_bootstrap.py
git commit -m "fix(live_stats): SF-2 follow-up: 'partial' status now actually fires when <70% saved"
```

---

## Task 5: SF-5 depth_charts — MLB Stats API fallback

**Files:**
- Modify: `src/depth_charts.py` — add `fetch_depth_charts_via_statsapi()` fallback
- Modify: `src/data_bootstrap.py:864-887` — use 3-tier fallback (`fetch_with_fallback`)
- Test: `tests/test_sf5_depth_charts_fallback.py` (new)

**Goal:** Roster Resource scraping returns empty (likely JS-gated or 403). Add MLB Stats API fallback that pulls each team's roster + uses `depthChartPosition` hydration to derive starter/reliever roles. Persist `depth_chart_role` and `lineup_slot` to the players table so closer monitor and lineup protection unblock.

- [ ] **Step 1: Investigate MLB Stats API depth chart capability**

Run a one-off:

```bash
python -c "
import statsapi
r = statsapi.get('team_roster', {'teamId': 147, 'rosterType': 'depthChart'})
import json
print(json.dumps(r, indent=2)[:3000])
"
```

Verify the response contains starter/reliever/closer info. If yes → use this. If no → use the basic 40-man roster + heuristic (GS >= 5 = starter, SV >= 5 = closer).

- [ ] **Step 2: Write failing test for the fallback**

```python
# tests/test_sf5_depth_charts_fallback.py
"""SF-5: depth_charts must populate depth_chart_role/lineup_slot via MLB
Stats API when Roster Resource fails."""
import sqlite3
from unittest.mock import patch, MagicMock
from src.data_bootstrap import _bootstrap_depth_charts, BootstrapProgress


def test_falls_back_to_statsapi_when_roster_resource_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT,
                              team TEXT, mlb_id INTEGER,
                              depth_chart_role TEXT, lineup_slot INTEGER);
        CREATE TABLE refresh_log (source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT,
                                  rows_written INTEGER, rows_expected_min INTEGER,
                                  message TEXT, tier TEXT);
        INSERT INTO players VALUES (1, 'Aaron Judge', 'NYY', 592450, NULL, NULL);
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.database.DB_PATH", str(db_path))

    # Mock Roster Resource to return empty (primary tier fails)
    with patch("src.depth_charts.fetch_depth_charts", return_value={}):
        # Mock MLB Stats API to return Aaron Judge as RF in the lineup
        fake_statsapi_resp = {
            "roster": [{"person": {"id": 592450, "fullName": "Aaron Judge"},
                        "depthChartPosition": {"name": "Right Field"},
                        "battingOrder": "300"}]
        }
        with patch("src.depth_charts.fetch_depth_charts_via_statsapi",
                   return_value={"NYY": {"lineup": [{"name": "Aaron Judge", "slot": 3}]}}):
            progress = BootstrapProgress()
            _bootstrap_depth_charts(progress)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT depth_chart_role, lineup_slot FROM players WHERE player_id=1").fetchone()
    status = conn.execute("SELECT status, tier FROM refresh_log WHERE source='depth_charts'").fetchone()
    conn.close()
    assert row[0] is not None, f"Expected role populated, got {row[0]}"
    assert row[1] == 3, f"Expected slot 3, got {row[1]}"
    assert status[0] == "success", f"Expected 'success', got {status[0]}"
    assert status[1] == "fallback", f"Expected tier='fallback', got {status[1]}"
```

- [ ] **Step 3: Run test to confirm failure**

Run: `python -m pytest tests/test_sf5_depth_charts_fallback.py -v`
Expected: FAIL — `fetch_depth_charts_via_statsapi` doesn't exist.

- [ ] **Step 4: Implement the fallback**

Add to `src/depth_charts.py`:

```python
def fetch_depth_charts_via_statsapi() -> dict[str, dict]:
    """MLB Stats API fallback for depth charts.

    Returns same shape as fetch_depth_charts(): {team_code: {"lineup": [...],
    "rotation": [...], "bullpen": [...]}}.
    """
    try:
        import statsapi
    except ImportError:
        return {}

    # MLB team IDs for all 30 teams (same mapping as elsewhere in codebase)
    from src.depth_charts import _TEAM_SLUG_TO_CODE
    # Prefer to import from a single source — check if a team_id mapping exists
    # in the codebase first; otherwise hardcode the 30 IDs here.
    TEAM_ID_TO_CODE = {
        108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
        113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
        118: "KC",  119: "LAD", 120: "WSN", 121: "NYM", 133: "OAK",
        134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
        139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
        144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
    }
    out: dict[str, dict] = {}
    for team_id, code in TEAM_ID_TO_CODE.items():
        try:
            resp = statsapi.get("team_roster", {
                "teamId": team_id,
                "rosterType": "active",
                "hydrate": "person(stats(group=[hitting,pitching],type=[season]))",
            })
            lineup, rotation, bullpen = [], [], []
            for member in resp.get("roster", []):
                person = member.get("person", {})
                name = person.get("fullName", "")
                pos = member.get("position", {}).get("abbreviation", "")
                # Heuristic: starter if GS >= 5; closer if SV >= 5; else bench
                stats = person.get("stats") or []
                gs, sv = 0, 0
                for s in stats:
                    splits = s.get("splits") or []
                    for sp in splits:
                        st = sp.get("stat", {})
                        gs = max(gs, int(st.get("gamesStarted", 0) or 0))
                        sv = max(sv, int(st.get("saves", 0) or 0))
                if pos == "P":
                    if gs >= 5:
                        rotation.append({"name": name, "slot": len(rotation) + 1})
                    elif sv >= 5:
                        bullpen.insert(0, {"name": name, "role": "closer"})
                    else:
                        bullpen.append({"name": name, "role": "reliever"})
                elif pos and pos not in ("P", "DH"):
                    # Position player — append to lineup with sequential slot
                    lineup.append({"name": name, "slot": len(lineup) + 1, "pos": pos})
            out[code] = {"lineup": lineup, "rotation": rotation, "bullpen": bullpen}
        except Exception:
            continue
    return out
```

Modify `_bootstrap_depth_charts` in `src/data_bootstrap.py`:

```python
def _bootstrap_depth_charts(progress: BootstrapProgress) -> str:
    progress.phase = "Depth Charts"
    progress.detail = "Fetching depth charts..."
    try:
        from src.database import update_refresh_log
        from src.depth_charts import fetch_depth_charts, fetch_depth_charts_via_statsapi

        # Tier 1: Roster Resource scrape
        depth_data = fetch_depth_charts()
        tier = "primary"
        if not depth_data:
            # Tier 2: MLB Stats API fallback
            depth_data = fetch_depth_charts_via_statsapi()
            tier = "fallback" if depth_data else None
        if depth_data:
            count = _persist_depth_chart_roles(depth_data)
            update_refresh_log("depth_charts", "success", tier=tier)
            return f"Depth charts: {len(depth_data)} teams, {count} roles persisted ({tier})"
        update_refresh_log("depth_charts", "no_data")
        return "Skipped: depth chart endpoints returned no data"
    except Exception as e:
        logger.warning("Depth chart bootstrap failed: %s", e)
        return _format_fetch_error(e, "Depth charts")
```

- [ ] **Step 5: Run test passes**

Run: `python -m pytest tests/test_sf5_depth_charts_fallback.py -v`
Expected: PASS.

- [ ] **Step 6: Run live bootstrap to verify on real data**

```bash
python -c "from src.data_bootstrap import _bootstrap_depth_charts, BootstrapProgress; print(_bootstrap_depth_charts(BootstrapProgress()))"
python -c "import sqlite3; c = sqlite3.connect('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db'); print('roles:', c.execute('SELECT COUNT(*) FROM players WHERE depth_chart_role IS NOT NULL').fetchone()[0]); print('slots:', c.execute('SELECT COUNT(*) FROM players WHERE lineup_slot IS NOT NULL').fetchone()[0])"
```

Expected: roles > 0, slots > 0 (probably ~270+ across rostered players).

- [ ] **Step 7: Commit**

```bash
git add tests/test_sf5_depth_charts_fallback.py src/depth_charts.py src/data_bootstrap.py
git commit -m "fix(depth_charts): SF-5 add MLB Stats API fallback when Roster Resource empty"
```

---

## Task 6: SF-6 stuff_plus + batting_stats — Document as Known Limitation

**Files:**
- Modify: `src/data_bootstrap.py:1000-1199` (`_bootstrap_stuff_plus`, `_bootstrap_batting_stats`)
- Test: `tests/test_sf6_known_limitation.py` (new)

**Goal:** FanGraphs `leaders-legacy.aspx` returns 403 to non-browser scrapers. We've already documented this as a known limitation in CLAUDE.md. The fix here: make the refresh_log message link to the limitation explicitly, and ensure the optimizer's Stuff+ K-boost path (`daily_optimizer.py:963-971`) gracefully no-ops with `stuff_plus=NULL`.

**DECISION POINT (user input):** Three options for SF-6 — pick which to implement:
- **(A) Accept and document**: cleanest message in Data Status, accept neutral defaults in the K-boost path
- **(B) Wire FIP/xFIP as proxy**: when Stuff+ NULL, use `(5.0 - fip) / 5.0` clipped to [0.85, 1.15] as a Stuff+ proxy
- **(C) Build a browser-headers scraper**: try `fetch_with_fallback()` from `data_fetch_utils.py` with full browser headers + retry; this might bypass the 403

Default to **(A)** if user doesn't specify.

- [ ] **Step 1: Confirm K-boost path no-ops on NULL**

Read `src/optimizer/daily_optimizer.py:963-971`. Verify behavior when `stuff_plus` is None or 0 — should be a passive no-op (multiplier = 1.0).

- [ ] **Step 2: Write the test for option (A)**

```python
# tests/test_sf6_known_limitation.py
"""SF-6: When FG is 403, refresh_log message must clearly state the
limitation, and the K-boost code path must gracefully no-op."""
from src.optimizer.daily_optimizer import _stuff_plus_k_boost  # may need to expose this


def test_k_boost_neutral_when_stuff_plus_null():
    """With stuff_plus=None, the multiplier on K should be 1.0."""
    assert _stuff_plus_k_boost(None) == 1.0
    assert _stuff_plus_k_boost(0) == 1.0
    assert _stuff_plus_k_boost(float("nan")) == 1.0
```

(If `_stuff_plus_k_boost` doesn't exist as a separate function, refactor the inline math at lines 963-971 into a helper for testability.)

- [ ] **Step 3: Apply fix (option A)**

Modify the 403 message in `_bootstrap_stuff_plus`:

```python
return ("Skipped: FanGraphs Stuff+ unavailable (HTTP 403 — known limitation, "
        "see CLAUDE.md SF-6). Optimizer K-boost defaults to neutral 1.0×.")
```

Same shape for `_bootstrap_batting_stats`.

If user picks option (B) — FIP proxy — modify the K-boost helper to fall back to FIP when stuff_plus is null:

```python
def _stuff_plus_k_boost(stuff_plus: float | None, fip: float | None = None) -> float:
    if stuff_plus and stuff_plus > 0 and not math.isnan(stuff_plus):
        # Stuff+ 100 = league avg, +/- 10 per stdev
        return max(0.85, min(1.15, 1.0 + (stuff_plus - 100) / 100))
    if fip and fip > 0:
        # Lower FIP = better stuff
        return max(0.85, min(1.15, 1.0 + (4.0 - fip) / 10.0))
    return 1.0
```

- [ ] **Step 4: Test passes, commit**

Run: `python -m pytest tests/test_sf6_known_limitation.py -v`

```bash
git add tests/test_sf6_known_limitation.py src/data_bootstrap.py src/optimizer/daily_optimizer.py
git commit -m "fix(optimizer): SF-6 K-boost neutral on null Stuff+, honest 403 message"
```

---

## Task 7: SF-7 catcher_framing + umpire_tendencies

**Files:**
- Modify: `src/data_bootstrap.py:1414-1737`

**Goal:** All sources for catcher framing and umpire tendencies failed. Both tables empty → matchup multipliers use neutral defaults silently. Make this honest.

**DECISION POINT (user input):**
- **(A) Document as known limitation**: like SF-6, accept neutral defaults
- **(B) Try Baseball Savant CSV downloads with explicit browser headers**: may succeed where the Python wrappers fail
- **(C) Build a one-off seed file**: download the data manually once, ship it as a JSON in `data/seed/`, ship a small migration that loads it on first boot — gives a known-good baseline that ages

Default to **(A) + (C)**: document, but ship a seed file with last-known-good values so the multipliers are at least informed.

- [ ] **Step 1: Write test asserting the documented behavior**

```python
# tests/test_sf7_neutral_defaults.py
"""SF-7: catcher_framing + umpire_tendencies empty → neutral multipliers."""
from src.optimizer.matchup_adjustments import (
    catcher_framing_adjustment, umpire_adjustment
)


def test_catcher_framing_neutral_when_table_empty():
    """When catcher_framing has 0 rows, the multiplier must be 1.0."""
    # If the function takes a catcher name, pass an unknown one
    assert catcher_framing_adjustment("nonexistent_catcher") == 1.0


def test_umpire_neutral_when_table_empty():
    assert umpire_adjustment("nonexistent_umpire") == 1.0
```

- [ ] **Step 2: Confirm behavior, fix if needed**

Read the actual functions in `src/optimizer/matchup_adjustments.py` to verify they return 1.0 on missing data. If they raise or return None, fix to return 1.0 with a debug log.

- [ ] **Step 3: Update the bootstrap message to be explicit**

In `_bootstrap_catcher_framing` and `_bootstrap_umpire_tendencies`, change `"no_data"` messages to:

```python
return ("Skipped: all framing sources failed (Savant + FanGraphs + statsapi). "
        "Optimizer uses neutral 1.0× framing multiplier — see CLAUDE.md SF-7.")
```

- [ ] **Step 4: (Optional, decision point C) Create seed file**

Create `data/seed/catcher_framing_2024.json` with 2024 framing values you've verified manually. On bootstrap, if the table is empty AND the seed file exists, load it as a "warm-start" baseline (mark with `tier='emergency'`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_sf7_neutral_defaults.py src/data_bootstrap.py src/optimizer/matchup_adjustments.py
git commit -m "fix(matchup): SF-7 neutral framing/umpire multipliers, honest message"
```

---

## Task 8: SF-10 mlb_id Coverage Investigation

**Files:**
- Investigate: `src/data_bootstrap.py:221-253` (`_bootstrap_players`)
- Test: `tests/test_sf10_mlb_id_coverage.py` (new)

**Goal:** Currently 1138/9576 = 12% of players have `mlb_id`. Investigate root cause: are we fetching the wrong endpoint? Are we deduping aggressively and dropping mlb_id? This blocks SF-2 (season_stats join), SF-7 (game_logs join), and others.

**Note:** This is largely investigative. Final fix may be one-line or may require restructuring the player upsert.

- [ ] **Step 1: Diagnostic queries**

```bash
python -c "
import sqlite3
c = sqlite3.connect('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db')
print('total players:', c.execute('SELECT COUNT(*) FROM players').fetchone()[0])
print('with mlb_id:', c.execute('SELECT COUNT(*) FROM players WHERE mlb_id IS NOT NULL').fetchone()[0])
print('rostered players w/ mlb_id:',
      c.execute('SELECT COUNT(*) FROM players p JOIN league_rosters lr ON p.player_id=lr.player_id WHERE p.mlb_id IS NOT NULL').fetchone()[0])
print('rostered players total:',
      c.execute('SELECT COUNT(*) FROM players p JOIN league_rosters lr ON p.player_id=lr.player_id').fetchone()[0])
print('breakdown by source (where players were inserted):')
for r in c.execute('SELECT COUNT(*), team FROM players WHERE mlb_id IS NULL GROUP BY team ORDER BY COUNT(*) DESC LIMIT 10'):
    print(' ', r)
"
```

This tells us whether the gap is in the historical/scraped player records or in the active 40-man rosters (which is what matters for joins).

- [ ] **Step 2: Triage decision**

If rostered players have ~95%+ mlb_id coverage → the 88% gap is in historical/inactive records, low impact. Document and move on.

If rostered players have <90% mlb_id coverage → we have a real gap that breaks downstream joins. Trace `_bootstrap_players` and `_bootstrap_extended_roster` to find where mlb_id is dropped.

- [ ] **Step 3: Apply fix or document**

Depending on Step 2, either:
- (Fix) Add a backfill phase that runs after `_bootstrap_players` and looks up mlb_id by name+team for any active-roster players still missing it
- (Document) Add a comment to CLAUDE.md SF-10 noting that the 88% gap is concentrated in inactive records and doesn't affect downstream joins

- [ ] **Step 4: Commit**

```bash
git add tests/test_sf10_mlb_id_coverage.py [src/...]
git commit -m "fix(players): SF-10 backfill mlb_id for active-roster players via name+team"
```

---

## Task 10: Data Currency & Formula Correctness Audit (Parallel)

**Goal:** Independently verify that (a) every data source in the app refreshes at an appropriate cadence and is genuinely up to date, and (b) every formula throughout the codebase reads from the correct, current data columns. Find silent uses of stale or wrong data.

**Why this matters:** The 2026-04-18 audit caught 14 silent-fail patterns. This task is the proactive sweep: instead of waiting for a user-visible symptom, audit the codebase end-to-end for the same class of bugs in places nobody has looked yet.

**Structure:** Five independent read-only sub-audits dispatched as parallel agents. Each returns a structured report. Findings are aggregated into Task 11 (fix list).

### Acceptance Criteria

A finding is **actionable** if it meets at least one of:
- Code reads from a column/table that's empty or known-stale
- Formula uses hardcoded constant where a centralized source-of-truth exists (e.g., hardcoded `["R","HR","RBI",...]` instead of `LeagueConfig.hitting_categories`)
- Data source has no TTL check or has TTL longer than the data's natural change rate
- Pool enrichment skips a column that downstream consumers need
- Two paths compute the same thing differently (silent disagreement)

A finding is **NOT actionable** (skip in report) if:
- It's a pre-existing known limitation already documented in CLAUDE.md
- It's a defensive fallback that's correct by design (e.g., `1.0` neutral multiplier when data missing)

### Sub-Audit 10a: Bootstrap Phase Completeness

**Scope:** `src/data_bootstrap.py` (all 33 phases) + `refresh_log` table state

**Questions to answer:**
1. Is every defined `_bootstrap_*` function actually called from `bootstrap_all_data()`?
2. For each phase, is the TTL appropriate for the data's volatility? (e.g., live stats during games should be ≤30 min; player metadata can be 7 days)
3. Are there any phases that fetch data but never persist it?
4. Are there any phases logging `'success'` while leaving target tables empty? (verify against live DB row counts)
5. Does the parallel-execution block in `bootstrap_all_data()` have any phases that should be sequential (write-conflict risk)?

**Method:** Read all `_bootstrap_*` functions. Cross-reference with the call site. Query live `refresh_log` and target table row counts.

**Output format:**
```
PHASE                STATUS       TARGET_TABLE        ROWS    TTL_OK   ISSUE
players              success      players             9576    YES      None
season_stats         success      season_stats        1073    YES      Should be 'partial' — telemetry bug
catcher_framing      no_data      catcher_framing     0       N/A      Known limitation (SF-7)
...
```

### Sub-Audit 10b: Optimizer Formula Data References

**Scope:** All files in `src/optimizer/` (21 modules)

**Questions:**
1. Does every formula read from the correct DataFrame column? (e.g., `roster["fp_proj"]` not `roster["proj_fp"]`)
2. Are inverse stats (L, ERA, WHIP) consistently sign-flipped before SGP?
3. Are rate stats (AVG, OBP, ERA, WHIP) aggregated weighted by AB/IP, never simple-averaged?
4. Are there any references to deprecated columns or columns that don't exist in the player pool?
5. Are constants pulled from `constants_registry.py` everywhere, or are some hardcoded?
6. Is `LeagueConfig` the only source of category lists, or are there local `["R","HR",...]` arrays?

**Method:** Read each module. Cross-reference column names against `database.py` schema and `_build_player_pool()` output. Grep for hardcoded category lists.

**Output format:**
```
FILE:LINE                           ISSUE                                       SEVERITY
optimizer/foo.py:42                 reads roster["xfip"] but column not in pool MEDIUM
optimizer/bar.py:101                hardcoded ["R","HR","RBI"] (use LeagueConfig)LOW
...
```

### Sub-Audit 10c: Trade Engine Formula Data References

**Scope:** All files in `src/engine/` (6 phases, 7 sub-packages)

**Questions:** Same as 10b but for the trade analyzer. Plus:
- Does every Monte Carlo simulator use the paired-seed discipline?
- Does the cosine→1v1→2v1 scan in `trade_finder.py` reference current pool data?
- Are SGP denominators recomputed against current standings, not stale ones?

**Method:** Same as 10b.

### Sub-Audit 10d: Page-Level Data Flow

**Scope:** All files in `pages/` (11 pages)

**Questions:**
1. Does every page that shows live data go through `yahoo_data_service`?
2. Does every page use `format_stat()` for stat display, or are there local format strings?
3. Are there pages computing things that already exist as utility functions (DRY violation = correctness risk)?
4. Are there pages reading directly from SQL when the player pool already has the data?
5. Are there pages with hardcoded TTLs that diverge from `yahoo_data_service` TTLs?

**Method:** Read each page. Cross-reference imports against the unified services in `src/`.

### Sub-Audit 10e: Cross-Codebase Consistency

**Scope:** Repo-wide grep + structural checks

**Questions:**
1. How many distinct hardcoded category lists exist? (`grep -r '"R", "HR", "RBI"' src/ pages/`)
2. How many places call `sqlite3.connect()` directly instead of `get_connection()`? (WAL+busy_timeout setup gets bypassed)
3. How many places format stats with custom `f"{x:.3f}"` instead of `format_stat()`?
4. How many places import `LeagueConfig` vs how many should but don't?
5. Are there any places still using `datetime.utcnow()` (deprecated) instead of `datetime.now(UTC)`?

**Method:** Repo-wide greps. Structural sampling.

**Output format:** Categorical counts + file:line list for each.

---

### Task 10 Aggregation

After all 5 sub-audits return:

- [ ] **Step 1: Compile findings**

Combine all 5 reports into one consolidated finding list. Categorize by severity:
- **HIGH**: Silently wrong data, user-visible
- **MEDIUM**: Silently wrong data, masked by current defaults
- **LOW**: Style/consistency, no data wrongness

- [ ] **Step 2: Add fix tasks to this plan**

For each HIGH and MEDIUM finding, append a new Task (Task 11+) with the same TDD structure used elsewhere in this plan.

LOW findings get a single bulk-cleanup task (Task N: "Style consistency cleanup").

- [ ] **Step 3: Re-run self-review on the expanded plan**

Confirm no placeholders, type consistency, etc.

---

## Task 11+: Fix Tasks From Audit (RESOLVED 2026-05-10 — 5 audit agents returned)

The 5 parallel audit agents identified 14 NEW silent-fail patterns (SF-15..SF-28) in addition to the original SF-1..SF-14. Each fix below follows the TDD pattern: failing test → minimal fix → passing test → commit.

### Task 11: SF-19 — sprint_speed column not in player pool (HIGH)

**Files:** `src/database.py` (`_load_player_pool_impl`), `tests/test_sf19_pool_columns.py` (new)

**Bug:** `src/optimizer/daily_optimizer.py:958` and `src/optimizer/streaming.py:268` read `player.get("sprint_speed", 0)` but the pool SQL never selects `sprint_speed` from `statcast_archive` — only xwoba, xba, barrel_pct, hard_hit_pct, ev_mean, stuff_plus, babip. Every read returns 0, silently disabling the SB sprint-speed boost and the SB-streaming K1 score.

**Fix:** Add `sa.sprint_speed AS sprint_speed` to both pool SQL queries (main + fallback).

**Test:** Build a pool, assert `"sprint_speed" in pool.columns` AND there's at least one non-null value when statcast_archive has data.

---

### Task 12: SF-20 — Database fallback rate-stat aggregation (HIGH)

**Files:** `src/database.py:1655-1661`, `tests/test_sf20_rate_aggregation.py` (new)

**Bug:** Fallback projection-blending query uses `AVG(proj.avg)`, `AVG(proj.obp)`, `AVG(proj.era)`, `AVG(proj.whip)` — simple-mean across projection systems, NOT component-weighted. AVG should be `SUM(h)/SUM(ab)`, ERA should be `SUM(er)*9/SUM(ip)`. This path activates whenever blended projections don't exist (sample-data mode + edge cases).

**Fix:** Replace the AVG aggregations with weighted formulas using the per-system H/AB/ER/IP/BB/HBP/SF columns.

**Test:** Build sample data with two divergent projection systems, assert the fallback rate stats match the weighted formula not the simple mean.

---

### Task 13: SF-21 — Stale `_LC` singleton in category_analysis (HIGH)

**Files:** `src/engine/portfolio/category_analysis.py:24,103`, `tests/test_sf21_stale_lc.py` (new)

**Bug:** `_LC = _LC_Class()` is constructed once at import time. `compute_marginal_sgp` reads `_LC.sgp_denominators` at line 103. When `build_valuation_context` updates denominators from live standings, the updated `local_config.sgp_denominators` never reaches `_LC`. Direct callers of `compute_marginal_sgp` always see pre-season defaults.

**Fix:** Make `compute_marginal_sgp` accept `config: LeagueConfig` as a required parameter and read denominators from it. Remove module-level `_LC`.

**Test:** Call `compute_marginal_sgp` once with default config, then with a config whose denominators are doubled, assert the returned SGP differs accordingly.

---

### Task 14: SF-22 — park_factors Tier 1 pybaseball data ignored (HIGH)

**Files:** `src/data_bootstrap.py:_bootstrap_park_factors` (around line 426), `tests/test_sf22_park_factor_tier.py` (new)

**Bug:** Audit 10a found that `source_dict = _PARK_FACTORS_EMERGENCY_2026` is set unconditionally before an `if isinstance(data, dict)` overwrite that may not fire. Effect: even when pybaseball returns valid 2026 park factors, the emergency dict gets used.

**Fix:** Reorder so `source_dict` is only set to `_PARK_FACTORS_EMERGENCY_2026` when Tier 1 returns nothing. Mark `tier='primary'` vs `tier='emergency'` in refresh_log.

**Test:** Mock pybaseball to return valid data, call `_bootstrap_park_factors`, assert `tier='primary'` and the saved factors match the mock data not the emergency constants.

---

### Task 15: SF-23 — 7 pages bypass Yahoo service for live league data (HIGH)

**Files:** `pages/15_Weekly_Recap.py`, `pages/7_Weekly_Dashboard.py`, `pages/10_Category_Tracker.py`, `pages/9_Waiver_Wire.py`, `pages/8_Trade_Values.py`, `pages/14_Punt_Analyzer.py`, `pages/12_Playoff_Odds.py`

**Bug:** Each page calls `load_league_rosters()` / `load_league_standings()` which read raw SQLite cached from the last bootstrap. No TTL enforcement, no Yahoo refresh trigger, no live-data badge. Users see hours-old standings/rosters with no warning.

**Fix:** For each page, replace the load with `get_yahoo_data_service().get_rosters()` / `.get_standings()`. The service handles 30-min TTL, automatic refresh, and 429 backoff.

**Test (per page):** Mock `yahoo_data_service.get_rosters` to return a known DataFrame, render the page (or its data-load path), assert the page consumed data from the mock not from raw SQL.

**Note:** This is 7 file changes but the pattern is identical. Single agent can do them as a batch.

---

### Task 16: SF-25 — 7 SGP local reinventions consolidated to SGPCalculator (HIGH)

**Files (7 sites):**
- `src/trade_finder.py:1503` (`_totals_sgp`)
- `src/trade_finder.py:1475` (`_weighted_totals_sgp`)
- `src/waiver_wire.py:502` (`_totals_to_sgp`)
- `src/engine/monte_carlo/trade_simulator.py:221` (`_simulate_roster_sgp`)
- `src/in_season.py:147-149` (inline)
- `src/start_sit.py:801-816` (inline)
- `src/optimizer/daily_optimizer.py:938-940` (inline)
- `src/engine/output/trade_evaluator.py:908-927` (inline Phase 1 SGP delta)

**Bug:** Each site replicates the `(value - replacement) / denom` SGP formula with its own inverse-stat sign-flip handling. Any future change to `SGPCalculator` (e.g., volume weighting on rate stats) won't propagate.

**Fix strategy:** Add a `SGPCalculator.totals_sgp(totals: dict, weights: dict|None=None) -> float` method that consolidates the loop. Migrate each call site to use it. Delete the local helpers.

**Test:** Each migrated site gets a test asserting parity with the original local computation on a fixed input. Then add ONE test on `SGPCalculator.totals_sgp` itself covering inverse stats, weighted/unweighted, and missing categories.

**Acceptance:** `grep -rn "_totals_sgp\|_totals_to_sgp\|_simulate_roster_sgp" src/` returns 0 hits outside `src/valuation.py`.

---

### Task 17: 9_League_Standings — duplicate projected_team_totals (HIGH)

**Files:** `pages/9_League_Standings.py:110-186`

**Bug:** `_compute_projected_team_totals()` is a 70-line reimplementation of `standings_utils.get_all_team_totals()` that runs cold SQL on every render. The canonical version is session-cached.

**Fix:** Replace `_compute_projected_team_totals()` calls with `standings_utils.get_all_team_totals(...)`. Delete the local function.

**Test:** Render the page (or its compute path), assert it consumed the cached service.

---

### Task 18: Engine hardcoded categories + SGP_DENOMS centralization (HIGH)

**Files:**
- `src/engine/portfolio/valuation.py:52-53` (CATEGORIES, INVERSE_CATEGORIES)
- `src/engine/portfolio/copula.py:38,67` (CATEGORIES, INVERSE_CATEGORIES)
- `src/engine/game_theory/opponent_valuation.py:26-56` (CATEGORIES + DEFAULT_SGP_DENOMS)
- `src/engine/monte_carlo/trade_simulator.py:32` (imports stale module constants)
- `src/engine/game_theory/sensitivity.py:37` (INVERSE_CATEGORIES)
- `src/optimizer/scenario_generator.py:429-430` (hitting_cats, pitching_cats)
- `src/optimizer/dual_objective.py:47`, `pipeline.py:121`, `shared_data_layer.py:433` (inverse cats)
- `src/optimizer/projections.py:40`, `shared_data_layer.py:883` (RATE_CATS)
- `src/war_room.py:20-23`, `src/ui_shared.py:2241`, `src/leaders.py:19`, `src/player_databank.py:26-27`
- `app.py:2056,2136`, `pages/12_Matchup_Planner.py:59`, `pages/9_League_Standings.py:72-76`, `pages/1_My_Team.py:1805,1404`, `pages/10_Leaders.py:328`, `pages/6_Line-up_Optimizer.py:1071-1083`

**Bug:** 12+ hardcoded CATEGORIES lists, 9+ hardcoded `{"L","ERA","WHIP"}` sets, 4+ hardcoded `RATE_CATS` sets. `pages/1_My_Team.py:1404` even drops "L" from inverse — silent miscategorization.

**Fix:** Replace each with `LeagueConfig.all_categories` / `.inverse_stats` / `.rate_stats`. For `engine/game_theory/opponent_valuation.py`, accept config as a parameter and read denominators from it.

**Test:** A repo-wide structural test (`tests/test_no_hardcoded_categories.py`) that greps for the bad patterns and asserts count == 0.

---

### Task 19: SF-15 — Race conditions on statcast_archive concurrent writes (MEDIUM)

**Files:** `src/data_bootstrap.py` Phase 22-24 executor block

**Bug:** `_bootstrap_batting_stats` and `_bootstrap_sprint_speed` both write to `statcast_archive` in the same parallel pool. UNIQUE constraint violation possible for hitters appearing in both datasets.

**Fix:** Either serialize the two phases OR change INSERT to `INSERT OR REPLACE` / `ON CONFLICT(player_id, season) DO UPDATE`.

**Test:** Concurrent-write reproducer using ThreadPoolExecutor + the same player_id from two threads, assert both succeed without error.

---

### Task 20: SF-16 — 7 phases use hardcoded TTLs instead of StalenessConfig (MEDIUM)

**Files:** `src/data_bootstrap.py` (`_bootstrap_adp_sources`, `_bootstrap_depth_charts`, `_bootstrap_contracts`, `_bootstrap_dynamic_park_factors`, `_bootstrap_bat_speed`, `_bootstrap_forty_man`, game_logs phase)

**Bug:** These phases pass integer literals (`24`, `168`, `720`, etc.) to `check_staleness()`. Callers passing custom `StalenessConfig` cannot override these. The config is misleadingly incomplete.

**Fix:** Add `adp_sources_hours`, `depth_charts_hours`, `contracts_hours`, `dynamic_park_factors_hours`, `bat_speed_hours`, `forty_man_hours`, `game_logs_hours` fields to `StalenessConfig` (with the current defaults). Update each phase to read from config.

**Test:** Pass custom config with all fields halved, run bootstrap with prior data 12 hours old, assert the stale-trigger logic fires for all 7 phases.

---

### Task 21: SF-17 — ROS projections TTL too aggressive + ordering dependency (MEDIUM)

**Files:** `src/data_bootstrap.py` Phase 19 (`_bootstrap_ros_projections` inline block)

**Bug:** Uses `_live_stats_ttl_hours` (15-60 min) for an expensive PyMC MCMC computation. Also runs BEFORE `team_strength` (P21) and `game_day` (P20) but `update_ros_projections` depends on team strength.

**Fix:** Use a fixed 4h or 24h TTL for ROS projections. Move Phase 19 to AFTER Phase 21 (team_strength).

**Test:** Verify Phase 19 fires only when ROS projections are >4h old, and verify Phase 19 runs after Phase 21 in the call order.

---

### Task 22: SF-18 — Yahoo transactions/FA phases have no staleness gate (MEDIUM)

**Files:** `src/data_bootstrap.py` Phases 17, 18

**Bug:** Phase 17 (`yahoo_transactions`) and Phase 18 (`yahoo_free_agents`) run on every bootstrap. P18 has no `update_refresh_log` entry at all → invisible in Data Status panel.

**Fix:** Wrap both in `check_staleness()` with the same TTL as `yahoo_data_service` (15min for transactions, 1h for FA). Add `update_refresh_log` for P18.

**Test:** With recently-fetched data, assert both phases skip. With stale data, both run.

---

### Task 23: SF-24 — 5+ pages format ERA/WHIP at .3f instead of .2f (MEDIUM)

**Files:** `pages/15_Weekly_Recap.py:119-121`, `pages/7_Weekly_Dashboard.py:534-535`, `pages/10_Category_Tracker.py:135-141`, `pages/14_Punt_Analyzer.py:200`, `pages/5_Free_Agents.py:784,802,820`, `pages/1_My_Team.py:2037-2043`

**Bug:** Inline `f"{x:.3f}"` for rate stats including ERA/WHIP. ERA `3.850` displayed instead of `3.85`. Bypasses `format_stat()`.

**Fix:** Replace each with `format_stat(x, stat_type)`.

**Test:** Per-page render snapshot test.

---

### Task 24: SF-27 — MatchupContextService bypasses (MEDIUM)

**Files:** `pages/10_Category_Tracker.py:102` (calls `compute_category_weights_from_analysis` direct), `pages/6_Line-up_Optimizer.py:3201` (calls `compute_h2h_category_weights` direct)

**Bug:** Bypasses produce divergent weight vectors vs. canonical service.

**Fix:** Route through `get_matchup_context()` mode='matchup' or 'blended'.

**Test:** Per page assert weights match the service output.

---

### Task 25: SF-28 — 3 scripts use direct sqlite3.connect (LOW)

**Files:** `scripts/draft_vs_current.py:8`, `scripts/extract_trade_data.py:18`, `scripts/optimal_roster_sim.py:8`

**Bug:** Bypass WAL+busy_timeout. Running these while the app is live can cause lock contention (SF-4 pattern).

**Fix:** Replace each with `from src.database import get_connection; conn = get_connection()`.

**Test:** Static grep test asserting `sqlite3.connect` count outside `src/database.py` and `tests/` is 0.

---

### Cross-cutting bonus findings (one-line fixes)

- `src/optimizer/streaming.py:608` — replace inline `team_weekly_ip = 55.0` with `_CONSTANTS.get("team_weekly_ip", 55.0)`
- `src/optimizer/streaming.py:166` — register `streaming_baseline_whip = 1.25` in constants_registry, read it
- `src/optimizer/category_urgency.py:22-23` — read `COUNTING_STAT_K`/`RATE_STAT_K` from `CONSTANTS_REGISTRY` (currently dead-keys)
- `src/optimizer/shared_data_layer.py:319` — add `tracker.record("team_strength", ...)` after team_strength load
- `src/data_bootstrap.py:_bootstrap_contracts` — return `update_refresh_log_auto` with proper `expected_min` so 0-rows-updated is `partial`/`no_data` not `success`
- `src/engine/portfolio/valuation.py:247-248` — replace `__dict__.update` with proper config copy
- `src/engine/monte_carlo/trade_simulator.py:144` — fix the antithetic-variate comment OR implement true antithetic via negated quantiles
- `app.py:2056,2136` — remove the two app-level hardcoded category lists

---

## Task 9: Update CLAUDE.md Audit Status

**Files:**
- Modify: `CLAUDE.md` — the "Known Data Issues — 2026-04-18 Audit" section

**Goal:** After all fixes land, mark each SF-N item with its final state and update the "Unresolved contradiction" section.

- [ ] **Step 1: Update each task with status**

For each SF-N now resolved, change the heading to e.g. `**SF-5 · depth_charts table has 0 rows** ✅ RESOLVED 2026-05-10` and add a short note pointing to the commit hash.

- [ ] **Step 2: Replace "Unresolved contradiction" section**

After Task 1 lands, replace the section with the actual finding (e.g. "the gate was dead code for pure-`P` positions; patched at commit X to also handle solo-`P` markers").

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): close out 2026-04-18 audit — all 14 findings resolved or documented"
```

---

## Self-Review

**1. Spec coverage:** All 14 SF items + the unresolved contradiction are addressed.
- SF-1 ✅ (already fixed, no task)
- SF-2 → Task 4 (follow-up)
- SF-3 ✅ (already fixed)
- SF-4 ✅ (already fixed)
- SF-5 → Task 5
- SF-6 → Task 6
- SF-7 → Task 7
- SF-8 ✅ (already fixed)
- SF-9 ✅ (already fixed)
- SF-10 → Task 8
- SF-11 → Task 2
- SF-12 ✅ (already fixed)
- SF-13 → Task 3
- SF-14 ✅ (already fixed)
- Unresolved → Task 1
- Doc cleanup → Task 9

**2. Placeholder scan:** Tasks 6, 7 have decision points marked clearly. No "TBD" / "implement later" lurking.

**3. Type consistency:** `update_refresh_log_auto` and `update_refresh_log` signatures consistent across tasks. `_bootstrap_*` functions follow the same `(progress: BootstrapProgress) -> str` shape.

**4. Risk:** Task 5 (SF-5 fallback) has the most novel code — the MLB Stats API depth chart hydration. Step 1 of that task is a manual probe to confirm the API actually returns the data we need before committing to the implementation. If the probe fails, fall back to the 40-man + GS heuristic.
