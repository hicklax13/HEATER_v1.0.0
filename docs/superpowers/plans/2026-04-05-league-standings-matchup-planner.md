# League Standings + Matchup Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:dispatching-parallel-agents to implement this plan. Tasks are organized into waves — all tasks within a wave can run in parallel. Wait for a wave to complete before starting the next.

**Goal:** Build a new League Standings page with live Yahoo H2H standings and enhanced MC projections, and redesign the Matchup Planner with per-category win probabilities and week-by-week navigation.

**Architecture:** A shared `src/standings_engine.py` module provides all computation (Bayesian-updated Normal + Gaussian copula for category win probabilities, schedule-aware MC season simulation, magic numbers, team strength profiles). Two Streamlit pages consume this engine. Yahoo data layer enhanced with full league schedule fetching and W-L-T record storage.

**Tech Stack:** Python, Streamlit, NumPy, SciPy (norm.cdf, Cholesky), pandas, SQLite, yfpy (Yahoo Fantasy API)

**Spec:** `docs/superpowers/specs/2026-04-05-league-standings-matchup-planner-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/database.py` | MODIFY | Add `league_schedule_full` + `league_records` tables, CRUD functions |
| `src/yahoo_data_service.py` | MODIFY | Add `get_full_league_schedule()`, `get_week_scoreboard()`, capture W-L-T meta columns |
| `src/standings_engine.py` | CREATE | Category win probabilities, enhanced MC simulation, magic numbers, team strength |
| `pages/8_League_Standings.py` | CREATE | New 2-tab page: Current Standings + Season Projections |
| `pages/8_Standings.py` | DELETE | Replaced by League Standings |
| `pages/11_Matchup_Planner.py` | REWRITE | Add Category Probabilities tab + week navigator |
| `src/ui_shared.py` | MODIFY | Add "league_standings" sidebar icon entry + PAGE_ICONS key |
| `tests/test_standings_engine.py` | CREATE | Unit tests for all engine functions |
| `tests/test_league_standings_integration.py` | CREATE | Page-level integration tests |

---

## Wave 1: Foundation (3 parallel agents)

All three tasks in this wave are independent and run simultaneously.

---

### Task 1: Database Schema Additions

**Agent focus:** Add two new tables and CRUD functions to `src/database.py`. Test-driven.

**Files:**
- Modify: `src/database.py` (lines 177-249 for CREATE TABLE section, ~line 1362 for new functions)
- Create: `tests/test_standings_engine.py` (DB-related tests only — other agents will append to this file later)

- [ ] **Step 1: Write failing tests for new DB tables and functions**

Create `tests/test_standings_engine.py` with these tests:

```python
"""Tests for standings engine — database layer."""

from __future__ import annotations

import pytest
import pandas as pd

from src.database import (
    init_db,
    get_connection,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure fresh DB for each test."""
    init_db()


class TestLeagueScheduleFullTable:
    """Tests for league_schedule_full table CRUD."""

    def test_upsert_and_load_full_schedule(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team C", "Team D")
        upsert_league_schedule_full(2, "Team A", "Team C")

        result = load_league_schedule_full()
        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2  # 2 matchups in week 1
        assert ("Team A", "Team B") in result[1]

    def test_upsert_full_schedule_idempotent(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team A", "Team B")  # duplicate

        result = load_league_schedule_full()
        assert len(result[1]) == 1  # no duplicate

    def test_load_empty_full_schedule(self):
        from src.database import load_league_schedule_full

        result = load_league_schedule_full()
        assert result == {}


class TestLeagueRecordsTable:
    """Tests for league_records table CRUD."""

    def test_upsert_and_load_records(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team Hickey", wins=42, losses=32, ties=6,
                             win_pct=0.563, streak="L1", rank=3)
        upsert_league_record("Jonny Jockstrap", wins=48, losses=26, ties=6,
                             win_pct=0.638, streak="W3", rank=1)

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.loc[df["team_name"] == "Team Hickey", "wins"].iloc[0] == 42

    def test_upsert_record_overwrites(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team A", wins=10, losses=5, ties=1,
                             win_pct=0.656, streak="W1", rank=1)
        upsert_league_record("Team A", wins=11, losses=5, ties=1,
                             win_pct=0.688, streak="W2", rank=1)

        df = load_league_records()
        assert len(df) == 1
        assert df.iloc[0]["wins"] == 11

    def test_load_empty_records(self):
        from src.database import load_league_records

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_standings_engine.py -v`
Expected: FAIL with ImportError (functions don't exist yet)

- [ ] **Step 3: Add CREATE TABLE statements to `init_db()`**

In `src/database.py`, after the `league_schedule` CREATE TABLE (line ~364), add:

```python
    # ── League schedule (full — all teams, all weeks) ──────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS league_schedule_full (
            week INTEGER NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            PRIMARY KEY (week, team_a, team_b)
        )
    """)

    # ── League W-L-T records ───────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS league_records (
            team_name TEXT PRIMARY KEY,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            ties INTEGER DEFAULT 0,
            win_pct REAL DEFAULT 0.0,
            points_for REAL DEFAULT 0.0,
            points_against REAL DEFAULT 0.0,
            streak TEXT DEFAULT '',
            rank INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
```

- [ ] **Step 4: Add CRUD functions**

After `load_league_schedule()` (line ~1375), add:

```python
def upsert_league_schedule_full(week: int, team_a: str, team_b: str) -> None:
    """Insert or replace a full-league matchup entry."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO league_schedule_full (week, team_a, team_b) VALUES (?, ?, ?)",
            (week, team_a, team_b),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_schedule_full() -> dict[int, list[tuple[str, str]]]:
    """Load full league schedule: {week: [(team_a, team_b), ...]}."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT week, team_a, team_b FROM league_schedule_full ORDER BY week"
        ).fetchall()
    finally:
        conn.close()
    result: dict[int, list[tuple[str, str]]] = {}
    for week, team_a, team_b in rows:
        week = int(week)
        result.setdefault(week, []).append((str(team_a), str(team_b)))
    return result


def upsert_league_record(
    team_name: str,
    wins: int = 0,
    losses: int = 0,
    ties: int = 0,
    win_pct: float = 0.0,
    points_for: float = 0.0,
    points_against: float = 0.0,
    streak: str = "",
    rank: int = 0,
) -> None:
    """Insert or replace a team's W-L-T record."""
    from datetime import datetime, UTC

    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO league_records
               (team_name, wins, losses, ties, win_pct, points_for,
                points_against, streak, rank, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (team_name, wins, losses, ties, win_pct, points_for,
             points_against, streak, rank, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_records() -> pd.DataFrame:
    """Load all team W-L-T records."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM league_records ORDER BY rank",
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=[
            "team_name", "wins", "losses", "ties", "win_pct",
            "points_for", "points_against", "streak", "rank", "updated_at",
        ])
    for col in ("wins", "losses", "ties", "rank"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce").fillna(0.0)
    return df
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_standings_engine.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest -x -q`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/database.py tests/test_standings_engine.py
git commit -m "feat: add league_schedule_full and league_records DB tables with CRUD"
```

---

### Task 2: Yahoo Data Service Enhancements

**Agent focus:** Add `get_full_league_schedule()`, `get_week_scoreboard()`, and capture W-L-T meta columns from standings. Test-driven.

**Files:**
- Modify: `src/yahoo_data_service.py` (add methods after existing get_schedule at line ~352)
- Modify: `src/yahoo_api.py` (add `get_league_scoreboard_for_week()` wrapper if needed)
- Create: `tests/test_yahoo_schedule.py`

**Note:** These functions call Yahoo API which requires auth. Tests must mock the Yahoo client.

- [ ] **Step 1: Write failing tests**

Create `tests/test_yahoo_schedule.py`:

```python
"""Tests for Yahoo schedule and records fetching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from src.database import init_db


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db()


class TestFetchAndSyncRecords:
    """Test W-L-T record capture from standings."""

    def test_sync_records_stores_wlt(self):
        from src.database import load_league_records, upsert_league_record

        upsert_league_record("Team A", wins=10, losses=5, ties=1,
                             win_pct=0.656, streak="W2", rank=1)
        df = load_league_records()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["wins"] == 10
        assert row["losses"] == 5
        assert row["streak"] == "W2"


class TestFullLeagueScheduleParsing:
    """Test schedule parsing logic (no Yahoo API call)."""

    def test_parse_scoreboard_matchups(self):
        """Test that we correctly parse matchup pairs from a scoreboard response."""
        # Simulate the structure returned by yfpy scoreboard
        from src.standings_engine import parse_scoreboard_matchups

        mock_matchups = [
            {"team_a": "Team Hickey", "team_b": "Baty Babies"},
            {"team_a": "Jonny Jockstrap", "team_b": "Shohei Show"},
            {"team_a": "The Good The Vlad The Ugly", "team_b": "Team F"},
        ]
        result = parse_scoreboard_matchups(mock_matchups)
        assert len(result) == 3
        assert ("Team Hickey", "Baty Babies") in result

    def test_find_user_opponent_in_schedule(self):
        """Test finding user's opponent for a given week."""
        from src.standings_engine import find_user_opponent

        schedule = {
            1: [("Team Hickey", "Baty Babies"), ("Team C", "Team D")],
            2: [("Jonny Jockstrap", "Team Hickey"), ("Team C", "Team D")],
        }
        assert find_user_opponent(schedule, 1, "Team Hickey") == "Baty Babies"
        assert find_user_opponent(schedule, 2, "Team Hickey") == "Jonny Jockstrap"
        assert find_user_opponent(schedule, 3, "Team Hickey") is None


class TestWeekScoreboardParsing:
    """Test scoreboard result parsing for past weeks."""

    def test_parse_week_results(self):
        """Test parsing category W/L results from a completed week."""
        from src.standings_engine import parse_week_category_results

        # Simulate category-level results
        categories = [
            {"name": "R", "user_val": 38, "opp_val": 33, "is_inverse": False},
            {"name": "ERA", "user_val": 3.42, "opp_val": 3.10, "is_inverse": True},
            {"name": "SB", "user_val": 5, "opp_val": 5, "is_inverse": False},
        ]
        result = parse_week_category_results(categories)
        assert result[0]["result"] == "W"   # R: 38 > 33
        assert result[1]["result"] == "L"   # ERA: 3.42 > 3.10 (inverse, lower wins)
        assert result[2]["result"] == "T"   # SB: tied
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_yahoo_schedule.py -v`
Expected: FAIL with ImportError (standings_engine doesn't exist yet)

- [ ] **Step 3: Create stub functions in `src/standings_engine.py`**

Create `src/standings_engine.py` with the helper functions that the Yahoo layer and tests need:

```python
"""Standings engine — shared computation for League Standings + Matchup Planner.

Pure functions (no Streamlit dependency). All Yahoo/DB I/O happens in callers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Schedule helpers ────────────────────────────────────────────────


def parse_scoreboard_matchups(
    matchups: list[dict],
) -> list[tuple[str, str]]:
    """Extract (team_a, team_b) pairs from a scoreboard response.

    Args:
        matchups: List of dicts with "team_a" and "team_b" keys.

    Returns:
        List of (team_a_name, team_b_name) tuples.
    """
    result = []
    for m in matchups:
        team_a = str(m.get("team_a", ""))
        team_b = str(m.get("team_b", ""))
        if team_a and team_b:
            result.append((team_a, team_b))
    return result


def find_user_opponent(
    schedule: dict[int, list[tuple[str, str]]],
    week: int,
    user_team_name: str,
) -> str | None:
    """Find the user's opponent for a specific week.

    Args:
        schedule: {week: [(team_a, team_b), ...]}
        week: Week number to look up.
        user_team_name: User's team name.

    Returns:
        Opponent team name, or None if not found.
    """
    matchups = schedule.get(week, [])
    for team_a, team_b in matchups:
        if team_a == user_team_name:
            return team_b
        if team_b == user_team_name:
            return team_a
    return None


def parse_week_category_results(
    categories: list[dict],
) -> list[dict]:
    """Parse category-level W/L/T results from a completed matchup.

    Args:
        categories: List of dicts with name, user_val, opp_val, is_inverse.

    Returns:
        List of dicts with name, user_val, opp_val, result ("W"/"L"/"T").
    """
    results = []
    for cat in categories:
        name = cat["name"]
        user_val = float(cat["user_val"])
        opp_val = float(cat["opp_val"])
        is_inverse = bool(cat.get("is_inverse", False))

        if user_val == opp_val:
            result = "T"
        elif is_inverse:
            result = "W" if user_val < opp_val else "L"
        else:
            result = "W" if user_val > opp_val else "L"

        results.append({
            "name": name,
            "user_val": user_val,
            "opp_val": opp_val,
            "result": result,
        })
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_yahoo_schedule.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Enhance `_fetch_and_sync_standings()` to capture W-L-T meta columns**

In `src/yahoo_data_service.py`, modify `_fetch_and_sync_standings()` (lines 500-543). After the existing code that filters out meta_cols and upserts category data, add this block to also save W-L-T records:

```python
        # ── Also capture W-L-T records ──────────────────────────
        from src.database import upsert_league_record

        meta_col_map = {
            "wins": "wins", "losses": "losses", "ties": "ties",
            "percentage": "win_pct", "points_for": "points_for",
            "points_against": "points_against", "streak": "streak",
            "rank": "rank",
        }
        for _, row in standings_df.iterrows():
            team_name = str(row.get("team_name", ""))
            if not team_name:
                continue
            record_kwargs = {}
            for src_col, dst_key in meta_col_map.items():
                val = row.get(src_col)
                if val is not None:
                    if dst_key in ("wins", "losses", "ties", "rank"):
                        try:
                            record_kwargs[dst_key] = int(float(val))
                        except (ValueError, TypeError):
                            record_kwargs[dst_key] = 0
                    elif dst_key == "win_pct":
                        try:
                            record_kwargs[dst_key] = float(val)
                        except (ValueError, TypeError):
                            record_kwargs[dst_key] = 0.0
                    else:
                        record_kwargs[dst_key] = str(val) if val else ""
            if record_kwargs:
                upsert_league_record(team_name, **record_kwargs)
```

- [ ] **Step 6: Add `get_full_league_schedule()` method to `YahooDataService`**

In `src/yahoo_data_service.py`, after `get_schedule()` (line ~352), add:

```python
    def get_full_league_schedule(
        self, force_refresh: bool = False, total_weeks: int = 24,
    ) -> dict[int, list[tuple[str, str]]]:
        """Fetch all matchups for all weeks (all 12 teams).

        Returns: {week: [(team_a, team_b), ...]} with 6 matchups per week.
        Cached in session_state with 24h TTL. Stored in league_schedule_full table.
        """
        import streamlit as st
        from src.database import load_league_schedule_full, upsert_league_schedule_full

        cache_key = "_full_league_schedule"
        ts_key = "_full_league_schedule_ts"

        # Check session cache
        if not force_refresh and cache_key in st.session_state:
            cached_ts = st.session_state.get(ts_key, 0)
            if (time.monotonic() - cached_ts) < 86400:  # 24h TTL
                return st.session_state[cache_key]

        # Check DB cache
        db_schedule = load_league_schedule_full()
        if db_schedule and not force_refresh:
            st.session_state[cache_key] = db_schedule
            st.session_state[ts_key] = time.monotonic()
            return db_schedule

        # Fetch from Yahoo API
        if not self._client or not self.is_connected():
            logger.warning("Yahoo not connected — returning DB/empty schedule")
            return db_schedule or {}

        schedule: dict[int, list[tuple[str, str]]] = {}
        try:
            for week in range(1, total_weeks + 1):
                try:
                    scoreboard = self._client._query.get_league_scoreboard_by_week(
                        chosen_week=week
                    )
                    if not scoreboard or not hasattr(scoreboard, "matchups"):
                        continue
                    week_matchups = []
                    for matchup in scoreboard.matchups:
                        teams = getattr(matchup, "teams", None)
                        if not teams or len(teams) < 2:
                            continue
                        team_a_name = ""
                        team_b_name = ""
                        for i, team_entry in enumerate(teams):
                            team_obj = team_entry.get("team") if isinstance(team_entry, dict) else team_entry
                            name = getattr(team_obj, "name", None)
                            if isinstance(name, bytes):
                                name = name.decode("utf-8", errors="replace")
                            name = str(name) if name else f"Team_{i}"
                            if i == 0:
                                team_a_name = name
                            else:
                                team_b_name = name
                        if team_a_name and team_b_name:
                            week_matchups.append((team_a_name, team_b_name))
                            upsert_league_schedule_full(week, team_a_name, team_b_name)
                    schedule[week] = week_matchups
                    time.sleep(0.5)  # Rate limit
                except Exception as exc:
                    logger.warning("Failed to fetch week %d schedule: %s", week, exc)
                    continue
        except Exception as exc:
            logger.error("Full schedule fetch failed: %s", exc)

        # Merge with DB for any missing weeks
        if not schedule:
            schedule = db_schedule or {}
        else:
            for week, matchups in (db_schedule or {}).items():
                if week not in schedule:
                    schedule[week] = matchups

        st.session_state[cache_key] = schedule
        st.session_state[ts_key] = time.monotonic()
        return schedule
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest -x -q`
Expected: All tests pass (Yahoo methods are only called via live connection, not in unit tests)

- [ ] **Step 8: Commit**

```bash
git add src/standings_engine.py src/yahoo_data_service.py tests/test_yahoo_schedule.py
git commit -m "feat: add full league schedule fetching and W-L-T record capture from Yahoo"
```

---

### Task 3: UI Shared Enhancements

**Agent focus:** Add missing "league_standings" icon to `PAGE_ICONS` and sidebar JS icon map in `src/ui_shared.py`.

**Files:**
- Modify: `src/ui_shared.py` (PAGE_ICONS dict near line ~24, sidebar JS icons inside `inject_custom_css()` near line ~503)

- [ ] **Step 1: Add "league_standings" icon to PAGE_ICONS**

In `src/ui_shared.py`, in the `PAGE_ICONS` dict (after the existing icon entries, around line ~260), add:

```python
    "league_standings": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 17V9"/><path d="M11 17V5"/><path d="M15 17v-4"/><path d="M19 17v-8"/></svg>',
```

This is a bar chart icon (ascending bars) that represents standings/rankings.

- [ ] **Step 2: Add sidebar icon entry for "League Standings" page**

In `inject_custom_css()`, find the JavaScript `icons` dict that maps sidebar link text to SVG icons. Add an entry for the new page name:

```javascript
'League Standings': '<svg ...same SVG as above...>',
```

The sidebar icon map keys must match the sidebar link text exactly (which comes from the filename: `8_League_Standings.py` renders as "League Standings").

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `python -m pytest tests/test_ui_shared.py -v`
Expected: All existing UI tests pass

- [ ] **Step 4: Commit**

```bash
git add src/ui_shared.py
git commit -m "feat: add league_standings icon to PAGE_ICONS and sidebar"
```

---

## Wave 2: Core Engine (2 parallel agents)

Depends on Wave 1 completion (database + standings_engine.py stub exist).

---

### Task 4: Category Win Probability Engine

**Agent focus:** Implement the Bayesian-updated Normal + Gaussian copula category win probability computation in `src/standings_engine.py`. This is the core algorithm. Test-driven.

**Files:**
- Modify: `src/standings_engine.py` (append to file created in Task 2)
- Append to: `tests/test_standings_engine.py`

**Key references:**
- `src/valuation.py:12-106` — `LeagueConfig` class, `config.inverse_stats`, `config.rate_stats`
- `src/standings_projection.py:15-28` — `WEEKLY_TAU` dict
- `scipy.stats.norm` — Normal CDF for P(win)
- `numpy.linalg.cholesky` — Cholesky decomposition for correlated sampling

- [ ] **Step 1: Write failing tests for category win probabilities**

Append to `tests/test_standings_engine.py`:

```python
import numpy as np
from src.valuation import LeagueConfig


class TestCategoryWinProbabilities:
    """Tests for per-category win probability computation."""

    def _make_pool(self, players: list[dict]) -> pd.DataFrame:
        """Build minimal player pool DataFrame for testing."""
        base = {
            "player_id": 0, "name": "Test", "team": "TST", "positions": "OF",
            "is_hitter": 1, "is_injured": 0,
            "pa": 600, "ab": 550, "h": 150, "r": 80, "hr": 25, "rbi": 85,
            "sb": 10, "avg": 0.273, "obp": 0.340, "bb": 50, "hbp": 5, "sf": 5,
            "ip": 0, "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0.0, "whip": 0.0,
            "er": 0, "bb_allowed": 0, "h_allowed": 0, "adp": 100,
        }
        rows = []
        for i, overrides in enumerate(players):
            row = {**base, "player_id": i + 1, **overrides}
            rows.append(row)
        return pd.DataFrame(rows)

    def test_equal_teams_near_50_percent(self):
        """Two identical teams should have ~50% win probability per category."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        # 5 identical hitters per team
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 0, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [hitter] * 5 + [pitcher] * 3 +  # user team (ids 1-8)
            [hitter] * 5 + [pitcher] * 3     # opp team (ids 9-16)
        )
        user_ids = list(range(1, 9))
        opp_ids = list(range(9, 17))

        result = compute_category_win_probabilities(user_ids, opp_ids, pool, config)

        assert "overall_win_pct" in result
        assert "categories" in result
        assert len(result["categories"]) == 12

        # Equal teams: overall should be near 50%
        assert 0.35 <= result["overall_win_pct"] <= 0.65

        # Each category should be near 50%
        for cat in result["categories"]:
            assert 0.30 <= cat["win_pct"] <= 0.70, f"{cat['name']} win_pct out of range: {cat['win_pct']}"

    def test_dominant_team_high_probability(self):
        """A much stronger team should have >80% win probability."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        strong_hitter = {"name": "Strong", "is_hitter": 1, "r": 120, "hr": 45,
                         "rbi": 130, "sb": 25, "avg": 0.310, "obp": 0.400,
                         "pa": 700, "ab": 600, "h": 186, "bb": 80, "hbp": 10, "sf": 5}
        weak_hitter = {"name": "Weak", "is_hitter": 1, "r": 40, "hr": 8,
                       "rbi": 35, "sb": 2, "avg": 0.220, "obp": 0.270,
                       "pa": 400, "ab": 360, "h": 79, "bb": 30, "hbp": 3, "sf": 3}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 5, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [strong_hitter] * 5 + [pitcher] * 3 +  # user (strong)
            [weak_hitter] * 5 + [pitcher] * 3       # opp (weak)
        )

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )
        assert result["overall_win_pct"] > 0.70

    def test_inverse_categories_correct_direction(self):
        """For ERA/WHIP/L, lower values should mean higher win probability."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        good_pitcher = {"name": "GP", "is_hitter": 0, "ip": 200, "w": 15, "l": 5,
                        "sv": 0, "k": 200, "era": 2.50, "whip": 1.00,
                        "er": 56, "bb_allowed": 40, "h_allowed": 160,
                        "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        bad_pitcher = {"name": "BP", "is_hitter": 0, "ip": 150, "w": 6, "l": 14,
                       "sv": 0, "k": 100, "era": 5.50, "whip": 1.60,
                       "er": 92, "bb_allowed": 70, "h_allowed": 170,
                       "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [hitter] * 5 + [good_pitcher] * 3 +
            [hitter] * 5 + [bad_pitcher] * 3
        )

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )
        era_cat = next(c for c in result["categories"] if c["name"] == "ERA")
        whip_cat = next(c for c in result["categories"] if c["name"] == "WHIP")

        # User has better (lower) ERA/WHIP → should have >50% win probability
        assert era_cat["win_pct"] > 0.55, f"ERA win_pct should be >0.55: {era_cat['win_pct']}"
        assert whip_cat["win_pct"] > 0.55, f"WHIP win_pct should be >0.55: {whip_cat['win_pct']}"

    def test_output_schema(self):
        """Verify the output dict has all required keys."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 5, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool([hitter] * 5 + [pitcher] * 3 + [hitter] * 5 + [pitcher] * 3)

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )

        # Top-level keys
        assert "overall_win_pct" in result
        assert "overall_tie_pct" in result
        assert "overall_loss_pct" in result
        assert "projected_score" in result
        assert "categories" in result

        # Probabilities sum to ~1.0
        total = result["overall_win_pct"] + result["overall_tie_pct"] + result["overall_loss_pct"]
        assert 0.99 <= total <= 1.01

        # Per-category schema
        for cat in result["categories"]:
            assert "name" in cat
            assert "user_proj" in cat
            assert "opp_proj" in cat
            assert "win_pct" in cat
            assert "is_inverse" in cat
            assert 0.0 <= cat["win_pct"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_standings_engine.py::TestCategoryWinProbabilities -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `compute_category_win_probabilities()` in `src/standings_engine.py`**

Append to `src/standings_engine.py`:

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.standings_projection import WEEKLY_TAU, INVERSE_CATS
from src.valuation import LeagueConfig

# ── Category correlation matrix ─────────────────────────────────────

CATEGORY_CORRELATIONS: dict[tuple[str, str], float] = {
    ("HR", "R"): 0.72,   ("HR", "RBI"): 0.68,   ("R", "RBI"): 0.65,
    ("AVG", "OBP"): 0.85, ("ERA", "WHIP"): 0.78,
    ("W", "K"): 0.45,     ("SB", "AVG"): -0.15,
    ("W", "L"): -0.30,    ("SV", "W"): -0.10,
}

ALL_CATEGORIES: list[str] = [
    "R", "HR", "RBI", "SB", "AVG", "OBP",
    "W", "L", "SV", "K", "ERA", "WHIP",
]


def _build_correlation_matrix(categories: list[str]) -> np.ndarray:
    """Build an NxN correlation matrix from pairwise correlations."""
    n = len(categories)
    corr = np.eye(n)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for (a, b), rho in CATEGORY_CORRELATIONS.items():
        if a in cat_idx and b in cat_idx:
            i, j = cat_idx[a], cat_idx[b]
            corr[i, j] = rho
            corr[j, i] = rho
    # Ensure positive semi-definite via nearest PSD
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    return corr


def _estimate_team_weekly_stats(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    weeks_remaining: int = 16,
) -> dict[str, float]:
    """Estimate a team's expected weekly production per category.

    Returns: {category: weekly_mean}
    """
    roster = player_pool[player_pool["player_id"].isin(roster_ids)].copy()
    hitters = roster[roster["is_hitter"] == 1]
    pitchers = roster[roster["is_hitter"] == 0]
    weeks = max(weeks_remaining, 1)

    stats: dict[str, float] = {}

    # Counting stats — sum and scale to weekly
    for cat in ("r", "hr", "rbi", "sb"):
        stats[cat.upper()] = float(hitters[cat].sum()) / weeks if not hitters.empty else 0.0
    for cat in ("w", "l", "sv", "k"):
        stats[cat.upper()] = float(pitchers[cat].sum()) / weeks if not pitchers.empty else 0.0

    # Rate stats — weighted aggregation
    if not hitters.empty:
        total_ab = float(hitters["ab"].sum())
        total_h = float(hitters["h"].sum())
        total_pa = float(hitters["pa"].sum())
        total_bb = float(hitters["bb"].sum())
        total_hbp = float(hitters.get("hbp", pd.Series([0])).sum())
        total_sf = float(hitters.get("sf", pd.Series([0])).sum())
        stats["AVG"] = total_h / total_ab if total_ab > 0 else 0.250
        denom_obp = total_ab + total_bb + total_hbp + total_sf
        stats["OBP"] = (total_h + total_bb + total_hbp) / denom_obp if denom_obp > 0 else 0.320
    else:
        stats["AVG"] = 0.250
        stats["OBP"] = 0.320

    if not pitchers.empty:
        total_ip = float(pitchers["ip"].sum())
        total_er = float(pitchers["er"].sum())
        total_bb_p = float(pitchers["bb_allowed"].sum())
        total_h_p = float(pitchers["h_allowed"].sum())
        stats["ERA"] = (total_er * 9 / total_ip) if total_ip > 0 else 4.50
        stats["WHIP"] = (total_bb_p + total_h_p) / total_ip if total_ip > 0 else 1.30
    else:
        stats["ERA"] = 4.50
        stats["WHIP"] = 1.30

    return stats


def compute_category_win_probabilities(
    user_roster_ids: list[int],
    opp_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    weeks_played: int = 0,
    weeks_remaining: int = 16,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute per-category P(user wins) for a head-to-head matchup.

    Uses Bayesian-updated Normal margins with Gaussian copula for
    correlated overall matchup probability.

    Returns: {
        overall_win_pct, overall_tie_pct, overall_loss_pct,
        projected_score: {W, L, T},
        categories: [{name, user_proj, opp_proj, win_pct, confidence, is_inverse}]
    }
    """
    categories = ALL_CATEGORIES
    inverse = INVERSE_CATS

    user_stats = _estimate_team_weekly_stats(user_roster_ids, player_pool, config, weeks_remaining)
    opp_stats = _estimate_team_weekly_stats(opp_roster_ids, player_pool, config, weeks_remaining)

    # Per-category independent probabilities
    cat_results = []
    marginal_probs = []

    for cat in categories:
        mu_user = user_stats.get(cat, 0.0)
        mu_opp = opp_stats.get(cat, 0.0)
        is_inv = cat in inverse

        # Variance: base tau, with Bayesian shrinkage if weeks played > 0
        tau = WEEKLY_TAU.get(cat, 1.0)
        if tau is None:
            tau = 1.0  # ERA tau can be None from dynamic loading
        base_var = float(tau) ** 2

        # Bayesian shrinkage: more weeks played → tighter variance
        if weeks_played >= 4:
            shrinkage = 1.0 / (1.0 + weeks_played / 10.0)
            var_cat = base_var * shrinkage
        else:
            var_cat = base_var

        sigma_diff = np.sqrt(2 * var_cat)  # Both teams have same variance

        if sigma_diff < 1e-10:
            p_win = 0.5
        elif is_inv:
            p_win = float(norm.cdf((mu_opp - mu_user) / sigma_diff))
        else:
            p_win = float(norm.cdf((mu_user - mu_opp) / sigma_diff))

        p_win = max(0.01, min(0.99, p_win))

        confidence = "high" if weeks_played >= 8 else "medium" if weeks_played >= 4 else "low"

        cat_results.append({
            "name": cat,
            "user_proj": round(mu_user, 3),
            "opp_proj": round(mu_opp, 3),
            "win_pct": round(p_win, 4),
            "confidence": confidence,
            "is_inverse": is_inv,
        })
        marginal_probs.append(p_win)

    # Correlated overall matchup probability via Gaussian copula
    corr_matrix = _build_correlation_matrix(categories)
    rng = np.random.default_rng(seed)

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(len(categories))

    z = rng.standard_normal((n_sims, len(categories)))
    correlated_z = z @ L.T

    # Convert to win/loss using marginal probabilities
    # P(win cat i) = marginal_probs[i]
    # Win if Phi(z_i) < marginal_prob[i]
    uniform = norm.cdf(correlated_z)
    wins_matrix = uniform < np.array(marginal_probs)  # (n_sims, n_cats)

    cats_won = wins_matrix.sum(axis=1)  # per sim
    n_cats = len(categories)
    half = n_cats / 2.0

    overall_win = float(np.mean(cats_won > half))
    overall_tie = float(np.mean(cats_won == half))
    overall_loss = float(np.mean(cats_won < half))

    # Projected score (expected category W-L-T)
    expected_w = float(np.mean(cats_won))
    expected_l = float(n_cats - expected_w)

    return {
        "overall_win_pct": round(overall_win, 4),
        "overall_tie_pct": round(overall_tie, 4),
        "overall_loss_pct": round(overall_loss, 4),
        "projected_score": {
            "W": round(expected_w, 1),
            "L": round(expected_l, 1),
            "T": round(n_cats - expected_w - expected_l, 1) if expected_w + expected_l < n_cats else 0.0,
        },
        "categories": cat_results,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_standings_engine.py::TestCategoryWinProbabilities -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/standings_engine.py tests/test_standings_engine.py
git commit -m "feat: implement category win probability engine with Gaussian copula"
```

---

### Task 5: Enhanced MC Season Simulation + Magic Numbers + Team Strength

**Agent focus:** Implement schedule-aware MC simulation, magic number computation, and wired-up team strength profiles. Test-driven.

**Files:**
- Modify: `src/standings_engine.py` (append simulation functions)
- Append to: `tests/test_standings_engine.py`

**Key references:**
- `src/standings_projection.py:106` — existing `simulate_season()` (keep intact, new function is separate)
- `src/power_rankings.py:149` — `compute_power_rankings()` (reuse)
- `src/injury_model.py:68` — `compute_health_score()` (wire up)

- [ ] **Step 1: Write failing tests for MC simulation and magic numbers**

Append to `tests/test_standings_engine.py`:

```python
class TestSimulateSeasonEnhanced:
    """Tests for schedule-aware MC season simulation."""

    def test_carries_forward_current_record(self):
        """Simulation should start from current W-L-T, not zero."""
        from src.standings_engine import simulate_season_enhanced

        current = {
            "Team A": {"W": 40, "L": 30, "T": 10},
            "Team B": {"W": 30, "L": 40, "T": 10},
        }
        schedule = {
            i: [("Team A", "Team B")]
            for i in range(15, 25)  # weeks 15-24 remaining
        }
        # Minimal team totals
        team_totals = {
            "Team A": {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                        "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20},
            "Team B": {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                        "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20},
        }

        result = simulate_season_enhanced(
            current_standings=current,
            team_weekly_totals=team_totals,
            full_schedule=schedule,
            current_week=15,
            n_sims=100,
            seed=42,
        )

        assert "projected_records" in result
        # Team A started with 40W, should end with >= 40W
        assert result["projected_records"]["Team A"]["W"] >= 40
        # Team B started with 30W, should end with >= 30W
        assert result["projected_records"]["Team B"]["W"] >= 30

    def test_uses_actual_schedule(self):
        """Simulation should use specific matchups, not round-robin."""
        from src.standings_engine import simulate_season_enhanced

        current = {
            "Team A": {"W": 0, "L": 0, "T": 0},
            "Team B": {"W": 0, "L": 0, "T": 0},
            "Team C": {"W": 0, "L": 0, "T": 0},
            "Team D": {"W": 0, "L": 0, "T": 0},
        }
        # A always plays B, C always plays D
        schedule = {
            i: [("Team A", "Team B"), ("Team C", "Team D")]
            for i in range(1, 11)
        }
        team_totals = {
            t: {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20}
            for t in current
        }

        result = simulate_season_enhanced(
            current_standings=current,
            team_weekly_totals=team_totals,
            full_schedule=schedule,
            current_week=1,
            n_sims=100,
            seed=42,
        )

        # All teams should have exactly 10 matchups (10 weeks)
        for team in current:
            rec = result["projected_records"][team]
            assert rec["W"] + rec["L"] + rec["T"] == 10

    def test_output_schema(self):
        from src.standings_engine import simulate_season_enhanced

        current = {"A": {"W": 5, "L": 5, "T": 0}, "B": {"W": 5, "L": 5, "T": 0}}
        schedule = {i: [("A", "B")] for i in range(6, 11)}
        totals = {
            t: {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20}
            for t in current
        }

        result = simulate_season_enhanced(
            current, totals, schedule, current_week=6, n_sims=50, seed=42
        )

        assert "projected_records" in result
        assert "playoff_probability" in result
        assert "confidence_intervals" in result
        assert "strength_of_schedule" in result


class TestMagicNumbers:
    """Tests for magic number computation."""

    def test_already_clinched(self):
        from src.standings_engine import compute_magic_numbers

        # 4-team league, team already has huge lead
        standings = {"A": 50, "B": 40, "C": 30, "D": 20}
        remaining = 2  # only 2 games left
        result = compute_magic_numbers(standings, remaining, playoff_spots=2)
        assert result["A"] == 0  # already clinched

    def test_eliminated_team(self):
        from src.standings_engine import compute_magic_numbers

        standings = {"A": 50, "B": 45, "C": 40, "D": 5}
        remaining = 2
        result = compute_magic_numbers(standings, remaining, playoff_spots=2)
        assert result["D"] is None  # can't catch up

    def test_mid_pack(self):
        from src.standings_engine import compute_magic_numbers

        standings = {"A": 10, "B": 9, "C": 8, "D": 7, "E": 6, "F": 5}
        remaining = 10
        result = compute_magic_numbers(standings, remaining, playoff_spots=4)
        # Mid-pack teams should have positive magic numbers
        assert result["C"] is not None
        assert result["C"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_standings_engine.py::TestSimulateSeasonEnhanced tests/test_standings_engine.py::TestMagicNumbers -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `simulate_season_enhanced()` in `src/standings_engine.py`**

Append to `src/standings_engine.py`:

```python
def simulate_season_enhanced(
    current_standings: dict[str, dict[str, int]],
    team_weekly_totals: dict[str, dict[str, float]],
    full_schedule: dict[int, list[tuple[str, str]]],
    current_week: int = 1,
    n_sims: int = 1000,
    seed: int = 42,
    momentum_data: dict[str, float] | None = None,
    playoff_spots: int = 4,
) -> dict:
    """Schedule-aware MC season simulation.

    Args:
        current_standings: {team: {W, L, T}} — current H2H record.
        team_weekly_totals: {team: {cat: weekly_mean}} — projected weekly averages.
        full_schedule: {week: [(team_a, team_b), ...]} — actual matchups.
        current_week: First week to simulate (prior weeks use actual results).
        n_sims: Monte Carlo iterations.
        seed: RNG seed.
        momentum_data: {team: float} — momentum multiplier [0.5, 2.0].
        playoff_spots: Number of playoff spots (default 4).

    Returns: {
        projected_records: {team: {W, L, T, win_pct}},
        playoff_probability: {team: float},
        confidence_intervals: {team: {p5_wins, p95_wins}},
        strength_of_schedule: {team: float},
    }
    """
    teams = list(current_standings.keys())
    n_teams = len(teams)
    categories = ALL_CATEGORIES
    n_cats = len(categories)
    half_cats = n_cats / 2.0

    # Build correlation matrix and Cholesky
    corr_matrix = _build_correlation_matrix(categories)
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(n_cats)

    rng = np.random.default_rng(seed)

    # Get tau values
    tau_vals = []
    for cat in categories:
        t = WEEKLY_TAU.get(cat, 1.0)
        tau_vals.append(float(t) if t is not None else 1.0)
    tau_arr = np.array(tau_vals)

    # Build team means matrix (n_teams x n_cats)
    team_means = np.zeros((n_teams, n_cats))
    for ti, team in enumerate(teams):
        totals = team_weekly_totals.get(team, {})
        for ci, cat in enumerate(categories):
            val = totals.get(cat, 0.0)
            # Apply momentum adjustment if available
            if momentum_data and team in momentum_data:
                mom = momentum_data[team]
                # Momentum adjusts mean by +/-5% max
                adjustment = 1.0 + 0.05 * (mom - 1.0)
                val *= adjustment
            team_means[ti, ci] = val

    # Determine remaining weeks
    all_weeks = sorted(full_schedule.keys())
    remaining_weeks = [w for w in all_weeks if w >= current_week]
    n_remaining = len(remaining_weeks)

    # Inverse category mask
    inverse_mask = np.array([cat in INVERSE_CATS for cat in categories])

    # Simulation
    win_counts = np.zeros((n_sims, n_teams))

    for sim in range(n_sims):
        # Start with current records
        sim_wins = np.array([current_standings[t]["W"] for t in teams], dtype=float)

        for week in remaining_weeks:
            matchups = full_schedule.get(week, [])
            for team_a, team_b in matchups:
                if team_a not in teams or team_b not in teams:
                    continue
                ti_a = teams.index(team_a)
                ti_b = teams.index(team_b)

                # Generate correlated noise for both teams
                z_a = rng.standard_normal(n_cats)
                z_b = rng.standard_normal(n_cats)
                noise_a = L @ z_a * tau_arr
                noise_b = L @ z_b * tau_arr

                vals_a = team_means[ti_a] + noise_a
                vals_b = team_means[ti_b] + noise_b

                # Count category wins
                a_wins_cat = np.where(
                    inverse_mask,
                    vals_a < vals_b,  # inverse: lower is better
                    vals_a > vals_b,  # normal: higher is better
                )
                cats_won_a = int(a_wins_cat.sum())
                cats_won_b = n_cats - cats_won_a  # ties counted as losses for simplicity

                if cats_won_a > half_cats:
                    sim_wins[ti_a] += 1
                elif cats_won_b > half_cats:
                    sim_wins[ti_b] += 1
                # else: tie — neither gets a win

        win_counts[sim] = sim_wins

    # Aggregate results
    mean_wins = win_counts.mean(axis=0)
    p5_wins = np.percentile(win_counts, 5, axis=0)
    p95_wins = np.percentile(win_counts, 95, axis=0)

    # Playoff probability: top `playoff_spots` by wins
    playoff_counts = np.zeros(n_teams)
    for sim in range(n_sims):
        ranked = np.argsort(-win_counts[sim])
        for spot in range(min(playoff_spots, n_teams)):
            playoff_counts[ranked[spot]] += 1
    playoff_probs = playoff_counts / n_sims

    # Strength of schedule: avg opponent quality
    total_games = current_standings[teams[0]]["W"] + current_standings[teams[0]]["L"] + current_standings[teams[0]]["T"]
    total_possible = total_games + n_remaining
    sos = {}
    for ti, team in enumerate(teams):
        opp_quality = []
        for week in remaining_weeks:
            for ta, tb in full_schedule.get(week, []):
                if ta == team:
                    oi = teams.index(tb)
                    opp_quality.append(mean_wins[oi] / max(total_possible, 1))
                elif tb == team:
                    oi = teams.index(ta)
                    opp_quality.append(mean_wins[oi] / max(total_possible, 1))
        sos[team] = round(float(np.mean(opp_quality)) if opp_quality else 0.5, 3)

    projected_records = {}
    confidence_intervals = {}
    playoff_probability = {}

    for ti, team in enumerate(teams):
        total_w = float(mean_wins[ti])
        total_matchups = total_games + n_remaining
        total_l = total_matchups - total_w
        projected_records[team] = {
            "W": round(total_w, 1),
            "L": round(total_l, 1),
            "T": 0,  # Simplified — ties rare enough
            "win_pct": round(total_w / max(total_matchups, 1), 3),
        }
        confidence_intervals[team] = {
            "p5_wins": round(float(p5_wins[ti]), 1),
            "p95_wins": round(float(p95_wins[ti]), 1),
        }
        playoff_probability[team] = round(float(playoff_probs[ti]), 4)

    return {
        "projected_records": projected_records,
        "playoff_probability": playoff_probability,
        "confidence_intervals": confidence_intervals,
        "strength_of_schedule": sos,
    }


def compute_magic_numbers(
    current_wins: dict[str, int | float],
    remaining_matchups: int,
    playoff_spots: int = 4,
) -> dict[str, int | None]:
    """Compute magic number (wins to clinch playoff spot) per team.

    Classic formula adapted for H2H: magic = remaining + 1 - (my_wins - Nth_place_wins)
    where N = playoff_spots + 1 (the first team outside playoffs).

    Returns: {team: magic_number} where 0 = clinched, None = eliminated.
    """
    teams_sorted = sorted(current_wins.keys(), key=lambda t: -current_wins[t])
    results = {}

    for team in teams_sorted:
        my_wins = current_wins[team]

        # The team we need to stay ahead of: (playoff_spots + 1)th place
        # or the best non-playoff team
        chase_teams = [t for t in teams_sorted if t != team]
        # Sort others by wins descending
        chase_teams.sort(key=lambda t: -current_wins[t])

        if len(chase_teams) < playoff_spots:
            results[team] = 0  # Auto-clinch (fewer teams than spots)
            continue

        # Nth place team's max possible wins
        nth_team = chase_teams[playoff_spots - 1]  # (playoff_spots)th OTHER team
        nth_wins = current_wins[nth_team]
        nth_max = nth_wins + remaining_matchups

        # Magic number: wins I need so that even if Nth team wins everything, I'm still ahead
        magic = max(0, nth_max - my_wins + 1)

        if magic > remaining_matchups:
            results[team] = None  # Eliminated — can't get enough wins
        elif magic <= 0:
            results[team] = 0  # Clinched
        else:
            results[team] = int(magic)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_standings_engine.py::TestSimulateSeasonEnhanced tests/test_standings_engine.py::TestMagicNumbers -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/standings_engine.py tests/test_standings_engine.py
git commit -m "feat: implement schedule-aware MC simulation and magic number computation"
```

---

### Task 5b: Team Strength Profiles (within Task 5 agent)

After completing magic numbers, also implement `compute_team_strength_profiles()` in `src/standings_engine.py`:

```python
def compute_team_strength_profiles(
    team_weekly_totals: dict[str, dict[str, float]],
    full_schedule: dict[int, list[tuple[str, str]]] | None = None,
    current_week: int = 1,
    momentum_data: dict[str, float] | None = None,
) -> list[dict]:
    """Compute 5-factor team strength profiles using existing power_rankings.py.

    Wires up all 5 factors:
    - roster_quality (40%): Z-score of team's avg category rank
    - category_balance (25%): Fraction of cats beating median
    - schedule_strength (15%): Avg opponent roster_quality for remaining weeks
    - injury_exposure (10%): Placeholder (requires live health data from caller)
    - momentum (10%): From momentum_data if provided
    """
    from src.power_rankings import compute_power_rankings, bootstrap_confidence_interval
    from src.standings_projection import INVERSE_CATS

    teams = list(team_weekly_totals.keys())
    n_teams = len(teams)
    categories = ALL_CATEGORIES

    # Compute z-scores per category per team
    cat_values = {cat: [] for cat in categories}
    for team in teams:
        for cat in categories:
            cat_values[cat].append(team_weekly_totals[team].get(cat, 0.0))

    # Roster quality: normalized z-score of category ranks
    roster_quality = {}
    cat_balance = {}
    for ti, team in enumerate(teams):
        ranks = []
        above_median = 0
        for cat in categories:
            vals = cat_values[cat]
            team_val = vals[ti]
            if cat in INVERSE_CATS:
                rank = sum(1 for v in vals if v < team_val) + 1
                median_val = sorted(vals)[n_teams // 2]
                if team_val <= median_val:
                    above_median += 1
            else:
                rank = sum(1 for v in vals if v > team_val) + 1
                median_val = sorted(vals)[n_teams // 2]
                if team_val >= median_val:
                    above_median += 1
            ranks.append(rank)

        avg_rank = sum(ranks) / len(ranks)
        # Normalize: best possible = 1.0, worst = 0.0
        rq = max(0.0, min(1.0, 1.0 - (avg_rank - 1) / max(n_teams - 1, 1)))
        roster_quality[team] = rq
        cat_balance[team] = max(0.1, above_median / len(categories))

    # Schedule strength: avg opponent roster_quality for remaining weeks
    sos = {}
    remaining_weeks = [w for w in sorted(full_schedule.keys()) if w >= current_week] if full_schedule else []
    for team in teams:
        opp_qualities = []
        for week in remaining_weeks:
            for ta, tb in full_schedule.get(week, []):
                if ta == team and tb in roster_quality:
                    opp_qualities.append(roster_quality[tb])
                elif tb == team and ta in roster_quality:
                    opp_qualities.append(roster_quality[ta])
        sos[team] = float(np.mean(opp_qualities)) if opp_qualities else 0.5

    # Build power ranking input
    team_data = []
    for team in teams:
        entry = {
            "team_name": team,
            "roster_quality": roster_quality[team],
            "category_balance": cat_balance[team],
            "schedule_strength": sos[team],
            "injury_exposure": None,  # caller can provide via health data
            "momentum": momentum_data.get(team) if momentum_data else None,
        }
        team_data.append(entry)

    pr_df = compute_power_rankings(team_data)

    # Add bootstrap CIs
    results = []
    for _, row in pr_df.iterrows():
        p5, p95 = bootstrap_confidence_interval(row["power_rating"])
        results.append({
            "team_name": row["team_name"],
            "power_rating": row["power_rating"],
            "roster_quality": row["roster_quality"],
            "category_balance": row["category_balance"],
            "schedule_strength": row.get("schedule_strength"),
            "injury_exposure": row.get("injury_exposure"),
            "momentum": row.get("momentum"),
            "rank": row["rank"],
            "ci_low": p5,
            "ci_high": p95,
        })
    return results
```

Add a test for this in `tests/test_standings_engine.py`:

```python
class TestTeamStrengthProfiles:
    def test_all_teams_ranked(self):
        from src.standings_engine import compute_team_strength_profiles

        totals = {
            "A": {"R": 5, "HR": 3, "RBI": 5, "SB": 2, "AVG": 0.280, "OBP": 0.350,
                  "W": 2, "L": 1, "SV": 1, "K": 10, "ERA": 3.00, "WHIP": 1.10},
            "B": {"R": 4, "HR": 2, "RBI": 4, "SB": 1, "AVG": 0.260, "OBP": 0.330,
                  "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 4.00, "WHIP": 1.30},
        }
        schedule = {1: [("A", "B")]}
        result = compute_team_strength_profiles(totals, schedule, current_week=1)
        assert len(result) == 2
        assert result[0]["power_rating"] > result[1]["power_rating"]  # A is stronger
        assert all("ci_low" in r and "ci_high" in r for r in result)
```

---

## Wave 3: Pages (2 parallel agents)

Depends on Wave 2 completion (engine functions exist and are tested).

---

### Task 6: League Standings Page

**Agent focus:** Create `pages/8_League_Standings.py` with two tabs: Current Standings (live Yahoo W-L-T + category grid) and Season Projections (enhanced MC + scenario explorer). Delete `pages/8_Standings.py`.

**Files:**
- Create: `pages/8_League_Standings.py`
- Delete: `pages/8_Standings.py`

**Key references:**
- `src/ui_shared.py` — all render_* functions (see Task 3 for line numbers)
- `src/standings_engine.py` — `compute_category_win_probabilities()`, `simulate_season_enhanced()`, `compute_magic_numbers()`
- `src/database.py` — `load_league_records()`, `load_league_standings()`
- `src/yahoo_data_service.py` — `get_yahoo_data_service()`, `get_standings()`, `get_matchup()`, `get_full_league_schedule()`

**Implementation is too large to include every line of code here.** The agent should:

- [ ] **Step 1: Read the existing `pages/8_Standings.py` to understand current patterns**

Read: `pages/8_Standings.py`
Note: session_state keys, import patterns, fallback logic

- [ ] **Step 2: Read the design spec for page layout requirements**

Read: `docs/superpowers/specs/2026-04-05-league-standings-matchup-planner-design.md` — Sections 2.1 through 2.5

- [ ] **Step 3: Create `pages/8_League_Standings.py` with initialization**

Follow existing page pattern:
```python
"""League Standings — Live H2H standings + season projections."""
from __future__ import annotations
import logging
import pandas as pd
import streamlit as st

from src.database import init_db, load_league_records, load_league_standings
from src.ui_shared import (
    THEME, inject_custom_css, page_timer_footer, page_timer_start,
    build_compact_table_html, render_compact_table, render_context_card,
    render_context_columns, render_data_freshness_card, render_page_layout,
)
from src.yahoo_data_service import get_yahoo_data_service
from src.valuation import LeagueConfig

# Guard imports
try:
    from src.standings_engine import (
        compute_category_win_probabilities, simulate_season_enhanced,
        compute_magic_numbers, find_user_opponent,
        _estimate_team_weekly_stats,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)
T = THEME

st.set_page_config(page_title="Heater | League Standings", page_icon="", layout="wide", initial_sidebar_state="collapsed")
init_db()
inject_custom_css()
page_timer_start()
```

- [ ] **Step 4: Implement recommendation banner**

Use `yds.get_matchup()` to build the matchup preview teaser:
```python
# Build banner teaser from live matchup
yds = get_yahoo_data_service()
matchup = yds.get_matchup()
if matchup:
    opp = matchup.get("opp_name", "opponent")
    wins = matchup.get("wins", 0)
    losses = matchup.get("losses", 0)
    ties = 12 - wins - losses
    teaser = f"This week vs {opp}: {'leading' if wins > losses else 'trailing'} {wins}-{losses}-{ties} in categories."
else:
    teaser = "Connect your Yahoo league to see live matchup analysis."

render_page_layout("LEAGUE STANDINGS", banner_teaser=teaser, banner_icon="league_standings")
```

- [ ] **Step 5: Implement context panel (left column)**

Four context cards:
1. YOUR POSITION — rank, W-L-T, GB from 1st, GB from 4th
2. THIS WEEK — opponent name, live category score
3. PLAYOFF ODDS — probability + magic number (from cached simulation)
4. DATA FRESHNESS — `render_data_freshness_card()`

- [ ] **Step 6: Implement Tab 1: Current Standings**

Two sections:
1. **H2H Record Table**: Load from `load_league_records()`. Columns: Rank, Team, W, L, T, Win%, GB, Streak. User team highlighted. Dashed line between rank 4 and 5. Use `build_compact_table_html()`.
2. **Category Standings Grid**: Load from `load_league_standings()`, pivot to wide. 12 teams x 12 categories. Cell content = rank with color-coded badge (green 1-4, blue 5-8, red 9-12). Use `build_compact_table_html()` with custom highlight_cols.

- [ ] **Step 7: Implement Tab 2: Season Projections**

1. **Auto-run simulation**: On page load, check `session_state["standings_sim_result"]`. If stale or missing, compute team weekly totals from standings and run `simulate_season_enhanced()`.
2. **Projected Final Standings Table**: Show projected W, L, Win%, Playoff%, Magic#, SOS from simulation results.
3. **Scenario Explorer**: Three `st.number_input` fields (W, L, T constrained to sum 12). Button to re-run simulation with modified current-week result. Show delta in playoff odds.

- [ ] **Step 8: Delete old `pages/8_Standings.py`**

```bash
git rm pages/8_Standings.py
```

- [ ] **Step 9: Test the page manually**

Run: `streamlit run app.py`
Navigate to League Standings page. Verify:
- Both tabs render without errors
- Current Standings shows W-L-T table + category grid
- Season Projections auto-runs simulation
- Context panel shows position, matchup, odds

- [ ] **Step 10: Run full test suite**

Run: `python -m pytest -x -q`
Expected: All tests pass (old Standings page tests may need updating if they import from page 8)

- [ ] **Step 11: Commit**

```bash
git add pages/8_League_Standings.py
git commit -m "feat: create League Standings page with live Yahoo standings and MC projections"
```

---

### Task 7: Matchup Planner Redesign

**Agent focus:** Add Category Probabilities tab and week navigator to `pages/11_Matchup_Planner.py`. Preserve all existing tabs.

**Files:**
- Modify: `pages/11_Matchup_Planner.py`

**Key references:**
- `src/standings_engine.py` — `compute_category_win_probabilities()`, `find_user_opponent()`, `parse_week_category_results()`
- Design spec Section 3 (Matchup Planner Redesign)
- Current file structure: lines 1-477 (read in exploration phase)

- [ ] **Step 1: Read the current `pages/11_Matchup_Planner.py` in full**

Understand imports, session state keys, tab structure, and rendering patterns.

- [ ] **Step 2: Read the design spec for Matchup Planner requirements**

Read: `docs/superpowers/specs/2026-04-05-league-standings-matchup-planner-design.md` — Sections 3.1 through 3.6

- [ ] **Step 3: Add new imports and engine guard**

Add to imports section:
```python
try:
    from src.standings_engine import (
        compute_category_win_probabilities, find_user_opponent,
        parse_week_category_results, ALL_CATEGORIES,
    )
    from src.valuation import LeagueConfig
    WIN_PROB_AVAILABLE = True
except ImportError:
    WIN_PROB_AVAILABLE = False
```

- [ ] **Step 4: Add week navigator to context panel**

In the context panel (`with ctx:`), add before the existing filters:

```python
    # ── Week navigator ───────────────────────────────────────────
    yds = get_yahoo_data_service()
    matchup_data = yds.get_matchup()
    current_week = matchup_data.get("week", 1) if matchup_data else 1

    if "matchup_week" not in st.session_state:
        st.session_state["matchup_week"] = current_week

    nav_cols = st.columns([1, 2, 1])
    with nav_cols[0]:
        if st.button("◀", key="week_prev", help="Previous week"):
            st.session_state["matchup_week"] = max(1, st.session_state["matchup_week"] - 1)
            st.rerun()
    with nav_cols[1]:
        selected_week = st.session_state["matchup_week"]
        week_label = "Current" if selected_week == current_week else (
            "Past" if selected_week < current_week else "Future"
        )
        st.markdown(
            f'<div style="text-align:center;font-size:20px;font-weight:800;color:{T["primary"]};">W{selected_week}</div>'
            f'<div style="text-align:center;font-size:11px;color:{T["tx2"]};">{week_label}</div>',
            unsafe_allow_html=True,
        )
    with nav_cols[2]:
        if st.button("▶", key="week_next", help="Next week"):
            st.session_state["matchup_week"] = min(24, st.session_state["matchup_week"] + 1)
            st.rerun()

    # Find opponent for selected week
    full_schedule = yds.get_full_league_schedule() if yds.is_connected() else {}
    user_team = matchup_data.get("user_name", "Team Hickey") if matchup_data else "Team Hickey"
    week_opponent = find_user_opponent(full_schedule, selected_week, user_team) if full_schedule else None

    if week_opponent:
        render_context_card("Opponent", f'<div style="font-size:13px;font-weight:700;color:{T["tx"]};">{week_opponent}</div>')
```

- [ ] **Step 5: Add Category Probabilities tab as first tab**

Replace the existing `st.tabs()` call with:
```python
    tabs = st.tabs(["Category Probabilities", "Player Matchups", "Per-Game Detail", "Hitters Only", "Pitchers Only"])
    tab_probs, tab_summary, tab_detail, tab_hitters, tab_pitchers = tabs
```

Implement `tab_probs`:
```python
    with tab_probs:
        if not WIN_PROB_AVAILABLE:
            st.warning("Category probability engine not available.")
        elif not rosters.empty and week_opponent:
            # Get rosters for user and opponent
            user_roster = rosters[rosters["team_name"] == user_team]
            opp_roster = rosters[rosters["team_name"] == week_opponent]

            if not user_roster.empty and not opp_roster.empty:
                config = LeagueConfig()
                user_ids = user_roster["player_id"].tolist()
                opp_ids = opp_roster["player_id"].tolist()

                # For past weeks: show actual results
                selected_week = st.session_state["matchup_week"]
                if selected_week < current_week:
                    st.subheader(f"Week {selected_week} Results vs {week_opponent}")
                    st.info("Past week results — showing actual category outcomes.")
                    # TODO: Fetch actual results from Yahoo scoreboard
                else:
                    # Current or future week: show probabilities
                    probs = compute_category_win_probabilities(
                        user_ids, opp_ids, pool, config,
                        weeks_played=max(0, current_week - 1),
                    )

                    # Overall win probability in context panel
                    # (rendered via context card above)

                    st.subheader(f"Week {selected_week} vs {week_opponent}")

                    # Per-category probability bars
                    sorted_cats = sorted(probs["categories"], key=lambda c: -c["win_pct"])
                    for cat in sorted_cats:
                        pct = cat["win_pct"] * 100
                        # Color: green >= 70, blue 55-70, orange 45-55, red < 45
                        if pct >= 70:
                            bar_color = T["green"]
                        elif pct >= 55:
                            bar_color = T["sky"]
                        elif pct >= 45:
                            bar_color = T["hot"]
                        else:
                            bar_color = T["primary"]

                        is_hit = cat["name"] in config.hitting_categories
                        name_color = T["sky"] if is_hit else T["primary"]

                        bar_html = (
                            f'<div style="display:flex;align-items:center;gap:12px;'
                            f'padding:6px 10px;background:{T["card"]};border:1px solid {T["border"]};'
                            f'border-radius:8px;margin-bottom:4px;">'
                            f'<div style="width:45px;font-size:13px;font-weight:700;'
                            f'color:{name_color};font-family:Geist Mono,monospace;">{cat["name"]}</div>'
                            f'<div style="flex:1;height:22px;border-radius:4px;'
                            f'background:{T["bg"]};overflow:hidden;position:relative;">'
                            f'<div style="width:{pct}%;height:100%;background:{bar_color};'
                            f'border-radius:4px;"></div>'
                            f'<div style="position:absolute;right:8px;top:50%;transform:translateY(-50%);'
                            f'font-size:11px;font-weight:700;color:{T["tx"]};">{pct:.0f}%</div>'
                            f'</div>'
                            f'<div style="width:110px;text-align:right;font-size:11px;'
                            f'font-family:Geist Mono,monospace;color:{T["tx2"]};">'
                            f'<span style="color:{T["green"]};font-weight:700;">{cat["user_proj"]}</span>'
                            f' vs '
                            f'<span style="color:{T["primary"]};">{cat["opp_proj"]}</span>'
                            f'</div>'
                            f'</div>'
                        )
                        st.markdown(bar_html, unsafe_allow_html=True)
            else:
                st.info(f"Roster data not available for {week_opponent}.")
        else:
            st.info("Connect your Yahoo league and load roster data to see category probabilities.")
```

- [ ] **Step 6: Move existing tab content into new tab variables**

Move the content that was in `tab_summary` to the new `tab_summary` variable. Same for `tab_detail`, `tab_hitters`, `tab_pitchers`. No functional changes to these tabs — just re-assigned to new variable names.

- [ ] **Step 7: Add overall win probability to context panel**

After the week navigator, add a context card showing the overall win/tie/loss probability:
```python
    # Show win probability in context if computed
    if WIN_PROB_AVAILABLE and "matchup_probs" in st.session_state:
        probs = st.session_state["matchup_probs"]
        win_pct = probs["overall_win_pct"] * 100
        render_context_card(
            "Win Probability",
            f'<div style="font-size:28px;font-weight:800;color:{T["green"] if win_pct > 50 else T["primary"]};">'
            f'{win_pct:.0f}%</div>'
            f'<div style="font-size:11px;color:{T["tx2"]};">Win: {win_pct:.0f}% | '
            f'Tie: {probs["overall_tie_pct"]*100:.0f}% | '
            f'Loss: {probs["overall_loss_pct"]*100:.0f}%</div>',
        )
```

- [ ] **Step 8: Update recommendation banner**

Replace the existing banner_teaser with matchup probability info:
```python
render_page_layout(
    "MATCHUP PLANNER",
    banner_teaser=f"Week {selected_week} vs {week_opponent}: {win_pct:.0f}% chance to win"
    if WIN_PROB_AVAILABLE and week_opponent else
    "Weekly matchup ratings with per-game analysis",
    banner_icon="calendar",
)
```

- [ ] **Step 9: Test manually**

Run: `streamlit run app.py`
Navigate to Matchup Planner. Verify:
- Week navigator arrows work
- Category Probabilities tab renders bars
- Existing tabs still work
- Context panel shows win probability

- [ ] **Step 10: Run full test suite**

Run: `python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 11: Commit**

```bash
git add pages/11_Matchup_Planner.py
git commit -m "feat: add category win probabilities and week navigator to Matchup Planner"
```

---

## Wave 4: Integration & Cleanup (1 agent)

Depends on all prior waves.

---

### Task 8: Integration Tests + Ruff Lint + Final Cleanup

**Agent focus:** Run full test suite, fix any regressions, add integration tests, run ruff lint/format, final commit.

**Files:**
- Create: `tests/test_league_standings_integration.py`
- Potentially fix: any files with lint issues

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -x -q`
Fix any failures.

- [ ] **Step 2: Write integration tests**

Create `tests/test_league_standings_integration.py`:

```python
"""Integration tests for League Standings + Matchup Planner."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.database import init_db
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db()


class TestEndToEndSimulation:
    """Test the full simulation pipeline with realistic data."""

    def test_full_pipeline(self):
        """Run category probs + season sim + magic numbers end-to-end."""
        from src.standings_engine import (
            compute_category_win_probabilities,
            simulate_season_enhanced,
            compute_magic_numbers,
        )

        config = LeagueConfig()
        n_teams = 4

        # Build minimal player pool
        rows = []
        pid = 1
        for team_idx in range(n_teams):
            for _ in range(5):  # 5 hitters per team
                rows.append({
                    "player_id": pid, "name": f"H{pid}", "team": f"T{team_idx}",
                    "positions": "OF", "is_hitter": 1, "is_injured": 0,
                    "pa": 600, "ab": 550, "h": 140 + team_idx * 10,
                    "r": 70 + team_idx * 5, "hr": 20 + team_idx * 3,
                    "rbi": 75 + team_idx * 5, "sb": 8 + team_idx * 2,
                    "avg": 0.255 + team_idx * 0.01, "obp": 0.320 + team_idx * 0.01,
                    "bb": 50, "hbp": 5, "sf": 5,
                    "ip": 0, "w": 0, "l": 0, "sv": 0, "k": 0,
                    "era": 0, "whip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
                    "adp": 50 + pid,
                })
                pid += 1
            for _ in range(3):  # 3 pitchers per team
                rows.append({
                    "player_id": pid, "name": f"P{pid}", "team": f"T{team_idx}",
                    "positions": "SP", "is_hitter": 0, "is_injured": 0,
                    "pa": 0, "ab": 0, "h": 0, "r": 0, "hr": 0, "rbi": 0,
                    "sb": 0, "avg": 0, "obp": 0, "bb": 0, "hbp": 0, "sf": 0,
                    "ip": 180, "w": 10 + team_idx, "l": 10 - team_idx,
                    "sv": 2 * team_idx, "k": 150 + team_idx * 10,
                    "era": 4.0 - team_idx * 0.3, "whip": 1.3 - team_idx * 0.05,
                    "er": 80 - team_idx * 5, "bb_allowed": 55, "h_allowed": 170 - team_idx * 5,
                    "adp": 50 + pid,
                })
                pid += 1

        pool = pd.DataFrame(rows)
        teams = [f"Team {chr(65+i)}" for i in range(n_teams)]
        roster_ids = {}
        for i, team in enumerate(teams):
            start = i * 8 + 1
            roster_ids[team] = list(range(start, start + 8))

        # Category win probabilities
        probs = compute_category_win_probabilities(
            roster_ids[teams[0]], roster_ids[teams[3]], pool, config
        )
        assert 0.0 <= probs["overall_win_pct"] <= 1.0
        assert len(probs["categories"]) == 12

        # Season simulation
        current = {t: {"W": 5, "L": 5, "T": 0} for t in teams}
        schedule = {
            w: [(teams[0], teams[1]), (teams[2], teams[3])]
            for w in range(6, 15)
        }

        from src.standings_engine import _estimate_team_weekly_stats
        team_totals = {}
        for team in teams:
            team_totals[team] = _estimate_team_weekly_stats(
                roster_ids[team], pool, config, weeks_remaining=9
            )

        sim = simulate_season_enhanced(
            current, team_totals, schedule,
            current_week=6, n_sims=50, seed=42
        )
        assert set(sim.keys()) == {"projected_records", "playoff_probability", "confidence_intervals", "strength_of_schedule"}

        # Magic numbers
        wins = {t: int(sim["projected_records"][t]["W"]) for t in teams}
        magic = compute_magic_numbers(wins, remaining_matchups=5, playoff_spots=2)
        assert all(t in magic for t in teams)
```

- [ ] **Step 3: Run integration tests**

Run: `python -m pytest tests/test_league_standings_integration.py -v`
Expected: All pass

- [ ] **Step 4: Run ruff lint and format**

```bash
python -m ruff check . --fix
python -m ruff format .
```

Fix any remaining issues.

- [ ] **Step 5: Run full test suite one final time**

Run: `python -m pytest -x -q`
Expected: All 2300+ tests pass (plus new tests)

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: League Standings page + Matchup Planner redesign — complete implementation"
```

---

## Execution Summary

| Wave | Tasks | Agents | Dependency |
|------|-------|--------|-----------|
| **Wave 1** | Task 1 (DB), Task 2 (Yahoo), Task 3 (UI) | 3 parallel | None |
| **Wave 2** | Task 4 (Win Probs), Task 5 (MC Sim) | 2 parallel | Wave 1 |
| **Wave 3** | Task 6 (Standings Page), Task 7 (Matchup Page) | 2 parallel | Wave 2 |
| **Wave 4** | Task 8 (Integration + Cleanup) | 1 agent | Wave 3 |

**Total: 8 tasks across 4 waves, max 3 agents running simultaneously.**
