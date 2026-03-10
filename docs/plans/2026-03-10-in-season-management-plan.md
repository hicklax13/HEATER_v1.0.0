# In-Season Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the draft-only Fantasy Baseball Tool into a full in-season manager with live stats, trade analyzer, player comparison, and free agent analysis — all without Yahoo Sports API.

**Architecture:** 4 new Streamlit pages (`pages/`), 3 new backend modules (`src/live_stats.py`, `src/in_season.py`, `src/league_manager.py`), 1 shared UI module (`src/ui_shared.py`), 6 new database tables. Data from MLB Stats API + pybaseball, league-specific data via CSV upload. Trade analysis uses MC simulation + Live SGP reusing existing valuation engine.

**Tech Stack:** Python 3.12, Streamlit, SQLite, pandas, NumPy, SciPy, Plotly, MLB-StatsAPI, pybaseball

---

## Phase 1: Foundation

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add `MLB-StatsAPI` and `pybaseball` to `requirements.txt`:

```
streamlit>=1.37.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.18.0
MLB-StatsAPI>=1.7
pybaseball>=2.3
# Optional: Yahoo API integration
# yfpy>=15.0.0
# Optional: Excel cheat sheet export
# openpyxl>=3.1.0
```

**Step 2: Install new dependencies**

Run: `pip install MLB-StatsAPI pybaseball`
Expected: Both packages install successfully.

**Step 3: Verify imports work**

Run: `python -c "import statsapi; import pybaseball; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add MLB-StatsAPI and pybaseball dependencies"
```

---

### Task 2: Add 6 new database tables to init_db()

**Files:**
- Modify: `src/database.py:18-71` (extend `init_db()`)
- Test: `tests/test_database_schema.py`

**Step 1: Write the failing test**

Create `tests/test_database_schema.py`:

```python
"""Test that all database tables are created correctly."""

import sqlite3
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db, DB_PATH


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _get_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


import pytest


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    yield tmp.name
    db_mod.DB_PATH = original
    os.unlink(tmp.name)


def test_new_tables_exist(temp_db):
    """All 6 new tables should be created by init_db()."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    for table in [
        "season_stats",
        "ros_projections",
        "league_rosters",
        "league_standings",
        "park_factors",
        "refresh_log",
    ]:
        assert _table_exists(cursor, table), f"Table {table} not found"

    conn.close()


def test_season_stats_columns(temp_db):
    """season_stats table should have the expected columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "season_stats")
    expected = {
        "player_id", "season", "pa", "ab", "h", "r", "hr", "rbi",
        "sb", "avg", "ip", "w", "sv", "k", "era", "whip",
        "er", "bb_allowed", "h_allowed", "games_played", "last_updated",
    }
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()


def test_league_rosters_columns(temp_db):
    """league_rosters table should have the expected columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "league_rosters")
    expected = {"id", "team_name", "team_index", "player_id", "roster_slot", "is_user_team"}
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()


def test_refresh_log_columns(temp_db):
    """refresh_log table should have source, last_refresh, status columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "refresh_log")
    expected = {"source", "last_refresh", "status"}
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_database_schema.py -v`
Expected: FAIL — tables don't exist yet.

**Step 3: Add the 6 new tables to init_db()**

In `src/database.py`, find the `init_db()` function (line 18). After the existing `CREATE INDEX` statements and `conn.commit()` (line 70), add a second `conn = get_connection()` block (or merge into the existing executescript). Add before the `conn.close()`:

```python
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS season_stats (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL DEFAULT 2026,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
            k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (player_id, season),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS ros_projections (
            player_id INTEGER NOT NULL,
            system TEXT NOT NULL,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
            k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (player_id, system),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS league_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_index INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            roster_slot TEXT,
            is_user_team INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS league_standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            category TEXT NOT NULL,
            total REAL DEFAULT 0,
            rank INTEGER,
            points REAL
        );

        CREATE TABLE IF NOT EXISTS park_factors (
            team_code TEXT PRIMARY KEY,
            factor_hitting REAL DEFAULT 1.0,
            factor_pitching REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT DEFAULT 'success'
        );

        CREATE INDEX IF NOT EXISTS idx_season_stats_player ON season_stats(player_id);
        CREATE INDEX IF NOT EXISTS idx_ros_proj_player ON ros_projections(player_id);
        CREATE INDEX IF NOT EXISTS idx_league_rosters_team ON league_rosters(team_name);
    """)
    conn.commit()
    conn.close()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_database_schema.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/database.py tests/test_database_schema.py
git commit -m "feat: add 6 new database tables for in-season management"
```

---

### Task 3: Add database query functions for new tables

**Files:**
- Modify: `src/database.py` (add functions after `load_player_pool()`, after line 429)
- Test: `tests/test_database_queries.py`

**Step 1: Write the failing test**

Create `tests/test_database_queries.py`:

```python
"""Test database query functions for in-season tables."""

import sqlite3
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import (
    init_db,
    upsert_season_stats,
    upsert_ros_projection,
    upsert_league_roster_entry,
    upsert_league_standing,
    update_refresh_log,
    get_refresh_status,
    load_season_stats,
    load_league_rosters,
    clear_league_rosters,
)

import pytest


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    os.unlink(tmp.name)


def _insert_player(db_path, name="Test Player", team="NYY", positions="SS", is_hitter=1):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        (name, team, positions, is_hitter),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return pid


def test_upsert_season_stats(temp_db):
    pid = _insert_player(temp_db)
    upsert_season_stats(pid, {"r": 50, "hr": 20, "rbi": 60, "ab": 400, "h": 110})
    df = load_season_stats()
    assert len(df) == 1
    assert df.iloc[0]["r"] == 50
    assert df.iloc[0]["hr"] == 20


def test_upsert_season_stats_updates(temp_db):
    pid = _insert_player(temp_db)
    upsert_season_stats(pid, {"r": 50, "hr": 20})
    upsert_season_stats(pid, {"r": 60, "hr": 25})
    df = load_season_stats()
    assert len(df) == 1
    assert df.iloc[0]["r"] == 60


def test_upsert_ros_projection(temp_db):
    pid = _insert_player(temp_db)
    upsert_ros_projection(pid, "steamer_ros", {"r": 30, "hr": 10})
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT * FROM ros_projections WHERE player_id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row is not None


def test_league_roster_operations(temp_db):
    pid = _insert_player(temp_db)
    upsert_league_roster_entry("Team Hickey", 0, pid, "SS", is_user_team=True)
    df = load_league_rosters()
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "Team Hickey"
    assert df.iloc[0]["is_user_team"] == 1

    clear_league_rosters()
    df = load_league_rosters()
    assert len(df) == 0


def test_refresh_log(temp_db):
    update_refresh_log("mlb_stats", "success")
    status = get_refresh_status("mlb_stats")
    assert status is not None
    assert status["status"] == "success"
    assert status["last_refresh"] is not None

    status2 = get_refresh_status("nonexistent")
    assert status2 is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_database_queries.py -v`
Expected: FAIL — `ImportError` because functions don't exist yet.

**Step 3: Implement query functions**

Add these functions to `src/database.py` after `load_player_pool()` (after line 429), before the `# -- Helpers` section:

```python
# ── In-Season Data Functions ──────────────────────────────────────


def upsert_season_stats(player_id: int, stats: dict, season: int = 2026):
    """Insert or update a player's season stats."""
    from datetime import datetime

    conn = get_connection()
    cols = [
        "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg",
        "ip", "w", "sv", "k", "era", "whip", "er", "bb_allowed",
        "h_allowed", "games_played",
    ]
    values = {c: stats.get(c, 0) for c in cols}
    values["last_updated"] = datetime.utcnow().isoformat()

    conn.execute(
        f"""INSERT INTO season_stats (player_id, season, {', '.join(cols)}, last_updated)
            VALUES (?, ?, {', '.join('?' for _ in cols)}, ?)
            ON CONFLICT(player_id, season) DO UPDATE SET
            {', '.join(f'{c}=excluded.{c}' for c in cols)},
            last_updated=excluded.last_updated""",
        (player_id, season, *[values[c] for c in cols], values["last_updated"]),
    )
    conn.commit()
    conn.close()


def load_season_stats(season: int = 2026) -> pd.DataFrame:
    """Load all season stats as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM season_stats WHERE season = ?", conn, params=(season,)
    )
    conn.close()
    return df


def upsert_ros_projection(player_id: int, system: str, stats: dict):
    """Insert or update a ROS projection for a player."""
    from datetime import datetime

    conn = get_connection()
    cols = [
        "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg",
        "ip", "w", "sv", "k", "era", "whip", "er", "bb_allowed", "h_allowed",
    ]
    values = {c: stats.get(c, 0) for c in cols}
    values["updated_at"] = datetime.utcnow().isoformat()

    conn.execute(
        f"""INSERT INTO ros_projections (player_id, system, {', '.join(cols)}, updated_at)
            VALUES (?, ?, {', '.join('?' for _ in cols)}, ?)
            ON CONFLICT(player_id, system) DO UPDATE SET
            {', '.join(f'{c}=excluded.{c}' for c in cols)},
            updated_at=excluded.updated_at""",
        (player_id, system, *[values[c] for c in cols], values["updated_at"]),
    )
    conn.commit()
    conn.close()


def upsert_league_roster_entry(
    team_name: str,
    team_index: int,
    player_id: int,
    roster_slot: str = None,
    is_user_team: bool = False,
):
    """Add a player to a league roster."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team)
           VALUES (?, ?, ?, ?, ?)""",
        (team_name, team_index, player_id, roster_slot, 1 if is_user_team else 0),
    )
    conn.commit()
    conn.close()


def load_league_rosters() -> pd.DataFrame:
    """Load all league rosters as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM league_rosters", conn)
    conn.close()
    return df


def clear_league_rosters():
    """Remove all league roster entries (for re-import)."""
    conn = get_connection()
    conn.execute("DELETE FROM league_rosters")
    conn.commit()
    conn.close()


def upsert_league_standing(
    team_name: str, category: str, total: float, rank: int = None, points: float = None
):
    """Insert or update a league standing entry."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM league_standings WHERE team_name = ? AND category = ?",
        (team_name, category),
    )
    conn.execute(
        "INSERT INTO league_standings (team_name, category, total, rank, points) VALUES (?, ?, ?, ?, ?)",
        (team_name, category, total, rank, points),
    )
    conn.commit()
    conn.close()


def update_refresh_log(source: str, status: str = "success"):
    """Update the refresh log for a data source."""
    from datetime import datetime

    conn = get_connection()
    conn.execute(
        """INSERT INTO refresh_log (source, last_refresh, status)
           VALUES (?, ?, ?)
           ON CONFLICT(source) DO UPDATE SET
           last_refresh=excluded.last_refresh, status=excluded.status""",
        (source, datetime.utcnow().isoformat(), status),
    )
    conn.commit()
    conn.close()


def get_refresh_status(source: str) -> dict | None:
    """Get the last refresh status for a data source."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT source, last_refresh, status FROM refresh_log WHERE source = ?",
        (source,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {"source": row[0], "last_refresh": row[1], "status": row[2]}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_database_queries.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/database.py tests/test_database_queries.py
git commit -m "feat: add database query functions for in-season tables"
```

---

### Task 4: Extract shared UI module (src/ui_shared.py)

**Files:**
- Create: `src/ui_shared.py`
- Modify: `app.py:54-92` (import from ui_shared instead of defining inline)

**Step 1: Create src/ui_shared.py**

```python
"""Shared UI constants and helpers used by all pages."""

THEME = {
    "bg": "#0a0e1a",
    "card": "#1a1f2e",
    "card_h": "#252b3b",
    "amber": "#f59e0b",
    "amber_l": "#fbbf24",
    "teal": "#06b6d4",
    "ok": "#84cc16",
    "danger": "#f43f5e",
    "warn": "#fb923c",
    "tx": "#f0f0f0",
    "tx2": "#8b95a5",
    "tiers": [
        "#f59e0b",
        "#fbbf24",
        "#84cc16",
        "#06b6d4",
        "#8b5cf6",
        "#f97316",
        "#f43f5e",
        "#6b7280",
    ],
}

ROSTER_CONFIG = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "Util": 2,
    "SP": 2,
    "RP": 2,
    "P": 4,
    "BN": 5,
}

HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

T = THEME  # shorthand for f-strings
```

**Step 2: Update app.py**

Replace the `THEME = {...}` block (lines 54-76), `ROSTER_CONFIG = {...}` block (lines 78-90), and `T = THEME` (line 92) with:

```python
from src.ui_shared import THEME, ROSTER_CONFIG, T
```

Keep `inject_custom_css()` in `app.py` (it references `T` which is now imported).

**Step 3: Verify imports**

Run: `python -c "from src.ui_shared import THEME, ROSTER_CONFIG, T; print('OK')"`
Expected: `OK`

Run: `python -c "import app; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/ui_shared.py app.py
git commit -m "refactor: extract THEME and ROSTER_CONFIG into src/ui_shared.py"
```

---

### Task 5: Create src/league_manager.py

**Files:**
- Create: `src/league_manager.py`
- Test: `tests/test_league_manager.py`

**Step 1: Write the failing test**

Create `tests/test_league_manager.py`:

```python
"""Test league manager: CSV import, roster management, free agents."""

import csv
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.league_manager import (
    import_league_rosters_csv,
    import_standings_csv,
    get_free_agents,
    get_team_roster,
    add_player_to_roster,
    remove_player_from_roster,
)


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (2, 'Shohei Ohtani', 'LAD', 'DH', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (3, 'Trea Turner', 'PHI', 'SS', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (4, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (5, 'Free Agent Guy', 'FA', 'OF', 1)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    os.unlink(tmp.name)


@pytest.fixture
def roster_csv(tmp_path):
    csv_path = tmp_path / "rosters.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["team_name", "player_name", "position", "roster_slot"])
        writer.writerow(["Team Hickey", "Aaron Judge", "OF", "OF"])
        writer.writerow(["Team Hickey", "Shohei Ohtani", "DH", "Util"])
        writer.writerow(["Team 2", "Trea Turner", "SS", "SS"])
        writer.writerow(["Team 2", "Gerrit Cole", "SP", "SP"])
    return str(csv_path)


@pytest.fixture
def standings_csv(tmp_path):
    csv_path = tmp_path / "standings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "team_name", "R", "HR", "RBI", "SB", "AVG",
            "W", "SV", "K", "ERA", "WHIP",
        ])
        writer.writerow([
            "Team Hickey", "450", "120", "430", "65", ".272",
            "45", "30", "650", "3.50", "1.15",
        ])
        writer.writerow([
            "Team 2", "420", "110", "410", "70", ".265",
            "50", "35", "700", "3.70", "1.20",
        ])
    return str(csv_path)


def test_import_rosters_csv(roster_csv, temp_db):
    count = import_league_rosters_csv(roster_csv, user_team_name="Team Hickey")
    assert count == 4
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 2


def test_import_standings_csv(standings_csv, temp_db):
    count = import_standings_csv(standings_csv)
    assert count == 2


def test_free_agents(roster_csv, temp_db):
    import_league_rosters_csv(roster_csv, user_team_name="Team Hickey")
    all_players = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "name": [
            "Aaron Judge", "Shohei Ohtani", "Trea Turner",
            "Gerrit Cole", "Free Agent Guy",
        ],
    })
    fa = get_free_agents(all_players)
    assert len(fa) == 1
    assert fa.iloc[0]["name"] == "Free Agent Guy"


def test_add_remove_player(temp_db):
    add_player_to_roster("Team Hickey", 0, 3, "SS", is_user_team=True)
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 1

    remove_player_from_roster("Team Hickey", 3)
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_league_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement src/league_manager.py**

```python
"""League management: roster imports, standings, free agents."""

import pandas as pd

from src.database import (
    get_connection,
    upsert_league_roster_entry,
    upsert_league_standing,
    load_league_rosters,
    clear_league_rosters,
)


def import_league_rosters_csv(csv_path: str, user_team_name: str) -> int:
    """Import all teams' rosters from CSV.

    Expected columns: team_name, player_name, position, roster_slot
    Returns number of roster entries imported.
    """
    clear_league_rosters()
    conn = get_connection()
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    imported = 0

    teams = sorted(df["team_name"].unique())
    team_index_map = {}
    idx = 1
    for t in teams:
        if t == user_team_name:
            team_index_map[t] = 0
        else:
            team_index_map[t] = idx
            idx += 1

    for _, row in df.iterrows():
        team = str(row["team_name"]).strip()
        player_name = str(row["player_name"]).strip()
        roster_slot = (
            str(row.get("roster_slot", "")).strip()
            if "roster_slot" in row.index
            else None
        )

        cursor = conn.cursor()
        cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
        result = cursor.fetchone()
        if not result:
            parts = player_name.split()
            if len(parts) >= 2:
                cursor.execute(
                    "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                    (f"%{parts[0]}%", f"%{parts[-1]}%"),
                )
                result = cursor.fetchone()
        if not result:
            continue

        player_id = result[0]
        is_user = team == user_team_name
        team_idx = team_index_map.get(team, 0)

        upsert_league_roster_entry(
            team, team_idx, player_id, roster_slot, is_user_team=is_user
        )
        imported += 1

    conn.close()
    return imported


def import_standings_csv(csv_path: str) -> int:
    """Import current standings from CSV.

    Expected columns: team_name, R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP
    Returns number of teams imported.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    categories = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    imported = 0

    for _, row in df.iterrows():
        team = str(row["team_name"]).strip()
        for cat in categories:
            if cat in row.index:
                val = float(str(row[cat]).replace(",", ""))
                upsert_league_standing(team, cat, val)
        imported += 1

    return imported


def get_team_roster(team_name: str) -> pd.DataFrame:
    """Return roster for a specific team with player info."""
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT lr.*, p.name, p.team, p.positions, p.is_hitter
           FROM league_rosters lr
           JOIN players p ON lr.player_id = p.player_id
           WHERE lr.team_name = ?""",
        conn,
        params=(team_name,),
    )
    conn.close()
    return df


def get_free_agents(player_pool: pd.DataFrame) -> pd.DataFrame:
    """Return all players NOT on any league_rosters team."""
    rostered = load_league_rosters()
    if rostered.empty:
        return player_pool
    rostered_ids = set(rostered["player_id"].values)
    return player_pool[~player_pool["player_id"].isin(rostered_ids)].copy()


def add_player_to_roster(
    team_name: str,
    team_index: int,
    player_id: int,
    roster_slot: str = None,
    is_user_team: bool = False,
):
    """Add a player to a team roster."""
    upsert_league_roster_entry(
        team_name, team_index, player_id, roster_slot, is_user_team
    )


def remove_player_from_roster(team_name: str, player_id: int):
    """Remove a player from a team roster."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM league_rosters WHERE team_name = ? AND player_id = ?",
        (team_name, player_id),
    )
    conn.commit()
    conn.close()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_league_manager.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/league_manager.py tests/test_league_manager.py
git commit -m "feat: add league manager for roster/standings CSV import"
```

---

## Phase 2: Data Pipeline

### Task 6: Create src/live_stats.py — MLB Stats API integration

**Files:**
- Create: `src/live_stats.py`
- Test: `tests/test_live_stats.py`

**Step 1: Write the failing test**

Create `tests/test_live_stats.py`:

```python
"""Test live stats module — MLB Stats API + pybaseball integration."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (2, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    os.unlink(tmp.name)


def test_match_player_id(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Aaron Judge", "NYY")
    assert pid == 1


def test_match_player_id_not_found(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Nonexistent Player", "XXX")
    assert pid is None


@patch("src.live_stats.statsapi")
def test_fetch_season_stats_structure(mock_statsapi, temp_db):
    from src.live_stats import fetch_season_stats

    mock_statsapi.get.return_value = {
        "people": [
            {
                "id": 592450,
                "fullName": "Aaron Judge",
                "currentTeam": {"abbreviation": "NYY"},
                "stats": [
                    {
                        "splits": [
                            {
                                "stat": {
                                    "plateAppearances": 500,
                                    "atBats": 450,
                                    "hits": 130,
                                    "runs": 80,
                                    "homeRuns": 35,
                                    "rbi": 90,
                                    "stolenBases": 5,
                                    "avg": ".289",
                                    "gamesPlayed": 120,
                                }
                            }
                        ]
                    }
                ],
            }
        ]
    }

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    assert "player_name" in df.columns
    assert "hr" in df.columns


def test_get_refresh_age():
    from src.live_stats import _get_refresh_age_hours

    age = _get_refresh_age_hours("nonexistent_source")
    assert age > 24


@patch("src.live_stats.statsapi")
def test_refresh_all_stats(mock_statsapi, temp_db):
    from src.live_stats import refresh_all_stats

    mock_statsapi.get.return_value = {"people": []}

    result = refresh_all_stats(force=True)
    assert isinstance(result, dict)
    assert "season_stats" in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_stats.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement src/live_stats.py**

```python
"""Live stats data pipeline: MLB Stats API + pybaseball.

Fetches current season stats, ROS projections, and park factors
from free public data sources. No Yahoo dependency.
"""

from datetime import datetime

import pandas as pd

try:
    import statsapi
except ImportError:
    statsapi = None

from src.database import (
    get_connection,
    upsert_season_stats,
    update_refresh_log,
    get_refresh_status,
)


def match_player_id(player_name: str, team_abbr: str) -> int | None:
    """Match an external player name to our players table."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
    result = cursor.fetchone()
    if result:
        conn.close()
        return result[0]

    parts = player_name.replace(".", "").split()
    if len(parts) >= 2:
        last_name = parts[-1]
        cursor.execute(
            "SELECT player_id FROM players WHERE name LIKE ? AND team = ?",
            (f"%{last_name}%", team_abbr),
        )
        result = cursor.fetchone()
        if result:
            conn.close()
            return result[0]

    if len(parts) >= 1:
        last_name = parts[-1]
        cursor.execute(
            "SELECT player_id FROM players WHERE name LIKE ?",
            (f"% {last_name}",),
        )
        results = cursor.fetchall()
        if len(results) == 1:
            conn.close()
            return results[0][0]

    conn.close()
    return None


def fetch_season_stats(season: int = 2026) -> pd.DataFrame:
    """Pull current season stats for all MLB players via MLB Stats API."""
    if statsapi is None:
        raise ImportError(
            "MLB-StatsAPI is required. Install with: pip install MLB-StatsAPI"
        )

    rows = []

    try:
        hitting = statsapi.get(
            "sports_players",
            {
                "season": season,
                "gameType": "R",
                "group": "hitting",
                "stats": "season",
                "limit": 1000,
                "sportId": 1,
            },
        )
        for player in hitting.get("people", []):
            stats_list = player.get("stats", [])
            if not stats_list or not stats_list[0].get("splits"):
                continue
            s = stats_list[0]["splits"][0]["stat"]
            team_info = player.get("currentTeam", {})
            rows.append({
                "player_name": player.get("fullName", ""),
                "team": team_info.get("abbreviation", ""),
                "is_hitter": True,
                "pa": int(s.get("plateAppearances", 0)),
                "ab": int(s.get("atBats", 0)),
                "h": int(s.get("hits", 0)),
                "r": int(s.get("runs", 0)),
                "hr": int(s.get("homeRuns", 0)),
                "rbi": int(s.get("rbi", 0)),
                "sb": int(s.get("stolenBases", 0)),
                "avg": float(s.get("avg", "0") or 0),
                "games_played": int(s.get("gamesPlayed", 0)),
                "ip": 0, "w": 0, "sv": 0, "k": 0,
                "era": 0, "whip": 0, "er": 0,
                "bb_allowed": 0, "h_allowed": 0,
            })
    except Exception:
        pass

    try:
        pitching = statsapi.get(
            "sports_players",
            {
                "season": season,
                "gameType": "R",
                "group": "pitching",
                "stats": "season",
                "limit": 1000,
                "sportId": 1,
            },
        )
        for player in pitching.get("people", []):
            stats_list = player.get("stats", [])
            if not stats_list or not stats_list[0].get("splits"):
                continue
            s = stats_list[0]["splits"][0]["stat"]
            team_info = player.get("currentTeam", {})
            rows.append({
                "player_name": player.get("fullName", ""),
                "team": team_info.get("abbreviation", ""),
                "is_hitter": False,
                "pa": 0, "ab": 0, "h": 0, "r": 0,
                "hr": 0, "rbi": 0, "sb": 0, "avg": 0,
                "ip": float(s.get("inningsPitched", "0") or 0),
                "w": int(s.get("wins", 0)),
                "sv": int(s.get("saves", 0)),
                "k": int(s.get("strikeOuts", 0)),
                "era": float(s.get("era", "0") or 0),
                "whip": float(s.get("whip", "0") or 0),
                "er": int(s.get("earnedRuns", 0)),
                "bb_allowed": int(s.get("baseOnBalls", 0)),
                "h_allowed": int(s.get("hits", 0)),
                "games_played": int(s.get("gamesPlayed", 0)),
            })
    except Exception:
        pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def save_season_stats_to_db(stats_df: pd.DataFrame) -> int:
    """Match fetched stats to players table and save to season_stats."""
    saved = 0
    for _, row in stats_df.iterrows():
        player_id = match_player_id(row["player_name"], row.get("team", ""))
        if player_id is None:
            continue
        upsert_season_stats(player_id, row.to_dict())
        saved += 1
    return saved


def _get_refresh_age_hours(source: str) -> float:
    """How many hours since the last successful refresh."""
    status = get_refresh_status(source)
    if status is None or status["last_refresh"] is None:
        return 999.0
    try:
        last = datetime.fromisoformat(status["last_refresh"])
        return (datetime.utcnow() - last).total_seconds() / 3600
    except (ValueError, TypeError):
        return 999.0


def refresh_all_stats(force: bool = False) -> dict:
    """Orchestrator: refresh season stats if stale (>24h)."""
    results = {}

    age = _get_refresh_age_hours("season_stats")
    if force or age > 24:
        try:
            df = fetch_season_stats()
            if not df.empty:
                saved = save_season_stats_to_db(df)
                update_refresh_log("season_stats", "success")
                results["season_stats"] = f"Saved {saved} player stats"
            else:
                update_refresh_log("season_stats", "empty")
                results["season_stats"] = "No data returned"
        except Exception as e:
            update_refresh_log("season_stats", f"error: {e}")
            results["season_stats"] = f"Error: {e}"
    else:
        results["season_stats"] = f"Fresh (updated {age:.1f}h ago)"

    return results
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_live_stats.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/live_stats.py tests/test_live_stats.py
git commit -m "feat: add live stats pipeline with MLB Stats API integration"
```

---

## Phase 3: Core Algorithms

### Task 7: Create src/in_season.py — Trade Analyzer, Player Compare, FA Ranker

**Files:**
- Create: `src/in_season.py`
- Test: `tests/test_in_season.py`

**Step 1: Write the failing test**

Create `tests/test_in_season.py`:

```python
"""Test in-season analysis: trade analyzer, player comparison, FA ranker."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    players = [
        (1, "Aaron Judge", "NYY", "OF", 1),
        (2, "Shohei Ohtani", "LAD", "DH", 1),
        (3, "Trea Turner", "PHI", "SS", 1),
        (4, "Gerrit Cole", "NYY", "SP", 0),
        (5, "Corbin Burnes", "BAL", "SP", 0),
        (6, "Bobby Witt Jr", "KC", "SS", 1),
    ]
    for p in players:
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter) "
            "VALUES (?,?,?,?,?)",
            p,
        )
    hitter_projs = [
        (1, "blended", 600, 550, 160, 100, 42, 110, 8, 0.291),
        (2, "blended", 650, 590, 175, 110, 45, 115, 15, 0.297),
        (3, "blended", 580, 530, 155, 85, 18, 70, 30, 0.292),
        (6, "blended", 620, 570, 170, 95, 28, 90, 35, 0.298),
    ]
    pitcher_projs = [
        (4, "blended", 200, 15, 0, 220, 2.80, 1.05, 62, 50, 160),
        (5, "blended", 190, 13, 0, 200, 3.10, 1.10, 65, 55, 155),
    ]
    for p in hitter_projs:
        conn.execute(
            "INSERT INTO projections "
            "(player_id, system, pa, ab, h, r, hr, rbi, sb, avg) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            p,
        )
    for p in pitcher_projs:
        conn.execute(
            "INSERT INTO projections "
            "(player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            p,
        )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    os.unlink(tmp.name)


def _make_player_pool():
    from src.database import load_player_pool

    pool = load_player_pool()
    pool = pool.rename(columns={"name": "player_name"})
    return pool


def test_analyze_trade_returns_result(temp_db):
    from src.in_season import analyze_trade

    config = LeagueConfig()
    pool = _make_player_pool()

    result = analyze_trade(
        giving_ids=[1],
        receiving_ids=[4],
        user_roster_ids=[1, 2, 3],
        player_pool=pool,
        config=config,
    )
    assert result is not None
    assert "verdict" in result
    assert "category_impact" in result
    assert isinstance(result["category_impact"], dict)


def test_analyze_trade_category_impact(temp_db):
    from src.in_season import analyze_trade

    config = LeagueConfig()
    pool = _make_player_pool()

    result = analyze_trade(
        giving_ids=[1],
        receiving_ids=[4],
        user_roster_ids=[1, 2, 3],
        player_pool=pool,
        config=config,
    )
    assert result["category_impact"]["HR"] < 0


def test_compare_players(temp_db):
    from src.in_season import compare_players

    config = LeagueConfig()
    pool = _make_player_pool()

    result = compare_players(1, 2, pool, config)
    assert "player_a" in result
    assert "player_b" in result
    assert "z_scores_a" in result
    assert "z_scores_b" in result
    assert "composite_a" in result
    assert "composite_b" in result


def test_rank_free_agents(temp_db):
    from src.in_season import rank_free_agents

    config = LeagueConfig()
    pool = _make_player_pool()

    user_roster_ids = [1, 2, 3]
    fa_pool = pool[pool["player_id"].isin([6])]

    ranked = rank_free_agents(user_roster_ids, fa_pool, pool, config)
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) >= 1
    assert "marginal_value" in ranked.columns
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_in_season.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement src/in_season.py**

```python
"""In-season analysis: trade analyzer, player comparison, free agent ranker.

Reuses SGPCalculator, compute_replacement_levels, compute_category_weights
from src/valuation.py. MC simulation pattern adapted from src/simulation.py.
"""

import numpy as np
import pandas as pd

from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_category_weights,
)


def _roster_category_totals(roster_ids: list, player_pool: pd.DataFrame) -> dict:
    """Compute aggregate category totals for a set of player IDs."""
    roster = player_pool[player_pool["player_id"].isin(roster_ids)]
    totals = {
        "R": 0, "HR": 0, "RBI": 0, "SB": 0,
        "W": 0, "SV": 0, "K": 0,
        "ab": 0, "h": 0, "ip": 0, "er": 0,
        "bb_allowed": 0, "h_allowed": 0,
    }
    for _, p in roster.iterrows():
        totals["R"] += int(p.get("r", 0) or 0)
        totals["HR"] += int(p.get("hr", 0) or 0)
        totals["RBI"] += int(p.get("rbi", 0) or 0)
        totals["SB"] += int(p.get("sb", 0) or 0)
        totals["W"] += int(p.get("w", 0) or 0)
        totals["SV"] += int(p.get("sv", 0) or 0)
        totals["K"] += int(p.get("k", 0) or 0)
        totals["ab"] += int(p.get("ab", 0) or 0)
        totals["h"] += int(p.get("h", 0) or 0)
        totals["ip"] += float(p.get("ip", 0) or 0)
        totals["er"] += int(p.get("er", 0) or 0)
        totals["bb_allowed"] += int(p.get("bb_allowed", 0) or 0)
        totals["h_allowed"] += int(p.get("h_allowed", 0) or 0)

    if totals["ab"] > 0:
        totals["AVG"] = totals["h"] / totals["ab"]
    else:
        totals["AVG"] = 0
    if totals["ip"] > 0:
        totals["ERA"] = totals["er"] * 9 / totals["ip"]
        totals["WHIP"] = (totals["bb_allowed"] + totals["h_allowed"]) / totals["ip"]
    else:
        totals["ERA"] = 0
        totals["WHIP"] = 0

    return totals


def analyze_trade(
    giving_ids: list,
    receiving_ids: list,
    user_roster_ids: list,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    n_sims: int = 200,
) -> dict:
    """Analyze a trade proposal using Live SGP + Monte Carlo simulation.

    Returns dict with verdict, confidence_pct, category_impact,
    total_sgp_change, mc_mean, mc_std, risk_flags, before/after totals.
    """
    sgp_calc = SGPCalculator(config)

    before_ids = list(user_roster_ids)
    after_ids = [pid for pid in before_ids if pid not in giving_ids] + list(
        receiving_ids
    )

    before_totals = _roster_category_totals(before_ids, player_pool)
    after_totals = _roster_category_totals(after_ids, player_pool)

    category_impact = {}
    total_sgp_change = 0.0
    for cat in config.all_categories:
        denom = config.sgp_denominators[cat]
        if denom == 0:
            denom = 1.0

        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        raw_change = after_val - before_val

        if cat in config.inverse_stats:
            sgp_change = -raw_change / denom
        else:
            sgp_change = raw_change / denom

        category_impact[cat] = round(sgp_change, 3)
        total_sgp_change += sgp_change

    mc_results = []
    for _ in range(n_sims):
        noise = np.random.normal(1.0, 0.08, size=len(config.all_categories))
        sim_change = sum(
            category_impact[cat] * noise[i]
            for i, cat in enumerate(config.all_categories)
        )
        mc_results.append(sim_change)

    mc_results = np.array(mc_results)
    pct_positive = (mc_results > 0).mean() * 100

    risk_flags = []
    giving_players = player_pool[player_pool["player_id"].isin(giving_ids)]
    receiving_players = player_pool[player_pool["player_id"].isin(receiving_ids)]

    for _, p in receiving_players.iterrows():
        if p.get("is_injured", 0):
            risk_flags.append(
                f"{p.get('player_name', p.get('name', '?'))} is injured"
            )

    for _, p in giving_players.iterrows():
        sgp = sgp_calc.total_sgp(p)
        if sgp > 3.0:
            risk_flags.append(
                f"Trading away elite player: "
                f"{p.get('player_name', p.get('name', '?'))}"
            )

    verdict = "ACCEPT" if pct_positive >= 55 else "DECLINE"

    return {
        "verdict": verdict,
        "confidence_pct": round(pct_positive, 1),
        "category_impact": category_impact,
        "total_sgp_change": round(total_sgp_change, 3),
        "mc_mean": round(float(mc_results.mean()), 3),
        "mc_std": round(float(mc_results.std()), 3),
        "risk_flags": risk_flags,
        "before_totals": before_totals,
        "after_totals": after_totals,
    }


def compare_players(
    player_id_a: int,
    player_id_b: int,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
) -> dict:
    """Compare two players using z-score normalization across all categories."""
    player_a = player_pool[player_pool["player_id"] == player_id_a]
    player_b = player_pool[player_pool["player_id"] == player_id_b]

    if player_a.empty or player_b.empty:
        return {"error": "Player not found"}

    pa = player_a.iloc[0]
    pb = player_b.iloc[0]

    stat_map = {
        "R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb", "AVG": "avg",
        "W": "w", "SV": "sv", "K": "k", "ERA": "era", "WHIP": "whip",
    }

    z_a, z_b = {}, {}
    for cat in config.all_categories:
        col = stat_map[cat]
        vals = player_pool[col].dropna()
        mean = vals.mean()
        std = vals.std()
        if std == 0:
            std = 1.0

        val_a = float(pa.get(col, 0) or 0)
        val_b = float(pb.get(col, 0) or 0)

        if cat in config.inverse_stats:
            z_a[cat] = -(val_a - mean) / std
            z_b[cat] = -(val_b - mean) / std
        else:
            z_a[cat] = (val_a - mean) / std
            z_b[cat] = (val_b - mean) / std

    composite_a = sum(z_a.values())
    composite_b = sum(z_b.values())

    advantages = {}
    for cat in config.all_categories:
        if z_a[cat] > z_b[cat]:
            advantages[cat] = "A"
        elif z_b[cat] > z_a[cat]:
            advantages[cat] = "B"
        else:
            advantages[cat] = "TIE"

    return {
        "player_a": pa.get("player_name", pa.get("name", "?")),
        "player_b": pb.get("player_name", pb.get("name", "?")),
        "z_scores_a": z_a,
        "z_scores_b": z_b,
        "composite_a": round(composite_a, 3),
        "composite_b": round(composite_b, 3),
        "advantages": advantages,
    }


def rank_free_agents(
    user_roster_ids: list,
    fa_pool: pd.DataFrame,
    full_pool: pd.DataFrame,
    config: LeagueConfig,
) -> pd.DataFrame:
    """Rank free agents by marginal value relative to user's roster."""
    sgp_calc = SGPCalculator(config)
    roster_totals = _roster_category_totals(user_roster_ids, full_pool)
    weights = compute_category_weights(roster_totals, config)

    records = []
    for _, fa in fa_pool.iterrows():
        marginal = sgp_calc.marginal_sgp(fa, roster_totals, weights)
        total_marginal = sum(marginal.values())

        best_cat = max(marginal, key=marginal.get) if marginal else ""
        best_cat_val = marginal.get(best_cat, 0)

        records.append({
            "player_id": fa["player_id"],
            "player_name": fa.get("player_name", fa.get("name", "?")),
            "positions": fa.get("positions", ""),
            "marginal_value": round(total_marginal, 3),
            "best_category": best_cat,
            "best_cat_impact": round(best_cat_val, 3),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(
            "marginal_value", ascending=False
        ).reset_index(drop=True)
    return result
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_in_season.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/in_season.py tests/test_in_season.py
git commit -m "feat: add trade analyzer, player comparison, and FA ranker"
```

---

## Phase 4: UI Pages

### Task 8: Create Streamlit pages directory and all 4 pages

**Files:**
- Create: `pages/1_My_Team.py`
- Create: `pages/2_Trade_Analyzer.py`
- Create: `pages/3_Player_Compare.py`
- Create: `pages/4_Free_Agents.py`

These are Streamlit UI pages. They import and use the backend modules built in Phases 1-3. Since they run in Streamlit context, test by running the app and visually verifying. See the design document at `docs/plans/2026-03-10-in-season-management-design.md` for detailed layout specs.

**Step 1: Create `pages/` directory**

Run: `mkdir -p pages`

**Step 2: Create all 4 page files**

See the design document for exact UI layout. Each page should:
- Import from `src/ui_shared.py` for theme constants
- Import from backend modules (`src/database.py`, `src/league_manager.py`, `src/in_season.py`, `src/live_stats.py`)
- Apply Broadcast Booth dark theme via inline CSS
- Use `st.set_page_config()` with appropriate title and icon

Key page behaviors:

**`pages/1_My_Team.py`**: Team roster table, category standings heatmap (10 cats x 12 teams), strengths/weaknesses sidebar, refresh stats button.

**`pages/2_Trade_Analyzer.py`**: Two multiselect dropdowns (give/receive), "Analyze Trade" button, verdict banner (green ACCEPT / red DECLINE + confidence %), category impact table, MC results metrics, risk flags.

**`pages/3_Player_Compare.py`**: Two player selectboxes, Plotly radar chart overlay, side-by-side z-score table with advantage column.

**`pages/4_Free_Agents.py`**: Position filter dropdown, min marginal value slider, ranked table with player name, positions, marginal value, best category, category impact.

**Step 3: Verify pages create**

Run: `ls pages/`
Expected: Four `.py` files.

**Step 4: Commit**

```bash
git add pages/
git commit -m "feat: add 4 in-season Streamlit pages (My Team, Trade, Compare, Free Agents)"
```

---

## Phase 5: Integration

### Task 9: Add Setup Wizard Step 5 — League Import

**Files:**
- Modify: `app.py`

**Step 1: Add league_manager import to app.py**

Add near the top of `app.py` (after existing imports):
```python
from src.league_manager import import_league_rosters_csv, import_standings_csv
```

**Step 2: Add Step 5 to the wizard**

Locate the wizard section in `app.py`. After Step 4 and before the wizard completion logic, add Step 5 with:
- Text input for user team name
- File uploader for roster CSV
- File uploader for standings CSV
- Import buttons with success/error feedback

Update the wizard total steps from 4 to 5.

**Step 3: Verify app imports**

Run: `python -c "import app; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Step 5 to setup wizard for league import"
```

---

### Task 10: Remove yahoo_api.py dependency from app.py

**Files:**
- Modify: `app.py` (lines 30-36, plus all yahoo references in wizard)

**Step 1: Remove yahoo_api import block from app.py**

Delete lines 30-36:
```python
from src.yahoo_api import (
    YahooFantasyClient,
    has_credentials,
    load_credentials,
    save_credentials,
    sync_draft_picks,
)
```

**Step 2: Remove all yahoo_api references**

Search `app.py` for `yahoo`, `YahooFantasyClient`, `has_credentials`, etc. Remove or replace the Yahoo-specific wizard step (likely Step 3: "Connect to Yahoo") with a simpler step or renumber steps.

**Step 3: Verify and lint**

Run: `python -c "import app; print('OK')"`
Expected: `OK`

Run: `python -m ruff check app.py`
Expected: No errors.

**Step 4: Commit**

```bash
git add app.py
git commit -m "refactor: remove yahoo_api dependency from app.py"
```

---

### Task 11: Run full lint, format, and test suite

**Step 1: Lint**

Run: `python -m ruff check .`
Expected: No errors. If errors found, fix them.

**Step 2: Format**

Run: `python -m ruff format --check .`
Expected: All clean. If not, run `python -m ruff format .`

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS.

**Step 4: Verify existing draft still works**

Run: `python load_sample_data.py && python -c "import app; print('OK')"`
Expected: Both succeed.

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: lint and format cleanup for in-season features"
```

---

### Task 12: Update CLAUDE.md and memory

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add new sections documenting:
- New file structure entries (4 pages, 3 new modules, ui_shared.py)
- New database tables (6 tables)
- New dependencies (MLB-StatsAPI, pybaseball)
- In-season features overview
- Trade analyzer algorithm summary
- Updated architecture section

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with in-season management features"
```

---

## Phase 6: Verification

### Task 13: End-to-end verification

**Step 1: Verify all new modules import**

```bash
python -c "from src.live_stats import *; print('live_stats OK')"
python -c "from src.in_season import *; print('in_season OK')"
python -c "from src.league_manager import *; print('league_manager OK')"
python -c "from src.ui_shared import *; print('ui_shared OK')"
```

**Step 2: Verify database tables**

```bash
python -c "from src.database import init_db; init_db(); print('DB OK')"
```

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

**Step 4: Run CI checks locally**

```bash
python -m ruff check . --output-format=github
python -m ruff format --check .
python load_sample_data.py
python -m pytest tests/ -v
python -c "import app; print('app.py OK')"
```

**Step 5: Verify existing draft not broken**

```bash
python -c "
from src.database import init_db, load_player_pool
from src.valuation import LeagueConfig, SGPCalculator, compute_replacement_levels, compute_sgp_denominators, value_all_players
init_db()
pool = load_player_pool()
print(f'Player pool: {len(pool)} players')
lc = LeagueConfig()
sgp = SGPCalculator(lc)
repl = compute_replacement_levels(pool, lc, sgp)
valued = value_all_players(pool, lc, replacement_levels=repl)
print(f'Valued: {len(valued)} players')
print('Draft engine OK')
"
```

**Step 6: Final commit and push**

```bash
git add -A
git commit -m "feat: complete in-season management v1.0"
git push origin master
```

Verify CI pipeline passes all jobs (lint, test, build).
