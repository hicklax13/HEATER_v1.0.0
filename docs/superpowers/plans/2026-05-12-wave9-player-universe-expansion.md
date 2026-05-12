# Wave 9: Player Universe Expansion (INFRA-F5, Option B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the player universe from ~1,300 (40-man + spring training MLB) to ~2,200 by adding AAA/AA top-30 per team (~+900 minor leaguers), enabling visibility into call-up candidates and depth-chart prospects.

**Architecture:** Adds a `level` column to `players` (NULL/`"MLB"` default, `"AAA"`/`"AA"` for minors). New bootstrap phase queries MLB Stats API `sports_players` with sportId=11 (AAA) and sportId=12 (AA), caps at top-30 per minor-league team. Pool exposes `level`; pages get optional level filter. Yahoo can't see minor leaguers, so `percent_owned` stays NULL for them — pages mark "no Yahoo data" appropriately.

**Tech Stack:** Python 3.11+, MLB-StatsAPI, SQLite (WAL via `get_connection`), Streamlit pages, pytest.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md) (INFRA-F5)

---

## Cold-Start Context

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave9`
- Branch: `claude/audit-wave9-player-universe`
- HEAD: `1caa63a Merge pull request #23 from hicklax13/claude/audit-wave8d-code-simplifier`

### Key existing patterns to follow
- `_safe_add_column(conn, table, col, type_decl)` for schema migrations — see `src/database.py:618-640` for examples
- `statsapi.get("sports_players", {"season": Y, "sportId": N, "gameType": "R"})` for player fetches — see `src/live_stats.py:670-711` (`fetch_all_mlb_players`)
- 3-tier waterfall for bootstrap phases (primary → fallback → emergency seed) — though for this phase, Tier 1 is the only practical path (no minor-league seed available)
- `update_refresh_log_auto(name, count, expected_min, message)` for refresh_log status (Wave 7 SF-54 pattern)
- `_build_player_pool` in `src/database.py` already does `SELECT * FROM players p ...`; adding `level` to the table makes it flow through automatically

### Sport IDs (MLB Stats API)
- 1 = MLB (current)
- 11 = Triple-A (AAA)
- 12 = Double-A (AA)
- 13 = High-A
- 14 = Low-A
- 16 = Rookie

### Expected scale
- AAA: ~30 affiliates × ~25-30 roster = 750-900 players
- AA: ~30 affiliates × ~25-30 roster = 750-900 players
- After top-30-per-team cap: ~900 total combined (already close to natural roster size)

⚠️ All tasks: use `pytest -q 2>&1 | tail -10`. Commit explicitly per task.

---

## Task 1: Schema migration — add `level` column

**Files:**
- Modify: `src/database.py:618-640` (add `_safe_add_column` line)
- Test: `tests/test_wave9_level_column_migration.py` (create)

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_wave9_level_column_migration.py`:

```python
"""Wave 9 / Task 1: players table has `level` column after init_db."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    return db_path


def test_init_db_adds_level_column(temp_db):
    """After init_db(), players.level must exist (NULL for legacy rows)."""
    from src.database import init_db
    init_db()
    conn = sqlite3.connect(temp_db)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()]
    finally:
        conn.close()
    assert "level" in cols, (
        f"Wave 9 regression: players.level column missing after init_db. "
        f"Found columns: {cols}"
    )


def test_safe_add_column_level_idempotent(temp_db):
    """Calling _safe_add_column twice must not raise."""
    from src.database import _safe_add_column, get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        _safe_add_column(conn, "players", "level", "TEXT")
        _safe_add_column(conn, "players", "level", "TEXT")
    finally:
        conn.close()
```

- [ ] **Step 1.2: Run test (should FAIL)**

```bash
python -m pytest tests/test_wave9_level_column_migration.py -q 2>&1 | tail -10
```
Expected: FAIL — column not in table.

- [ ] **Step 1.3: Add the migration**

Edit `src/database.py` around line 636 (after existing `bats`/`throws` migrations). Add:

```python
    # Wave 9 INFRA-F5: minor-league universe expansion (Option B)
    _safe_add_column(conn, "players", "level", "TEXT")  # NULL/"MLB"/"AAA"/"AA"
```

Place it next to other Phase 2 bio-field migrations.

- [ ] **Step 1.4: Run test (should PASS)**

```bash
python -m pytest tests/test_wave9_level_column_migration.py -q 2>&1 | tail -10
```

- [ ] **Step 1.5: Lint + commit**

```bash
python -m ruff format src/database.py tests/test_wave9_level_column_migration.py
python -m ruff check src/database.py tests/test_wave9_level_column_migration.py
git add src/database.py tests/test_wave9_level_column_migration.py
git commit -m "feat(wave9): add players.level column for AAA/AA expansion (INFRA-F5 Task 1)"
```

---

## Task 2: `fetch_minor_league_players` in `live_stats.py`

**Files:**
- Modify: `src/live_stats.py` (add new function near `fetch_all_mlb_players`)
- Test: `tests/test_wave9_minor_league_fetch.py` (create)

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_wave9_minor_league_fetch.py`:

```python
"""Wave 9 / Task 2: fetch_minor_league_players from MLB Stats API."""

from unittest.mock import patch

import pandas as pd


def _make_fake_sports_players_response(level_code: str, num_teams: int = 2, players_per_team: int = 40):
    """Build a fake statsapi.get('sports_players') response for minor leagues."""
    people = []
    for team_idx in range(num_teams):
        team_id = 100 + team_idx
        for p_idx in range(players_per_team):
            people.append({
                "id": 600000 + team_idx * 1000 + p_idx,
                "fullName": f"Player {level_code}{team_idx}-{p_idx}",
                "active": True,
                "primaryPosition": {"abbreviation": "OF", "type": "Outfielder"},
                "currentTeam": {"id": team_id},
                "batSide": {"code": "R"},
                "pitchHand": {"code": "R"},
                "birthDate": "2000-01-01",
            })
    return {"people": people}


def test_fetch_minor_league_players_caps_at_top_n_per_team():
    """fetch_minor_league_players returns at most top_n_per_team rows per affiliate."""
    from src import live_stats

    fake = _make_fake_sports_players_response("AAA", num_teams=3, players_per_team=50)
    with patch.object(live_stats.statsapi, "get", return_value=fake), \
         patch.object(live_stats, "_build_team_id_map", return_value={100: "SCR", 101: "OMA", 102: "BUF"}):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",), top_n_per_team=30)

    assert not df.empty
    # 3 teams × 30 cap = 90 rows
    assert len(df) == 90, f"Expected 90 rows (3×30 cap); got {len(df)}"
    # Each team has exactly 30
    counts_per_team = df.groupby("team").size().to_dict()
    assert all(c == 30 for c in counts_per_team.values()), f"Cap violated: {counts_per_team}"


def test_fetch_minor_league_players_sets_level_column():
    """Each row gets level='AAA' or 'AA' matching the source sportId."""
    from src import live_stats

    aaa_resp = _make_fake_sports_players_response("AAA", num_teams=1, players_per_team=10)
    aa_resp = _make_fake_sports_players_response("AA", num_teams=1, players_per_team=10)

    def _fake_get(endpoint, params, **kwargs):
        if params.get("sportId") == 11:
            return aaa_resp
        if params.get("sportId") == 12:
            return aa_resp
        return {"people": []}

    with patch.object(live_stats.statsapi, "get", side_effect=_fake_get), \
         patch.object(live_stats, "_build_team_id_map", return_value={100: "SCR"}):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA", "AA"), top_n_per_team=30)

    assert "level" in df.columns
    assert set(df["level"].unique()) == {"AAA", "AA"}
    assert (df[df["level"] == "AAA"]["mlb_id"] >= 600000).all()
    assert (df[df["level"] == "AA"]["mlb_id"] >= 600000).all()


def test_fetch_minor_league_players_handles_empty_response():
    """Empty API response → empty DataFrame, no crash."""
    from src import live_stats
    with patch.object(live_stats.statsapi, "get", return_value={"people": []}), \
         patch.object(live_stats, "_build_team_id_map", return_value={}):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",))
    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_fetch_minor_league_players_handles_api_failure():
    """statsapi exception → empty DataFrame + logger.warning, no crash."""
    from src import live_stats
    with patch.object(live_stats.statsapi, "get", side_effect=ConnectionError("simulated")), \
         patch.object(live_stats, "_build_team_id_map", return_value={}):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",))
    assert df.empty
```

- [ ] **Step 2.2: Run test (should FAIL — function not defined)**

```bash
python -m pytest tests/test_wave9_minor_league_fetch.py -q 2>&1 | tail -10
```

- [ ] **Step 2.3: Implement the function**

Add to `src/live_stats.py` immediately after `fetch_all_mlb_players` (around line 712):

```python
# Wave 9: MLB Stats API sportId codes for minor leagues
_MINOR_LEAGUE_SPORT_IDS: dict[str, int] = {
    "AAA": 11,
    "AA": 12,
}


def fetch_minor_league_players(
    season: int = 2026,
    levels: tuple[str, ...] = ("AAA", "AA"),
    top_n_per_team: int = 30,
) -> pd.DataFrame:
    """Fetch AAA/AA active players, capped at top-N per affiliate.

    Wave 9 INFRA-F5 (Option B): expands the player universe beyond the
    40-man MLB roster + spring training pool to include the top
    minor-league prospects/depth-chart candidates.

    Returns a DataFrame with the same columns as fetch_all_mlb_players()
    plus a `level` column set to "AAA" or "AA" per the source sportId.
    Yahoo ownership data is NOT available for minor leaguers — consumers
    must handle NULL percent_owned appropriately.

    Args:
        season: Season year (default 2026).
        levels: Minor-league levels to fetch (default both AAA and AA).
        top_n_per_team: Cap per affiliate to control universe size.

    Returns:
        DataFrame indexed 0..N with columns matching fetch_all_mlb_players()
        plus `level`. Empty DataFrame on any failure.
    """
    if statsapi is None:
        logger.warning(
            "fetch_minor_league_players: MLB-StatsAPI unavailable; "
            "returning empty (no minor-league universe expansion)"
        )
        return pd.DataFrame()

    team_map = _build_team_id_map(season)

    all_rows: list[dict] = []
    for level in levels:
        sport_id = _MINOR_LEAGUE_SPORT_IDS.get(level)
        if sport_id is None:
            logger.warning(
                "fetch_minor_league_players: unknown level %r; skipping",
                level,
            )
            continue

        try:
            data = statsapi.get(
                "sports_players",
                {"season": season, "sportId": sport_id, "gameType": "R"},
                request_kwargs={"timeout": _API_TIMEOUT},
            )
        except Exception as exc:
            logger.warning(
                "fetch_minor_league_players: %s fetch failed (sportId=%d): %s",
                level,
                sport_id,
                exc,
                exc_info=True,
            )
            continue

        # Group raw rows by team_id so we can cap per affiliate.
        by_team: dict[int, list[dict]] = {}
        for p in data.get("people", []):
            if not p.get("active", False):
                continue
            team_id = p.get("currentTeam", {}).get("id")
            if team_id is None:
                continue
            by_team.setdefault(team_id, []).append(p)

        for team_id, players in by_team.items():
            # Cap at top_n_per_team — preserve API order (MLB sorts by
            # roster position, which approximates depth-chart prominence).
            for p in players[:top_n_per_team]:
                pos_info = p.get("primaryPosition", {})
                pos_abbr = pos_info.get("abbreviation", "Util")
                pos_type = pos_info.get("type", "")
                is_hitter = pos_type != "Pitcher" and pos_abbr != "P"
                all_rows.append(
                    {
                        "mlb_id": p.get("id"),
                        "name": p.get("fullName", ""),
                        "team": team_map.get(team_id, ""),
                        "positions": pos_abbr if pos_abbr not in ("0", "-", "") else "Util",
                        "is_hitter": is_hitter,
                        "bats": p.get("batSide", {}).get("code", ""),
                        "throws": p.get("pitchHand", {}).get("code", ""),
                        "birth_date": p.get("birthDate", ""),
                        "level": level,
                    }
                )

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
```

- [ ] **Step 2.4: Run tests (should PASS)**

```bash
python -m pytest tests/test_wave9_minor_league_fetch.py -q 2>&1 | tail -10
```

- [ ] **Step 2.5: Lint + commit**

```bash
python -m ruff format src/live_stats.py tests/test_wave9_minor_league_fetch.py
python -m ruff check src/live_stats.py tests/test_wave9_minor_league_fetch.py
git add src/live_stats.py tests/test_wave9_minor_league_fetch.py
git commit -m "feat(wave9): fetch_minor_league_players from MLB Stats API (INFRA-F5 Task 2)"
```

---

## Task 3: Bootstrap phase `_bootstrap_minor_league_rosters`

**Files:**
- Modify: `src/data_bootstrap.py` (add new function, wire into orchestrator)
- Modify: `src/database.py` (`upsert_player_bulk` already handles `level` if it's a column; verify SELECT path)
- Test: `tests/test_wave9_bootstrap_phase.py` (create)

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_wave9_bootstrap_phase.py`:

```python
"""Wave 9 / Task 3: _bootstrap_minor_league_rosters phase."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    from src.database import init_db
    init_db()
    return db_path


def test_bootstrap_minor_league_inserts_players_with_level(temp_db):
    """The phase writes rows with level='AAA'/'AA' and updates refresh_log."""
    from src.data_bootstrap import BootstrapProgress, _bootstrap_minor_league_rosters
    from src.database import get_connection

    fake_df = pd.DataFrame([
        {"mlb_id": 700001, "name": "AAA Prospect 1", "team": "SCR", "positions": "OF",
         "is_hitter": True, "bats": "R", "throws": "R", "birth_date": "2002-05-01",
         "level": "AAA"},
        {"mlb_id": 700002, "name": "AA Prospect 1", "team": "SOM", "positions": "SP",
         "is_hitter": False, "bats": "R", "throws": "R", "birth_date": "2003-08-15",
         "level": "AA"},
    ])

    with patch("src.live_stats.fetch_minor_league_players", return_value=fake_df):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_minor_league_rosters(progress)

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, team, positions, level FROM players WHERE level IN ('AAA', 'AA')"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert {r[3] for r in rows} == {"AAA", "AA"}
    # Result message describes what happened
    assert "2" in result or "minor" in result.lower()


def test_bootstrap_minor_league_handles_empty_response(temp_db):
    """Empty DataFrame from fetch → refresh_log logs 'no_data', no crash."""
    from src.data_bootstrap import BootstrapProgress, _bootstrap_minor_league_rosters
    from src.database import get_connection

    with patch("src.live_stats.fetch_minor_league_players", return_value=pd.DataFrame()):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_minor_league_rosters(progress)

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM refresh_log WHERE source = 'minor_league_rosters'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] in ("no_data", "error")
```

- [ ] **Step 3.2: Run tests (should FAIL — phase not defined)**

```bash
python -m pytest tests/test_wave9_bootstrap_phase.py -q 2>&1 | tail -10
```

- [ ] **Step 3.3: Implement the phase**

Add to `src/data_bootstrap.py` next to `_bootstrap_players` (after the function, around line 282):

```python
def _bootstrap_minor_league_rosters(progress: BootstrapProgress) -> str:
    """Wave 9 INFRA-F5: fetch AAA/AA rosters and upsert with level column.

    Adds ~900 minor-league players (top 30 per affiliate × ~60 affiliates,
    de-duped on mlb_id) to the player universe. These players lack Yahoo
    ownership data — consumers (UI filters) must handle NULL percent_owned.
    """
    from src.database import (
        update_refresh_log,
        update_refresh_log_auto,
        upsert_player_bulk,
    )
    from src.live_stats import fetch_minor_league_players

    progress.phase = "Minor League Rosters"
    progress.detail = "Fetching AAA + AA rosters..."
    try:
        df = fetch_minor_league_players(season=2026, levels=("AAA", "AA"), top_n_per_team=30)
        if df.empty:
            update_refresh_log(
                "minor_league_rosters",
                "no_data",
                rows_written=0,
                message="fetch_minor_league_players returned empty",
            )
            return "Skipped: minor-league API returned no data"
        players = df.to_dict("records")
        count = upsert_player_bulk(players)
        status = update_refresh_log_auto(
            "minor_league_rosters",
            count,
            expected_min=500,  # ~900 expected; 500 floor allows for partial AAA-only success
            message=f"{count} minor leaguers upserted from {len(df)} API rows",
        )
        return f"Saved {count} minor leaguers ({status})"
    except Exception as e:
        update_refresh_log("minor_league_rosters", "error", message=str(e)[:200])
        return f"Error: {e}"
```

⚠️ Verify `upsert_player_bulk` handles the `level` key. Read `src/database.py` definition (search for `def upsert_player_bulk`). If it uses explicit column list, add `level` to the INSERT. If it uses dict-keys-as-columns, no change needed.

- [ ] **Step 3.4: Wire into orchestrator**

In `src/data_bootstrap.py`, find `bootstrap_all_data` function. Find a sensible insertion point after `_bootstrap_players` (around line 2820 — check actual location). Add:

```python
# Phase: Minor league rosters (Wave 9 INFRA-F5)
_notify(0.06)  # adjust to fit existing progress curve
if force or check_staleness("minor_league_rosters", 168.0):  # 7-day staleness
    results["minor_league_rosters"] = _bootstrap_minor_league_rosters(progress)
else:
    results["minor_league_rosters"] = "Fresh"
```

- [ ] **Step 3.5: Run tests (should PASS)**

```bash
python -m pytest tests/test_wave9_bootstrap_phase.py -q 2>&1 | tail -10
```

- [ ] **Step 3.6: Lint + commit**

```bash
python -m ruff format src/data_bootstrap.py tests/test_wave9_bootstrap_phase.py
python -m ruff check src/data_bootstrap.py tests/test_wave9_bootstrap_phase.py
git add src/data_bootstrap.py tests/test_wave9_bootstrap_phase.py
git commit -m "feat(wave9): _bootstrap_minor_league_rosters phase (INFRA-F5 Task 3)"
```

---

## Task 4: Player pool exposes `level`

**Files:**
- Modify: `src/database.py` `_build_player_pool` (verify `level` is in the SELECT)
- Test: `tests/test_wave9_pool_includes_level.py` (create)

- [ ] **Step 4.1: Write failing test**

Create `tests/test_wave9_pool_includes_level.py`:

```python
"""Wave 9 / Task 4: load_player_pool() returns level column."""

import pytest


@pytest.fixture
def temp_db_with_players(tmp_path, monkeypatch):
    import sqlite3
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    from src.database import init_db
    init_db()
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO players (mlb_id, name, team, positions, is_hitter, level) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (700001, "MLB Player", "NYY", "OF", 1, "MLB"),
                (700002, "AAA Prospect", "SCR", "OF", 1, "AAA"),
                (700003, "AA Prospect", "SOM", "SP", 0, "AA"),
                (700004, "Legacy Player", "BOS", "1B", 1, None),  # legacy, level NULL
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_player_pool_exposes_level_column(temp_db_with_players):
    """load_player_pool returns a `level` column with values from players.level."""
    from src.database import load_player_pool
    pool = load_player_pool()
    assert "level" in pool.columns, (
        f"Wave 9 regression: load_player_pool() missing 'level' column. "
        f"Found: {list(pool.columns)}"
    )
    # Spot-check: AAA player has level='AAA'
    aaa = pool[pool["mlb_id"] == 700002]
    if not aaa.empty:
        assert aaa.iloc[0]["level"] == "AAA"
```

- [ ] **Step 4.2: Run test (PASS if `_build_player_pool` already uses `SELECT *`; FAIL if it has an explicit column list)**

```bash
python -m pytest tests/test_wave9_pool_includes_level.py -q 2>&1 | tail -10
```

- [ ] **Step 4.3: Fix if needed**

Read `src/database.py` `_build_player_pool` function. If it has an explicit `SELECT name, team, positions, ...` list, add `level` to the SELECT. If it uses `SELECT *` or `SELECT p.*`, no change needed.

If a fix is required, add `p.level` to the SELECT statement (with appropriate alias-prefix matching surrounding columns).

- [ ] **Step 4.4: Run test (PASS)**

- [ ] **Step 4.5: Lint + commit**

```bash
python -m ruff format src/database.py tests/test_wave9_pool_includes_level.py
python -m ruff check src/database.py tests/test_wave9_pool_includes_level.py
git add src/database.py tests/test_wave9_pool_includes_level.py
git commit -m "feat(wave9): player pool exposes level column (INFRA-F5 Task 4)"
```

---

## Task 5: Free Agents page — level filter

**Files:**
- Modify: `pages/5_Free_Agents.py` (or whatever the actual filename is — verify by `ls pages/ | grep -i free`)
- Test: `tests/test_wave9_free_agents_level_filter.py` (create)

- [ ] **Step 5.1: Identify the filename**

```bash
ls pages/ | grep -i free
```

The convention in this repo uses numeric-prefixed names. The Free Agents page exists; use that exact filename in the commands below.

- [ ] **Step 5.2: Write the structural test**

Create `tests/test_wave9_free_agents_level_filter.py`:

```python
"""Wave 9 / Task 5: Free Agents page exposes a level filter."""

from pathlib import Path


def test_free_agents_page_has_level_selectbox():
    """The Free Agents page must include a Streamlit selectbox or radio for
    filtering by player level (MLB / AAA / AA / All). This guards that
    Wave 9 minor leaguers don't silently appear in default FA lists when
    users expect MLB-only."""
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    matches = list(pages_dir.glob("*Free_Agents*.py"))
    assert matches, "Free Agents page not found in pages/"

    text = matches[0].read_text(encoding="utf-8")
    # Look for either a selectbox or a radio with "level" + ("MLB"|"AAA"|"AA"|"All")
    assert ("level" in text.lower()) and (
        '"AAA"' in text or '"AA"' in text or 'selectbox' in text.lower()
    ), (
        f"Wave 9 regression: Free Agents page {matches[0].name} appears "
        "to lack a level filter for MLB/AAA/AA. Add a selectbox or radio."
    )
```

- [ ] **Step 5.3: Run test (should FAIL)**

```bash
python -m pytest tests/test_wave9_free_agents_level_filter.py -q 2>&1 | tail -10
```

- [ ] **Step 5.4: Add the level filter**

Read the Free Agents page. Find where the player pool is loaded (typically `pool = load_player_pool()` or similar). After the pool load and BEFORE display, add:

```python
# Wave 9 INFRA-F5: filter by player level (MLB / AAA / AA / All)
# Minor leaguers lack Yahoo ownership data, so default to MLB-only.
_LEVEL_OPTIONS = ["MLB only", "MLB + AAA", "MLB + AAA + AA", "All"]
_level_filter = st.selectbox(
    "Player universe",
    _LEVEL_OPTIONS,
    index=0,
    help="MLB-only is the default. Expanding to AAA/AA shows minor-league "
    "depth-chart candidates but they lack Yahoo ownership data.",
)

# Apply filter
if _level_filter == "MLB only":
    pool = pool[(pool.get("level").isna()) | (pool.get("level") == "MLB")]
elif _level_filter == "MLB + AAA":
    pool = pool[(pool.get("level").isna()) | (pool.get("level").isin(["MLB", "AAA"]))]
elif _level_filter == "MLB + AAA + AA":
    pool = pool[(pool.get("level").isna()) | (pool.get("level").isin(["MLB", "AAA", "AA"]))]
# else "All" → no filter
```

⚠️ Use the actual `st` import name from the page (might be `import streamlit as st`).

- [ ] **Step 5.5: Run test (PASS)**

```bash
python -m pytest tests/test_wave9_free_agents_level_filter.py -q 2>&1 | tail -10
```

- [ ] **Step 5.6: Lint + commit**

```bash
python -m ruff format pages/ tests/test_wave9_free_agents_level_filter.py
python -m ruff check pages/ tests/test_wave9_free_agents_level_filter.py
git add pages/ tests/test_wave9_free_agents_level_filter.py
git commit -m "feat(wave9): Free Agents level filter (INFRA-F5 Task 5)"
```

---

## Task 6: Player Compare page — level filter

**Files:**
- Modify: `pages/6_Player_Compare.py` (or the actual filename)
- Test: `tests/test_wave9_player_compare_level_filter.py` (create)

- [ ] **Step 6.1: Identify filename**

```bash
ls pages/ | grep -i compare
```

- [ ] **Step 6.2: Write the structural test**

Create `tests/test_wave9_player_compare_level_filter.py`:

```python
"""Wave 9 / Task 6: Player Compare page exposes a level filter."""

from pathlib import Path


def test_player_compare_page_has_level_filter():
    """Player Compare must support filtering the candidate-player picker
    by level so users don't accidentally compare an MLB regular against
    an AAA prospect with no Yahoo ownership context."""
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    matches = list(pages_dir.glob("*Player_Compare*.py"))
    assert matches, "Player Compare page not found in pages/"

    text = matches[0].read_text(encoding="utf-8")
    assert ("level" in text.lower()) and (
        '"AAA"' in text or '"AA"' in text or 'selectbox' in text.lower()
    ), (
        f"Wave 9 regression: Player Compare page {matches[0].name} "
        "lacks a level filter."
    )
```

- [ ] **Step 6.3: Run test (FAIL)**

```bash
python -m pytest tests/test_wave9_player_compare_level_filter.py -q 2>&1 | tail -10
```

- [ ] **Step 6.4: Add the filter**

Same pattern as Task 5 — add a `selectbox` for level filter near the top of the page (after pool load, before player picker).

- [ ] **Step 6.5: Run test (PASS)**

- [ ] **Step 6.6: Lint + commit**

```bash
python -m ruff format pages/ tests/test_wave9_player_compare_level_filter.py
python -m ruff check pages/ tests/test_wave9_player_compare_level_filter.py
git add pages/ tests/test_wave9_player_compare_level_filter.py
git commit -m "feat(wave9): Player Compare level filter (INFRA-F5 Task 6)"
```

---

## Task 7: CLAUDE.md update

- [ ] **Step 7.1: Cumulative structural-invariant sweep**

```bash
python -m pytest tests/test_no_*.py tests/test_pages_*.py tests/test_wave*_*.py tests/test_engine_*.py -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 7.2: Update CLAUDE.md**

Find the "Data Audit History" section. After the Wave 8d paragraph, add:

```markdown
**2026-05-12 Wave 9 (SF-78..SF-83) — PLAYER UNIVERSE EXPANSION (INFRA-F5, Option B).** Expanded the player universe from ~1,300 (40-man + spring training MLB) to ~2,200 by adding top-30 per AAA + AA affiliate (~+900 rows). SF-78 (`players.level` column): added via `_safe_add_column` migration — NULL/"MLB"/"AAA"/"AA". SF-79 (`fetch_minor_league_players`): MLB Stats API `sports_players` with sportId=11 (AAA) and sportId=12 (AA), capped at top-30 per team. SF-80 (`_bootstrap_minor_league_rosters`): new bootstrap phase wired into orchestrator with 7-day staleness, expected_min=500. SF-81 (player pool): `level` column flows through `_build_player_pool`. SF-82 (Free Agents page): level selectbox defaulting to "MLB only" — minor leaguers visible on demand. SF-83 (Player Compare page): same level selectbox. Yahoo ownership data is unavailable for non-MLB players — `percent_owned` stays NULL and the UI displays accordingly.
```

Also add 6 rows to the Structural Invariants table near the Wave 8d additions:

```markdown
| `test_wave9_level_column_migration.py` | players.level column exists after init_db (SF-78) |
| `test_wave9_minor_league_fetch.py` | fetch_minor_league_players caps at top_n_per_team, handles failures (SF-79) |
| `test_wave9_bootstrap_phase.py` | _bootstrap_minor_league_rosters writes to DB + refresh_log (SF-80) |
| `test_wave9_pool_includes_level.py` | load_player_pool exposes level column (SF-81) |
| `test_wave9_free_agents_level_filter.py` | Free Agents page has level filter (SF-82) |
| `test_wave9_player_compare_level_filter.py` | Player Compare page has level filter (SF-83) |
```

- [ ] **Step 7.3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(wave9): CLAUDE.md history for INFRA-F5 (SF-78..SF-83)"
```

---

## Phase Final: Push + PR + Merge

- [ ] **Step F.1: Push**

```bash
git push -u origin claude/audit-wave9-player-universe
```

- [ ] **Step F.2: Create PR**

```bash
gh pr create --title "Wave 9: player universe expansion (INFRA-F5 Option B — AAA/AA top-30)" --body "..."
```

PR body should include:
- Per-task summary (Tasks 1-7)
- Schema impact (~+900 rows, +1 column on players)
- Yahoo-data caveat (no ownership for minors)
- Test count (~10 new tests)
- Default behavior (MLB-only filter; minors visible on opt-in)

- [ ] **Step F.3: Wait for CI, merge when green**

```bash
gh pr merge --merge --delete-branch
```

- [ ] **Step F.4: Sync master + cleanup worktree**

---

## Self-Review

**Spec coverage:**
- Schema change (Task 1) ✅
- API integration (Task 2) ✅
- Bootstrap orchestrator (Task 3) ✅
- Pool integration (Task 4) ✅
- UI filter on FA + Player Compare (Tasks 5-6) ✅
- Documentation (Task 7) ✅

**Placeholder scan:** No "TBD" / "TODO" / "implement later". All code blocks complete with exact code.

**Type consistency:**
- `level` column: TEXT, NULL-able, values `"MLB"`/`"AAA"`/`"AA"` (consistent across all 7 tasks)
- `fetch_minor_league_players(season, levels, top_n_per_team)` signature consistent in Task 2 spec + Task 3 caller
- `_bootstrap_minor_league_rosters` referenced consistently in Tasks 3 → 7

**Cold-start usability:** Worktree path, branch, HEAD, all imports specified. Sport IDs documented. Yahoo caveat called out.

**Granularity:** Each task is 4-6 steps of 2-5 min each. Total ~6 commits + final PR.

**Out-of-scope (intentionally deferred):**
- Yahoo-style "ownership %" simulation for minor leaguers (no data source)
- Prospect ranking enrichment via FanGraphs (separate from universe expansion)
- Player comparison views with minor-league context-aware metrics

These can be Wave 10+ if you want.
