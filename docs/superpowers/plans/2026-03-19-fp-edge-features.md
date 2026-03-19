# FantasyPros Edge Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform HEATER's 3 weakest FantasyPros-parity features from checkbox parity into genuine competitive advantages: (1) live Prospect Rankings Engine with FanGraphs Board API + MLB Stats API MiLB stats + computed MLB Readiness Score, (2) Multi-Platform ECR Consensus from 7 ranking sources with Trimmed Borda Count, (3) News/Transaction Intelligence from 4 sources with template-based analytical summaries.

**Architecture:** Three new/rewritten source modules (`src/prospect_engine.py`, `src/ecr.py` rewrite, `src/player_news.py`) plus modifications to `src/database.py` (5 new tables), `src/data_bootstrap.py` (3 new phases 13-15), `src/live_stats.py` (enhanced status endpoint), `src/yahoo_api.py` (injury/ownership fields). Each feature is self-contained with its own test file. DB schema is the shared foundation — implemented first.

**Tech Stack:** Python 3.11-3.14 (CI: 3.11-3.13, local: 3.14), SQLite, pandas, requests, BeautifulSoup (existing), feedparser (new dep), MLB-StatsAPI (existing), yfpy (existing), scipy.stats (existing)

---

## File Structure

### New Source Files (2)
| File | Responsibility |
|------|---------------|
| `src/prospect_engine.py` | FanGraphs Board API fetch, MLB Stats API MiLB stats, MLB Readiness Score computation, fallback chain |
| `src/player_news.py` | Multi-source news aggregation (MLB enhanced + ESPN + RotoWire RSS + Yahoo), template-based analytical summaries, ownership trends |

### Rewritten Source Files (1)
| File | Responsibility |
|------|---------------|
| `src/ecr.py` | Full rewrite: 7-source consensus builder (ESPN + Yahoo + CBS + NFBC + FG + FP + HEATER SGP), Trimmed Borda Count, player ID cross-ref, disagreement detection |

### New Test Files (3)
| File | Tests |
|------|-------|
| `tests/test_prospect_engine.py` | ~17 tests |
| `tests/test_ecr_consensus.py` | ~32 tests (replaces `tests/test_ecr.py`) |
| `tests/test_player_news.py` | ~27 tests |

### Rewritten Test Files (2)
| File | Change |
|------|--------|
| `tests/test_ecr.py` | Renamed to `tests/test_ecr_consensus.py`, fully rewritten |
| `tests/test_prospect_rankings.py` | Rewritten to test `src/prospect_engine.py` |

### Modified Files (8)
| File | Changes |
|------|---------|
| `src/database.py` | 5 new tables in `init_db()`: `prospect_rankings`, `ecr_consensus`, `player_id_map`, `player_news`, `ownership_trends` |
| `src/data_bootstrap.py` | 3 new phases (13-15), 3 new `StalenessConfig` fields |
| `src/live_stats.py` | Add `fetch_player_enhanced_status()` |
| `src/yahoo_api.py` | Extract `injury_note`, `status_full`, `percent_owned` during sync |
| `requirements.txt` | Add `feedparser` |
| `pages/9_Leaders.py` | Enhanced Prospects tab with Plotly radar chart |
| `pages/1_My_Team.py` | News & Alerts tab |
| `app.py` | Consensus Rank column + disagreement badges + Available Players consensus sort/filter |

### Unchanged Files (notable)
| File | Note |
|------|------|
| `src/news_fetcher.py` | **Unchanged** — `player_news.py` imports `fetch_recent_transactions()` from it; do NOT modify |

---

## Chunk 1: Database Foundation + Prospect Engine

**Features:** DB schema for all 3 features, Prospect Rankings Engine
**Rationale:** DB tables are the shared foundation — every other feature depends on them. Prospect Engine is self-contained (no cross-dependencies with ECR or News).

### Task 1: Database Schema — 5 New Tables

**Files:**
- Modify: `src/database.py` (add 5 tables + indexes inside `init_db()` at line 249, before the closing `""")`)

- [ ] **Step 1: Write failing test for new tables**

```python
# tests/test_fp_edge_schema.py
"""Tests for FP Edge feature database tables."""
import sqlite3
import pytest
from unittest.mock import patch

@pytest.fixture
def _temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        yield db_path

def _table_exists(db_path, table_name):
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def _get_columns(db_path, table_name):
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    conn.close()
    return cols

def test_prospect_rankings_table_exists(_temp_db):
    assert _table_exists(_temp_db, "prospect_rankings")

def test_prospect_rankings_columns(_temp_db):
    cols = _get_columns(_temp_db, "prospect_rankings")
    for expected in ["prospect_id", "mlb_id", "name", "fg_rank", "fg_fv",
                     "readiness_score", "milb_avg", "milb_era", "age"]:
        assert expected in cols, f"Missing column: {expected}"

def test_ecr_consensus_table_exists(_temp_db):
    assert _table_exists(_temp_db, "ecr_consensus")

def test_ecr_consensus_columns(_temp_db):
    cols = _get_columns(_temp_db, "ecr_consensus")
    for expected in ["player_id", "espn_rank", "yahoo_adp", "consensus_rank",
                     "consensus_avg", "rank_stddev", "n_sources"]:
        assert expected in cols, f"Missing column: {expected}"

def test_player_id_map_table_exists(_temp_db):
    assert _table_exists(_temp_db, "player_id_map")

def test_player_id_map_columns(_temp_db):
    cols = _get_columns(_temp_db, "player_id_map")
    for expected in ["player_id", "espn_id", "yahoo_key", "fg_id", "mlb_id"]:
        assert expected in cols, f"Missing column: {expected}"

def test_player_news_table_exists(_temp_db):
    assert _table_exists(_temp_db, "player_news")

def test_player_news_columns(_temp_db):
    cols = _get_columns(_temp_db, "player_news")
    for expected in ["player_id", "source", "headline", "news_type",
                     "il_status", "sentiment_score"]:
        assert expected in cols, f"Missing column: {expected}"

def test_ownership_trends_table_exists(_temp_db):
    assert _table_exists(_temp_db, "ownership_trends")

def test_ownership_trends_columns(_temp_db):
    cols = _get_columns(_temp_db, "ownership_trends")
    for expected in ["player_id", "date", "percent_owned", "delta_7d"]:
        assert expected in cols, f"Missing column: {expected}"

def test_player_id_map_unique_indexes(_temp_db):
    """Verify filtered unique indexes exist on player_id_map."""
    conn = sqlite3.connect(str(_temp_db))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_id_map'")
    indexes = [row[0] for row in cur.fetchall()]
    conn.close()
    assert "idx_id_map_espn" in indexes
    assert "idx_id_map_yahoo" in indexes
    assert "idx_id_map_fg" in indexes
    assert "idx_id_map_mlb" in indexes

def test_player_news_indexes(_temp_db):
    conn = sqlite3.connect(str(_temp_db))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_news'")
    indexes = [row[0] for row in cur.fetchall()]
    conn.close()
    assert "idx_player_news_player" in indexes
    assert "idx_player_news_type" in indexes
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_fp_edge_schema.py -v`
Expected: FAIL — tables don't exist yet

- [ ] **Step 3: Add 5 tables to database.py init_db()**

In `src/database.py`, inside `init_db()`, just before the line `    """)` (line 250), insert:

```sql
        CREATE TABLE IF NOT EXISTS prospect_rankings (
            prospect_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mlb_id INTEGER,
            name TEXT NOT NULL,
            team TEXT,
            position TEXT,
            fg_rank INTEGER,
            fg_fv INTEGER,
            fg_eta TEXT,
            fg_risk TEXT,
            age INTEGER,
            hit_present INTEGER, hit_future INTEGER,
            game_present INTEGER, game_future INTEGER,
            raw_present INTEGER, raw_future INTEGER,
            speed INTEGER, field INTEGER,
            ctrl_present INTEGER, ctrl_future INTEGER,
            scouting_report TEXT,
            tldr TEXT,
            milb_level TEXT,
            milb_avg REAL, milb_obp REAL, milb_slg REAL,
            milb_k_pct REAL, milb_bb_pct REAL, milb_hr INTEGER, milb_sb INTEGER,
            milb_ip REAL, milb_era REAL, milb_whip REAL, milb_k9 REAL, milb_bb9 REAL,
            readiness_score REAL,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS ecr_consensus (
            player_id INTEGER PRIMARY KEY,
            espn_rank INTEGER,
            yahoo_adp REAL,
            cbs_rank INTEGER,
            nfbc_adp REAL,
            fg_adp REAL,
            fp_ecr INTEGER,
            heater_sgp_rank INTEGER,
            consensus_rank INTEGER,
            consensus_avg REAL,
            rank_min INTEGER,
            rank_max INTEGER,
            rank_stddev REAL,
            n_sources INTEGER,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS player_id_map (
            player_id INTEGER PRIMARY KEY,
            espn_id INTEGER,
            yahoo_key TEXT,
            fg_id INTEGER,
            mlb_id INTEGER,
            cbs_id INTEGER,
            nfbc_id INTEGER,
            name TEXT,
            team TEXT,
            updated_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_espn ON player_id_map(espn_id) WHERE espn_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_yahoo ON player_id_map(yahoo_key) WHERE yahoo_key IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_fg ON player_id_map(fg_id) WHERE fg_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_mlb ON player_id_map(mlb_id) WHERE mlb_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS player_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            headline TEXT NOT NULL,
            detail TEXT,
            news_type TEXT,
            injury_body_part TEXT,
            il_status TEXT,
            sentiment_score REAL,
            published_at TEXT,
            fetched_at TEXT,
            UNIQUE(player_id, source, headline, published_at)
        );
        CREATE INDEX IF NOT EXISTS idx_player_news_player ON player_news(player_id);
        CREATE INDEX IF NOT EXISTS idx_player_news_type ON player_news(news_type);

        CREATE TABLE IF NOT EXISTS ownership_trends (
            player_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            percent_owned REAL,
            delta_7d REAL,
            PRIMARY KEY (player_id, date)
        );
```

**Important:** The 5 new `CREATE TABLE` statements go INSIDE the existing `conn.executescript("""...""")` block, before the closing `""")` at line 250. The `CREATE INDEX` and `CREATE UNIQUE INDEX` statements for `player_id_map` and `player_news` also go inside the same executescript block.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_fp_edge_schema.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest --tb=short -q`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/database.py tests/test_fp_edge_schema.py
git commit -m "feat: add 5 DB tables for prospect rankings, ECR consensus, player news, ID map, ownership"
```

---

### Task 2: Prospect Engine — Core Module

**Files:**
- Create: `src/prospect_engine.py`
- Create: `tests/test_prospect_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prospect_engine.py
"""Tests for prospect rankings engine."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Readiness score tests ────────────────────────────────────────────

def test_fv_normalized_bounds():
    from src.prospect_engine import _fv_normalized
    assert _fv_normalized(80) == 100.0
    assert _fv_normalized(20) == 0.0
    assert 0 <= _fv_normalized(50) <= 100

def test_fv_normalized_clamps():
    from src.prospect_engine import _fv_normalized
    assert _fv_normalized(10) == 0.0   # below min
    assert _fv_normalized(90) == 100.0  # above max

def test_eta_proximity_current_year():
    from src.prospect_engine import _eta_proximity
    assert _eta_proximity("2026") == 100.0

def test_eta_proximity_future():
    from src.prospect_engine import _eta_proximity
    assert _eta_proximity("2027") == 75.0
    assert _eta_proximity("2028") == 50.0
    assert _eta_proximity("2029") == 25.0
    assert _eta_proximity("2030") == 0.0

def test_eta_proximity_past():
    from src.prospect_engine import _eta_proximity
    assert _eta_proximity("2025") == 100.0  # already MLB-ready

def test_eta_proximity_empty():
    from src.prospect_engine import _eta_proximity
    assert _eta_proximity("") == 50.0  # unknown defaults to mid

def test_risk_factor():
    from src.prospect_engine import _risk_factor
    assert _risk_factor("Low") == 1.0
    assert _risk_factor("Medium") == 0.8
    assert _risk_factor("High") == 0.6
    assert _risk_factor("Extreme") == 0.4
    assert _risk_factor("Unknown") == 0.7  # default

def test_readiness_score_bounds():
    from src.prospect_engine import compute_mlb_readiness_score
    row = {
        "fg_fv": 60, "fg_eta": "2026", "fg_risk": "Medium",
        "milb_obp": 0.350, "milb_slg": 0.500, "milb_level": "AAA",
        "age": 23,
    }
    score = compute_mlb_readiness_score(row)
    assert 0 <= score <= 100

def test_readiness_score_high_fv_near_eta():
    from src.prospect_engine import compute_mlb_readiness_score
    elite = {
        "fg_fv": 70, "fg_eta": "2026", "fg_risk": "Low",
        "milb_obp": 0.400, "milb_slg": 0.550, "milb_level": "AAA",
        "age": 22,
    }
    mediocre = {
        "fg_fv": 40, "fg_eta": "2029", "fg_risk": "High",
        "milb_obp": 0.300, "milb_slg": 0.380, "milb_level": "A",
        "age": 20,
    }
    assert compute_mlb_readiness_score(elite) > compute_mlb_readiness_score(mediocre)

def test_readiness_score_missing_milb():
    from src.prospect_engine import compute_mlb_readiness_score
    row = {
        "fg_fv": 55, "fg_eta": "2027", "fg_risk": "Medium",
        "milb_obp": None, "milb_slg": None, "milb_level": None,
        "age": None,
    }
    score = compute_mlb_readiness_score(row)
    assert 0 <= score <= 100  # should still compute without MiLB data


# ── FanGraphs API parsing tests ──────────────────────────────────────

MOCK_FG_RESPONSE = [
    {
        "PlayerName": "Prospect Alpha",
        "Team": "NYY",
        "Position": "SS",
        "FV": "60",
        "ETA": "2026",
        "Risk": "Medium",
        "minorMasterId": 12345,
        "pHit": 55, "fHit": 60,
        "pGame": 50, "fGame": 55,
        "pRaw": 45, "fRaw": 50,
        "pSpd": 55, "Fld": 50,
        "Age": 21,
    },
    {
        "PlayerName": "Prospect Beta",
        "Team": "LAD",
        "Position": "SP",
        "FV": "55",
        "ETA": "2027",
        "Risk": "High",
        "minorMasterId": 67890,
        "pCtl": 45, "fCtl": 55,
        "pHit": None, "fHit": None,
        "pGame": None, "fGame": None,
        "Age": 19,
    },
]

def test_parse_fg_response():
    from src.prospect_engine import _parse_fg_prospects
    df = _parse_fg_prospects(MOCK_FG_RESPONSE)
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Prospect Alpha"
    assert df.iloc[0]["fg_fv"] == 60
    assert df.iloc[0]["hit_present"] == 55
    assert df.iloc[0]["hit_future"] == 60

def test_parse_fg_pitcher_fields():
    from src.prospect_engine import _parse_fg_prospects
    df = _parse_fg_prospects(MOCK_FG_RESPONSE)
    pitcher = df[df["name"] == "Prospect Beta"].iloc[0]
    assert pitcher["ctrl_present"] == 45
    assert pitcher["ctrl_future"] == 55


# ── Filtering tests ──────────────────────────────────────────────────

def test_get_prospect_rankings_top_n():
    from src.prospect_engine import get_prospect_rankings
    # Uses fallback static list when DB empty and FG unavailable
    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame()
        with patch("src.prospect_engine.refresh_prospect_rankings") as mock_refresh:
            mock_refresh.return_value = pd.DataFrame()
            df = get_prospect_rankings(top_n=5)
            # Should fall back to static list
            assert len(df) <= 20  # static list has 20

def test_filter_by_position():
    from src.prospect_engine import get_prospect_rankings
    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame([
            {"name": "A", "position": "SS", "readiness_score": 80, "fg_rank": 1},
            {"name": "B", "position": "SP", "readiness_score": 70, "fg_rank": 2},
            {"name": "C", "position": "OF", "readiness_score": 60, "fg_rank": 3},
        ])
        df = get_prospect_rankings(position="SP")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "B"

def test_filter_by_org():
    from src.prospect_engine import get_prospect_rankings
    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame([
            {"name": "A", "team": "NYY", "position": "SS", "readiness_score": 80, "fg_rank": 1},
            {"name": "B", "team": "LAD", "position": "SP", "readiness_score": 70, "fg_rank": 2},
        ])
        df = get_prospect_rankings(org="NYY")
        assert len(df) == 1


# ── DB round-trip: get_prospect_detail() ─────────────────────────────

def test_get_prospect_detail_round_trip(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.prospect_engine import _store_prospects, get_prospect_detail
        df = pd.DataFrame([{
            "mlb_id": 12345, "name": "Test Prospect", "team": "NYY",
            "position": "SS", "fg_rank": 1, "fg_fv": 60, "fg_eta": "2026",
            "fg_risk": "Medium", "age": 21, "readiness_score": 72.5,
            "hit_present": 55, "hit_future": 60,
        }])
        _store_prospects(df)
        detail = get_prospect_detail(1)  # prospect_id=1 (first inserted)
        assert detail is not None
        assert detail["name"] == "Test Prospect"
        assert detail["fg_fv"] == 60

def test_get_prospect_detail_missing(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.prospect_engine import get_prospect_detail
        assert get_prospect_detail(999) is None
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_prospect_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prospect_engine'`

- [ ] **Step 3: Implement src/prospect_engine.py**

```python
# src/prospect_engine.py
"""Prospect Rankings Engine — FanGraphs Board API + MLB Stats API MiLB stats.

Computes MLB Readiness Score (0-100) combining FV, age-level performance,
ETA proximity, and risk factor.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_CURRENT_SEASON = 2026

_LEVEL_AVG_WOBA = {"AAA": 0.330, "AA": 0.310, "High-A": 0.300, "A": 0.290, "A+": 0.300}
_LEVEL_AVG_AGE = {"AAA": 25, "AA": 23, "High-A": 22, "A": 21, "A+": 22}

_FG_BOARD_URL = (
    "https://www.fangraphs.com/api/prospects/board/data"
    "?draft=false&season={season}"
)
_MLB_MILB_STATS_URL = (
    "https://statsapi.mlb.com/api/v1/people/{mlb_id}"
    "/stats?stats=yearByYear&leagueListId=milb_all"
)

_RISK_MAP = {"Low": 1.0, "Medium": 0.8, "High": 0.6, "Extreme": 0.4}
_HEADERS = {"User-Agent": "HEATER-Fantasy-Tool/1.0"}

# ── Fallback static list (last resort) ──────────────────────────────

_STATIC_PROSPECTS = [
    {"rank": 1, "name": "Roki Sasaki", "team": "LAD", "position": "SP", "eta": "2025", "fv": 80},
    {"rank": 2, "name": "Roman Anthony", "team": "BOS", "position": "OF", "eta": "2025", "fv": 70},
    {"rank": 3, "name": "Travis Bazzana", "team": "CLE", "position": "2B", "eta": "2026", "fv": 65},
    {"rank": 4, "name": "Charlie Condon", "team": "COL", "position": "3B", "eta": "2027", "fv": 65},
    {"rank": 5, "name": "Jac Caglianone", "team": "KC", "position": "1B/SP", "eta": "2027", "fv": 65},
    {"rank": 6, "name": "Sebastian Walcott", "team": "TEX", "position": "SS", "eta": "2027", "fv": 65},
    {"rank": 7, "name": "Kristian Campbell", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
    {"rank": 8, "name": "Marcelo Mayer", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
    {"rank": 9, "name": "JJ Wetherholt", "team": "PIT", "position": "2B", "eta": "2026", "fv": 60},
    {"rank": 10, "name": "Coby Mayo", "team": "BAL", "position": "3B", "eta": "2025", "fv": 55},
    {"rank": 11, "name": "Nick Kurtz", "team": "OAK", "position": "1B", "eta": "2027", "fv": 60},
    {"rank": 12, "name": "James Wood", "team": "WSH", "position": "OF", "eta": "2025", "fv": 55},
    {"rank": 13, "name": "Bubba Chandler", "team": "PIT", "position": "SS/SP", "eta": "2026", "fv": 60},
    {"rank": 14, "name": "Chase Burns", "team": "CIN", "position": "SP", "eta": "2026", "fv": 60},
    {"rank": 15, "name": "Tink Hence", "team": "STL", "position": "SP", "eta": "2026", "fv": 55},
    {"rank": 16, "name": "Samuel Basallo", "team": "BAL", "position": "C", "eta": "2026", "fv": 55},
    {"rank": 17, "name": "Braden Montgomery", "team": "BOS", "position": "OF", "eta": "2027", "fv": 60},
    {"rank": 18, "name": "Leodalis De Vries", "team": "TEX", "position": "SS", "eta": "2028", "fv": 60},
    {"rank": 19, "name": "Colt Emerson", "team": "CLE", "position": "SS", "eta": "2028", "fv": 60},
    {"rank": 20, "name": "Ethan Salas", "team": "SD", "position": "C", "eta": "2026", "fv": 55},
]


# ── Readiness score components ──────────────────────────────────────

def _fv_normalized(fv: int | None) -> float:
    """Normalize FV (20-80 scale) to 0-100."""
    if fv is None:
        return 50.0
    return max(0.0, min(100.0, (fv - 20) / 60 * 100))


def _eta_proximity(eta: str | None) -> float:
    """Score ETA proximity (0-100). Current season or past = 100."""
    if not eta:
        return 50.0
    try:
        eta_year = int(eta)
    except (ValueError, TypeError):
        return 50.0
    diff = eta_year - _CURRENT_SEASON
    if diff <= 0:
        return 100.0
    mapping = {1: 75.0, 2: 50.0, 3: 25.0}
    return mapping.get(diff, 0.0)


def _risk_factor(risk: str | None) -> float:
    """Map risk level to multiplier."""
    if not risk:
        return 0.7
    return _RISK_MAP.get(risk, 0.7)


def _age_level_performance(row: dict) -> float:
    """Compute age-level performance score (0-100).
    wOBA proxy = (OBP * 1.2 + SLG * 0.8) / 2
    Bonus/penalty for age vs level average.
    """
    obp = row.get("milb_obp")
    slg = row.get("milb_slg")
    level = row.get("milb_level") or ""
    age = row.get("age")

    if obp is None or slg is None:
        return 50.0  # neutral when no MiLB data

    woba_proxy = (obp * 1.2 + slg * 0.8) / 2
    level_avg = _LEVEL_AVG_WOBA.get(level, 0.310)

    # Scale: each 0.020 above avg = +10 points from 50 baseline
    perf_score = 50.0 + (woba_proxy - level_avg) / 0.020 * 10.0

    # Age adjustment: younger than level avg = bonus
    if age is not None:
        level_age = _LEVEL_AVG_AGE.get(level, 23)
        age_bonus = (level_age - age) * 5.0  # +5 per year younger
        perf_score += age_bonus

    return max(0.0, min(100.0, perf_score))


def compute_mlb_readiness_score(row: dict) -> float:
    """Compute MLB Readiness Score (0-100).

    Formula (additive, per spec):
        0.40 * fv_normalized          (FV 80->100, FV 20->0)
      + 0.25 * age_level_performance  (wOBA vs level avg, age-adjusted, 0-100)
      + 0.20 * eta_proximity          (2026->100, 2027->75, 2028->50, 2029->25)
      + 0.15 * risk_factor_scaled     (Low->100, Med->80, High->60, Extreme->40)
    """
    fv = _fv_normalized(row.get("fg_fv"))
    perf = _age_level_performance(row)
    eta = _eta_proximity(row.get("fg_eta"))
    risk = _risk_factor(row.get("fg_risk"))

    # Additive: risk_factor is 0-1, scale to 0-100 for the 15% weight
    score = 0.40 * fv + 0.25 * perf + 0.20 * eta + 0.15 * (risk * 100)
    return round(max(0.0, min(100.0, score)), 1)


# ── FanGraphs Board API ─────────────────────────────────────────────

def _parse_fg_prospects(data: list[dict]) -> pd.DataFrame:
    """Parse FanGraphs Board API response into a DataFrame."""
    rows = []
    for item in data:
        try:
            fv_raw = item.get("FV", "0")
            fv = int(fv_raw) if fv_raw else 0
        except (ValueError, TypeError):
            fv = 0

        row = {
            "name": item.get("PlayerName", ""),
            "team": item.get("Team", ""),
            "position": item.get("Position", ""),
            "fg_rank": len(rows) + 1,
            "fg_fv": fv,
            "fg_eta": str(item.get("ETA", "")),
            "fg_risk": item.get("Risk", ""),
            "mlb_id": item.get("minorMasterId"),
            "age": item.get("Age"),
            "hit_present": item.get("pHit"),
            "hit_future": item.get("fHit"),
            "game_present": item.get("pGame"),
            "game_future": item.get("fGame"),
            "raw_present": item.get("pRaw"),
            "raw_future": item.get("fRaw"),
            "speed": item.get("pSpd"),
            "field": item.get("Fld"),
            "ctrl_present": item.get("pCtl"),
            "ctrl_future": item.get("fCtl"),
            "scouting_report": item.get("Report", ""),
            "tldr": item.get("TLDR", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_fangraphs_prospects(season: int = _CURRENT_SEASON) -> pd.DataFrame:
    """Fetch prospect data from FanGraphs Board API.
    Returns DataFrame or empty DataFrame on failure.
    """
    try:
        import requests
        url = _FG_BOARD_URL.format(season=season)
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("FanGraphs Board API returned empty/non-list response")
            return pd.DataFrame()
        return _parse_fg_prospects(data)
    except Exception:
        logger.warning("FanGraphs Board API fetch failed", exc_info=True)
        return pd.DataFrame()


# ── MLB Stats API MiLB stats ────────────────────────────────────────

def fetch_milb_stats(mlb_ids: list[int]) -> pd.DataFrame:
    """Fetch MiLB stats for a list of MLB IDs.
    Returns DataFrame with one row per player (most recent MiLB season).
    """
    rows = []
    for mlb_id in mlb_ids:
        if mlb_id is None:
            continue
        try:
            import requests
            url = _MLB_MILB_STATS_URL.format(mlb_id=mlb_id)
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            stats_list = data.get("stats", [])
            if not stats_list:
                continue
            # Find the most recent MiLB season
            splits = stats_list[0].get("splits", [])
            if not splits:
                continue
            latest = splits[-1]  # most recent
            stat = latest.get("stat", {})
            league = latest.get("league", {})
            level_name = league.get("name", "")
            # Determine if hitter or pitcher stats
            row = {
                "mlb_id": mlb_id,
                "milb_level": _normalize_level(level_name),
                "milb_avg": _safe_float(stat.get("avg")),
                "milb_obp": _safe_float(stat.get("obp")),
                "milb_slg": _safe_float(stat.get("slg")),
                "milb_hr": _safe_int(stat.get("homeRuns")),
                "milb_sb": _safe_int(stat.get("stolenBases")),
                "milb_k_pct": _compute_k_pct(stat),
                "milb_bb_pct": _compute_bb_pct(stat),
                "milb_ip": _safe_float(stat.get("inningsPitched")),
                "milb_era": _safe_float(stat.get("era")),
                "milb_whip": _safe_float(stat.get("whip")),
                "milb_k9": _safe_float(stat.get("strikeoutsPer9Inn")),
                "milb_bb9": _safe_float(stat.get("walksPer9Inn")),
            }
            rows.append(row)
            time.sleep(0.3)  # rate limit
        except Exception:
            logger.debug("MiLB stats fetch failed for mlb_id=%s", mlb_id, exc_info=True)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_level(level_name: str) -> str:
    name = level_name.lower()
    if "triple" in name or "aaa" in name or "international" in name or "pacific" in name:
        return "AAA"
    if "double" in name or "eastern" in name or "southern" in name or "texas" in name:
        return "AA"
    if "high" in name or "a+" in name:
        return "High-A"
    if "single" in name or "south atlantic" in name or "midwest" in name:
        return "A"
    return level_name


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _compute_k_pct(stat: dict) -> float | None:
    so = _safe_int(stat.get("strikeOuts"))
    pa = _safe_int(stat.get("plateAppearances")) or _safe_int(stat.get("atBats"))
    if so is not None and pa and pa > 0:
        return round(so / pa * 100, 1)
    return None


def _compute_bb_pct(stat: dict) -> float | None:
    bb = _safe_int(stat.get("baseOnBalls"))
    pa = _safe_int(stat.get("plateAppearances")) or _safe_int(stat.get("atBats"))
    if bb is not None and pa and pa > 0:
        return round(bb / pa * 100, 1)
    return None


# ── DB persistence ───────────────────────────────────────────────────

def _store_prospects(df: pd.DataFrame) -> int:
    """Store prospect rankings to DB. Returns count stored."""
    if df.empty:
        return 0
    from src.database import get_connection
    conn = get_connection()
    try:
        # Clear and re-insert
        conn.execute("DELETE FROM prospect_rankings")
        now = datetime.now(UTC).isoformat()
        count = 0
        for _, row in df.iterrows():
            conn.execute(
                """INSERT INTO prospect_rankings
                   (mlb_id, name, team, position, fg_rank, fg_fv, fg_eta, fg_risk,
                    age, hit_present, hit_future, game_present, game_future,
                    raw_present, raw_future, speed, field, ctrl_present, ctrl_future,
                    scouting_report, tldr, milb_level, milb_avg, milb_obp, milb_slg,
                    milb_k_pct, milb_bb_pct, milb_hr, milb_sb,
                    milb_ip, milb_era, milb_whip, milb_k9, milb_bb9,
                    readiness_score, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.get("mlb_id"), row.get("name"), row.get("team"),
                    row.get("position"), row.get("fg_rank"), row.get("fg_fv"),
                    row.get("fg_eta"), row.get("fg_risk"), row.get("age"),
                    row.get("hit_present"), row.get("hit_future"),
                    row.get("game_present"), row.get("game_future"),
                    row.get("raw_present"), row.get("raw_future"),
                    row.get("speed"), row.get("field"),
                    row.get("ctrl_present"), row.get("ctrl_future"),
                    row.get("scouting_report"), row.get("tldr"),
                    row.get("milb_level"), row.get("milb_avg"), row.get("milb_obp"),
                    row.get("milb_slg"), row.get("milb_k_pct"), row.get("milb_bb_pct"),
                    row.get("milb_hr"), row.get("milb_sb"),
                    row.get("milb_ip"), row.get("milb_era"), row.get("milb_whip"),
                    row.get("milb_k9"), row.get("milb_bb9"),
                    row.get("readiness_score"), now,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def _fetch_from_db(top_n: int = 100) -> pd.DataFrame:
    """Load prospect rankings from DB."""
    from src.database import get_connection
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM prospect_rankings ORDER BY fg_rank LIMIT ?",
            conn, params=(top_n,),
        )
    finally:
        conn.close()


# ── Public API ───────────────────────────────────────────────────────

def _scrape_mlb_pipeline() -> pd.DataFrame:
    """Fallback Level 2: Scrape MLB Pipeline prospect list.
    Returns DataFrame with basic ranking + stats (no scouting tools).
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        # MLB Pipeline prospect list page
        url = "https://www.mlb.com/prospects"
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        # Try JSON embedded in page first (MLB often embeds data)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for prospect data in script tags
        import json as _json
        for script in soup.find_all("script"):
            text = script.string or ""
            if "prospects" in text.lower() and "rank" in text.lower():
                # Try to extract JSON array
                for start in range(len(text)):
                    if text[start] == "[":
                        for end in range(len(text) - 1, start, -1):
                            if text[end] == "]":
                                try:
                                    data = _json.loads(text[start : end + 1])
                                    if isinstance(data, list) and len(data) > 5:
                                        rows = []
                                        for i, item in enumerate(data[:100]):
                                            rows.append({
                                                "name": item.get("name", item.get("fullName", "")),
                                                "team": item.get("team", {}).get("abbreviation", "")
                                                    if isinstance(item.get("team"), dict) else str(item.get("team", "")),
                                                "position": item.get("position", item.get("primaryPosition", "")),
                                                "fg_rank": i + 1,
                                                "mlb_id": item.get("playerId", item.get("id")),
                                            })
                                        if rows:
                                            logger.info("MLB Pipeline scrape: found %d prospects", len(rows))
                                            return pd.DataFrame(rows)
                                except (ValueError, TypeError):
                                    continue
                                break
        logger.warning("MLB Pipeline scrape: no parseable data found")
        return pd.DataFrame()
    except Exception:
        logger.warning("MLB Pipeline scrape failed", exc_info=True)
        return pd.DataFrame()


def refresh_prospect_rankings(force: bool = False) -> pd.DataFrame:
    """Refresh prospect rankings from external sources.

    Fallback chain:
    1. FanGraphs Board API (richest — scouting tools + reports)
    2. MLB Pipeline scrape (ranking + basic stats, no scouting tools)
    3. Static list (last resort)
    """
    from src.database import check_staleness

    if not force and not check_staleness("prospect_rankings", 168):
        return _fetch_from_db()

    def _enrich_and_store(df: pd.DataFrame) -> pd.DataFrame:
        """Fetch MiLB stats, compute readiness, store to DB."""
        mlb_ids = [mid for mid in df["mlb_id"].dropna().astype(int).tolist() if mid > 0]
        if mlb_ids:
            milb_df = fetch_milb_stats(mlb_ids)
            if not milb_df.empty:
                df = df.merge(milb_df, on="mlb_id", how="left", suffixes=("", "_milb"))
                for col in milb_df.columns:
                    if col != "mlb_id" and col in df.columns and f"{col}_milb" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_milb"])
                        df.drop(columns=[f"{col}_milb"], inplace=True)
        df["readiness_score"] = df.apply(
            lambda r: compute_mlb_readiness_score(r.to_dict()), axis=1
        )
        _store_prospects(df)
        return df

    # Level 1: FanGraphs Board API
    df = fetch_fangraphs_prospects()
    if not df.empty:
        return _enrich_and_store(df)

    # Level 2: MLB Pipeline scrape
    df = _scrape_mlb_pipeline()
    if not df.empty:
        logger.info("Using MLB Pipeline scrape (no scouting tools)")
        return _enrich_and_store(df)

    # Level 3: Static fallback
    logger.warning("All prospect sources failed — using static list")
    return pd.DataFrame(_STATIC_PROSPECTS)


def get_prospect_rankings(
    top_n: int = 100,
    position: str | None = None,
    org: str | None = None,
) -> pd.DataFrame:
    """Get prospect rankings, optionally filtered.

    Tries DB first, refreshes if empty.
    """
    df = _fetch_from_db(top_n=500)  # fetch more than needed for filtering
    if df.empty:
        df = refresh_prospect_rankings()
    if df.empty:
        # Ultimate fallback
        df = pd.DataFrame(_STATIC_PROSPECTS)

    if position:
        df = df[df["position"].str.contains(position, case=False, na=False)]
    if org:
        df = df[df["team"].str.upper() == org.upper()]

    return df.head(top_n).reset_index(drop=True)


def get_prospect_detail(prospect_id: int) -> dict | None:
    """Get full detail for a single prospect by prospect_id."""
    from src.database import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM prospect_rankings WHERE prospect_id = ?",
            (prospect_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_prospect_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Rewrite tests/test_prospect_rankings.py**

Update imports to use `src.prospect_engine` instead of `src.ecr`:

```python
# tests/test_prospect_rankings.py
"""Tests for prospect rankings (updated for prospect_engine)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.prospect_engine import get_prospect_rankings


def test_fetch_returns_dataframe():
    df = get_prospect_rankings()
    assert len(df) > 0
    assert "name" in df.columns


def test_top_n_limit():
    df = get_prospect_rankings(top_n=5)
    assert len(df) <= 5


def test_top_n_exceeds_available():
    df = get_prospect_rankings(top_n=500)
    assert len(df) <= 500


def test_position_filter_sp():
    df = get_prospect_rankings(position="SP")
    if len(df) > 0:
        assert all("SP" in pos for pos in df["position"])


def test_position_filter_no_match():
    df = get_prospect_rankings(position="DH")
    assert len(df) == 0


def test_org_filter():
    df = get_prospect_rankings(org="BOS")
    if len(df) > 0:
        assert all(t == "BOS" for t in df["team"])
```

- [ ] **Step 6: Run updated prospect_rankings tests**

Run: `python -m pytest tests/test_prospect_rankings.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All existing tests still pass

- [ ] **Step 8: Commit**

```bash
git add src/prospect_engine.py tests/test_prospect_engine.py tests/test_prospect_rankings.py
git commit -m "feat: add prospect engine with FG Board API, MiLB stats, MLB Readiness Score"
```

---

## Chunk 2: Multi-Platform ECR Consensus

**Features:** ECR rewrite with 7 sources, Trimmed Borda Count, player ID cross-reference
**Rationale:** Self-contained rewrite. All 7 sources either exist in the codebase (FG, NFBC, FP, Yahoo, HEATER SGP) or are new fetchers (ESPN, CBS). Must maintain backward-compat for `blend_ecr_with_projections()`.

### Task 3: ECR Consensus — Core Module

**Files:**
- Rewrite: `src/ecr.py`
- Rewrite: `tests/test_ecr.py` → rename to `tests/test_ecr_consensus.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ecr_consensus.py` with ~25 tests covering:

```python
# tests/test_ecr_consensus.py
"""Tests for multi-platform ECR consensus."""
from __future__ import annotations

import json
import statistics
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Trimmed Borda Count tests ────────────────────────────────────────

def test_consensus_single_source():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"fg_adp": 25})
    assert result["consensus_avg"] == 25.0
    assert result["n_sources"] == 1

def test_consensus_three_sources_no_trim():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 10, "b": 20, "c": 30})
    assert result["consensus_avg"] == 20.0
    assert result["n_sources"] == 3
    assert result["rank_min"] == 10
    assert result["rank_max"] == 30

def test_consensus_four_sources_trims_outliers():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 5, "b": 20, "c": 25, "d": 100})
    # Trim highest (100) and lowest (5), avg of [20, 25] = 22.5
    assert result["consensus_avg"] == 22.5
    assert result["n_sources"] == 4
    # min/max from ALL sources, not trimmed
    assert result["rank_min"] == 5
    assert result["rank_max"] == 100

def test_consensus_five_sources_trims():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 1, "b": 10, "c": 20, "d": 30, "e": 200})
    # Trim 1 and 200, avg of [10, 20, 30] = 20.0
    assert result["consensus_avg"] == 20.0

def test_consensus_stddev():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 10, "b": 20, "c": 30})
    expected = round(statistics.stdev([10, 20, 30]), 1)
    assert result["rank_stddev"] == expected

def test_consensus_stddev_single_source():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 10})
    assert result["rank_stddev"] == 0.0

def test_consensus_empty():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({})
    assert result["consensus_avg"] is None

def test_consensus_none_values_filtered():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({"a": 10, "b": None, "c": 30})
    assert result["n_sources"] == 2
    assert result["consensus_avg"] == 20.0

def test_consensus_seven_sources():
    from src.ecr import _compute_player_consensus
    result = _compute_player_consensus({
        "espn": 15, "yahoo": 20, "cbs": 18, "nfbc": 22,
        "fg": 16, "fp": 19, "heater": 17,
    })
    # Trim highest (22) and lowest (15), avg of [16, 17, 18, 19, 20] = 18.0
    assert result["consensus_avg"] == 18.0
    assert result["n_sources"] == 7


# ── Consensus rank assignment ────────────────────────────────────────

def test_assign_consensus_ranks_sequential():
    from src.ecr import assign_consensus_ranks
    players = [
        {"name": "A", "consensus_avg": 30.0},
        {"name": "B", "consensus_avg": 10.0},
        {"name": "C", "consensus_avg": 20.0},
    ]
    result = assign_consensus_ranks(players)
    ranked = {p["name"]: p["consensus_rank"] for p in result}
    assert ranked["B"] == 1  # lowest avg = rank 1
    assert ranked["C"] == 2
    assert ranked["A"] == 3

def test_assign_consensus_ranks_none_unranked():
    from src.ecr import assign_consensus_ranks
    players = [
        {"name": "A", "consensus_avg": 10.0},
        {"name": "B", "consensus_avg": None},
    ]
    result = assign_consensus_ranks(players)
    a = next(p for p in result if p["name"] == "A")
    b = next(p for p in result if p["name"] == "B")
    assert a["consensus_rank"] == 1
    assert b.get("consensus_rank") is None


# ── Disagreement detection ───────────────────────────────────────────

def test_disagreement_high():
    from src.ecr import compute_ecr_disagreement
    row = {"rank_stddev": 35.0, "n_sources": 4, "rank_min": 10, "rank_max": 90,
           "consensus_avg": 40.0, "espn_rank": 10, "yahoo_adp": 90}
    badge = compute_ecr_disagreement(row)
    assert badge is not None
    assert "High" in badge

def test_disagreement_moderate():
    from src.ecr import compute_ecr_disagreement
    row = {"rank_stddev": 20.0, "n_sources": 3, "rank_min": 20, "rank_max": 60,
           "consensus_avg": 35.0}
    badge = compute_ecr_disagreement(row)
    assert badge is not None
    assert "Moderate" in badge

def test_disagreement_none_low_stddev():
    from src.ecr import compute_ecr_disagreement
    row = {"rank_stddev": 5.0, "n_sources": 5}
    badge = compute_ecr_disagreement(row)
    assert badge is None

def test_disagreement_none_too_few_sources():
    from src.ecr import compute_ecr_disagreement
    row = {"rank_stddev": 50.0, "n_sources": 2}
    badge = compute_ecr_disagreement(row)
    assert badge is None


# ── ESPN API parsing ─────────────────────────────────────────────────

MOCK_ESPN_PLAYER = {
    "id": 12345,
    "fullName": "Mike Trout",
    "defaultPositionId": 8,
    "draftRanksByRankType": {
        "STANDARD": {"rank": 42}
    },
}

def test_parse_espn_player():
    from src.ecr import _parse_espn_player
    result = _parse_espn_player(MOCK_ESPN_PLAYER)
    assert result["espn_id"] == 12345
    assert result["name"] == "Mike Trout"
    assert result["espn_rank"] == 42

def test_parse_espn_player_missing_rank():
    from src.ecr import _parse_espn_player
    player = {"id": 1, "fullName": "Nobody", "draftRanksByRankType": {}}
    result = _parse_espn_player(player)
    assert result["espn_rank"] is None


# ── Backward compatibility ───────────────────────────────────────────

def test_blend_ecr_with_projections_signature():
    from src.ecr import blend_ecr_with_projections
    pool = pd.DataFrame([
        {"name": "Player A", "pick_score": 10.0},
        {"name": "Player B", "pick_score": 8.0},
    ])
    # Should accept both old ecr_df and new consensus_df forms
    result = blend_ecr_with_projections(pool, pd.DataFrame(), ecr_weight=0.15)
    assert "blended_rank" in result.columns
    assert "ecr_badge" in result.columns

def test_blend_preserves_columns():
    from src.ecr import blend_ecr_with_projections
    pool = pd.DataFrame([{"name": "A", "pick_score": 10.0}])
    result = blend_ecr_with_projections(pool, pd.DataFrame())
    assert "pick_score" in result.columns


# ── CBS parsing (best-effort) ────────────────────────────────────────

def test_parse_cbs_empty_html():
    from src.ecr import _parse_cbs_rankings
    result = _parse_cbs_rankings("")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ── DB round-trip ────────────────────────────────────────────────────

def test_store_and_load_consensus(tmp_path):
    from unittest.mock import patch
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.ecr import _store_consensus, load_ecr_consensus
        consensus_df = pd.DataFrame([{
            "player_id": 1, "espn_rank": 10, "yahoo_adp": 15.0,
            "cbs_rank": None, "nfbc_adp": 12.0, "fg_adp": 11.0,
            "fp_ecr": 13, "heater_sgp_rank": 8,
            "consensus_rank": 1, "consensus_avg": 11.5,
            "rank_min": 8, "rank_max": 15, "rank_stddev": 2.3,
            "n_sources": 6,
        }])
        count = _store_consensus(consensus_df)
        assert count == 1
        loaded = load_ecr_consensus()
        assert len(loaded) == 1
        assert loaded.iloc[0]["consensus_rank"] == 1


# ── ESPN pagination ─────────────────────────────────────────────────

def test_espn_pagination_fetches_multiple_pages():
    from src.ecr import fetch_espn_rankings
    page1 = [{"id": i, "fullName": f"Player {i}", "defaultPositionId": 8,
              "draftRanksByRankType": {"STANDARD": {"rank": i}}} for i in range(1, 51)]
    page2 = [{"id": i, "fullName": f"Player {i}", "defaultPositionId": 8,
              "draftRanksByRankType": {"STANDARD": {"rank": i}}} for i in range(51, 60)]
    with patch("src.ecr._espn_api_request") as mock_req:
        mock_req.side_effect = [page1, page2, []]  # 3 pages, last empty
        df = fetch_espn_rankings()
        assert len(df) == 59


# ── Yahoo ADP extraction ────────────────────────────────────────────

def test_fetch_yahoo_adp_from_client():
    from src.ecr import fetch_yahoo_adp
    mock_client = MagicMock()
    mock_client.get_league_draft_results.return_value = pd.DataFrame([
        {"player_name": "Player A", "pick": 5},
        {"player_name": "Player B", "pick": 12},
    ])
    with patch("src.ecr._get_yahoo_client", return_value=mock_client):
        df = fetch_yahoo_adp()
        assert "yahoo_adp" in df.columns or "pick" in df.columns

def test_fetch_yahoo_adp_no_client():
    from src.ecr import fetch_yahoo_adp
    with patch("src.ecr._get_yahoo_client", return_value=None):
        df = fetch_yahoo_adp()
        assert df.empty


# ── Player ID map DB round-trip ─────────────────────────────────────

def test_store_and_load_player_id_map(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.ecr import _upsert_player_id_map, _lookup_player_id_by_espn
        _upsert_player_id_map(player_id=42, espn_id=30836, name="Mike Trout")
        result = _lookup_player_id_by_espn(30836)
        assert result == 42

def test_player_id_map_upsert_updates(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.ecr import _upsert_player_id_map, _lookup_player_id_by_espn
        _upsert_player_id_map(player_id=42, espn_id=30836, name="Mike Trout")
        _upsert_player_id_map(player_id=42, espn_id=30836, yahoo_key="mlb.p.545361")
        # Should update, not duplicate
        from src.database import get_connection
        conn = get_connection()
        try:
            count = conn.execute("SELECT COUNT(*) FROM player_id_map WHERE player_id = 42").fetchone()[0]
            assert count == 1
        finally:
            conn.close()


# ── ID dedup/merge conflict ─────────────────────────────────────────

def test_resolve_player_ids_dedup():
    from src.ecr import _resolve_name_to_player_id
    # When two sources have different names for the same player,
    # fuzzy matching should still resolve to same player_id
    pool = pd.DataFrame([
        {"player_id": 1, "name": "Ronald Acuna Jr."},
        {"player_id": 2, "name": "Vladimir Guerrero Jr."},
    ])
    assert _resolve_name_to_player_id("Ronald Acuna", pool) == 1
    assert _resolve_name_to_player_id("Ronald Acuña Jr", pool) == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_ecr_consensus.py -v`
Expected: FAIL — functions don't exist yet

- [ ] **Step 3: Implement rewritten src/ecr.py**

Full rewrite with the following structure:
- `_compute_player_consensus(sources: dict) -> dict` — Trimmed Borda Count
- `assign_consensus_ranks(players: list[dict]) -> list[dict]` — sequential 1..N
- `compute_ecr_disagreement(consensus_row: dict) -> str | None` — new signature
- `_parse_espn_player(player: dict) -> dict` — ESPN API parsing
- `fetch_espn_rankings() -> pd.DataFrame` — ESPN Fantasy API
- `fetch_cbs_rankings() -> pd.DataFrame` — best-effort CBS scrape
- `_parse_cbs_rankings(html: str) -> pd.DataFrame` — CBS HTML parser
- `fetch_all_ranking_sources() -> dict[str, pd.DataFrame]` — aggregate all 7
- `resolve_player_ids_across_sources(sources: dict) -> pd.DataFrame` — cross-ref
- `compute_consensus(resolved_df: pd.DataFrame) -> pd.DataFrame` — full pipeline
- `refresh_ecr_consensus(force=False) -> pd.DataFrame` — staleness-aware refresh
- `_store_consensus(df) -> int` / `load_ecr_consensus() -> pd.DataFrame` — DB
- `blend_ecr_with_projections(valued_pool, consensus_df, ecr_weight=0.15)` — backward-compat
- `fetch_yahoo_adp() -> pd.DataFrame` — delegates to existing `YahooFantasyClient`

Key implementation details:
- ESPN: `_espn_api_request(offset, limit)` helper returns player list; `fetch_espn_rankings()` paginates (offset 0, 300, 600...) until empty page returned
- ESPN rank path: `player["draftRanksByRankType"]["STANDARD"]["rank"]`
- CBS: `requests.get(url)` with BeautifulSoup, graceful empty on JS-rendered
- Existing sources: import from `adp_sources.py` (FP ECR, NFBC) and `data_pipeline.py` (FG ADP)
- HEATER SGP rank: computed from `valuation.py`'s `value_all_players()` on the loaded pool
- Minimum viable: FG + NFBC + HEATER SGP (3 sources, always available)
- Yahoo ADP: `_get_yahoo_client()` reads from `st.session_state.yahoo_client`; `fetch_yahoo_adp()` returns empty DataFrame when not connected

**Player ID cross-reference and dedup/merge:**
- `_upsert_player_id_map(player_id, espn_id=None, yahoo_key=None, fg_id=None, mlb_id=None, name=None)` — inserts or updates `player_id_map` table via `INSERT OR REPLACE`
- `_lookup_player_id_by_espn(espn_id)` / `_lookup_player_id_by_yahoo(yahoo_key)` — reverse lookups
- `_resolve_name_to_player_id(name, player_pool)` — fuzzy match (reuses `fuzzy_match_player()` from `src/engine/projections/projection_client.py`) to resolve external source names to HEATER `player_id`
- **Dedup strategy:** When resolving names across sources, if two sources map different external IDs to the same HEATER `player_id`, both external IDs are stored in `player_id_map`. If two sources have slightly different name spellings (e.g., "Ronald Acuña Jr" vs "Ronald Acuna Jr."), fuzzy matching resolves both to the same `player_id`. Name normalization strips accents and suffixes before matching.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_ecr_consensus.py -v`
Expected: All PASS

- [ ] **Step 5: Delete old tests/test_ecr.py**

```bash
git rm tests/test_ecr.py
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass (old test_ecr.py removed, new test_ecr_consensus.py in place)

- [ ] **Step 7: Commit**

```bash
git add src/ecr.py tests/test_ecr_consensus.py
git commit -m "feat: rewrite ECR with 7-source consensus, Trimmed Borda Count, ESPN/CBS fetchers"
```

---

## Chunk 3: News/Transaction Intelligence

**Features:** Multi-source news aggregation, template-based analytical summaries, ownership trends
**Rationale:** Self-contained new module. Imports from existing `news_fetcher.py` (unchanged) and `il_manager.py`.

### Task 4: Player News — Core Module

**Files:**
- Create: `src/player_news.py`
- Create: `tests/test_player_news.py`
- Modify: `src/live_stats.py` (add `fetch_player_enhanced_status()`)
- Modify: `requirements.txt` (add `feedparser`)

- [ ] **Step 1: Add feedparser to requirements.txt**

Add `feedparser>=6.0` to `requirements.txt`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_player_news.py
"""Tests for player news intelligence module."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── ESPN news parsing ────────────────────────────────────────────────

MOCK_ESPN_NEWS = {
    "articles": [
        {
            "headline": "Mike Trout placed on 10-day IL with knee injury",
            "description": "Angels CF Mike Trout was placed on the 10-day IL...",
            "published": "2026-03-18T12:00:00Z",
            "categories": [{"athleteId": 30836}],
        },
        {
            "headline": "MLB Trade Rumors roundup",
            "description": "General news with no specific player",
            "published": "2026-03-18T10:00:00Z",
            "categories": [],
        },
    ]
}

def test_parse_espn_news():
    from src.player_news import _parse_espn_news
    items = _parse_espn_news(MOCK_ESPN_NEWS)
    assert len(items) == 1  # second article has no athleteId
    assert items[0]["headline"] == "Mike Trout placed on 10-day IL with knee injury"
    assert items[0]["source"] == "espn"
    assert items[0]["espn_athlete_id"] == 30836


# ── RotoWire RSS parsing ────────────────────────────────────────────

MOCK_RSS_XML = """<?xml version="1.0"?>
<rss version="2.0">
<channel>
<item>
  <title>Shohei Ohtani (elbow) expected back next week</title>
  <description>Ohtani is progressing well from surgery...</description>
  <pubDate>Mon, 18 Mar 2026 14:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""

def test_parse_rotowire_rss():
    from src.player_news import _parse_rotowire_entries
    import feedparser
    feed = feedparser.parse(MOCK_RSS_XML)
    items = _parse_rotowire_entries(feed.entries)
    assert len(items) == 1
    assert "Ohtani" in items[0]["headline"]
    assert items[0]["source"] == "rotowire"


# ── MLB enhanced status parsing ──────────────────────────────────────

def test_parse_mlb_enhanced_status():
    from src.player_news import _parse_mlb_enhanced_status
    mock_data = {
        "people": [{
            "id": 545361,
            "fullName": "Mike Trout",
            "currentTeam": {"name": "Los Angeles Angels"},
            "rosterEntries": [{"status": {"code": "D10", "description": "10-Day Injured List"}}],
            "transactions": [
                {"description": "placed on 10-day IL with right knee inflammation",
                 "date": "2026-03-15"},
            ],
        }]
    }
    items = _parse_mlb_enhanced_status(mock_data, 545361)
    assert len(items) >= 1
    assert items[0]["il_status"] == "IL10"


# ── News type classification ────────────────────────────────────────

def test_classify_news_type_injury():
    from src.player_news import _classify_news_type
    assert _classify_news_type("placed on 10-day IL with knee strain") == "injury"

def test_classify_news_type_transaction():
    from src.player_news import _classify_news_type
    assert _classify_news_type("traded to the New York Yankees") == "transaction"

def test_classify_news_type_callup():
    from src.player_news import _classify_news_type
    assert _classify_news_type("called up from Triple-A") == "callup"

def test_classify_news_type_lineup():
    from src.player_news import _classify_news_type
    assert _classify_news_type("moved to leadoff in the batting order") == "lineup"

def test_classify_news_type_general():
    from src.player_news import _classify_news_type
    assert _classify_news_type("some random update") == "general"


# ── Template summary generation ──────────────────────────────────────

def test_generate_injury_summary():
    from src.player_news import generate_intel_summary
    intel = {
        "news_type": "injury",
        "il_status": "IL10",
        "injury_body_part": "knee",
        "duration_weeks": 2.0,
        "lost_sgp": -1.5,
        "replacement_name": "John Doe",
        "replacement_sgp": 0.8,
        "ownership_trend": "down",
        "ownership_delta": -5.0,
    }
    summary = generate_intel_summary(intel)
    assert "knee" in summary.lower()
    assert "2.0" in summary
    assert "John Doe" in summary

def test_generate_trade_summary():
    from src.player_news import generate_intel_summary
    intel = {
        "news_type": "transaction",
        "new_team": "NYY",
        "park_factor": 1.05,
        "park_context": "Hitter-friendly park",
        "sgp_delta": 0.3,
    }
    summary = generate_intel_summary(intel)
    assert "NYY" in summary

def test_generate_callup_summary():
    from src.player_news import generate_intel_summary
    intel = {
        "news_type": "callup",
        "milb_level": "AAA",
        "milb_avg": 0.310,
        "milb_obp": 0.380,
        "milb_slg": 0.520,
        "projected_sgp": 2.5,
        "roster_context": "Replaces injured starter",
    }
    summary = generate_intel_summary(intel)
    assert "AAA" in summary

def test_generate_lineup_summary():
    from src.player_news import generate_intel_summary
    intel = {
        "news_type": "lineup",
        "batting_slot": 2,
        "projected_pa_week": 30.5,
        "pa_delta": 2.3,
        "sgp_delta": 0.15,
    }
    summary = generate_intel_summary(intel)
    assert "30.5" in summary

def test_generate_summary_unknown_type():
    from src.player_news import generate_intel_summary
    intel = {"news_type": "general", "headline": "Some news"}
    summary = generate_intel_summary(intel)
    assert isinstance(summary, str)


# ── Ownership trend computation ──────────────────────────────────────

def test_compute_ownership_trend_no_data():
    from src.player_news import compute_ownership_trend
    with patch("src.player_news._load_ownership_history") as mock:
        mock.return_value = []
        result = compute_ownership_trend(1)
        assert result == {}

def test_compute_ownership_trend_with_data():
    from src.player_news import compute_ownership_trend
    with patch("src.player_news._load_ownership_history") as mock:
        mock.return_value = [
            {"date": "2026-03-18", "percent_owned": 85.0},
            {"date": "2026-03-11", "percent_owned": 80.0},
        ]
        result = compute_ownership_trend(1)
        assert result["current"] == 85.0
        assert result["delta_7d"] == 5.0
        assert result["direction"] == "up"


# ── DB round-trip ────────────────────────────────────────────────────

def test_store_and_query_news(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.player_news import _store_news_items, _query_player_news
        items = [{
            "player_id": 1,
            "source": "espn",
            "headline": "Test headline",
            "detail": "Test detail",
            "news_type": "injury",
            "injury_body_part": "knee",
            "il_status": "IL10",
            "sentiment_score": -0.5,
            "published_at": "2026-03-18T12:00:00Z",
        }]
        count = _store_news_items(items)
        assert count == 1
        rows = _query_player_news(1)
        assert len(rows) == 1
        assert rows[0]["headline"] == "Test headline"

def test_store_duplicate_ignored(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        init_db()
        from src.player_news import _store_news_items
        item = {
            "player_id": 1, "source": "espn",
            "headline": "Duplicate", "published_at": "2026-03-18",
        }
        _store_news_items([item])
        count = _store_news_items([item])  # second insert
        assert count == 0  # UNIQUE constraint prevents duplicate


# ── Yahoo injury field extraction ─────────────────────────────────────

def test_extract_yahoo_injury_fields():
    from src.player_news import _extract_yahoo_injury_news
    player_data = {
        "player_id": 42,
        "name": "Mike Trout",
        "injury_note": "Right knee inflammation",
        "status_full": "10-Day Injured List",
        "percent_owned": 99.5,
    }
    items = _extract_yahoo_injury_news(player_data)
    assert len(items) == 1
    assert items[0]["source"] == "yahoo"
    assert items[0]["il_status"] == "IL10"

def test_extract_yahoo_no_injury():
    from src.player_news import _extract_yahoo_injury_news
    player_data = {"player_id": 42, "name": "Healthy Player"}
    items = _extract_yahoo_injury_news(player_data)
    assert len(items) == 0


# ── ESPN athleteId cross-reference ────────────────────────────────────

def test_espn_athlete_id_to_player_id():
    from src.player_news import _resolve_espn_athlete_id
    with patch("src.player_news._lookup_player_by_espn_id") as mock:
        mock.return_value = 42
        result = _resolve_espn_athlete_id(30836)
        assert result == 42

def test_espn_athlete_id_unknown():
    from src.player_news import _resolve_espn_athlete_id
    with patch("src.player_news._lookup_player_by_espn_id", return_value=None):
        result = _resolve_espn_athlete_id(99999)
        assert result is None


# ── Fallback with missing sources ─────────────────────────────────────

def test_aggregate_news_graceful_on_all_failures():
    from src.player_news import aggregate_news
    with patch("src.player_news.fetch_espn_news", side_effect=Exception("fail")), \
         patch("src.player_news.fetch_rotowire_rss", side_effect=Exception("fail")), \
         patch("src.player_news.fetch_mlb_enhanced_status", return_value=[]), \
         patch("src.player_news._query_player_news", return_value=[]):
        result = aggregate_news(player_id=42)
        assert isinstance(result, list)
        assert len(result) == 0  # graceful empty, not exception


# ── Batch roster intel ─────────────────────────────────────────────────

def test_generate_roster_intel_batch():
    from src.player_news import generate_roster_intel
    with patch("src.player_news.aggregate_news", return_value=[]) as mock_agg, \
         patch("src.player_news.compute_ownership_trend", return_value={}):
        pool = pd.DataFrame([
            {"player_id": 1, "name": "A"},
            {"player_id": 2, "name": "B"},
        ])
        result = generate_roster_intel([1, 2], pool)
        assert isinstance(result, dict)
        assert mock_agg.call_count == 2
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_player_news.py -v`
Expected: FAIL

- [ ] **Step 3: Add `fetch_player_enhanced_status()` to src/live_stats.py**

Add near the end of the file:

```python
def fetch_player_enhanced_status(mlb_id: int) -> dict | None:
    """Fetch enhanced player status with roster entries and transactions.
    Uses MLB Stats API hydrate=rosterEntries,transactions.
    """
    if statsapi is None:
        return None
    try:
        data = statsapi.get(
            "person",
            {"personIds": mlb_id, "hydrate": "rosterEntries,transactions"},
        )
        people = data.get("people", [])
        return people[0] if people else None
    except Exception:
        logger.warning("Enhanced status fetch failed for mlb_id=%s", mlb_id, exc_info=True)
        return None
```

- [ ] **Step 4: Implement src/player_news.py**

Full implementation with:
- `_parse_espn_news(data)` — parse ESPN news JSON
- `_parse_rotowire_entries(entries)` — parse feedparser entries
- `_parse_mlb_enhanced_status(data, mlb_id)` — parse MLB API hydrated response
- `_classify_news_type(text)` — keyword-based classification
- `_classify_il_status(text)` — extract IL type from description
- `fetch_espn_news()` — ESPN news API call
- `fetch_rotowire_rss()` — tries multiple RSS URLs
- `fetch_mlb_enhanced_status(player_ids)` — batch MLB API
- `aggregate_news(player_id)` — combine all sources for one player
- `generate_intel_summary(intel)` — template-based summaries
- `generate_player_intel(player_id, player_pool, config)` — full intel dict
- `generate_injury_context(player_id, il_status, body_part)` — uses `il_manager.py`
- `find_replacement_options(player_id, player_pool, config, top_n)` — uses `waiver_wire.py` pattern
- `compute_ownership_trend(player_id, lookback_days)` — reads `ownership_trends` table
- `_load_ownership_history(player_id)` — DB query
- `_store_news_items(items)` / `_query_player_news(player_id)` — DB persistence
- `refresh_all_news(yahoo_client, force)` — batch refresh all sources
- `generate_roster_intel(roster_ids, player_pool, config)` — batch for My Team page

Key patterns:
- `FEEDPARSER_AVAILABLE` flag with `try/except ImportError`
- Rate limiting: 0.5s between external requests
- Per-source error counting (3 failures → skip)
- Imports `fetch_recent_transactions()` from `news_fetcher.py` (not duplicated)
- Imports `classify_il_type()` and `estimate_il_duration()` from `il_manager.py`
- Imports `compute_news_sentiment()` from `news_sentiment.py`

- [ ] **Step 5: Run tests to verify GREEN**

Run: `python -m pytest tests/test_player_news.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/player_news.py tests/test_player_news.py src/live_stats.py requirements.txt
git commit -m "feat: add multi-source news intelligence with template-based analytical summaries"
```

---

## Chunk 4: Bootstrap Integration + UI Wiring

**Features:** Bootstrap phases 13-15, StalenessConfig update, UI changes to Leaders and My Team pages
**Rationale:** Wiring layer that connects the 3 new modules into the app lifecycle.

### Task 5: Bootstrap Phases 13-15

**Files:**
- Modify: `src/data_bootstrap.py` (add 3 phases + StalenessConfig fields)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_fp_edge_schema.py`:

```python
def test_staleness_config_has_new_fields():
    from src.data_bootstrap import StalenessConfig
    sc = StalenessConfig()
    assert hasattr(sc, "prospects_hours")
    assert hasattr(sc, "news_hours")
    assert hasattr(sc, "ecr_consensus_hours")
    assert sc.prospects_hours == 168
    assert sc.news_hours == 1
    assert sc.ecr_consensus_hours == 24
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_fp_edge_schema.py::test_staleness_config_has_new_fields -v`
Expected: FAIL — fields don't exist

- [ ] **Step 3: Add StalenessConfig fields**

In `src/data_bootstrap.py`, add 3 fields to `StalenessConfig`:

```python
prospects_hours: float = 168      # 7 days
news_hours: float = 1             # 1 hour
ecr_consensus_hours: float = 24   # 24 hours
```

- [ ] **Step 4: Add 3 bootstrap phase functions**

After the existing `_bootstrap_news()` and before `bootstrap_all_data()`, add:

```python
def _bootstrap_prospects(progress: BootstrapProgress) -> str:
    """Phase 13: Prospect rankings from FanGraphs + MiLB stats."""
    progress.phase = "Prospects"
    progress.detail = "Refreshing prospect rankings..."
    try:
        from src.database import update_refresh_log
        from src.prospect_engine import refresh_prospect_rankings

        df = refresh_prospect_rankings(force=True)
        update_refresh_log("prospect_rankings", "success")
        return f"Prospects: {len(df)} ranked"
    except Exception as e:
        logger.warning("Prospect bootstrap failed: %s", e)
        return f"Prospects: error ({e})"


def _bootstrap_news_intel(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Phase 14: Multi-source news intelligence."""
    progress.phase = "News Intelligence"
    progress.detail = "Fetching news from ESPN, RotoWire, MLB API..."
    try:
        from src.database import update_refresh_log
        from src.player_news import refresh_all_news

        count = refresh_all_news(yahoo_client=yahoo_client, force=True)
        update_refresh_log("news_intelligence", "success")
        return f"News: {count} items from multi-source"
    except Exception as e:
        logger.warning("News intelligence bootstrap failed: %s", e)
        return f"News intel: error ({e})"


def _bootstrap_ecr_consensus(progress: BootstrapProgress) -> str:
    """Phase 15: ECR consensus from 7 ranking sources."""
    progress.phase = "ECR Consensus"
    progress.detail = "Building multi-platform ranking consensus..."
    try:
        from src.database import update_refresh_log
        from src.ecr import refresh_ecr_consensus

        df = refresh_ecr_consensus(force=True)
        update_refresh_log("ecr_consensus", "success")
        return f"ECR Consensus: {len(df)} players ranked"
    except Exception as e:
        logger.warning("ECR consensus bootstrap failed: %s", e)
        return f"ECR consensus: error ({e})"
```

- [ ] **Step 5: Add phases to `bootstrap_all_data()`**

In `bootstrap_all_data()`, after Phase 12 (deduplication) block and before `_notify(1.0)`, insert:

```python
    # Phase 13: Prospect rankings
    _notify(0.96)
    if force or check_staleness("prospect_rankings", staleness.prospects_hours):
        results["prospects"] = _bootstrap_prospects(progress)
    else:
        results["prospects"] = "Fresh"

    # Phase 14: News intelligence (multi-source)
    _notify(0.97)
    if force or check_staleness("news_intelligence", staleness.news_hours):
        results["news_intelligence"] = _bootstrap_news_intel(progress, yahoo_client)
    else:
        results["news_intelligence"] = "Fresh"

    # Phase 15: ECR consensus (depends on Phase 3 projections + Phase 9 ADP)
    _notify(0.99)
    if force or check_staleness("ecr_consensus", staleness.ecr_consensus_hours):
        results["ecr_consensus"] = _bootstrap_ecr_consensus(progress)
    else:
        results["ecr_consensus"] = "Fresh"
```

- [ ] **Step 6: Run test to verify GREEN**

Run: `python -m pytest tests/test_fp_edge_schema.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add src/data_bootstrap.py tests/test_fp_edge_schema.py
git commit -m "feat: add bootstrap phases 13-15 for prospects, news intel, ECR consensus"
```

---

### Task 6: Yahoo API — Injury/Ownership Field Extraction

**Files:**
- Modify: `src/yahoo_api.py` (extract `injury_note`, `status_full`, `percent_owned`)

- [ ] **Step 1: Identify the sync method that processes player data**

Read `src/yahoo_api.py` to find where player data is processed during `sync_to_db()`. The injury/ownership fields need to be extracted from the yfpy Player model during roster sync.

- [ ] **Step 2: Add field extraction**

In the roster sync loop where players are processed, extract:
```python
injury_note = getattr(player, "injury_note", None)
status_full = getattr(player, "status_full", None)
percent_owned_raw = getattr(player, "percent_owned", None)
# percent_owned may be a nested object or float
if hasattr(percent_owned_raw, "value"):
    percent_owned = float(percent_owned_raw.value)
elif percent_owned_raw is not None:
    percent_owned = float(percent_owned_raw)
else:
    percent_owned = None
```

Store these in the `ownership_trends` and `player_news` tables when available.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All pass (Yahoo tests use mocks)

- [ ] **Step 4: Commit**

```bash
git add src/yahoo_api.py
git commit -m "feat: extract injury_note, status_full, percent_owned from Yahoo sync"
```

---

### Task 7: Leaders Page — Enhanced Prospects Tab

**Files:**
- Modify: `pages/9_Leaders.py`

- [ ] **Step 1: Read current Leaders page structure**

Read `pages/9_Leaders.py` to identify where the existing Prospects tab is and how it uses `fetch_prospect_rankings()` from `src/ecr.py`.

- [ ] **Step 2: Update import and enhance Prospects tab**

Change imports from `src.ecr` to `src.prospect_engine`:
```python
from src.prospect_engine import get_prospect_rankings, get_prospect_detail
```

Enhance the Prospects tab with:
- Position filter pills (SS, OF, SP, etc.)
- Organization filter dropdown
- ETA year filter
- Sortable table with readiness score column
- Expandable rows with scouting report text and tool grades
- "Fantasy Relevant" badge when available
- **Plotly radar chart** for scouting tool visualization:

```python
import plotly.graph_objects as go

def _render_prospect_radar(prospect: dict):
    """Render scouting tool radar chart for a prospect."""
    # Hitter tools: Hit, Game, Raw, Speed, Field
    # Pitcher tools: Ctrl (present/future only — no hit/game/raw for pitchers)
    is_pitcher = "SP" in (prospect.get("position") or "") or "P" in (prospect.get("position") or "")

    if is_pitcher:
        categories = ["Ctrl (Pres)", "Ctrl (Fut)"]
        present = [prospect.get("ctrl_present") or 0, prospect.get("ctrl_future") or 0]
        # Pitchers don't have the full 5-tool radar — show available tools only
        if not any(present):
            st.caption("No scouting tool grades available")
            return
    else:
        categories = ["Hit", "Game Power", "Raw Power", "Speed", "Field"]
        present = [
            prospect.get("hit_present") or 0, prospect.get("game_present") or 0,
            prospect.get("raw_present") or 0, prospect.get("speed") or 0,
            prospect.get("field") or 0,
        ]
        future = [
            prospect.get("hit_future") or 0, prospect.get("game_future") or 0,
            prospect.get("raw_future") or 0, prospect.get("speed") or 0,
            prospect.get("field") or 0,
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=present, theta=categories, fill="toself",
        name="Present", line_color=T["sky"],
    ))
    if not is_pitcher:
        fig.add_trace(go.Scatterpolar(
            r=future, theta=categories, fill="toself",
            name="Future", line_color=T["hot"], opacity=0.5,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[20, 80])),
        showlegend=True, height=300, margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
```

Call `_render_prospect_radar(prospect)` inside each expandable row when scouting tool data is available.

- [ ] **Step 3: Run app to verify UI renders**

Run: `streamlit run app.py` and navigate to Leaders → Prospects tab. Verify it loads without errors.

- [ ] **Step 4: Commit**

```bash
git add pages/9_Leaders.py
git commit -m "feat: enhance Prospects tab with FG scouting tools, readiness score, filters"
```

---

### Task 8: My Team Page — News & Alerts Tab

**Files:**
- Modify: `pages/1_My_Team.py`

- [ ] **Step 1: Read current My Team page structure**

Read `pages/1_My_Team.py` to identify existing tabs and where to add News & Alerts.

- [ ] **Step 2: Add News & Alerts tab**

Add a new tab with:
- Import `from src.player_news import generate_roster_intel, generate_intel_summary`
- Player intel cards for all roster players with recent news
- Each card shows: headline, source badge, injury context block, ownership trend arrow, sentiment indicator
- Sort options: recency, severity, SGP impact
- Graceful "No recent news" message when empty

- [ ] **Step 3: Run app to verify UI renders**

Run: `streamlit run app.py` and navigate to My Team → News & Alerts tab.

- [ ] **Step 4: Commit**

```bash
git add pages/1_My_Team.py
git commit -m "feat: add News & Alerts tab to My Team page with multi-source intel"
```

---

### Task 9: Draft Board & Available Players — Consensus Rank Integration

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update draft board to use consensus rank**

In `app.py`, find the draft board table rendering and:
- Change "ECR Rank" column header to "Consensus Rank"
- Add range tooltip showing all source ranks on hover
- Update `blend_ecr_with_projections()` call to use new consensus data
- Add "HEATER vs Consensus" delta column

- [ ] **Step 2: Update Available Players tab with consensus sort and disagreement filter**

In `app.py`, find the Available Players tab (~line 1638) and:
- Add "Sort by" selectbox with options: `["HEATER Rank", "Consensus Rank", "ADP", "Name"]`
  ```python
  sort_option = st.selectbox("Sort by", ["HEATER Rank", "Consensus Rank", "ADP", "Name"], key="avail_sort")
  sort_map = {
      "HEATER Rank": "enhanced_rank",
      "Consensus Rank": "consensus_rank",
      "ADP": "adp",
      "Name": "name",
  }
  available_display = available_display.sort_values(sort_map[sort_option], ascending=True, na_position="last")
  ```
- Add "Disagreement Level" filter pill buttons (All / High / Medium / Low):
  ```python
  disagree_filter = st.radio(
      "Disagreement Level", ["All", "High", "Medium", "Low"],
      horizontal=True, key="disagree_filter"
  )
  if disagree_filter != "All":
      level_map = {"High": 30, "Medium": 15, "Low": 0}
      threshold = level_map[disagree_filter]
      if disagree_filter == "High":
          available_display = available_display[
              available_display["consensus_delta"].abs() > threshold
          ]
      elif disagree_filter == "Medium":
          available_display = available_display[
              (available_display["consensus_delta"].abs() > 15) &
              (available_display["consensus_delta"].abs() <= 30)
          ]
      else:  # Low
          available_display = available_display[
              available_display["consensus_delta"].abs() <= 15
          ]
  ```
- Add `consensus_rank` and `consensus_delta` columns to the displayed dataframe
- Color-code delta column: green when HEATER ranks higher (negative delta = value pick), red when consensus ranks higher

- [ ] **Step 3: Run app to verify**

Run: `streamlit run app.py` and check draft board + Available Players tab.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: update draft board and available players with consensus rank sort and disagreement filter"
```

---

## Chunk 5: Verification & Cleanup

### Task 10: Full Test Suite + Lint + Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: All existing tests + ~77 new tests pass

- [ ] **Step 2: Run linter**

Run: `python -m ruff check .`
Fix any issues.

Run: `python -m ruff format .`

- [ ] **Step 3: Verify test count**

Expected new test counts:
- `tests/test_fp_edge_schema.py`: ~13 tests
- `tests/test_prospect_engine.py`: ~17 tests (includes detail round-trip + missing detail)
- `tests/test_prospect_rankings.py`: ~6 tests (rewritten)
- `tests/test_ecr_consensus.py`: ~32 tests (includes ESPN pagination, Yahoo ADP, player ID map CRUD, dedup)
- `tests/test_player_news.py`: ~27 tests (includes Yahoo injury extraction, ESPN cross-ref, graceful failures, batch intel)

Total new: ~77 net-new tests (12 existing ECR + 6 existing prospect tests rewritten)

- [ ] **Step 4: Commit any lint fixes**

```bash
git add src/prospect_engine.py src/ecr.py src/player_news.py src/database.py src/data_bootstrap.py src/yahoo_api.py pages/9_Leaders.py pages/1_My_Team.py app.py tests/test_fp_edge_schema.py tests/test_prospect_engine.py tests/test_prospect_rankings.py tests/test_ecr_consensus.py tests/test_player_news.py
git commit -m "chore: lint and format fixes for FP edge features"
```

---

## Implementation Sequence (Recommended)

| Order | Task | Feature | Dependencies | New Tests |
|-------|------|---------|-------------|-----------|
| 1 | Task 1 | DB Schema (5 tables) | None | ~13 |
| 2 | Task 2 | Prospect Engine | Task 1 (DB) | ~17 + 6 rewritten |
| 3 | Task 3 | ECR Consensus Rewrite | Task 1 (DB) | ~32 (replaces 12) |
| 4 | Task 4 | Player News | Task 1 (DB) | ~27 |
| 5 | Task 5 | Bootstrap Phases 13-15 | Tasks 2-4 | 1 |
| 6 | Task 6 | Yahoo Injury/Ownership | Task 4 (player_news) | 0 |
| 7 | Task 7 | Leaders Page Enhancement | Task 2 (prospect_engine) | 0 |
| 8 | Task 8 | My Team News Tab | Task 4 (player_news) | 0 |
| 9 | Task 9 | Draft Board Consensus | Task 3 (ecr.py) | 0 |
| 10 | Task 10 | Verification & Cleanup | All | 0 |

**Total:** 10 tasks, ~77 net-new tests, 2 new source files, 1 rewritten source file, 5 new DB tables, 3 bootstrap phases

---

## Subagent Strategy

Tasks 2, 3, and 4 are independent after Task 1 completes (they only share the DB schema). Execute them in parallel:

- **Sequential:** Task 1 (DB foundation)
- **Parallel:** Tasks 2, 3, 4 (prospect engine || ECR consensus || player news)
- **Sequential after parallel:** Task 5 (bootstrap — needs all 3 modules)
- **Parallel:** Tasks 6, 7, 8, 9 (all UI wiring — independent pages)
- **Final:** Task 10 (verification)

---

## Verification

After all tasks complete:

1. **Run full test suite:** `python -m pytest -v` — expect all existing + ~77 new tests to pass
2. **Lint:** `python -m ruff check .` and `python -m ruff format .`
3. **Launch app:** `streamlit run app.py` — verify:
   - Leaders page: Prospects tab with readiness score, FG scouting tools, position/org filters
   - My Team page: News & Alerts tab with multi-source intel cards
   - Draft board: Consensus Rank column, disagreement badges, HEATER vs Consensus delta
   - Available Players tab: Consensus Rank sort option, disagreement level filter
   - Bootstrap: Phases 13-15 appear in splash screen progress
4. **Commit and push:**
   ```bash
   git add src/prospect_engine.py src/ecr.py src/player_news.py src/database.py src/data_bootstrap.py src/yahoo_api.py pages/9_Leaders.py pages/1_My_Team.py app.py tests/test_fp_edge_schema.py tests/test_prospect_engine.py tests/test_prospect_rankings.py tests/test_ecr_consensus.py tests/test_player_news.py
   git commit -m "feat: implement 3 FP edge features — prospect engine, ECR consensus, news intelligence"
   ```
