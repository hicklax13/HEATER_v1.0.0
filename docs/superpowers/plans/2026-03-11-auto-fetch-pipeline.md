# Auto-Fetch Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically fetch Steamer, ZiPS, and Depth Charts projections (plus ADP) from FanGraphs on app startup, eliminating manual CSV uploads.

**Architecture:** A new `src/data_pipeline.py` module fetches projections from FanGraphs' internal JSON API (`/api/projections`) on first visit to Step 1. It normalizes the JSON responses to match the existing DB schema, upserts players, stores projections, blends them, extracts ADP, and writes a refresh log entry. CSV upload remains as fallback. The module reuses `database.py`'s `_upsert_player()`, `create_blended_projections()`, and `update_refresh_log()` patterns.

**Tech Stack:** requests (HTTP), pandas (normalization), sqlite3 (storage), Streamlit (session state + UI)

**Spec:** `docs/superpowers/specs/2026-03-11-auto-fetch-pipeline-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/data_pipeline.py` | Fetch, normalize, store FanGraphs projections + ADP |
| Create | `tests/test_data_pipeline.py` | ~28 tests for the pipeline module |
| Modify | `app.py:713-793` | Wire auto-fetch into Step 1, fix CSV import bug |
| Modify | `requirements.txt` | Add `requests` explicitly |

---

## Chunk 1: Foundation + Normalization

### Task 1: Add requests dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add requests to requirements.txt**

Add `requests>=2.28.0` after `plotly`:

```
streamlit>=1.37.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.18.0
requests>=2.28.0
MLB-StatsAPI>=1.7
```

- [ ] **Step 2: Verify install**

Run: `pip install -r requirements.txt`
Expected: All packages satisfied (requests was already a transitive dep of streamlit, now explicit)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requests as explicit dependency for FG data pipeline"
```

---

### Task 2: Create data_pipeline.py skeleton + hitter normalization

**Files:**
- Create: `src/data_pipeline.py`
- Create: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write the failing test for hitter normalization**

Create `tests/test_data_pipeline.py`:

```python
"""Tests for the FanGraphs auto-fetch data pipeline."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    """Use a temp database for every test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    # Also patch data_pipeline's reference to database
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


# ── Sample FanGraphs JSON records ──────────────────────────────────

SAMPLE_HITTER_JSON = [
    {
        "PlayerName": "Aaron Judge",
        "Team": "NYY",
        "minpos": "OF",
        "PA": 650,
        "AB": 550,
        "H": 160,
        "R": 110,
        "HR": 45,
        "RBI": 120,
        "SB": 5,
        "AVG": 0.291,
        "ADP": 3.5,
    },
    {
        "PlayerName": "Mookie Betts",
        "Team": "LAD",
        "minpos": "SS",
        "PA": 620,
        "AB": 540,
        "H": 170,
        "R": 105,
        "HR": 25,
        "RBI": 85,
        "SB": 15,
        "AVG": 0.315,
        "ADP": 5.0,
    },
]

SAMPLE_PITCHER_JSON = [
    {
        "PlayerName": "Gerrit Cole",
        "Team": "NYY",
        "IP": 200.0,
        "W": 16,
        "SV": 0,
        "SO": 250,
        "ERA": 2.95,
        "WHIP": 1.05,
        "ER": 66,
        "BB": 45,
        "H": 165,
        "GS": 33,
        "G": 33,
        "ADP": 15.0,
    },
    {
        "PlayerName": "Josh Hader",
        "Team": "HOU",
        "IP": 65.0,
        "W": 4,
        "SV": 38,
        "SO": 85,
        "ERA": 2.50,
        "WHIP": 0.95,
        "ER": 18,
        "BB": 20,
        "H": 42,
        "GS": 0,
        "G": 65,
        "ADP": 55.0,
    },
    {
        "PlayerName": "Dual Starter",
        "Team": "SEA",
        "IP": 70.0,
        "W": 5,
        "SV": 1,
        "SO": 70,
        "ERA": 3.80,
        "WHIP": 1.20,
        "ER": 30,
        "BB": 25,
        "H": 59,
        "GS": 3,
        "G": 40,
        "ADP": 250.0,
    },
]


# ── Tests ──────────────────────────────────────────────────────────


class TestNormalizeHitterJson:
    def test_column_mapping(self):
        """FG JSON fields map to correct DB column names."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        expected_cols = {"name", "team", "positions", "is_hitter", "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg"}
        assert expected_cols.issubset(set(df.columns))

    def test_values(self):
        """Numeric values are preserved correctly."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["pa"] == 650
        assert judge["hr"] == 45
        assert judge["avg"] == pytest.approx(0.291)
        assert judge["is_hitter"] is True or judge["is_hitter"] == 1

    def test_position_from_minpos(self):
        """minpos field correctly maps to positions column."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["positions"] == "OF"
        betts = df[df["name"] == "Mookie Betts"].iloc[0]
        assert betts["positions"] == "SS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestNormalizeHitterJson -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.data_pipeline'" or "ImportError"

- [ ] **Step 3: Write minimal implementation — data_pipeline.py skeleton + normalize_hitter_json**

Create `src/data_pipeline.py`:

```python
"""FanGraphs auto-fetch data pipeline.

Fetches Steamer, ZiPS, and Depth Charts projections from FanGraphs'
internal JSON API on app startup. Normalizes JSON to DB schema, upserts
players, stores projections, creates blended projections, and extracts ADP.

CSV upload remains available as a manual fallback.
"""

import logging
import time

import pandas as pd
import requests

from src.database import (
    create_blended_projections,
    get_connection,
    init_db,
    update_refresh_log,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_BASE_URL = "https://www.fangraphs.com/api/projections"
_TIMEOUT = 10  # seconds per request
_RATE_LIMIT = 0.5  # seconds between requests

# FG API type parameter → DB system column value
SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",
}

SYSTEMS = ["steamer", "zips", "fangraphsdc"]


class FetchError(Exception):
    """Raised when an API fetch fails."""


# ── Normalization ──────────────────────────────────────────────────


def normalize_hitter_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON hitter fields to DB schema.

    Sets is_hitter=True. Extracts primary position from 'minpos' field.
    NOTE: FG 'minpos' provides primary position only (e.g., "SS"), not
    multi-position eligibility. This is a known limitation vs CSV import.
    """
    records = []
    for player in raw:
        records.append(
            {
                "name": player.get("PlayerName", ""),
                "team": player.get("Team", ""),
                "positions": player.get("minpos", "Util") or "Util",
                "is_hitter": True,
                "pa": int(player.get("PA", 0) or 0),
                "ab": int(player.get("AB", 0) or 0),
                "h": int(player.get("H", 0) or 0),
                "r": int(player.get("R", 0) or 0),
                "hr": int(player.get("HR", 0) or 0),
                "rbi": int(player.get("RBI", 0) or 0),
                "sb": int(player.get("SB", 0) or 0),
                "avg": float(player.get("AVG", 0) or 0),
            }
        )
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestNormalizeHitterJson -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: data pipeline skeleton + hitter JSON normalization"
```

---

### Task 3: Pitcher normalization with SP/RP classification

**Files:**
- Modify: `src/data_pipeline.py` (add `normalize_pitcher_json`)
- Modify: `tests/test_data_pipeline.py` (add pitcher tests)

- [ ] **Step 1: Write failing tests for pitcher normalization**

Add to `tests/test_data_pipeline.py`:

```python
class TestNormalizePitcherJson:
    def test_column_mapping(self):
        """FG JSON pitcher fields map to correct DB column names."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        expected_cols = {"name", "team", "positions", "is_hitter", "ip", "w", "sv", "k", "era", "whip", "er", "bb_allowed", "h_allowed"}
        assert expected_cols.issubset(set(df.columns))

    def test_so_to_k_mapping(self):
        """FG 'SO' field maps to DB 'k' column."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["k"] == 250

    def test_sp_classification(self):
        """GS >= 5 → 'SP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["positions"] == "SP"

    def test_rp_classification(self):
        """SV >= 3 → 'RP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        hader = df[df["name"] == "Josh Hader"].iloc[0]
        assert hader["positions"] == "RP"

    def test_dual_classification(self):
        """GS >= 1 but < 5, SV < 3 → 'SP,RP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        dual = df[df["name"] == "Dual Starter"].iloc[0]
        assert dual["positions"] == "SP,RP"

    def test_is_hitter_false(self):
        """All pitchers have is_hitter=False."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        assert all(row == False or row == 0 for row in df["is_hitter"])  # noqa: E712
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestNormalizePitcherJson -v`
Expected: FAIL with "ImportError: cannot import name 'normalize_pitcher_json'"

- [ ] **Step 3: Implement normalize_pitcher_json**

Add to `src/data_pipeline.py` after `normalize_hitter_json`:

```python
def normalize_pitcher_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON pitcher fields to DB schema.

    Sets is_hitter=False. Classifies SP/RP using existing logic from
    import_pitcher_csv(): GS >= 5 or IP >= 80 → "SP", SV >= 3 → "RP",
    GS >= 1 → "SP,RP", else → "RP".
    """
    records = []
    for player in raw:
        ip = float(player.get("IP", 0) or 0)
        gs = int(player.get("GS", 0) or 0)
        sv = int(player.get("SV", 0) or 0)

        # SP/RP classification (mirrors database.py import_pitcher_csv)
        if gs >= 5 or ip >= 80:
            positions = "SP"
        elif sv >= 3:
            positions = "RP"
        elif gs >= 1:
            positions = "SP,RP"
        else:
            positions = "RP"

        er = int(player.get("ER", 0) or 0)
        era = float(player.get("ERA", 0) or 0)
        bb_allowed = int(player.get("BB", 0) or 0)
        h_allowed = int(player.get("H", 0) or 0)
        whip = float(player.get("WHIP", 0) or 0)

        # Compute ER from ERA if not available (mirrors database.py)
        if er == 0 and era > 0 and ip > 0:
            er = int(round(era * ip / 9))
        # Compute H/BB from WHIP if both missing
        if h_allowed == 0 and bb_allowed == 0 and whip > 0 and ip > 0:
            total = whip * ip
            bb_allowed = int(total * 0.3)
            h_allowed = int(total * 0.7)

        records.append(
            {
                "name": player.get("PlayerName", ""),
                "team": player.get("Team", ""),
                "positions": positions,
                "is_hitter": False,
                "ip": ip,
                "w": int(player.get("W", 0) or 0),
                "sv": sv,
                "k": int(player.get("SO", 0) or 0),
                "era": era,
                "whip": whip,
                "er": er,
                "bb_allowed": bb_allowed,
                "h_allowed": h_allowed,
            }
        )
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestNormalizePitcherJson -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: pitcher JSON normalization with SP/RP classification"
```

---

## Chunk 2: Fetching + ADP Extraction

### Task 4: System name mapping test

**Files:**
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write test for system name mapping**

Add to `tests/test_data_pipeline.py`:

```python
class TestSystemMapping:
    def test_fangraphsdc_maps_to_depthcharts(self):
        """FG API 'fangraphsdc' must map to DB 'depthcharts'."""
        from src.data_pipeline import SYSTEM_MAP

        assert SYSTEM_MAP["fangraphsdc"] == "depthcharts"

    def test_all_systems_present(self):
        """All 3 projection systems are in SYSTEM_MAP."""
        from src.data_pipeline import SYSTEM_MAP

        assert set(SYSTEM_MAP.keys()) == {"steamer", "zips", "fangraphsdc"}

    def test_steamer_identity(self):
        """Steamer maps to itself."""
        from src.data_pipeline import SYSTEM_MAP

        assert SYSTEM_MAP["steamer"] == "steamer"
```

- [ ] **Step 2: Run test to verify it passes** (constants already exist)

Run: `python -m pytest tests/test_data_pipeline.py::TestSystemMapping -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_pipeline.py
git commit -m "test: system name mapping (fangraphsdc → depthcharts)"
```

---

### Task 5: fetch_projections with mocked HTTP

**Files:**
- Modify: `src/data_pipeline.py` (add `fetch_projections`)
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write failing test for fetch_projections**

Add to `tests/test_data_pipeline.py`:

```python
class TestFetchProjections:
    @patch("src.data_pipeline.requests.get")
    def test_success_hitters(self, mock_get):
        """Successful fetch returns normalized hitter DataFrame + raw JSON."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_HITTER_JSON
        mock_get.return_value = mock_response

        df, raw = fetch_projections("steamer", "bat")
        assert len(df) == 2
        assert "name" in df.columns
        assert df.iloc[0]["is_hitter"] is True or df.iloc[0]["is_hitter"] == 1
        assert raw == SAMPLE_HITTER_JSON

    @patch("src.data_pipeline.requests.get")
    def test_success_pitchers(self, mock_get):
        """Successful fetch returns normalized pitcher DataFrame + raw JSON."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PITCHER_JSON
        mock_get.return_value = mock_response

        df, raw = fetch_projections("steamer", "pit")
        assert len(df) == 3
        assert "k" in df.columns

    @patch("src.data_pipeline.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network error raises FetchError."""
        from src.data_pipeline import FetchError, fetch_projections

        mock_get.side_effect = requests.exceptions.ConnectionError("No network")
        with pytest.raises(FetchError):
            fetch_projections("steamer", "bat")

    @patch("src.data_pipeline.requests.get")
    def test_correct_url_params(self, mock_get):
        """Verifies the correct URL and query params are sent."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        fetch_projections("zips", "bat")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["type"] == "zips"
        assert call_kwargs[1]["params"]["stats"] == "bat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestFetchProjections -v`
Expected: FAIL with "ImportError: cannot import name 'fetch_projections'"

- [ ] **Step 3: Implement fetch_projections**

Add to `src/data_pipeline.py`:

```python
def fetch_projections(system: str, stats: str) -> tuple[pd.DataFrame, list[dict]]:
    """Single API call to FanGraphs projections endpoint.

    Args:
        system: FG API type — "steamer", "zips", or "fangraphsdc"
        stats: "bat" or "pit"

    Returns:
        Tuple of (normalized DataFrame, raw JSON list).
        The raw JSON is preserved for ADP extraction.

    Raises:
        FetchError: On network or parse failure.
    """
    params = {
        "type": system,
        "stats": stats,
        "pos": "all",
        "team": "0",
        "lg": "all",
        "players": "0",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        resp = requests.get(
            _BASE_URL, params=params, headers=headers, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"Failed to fetch {system}/{stats}: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"Invalid JSON from {system}/{stats}: {exc}") from exc

    if stats == "bat":
        return normalize_hitter_json(raw), raw
    else:
        return normalize_pitcher_json(raw), raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestFetchProjections -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: fetch_projections with mocked HTTP + FetchError"
```

---

### Task 6: ADP extraction

**Files:**
- Modify: `src/data_pipeline.py` (add `extract_adp`)
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write failing tests for extract_adp**

Add to `tests/test_data_pipeline.py`:

```python
class TestExtractAdp:
    def test_filters_high_adp(self):
        """ADP >= 999 is excluded."""
        from src.data_pipeline import extract_adp

        raw_with_high_adp = SAMPLE_HITTER_JSON + [
            {"PlayerName": "Nobody", "Team": "FA", "minpos": "Util",
             "PA": 100, "AB": 90, "H": 20, "R": 10, "HR": 2, "RBI": 8,
             "SB": 0, "AVG": 0.222, "ADP": 999},
        ]
        adp_df = extract_adp(raw_with_high_adp, [])
        assert "Nobody" not in adp_df["name"].values

    def test_filters_null_adp(self):
        """Null ADP values are excluded."""
        from src.data_pipeline import extract_adp

        raw_with_null = SAMPLE_HITTER_JSON + [
            {"PlayerName": "Ghost", "Team": "FA", "minpos": "Util",
             "PA": 100, "AB": 90, "H": 20, "R": 10, "HR": 2, "RBI": 8,
             "SB": 0, "AVG": 0.222, "ADP": None},
        ]
        adp_df = extract_adp(raw_with_null, [])
        assert "Ghost" not in adp_df["name"].values

    def test_valid_adp_preserved(self):
        """Valid ADP values are kept."""
        from src.data_pipeline import extract_adp

        adp_df = extract_adp(SAMPLE_HITTER_JSON, SAMPLE_PITCHER_JSON)
        assert len(adp_df) > 0
        judge_adp = adp_df[adp_df["name"] == "Aaron Judge"]["adp"].iloc[0]
        assert judge_adp == pytest.approx(3.5)

    def test_combines_hitters_and_pitchers(self):
        """ADP is extracted from both hitter and pitcher records."""
        from src.data_pipeline import extract_adp

        adp_df = extract_adp(SAMPLE_HITTER_JSON, SAMPLE_PITCHER_JSON)
        names = set(adp_df["name"].values)
        assert "Aaron Judge" in names
        assert "Gerrit Cole" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestExtractAdp -v`
Expected: FAIL with "ImportError: cannot import name 'extract_adp'"

- [ ] **Step 3: Implement extract_adp**

Add to `src/data_pipeline.py`:

```python
def extract_adp(
    hitters_raw: list[dict], pitchers_raw: list[dict]
) -> pd.DataFrame:
    """Pull ADP from raw FanGraphs JSON (Steamer responses are most complete).

    Filters out ADP >= 999 and null values.
    Returns DataFrame with columns: name, adp.
    The caller (_store_adp) resolves name → player_id before DB insert.
    """
    records = []
    for player in hitters_raw + pitchers_raw:
        name = player.get("PlayerName", "")
        adp = player.get("ADP")
        if adp is None or adp >= 999:
            continue
        try:
            adp_val = float(adp)
        except (ValueError, TypeError):
            continue
        records.append({"name": name, "adp": adp_val})

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["name", "adp"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestExtractAdp -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: ADP extraction with filtering (null, >=999)"
```

---

## Chunk 3: Storage + Orchestration

### Task 7: _store_projections and _store_adp

**Files:**
- Modify: `src/data_pipeline.py` (add storage helpers)
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write failing tests for storage**

Add to `tests/test_data_pipeline.py`:

```python
class TestStoreProjections:
    def test_stores_and_resolves_player_ids(self):
        """Players are upserted and projections get valid player_ids."""
        from src.data_pipeline import _store_projections, normalize_hitter_json

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        projections = {"steamer_bat": hitters}
        count = _store_projections(projections)
        assert count == 2

        conn = db_mod.get_connection()
        rows = conn.execute(
            "SELECT p.name, pr.system, pr.hr "
            "FROM projections pr JOIN players p ON pr.player_id = p.player_id "
            "WHERE pr.system = 'steamer'"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        names = {r[0] for r in rows}
        assert "Aaron Judge" in names

    def test_idempotent(self):
        """Storing the same system twice doesn't create duplicates."""
        from src.data_pipeline import _store_projections, normalize_hitter_json

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        projections = {"steamer_bat": hitters}
        _store_projections(projections)
        _store_projections(projections)  # second store

        conn = db_mod.get_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM projections WHERE system = 'steamer'"
        ).fetchone()[0]
        conn.close()
        assert count == 2  # Not 4


class TestStoreAdp:
    def test_stores_with_player_id(self):
        """ADP records are stored with resolved player_ids."""
        from src.data_pipeline import (
            _store_adp,
            _store_projections,
            normalize_hitter_json,
        )

        init_db()
        # Must store players first so ADP can resolve names → player_ids
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        _store_projections({"steamer_bat": hitters})

        adp_df = pd.DataFrame([
            {"name": "Aaron Judge", "adp": 3.5},
            {"name": "Mookie Betts", "adp": 5.0},
        ])
        count = _store_adp(adp_df)
        assert count == 2

        conn = db_mod.get_connection()
        rows = conn.execute("SELECT player_id, adp FROM adp").fetchall()
        conn.close()
        assert len(rows) == 2
        assert all(r[0] is not None and r[0] > 0 for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestStoreProjections tests/test_data_pipeline.py::TestStoreAdp -v`
Expected: FAIL with "ImportError: cannot import name '_store_projections'"

- [ ] **Step 3: Implement _store_projections, _store_adp, _upsert_player**

Add to `src/data_pipeline.py`:

```python
# ── Storage ────────────────────────────────────────────────────────


def _upsert_player(
    cursor, name: str, team: str, positions: str, is_hitter: bool
) -> int:
    """Insert or find a player. Returns player_id.

    Mirrors database.py's _upsert_player() — match on (name, team),
    merge positions if existing.
    """
    cursor.execute(
        "SELECT player_id, positions FROM players WHERE name = ? AND team = ?",
        (name, team),
    )
    result = cursor.fetchone()
    if result:
        existing = set(result[1].split(","))
        new = set(positions.split(","))
        merged = ",".join(sorted(existing | new))
        if merged != result[1]:
            cursor.execute(
                "UPDATE players SET positions = ? WHERE player_id = ?",
                (merged, result[0]),
            )
        return result[0]
    else:
        cursor.execute(
            "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
            (name, team, positions, 1 if is_hitter else 0),
        )
        return cursor.lastrowid


def _store_projections(projections: dict[str, pd.DataFrame]) -> int:
    """Upsert players then store projections in DB.

    Keys are "{db_system}_{bat|pit}" e.g. "steamer_bat", "depthcharts_pit".
    For each system: DELETE existing rows → upsert players → INSERT projections.
    Returns total row count inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    total = 0

    # Collect all unique systems being stored
    systems_stored = set()
    for key in projections:
        db_system = key.rsplit("_", 1)[0]  # "steamer_bat" → "steamer"
        systems_stored.add(db_system)

    # Delete old projections for each system (idempotency)
    for system in systems_stored:
        cursor.execute("DELETE FROM projections WHERE system = ?", (system,))

    # Insert new projections
    for key, df in projections.items():
        db_system = key.rsplit("_", 1)[0]
        for _, row in df.iterrows():
            is_hitter = bool(row.get("is_hitter", True))
            player_id = _upsert_player(
                cursor,
                str(row["name"]),
                str(row.get("team", "")),
                str(row.get("positions", "Util")),
                is_hitter,
            )

            if is_hitter:
                cursor.execute(
                    """INSERT INTO projections
                       (player_id, system, pa, ab, h, r, hr, rbi, sb, avg)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        player_id,
                        db_system,
                        int(row.get("pa", 0)),
                        int(row.get("ab", 0)),
                        int(row.get("h", 0)),
                        int(row.get("r", 0)),
                        int(row.get("hr", 0)),
                        int(row.get("rbi", 0)),
                        int(row.get("sb", 0)),
                        float(row.get("avg", 0)),
                    ),
                )
            else:
                cursor.execute(
                    """INSERT INTO projections
                       (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        player_id,
                        db_system,
                        float(row.get("ip", 0)),
                        int(row.get("w", 0)),
                        int(row.get("sv", 0)),
                        int(row.get("k", 0)),
                        float(row.get("era", 0)),
                        float(row.get("whip", 0)),
                        int(row.get("er", 0)),
                        int(row.get("bb_allowed", 0)),
                        int(row.get("h_allowed", 0)),
                    ),
                )
            total += 1

    conn.commit()
    conn.close()
    return total


def _store_adp(adp_df: pd.DataFrame) -> int:
    """Resolve player names to player_ids and store ADP.

    Uses exact-match against players table (name field).
    DELETE FROM adp before INSERT (idempotency).
    Returns row count inserted.
    """
    if adp_df.empty:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing ADP (idempotency)
    cursor.execute("DELETE FROM adp")

    count = 0
    for _, row in adp_df.iterrows():
        name = str(row["name"])
        adp_val = float(row["adp"])

        # Resolve name → player_id (exact match first, fuzzy fallback)
        cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
        result = cursor.fetchone()
        if result is None:
            # Fuzzy fallback: first + last name LIKE match
            parts = name.split()
            if len(parts) >= 2:
                cursor.execute(
                    "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                    (f"%{parts[0]}%", f"%{parts[-1]}%"),
                )
                result = cursor.fetchone()
        if result is None:
            logger.warning("ADP: no player_id found for '%s', skipping", name)
            continue

        player_id = result[0]
        cursor.execute(
            """INSERT INTO adp (player_id, adp) VALUES (?, ?)
               ON CONFLICT(player_id) DO UPDATE SET adp=excluded.adp""",
            (player_id, adp_val),
        )
        count += 1

    conn.commit()
    conn.close()
    return count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestStoreProjections tests/test_data_pipeline.py::TestStoreAdp -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: projection + ADP storage with idempotent upsert"
```

---

### Task 8: fetch_all_projections + refresh_if_stale orchestrator

**Files:**
- Modify: `src/data_pipeline.py` (add `fetch_all_projections`, `refresh_if_stale`)
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write failing tests for orchestration**

Add to `tests/test_data_pipeline.py`:

```python
class TestFetchAllProjections:
    @patch("src.data_pipeline.fetch_projections")
    def test_partial_failure(self, mock_fetch):
        """1 system fails, 2 succeed → returns dicts with 4 entries."""
        from src.data_pipeline import FetchError, fetch_all_projections

        def side_effect(system, stats):
            if system == "zips":
                raise FetchError("ZiPS unavailable")
            if stats == "bat":
                df = pd.DataFrame({"name": ["Test"], "team": ["TST"],
                                   "positions": ["OF"], "is_hitter": [True],
                                   "pa": [500], "ab": [450], "h": [120],
                                   "r": [70], "hr": [20], "rbi": [60],
                                   "sb": [5], "avg": [0.267]})
                return df, [{"PlayerName": "Test", "ADP": 100}]
            else:
                df = pd.DataFrame({"name": ["Ace"], "team": ["TST"],
                                   "positions": ["SP"], "is_hitter": [False],
                                   "ip": [180.0], "w": [12], "sv": [0],
                                   "k": [200], "era": [3.20], "whip": [1.10],
                                   "er": [64], "bb_allowed": [50],
                                   "h_allowed": [148]})
                return df, [{"PlayerName": "Ace", "ADP": 50}]

        mock_fetch.side_effect = side_effect
        projections, raw_data = fetch_all_projections()
        assert len(projections) == 4
        assert "steamer_bat" in projections
        assert "depthcharts_pit" in projections
        assert "zips_bat" not in projections

    @patch("src.data_pipeline.fetch_projections")
    def test_total_failure(self, mock_fetch):
        """All systems fail → returns empty dicts."""
        from src.data_pipeline import FetchError, fetch_all_projections

        mock_fetch.side_effect = FetchError("down")
        projections, raw_data = fetch_all_projections()
        assert projections == {}
        assert raw_data == {}


class TestRefreshIfStale:
    @patch("src.data_pipeline.fetch_all_projections")
    def test_success(self, mock_fetch_all):
        """Successful refresh returns True and populates DB."""
        from src.data_pipeline import normalize_hitter_json, refresh_if_stale

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        mock_fetch_all.return_value = (
            {"steamer_bat": hitters},
            {"steamer_bat": SAMPLE_HITTER_JSON},
        )

        result = refresh_if_stale(force=True)
        assert result is True

        # Verify data in DB
        conn = db_mod.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM projections").fetchone()[0]
        conn.close()
        assert count > 0

    @patch("src.data_pipeline.fetch_all_projections")
    def test_total_failure_returns_false(self, mock_fetch_all):
        """Total failure returns False."""
        from src.data_pipeline import refresh_if_stale

        init_db()
        mock_fetch_all.return_value = ({}, {})
        result = refresh_if_stale(force=True)
        assert result is False

    @patch("src.data_pipeline.fetch_all_projections")
    def test_skips_when_data_exists(self, mock_fetch_all):
        """When force=False and data exists, skip fetch and return True."""
        from src.data_pipeline import _store_projections, normalize_hitter_json, refresh_if_stale

        init_db()
        # Pre-populate DB with some projections
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        _store_projections({"steamer_bat": hitters})

        result = refresh_if_stale(force=False)
        assert result is True
        # fetch_all_projections should NOT have been called
        mock_fetch_all.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::TestFetchAllProjections tests/test_data_pipeline.py::TestRefreshIfStale -v`
Expected: FAIL with "ImportError: cannot import name 'fetch_all_projections'"

- [ ] **Step 3: Implement fetch_all_projections + refresh_if_stale**

Add to `src/data_pipeline.py`:

```python
# ── Orchestration ──────────────────────────────────────────────────


def fetch_all_projections() -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    """Fetch all 6 endpoints (3 systems × bat/pit).

    Returns:
        Tuple of (projections_dict, raw_json_dict).
        projections_dict: keyed by "{db_system}_{stats}" e.g. "steamer_bat"
        raw_json_dict: same keys, contains raw JSON for ADP extraction
    """
    projections: dict[str, pd.DataFrame] = {}
    raw_data: dict[str, list[dict]] = {}
    first = True

    for fg_system in SYSTEMS:
        db_system = SYSTEM_MAP[fg_system]
        for stats in ("bat", "pit"):
            if not first:
                time.sleep(_RATE_LIMIT)
            first = False

            try:
                df, raw = fetch_projections(fg_system, stats)
                key = f"{db_system}_{stats}"
                projections[key] = df
                raw_data[key] = raw
                logger.info(
                    "Fetched %s/%s: %d players", fg_system, stats, len(df)
                )
            except FetchError as exc:
                logger.warning("Failed to fetch %s/%s: %s", fg_system, stats, exc)

    return projections, raw_data


def refresh_if_stale(force: bool = False) -> bool:
    """Fetch projections if not yet fetched this session.

    Calls init_db() to ensure tables exist.
    When force=False, skips fetch if projections table already has data.
    Orchestrates: fetch → normalize → upsert players → store projections
                  → blend → ADP → refresh log.
    Returns True if data was refreshed or already exists,
    False if all fetches failed.
    """
    init_db()

    # Skip if data already exists and not forcing refresh
    if not force:
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM projections").fetchone()[0]
        conn.close()
        if count > 0:
            return True  # Data already exists, skip fetch

    # Fetch from FanGraphs
    projections, raw_data = fetch_all_projections()
    if not projections:
        logger.error("All FanGraphs fetches failed")
        update_refresh_log("fangraphs_projections", status="failed")
        return False

    # Store projections (upserts players automatically)
    total = _store_projections(projections)
    logger.info("Stored %d projection rows", total)

    # Create blended projections from all available systems
    try:
        create_blended_projections()
        logger.info("Blended projections created")
    except Exception as exc:
        logger.warning("Blended projection creation failed: %s", exc)

    # Extract and store ADP (prefer Steamer data)
    hitters_raw = raw_data.get("steamer_bat", [])
    pitchers_raw = raw_data.get("steamer_pit", [])
    if hitters_raw or pitchers_raw:
        adp_df = extract_adp(hitters_raw, pitchers_raw)
        adp_count = _store_adp(adp_df)
        logger.info("Stored %d ADP records", adp_count)

    # Log success
    update_refresh_log("fangraphs_projections", status="success")
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py::TestFetchAllProjections tests/test_data_pipeline.py::TestRefreshIfStale -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: fetch_all_projections + refresh_if_stale orchestrator"
```

---

## Chunk 4: App Integration + Final

### Task 9: Integrate into app.py Step 1

**Files:**
- Modify: `app.py:713-793` (render_step_import)

- [ ] **Step 1: Add data_pipeline import to app.py**

At the top of `app.py`, after the existing imports (around line 44), add:

```python
try:
    from src.data_pipeline import refresh_if_stale as fetch_projections_from_fg

    HAS_DATA_PIPELINE = True
except ImportError:
    HAS_DATA_PIPELINE = False
```

- [ ] **Step 2: Add auto-fetch block to render_step_import**

In `render_step_import()` (line 713), insert the auto-fetch block **after** the Yahoo connect section and **before** the CSV uploader columns. Insert after line 753 (the `── or ──` divider) and before line 755 (`col1, col2 = st.columns(2)`):

```python
    # ── Auto-Fetch Projections from FanGraphs ──────────────────────────
    if HAS_DATA_PIPELINE:
        if "projections_fetched" not in st.session_state:
            with st.spinner("Fetching latest projections from FanGraphs..."):
                try:
                    result = fetch_projections_from_fg(force=False)
                    st.session_state.projections_fetched = result
                except Exception as exc:
                    logger.warning("Auto-fetch failed: %s", exc)
                    st.session_state.projections_fetched = False

            if st.session_state.projections_fetched:
                st.toast("Projections updated!", icon="✅")

        if st.session_state.get("projections_fetched"):
            st.markdown(
                f'<div style="background:{T["card"]};border:1px solid #2d6a4f;'
                f'border-radius:12px;padding:16px;margin-bottom:16px;">'
                f'<div style="font-family:Oswald,sans-serif;color:#52b788;'
                f'font-size:16px;">✅ Projections Loaded</div>'
                f'<div style="color:{T["tx2"]};font-size:13px;margin-top:4px;">'
                f"Steamer + ZiPS + Depth Charts fetched from FanGraphs</div></div>",
                unsafe_allow_html=True,
            )
            st.session_state.hitter_data = True
            st.session_state.pitcher_data = True

            if st.button("🔄 Refresh Projections", type="secondary"):
                with st.spinner("Refreshing..."):
                    result = fetch_projections_from_fg(force=True)
                    st.session_state.projections_fetched = result
                    if result:
                        st.toast("Projections refreshed!", icon="✅")
                    else:
                        st.warning("Refresh failed. Data may be stale.")
                    st.rerun()
        else:
            st.markdown(
                f'<div style="background:{T["card"]};border:1px solid {T["card_h"]};'
                f'border-radius:12px;padding:16px;margin-bottom:16px;">'
                f'<div style="font-family:Oswald,sans-serif;color:{T["amber"]};'
                f'font-size:16px;">⚠️ Auto-Fetch Unavailable</div>'
                f'<div style="color:{T["tx2"]};font-size:13px;margin-top:4px;">'
                f"Upload CSV files below to load projections manually</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="text-align:center;color:{T["tx2"]};margin:12px 0;">'
            f"── Manual Override ──</div>",
            unsafe_allow_html=True,
        )
```

- [ ] **Step 3: Fix pre-existing CSV import bug (missing system arg)**

On line 767, change:
```python
                import_hitter_csv(hitter_file)
```
to:
```python
                import_hitter_csv(hitter_file, system="steamer")
```

On line 784, change:
```python
                import_pitcher_csv(pitcher_file)
```
to:
```python
                import_pitcher_csv(pitcher_file, system="steamer")
```

- [ ] **Step 4: Add logger import if not present**

Check if `import logging` and `logger = logging.getLogger(__name__)` exist in app.py. If not, add `import logging` to imports and `logger = logging.getLogger(__name__)` after the import block.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: wire auto-fetch into Step 1 + fix CSV import system arg bug"
```

---

### Task 10: Final lint, format, test, docs

**Files:**
- All modified files
- Modify: `CLAUDE.md` (update docs)

- [ ] **Step 1: Run ruff lint and format**

Run: `ruff check . --fix && ruff format .`
Expected: All files clean

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -v`
Expected: All tests pass (existing ~154 + new ~28 = ~182)

- [ ] **Step 3: Update CLAUDE.md**

Add to File Structure section:
```
  data_pipeline.py      — FanGraphs auto-fetch: projections + ADP from internal API
```

Add to tests section:
```
  test_data_pipeline.py     — Data pipeline: fetch, normalize, store, ADP extraction (~28 tests)
```

Update test count from ~154 to ~182.

Add to Architecture > Draft Valuation Pipeline:
```
0. **Auto-fetch** — FanGraphs internal JSON API fetches Steamer/ZiPS/Depth Charts on startup (session-once)
```

Add to Gotchas:
```
- **FG API system mapping** — `type=fangraphsdc` in API → `system='depthcharts'` in DB. SYSTEM_MAP in data_pipeline.py handles this.
- **FG `minpos` = single position** — Auto-fetch gives primary position only, not multi-position eligibility. CSV upload or future enhancement needed for full eligibility.
- **CSV import requires `system` arg** — `import_hitter_csv(path, system)` and `import_pitcher_csv(path, system)` require the system parameter. Pre-existing bug was fixed to default to "steamer".
```

Add to Data Sources:
```
- **Auto-fetch projections:** FanGraphs internal JSON API (`/api/projections`) — Steamer, ZiPS, Depth Charts (fetched on startup)
```

- [ ] **Step 4: Commit all changes**

```bash
git add -A
git commit -m "docs: update CLAUDE.md for auto-fetch data pipeline"
```

- [ ] **Step 5: Push to master**

Run: `git push origin master`

---

## Summary

| Task | What | Tests | Files |
|------|------|-------|-------|
| 1 | Add requests dep | — | requirements.txt |
| 2 | Hitter normalization | 3 | data_pipeline.py, test_data_pipeline.py |
| 3 | Pitcher normalization + SP/RP | 6 | data_pipeline.py, test_data_pipeline.py |
| 4 | System name mapping | 3 | test_data_pipeline.py |
| 5 | fetch_projections (HTTP, returns tuple) | 4 | data_pipeline.py, test_data_pipeline.py |
| 6 | ADP extraction | 4 | data_pipeline.py, test_data_pipeline.py |
| 7 | _store_projections + _store_adp | 3 | data_pipeline.py, test_data_pipeline.py |
| 8 | Orchestrator (fetch_all + refresh) | 5 | data_pipeline.py, test_data_pipeline.py |
| 9 | app.py integration + CSV fix | — | app.py |
| 10 | Lint, test, docs, push | — | CLAUDE.md, all |

**Total new tests:** ~28
**New files:** 2 (src/data_pipeline.py, tests/test_data_pipeline.py)
**Modified files:** 3 (app.py, requirements.txt, CLAUDE.md)
