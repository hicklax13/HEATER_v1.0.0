# Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 29 actionable gaps between the original draft engine spec and the current implementation — data sources, schema, engine output, UI, and infrastructure.

**Architecture:** Additive changes only. 14 new files + 15 modified files. Each new module follows the existing pattern: graceful fallback on failure, staleness-based caching, `try/except ImportError` for optional deps. All scrapers return empty DataFrames on HTTP failure. Engine output enrichment happens in `DraftRecommendationEngine.recommend()`, not in `DraftSimulator`.

**Tech Stack:** Python 3.11+, Streamlit, SQLite, MLB-StatsAPI, pybaseball, requests, beautifulsoup4 (new), schedule (new)

**Spec:** `docs/superpowers/specs/2026-03-18-gap-closure-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `src/marcel.py` | Marcel projection system (local 3yr weighted avg) |
| `src/adp_sources.py` | FantasyPros ECR + NFBC ADP fetchers |
| `src/depth_charts.py` | Roster Resource depth chart scraper |
| `src/contract_data.py` | BB-Ref contract year + arb scraper |
| `src/news_fetcher.py` | News aggregation from MLB Stats API transactions |
| `src/scheduler.py` | Background refresh scheduler |
| `tests/test_marcel.py` | Marcel projection tests |
| `tests/test_adp_sources.py` | ADP fetcher tests |
| `tests/test_depth_charts.py` | Depth chart tests |
| `tests/test_contract_data.py` | Contract data tests |
| `tests/test_news_fetcher.py` | News fetcher tests |
| `tests/test_engine_output.py` | Engine output enrichment tests |
| `tests/test_scheduler.py` | Scheduler tests |
| `docs/ROADMAP.md` | Timeline + done criteria |

### Modified Files
| File | Changes |
|------|---------|
| `src/data_pipeline.py` | Add ATC, THE BAT, THE BAT X to SYSTEM_MAP |
| `src/live_stats.py` | Add `fetch_extended_roster()`, `fetch_current_injuries()` |
| `src/database.py` | ALTER TABLE for 7 new columns |
| `src/data_bootstrap.py` | Wire new data sources into bootstrap phases |
| `src/draft_engine.py` | Add output enrichment (composite_value, position_rank, overall_rank, impact columns, confidence_level, news sentiment) |
| `src/yahoo_api.py` | Add `fetch_yahoo_adp()` |
| `src/engine/signals/statcast.py` | Add `fetch_stuff_plus()` |
| `src/contextual_factors.py` | Accept depth chart lineup slot |
| `src/ui_shared.py` | Add LAST CHANCE badge CSS |
| `app.py` | LAST CHANCE badge, top_n=10, category impact expander |
| `pages/2_Draft_Simulator.py` | Same UI updates |
| `requirements.txt` | Add beautifulsoup4, schedule |
| `CLAUDE.md` | Document new modules |
| `.github/workflows/refresh.yml` | Scheduled data refresh |

---

## Chunk 1: Independent Data Source Modules (Tasks 1-7)

> **Parallel dispatch:** All 7 tasks in this chunk are fully independent — dispatch as parallel agents.

---

### Task 1: Marcel Projections (`src/marcel.py`)

**Gaps:** G7 (Marcel projections)

**Files:**
- Create: `src/marcel.py`
- Create: `tests/test_marcel.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_marcel.py
"""Tests for Marcel projection system."""
import pandas as pd
import pytest
from src.marcel import compute_marcel_projection, marcel_age_adjustment

class TestMarcelProjection:
    def test_three_year_weighting(self):
        """Marcel uses 5/4/3 weighting for recent 3 years."""
        stats = {2025: {"hr": 30}, 2024: {"hr": 25}, 2023: {"hr": 20}}
        result = compute_marcel_projection(stats, stat="hr")
        # Weighted: (30*5 + 25*4 + 20*3) / (5+4+3) = 310/12 = 25.83
        assert result == pytest.approx(25.83, abs=0.1)

    def test_two_years_available(self):
        """With only 2 years, use 5/4 weighting."""
        stats = {2025: {"hr": 30}, 2024: {"hr": 20}}
        result = compute_marcel_projection(stats, stat="hr")
        assert result == pytest.approx(26.67, abs=0.1)

    def test_one_year_available(self):
        """With only 1 year, use that year + regression."""
        stats = {2025: {"hr": 40}}
        result = compute_marcel_projection(stats, stat="hr")
        assert result < 40  # regressed toward mean

    def test_empty_stats_returns_league_avg(self):
        """No historical data returns league average."""
        result = compute_marcel_projection({}, stat="hr")
        assert result > 0  # league average

    def test_regression_to_mean(self):
        """Marcel regresses toward league average."""
        high = compute_marcel_projection({2025: {"hr": 50}}, stat="hr")
        low = compute_marcel_projection({2025: {"hr": 5}}, stat="hr")
        assert high < 50
        assert low > 5

    def test_age_adjustment_young(self):
        """Young hitters get positive age adjustment."""
        adj = marcel_age_adjustment(age=24, is_hitter=True)
        assert adj > 1.0

    def test_age_adjustment_old(self):
        """Old hitters get negative age adjustment."""
        adj = marcel_age_adjustment(age=36, is_hitter=True)
        assert adj < 1.0

    def test_age_adjustment_peak(self):
        """Peak-age players get ~1.0 adjustment."""
        adj = marcel_age_adjustment(age=27, is_hitter=True)
        assert adj == pytest.approx(1.0, abs=0.02)

    def test_rate_stat_weighted_by_pa(self):
        """Rate stats use PA-weighted average, not simple average."""
        stats = {2025: {"avg": 0.300, "pa": 600}, 2024: {"avg": 0.250, "pa": 300}}
        result = compute_marcel_projection(stats, stat="avg", is_rate=True, pa_key="pa")
        # PA-weighted: (0.300*600*5 + 0.250*300*4) / (600*5 + 300*4) = 1200/4200 ≈ 0.286
        assert 0.270 < result < 0.300

    def test_full_player_projection(self):
        """Project all stats for a player."""
        from src.marcel import project_player_marcel
        hist = {
            2025: {"pa": 600, "hr": 30, "r": 90, "rbi": 100, "sb": 10, "avg": 0.280},
            2024: {"pa": 550, "hr": 25, "r": 80, "rbi": 85, "sb": 12, "avg": 0.270},
        }
        proj = project_player_marcel(hist, age=28, is_hitter=True)
        assert "hr" in proj
        assert "avg" in proj
        assert proj["hr"] > 0
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
python -m pytest tests/test_marcel.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.marcel'`

- [ ] **Step 3: Implement `src/marcel.py`**

```python
# src/marcel.py
"""Marcel projection system — local computation from historical stats.

Marcel (named after the monkey) is the simplest useful projection system.
Formula: 3yr weighted average (5/4/3) + regression to league mean + age adjustment.
Reference: Tom Tango's Marcel the Monkey Forecasting System.
"""
import numpy as np

# League average baselines (2023-2025 MLB averages, per 600 PA / 200 IP)
LEAGUE_AVG_HITTING = {
    "pa": 600, "ab": 540, "h": 140, "r": 72, "hr": 18, "rbi": 68,
    "sb": 10, "avg": 0.248, "obp": 0.315, "bb": 52, "hbp": 7, "sf": 4,
}
LEAGUE_AVG_PITCHING = {
    "ip": 170, "w": 9, "l": 9, "sv": 0, "k": 160, "era": 4.20,
    "whip": 1.28, "er": 80, "bb_allowed": 55, "h_allowed": 155,
}
RATE_STATS = {"avg", "obp", "era", "whip"}
WEIGHTS = [5, 4, 3]  # most recent first
REGRESSION_PA = 1200  # PA of regression toward mean (Tango recommendation)
REGRESSION_IP = 450   # IP equivalent for pitchers


def compute_marcel_projection(
    historical_stats: dict[int, dict[str, float]],
    stat: str,
    is_rate: bool = False,
    pa_key: str = "pa",
) -> float:
    """Compute Marcel projection for a single stat.

    Args:
        historical_stats: {year: {stat: value}} — keys are years, most recent first
        stat: stat name to project
        is_rate: if True, weight by PA/IP instead of simple weight
        pa_key: key for playing time denominator (pa or ip)
    """
    league_avg = {**LEAGUE_AVG_HITTING, **LEAGUE_AVG_PITCHING}
    if not historical_stats:
        return league_avg.get(stat, 0.0)

    sorted_years = sorted(historical_stats.keys(), reverse=True)[:3]
    values = []
    weights = []
    pa_values = []

    for i, year in enumerate(sorted_years):
        yr_stats = historical_stats[year]
        val = yr_stats.get(stat, 0.0)
        pa = yr_stats.get(pa_key, 600)
        w = WEIGHTS[i] if i < len(WEIGHTS) else WEIGHTS[-1]
        values.append(val)
        weights.append(w)
        pa_values.append(pa)

    if is_rate and pa_values:
        # PA-weighted average for rate stats
        total_weighted_pa = sum(pa * w for pa, w in zip(pa_values, weights))
        if total_weighted_pa == 0:
            return league_avg.get(stat, 0.0)
        weighted_sum = sum(v * pa * w for v, pa, w in zip(values, pa_values, weights))
        raw = weighted_sum / total_weighted_pa
    else:
        # Simple weighted average for counting stats
        total_weight = sum(weights)
        if total_weight == 0:
            return league_avg.get(stat, 0.0)
        raw = sum(v * w for v, w in zip(values, weights)) / total_weight

    # Regression to mean
    mean_val = league_avg.get(stat, raw)
    total_pa = sum(pa_values)
    reg_pa = REGRESSION_PA if pa_key == "pa" else REGRESSION_IP
    reliability = total_pa / (total_pa + reg_pa)
    regressed = reliability * raw + (1 - reliability) * mean_val

    return regressed


def marcel_age_adjustment(age: int, is_hitter: bool = True) -> float:
    """Age-based multiplier for Marcel projections.

    Peak ages: hitters 27, pitchers 26. Decline: hitters 0.5%/yr, pitchers 0.7%/yr.
    """
    peak = 27 if is_hitter else 26
    decline_rate = 0.005 if is_hitter else 0.007
    delta = age - peak
    if delta <= 0:
        # Young players: slight improvement toward peak
        return 1.0 + abs(delta) * 0.003
    else:
        return max(0.85, 1.0 - delta * decline_rate)


def project_player_marcel(
    historical_stats: dict[int, dict[str, float]],
    age: int = 27,
    is_hitter: bool = True,
) -> dict[str, float]:
    """Project all stats for a single player using Marcel."""
    if is_hitter:
        stats_to_project = ["pa", "ab", "h", "r", "hr", "rbi", "sb", "avg", "obp"]
        pa_key = "pa"
    else:
        stats_to_project = ["ip", "w", "l", "sv", "k", "era", "whip", "er", "bb_allowed", "h_allowed"]
        pa_key = "ip"

    age_adj = marcel_age_adjustment(age, is_hitter)
    result = {}

    for stat in stats_to_project:
        is_rate = stat in RATE_STATS
        val = compute_marcel_projection(historical_stats, stat, is_rate=is_rate, pa_key=pa_key)
        if not is_rate:
            val *= age_adj
        result[stat] = round(val, 3) if is_rate else round(val)

    return result
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
python -m pytest tests/test_marcel.py -v
```
Expected: 10 passed

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff format src/marcel.py tests/test_marcel.py
python -m ruff check src/marcel.py tests/test_marcel.py
git add src/marcel.py tests/test_marcel.py
git commit -m "feat: add Marcel projection system (local 3yr weighted avg)"
```

---

### Task 2: Extended Roster Fetch (`src/live_stats.py`)

**Gaps:** G4 (40-man rosters, spring training invitees)

**Files:**
- Modify: `src/live_stats.py`
- Modify: `src/database.py` (add `roster_type` column)
- Create: tests in existing `tests/test_live_stats.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_live_stats.py`:

```python
class TestExtendedRoster:
    def test_fetch_extended_roster_returns_dataframe(self):
        """Extended roster returns a DataFrame with expected columns."""
        from src.live_stats import fetch_extended_roster
        # Mock the API call
        with patch("src.live_stats.statsapi.get") as mock_get:
            mock_get.return_value = {"people": [
                {"id": 1, "fullName": "Test Player", "primaryPosition": {"abbreviation": "SS"},
                 "currentTeam": {"abbreviation": "NYY"}, "active": True,
                 "batSide": {"code": "R"}, "pitchHand": {"code": "R"}, "birthDate": "1995-01-01"}
            ]}
            df = fetch_extended_roster(season=2026)
            assert "roster_type" in df.columns
            assert len(df) >= 1

    def test_extended_roster_deduplicates_by_mlb_id(self):
        """Players appearing in multiple roster types are deduplicated."""
        from src.live_stats import fetch_extended_roster
        with patch("src.live_stats.statsapi.get") as mock_get:
            mock_get.return_value = {"people": [
                {"id": 1, "fullName": "Dupe Player", "primaryPosition": {"abbreviation": "1B"},
                 "currentTeam": {"abbreviation": "BOS"}, "active": True,
                 "batSide": {"code": "L"}, "pitchHand": {"code": "R"}, "birthDate": "1998-06-15"}
            ]}
            df = fetch_extended_roster(season=2026)
            assert df["mlb_id"].nunique() == len(df)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
python -m pytest tests/test_live_stats.py::TestExtendedRoster -v
```

- [ ] **Step 3: Implement `fetch_extended_roster()` in `src/live_stats.py`**

```python
def fetch_extended_roster(season: int = 2026) -> pd.DataFrame:
    """Fetch extended player pool: active + 40-man + spring training.

    Queries MLB Stats API with multiple gameTypes to capture players
    beyond the active roster. Deduplicates by mlb_id.
    """
    all_players = []
    roster_configs = [
        ("R", "active"),      # Regular season active roster
        ("S", "spring"),      # Spring training roster
    ]

    for game_type, roster_label in roster_configs:
        try:
            data = statsapi.get(
                "sports_players",
                {"season": season, "sportId": 1, "gameType": game_type},
            )
            for p in data.get("people", []):
                if not p.get("active", False):
                    continue
                pos = p.get("primaryPosition", {}).get("abbreviation", "Util")
                team = p.get("currentTeam", {}).get("abbreviation", "FA")
                is_pitcher = pos in ("P", "SP", "RP", "CL")
                all_players.append({
                    "mlb_id": p["id"],
                    "name": p.get("fullName", "Unknown"),
                    "team": team,
                    "positions": _normalize_position(pos),
                    "is_hitter": 0 if is_pitcher else 1,
                    "bats": p.get("batSide", {}).get("code", ""),
                    "throws": p.get("pitchHand", {}).get("code", ""),
                    "birth_date": p.get("birthDate", ""),
                    "roster_type": roster_label,
                })
        except Exception:
            logger.warning("Failed to fetch %s roster for %d", game_type, season)

    if not all_players:
        # Fallback to existing function
        return fetch_all_mlb_players(season)

    df = pd.DataFrame(all_players)
    # Deduplicate — keep first occurrence (active > spring)
    df = df.drop_duplicates(subset="mlb_id", keep="first")
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Add `roster_type` column to database schema**

In `src/database.py`, add to the ALTER TABLE block:

```python
_safe_alter(conn, "players", "roster_type", "TEXT DEFAULT 'active'")
```

- [ ] **Step 5: Run tests — verify they PASS**

```bash
python -m pytest tests/test_live_stats.py::TestExtendedRoster -v
```

- [ ] **Step 6: Commit**

```bash
git add src/live_stats.py src/database.py tests/test_live_stats.py
git commit -m "feat: add extended roster fetch (40-man + spring training)"
```

---

### Task 3: ADP Sources (`src/adp_sources.py`)

**Gaps:** G9 (Yahoo ADP), G10 (NFBC), G11 (FantasyPros ECR)

**Files:**
- Create: `src/adp_sources.py`
- Create: `tests/test_adp_sources.py`
- Modify: `src/database.py` (add `nfbc_adp` column)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adp_sources.py
"""Tests for multi-source ADP fetching."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.adp_sources import (
    fetch_fantasypros_ecr,
    fetch_nfbc_adp,
    compute_composite_adp,
)

class TestFantasyProsECR:
    def test_returns_dataframe_with_required_columns(self):
        result = fetch_fantasypros_ecr()
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "player_name" in result.columns
            assert "ecr_rank" in result.columns

    def test_empty_on_http_failure(self):
        with patch("src.adp_sources.requests.get", side_effect=Exception("HTTP error")):
            result = fetch_fantasypros_ecr()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

class TestNFBCADP:
    def test_returns_dataframe(self):
        result = fetch_nfbc_adp()
        assert isinstance(result, pd.DataFrame)

    def test_empty_on_failure(self):
        with patch("src.adp_sources.requests.get", side_effect=Exception("fail")):
            result = fetch_nfbc_adp()
            assert result.empty

class TestCompositeADP:
    def test_coalesce_priority(self):
        """Yahoo > FantasyPros > NFBC > Steamer."""
        row = {"yahoo_adp": 10, "fantasypros_adp": 15, "nfbc_adp": 20, "adp": 25}
        assert compute_composite_adp(row) == 10

    def test_fallback_to_fantasypros(self):
        row = {"yahoo_adp": None, "fantasypros_adp": 15, "nfbc_adp": 20, "adp": 25}
        assert compute_composite_adp(row) == 15

    def test_fallback_to_steamer(self):
        row = {"yahoo_adp": None, "fantasypros_adp": None, "nfbc_adp": None, "adp": 25}
        assert compute_composite_adp(row) == 25

    def test_all_none_returns_999(self):
        row = {"yahoo_adp": None, "fantasypros_adp": None, "nfbc_adp": None, "adp": None}
        assert compute_composite_adp(row) == 999
```

- [ ] **Step 2: Run tests — FAIL**
- [ ] **Step 3: Implement `src/adp_sources.py`**

Core structure:
```python
"""Multi-source ADP fetching: FantasyPros ECR, NFBC ADP."""
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
USER_AGENT = "Mozilla/5.0 (Fantasy Baseball Draft Tool)"
TIMEOUT = 15

def fetch_fantasypros_ecr() -> pd.DataFrame:
    """Scrape FantasyPros consensus rankings. Returns empty DataFrame on failure."""
    try:
        url = "https://www.fantasypros.com/mlb/rankings/overall.php"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Parse the rankings table
        rows = []
        table = soup.find("table", {"id": "ranking-table"})
        if table is None:
            return pd.DataFrame()
        for tr in table.find_all("tr")[1:]:  # skip header
            cells = tr.find_all("td")
            if len(cells) >= 4:
                rows.append({
                    "ecr_rank": int(cells[0].text.strip()),
                    "player_name": cells[1].text.strip(),
                    "team": cells[2].text.strip(),
                    "position": cells[3].text.strip(),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("FantasyPros ECR fetch failed: %s", e)
        return pd.DataFrame()

def fetch_nfbc_adp() -> pd.DataFrame:
    """Scrape NFBC ADP data. Returns empty DataFrame on failure."""
    try:
        url = "https://nfc.shgn.com/adp/baseball"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = []
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) >= 3:
                rows.append({
                    "player_name": cells[0].text.strip(),
                    "nfbc_adp": float(cells[1].text.strip()),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("NFBC ADP fetch failed: %s", e)
        return pd.DataFrame()

def compute_composite_adp(row: dict) -> float:
    """Compute composite ADP using coalesce priority: Yahoo > FP > NFBC > Steamer."""
    for key in ["yahoo_adp", "fantasypros_adp", "nfbc_adp", "adp"]:
        val = row.get(key)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return float(val)
    return 999.0
```

- [ ] **Step 4: Add `nfbc_adp` column to schema**

```python
# In src/database.py ALTER block
_safe_alter(conn, "adp", "nfbc_adp", "REAL")
```

- [ ] **Step 5: Run tests — PASS**
- [ ] **Step 6: Commit**

```bash
git add src/adp_sources.py tests/test_adp_sources.py src/database.py
git commit -m "feat: add FantasyPros ECR + NFBC ADP sources"
```

---

### Task 4: Depth Charts (`src/depth_charts.py`)

**Gaps:** G17 (Roster Resource), G23 (role_status)

**Files:**
- Create: `src/depth_charts.py`
- Create: `tests/test_depth_charts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_depth_charts.py
"""Tests for depth chart scraper."""
import pytest
from unittest.mock import patch
from src.depth_charts import fetch_depth_charts, classify_role

class TestFetchDepthCharts:
    def test_returns_dict_of_teams(self):
        result = fetch_depth_charts()
        assert isinstance(result, dict)

    def test_empty_on_failure(self):
        with patch("src.depth_charts.requests.get", side_effect=Exception("fail")):
            result = fetch_depth_charts()
            assert result == {}

class TestClassifyRole:
    def test_starter_lineup_slot(self):
        assert classify_role(lineup_slot=3, rotation_slot=None, bullpen_role=None) == "starter"

    def test_closer_role(self):
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="CL") == "closer"

    def test_setup_role(self):
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="SU") == "setup"

    def test_rotation_starter(self):
        assert classify_role(lineup_slot=None, rotation_slot=2, bullpen_role=None) == "starter"

    def test_bench_player(self):
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role=None) == "bench"
```

- [ ] **Step 2: Run tests — FAIL**
- [ ] **Step 3: Implement `src/depth_charts.py`**
- [ ] **Step 4: Run tests — PASS**
- [ ] **Step 5: Commit**

```bash
git add src/depth_charts.py tests/test_depth_charts.py
git commit -m "feat: add depth chart scraper with role classification"
```

---

### Task 5: Contract Data (`src/contract_data.py`)

**Gaps:** G18-G19 (contract/salary), G24 (contract_details, arbitration_eligible)

**Files:**
- Create: `src/contract_data.py`
- Create: `tests/test_contract_data.py`
- Modify: `src/database.py` (add `arbitration_eligible` column)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_contract_data.py
"""Tests for contract year data."""
import pytest
from unittest.mock import patch
from src.contract_data import fetch_contract_year_players, is_contract_year

class TestContractYear:
    def test_returns_set_of_names(self):
        result = fetch_contract_year_players()
        assert isinstance(result, set)

    def test_empty_on_failure(self):
        with patch("src.contract_data.requests.get", side_effect=Exception("fail")):
            result = fetch_contract_year_players()
            assert result == set()

    def test_is_contract_year_true(self):
        assert is_contract_year("Mike Trout", {"Mike Trout", "Shohei Ohtani"}) is True

    def test_is_contract_year_false(self):
        assert is_contract_year("Unknown Player", {"Mike Trout"}) is False

    def test_case_insensitive_match(self):
        assert is_contract_year("mike trout", {"Mike Trout"}) is True
```

- [ ] **Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: add contract year data from BB-Ref free agent list"
```

---

### Task 6: News Fetcher (`src/news_fetcher.py`)

**Gaps:** G20 (news API integration), G32 (spring training signal)

**Files:**
- Create: `src/news_fetcher.py`
- Create: `tests/test_news_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_news_fetcher.py
"""Tests for news fetching and aggregation."""
import pytest
from unittest.mock import patch
from src.news_fetcher import fetch_recent_transactions, aggregate_player_news

class TestFetchTransactions:
    def test_returns_list_of_dicts(self):
        result = fetch_recent_transactions(days_back=7)
        assert isinstance(result, list)

    def test_empty_on_failure(self):
        with patch("src.news_fetcher.statsapi.get", side_effect=Exception("fail")):
            result = fetch_recent_transactions()
            assert result == []

class TestAggregatePlayerNews:
    def test_maps_player_id_to_news_items(self):
        transactions = [
            {"player_name": "Test Player", "description": "Placed on IL with elbow strain"},
        ]
        player_map = {"Test Player": 42}
        result = aggregate_player_news(transactions, player_map)
        assert 42 in result
        assert len(result[42]) > 0

    def test_unmatched_players_excluded(self):
        transactions = [{"player_name": "Unknown Guy", "description": "Traded to NL"}]
        result = aggregate_player_news(transactions, {})
        assert result == {}
```

- [ ] **Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: add news fetcher from MLB Stats API transactions"
```

---

### Task 7: Engine Output Enrichment (`src/draft_engine.py`)

**Gaps:** G27 (composite 0-100), G28 (position rank), G29 (overall rank), G30 (category impact), G31 (confidence level)

**Files:**
- Modify: `src/draft_engine.py` — add `_enrich_output()` method
- Create: `tests/test_engine_output.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine_output.py
"""Tests for engine output enrichment fields."""
import numpy as np
import pandas as pd
import pytest
from src.draft_engine import DraftRecommendationEngine
from src.valuation import LeagueConfig

def _make_candidates():
    """Create mock recommendation output."""
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Player A", "Player B", "Player C"],
        "positions": ["SS", "SS,2B", "OF"],
        "combined_score": [8.5, 6.2, 7.0],
        "mc_mean_sgp": [5.0, 3.5, 4.2],
        "mc_std_sgp": [0.5, 1.2, 0.3],
        "pick_score": [7.0, 5.5, 6.0],
    })

class TestCompositeValue:
    def test_normalized_to_0_100(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        assert enriched["composite_value"].max() == 100.0
        assert enriched["composite_value"].min() == 0.0

    def test_single_candidate_gets_50(self):
        df = _make_candidates().head(1)
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        assert enriched["composite_value"].iloc[0] == 50.0

class TestPositionRank:
    def test_ss_ranked_among_ss(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        # Player A (SS only) and Player B (SS,2B) both rank at SS
        assert "SS:" in enriched.loc[0, "position_rank"]

    def test_multi_position_ranked_in_both(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        rank_b = enriched.loc[1, "position_rank"]
        assert "SS:" in rank_b
        assert "2B:" in rank_b

class TestOverallRank:
    def test_rank_is_sequential(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        assert list(enriched["overall_rank"]) == [1, 2, 3]

class TestConfidenceLevel:
    def test_low_cv_is_high_confidence(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        # Player C: mc_std=0.3, mc_mean=4.2 → cv=0.071 → HIGH
        assert enriched.loc[2, "confidence_level"] == "HIGH"

    def test_high_cv_is_low_confidence(self):
        df = _make_candidates()
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        # Player B: mc_std=1.2, mc_mean=3.5 → cv=0.343 → MEDIUM
        assert enriched.loc[1, "confidence_level"] == "MEDIUM"

    def test_zero_mean_is_low_confidence(self):
        df = _make_candidates()
        df.loc[0, "mc_mean_sgp"] = 0
        engine = DraftRecommendationEngine(LeagueConfig(), mode="quick")
        enriched = engine._enrich_output(df)
        assert enriched.loc[0, "confidence_level"] == "LOW"
```

- [ ] **Step 2: Run tests — FAIL**
- [ ] **Step 3: Implement `_enrich_output()` in `src/draft_engine.py`**

Add after `recommend()` method:

```python
def _enrich_output(self, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add composite_value, position_rank, overall_rank, confidence_level."""
    if candidates.empty:
        return candidates

    df = candidates.copy()

    # G29: Overall rank (trivial — already sorted by combined_score)
    df["overall_rank"] = range(1, len(df) + 1)

    # G27: Composite value 0-100
    scores = df["combined_score"]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        df["composite_value"] = ((scores - s_min) / (s_max - s_min) * 100).round(1)
    else:
        df["composite_value"] = 50.0

    # G28: Position rank
    pos_ranks = []
    for _, row in df.iterrows():
        positions = str(row.get("positions", "Util")).split(",")
        parts = []
        for pos in positions:
            pos = pos.strip()
            mask = df["positions"].str.contains(pos, na=False)
            rank_in_pos = (df.loc[mask, "combined_score"] >= row["combined_score"]).sum()
            parts.append(f"{pos}:{rank_in_pos}")
        pos_ranks.append("/".join(parts))
    df["position_rank"] = pos_ranks

    # G31: Confidence level
    mc_std = pd.to_numeric(df.get("mc_std_sgp", 0), errors="coerce").fillna(0)
    mc_mean = pd.to_numeric(df.get("mc_mean_sgp", 0), errors="coerce").fillna(0)
    cv = mc_std / mc_mean.clip(lower=0.01)
    df["confidence_level"] = pd.cut(
        cv, bins=[-float("inf"), 0.15, 0.35, float("inf")],
        labels=["HIGH", "MEDIUM", "LOW"],
    ).astype(str)

    return df
```

Then modify `recommend()` to call `_enrich_output()` before returning:

```python
# At end of recommend(), before return:
candidates = self._enrich_output(candidates)
```

- [ ] **Step 4: Run tests — PASS**
- [ ] **Step 5: Commit**

```bash
git add src/draft_engine.py tests/test_engine_output.py
git commit -m "feat: add engine output enrichment (composite, rank, confidence)"
```

---

## Chunk 2: Integration + Schema + Statcast (Tasks 8-11)

> **Parallel dispatch:** Tasks 8, 9, 10, 11 can run in parallel.

---

### Task 8: FanGraphs Projection Systems Expansion (`src/data_pipeline.py`)

**Gaps:** G5 (ATC), G6 (THE BAT/BAT X)

**Files:**
- Modify: `src/data_pipeline.py` — expand SYSTEM_MAP

- [ ] **Step 1: Expand SYSTEM_MAP**

```python
# In src/data_pipeline.py, replace SYSTEM_MAP:
SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",
    "atc": "atc",
    "thebat": "thebat",
    "thebatx": "thebatx",
}
```

- [ ] **Step 2: Update SYSTEMS list derivation** (already `SYSTEMS = list(SYSTEM_MAP.keys())` — auto-expands)

- [ ] **Step 3: Run existing data_pipeline tests**

```bash
python -m pytest tests/test_data_pipeline.py -v
```
Expected: All pass (SYSTEM_MAP change is additive)

- [ ] **Step 4: Commit**

```bash
git add src/data_pipeline.py
git commit -m "feat: add ATC, THE BAT, THE BAT X projection systems"
```

---

### Task 9: Schema Migrations (`src/database.py`)

**Gaps:** G21-G22 (FanGraphs/Yahoo IDs), G23-G25 (role_status, contract_details, spring_training_stats)

**Files:**
- Modify: `src/database.py`

- [ ] **Step 1: Add ALTER TABLE statements**

In the `_ensure_schema()` function, add:

```python
# Cross-reference IDs
_safe_alter(conn, "players", "fangraphs_id", "TEXT")
_safe_alter(conn, "players", "yahoo_id", "TEXT")

# Role and contract
_safe_alter(conn, "players", "roster_type", "TEXT DEFAULT 'active'")
_safe_alter(conn, "players", "role_status", "TEXT")
_safe_alter(conn, "players", "contract_details", "TEXT")
_safe_alter(conn, "players", "arbitration_eligible", "INTEGER DEFAULT 0")
_safe_alter(conn, "players", "spring_training_stats", "TEXT")

# Statcast Stuff+
_safe_alter(conn, "projections", "stuff_plus", "REAL")
_safe_alter(conn, "projections", "location_plus", "REAL")
_safe_alter(conn, "projections", "pitching_plus", "REAL")
```

- [ ] **Step 2: Run schema tests**

```bash
python -m pytest tests/test_database_schema.py -v
```

- [ ] **Step 3: Commit**

```bash
git add src/database.py
git commit -m "feat: add 10 new schema columns (IDs, role, contract, Stuff+)"
```

---

### Task 10: Statcast Stuff+ / Location+ / Pitching+ (`src/engine/signals/statcast.py`)

**Gaps:** G16

**Files:**
- Modify: `src/engine/signals/statcast.py`

- [ ] **Step 1: Add `fetch_stuff_plus()` function**

```python
def fetch_stuff_plus(season: int = 2026) -> pd.DataFrame:
    """Scrape Stuff+/Location+/Pitching+ from Baseball Savant.

    Returns DataFrame with columns: player_name, stuff_plus, location_plus, pitching_plus.
    Returns empty DataFrame on failure.
    """
    try:
        from bs4 import BeautifulSoup
        url = f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?year={season}&min=50&type=overall"
        resp = requests.get(url, headers={"User-Agent": "Fantasy Draft Tool"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Parse table — columns vary by year
        rows = []
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()
        headers = [th.text.strip().lower() for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) >= len(headers):
                row_dict = dict(zip(headers, cells))
                rows.append({
                    "player_name": row_dict.get("player", ""),
                    "stuff_plus": _safe_float(row_dict.get("stuff+", "")),
                    "location_plus": _safe_float(row_dict.get("location+", "")),
                    "pitching_plus": _safe_float(row_dict.get("pitching+", "")),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("Stuff+ fetch failed: %s", e)
        return pd.DataFrame()

def _safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 2: Run existing statcast tests**
- [ ] **Step 3: Commit**

```bash
git add src/engine/signals/statcast.py
git commit -m "feat: add Stuff+/Location+/Pitching+ scraper from Savant"
```

---

### Task 11: UI Polish (`app.py`, `pages/2_Draft_Simulator.py`)

**Gaps:** G33 (LAST CHANCE badge), G34 (top 10)

**Files:**
- Modify: `app.py`
- Modify: `pages/2_Draft_Simulator.py`
- Modify: `src/ui_shared.py`

- [ ] **Step 1: Add LAST CHANCE CSS to `src/ui_shared.py`**

```css
.badge-last-chance {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background: #e63946;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: 8px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
```

- [ ] **Step 2: Add LAST CHANCE badge to `app.py` hero card**

In `render_hero_pick()`, after survival gauge, add:

```python
last_chance_html = ""
if surv < 0.20:
    last_chance_html = '<span class="badge-last-chance">LAST CHANCE</span>'
```

Insert `{last_chance_html}` into the hero card HTML after `{inj_prob_html}`.

- [ ] **Step 3: Change `top_n` from 8 to 10**

In `app.py` recommend() call:
```python
top_n=10,  # was 8
```

In `pages/2_Draft_Simulator.py` recommend() call:
```python
top_n=10,  # was 8
```

- [ ] **Step 4: Adjust alternatives grid for 9 cards (3×3)**

```python
# In render_alternatives(), change:
cols = st.columns(min(len(alts), 3))  # 3 columns instead of 5
# Show up to 9 alternatives (indices 1-9)
for i, (_, row) in enumerate(alts.iterrows()):
    if i >= 9:
        break
    with cols[i % 3]:
```

- [ ] **Step 5: Commit**

```bash
git add app.py pages/2_Draft_Simulator.py src/ui_shared.py
git commit -m "feat: add LAST CHANCE badge + top 10 recommendations"
```

---

## Chunk 3: Bootstrap Integration + Infrastructure (Tasks 12-15)

> **Sequential:** Task 12 depends on Tasks 1-6. Tasks 13-15 are independent of each other after Task 12.

---

### Task 12: Bootstrap Integration (`src/data_bootstrap.py`)

**Gaps:** Wires Tasks 1-6 into bootstrap pipeline

**Files:**
- Modify: `src/data_bootstrap.py`

- [ ] **Step 1: Add new bootstrap phases**

Add after existing 7 phases:

```python
def _bootstrap_extended_roster(progress):
    """Phase 1b: Extended roster (40-man + spring training)."""
    from src.live_stats import fetch_extended_roster
    if check_staleness("extended_roster", 168):
        progress.phase = "Extended Roster"
        progress.detail = "Fetching 40-man + spring training..."
        df = fetch_extended_roster()
        if not df.empty:
            upsert_player_bulk(df.to_dict("records"))
            return f"Extended roster: {len(df)} players"
    return "Extended roster: fresh"

def _bootstrap_adp_sources(progress):
    """Phase 8: Multi-source ADP."""
    from src.adp_sources import fetch_fantasypros_ecr, fetch_nfbc_adp
    if check_staleness("adp_sources", 24):
        progress.phase = "ADP Sources"
        progress.detail = "Fetching FantasyPros + NFBC..."
        # Implementation: fetch, fuzzy match names, upsert to adp table
        return "ADP sources: updated"
    return "ADP sources: fresh"

def _bootstrap_depth_charts(progress):
    """Phase 9: Depth charts."""
    from src.depth_charts import fetch_depth_charts
    if check_staleness("depth_charts", 168):
        progress.phase = "Depth Charts"
        progress.detail = "Fetching Roster Resource..."
        data = fetch_depth_charts()
        return f"Depth charts: {len(data)} teams"
    return "Depth charts: fresh"

def _bootstrap_contracts(progress):
    """Phase 10: Contract year data."""
    from src.contract_data import fetch_contract_year_players
    if check_staleness("contracts", 720):
        progress.phase = "Contract Data"
        progress.detail = "Fetching BB-Ref free agents..."
        names = fetch_contract_year_players()
        return f"Contract data: {len(names)} players"
    return "Contract data: fresh"

def _bootstrap_news(progress):
    """Phase 11: Recent news/transactions."""
    from src.news_fetcher import fetch_recent_transactions
    if check_staleness("news", 6):
        progress.phase = "News"
        progress.detail = "Fetching recent transactions..."
        items = fetch_recent_transactions()
        return f"News: {len(items)} items"
    return "News: fresh"
```

- [ ] **Step 2: Wire into `bootstrap_all_data()` orchestrator**

Add phases 8-11 after phase 7 (Yahoo sync).

- [ ] **Step 3: Run bootstrap tests**

```bash
python -m pytest tests/test_data_bootstrap.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/data_bootstrap.py
git commit -m "feat: wire 5 new data sources into bootstrap pipeline"
```

---

### Task 13: Scheduler (`src/scheduler.py`)

**Gaps:** G26 (automated update pipeline)

**Files:**
- Create: `src/scheduler.py`
- Create: `tests/test_scheduler.py`
- Create: `.github/workflows/refresh.yml`

- [ ] **Step 1: Write scheduler + tests**

```python
# src/scheduler.py
"""Background data refresh scheduler.

Runs staleness checks on a schedule. Safe to call in Streamlit —
uses a daemon thread that dies with the main process.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_scheduler_running = False
_scheduler_thread = None

REFRESH_INTERVALS = {
    "live_stats": 3600,      # 1h
    "yahoo_sync": 21600,     # 6h
    "adp_sources": 86400,    # 24h
    "projections": 604800,   # 7d
    "depth_charts": 604800,  # 7d
}

def start_background_refresh():
    """Start background refresh thread (idempotent)."""
    global _scheduler_running, _scheduler_thread
    if _scheduler_running:
        return
    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_refresh_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("Background refresh scheduler started")

def stop_background_refresh():
    global _scheduler_running
    _scheduler_running = False

def _refresh_loop():
    """Main scheduler loop — checks staleness every 60s."""
    from src.data_bootstrap import bootstrap_all_data
    while _scheduler_running:
        try:
            bootstrap_all_data(force=False)
        except Exception as e:
            logger.warning("Background refresh error: %s", e)
        time.sleep(300)  # Check every 5 minutes
```

- [ ] **Step 2: Create GitHub Actions scheduled workflow**

```yaml
# .github/workflows/refresh.yml
name: Scheduled Data Refresh
on:
  schedule:
    - cron: '17 9 * * *'  # Daily at 9:17 AM UTC
  workflow_dispatch:
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install -r requirements.txt
      - run: python -c "from src.data_bootstrap import bootstrap_all_data; bootstrap_all_data(force=True)"
```

- [ ] **Step 3: Tests + commit**

```bash
git add src/scheduler.py tests/test_scheduler.py .github/workflows/refresh.yml
git commit -m "feat: add background refresh scheduler + GitHub Actions cron"
```

---

### Task 14: Backtesting Framework

**Gaps:** G35

**Files:**
- Create: `tests/backtest/backtest_2025.py`

- [ ] **Step 1: Implement backtest runner**

```python
# tests/backtest/backtest_2025.py
"""Backtesting framework: compare 2025 preseason recommendations vs actual results.

Usage: python -m tests.backtest.backtest_2025
"""
import json
import sys
from pathlib import Path
from scipy.stats import spearmanr
# ... implementation that loads 2025 data, runs engine, computes accuracy metrics
```

- [ ] **Step 2: Commit**

```bash
git add tests/backtest/backtest_2025.py
git commit -m "feat: add backtesting framework for 2025 season validation"
```

---

### Task 15: Documentation + Requirements

**Gaps:** G36-G37 (timeline, done criteria), plus requirements.txt updates

**Files:**
- Create: `docs/ROADMAP.md`
- Modify: `requirements.txt`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create ROADMAP.md**

```markdown
# HEATER Roadmap

## Completed
- Phase 0: League config (Dec 2025)
- Phase 1: Data pipeline + bootstrap (Jan 2026)
- Phase 2: Draft recommendation engine (Mar 2026)
- Phase 3: Live draft interface (Mar 2026)
- Phase 4: Planning + testing (Mar 2026)
- Phase 5: Gap closure (Mar 2026)

## Acceptance Criteria
- ≥1,000 players in pool (currently ~750, target ~1,200 with extended roster)
- ≥5 projection systems blended (Steamer, ZiPS, DC, ATC, THE BAT, THE BAT X, Marcel)
- ≥2 ADP sources (FG Steamer + FantasyPros ECR)
- All 8 draft board output fields present
- LAST CHANCE badge + top 10 display
- Background scheduler running
- ≥1,200 tests passing
```

- [ ] **Step 2: Update requirements.txt**

```
beautifulsoup4>=4.12.0
schedule>=1.2.0
```

- [ ] **Step 3: Update CLAUDE.md with new modules**

Add entries for: `src/marcel.py`, `src/adp_sources.py`, `src/depth_charts.py`, `src/contract_data.py`, `src/news_fetcher.py`, `src/scheduler.py`

- [ ] **Step 4: Commit**

```bash
git add docs/ROADMAP.md requirements.txt CLAUDE.md
git commit -m "docs: add ROADMAP, update requirements + CLAUDE.md for gap closure"
```

---

## Final Integration Test

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest --tb=line -q
```
Expected: ≥1,200 passed (current 1,108 + ~100 new tests)

- [ ] **Step 2: Run lint**

```bash
python -m ruff check .
python -m ruff format --check .
```

- [ ] **Step 3: Push to master**

```bash
git push
```

- [ ] **Step 4: Verify CI green**

```bash
gh run list --limit 1
```
