# Trade Intelligence V2 — 8-Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all existing intelligence modules into the trade pipeline, fix pre-existing bugs, enhance the daily refresh, add opponent tracking, and prepare the system for mid-season signal intelligence.

**Architecture:** Tasks 1-3 and 5 are independent and can be dispatched as parallel subagents. Tasks 4, 6, 7, 8 each build on the foundation but are also independent of each other. Task 2 (browser testing) should run after Task 1 and 3 are complete.

**Tech Stack:** Python 3.13, Streamlit, SQLite, pandas, NumPy, SciPy, pytest, GitHub Actions

**Parallelization Map:**
```
Parallel Group A (independent):  Task 1, Task 3, Task 4, Task 5
Sequential:                      Task 2 (after Task 1 + 3)
Parallel Group B (independent):  Task 6, Task 7, Task 8
```

---

### Task 1: Wire Trade Analyzer Page to Trade Intelligence

**Files:**
- Modify: `pages/3_Trade_Analyzer.py` (lines 240-275)
- Test: `python -m pytest tests/test_trade_engine.py -q` (existing tests must still pass)

**Context:** The Trade Analyzer page calls `evaluate_trade()` from the engine but doesn't apply health adjustment, category weighting, or FA gating to the player pool before evaluation. The Trade Finder already does this via `find_trade_opportunities()` in `src/trade_finder.py`. We need to apply the same pre-processing to the Trade Analyzer's player pool.

- [ ] **Step 1: Add trade_intelligence import to Trade Analyzer page**

In `pages/3_Trade_Analyzer.py`, add after the existing imports (around line 15):

```python
try:
    from src.trade_intelligence import get_health_adjusted_pool, apply_scarcity_flags
    TRADE_INTEL_AVAILABLE = True
except ImportError:
    TRADE_INTEL_AVAILABLE = False
```

- [ ] **Step 2: Apply health adjustment + scarcity to pool before evaluate_trade**

In `pages/3_Trade_Analyzer.py`, find the line `pool = load_player_pool()` (line 68) and add after `pool = coerce_numeric_df(pool)`:

```python
# Apply trade intelligence: health-adjust projections and add scarcity flags
if TRADE_INTEL_AVAILABLE:
    try:
        pool = get_health_adjusted_pool(pool)
        pool = apply_scarcity_flags(pool)
    except Exception:
        pass  # Graceful degradation — use raw pool if intelligence fails
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_trade_engine.py tests/test_trade_intelligence.py -q --tb=short`
Expected: All tests pass (no new tests needed — this is a wiring change)

- [ ] **Step 4: Lint and format**

Run: `python -m ruff check pages/3_Trade_Analyzer.py --fix && python -m ruff format pages/3_Trade_Analyzer.py`

- [ ] **Step 5: Commit**

```bash
git add pages/3_Trade_Analyzer.py
git commit -m "feat: wire Trade Analyzer page to trade intelligence module

Apply health-adjusted projections and scarcity flags to the player
pool before calling evaluate_trade(). IL/DTD players now get reduced
valuations, closers get scarcity premium, NA players are excluded."
```

---

### Task 2: Live Browser Test of Enhanced Trade Finder

**Files:**
- No code changes — this is a verification task
- Requires: `streamlit run app.py` running on localhost:8501

**Context:** After Tasks 1 and 3, verify that the Trade Finder's elite player protection and category weight cap produce sensible recommendations. The previous test showed "trade Yordan Alvarez for Trevor Story" which should now be blocked.

- [ ] **Step 1: Start the app**

Run: `streamlit run app.py --server.port 8501 --server.headless true`

- [ ] **Step 2: Navigate to Trade Finder and check By Value tab**

Using Playwright browser automation:
1. Navigate to `http://localhost:8501/Trade_Finder`
2. Wait for page load (scan takes ~8-10 seconds)
3. Click "By Value" tab
4. Read the top 5 trade recommendations
5. Verify: No elite player (Judge, Alvarez, Seager) is being traded for a scrub
6. Verify: "Health" and "FA Alt" columns appear in the table

- [ ] **Step 3: Check Trade Readiness tab**

1. Click "Trade Readiness" tab
2. Verify: No duplicate players
3. Verify: Health scores differentiate (IL players show < 85)
4. Verify: Position filter dropdown works
5. Verify: All 9 columns render (Player, Pos, Readiness, Cat Fit, Proj Conf, Health, Scarcity, FA Edge, Best FA)

- [ ] **Step 4: Check By Partner tab**

1. Click "By Partner" tab
2. Expand BUBBA CROSBY trades
3. Verify: "Health" and "FA Alt" columns visible
4. Verify: No NA/minors players in recommendations

- [ ] **Step 5: Catalog any issues found**

If issues found, create a bug report in `docs/AUDIT_TRADE_FINDER_V2.md` with format:
```
### BUG-NNN: [Description]
- File: [path:line]
- Severity: CRITICAL/HIGH/MEDIUM/LOW
- Description: [what's wrong]
- Fix: [suggested fix]
```

---

### Task 3: Fix Pre-Existing Errors (My Team News + Standings Columns)

**Files:**
- Modify: `pages/1_My_Team.py` (line 185, 257)
- Modify: `pages/8_Standings.py` (lines 233-244)
- Test: `python -m pytest -x -q --tb=short` (full suite)

**Context:** Two pre-existing errors found during browser testing:
1. `_render_news_tab` crashes with `'NoneType' object has no attribute 'replace'` when `detail` is `None`
2. Standings table shows LOSSES/WINS/PERCENTAGE columns alongside category stats because Yahoo's `sync_to_db()` writes W/L metadata as "categories"

#### Fix 3a: News tab NoneType error

- [ ] **Step 1: Fix None detail in _render_news_card**

In `pages/1_My_Team.py`, change line 185 from:
```python
    detail = news_item.get("detail", "")
```
to:
```python
    detail = news_item.get("detail") or ""
```

This handles the case where `"detail"` key exists but has value `None`. The `or ""` converts `None` to empty string.

#### Fix 3b: Standings metadata columns

- [ ] **Step 2: Filter standings pivot to only show scoring categories**

In `pages/8_Standings.py`, after line 232 (`standings_df = yds.get_standings()`), add filtering before the pivot:

```python
            if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
                # Filter to only scoring categories (exclude Yahoo metadata like WINS, LOSSES, etc.)
                from src.valuation import LeagueConfig
                _scoring_cats = set(c.upper() for c in LeagueConfig().all_categories)
                standings_df = standings_df[standings_df["category"].str.upper().isin(_scoring_cats)]
```

Replace the existing `if` block at line 233 with the above (add the filtering before the existing pivot logic).

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_database_queries.py -q --tb=short`
Expected: All pass

- [ ] **Step 4: Lint and commit**

```bash
python -m ruff check pages/1_My_Team.py pages/8_Standings.py --fix
python -m ruff format pages/1_My_Team.py pages/8_Standings.py
git add pages/1_My_Team.py pages/8_Standings.py
git commit -m "fix: news tab NoneType error and standings metadata columns

1. _render_news_card: use 'or' to convert None detail to empty string
2. Standings pivot: filter to scoring categories only, excluding
   Yahoo metadata columns (WINS, LOSSES, PERCENTAGE, etc.)"
```

---

### Task 4: Enhance Daily Refresh GitHub Action

**Files:**
- Modify: `.github/workflows/refresh.yml`

**Context:** The daily refresh runs `bootstrap_all_data(force=True)` which fetches MLB data but doesn't trigger the Bayesian ROS updater. The `update_ros_projections()` function in `src/bayesian.py` blends historical stats with 2026 live data — running it daily keeps projections current.

- [ ] **Step 1: Add Bayesian ROS step to refresh.yml**

In `.github/workflows/refresh.yml`, add after the existing "Initialize database" step (after line 27):

```yaml
      - name: Update Bayesian ROS projections
        run: |
          python -c "
          from src.database import init_db
          init_db()
          try:
              from src.bayesian import update_ros_projections
              count = update_ros_projections()
              print(f'Bayesian ROS: updated {count} projections')
          except Exception as e:
              print(f'Bayesian ROS update failed (non-fatal): {e}')
          "
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/refresh.yml
git commit -m "feat: add Bayesian ROS updater to daily refresh Action

Runs update_ros_projections() after bootstrap to keep player
projections current with live 2026 stats. Non-fatal on error
so the refresh Action still succeeds even if Bayesian fails."
```

---

### Task 5: Update CLAUDE.md with Trade Intelligence Architecture

**Files:**
- Modify: `CLAUDE.md`

**Context:** The `trade_intelligence.py` module, Trade Readiness tab, elite player protection, and category weight cap are not documented in CLAUDE.md. Future sessions need this context.

- [ ] **Step 1: Add trade_intelligence.py to file structure**

In CLAUDE.md file structure section, add after the `yahoo_data_service.py` entry:

```
  trade_intelligence.py — Trade valuation layer: health, scarcity, FA gating, Trade Readiness
```

- [ ] **Step 2: Add Trade Intelligence architecture section**

After the "Yahoo Data Service" section, add:

```markdown
### Trade Intelligence Layer
All trade valuations (Trade Finder + Trade Analyzer) flow through `src/trade_intelligence.py`:
1. **Health adjustment** — IL15=0.84x, IL60=0.55x, DTD=0.95x, NA excluded. Uses `injury_model.compute_health_score()` with 3-year history + current Yahoo IL/DTD status from `league_rosters.status` column.
2. **Category-weighted SGP** — `category_gap_analysis()` computes marginal weights per category. Categories where you can gain standings positions weighted higher. Punted categories get zero weight. Max weight capped at 3x average to prevent single-category dominance.
3. **FA gating** — Flags trades where a free agent at >= 70% of the trade target's value exists. Prevents wasting trade capital when a comparable FA is available.
4. **Closer scarcity premium** — SV >= 5 players get 1.3x multiplier. C/SS/2B positions get 1.15x VORP premium.
5. **Elite player protection** — Players in the top 20% by raw SGP require the return to be >= 50% as valuable. Prevents trading stars for role players.
6. **Trade Readiness tab** — 0-100 composite: 40% category fit + 25% projection confidence + 15% health + 10% scarcity + 10% FA advantage.
```

- [ ] **Step 3: Add key API signatures**

In the Key API Signatures section, add:

```python
# Trade Intelligence (src/trade_intelligence.py)
get_health_adjusted_pool(player_pool, config) -> pd.DataFrame  # IL/DTD/NA adjusted
get_category_weights(user_team_name, all_team_totals, config) -> dict[str, float]
compute_fa_comparisons(opp_ids, user_ids, fa_pool, pool, config) -> dict[int, dict]
apply_scarcity_flags(player_pool) -> pd.DataFrame  # adds is_closer, scarcity_mult
compute_trade_readiness(player_id, ...) -> dict  # 0-100 composite score
compute_trade_readiness_batch(player_ids, ...) -> pd.DataFrame  # batch scoring
```

- [ ] **Step 4: Update test count**

Change the test count to reflect current state (2139 passing).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add trade intelligence architecture to CLAUDE.md"
```

---

### Task 6: Dynamic FA Gate Threshold Based on Season Progress

**Files:**
- Modify: `src/trade_intelligence.py` (lines 47, 298-304)
- Create: `tests/test_trade_intelligence_dynamic_gate.py`

**Context:** The FA gate threshold (70%) should adjust based on how much 2026 data exists. Early season (< 50 PA average), projections are uncertain — a FA with 70% of projected value might actually be better. Mid-season (200+ PA), the projections are confident and the 70% gate is appropriate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trade_intelligence_dynamic_gate.py`:

```python
"""Tests for dynamic FA gate threshold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import compute_dynamic_fa_threshold


def test_early_season_threshold_is_higher():
    """Early season: FA must be >= 85% to flag (less aggressive gating)."""
    threshold = compute_dynamic_fa_threshold(avg_pa=20)
    assert threshold >= 0.80


def test_mid_season_threshold_is_standard():
    """Mid season: FA at 70% triggers the gate (standard)."""
    threshold = compute_dynamic_fa_threshold(avg_pa=250)
    assert 0.65 <= threshold <= 0.75


def test_late_season_threshold_is_lower():
    """Late season: FA at 60% triggers (more aggressive gating)."""
    threshold = compute_dynamic_fa_threshold(avg_pa=500)
    assert threshold <= 0.65


def test_zero_pa_uses_max_threshold():
    """No data: use highest threshold (least aggressive gating)."""
    threshold = compute_dynamic_fa_threshold(avg_pa=0)
    assert threshold >= 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trade_intelligence_dynamic_gate.py -v`
Expected: FAIL — `compute_dynamic_fa_threshold` doesn't exist yet

- [ ] **Step 3: Implement dynamic threshold**

In `src/trade_intelligence.py`, add after the `FA_GATE_THRESHOLD` constant (line 47):

```python
def compute_dynamic_fa_threshold(avg_pa: float) -> float:
    """Compute FA gate threshold that adapts to season progress.

    Early season (< 50 PA): threshold = 0.85 (less aggressive — projections uncertain)
    Mid season (200 PA): threshold = 0.70 (standard)
    Late season (500+ PA): threshold = 0.60 (more aggressive — data is reliable)

    Uses a sigmoid-like decay from 0.85 to 0.60 centered at 200 PA.
    """
    import math
    # Sigmoid decay: 0.85 at PA=0, ~0.70 at PA=200, ~0.60 at PA=500
    decay = 0.25 / (1 + math.exp(-0.015 * (avg_pa - 200)))
    return max(0.55, 0.85 - decay)
```

- [ ] **Step 4: Wire dynamic threshold into compute_fa_comparisons**

In `compute_fa_comparisons()`, replace the static `FA_GATE_THRESHOLD` usage at line ~300:

```python
        # Use dynamic threshold based on average PA across the pool
        avg_pool_pa = float(player_pool["pa"].mean()) if "pa" in player_pool.columns else 0
        dynamic_threshold = compute_dynamic_fa_threshold(avg_pool_pa)
        fa_pct = best_fa_value / target_sgp if target_sgp > 0 else 0.0
        results[pid] = {
            "has_alternative": fa_pct >= dynamic_threshold,
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_trade_intelligence_dynamic_gate.py tests/test_trade_intelligence.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/trade_intelligence.py tests/test_trade_intelligence_dynamic_gate.py
git commit -m "feat: dynamic FA gate threshold adapts to season progress

Early season (< 50 PA): 85% threshold (less aggressive — projections uncertain)
Mid season (200 PA): 70% threshold (standard)
Late season (500+ PA): 60% threshold (more aggressive — data reliable)

Uses sigmoid decay centered at 200 PA."
```

---

### Task 7: Wire Kalman Filter + Regime Detection into Trade Valuations

**Files:**
- Create: `src/trade_signals.py` (~100 lines)
- Create: `tests/test_trade_signals.py` (~80 lines)
- Modify: `src/trade_intelligence.py` (add signal adjustment to Trade Readiness)

**Context:** `src/engine/signals/kalman.py` separates true talent from noise, and `src/engine/signals/regime.py` detects hot/cold streaks. These should produce a "trend adjustment" that modifies the Trade Readiness score — a player on a hot streak gets a small bonus, a player in decline gets a penalty.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trade_signals.py`:

```python
"""Tests for trade signal integration — Kalman + regime adjustments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.trade_signals import compute_trend_adjustment


def test_hot_streak_positive_adjustment():
    """Player trending up should get positive adjustment."""
    # Observations trending upward: 0.250, 0.270, 0.290, 0.310
    observations = np.array([0.250, 0.270, 0.290, 0.310])
    adj = compute_trend_adjustment(observations, prior_mean=0.260)
    assert adj > 0


def test_cold_streak_negative_adjustment():
    """Player trending down should get negative adjustment."""
    observations = np.array([0.310, 0.290, 0.270, 0.250])
    adj = compute_trend_adjustment(observations, prior_mean=0.280)
    assert adj < 0


def test_stable_near_zero():
    """Stable performance gives near-zero adjustment."""
    observations = np.array([0.270, 0.268, 0.272, 0.269])
    adj = compute_trend_adjustment(observations, prior_mean=0.270)
    assert abs(adj) < 5.0  # Within 5 points on 0-100 scale


def test_too_few_observations_returns_zero():
    """Need at least 3 data points for trend detection."""
    observations = np.array([0.300])
    adj = compute_trend_adjustment(observations, prior_mean=0.280)
    assert adj == 0.0


def test_adjustment_bounded():
    """Adjustment should be bounded to [-15, +15]."""
    # Extreme trend
    observations = np.array([0.200, 0.250, 0.300, 0.350, 0.400])
    adj = compute_trend_adjustment(observations, prior_mean=0.250)
    assert -15 <= adj <= 15
```

- [ ] **Step 2: Implement trade_signals.py**

Create `src/trade_signals.py`:

```python
"""Trade Signals — Kalman + regime trend adjustment for trade valuations.

Produces a -15 to +15 trend adjustment score that modifies the Trade
Readiness composite. Uses the Kalman filter for true-talent estimation
and simple slope detection for trend direction.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

MAX_ADJUSTMENT = 15.0  # Cap at +/- 15 points on 0-100 scale
MIN_OBSERVATIONS = 3   # Need at least 3 data points


def compute_trend_adjustment(
    observations: np.ndarray,
    prior_mean: float,
    process_variance: float = 0.001,
) -> float:
    """Compute a trend adjustment score from recent observations.

    Uses a lightweight Kalman filter to estimate true talent, then
    compares the slope of the filtered signal to detect trends.

    Args:
        observations: Array of recent performance values (e.g., rolling AVG).
        prior_mean: Expected baseline (e.g., projected AVG).
        process_variance: How much true talent can shift per period.

    Returns:
        Float in [-15, +15]. Positive = trending up, negative = trending down.
        Returns 0.0 if insufficient data.
    """
    if len(observations) < MIN_OBSERVATIONS:
        return 0.0

    try:
        from src.engine.signals.kalman import kalman_true_talent

        # Observation variance: binomial approximation for rates
        obs_var = np.full(len(observations), 0.01)

        filtered_means, _ = kalman_true_talent(
            observations=observations,
            obs_variance=obs_var,
            process_variance=process_variance,
            prior_mean=prior_mean,
            prior_variance=process_variance * 10,
        )

        # Compute slope of filtered signal (linear regression on last N points)
        n = len(filtered_means)
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, filtered_means, 1)[0]

        # Normalize slope to adjustment scale
        # A slope of 0.01 per period (e.g., AVG going up 10 points/week) = +10
        adjustment = slope * 1000  # Scale factor for batting-average-like metrics
        return float(np.clip(adjustment, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))

    except ImportError:
        logger.debug("Kalman filter not available, returning zero adjustment")
        return 0.0
    except Exception:
        logger.debug("Trend adjustment failed", exc_info=True)
        return 0.0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_trade_signals.py -v`
Expected: All pass

- [ ] **Step 4: Wire trend adjustment into Trade Readiness (optional enhancement)**

In `src/trade_intelligence.py`, in the `compute_trade_readiness_batch()` function, add trend adjustment as a bonus/penalty on the final score. This is optional — the trend signal can be displayed as a column without affecting the composite initially.

Add to the batch computation loop:

```python
        # Trend adjustment (optional — from Kalman filter)
        try:
            from src.trade_signals import compute_trend_adjustment
            # Use recent season stats as observations
            # For now, this returns 0 until we have per-player rolling stats
            trend_adj = 0.0
        except ImportError:
            trend_adj = 0.0
```

- [ ] **Step 5: Commit**

```bash
git add src/trade_signals.py tests/test_trade_signals.py src/trade_intelligence.py
git commit -m "feat: Kalman trend adjustment for trade signal intelligence

compute_trend_adjustment() uses Kalman filter to detect hot/cold
streaks. Returns -15 to +15 adjustment for Trade Readiness composite.
Requires 3+ observations for trend detection. Graceful fallback to 0
when Kalman module unavailable."
```

---

### Task 8: Opponent Roster Tracking Over Time

**Files:**
- Modify: `src/database.py` (add `roster_snapshots` table + functions)
- Modify: `src/yahoo_data_service.py` (snapshot on roster fetch)
- Create: `tests/test_roster_tracking.py`

**Context:** Track which players each opponent adds/drops each week. This gives intelligence on what positions opponents are targeting — useful for anticipating counter-offers and identifying buy-low windows.

- [ ] **Step 1: Write failing test**

Create `tests/test_roster_tracking.py`:

```python
"""Tests for opponent roster change tracking."""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import (
    init_db,
    snapshot_league_rosters,
    get_roster_changes,
)


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def test_snapshot_creates_entries():
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 1, 'OF', 0)"
    )
    conn.commit()
    conn.close()

    count = snapshot_league_rosters()
    assert count >= 1


def test_detects_added_player():
    # Snapshot 1: Team A has player 1
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 1, 'OF', 0)"
    )
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    # Add player 2
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 2, 'SP', 0)"
    )
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    changes = get_roster_changes("Team A", days=7)
    adds = [c for c in changes if c["change_type"] == "add"]
    assert len(adds) >= 1
    assert any(c["player_id"] == 2 for c in adds)


def test_detects_dropped_player():
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 1, 'OF', 0)"
    )
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 2, 'SP', 0)"
    )
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    # Remove player 2
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute("DELETE FROM league_rosters WHERE player_id = 2")
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    changes = get_roster_changes("Team A", days=7)
    drops = [c for c in changes if c["change_type"] == "drop"]
    assert len(drops) >= 1
    assert any(c["player_id"] == 2 for c in drops)
```

- [ ] **Step 2: Add roster_snapshots table and functions to database.py**

In `src/database.py`, add the table in `init_db()` schema (after the league_schedule table):

```sql
CREATE TABLE IF NOT EXISTS roster_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    team_name TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    roster_slot TEXT,
    UNIQUE(snapshot_date, team_name, player_id)
);
```

Add `"roster_snapshots"` to `_VALID_TABLE_NAMES` frozenset.

Add functions after `load_league_schedule()`:

```python
def snapshot_league_rosters() -> int:
    """Take a snapshot of current league_rosters for change tracking.

    Stores today's date + all roster entries. Compares with previous
    snapshots to detect adds/drops per team.

    Returns:
        Number of snapshot entries written.
    """
    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        # Load current rosters
        cursor = conn.cursor()
        cursor.execute("SELECT team_name, player_id, roster_slot FROM league_rosters")
        rows = cursor.fetchall()

        count = 0
        for team_name, player_id, slot in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO roster_snapshots "
                    "(snapshot_date, team_name, player_id, roster_slot) "
                    "VALUES (?, ?, ?, ?)",
                    (today, team_name, int(player_id), slot),
                )
                count += 1
            except Exception:
                pass
        conn.commit()
        return count
    finally:
        conn.close()


def get_roster_changes(team_name: str, days: int = 7) -> list[dict]:
    """Detect roster adds/drops for a team over the last N days.

    Compares the most recent snapshot with the oldest snapshot within
    the date range.

    Returns:
        List of dicts with keys: change_type ('add'/'drop'), player_id,
        team_name, detected_date.
    """
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get the earliest and latest snapshot dates in range
        cursor.execute(
            "SELECT MIN(snapshot_date), MAX(snapshot_date) FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date >= ?",
            (team_name, cutoff),
        )
        row = cursor.fetchone()
        if not row or row[0] is None or row[1] is None or row[0] == row[1]:
            return []

        earliest, latest = row

        # Players in latest but not earliest = adds
        cursor.execute(
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ? "
            "EXCEPT "
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ?",
            (team_name, latest, team_name, earliest),
        )
        adds = [{"change_type": "add", "player_id": int(r[0]),
                 "team_name": team_name, "detected_date": latest}
                for r in cursor.fetchall()]

        # Players in earliest but not latest = drops
        cursor.execute(
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ? "
            "EXCEPT "
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ?",
            (team_name, earliest, team_name, latest),
        )
        drops = [{"change_type": "drop", "player_id": int(r[0]),
                  "team_name": team_name, "detected_date": latest}
                 for r in cursor.fetchall()]

        return adds + drops
    finally:
        conn.close()
```

- [ ] **Step 3: Wire snapshot into YahooDataService roster fetch**

In `src/yahoo_data_service.py`, in `_fetch_and_sync_rosters()`, add after `update_refresh_log("yahoo_data", "success")`:

```python
        # Take a roster snapshot for change tracking
        try:
            from src.database import snapshot_league_rosters
            snapshot_league_rosters()
        except Exception:
            pass  # Non-fatal — tracking is optional
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_roster_tracking.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/database.py src/yahoo_data_service.py tests/test_roster_tracking.py
git commit -m "feat: opponent roster tracking with daily snapshots

New roster_snapshots table stores daily roster state per team.
snapshot_league_rosters() runs on every Yahoo roster sync.
get_roster_changes(team, days) detects adds/drops by comparing
earliest vs latest snapshot within the date range.

Enables future opponent intelligence: what positions are they
targeting? Who did they just drop (buy-low window)?"
```

---

## Verification Checklist

After all 8 tasks are complete:

- [ ] `python -m pytest -x -q` — full suite passes (2139+ tests)
- [ ] `python -m ruff check .` — no lint errors
- [ ] `python -m ruff format --check .` — all files formatted
- [ ] `git push origin master` — CI passes all 5 jobs
- [ ] Browser test: Trade Finder shows sensible recommendations
- [ ] Browser test: My Team news tab loads without errors
- [ ] Browser test: Standings table shows R/HR/RBI (not WINS/LOSSES)
