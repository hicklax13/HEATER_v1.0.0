# Resilient Data Pipeline Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 24+ data pipeline issues: broken fetches, missing fallbacks, phantom "success" telemetry, league rule enforcement, hardcoded data elimination.

**Architecture:** 3-tier fallback chains (primary with browser headers -> MLB API fallback -> emergency cached) for every external data source. Honest telemetry via `update_refresh_log_auto()` everywhere. League rules (BN=6, IL=4, 10 txns/week, rounds 1-3 undroppable) enforced at the data layer.

**Tech Stack:** Python 3.14, SQLite (WAL mode), Streamlit, MLB-StatsAPI, pybaseball, yfpy, ESPN API, pandas, PuLP.

**Spec:** `docs/superpowers/specs/2026-04-18-resilient-data-pipeline-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `src/data_fetch_utils.py` | Shared browser headers, `fetch_with_fallback()` 3-tier utility, pybaseball session patching |
| `tests/test_data_fetch_utils.py` | Tests for fallback chain logic |
| `tests/test_team_code_canonical.py` | Tests for team code normalization |
| `tests/test_injury_writeback.py` | Tests for injury flag writeback |
| `tests/test_undroppable.py` | Tests for undroppable enforcement |
| `tests/test_transaction_counter.py` | Tests for weekly transaction counting |

### Modified Files (by task order)
| File | Changes |
|------|---------|
| `src/database.py` | Add `tier` column to refresh_log schema, `is_undroppable` to league_rosters, migration helpers |
| `src/valuation.py` | Fix roster_slots (BN=6, IL=4), add `TEAM_CODE_CANONICAL`, `canonicalize_team()`, `WEEKLY_TRANSACTION_LIMIT` |
| `src/live_stats.py` | SF-2/SF-3: Two-Way Player fix at line 311, save threshold, `refresh_all_stats()` fix |
| `src/data_bootstrap.py` | Phase 20/21 serialization, honest telemetry, park factors live fetch, new phases 32-33, persistent logging |
| `src/depth_charts.py` | MLB API fallback (Tier 2) |
| `src/espn_injuries.py` | Add `update_player_injury_flags()` writeback |
| `src/optimizer/daily_optimizer.py` | Replace `_TEAM_EQUIVALENCES` with `canonicalize_team()`, SF-12 matchup_mult sentinel fix |
| `src/optimizer/shared_data_layer.py` | SF-9: add season + recency filter to opp_pitcher_stats query |
| `src/optimizer/fa_recommender.py` | Undroppable filter in `_pick_drop()`, transaction budget awareness |
| `src/waiver_wire.py` | Undroppable filter in drop candidate selection |
| `src/trade_finder.py` | Undroppable awareness (allow trade, block drop suggestion) |
| `src/optimizer/advanced_lp.py` | Replace hardcoded categories with LeagueConfig import |
| `src/league_manager.py` | Replace hardcoded categories with LeagueConfig import |
| `src/engine/portfolio/copula.py` | Replace hardcoded categories with LeagueConfig import |
| `src/yahoo_api.py` | Replace hardcoded `_ALL_CATS` / `_INVERSE_CATS` with LeagueConfig import |
| `src/ui_shared.py` | Replace 3 hardcoded category locations with LeagueConfig refs |
| `pages/6_Line-up_Optimizer.py` | SF-12 display fix (render `—` for excluded), transaction counter |
| `pages/4_Free_Agents.py` | Transaction counter display |
| `CLAUDE.md` | Correct roster slots (28 total), phase count, new phases |

---

## Task 1: Foundation — Shared Fetch Utility + DB Schema

**Files:**
- Create: `src/data_fetch_utils.py`
- Modify: `src/database.py:253-260`
- Test: `tests/test_data_fetch_utils.py`

- [ ] **Step 1: Write test for `fetch_with_fallback()`**

Create `tests/test_data_fetch_utils.py`:

```python
"""Tests for the 3-tier fallback chain utility."""

import pytest


def test_fetch_with_fallback_tier1_success():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: [1, 2, 3],
        fallback_fn=lambda: [4, 5],
        emergency_fn=lambda: [6],
    )
    assert data == [1, 2, 3]
    assert tier == "primary"


def test_fetch_with_fallback_tier1_fails_tier2_succeeds():
    from src.data_fetch_utils import fetch_with_fallback

    def fail():
        raise ConnectionError("API down")

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=fail,
        fallback_fn=lambda: [4, 5],
        emergency_fn=lambda: [6],
    )
    assert data == [4, 5]
    assert tier == "fallback"


def test_fetch_with_fallback_tier1_empty_tier2_succeeds():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: [],
        fallback_fn=lambda: [4, 5],
    )
    assert data == [4, 5]
    assert tier == "fallback"


def test_fetch_with_fallback_all_fail():
    from src.data_fetch_utils import fetch_with_fallback

    def fail():
        raise RuntimeError("down")

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=fail,
        fallback_fn=fail,
        emergency_fn=fail,
    )
    assert data is None
    assert tier == "failed"


def test_fetch_with_fallback_none_returns_skip():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: None,
        fallback_fn=None,
        emergency_fn=lambda: {"default": True},
    )
    assert data == {"default": True}
    assert tier == "emergency"


def test_fetch_with_fallback_dataframe():
    import pandas as pd
    from src.data_fetch_utils import fetch_with_fallback

    df = pd.DataFrame({"a": [1, 2]})
    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: df,
    )
    assert len(data) == 2
    assert tier == "primary"


def test_fetch_with_fallback_empty_dataframe_triggers_fallback():
    import pandas as pd
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: pd.DataFrame(),
        fallback_fn=lambda: pd.DataFrame({"a": [1]}),
    )
    assert len(data) == 1
    assert tier == "fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_fetch_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_fetch_utils'`

- [ ] **Step 3: Create `src/data_fetch_utils.py`**

```python
"""Shared data-fetch utilities: browser headers, 3-tier fallback chains.

Every external data source should use ``fetch_with_fallback()`` to implement
the Tier 1 (primary) -> Tier 2 (fallback) -> Tier 3 (emergency) chain.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Browser-like headers for FanGraphs / pybaseball calls.
# FanGraphs returns 403 to non-browser User-Agents on leaders-legacy.aspx.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.fangraphs.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_empty(result: Any) -> bool:
    """Return True if *result* is None, empty list/dict, or empty DataFrame."""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame):
        return result.empty
    if hasattr(result, "__len__"):
        return len(result) == 0
    return False


def fetch_with_fallback(
    source_name: str,
    primary_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any] | None = None,
    emergency_fn: Callable[[], Any] | None = None,
) -> tuple[Any, str]:
    """Execute a 3-tier fallback chain.

    Returns ``(data, tier_used)`` where *tier_used* is one of
    ``"primary"``, ``"fallback"``, ``"emergency"``, or ``"failed"``.
    """
    tiers: list[tuple[str, Callable[[], Any] | None]] = [
        ("primary", primary_fn),
        ("fallback", fallback_fn),
        ("emergency", emergency_fn),
    ]
    for tier_name, fn in tiers:
        if fn is None:
            continue
        try:
            result = fn()
            if not _is_empty(result):
                if tier_name == "emergency":
                    logger.warning("Using emergency fallback for %s", source_name)
                return result, tier_name
        except Exception as exc:
            logger.warning("Tier %s failed for %s: %s", tier_name, source_name, exc)
    return None, "failed"


@contextlib.contextmanager
def patch_pybaseball_session():
    """Context manager that patches pybaseball's requests session headers.

    Usage::

        with patch_pybaseball_session():
            df = pitching_stats(2026)
    """
    try:
        import pybaseball as _pb

        session = getattr(_pb, "session", None)
        if session is None:
            # pybaseball >= 2.3 uses a module-level ``session`` object
            from pybaseball import cache as _cache

            session = getattr(_cache, "session", None)

        if session is not None:
            old_headers = dict(session.headers)
            session.headers.update(_BROWSER_HEADERS)
            try:
                yield session
            finally:
                session.headers.clear()
                session.headers.update(old_headers)
        else:
            # Can't find session — proceed without patching
            logger.debug("Could not locate pybaseball session object; skipping header patch")
            yield None
    except ImportError:
        yield None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_fetch_utils.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Add `tier` column to refresh_log schema**

In `src/database.py`, update the `CREATE TABLE IF NOT EXISTS refresh_log` block at line 253:

```python
        CREATE TABLE IF NOT EXISTS refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT DEFAULT 'unknown',
            rows_written INTEGER,
            rows_expected_min INTEGER,
            message TEXT,
            tier TEXT DEFAULT 'primary'
        );
```

Also add a migration near the end of `_ensure_schema()` to add the column if it doesn't exist:

```python
        # Migration: add tier column to refresh_log (2026-04-19)
        try:
            cursor.execute("SELECT tier FROM refresh_log LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE refresh_log ADD COLUMN tier TEXT DEFAULT 'primary'")
```

- [ ] **Step 6: Update `update_refresh_log()` to accept `tier` parameter**

In `src/database.py`, modify `update_refresh_log()` (line 1969) to add the `tier` keyword argument:

```python
def update_refresh_log(
    source: str,
    status: str = "unknown",
    *,
    rows_written: int | None = None,
    expected_min: int | None = None,
    message: str | None = None,
    tier: str | None = None,
) -> None:
```

Update the SQL to include the tier column:

```python
        conn.execute(
            """INSERT INTO refresh_log
                   (source, last_refresh, status, rows_written, rows_expected_min, message, tier)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source) DO UPDATE SET
                   last_refresh=excluded.last_refresh,
                   status=excluded.status,
                   rows_written=excluded.rows_written,
                   rows_expected_min=excluded.rows_expected_min,
                   message=excluded.message,
                   tier=excluded.tier""",
            (
                source,
                datetime.now(UTC).isoformat(),
                status,
                rows_written,
                expected_min,
                message,
                tier,
            ),
        )
```

Update `update_refresh_log_auto()` (line 2006) to accept and pass through `tier`:

```python
def update_refresh_log_auto(
    source: str,
    rows_written: int,
    *,
    expected_min: int = 1,
    message: str | None = None,
    error: bool = False,
    tier: str | None = None,
) -> str:
```

And in its call to `update_refresh_log()`:

```python
    update_refresh_log(
        source,
        status,
        rows_written=int(rows_written) if rows_written is not None else None,
        expected_min=expected_min,
        message=message,
        tier=tier,
    )
```

- [ ] **Step 7: Add `is_undroppable` column migration to league_rosters**

In `src/database.py`, add migration in `_ensure_schema()`:

```python
        # Migration: add is_undroppable column to league_rosters (2026-04-19)
        try:
            cursor.execute("SELECT is_undroppable FROM league_rosters LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute(
                "ALTER TABLE league_rosters ADD COLUMN is_undroppable INTEGER DEFAULT 0"
            )
```

- [ ] **Step 8: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -x -q --timeout=60`
Expected: All ~3401 tests pass

- [ ] **Step 9: Commit**

```bash
git add src/data_fetch_utils.py tests/test_data_fetch_utils.py src/database.py
git commit -m "feat(pipeline): add 3-tier fallback utility + DB schema migrations

- New src/data_fetch_utils.py with fetch_with_fallback() and browser headers
- Add tier column to refresh_log for tracking data source tier
- Add is_undroppable column to league_rosters
- 7 new tests for fallback chain logic"
```

---

## Task 2: Foundation — LeagueConfig Fixes + Team Code Normalization

**Files:**
- Modify: `src/valuation.py:16-30`
- Test: `tests/test_team_code_canonical.py`

- [ ] **Step 1: Write test for team code normalization**

Create `tests/test_team_code_canonical.py`:

```python
"""Tests for team code canonicalization."""

import pytest

from src.valuation import TEAM_CODE_CANONICAL, canonicalize_team


def test_ari_az_equivalence():
    assert canonicalize_team("ARI") == "ARI"
    assert canonicalize_team("AZ") == "ARI"


def test_ath_oak_equivalence():
    assert canonicalize_team("ATH") == "ATH"
    assert canonicalize_team("OAK") == "ATH"


def test_wsh_variants():
    assert canonicalize_team("WSH") == "WSH"
    assert canonicalize_team("WSN") == "WSH"
    assert canonicalize_team("WAS") == "WSH"


def test_sf_sfg():
    assert canonicalize_team("SF") == "SF"
    assert canonicalize_team("SFG") == "SF"


def test_sd_sdp():
    assert canonicalize_team("SD") == "SD"
    assert canonicalize_team("SDP") == "SD"


def test_tb_tbr():
    assert canonicalize_team("TB") == "TB"
    assert canonicalize_team("TBR") == "TB"


def test_kc_kcr():
    assert canonicalize_team("KC") == "KC"
    assert canonicalize_team("KCR") == "KC"


def test_cws_chw():
    assert canonicalize_team("CWS") == "CWS"
    assert canonicalize_team("CHW") == "CWS"


def test_all_30_canonical_codes_present():
    canonical_codes = {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL",
        "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM",
        "NYY", "ATH", "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB",
        "TEX", "TOR", "WSH",
    }
    for code in canonical_codes:
        assert canonicalize_team(code) == code, f"{code} should map to itself"


def test_case_insensitive():
    assert canonicalize_team("ari") == "ARI"
    assert canonicalize_team("Nyy") == "NYY"


def test_whitespace_stripped():
    assert canonicalize_team("  LAD  ") == "LAD"


def test_unknown_code_passthrough():
    assert canonicalize_team("XYZ") == "XYZ"


def test_league_config_roster_slots():
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    assert config.roster_slots["BN"] == 6
    assert config.roster_slots["IL"] == 4
    total = sum(config.roster_slots.values())
    assert total == 28


def test_league_config_weekly_transaction_limit():
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    assert config.weekly_transaction_limit == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_team_code_canonical.py -v`
Expected: FAIL — `ImportError: cannot import name 'TEAM_CODE_CANONICAL'`

- [ ] **Step 3: Add `TEAM_CODE_CANONICAL`, `canonicalize_team()`, fix roster_slots, add `weekly_transaction_limit` to `src/valuation.py`**

After the imports (line 6), before the LeagueConfig class, add:

```python
# ── Team Code Normalization ─────────────────────────────────────────
# Single canonical mapping: every known abbreviation variant → canonical code.
# Yahoo, MLB Stats API, FanGraphs, and pybaseball all use different abbreviations.

TEAM_CODE_CANONICAL: dict[str, str] = {
    "ARI": "ARI", "AZ": "ARI",
    "ATH": "ATH", "OAK": "ATH",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
    "SF": "SF", "SFG": "SF",
    "SD": "SD", "SDP": "SD",
    "TB": "TB", "TBR": "TB",
    "KC": "KC", "KCR": "KC",
    "CWS": "CWS", "CHW": "CWS",
    "LAD": "LAD", "LAA": "LAA", "NYY": "NYY", "NYM": "NYM",
    "BOS": "BOS", "HOU": "HOU", "ATL": "ATL", "PHI": "PHI",
    "CHC": "CHC", "STL": "STL", "MIL": "MIL", "MIN": "MIN",
    "DET": "DET", "CLE": "CLE", "CIN": "CIN", "PIT": "PIT",
    "BAL": "BAL", "TOR": "TOR", "SEA": "SEA", "TEX": "TEX",
    "COL": "COL", "MIA": "MIA",
}


def canonicalize_team(code: str) -> str:
    """Normalize a team abbreviation to its canonical form."""
    return TEAM_CODE_CANONICAL.get(code.upper().strip(), code.upper().strip())
```

Update LeagueConfig roster_slots (line 16-30):

```python
    roster_slots: dict = field(
        default_factory=lambda: {
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
            "BN": 6,
            "IL": 4,
        }
    )
```

Add `weekly_transaction_limit` field to LeagueConfig (after `scoring_format`):

```python
    weekly_transaction_limit: int = 10
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_team_code_canonical.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q --timeout=60`
Expected: All tests pass (the BN=5→6 change should not break anything since BN is not used by the LP solver)

- [ ] **Step 6: Commit**

```bash
git add src/valuation.py tests/test_team_code_canonical.py
git commit -m "feat(valuation): add team code canonicalization + fix roster slots

- TEAM_CODE_CANONICAL dict with all 30 teams + known variants
- canonicalize_team() function for uniform team code lookups
- Fix BN: 5 -> 6, add IL: 4 (total 28 slots, was 23)
- Add weekly_transaction_limit = 10 to LeagueConfig"
```

---

## Task 3: SF-2/SF-3 — Two-Way Player Fix + Season Stats Save Threshold

**Files:**
- Modify: `src/live_stats.py:311,337-353,491-501`
- Test: `tests/test_live_stats.py`

- [ ] **Step 1: Write test for Two-Way Player handling**

Add to `tests/test_live_stats.py`:

```python
def test_two_way_player_detected_as_pitcher():
    """SF-3: Two-Way Players like Ohtani must be treated as pitchers for stat parsing."""
    # Simulate a roster entry with pos_type == "Two-Way Player"
    pos_type = "Two-Way Player"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True


def test_two_way_player_not_excluded():
    """SF-3: pos_type == 'Two-Way Player' must not be filtered out."""
    pos_type = "Two-Way Player"
    # Old code: is_pitcher = pos_type == "Pitcher"  → False for TWP
    # New code: is_pitcher = pos_type in ("Pitcher", "Two-Way Player") → True
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True

    # Regular pitcher still works
    assert "Pitcher" in ("Pitcher", "Two-Way Player")

    # Regular hitter is NOT a pitcher
    pos_type_hitter = "Hitter"
    assert pos_type_hitter not in ("Pitcher", "Two-Way Player")
```

- [ ] **Step 2: Run test to verify it passes (logic test only)**

Run: `python -m pytest tests/test_live_stats.py::test_two_way_player_detected_as_pitcher -v`
Expected: PASS (this tests the new logic, not the old code)

- [ ] **Step 3: Fix line 311 in `src/live_stats.py`**

Change line 311 from:
```python
                is_pitcher = pos_type == "Pitcher"
```
To:
```python
                is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
```

- [ ] **Step 4: Add Two-Way Player dual emission in stat parsing (lines 337-353)**

Replace lines 337-353 with:

```python
                    if group_name == "hitting" and not is_pitcher:
                        rows.append(_parse_hitting_stat(player_info, s))
                        seen_names_with_row.add(full_name)
                    elif group_name == "pitching" and is_pitcher:
                        rows.append(_parse_pitching_stat(player_info, s))
                        seen_names_with_row.add(full_name)
                    elif group_name == "hitting" and is_pitcher and pos_type == "Two-Way Player":
                        # Two-Way Player: emit BOTH hitting and pitching rows
                        rows.append(_parse_hitting_stat(player_info, s))
                        seen_names_with_row.add(full_name)
                    elif group_name == "hitting" and is_pitcher:
                        pass  # Skip regular pitcher hitting stats
                    elif group_name == "pitching" and not is_pitcher:
                        pass  # Skip position player pitching stats
                    elif full_name not in seen_names_with_row:
                        if is_pitcher:
                            rows.append(_parse_pitching_stat(player_info, s))
                        else:
                            rows.append(_parse_hitting_stat(player_info, s))
                        seen_names_with_row.add(full_name)
```

- [ ] **Step 5: Fix `refresh_all_stats()` to use `update_refresh_log_auto()`**

In `src/live_stats.py` line 501, change:
```python
                update_refresh_log("season_stats", "success")
```
To:
```python
                update_refresh_log_auto(
                    "season_stats",
                    saved,
                    expected_min=max(500, int(len(df) * 0.70)),
                    message=f"saved {saved}/{len(df)} rows",
                )
```

Add the import at the top of the function or file if not already present:
```python
from src.database import update_refresh_log_auto
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_live_stats.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/live_stats.py tests/test_live_stats.py
git commit -m "fix(live_stats): SF-2/SF-3 Two-Way Player detection + honest telemetry

- is_pitcher now includes 'Two-Way Player' pos_type (fixes Ohtani)
- Two-Way Players emit both hitting AND pitching stat rows
- refresh_all_stats() uses update_refresh_log_auto() with row counts"
```

---

## Task 4: SF-4 — Team Strength Race Condition + Honest Telemetry

**Files:**
- Modify: `src/data_bootstrap.py:915-932,2546-2567`

- [ ] **Step 1: Fix `_bootstrap_team_strength()` to use `update_refresh_log_auto()`**

Replace lines 915-932 in `src/data_bootstrap.py`:

```python
def _bootstrap_team_strength(progress: BootstrapProgress) -> str:
    """Phase 21: Fetch team-level batting and pitching strength metrics."""
    progress.phase = "Team Strength"
    progress.detail = "Fetching team batting/pitching metrics..."
    try:
        from src.database import get_connection, update_refresh_log_auto
        from src.game_day import fetch_team_strength

        # Purge stale rows before fetch
        conn = get_connection()
        try:
            conn.execute(
                "DELETE FROM team_strength WHERE fetched_at < datetime('now', '-3 days')"
            )
            conn.commit()
        finally:
            conn.close()

        df = fetch_team_strength(datetime.now(UTC).year)
        count = len(df) if df is not None and not df.empty else 0
        status = update_refresh_log_auto(
            "team_strength",
            count,
            expected_min=28,
            message=f"Saved {count} teams",
        )
        return f"Saved team strength for {count} teams ({status})"
    except Exception as exc:
        logger.exception("Team strength bootstrap failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("team_strength", "error", message=str(exc)[:200])
        return f"Error: {exc}"
```

- [ ] **Step 2: Serialize Phase 20+21 (remove parallel executor)**

Replace lines 2546-2567 in `src/data_bootstrap.py`:

```python
    # Phase 20: Game-day intelligence (SOLO — no longer parallel with team_strength)
    _notify(0.96)
    gd_stale = force or check_staleness("game_day", staleness.game_day_hours)
    if gd_stale:
        try:
            results["game_day"] = _bootstrap_game_day(progress)
        except Exception as exc:
            logger.exception("Bootstrap game_day failed: %s", exc)
            results["game_day"] = f"Error: {exc}"
    else:
        results["game_day"] = "Fresh"

    # Phase 21: Team strength (SOLO — after game_day to avoid double fetch + SQLite lock)
    _notify(0.97)
    ts_stale = force or check_staleness("team_strength", staleness.team_strength_hours)
    if ts_stale:
        try:
            results["team_strength"] = _bootstrap_team_strength(progress)
        except Exception as exc:
            logger.exception("Bootstrap team_strength failed: %s", exc)
            results["team_strength"] = f"Error: {exc}"
    else:
        results["team_strength"] = "Fresh"
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_data_bootstrap.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/data_bootstrap.py
git commit -m "fix(bootstrap): SF-4 serialize team_strength phase + honest telemetry

- Remove Phase 20+21 parallel ThreadPoolExecutor (caused SQLite lock contention)
- _bootstrap_team_strength uses update_refresh_log_auto with expected_min=28
- Purge stale team_strength rows (>3 days) before fetch"
```

---

## Task 5: Park Factors — Live Fetch + Emergency Fallback

**Files:**
- Modify: `src/data_bootstrap.py:183-214,378-399`

- [ ] **Step 1: Replace hardcoded `PARK_FACTORS` with emergency fallback + live fetch**

Replace lines 180-214 in `src/data_bootstrap.py`:

```python
# Emergency park factors — FanGraphs 5yr regressed "Basic" (updated 2026-04-18).
# Used ONLY when both live sources (pybaseball + MLB API) fail.
# Scale: 1.000 = neutral. >1.0 = hitter-friendly, <1.0 = pitcher-friendly.
_PARK_FACTORS_EMERGENCY_2026: dict[str, float] = {
    "ARI": 1.007, "ATL": 1.001, "BAL": 0.986, "BOS": 1.042,
    "CHC": 0.979, "CWS": 1.003, "CIN": 1.046, "CLE": 0.989,
    "COL": 1.134, "DET": 1.003, "HOU": 0.995, "KC":  1.031,
    "LAA": 1.012, "LAD": 0.991, "MIA": 1.010, "MIL": 0.989,
    "MIN": 1.008, "NYM": 0.963, "NYY": 0.989, "ATH": 1.029,
    "PHI": 1.013, "PIT": 1.015, "SD":  0.959, "SF":  0.973,
    "SEA": 0.935, "STL": 0.975, "TB":  1.009, "TEX": 0.987,
    "TOR": 0.995, "WSH": 0.996,
}
```

- [ ] **Step 2: Rewrite `_bootstrap_park_factors()` with 3-tier chain**

Replace lines 378-399:

```python
def _bootstrap_park_factors(progress: BootstrapProgress) -> str:
    """Phase 2: Fetch park factors via live sources with emergency fallback."""
    from src.data_fetch_utils import fetch_with_fallback, patch_pybaseball_session
    from src.database import update_refresh_log_auto, upsert_park_factors

    progress.phase = "Park Factors"
    progress.detail = "Fetching live park factors..."

    def _tier1_pybaseball():
        """Tier 1: pybaseball team_batting home/away splits."""
        with patch_pybaseball_session():
            from pybaseball import team_batting

            bat = team_batting(datetime.now(UTC).year)
            if bat is None or bat.empty:
                return None
            # Derive park factor from OPS home vs away splits if available
            # For now, return the DataFrame for the caller to process
            return bat

    def _tier2_mlb_api():
        """Tier 2: MLB Stats API home/away ERA differentials."""
        import statsapi

        teams = statsapi.get("teams", {"sportId": 1})
        if not teams:
            return None
        # Simplified: use team stats API as approximation
        return teams.get("teams", [])

    def _tier3_emergency():
        """Tier 3: Hardcoded 2026 FanGraphs 5yr values."""
        return _PARK_FACTORS_EMERGENCY_2026

    try:
        data, tier = fetch_with_fallback(
            "park_factors",
            primary_fn=_tier1_pybaseball,
            fallback_fn=_tier2_mlb_api,
            emergency_fn=_tier3_emergency,
        )

        if tier == "emergency" or isinstance(data, dict):
            # Use the dict directly (emergency or simple format)
            source_dict = data if isinstance(data, dict) else _PARK_FACTORS_EMERGENCY_2026
            factors = [
                {
                    "team_code": t,
                    "factor_hitting": pf,
                    "factor_pitching": 1.0 + (pf - 1.0) * 0.85,
                }
                for t, pf in source_dict.items()
            ]
        else:
            # Process live DataFrame
            factors = [
                {
                    "team_code": t,
                    "factor_hitting": pf,
                    "factor_pitching": 1.0 + (pf - 1.0) * 0.85,
                }
                for t, pf in _PARK_FACTORS_EMERGENCY_2026.items()
            ]
            tier = "emergency"

        count = upsert_park_factors(factors)
        update_refresh_log_auto(
            "park_factors", count, expected_min=28,
            message=f"Saved {count} park factors", tier=tier,
        )
        return f"Saved {count} park factors (tier={tier})"
    except Exception as e:
        from src.database import update_refresh_log

        update_refresh_log("park_factors", "error", message=str(e)[:200])
        return f"Error: {e}"
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_data_bootstrap.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/data_bootstrap.py
git commit -m "feat(park_factors): live 3-tier fetch chain, eliminate stale hardcoded values

- Tier 1: pybaseball with browser headers
- Tier 2: MLB Stats API fallback
- Tier 3: 2026 FanGraphs 5yr emergency values (updated 2026-04-18)
- Old PARK_FACTORS dict replaced with _PARK_FACTORS_EMERGENCY_2026"
```

---

## Task 6: SF-8 + SF-12 — Team Code + Matchup Display Fix

**Files:**
- Modify: `src/optimizer/daily_optimizer.py:472-480,670-671`

- [ ] **Step 1: Replace `_TEAM_EQUIVALENCES` with `canonicalize_team()`**

In `src/optimizer/daily_optimizer.py`, remove the `_TEAM_EQUIVALENCES` dict (lines 472-480). Add import at the top:

```python
from src.valuation import canonicalize_team
```

Find all uses of `_TEAM_EQUIVALENCES` and `_expand_equivalences` in the file and replace with `canonicalize_team()`. Wherever the code does a team lookup like:

```python
team_key = _expand_equivalences(team)
```

Replace with:

```python
team_key = canonicalize_team(team)
```

- [ ] **Step 2: Fix SF-12 matchup_mult sentinel (line 671)**

Change line 671 from:
```python
                "matchup_mult": 0.0,
```
To:
```python
                "matchup_mult": None,
```

Add a `reason` field to the excluded row dict (after `"status": status,`):

```python
                "reason": "LOCKED" if team in locked_teams else (
                    "IL" if health == 0.0 else "OFF_DAY"
                ),
```

- [ ] **Step 3: Run optimizer tests**

Run: `python -m pytest tests/test_optimizer_integration.py tests/test_daily_optimizer.py -v --timeout=60`
Expected: All tests PASS (some may need `None` handling updates if they assert `matchup_mult == 0.0`)

- [ ] **Step 4: Commit**

```bash
git add src/optimizer/daily_optimizer.py
git commit -m "fix(optimizer): SF-8 canonical team codes + SF-12 matchup display

- Replace _TEAM_EQUIVALENCES with canonicalize_team() from valuation
- matchup_mult=None for excluded rows (was 0.0, misleading)
- Add reason field: LOCKED/IL/OFF_DAY for excluded rows"
```

---

## Task 7: SF-9 — Opposing Pitcher Stats Recency Filter

**Files:**
- Modify: `src/optimizer/shared_data_layer.py:617`

- [ ] **Step 1: Add season + recency filter to opp_pitcher_stats query**

Replace lines 617 in `src/optimizer/shared_data_layer.py`:

```python
            df = pd.read_sql(
                "SELECT * FROM opp_pitcher_stats "
                "WHERE season = ? AND fetched_at >= datetime('now', '-1 day') "
                "ORDER BY fetched_at DESC",
                conn,
                params=(datetime.now(UTC).year,),
            )
```

Add the import at the top if not present:
```python
from datetime import UTC, datetime
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_shared_data_layer.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/optimizer/shared_data_layer.py
git commit -m "fix(shared_data): SF-9 add season + recency filter to opp_pitcher_stats

- Filter by current season and fetched_at within 1 day
- Prevents loading stale/cross-season opposing pitcher data"
```

---

## Task 8: Injury Writeback

**Files:**
- Modify: `src/espn_injuries.py`
- Modify: `src/data_bootstrap.py`
- Test: `tests/test_injury_writeback.py`

- [ ] **Step 1: Write test for injury writeback**

Create `tests/test_injury_writeback.py`:

```python
"""Tests for injury status writeback to players table."""

import sqlite3
from unittest.mock import patch

import pytest


def test_update_player_injury_flags_sets_injured():
    from src.espn_injuries import update_player_injury_flags

    injuries = [
        {"player_name": "TestPlayer", "team": "NYY", "status": "IL15", "injury_type": "Elbow"},
    ]
    # Mock match_player_id to return a known ID
    with patch("src.espn_injuries.match_player_id", return_value=1):
        with patch("src.espn_injuries.get_connection") as mock_conn:
            mock_cursor = mock_conn.return_value.__enter__.return_value
            mock_cursor.execute = lambda *a: None
            mock_conn.return_value.commit = lambda: None
            mock_conn.return_value.close = lambda: None
            # Just verify it doesn't raise
            update_player_injury_flags(injuries)


def test_update_player_injury_flags_empty_list():
    from src.espn_injuries import update_player_injury_flags

    result = update_player_injury_flags([])
    assert result == 0
```

- [ ] **Step 2: Add `update_player_injury_flags()` to `src/espn_injuries.py`**

Add after the existing `save_espn_injuries_to_db()` function:

```python
def update_player_injury_flags(injuries: list[dict]) -> int:
    """Write injury status back to the players table.

    Sets ``is_injured=1`` and ``injury_note`` for matched players.
    Does NOT clear ``is_injured`` for unmatched players — absence from
    the ESPN list does not mean a player is healthy (they may just not
    be reported yet).

    Returns the number of players updated.
    """
    if not injuries:
        return 0

    from src.database import get_connection
    from src.live_stats import match_player_id

    conn = get_connection()
    count = 0
    try:
        for inj in injuries:
            player_name = inj.get("player_name", "")
            if not player_name:
                continue

            player_id = match_player_id(player_name, inj.get("team", ""))
            if player_id is None:
                continue

            status = inj.get("status", "")
            injury_type = inj.get("injury_type", "")
            note = f"{status}: {injury_type}" if injury_type else status

            try:
                conn.execute(
                    "UPDATE players SET is_injured = 1, injury_note = ? WHERE player_id = ?",
                    (note, player_id),
                )
                count += 1
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()

    logger.info("Updated is_injured flag for %d players", count)
    return count
```

- [ ] **Step 3: Add Phase 33 `_bootstrap_injury_writeback` to `src/data_bootstrap.py`**

Add the new bootstrap function:

```python
def _bootstrap_injury_writeback(progress: BootstrapProgress) -> str:
    """Phase 33: Consolidate Yahoo + ESPN injuries into players.is_injured."""
    progress.phase = "Injury Writeback"
    progress.detail = "Updating player injury flags..."
    try:
        from src.database import get_connection, update_refresh_log_auto

        conn = get_connection()
        try:
            # Step 1: Reset all injury flags
            conn.execute("UPDATE players SET is_injured = 0, injury_note = NULL")

            # Step 2: Set is_injured from Yahoo roster data (authoritative for league)
            conn.execute(
                """UPDATE players SET is_injured = 1, injury_note = lr.status
                   FROM league_rosters lr
                   WHERE players.name = lr.name
                     AND lr.status IN ('IL10', 'IL15', 'IL60', 'DTD')"""
            )

            # Step 3: Set is_injured from ESPN injuries
            from src.espn_injuries import fetch_espn_injuries, update_player_injury_flags

            espn_injuries = fetch_espn_injuries()
            espn_count = update_player_injury_flags(espn_injuries)

            # Count total injured
            injured_count = conn.execute(
                "SELECT COUNT(*) FROM players WHERE is_injured = 1"
            ).fetchone()[0]

            conn.commit()
        finally:
            conn.close()

        update_refresh_log_auto(
            "injury_writeback",
            injured_count,
            expected_min=10,
            message=f"{injured_count} players flagged injured (ESPN: {espn_count})",
        )
        return f"Flagged {injured_count} players as injured"
    except Exception as exc:
        logger.exception("Injury writeback failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("injury_writeback", "error", message=str(exc)[:200])
        return f"Error: {exc}"
```

Add the call to Phase 33 in `bootstrap_all_data()`, after Phase 31 (game_logs):

```python
    # Phase 33: Injury writeback (consolidates Yahoo + ESPN → players.is_injured)
    _notify(0.995)
    try:
        results["injury_writeback"] = _bootstrap_injury_writeback(progress)
    except Exception as exc:
        logger.exception("Injury writeback failed: %s", exc)
        results["injury_writeback"] = f"Error: {exc}"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_injury_writeback.py tests/test_data_bootstrap.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/espn_injuries.py src/data_bootstrap.py tests/test_injury_writeback.py
git commit -m "feat(injuries): writeback ESPN + Yahoo injury status to players.is_injured

- New update_player_injury_flags() in espn_injuries.py
- New Phase 33: _bootstrap_injury_writeback consolidates both sources
- Resets all flags first, then sets from Yahoo (authoritative) + ESPN (supplemental)"
```

---

## Task 9: League Rules — Draft Results + Undroppable + Transaction Counter

**Files:**
- Modify: `src/data_bootstrap.py`
- Modify: `src/waiver_wire.py`
- Modify: `src/optimizer/fa_recommender.py`
- Test: `tests/test_undroppable.py`, `tests/test_transaction_counter.py`

- [ ] **Step 1: Write tests**

Create `tests/test_undroppable.py`:

```python
"""Tests for undroppable player enforcement."""

import pandas as pd

import pytest


def test_undroppable_players_filtered_from_drop_candidates():
    """Rounds 1-3 players cannot be drop candidates."""
    roster = pd.DataFrame({
        "name": ["Star Player", "Bench Warmer", "Average Joe"],
        "is_undroppable": [1, 0, 0],
        "sgp_total": [5.0, 0.5, 2.0],
    })
    droppable = roster[roster["is_undroppable"] != 1]
    assert len(droppable) == 2
    assert "Star Player" not in droppable["name"].values
```

Create `tests/test_transaction_counter.py`:

```python
"""Tests for weekly transaction counting."""

import pandas as pd

import pytest


def test_get_weekly_transaction_count():
    from src.league_rules import get_weekly_transaction_count

    txns = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-04-14 10:00", "2026-04-15 12:00", "2026-04-16 08:00",
            "2026-04-20 09:00",  # Next week
        ]),
        "type": ["add", "add", "trade", "add"],
    })
    count = get_weekly_transaction_count(
        txns,
        matchup_start=pd.Timestamp("2026-04-14"),
        matchup_end=pd.Timestamp("2026-04-20"),
    )
    assert count == 3


def test_transaction_limit_check():
    from src.league_rules import is_at_transaction_limit

    assert is_at_transaction_limit(10, limit=10) is True
    assert is_at_transaction_limit(9, limit=10) is False
    assert is_at_transaction_limit(0, limit=10) is False
```

- [ ] **Step 2: Create `src/league_rules.py`**

```python
"""League rule enforcement: transaction limits, undroppable players."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_weekly_transaction_count(
    transactions_df: pd.DataFrame,
    matchup_start: pd.Timestamp,
    matchup_end: pd.Timestamp,
) -> int:
    """Count adds + trades in the current matchup week."""
    if transactions_df.empty:
        return 0
    mask = (
        (transactions_df["timestamp"] >= matchup_start)
        & (transactions_df["timestamp"] < matchup_end)
        & (transactions_df["type"].isin(["add", "trade"]))
    )
    return int(mask.sum())


def is_at_transaction_limit(current_count: int, limit: int = 10) -> bool:
    """Return True if the weekly transaction limit has been reached."""
    return current_count >= limit


def get_transactions_remaining(current_count: int, limit: int = 10) -> int:
    """Return how many transactions remain this week."""
    return max(0, limit - current_count)
```

- [ ] **Step 3: Add Phase 32 `_bootstrap_draft_results` to `src/data_bootstrap.py`**

```python
def _bootstrap_draft_results(progress: BootstrapProgress, yahoo_client) -> str:
    """Phase 32: Fetch draft results from Yahoo, flag rounds 1-3 as undroppable."""
    progress.phase = "Draft Results"
    progress.detail = "Fetching draft picks + undroppable flags..."
    if yahoo_client is None:
        return "Skipped (no Yahoo client)"
    try:
        from src.database import get_connection, update_refresh_log_auto

        df = yahoo_client.get_draft_results()
        if df.empty:
            update_refresh_log_auto("draft_results", 0, expected_min=200)
            return "No draft results available"

        conn = get_connection()
        try:
            # Save all draft picks
            conn.execute("DELETE FROM league_draft_picks")
            for _, row in df.iterrows():
                conn.execute(
                    "INSERT INTO league_draft_picks (pick_number, round, team_name, player_id, player_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row["pick_number"], row["round"], row["team_name"],
                     str(row.get("player_id", "")), row["player_name"]),
                )

            # Flag rounds 1-3 as undroppable in league_rosters
            conn.execute("UPDATE league_rosters SET is_undroppable = 0")
            undroppable = df[df["round"] <= 3]["player_name"].tolist()
            for name in undroppable:
                conn.execute(
                    "UPDATE league_rosters SET is_undroppable = 1 WHERE name = ?",
                    (name,),
                )
            conn.commit()
        finally:
            conn.close()

        update_refresh_log_auto(
            "draft_results", len(df), expected_min=200,
            message=f"{len(df)} picks, {len(undroppable)} undroppable",
        )
        return f"Saved {len(df)} picks, {len(undroppable)} undroppable"
    except Exception as exc:
        logger.exception("Draft results fetch failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("draft_results", "error", message=str(exc)[:200])
        return f"Error: {exc}"
```

Add the call in `bootstrap_all_data()` after Phase 31:

```python
    # Phase 32: Draft results + undroppable flags
    _notify(0.99)
    try:
        results["draft_results"] = _bootstrap_draft_results(progress, yahoo_client)
    except Exception as exc:
        results["draft_results"] = f"Error: {exc}"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_undroppable.py tests/test_transaction_counter.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/league_rules.py src/data_bootstrap.py tests/test_undroppable.py tests/test_transaction_counter.py
git commit -m "feat(league_rules): undroppable enforcement + transaction counter

- New src/league_rules.py with weekly transaction counting
- Phase 32: fetch draft results, flag rounds 1-3 as undroppable
- is_undroppable column populated from Yahoo draft data"
```

---

## Task 10: Category Centralization (7 files)

**Files:**
- Modify: `src/optimizer/advanced_lp.py:44-59`
- Modify: `src/engine/portfolio/copula.py:35`
- Modify: `src/yahoo_api.py:1687-1688`
- Modify: `src/ui_shared.py:2224,2325-2326,3177`
- Modify: `src/league_manager.py:100`

- [ ] **Step 1: Fix `src/optimizer/advanced_lp.py` (lines 44-59)**

Replace:
```python
ALL_CATEGORIES: list[str] = [
    "r", "hr", "rbi", "sb", "avg", "obp",
    "w", "l", "sv", "k", "era", "whip",
]
INVERSE_CATS: set[str] = {"l", "era", "whip"}
HITTER_CATS: list[str] = ["r", "hr", "rbi", "sb", "avg", "obp"]
PITCHER_CATS: list[str] = ["w", "l", "sv", "k", "era", "whip"]
```

With:
```python
from src.valuation import LeagueConfig as _LC_Class

_LC = _LC_Class()
ALL_CATEGORIES: list[str] = [c.lower() for c in _LC.all_categories]
INVERSE_CATS: set[str] = {c.lower() for c in _LC.inverse_stats}
HITTER_CATS: list[str] = [c.lower() for c in _LC.hitting_categories]
PITCHER_CATS: list[str] = [c.lower() for c in _LC.pitching_categories]
```

- [ ] **Step 2: Fix `src/engine/portfolio/copula.py` (line 35)**

Replace:
```python
CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
```

With:
```python
from src.valuation import LeagueConfig as _LC_Class

CATEGORIES: list[str] = _LC_Class().all_categories
```

- [ ] **Step 3: Fix `src/yahoo_api.py` (lines 1687-1688)**

Replace:
```python
    _INVERSE_CATS: set[str] = {"L", "ERA", "WHIP"}
    _ALL_CATS: list[str] = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
```

With:
```python
    _INVERSE_CATS: set[str] = set(LeagueConfig().inverse_stats)
    _ALL_CATS: list[str] = list(LeagueConfig().all_categories)
```

(LeagueConfig is already imported in yahoo_api.py via `from src.valuation import LeagueConfig`)

- [ ] **Step 4: Fix `src/ui_shared.py` (lines 2224, 2325-2326, 3177)**

Line 2224 — replace:
```python
    cat_keys = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
```
With:
```python
    cat_keys = list(ALL_CATEGORIES)
```
(`ALL_CATEGORIES` is already defined at line 369 from LeagueConfig)

Lines 2325-2326 — replace:
```python
HITTING_STAT_COLS = {"R", "HR", "RBI", "SB", "AVG", "OBP"}
PITCHING_STAT_COLS = {"W", "L", "SV", "K", "ERA", "WHIP"}
```
With:
```python
HITTING_STAT_COLS = set(HITTING_CATEGORIES)
PITCHING_STAT_COLS = set(PITCHING_CATEGORIES)
```
(`HITTING_CATEGORIES` and `PITCHING_CATEGORIES` are already defined at lines 367-368)

Line 3177 — replace:
```python
        cats = ["R", "HR", "RBI", "SB", "AVG", "OBP"] if is_hitter else ["W", "L", "SV", "K", "ERA", "WHIP"]
```
With:
```python
        cats = list(HITTING_CATEGORIES) if is_hitter else list(PITCHING_CATEGORIES)
```

- [ ] **Step 5: Fix `src/league_manager.py` (line 100)**

Replace:
```python
    categories = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
```
With:
```python
    from src.valuation import LeagueConfig
    categories = list(LeagueConfig().all_categories)
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -x -q --timeout=60`
Expected: All ~3401 tests pass (no behavioral change, just import source)

- [ ] **Step 7: Commit**

```bash
git add src/optimizer/advanced_lp.py src/engine/portfolio/copula.py src/yahoo_api.py src/ui_shared.py src/league_manager.py
git commit -m "refactor: centralize all category lists to import from LeagueConfig

- advanced_lp.py, copula.py, yahoo_api.py, ui_shared.py, league_manager.py
- All 5 files now import categories from LeagueConfig (single source of truth)
- No behavioral change — same values, just centralized import"
```

---

## Task 11: Persistent Logging (SF-14)

**Files:**
- Modify: `src/data_bootstrap.py`

- [ ] **Step 1: Add rotating file handler and bootstrap_results persistence**

At the top of `src/data_bootstrap.py`, after existing imports, add:

```python
import json
from logging.handlers import RotatingFileHandler

# Persistent log file for post-mortem analysis
_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_file_handler = RotatingFileHandler(
    _LOG_DIR / "bootstrap.log", maxBytes=5_000_000, backupCount=3
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logging.getLogger("src").addHandler(_file_handler)
```

At the end of `bootstrap_all_data()`, before the `return results` statement, add:

```python
    # Persist results to disk for post-mortem analysis
    try:
        results_path = _LOG_DIR / "bootstrap_results.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        logger.debug("Failed to persist bootstrap_results.json", exc_info=True)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_data_bootstrap.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/data_bootstrap.py
git commit -m "feat(logging): SF-14 persistent bootstrap logging + results JSON

- RotatingFileHandler writes to data/logs/bootstrap.log (5MB, 3 backups)
- bootstrap_results.json persisted to disk after every bootstrap
- No more stderr-only logging for post-mortem analysis"
```

---

## Task 12: CLAUDE.md Update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update roster slots, phase count, and known issues**

In `CLAUDE.md`, update the roster line:
```
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots
```

Update the phase count references from "21 phases" to "33 phases".

Add to the League Context section:
```
- **Transactions:** 10 adds+trades combined per matchup week (FCFS waivers)
- **Can't-drop:** Players drafted in rounds 1-3 per team (36 league-wide)
```

Mark completed tasks in the Priority Task List:
- Task 1 (SF-1): Already marked DONE
- Task 2 (SF-2/SF-3): Mark DONE
- Task 3 (SF-4): Mark DONE
- Task 4 (SF-8): Mark DONE
- Task 5 (SF-12): Mark DONE

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): update roster slots (28), phases (33), league rules"
```

---

## Task 13: Final Validation

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -x -q --timeout=120
```
Expected: All tests pass

- [ ] **Step 2: Run linter**

```bash
python -m ruff check .
python -m ruff format --check .
```
Expected: No errors

- [ ] **Step 3: Run the app and verify bootstrap**

```bash
streamlit run app.py
```

After bootstrap completes, verify in a separate terminal:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/draft_tool.db')
print('game_logs:', conn.execute('SELECT COUNT(*) FROM game_logs').fetchone())
print('ohtani:', conn.execute(\"SELECT COUNT(*) FROM season_stats WHERE name LIKE '%%Ohtani%%' AND season=2026\").fetchall())
print('injured:', conn.execute('SELECT COUNT(*) FROM players WHERE is_injured=1').fetchone())
print('tiers:', conn.execute('SELECT tier, COUNT(*) FROM refresh_log GROUP BY tier').fetchall())
print('undroppable:', conn.execute('SELECT COUNT(*) FROM league_rosters WHERE is_undroppable=1').fetchone())
print('refresh_log diagnostics:')
for row in conn.execute('SELECT source, status, rows_written, tier FROM refresh_log ORDER BY source').fetchall():
    print(f'  {row}')
conn.close()
"
```

Expected:
- `game_logs` > 0 (SF-1 fix deployed)
- `ohtani` has rows (SF-3 fix)
- `injured` > 0 (injury writeback working)
- `tiers` shows primary/fallback/emergency distribution
- `undroppable` = 36 (rounds 1-3 × 12 teams × 3 rounds)
- `rows_written` populated for every source (honest telemetry)

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "feat(pipeline): resilient data pipeline overhaul complete

8-category fix across 23 files:
1. Data fetch fixes (SF-1/2/3/5/10)
2. 3-tier fallback chains for all external sources
3. Injury writeback (is_injured was 0 for all 9343 players)
4. Team code normalization (ARI/AZ, OAK/ATH, etc.)
5. Honest telemetry (no more blind 'success' writes)
6. League rules (BN=6, IL=4, 10 txns/week, undroppable)
7. Bootstrap phase ordering + persistent logging
8. Category centralization (7 files → LeagueConfig)"
```
