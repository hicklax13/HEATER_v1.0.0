# Resilient Data Pipeline Overhaul — Design Spec

**Date:** 2026-04-18
**Author:** Claude + Connor Hickey
**Status:** Approved (pending implementation plan)
**Scope:** Fix all 24+ data pipeline issues, add tiered fallback chains, enforce league rules, eliminate hardcoded data

---

## 1. Context & Motivation

A comprehensive audit on 2026-04-18 revealed **14 silent-failure patterns (SF-1 through SF-14)**, plus **10+ additional issues** including missing league rule enforcement, hardcoded data, and broken injury detection. The most critical: game logs has 0 rows, Ohtani is completely invisible, `is_injured=0` for all 9,343 players, and the refresh_log lies "success" when saving 0 rows.

### User Requirements
- **Always refresh ALL data on app launch** (`force=True` every session)
- **Active game-day manager** — near-real-time during 7-11 PM ET game window
- **Zero hardcoded data** — all data sourced live, hardcoded values only as last-resort emergency fallback with UI warning
- **Live-first with auto-fallback** — try browser headers on FanGraphs, fall back to MLB API, then emergency static

### League Configuration (FourzynBurn, Yahoo)
- **Format:** 12-team H2H Categories, FCFS waivers
- **Hitting (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching (6):** W, L, SV, K, ERA, WHIP
- **Inverse cats:** L, ERA, WHIP (lower is better)
- **Rate stats:** AVG, OBP (weighted by PA/AB), ERA, WHIP (weighted by IP)
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = **28 max slots** (18 active + 6 bench + 4 IL)
- **Transactions:** 10 adds+trades combined per matchup week
- **Can't-drop:** Players drafted in rounds 1-3 by each team (36 players league-wide)
- **Yahoo game_key:** 469

---

## 2. Architecture Overview

### Universal Fallback Pattern

Every external data source follows a 3-tier chain:

```
Tier 1: Primary source (pybaseball/FanGraphs with browser headers, or direct API)
   |
   v fails (403, timeout, empty response, parse error)
   |
Tier 2: Fallback source (different API — usually MLB Stats API)
   |
   v fails
   |
Tier 3: Emergency (cached DB rows, or hardcoded 2026 values with UI warning)
```

**Shared utility** in `src/data_fetch_utils.py` (new file):

```python
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://www.fangraphs.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_with_fallback(
    source_name: str,
    primary_fn: Callable,
    fallback_fn: Callable | None = None,
    emergency_fn: Callable | None = None,
    expected_min: int = 1,
) -> tuple[Any, str]:
    """Execute 3-tier fallback chain. Returns (data, tier_used)."""
    for tier, fn in [("primary", primary_fn), ("fallback", fallback_fn), ("emergency", emergency_fn)]:
        if fn is None:
            continue
        try:
            result = fn()
            if result is not None and (not hasattr(result, '__len__') or len(result) > 0):
                if tier == "emergency":
                    logger.warning("Using emergency fallback for %s", source_name)
                return result, tier
        except Exception as e:
            logger.warning("Tier %s failed for %s: %s", tier, source_name, e)
    return None, "failed"
```

### Honest Telemetry

All bootstrap phases use `update_refresh_log_auto()` with:
- `rows_written`: actual count of rows stored
- `expected_min`: threshold for "partial" vs "success"
- `message`: human-readable summary including tier used
- `tier`: new column — "primary", "fallback", "emergency", or "failed"

No more `update_refresh_log("source", "success")` without row counts.

---

## 3. Fix Categories

### 3.1 Data Fetch Fixes (SF-1, SF-2, SF-3, SF-5, SF-10)

#### SF-1: Game Logs (already fixed in bcb7c86, needs bootstrap re-run)

No code change needed. The fix uses the correct endpoint:
```python
statsapi.get("person", {
    "personId": mlb_id,
    "hydrate": f"stats(group=[{group}],type=[gameLog],season={season})",
})
```
Response shape: `resp["people"][0]["stats"][0]["splits"][]`.

The next app launch with `force=True` will populate `game_logs`.

#### SF-2 + SF-3: Season Stats Discard + Two-Way Players

**File:** `src/live_stats.py`

**Change 1 — Line 311:** Fix Two-Way Player detection:
```python
# Before:
is_pitcher = pos_type == "Pitcher"
# After:
is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
```

**Change 2 — Two-Way Player dual emission:** For `pos_type == "Two-Way Player"`, emit TWO stat rows — one with hitting stats (`is_hitter=1`) and one with pitching stats (`is_hitter=0`). Both feed into their respective category SGPs.

**Change 3 — Save threshold:** In `save_season_stats_to_db()`:
```python
update_refresh_log_auto(
    "season_stats",
    saved_count,
    expected_min=max(500, int(len(fetched_df) * 0.70)),
    message=f"saved {saved_count}/{len(fetched_df)}, "
            f"no_match={no_match}, type_skip={type_skip}, backfilled={backfilled}",
)
```

#### SF-10: Players mlb_id Backfill

**File:** `src/data_bootstrap.py`

After the `_bootstrap_players` phase, add a **mlb_id backfill pass**:
1. Query all players with `mlb_id IS NULL`
2. For each, attempt to match against MLB Stats API by normalized name + team
3. Update `players.mlb_id` for matches

Also raise `expected_min` from 500 to `max(500, int(len(df) * 0.80))`.

#### SF-5: Depth Charts Fallback

**File:** `src/depth_charts.py`, `src/data_bootstrap.py`

Add tiered fallback:
- **Tier 1:** Roster Resource scrape (current code)
- **Tier 2:** MLB Stats API `team_roster` with `hydrate=depthChartPosition`
- **Tier 3:** Infer lineup order from most recent game logs (batting order position)

### 3.2 Tiered Fallback Chains

#### Source-by-source fallback matrix:

| Source | Tier 1 (Primary) | Tier 2 (Fallback) | Tier 3 (Emergency) |
|--------|------------------|-------------------|-------------------|
| Team Strength | pybaseball + browser headers | `_fetch_team_strength_statsapi()` | Cached DB rows + orange warning |
| Stuff+/Batting | pybaseball `pitching_stats()` + headers | MLB API K%, SwStr%, BB% | Neutral defaults + red warning |
| Park Factors | pybaseball home/away splits + headers | MLB API home/away ERA/OPS | 2026 FanGraphs 5yr values + orange warning |
| Depth Charts | Roster Resource | MLB API depthChartPosition | Game log lineup inference |
| Catcher Framing | pybaseball `statcast_catcher_framing()` | Baseball Savant API | Neutral (0 framing runs) |
| Umpire Tendencies | MLB API boxscores (fix parse) | Baseball Savant umpire cards | Neutral defaults |
| ADP | Yahoo draft results ADP | FanGraphs ADP from projections | ECR rankings as proxy |
| Sprint Speed | pybaseball `statcast_sprint_speed()` | Baseball Savant direct | League-average (27.0 ft/s) |
| Bat Speed | pybaseball bat tracking + headers | Baseball Savant direct | Neutral |

#### Emergency Park Factors (2026 FanGraphs 5yr, updated 2026-04-18):

```python
_PARK_FACTORS_EMERGENCY_2026 = {
    "ARI": 1.007, "ATL": 1.001, "BAL": 0.986, "BOS": 1.042,
    "CHC": 0.979, "CWS": 1.003, "CIN": 1.046, "CLE": 0.989,
    "COL": 1.134, "DET": 1.003, "HOU": 0.995, "KC":  1.031,
    "LAA": 1.012, "LAD": 0.991, "MIA": 1.010, "MIL": 0.989,
    "MIN": 1.008, "NYM": 0.963, "NYY": 0.989, "ATH": 1.029,
    "PHI": 1.013, "PIT": 1.015, "SD":  0.959, "SF":  0.973,
    "SEA": 0.935, "STL": 0.975, "TB":  1.009, "TEX": 0.987,
    "TOR": 0.995, "WSH": 0.996,
}
# Source: FanGraphs 5yr regressed "Basic" column, fangraphs.com/guts.aspx?type=pf
# Scale: 1.000 = neutral. Values >1.0 = hitter-friendly, <1.0 = pitcher-friendly.
# Note: KC (1.031) may be low — Kauffman Stadium walls moved in 8-10ft for 2026.
```

#### Browser Headers Implementation

**New file:** `src/data_fetch_utils.py`

Provides `patch_pybaseball_session()` context manager that temporarily overrides pybaseball's requests session headers with browser-like values. Used by all pybaseball-dependent bootstrap phases.

If pybaseball calls fail even with browser headers, the context manager logs the failure and the caller falls through to Tier 2.

### 3.3 Injury Writeback

**Problem:** `is_injured=0` for ALL 9,343 players despite known IL placements.

**Root cause:** ESPN injuries write to `player_news` table only. Yahoo roster data has `status` field (`IL10`, `IL15`, `IL60`, `DTD`, `active`) but it's never written back to `players.is_injured`.

**Fix — dual-source writeback:**

1. **Yahoo source (authoritative for your roster):** During Yahoo sync phase, for each player in `league_rosters`:
   - If `status` in (`IL10`, `IL15`, `IL60`, `DTD`) → set `players.is_injured=1`, `players.injury_note=status_full`
   - If `status == "active"` → set `players.is_injured=0`

2. **ESPN source (authoritative for league-wide):** After `save_espn_injuries_to_db()`:
   - Match ESPN injuries to `players` table by name+team
   - Set `is_injured=1` and `injury_note` for matched players
   - Do NOT clear `is_injured` for unmatched players (they might just not be on the ESPN list yet)

3. **New bootstrap phase (Phase 33):** `_bootstrap_injury_writeback` — runs AFTER Yahoo sync (Phase 7) and news/injuries (Phase 14). Consolidates both sources into `players.is_injured`.

4. **IL slot awareness:** The LP optimizer's `_il_statuses` set already correctly excludes IL/DTD players from starter slots. Once `is_injured` is actually populated, auto-IL alerts in `src/alerts.py` will fire correctly.

### 3.4 Team Code Normalization (SF-8)

**Problem:** Multiple modules have independent team code mappings. `ARI` and `AZ` coexist as separate rows. `_TEAM_EQUIVALENCES` in `daily_optimizer.py` is incomplete.

**Fix — single canonical mapping in `src/valuation.py`:**

```python
TEAM_CODE_CANONICAL: dict[str, str] = {
    "ARI": "ARI", "AZ": "ARI",
    "ATH": "ATH", "OAK": "ATH",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
    "SF": "SF", "SFG": "SF",
    "SD": "SD", "SDP": "SD",
    "TB": "TB", "TBR": "TB",
    "KC": "KC", "KCR": "KC",
    "CWS": "CWS", "CHW": "CWS",
    # All 30 canonical codes also map to themselves
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

**Files to update:** Remove `_TEAM_EQUIVALENCES` from `daily_optimizer.py`, remove `FG_TEAM_TO_ABBR` from `game_day.py` (replace with `canonicalize_team()`), and use `canonicalize_team()` in all team code lookups.

**DB cleanup at bootstrap start:**
```sql
DELETE FROM team_strength WHERE fetched_at < datetime('now', '-3 days');
```

### 3.5 Honest Telemetry (SF-4, SF-13, SF-14)

#### 5a. All phases use `update_refresh_log_auto()`

Audit all 27+ `_bootstrap_*` functions. Every one must call `update_refresh_log_auto()` with actual `rows_written` and appropriate `expected_min`. No more `update_refresh_log("source", "success")` without validation.

#### 5b. Race condition fix (SF-4)

Remove `_bootstrap_team_strength` from the Phase 20+21 ThreadPoolExecutor. Run sequentially:
```python
# Phase 20: Game Day Intel (solo)
if gd_stale:
    results["game_day"] = _bootstrap_game_day(progress)
# Phase 21: Team Strength (solo, after game_day — avoids double fetch_team_strength)
if ts_stale:
    results["team_strength"] = _bootstrap_team_strength(progress)
```

Also: `_bootstrap_game_day` should NOT call `fetch_team_strength()` internally. It should read from the DB after Phase 21 writes it, or accept team_strength as a parameter.

#### 5c. Persistent logging (SF-14)

Add rotating file handler in `src/data_bootstrap.py`:
```python
from logging.handlers import RotatingFileHandler

_log_dir = Path("data/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = RotatingFileHandler(
    _log_dir / "bootstrap.log", maxBytes=5_000_000, backupCount=3
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
))
logging.getLogger("src").addHandler(_file_handler)
```

Also persist `bootstrap_results` dict to `data/logs/bootstrap_results.json` at bootstrap end.

#### 5d. Tier tracking in refresh_log

Add `tier` column:
```sql
ALTER TABLE refresh_log ADD COLUMN tier TEXT DEFAULT 'primary';
```

Values: `primary`, `fallback`, `emergency`, `failed`.

Written by `update_refresh_log_auto()` (add `tier` parameter).

### 3.6 League Rules Enforcement

#### 6a. Correct LeagueConfig roster slots

**File:** `src/valuation.py`

```python
roster_slots: dict = field(default_factory=lambda: {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "OF": 3, "Util": 2,
    "SP": 2, "RP": 2, "P": 4,
    "BN": 6,   # was 5
    "IL": 4,   # was missing entirely
})
# Total: 28 max (18 active + 6 BN + 4 IL)
```

Update `CLAUDE.md` to reflect 28 total slots (not 23).

#### 6b. Undroppable players (rounds 1-3)

**New bootstrap phase (Phase 32):** `_bootstrap_draft_results`

```python
def _bootstrap_draft_results(progress, yahoo_client):
    """Fetch draft results from Yahoo and flag rounds 1-3 as undroppable."""
    if yahoo_client is None:
        return "Skipped (no Yahoo client)"

    df = yahoo_client.get_draft_results()
    if df.empty:
        return "No draft results available"

    # Save to league_draft_picks table
    save_draft_picks_to_db(df)

    # Flag rounds 1-3 as undroppable
    undroppable_names = df[df["round"] <= 3]["player_name"].tolist()
    mark_undroppable(undroppable_names)

    return f"Saved {len(df)} picks, {len(undroppable_names)} undroppable"
```

**Schema addition to `league_rosters`:**
```sql
ALTER TABLE league_rosters ADD COLUMN is_undroppable INTEGER DEFAULT 0;
```

**Enforcement points:**
- `src/waiver_wire.py` — filter `is_undroppable=1` from drop candidates
- `src/optimizer/fa_recommender.py` — filter from `_pick_drop()` candidates
- `src/trade_finder.py` — allow trading undroppable players (just can't drop)
- Streaming module — never suggest dropping an undroppable player

#### 6c. Weekly transaction counter

**Data source:** `yds.get_transactions()` filtered to current matchup week dates.

**Computation:**
```python
def get_weekly_transaction_count(transactions_df, matchup_start, matchup_end):
    """Count adds + trades in the current matchup week."""
    week_txns = transactions_df[
        (transactions_df["timestamp"] >= matchup_start) &
        (transactions_df["timestamp"] <= matchup_end) &
        (transactions_df["type"].isin(["add", "trade"]))
    ]
    return len(week_txns)
```

**Display:** In optimizer and FA pages: `"Moves: 7/10 this week"`.

**Enforcement:**
- At 10/10: disable streaming suggestions, show warning banner
- At 8+/10: FA recommender prioritizes high-impact adds only, warns "2 moves remaining"

**Constant:** `WEEKLY_TRANSACTION_LIMIT = 10` in `LeagueConfig`.

#### 6d. Player disambiguation

When duplicate player names exist (e.g., two Max Muncys), display as `"Max Muncy (LAD)"` vs `"Max Muncy (ATH)"`. The `mlb_id` backfill (SF-10 fix) ensures they're differentiated internally. The display name logic applies in:
- All Streamlit dataframe renderers
- Trade finder results
- FA recommender results
- Player comparison page

### 3.7 Bootstrap Fixes

#### Phase ordering (revised)

```
Phase 1:     Players (MLB Stats API)
Phase 2-3:   Park Factors + Projections (parallel)
Phase 3b:    Extended Roster
Phase 4-5:   Live Stats + Historical (parallel) — with SF-2/SF-3 fixes
Phase 6:     Injury Data (derived)
Phase 7:     Yahoo Sync (rosters, standings, FA, transactions)
Phase 8:     (reserved)
Phase 9:     ADP Sources (with fallback)
Phase 9b:    Depth Charts (with fallback chain)
Phase 10:    Contracts
Phase 11:    News/Transactions
Phase 12:    Deduplication
Phase 13:    Prospects
Phase 14:    News Intelligence + ESPN Injuries
Phase 15:    ECR Consensus
Phase 16:    Player ID Map + mlb_id Backfill (ENHANCED)
Phase 17:    Yahoo Transactions
Phase 18:    Yahoo Free Agents
Phase 19:    ROS Projections (Bayesian)
Phase 20:    Game Day Intel (SOLO — no longer parallel with 21)
Phase 21:    Team Strength (SOLO — after 20, no double fetch)
Phase 22-24: Stuff+/Batting/Sprint (parallel, with browser headers + fallback)
Phase 25-27: Dynamic PF/Bat Speed/40-Man (parallel)
Phase 28-30: Umpire/Catcher/PvB (parallel, with fallback chains)
Phase 31:    Game Logs
Phase 32:    Draft Results + Undroppable Flags (NEW)
Phase 33:    Injury Writeback (NEW — consolidates Yahoo + ESPN → players.is_injured)
```

#### expected_min threshold audit

| Phase | Current | New | Rationale |
|-------|---------|-----|-----------|
| Players | 500 | `max(500, int(len(df) * 0.80))` | Flag when >20% drops |
| Season stats | 500 | `max(500, int(len(df) * 0.70))` | Flag when >30% drops |
| Game logs | 200 | `max(100, len(rostered_ids) * 10)` | ~10 games/player |
| Team strength | (none) | 28 | At least 28/30 teams |
| Park factors | (none) | 28 | At least 28/30 parks |
| Draft results | (none) | 200 | ~276 picks in 23-round draft |
| Depth charts | (none) | 20 | At least 20/30 teams |

#### Dynamic game-window refresh

The existing `_live_stats_ttl_hours()` returns 0.25h during 19:00-00:59 ET. Extend to also apply to:
- Game logs: 0.5h during game window (picks up in-progress stats)
- Matchup data: 5 min always (already exists)
- Team strength: 4h always (doesn't change intra-day)

### 3.8 UI Transparency + Category Centralization

#### Data tier badges

In the Data Status panel (splash screen and sidebar), each data source shows:
- **Green dot** = Tier 1 live primary data
- **Orange dot** = Tier 2 fallback + tooltip: "Using MLB Stats API fallback"
- **Red dot** = Tier 3 emergency + text: "Using 2026-04-18 cached values"
- **Gray dot** = Failed / no data

Source: `refresh_log.tier` column.

#### Locked game display (SF-12)

**File:** `src/optimizer/daily_optimizer.py`

Change the early-return branch at line 670:
```python
# Before:
matchup_mult = 0.0
# After:
matchup_mult = None  # Sentinel for "excluded, not computed"
```

**File:** `pages/6_Line-up_Optimizer.py`

Table renderer: when `matchup_mult is None` or `NaN`, display `—` instead of `0.00`.

Add `reason` column to DCV output:
- `LOCKED` — game in progress or finished
- `IL` — player on injured list
- `OFF_DAY` — team not playing today
- `NOT_PROBABLE` — pure SP not in probable starters

#### Transaction budget display

In optimizer page header and FA page header:
```
Moves: 7/10 this week | 3 remaining
```

Orange text when 8+/10. Red when 10/10.

#### Category constant centralization

Refactor these 7 files to import from `LeagueConfig` instead of hardcoding category lists:

| File | Current Hardcoding | Fix |
|------|-------------------|-----|
| `src/optimizer/advanced_lp.py` (L44-60) | Hardcoded hitting/pitching cats | `from src.valuation import LeagueConfig; config = LeagueConfig()` |
| `src/league_manager.py` (L100) | Hardcoded cat list | Import from LeagueConfig |
| `src/player_databank.py` (L63-64) | `HITTING_CATS` constant | Import from LeagueConfig |
| `src/ui_shared.py` | Multiple hardcoded lists | Import from LeagueConfig |
| `src/yahoo_api.py` | `_ALL_CATS` | Import from LeagueConfig |
| `src/engine/portfolio/copula.py` | Duplicated cats | Import from LeagueConfig |
| `src/engine/portfolio/valuation.py` | Duplicated cats | Import from LeagueConfig |

---

## 4. Files Modified (Estimated)

| File | Changes |
|------|---------|
| `src/live_stats.py` | SF-2/SF-3 fix (Two-Way Player, save threshold) |
| `src/data_bootstrap.py` | Phase ordering, expected_min, race condition, new phases 32-33, persistent logging, browser headers |
| `src/data_fetch_utils.py` | **NEW** — shared browser headers, `fetch_with_fallback()` |
| `src/depth_charts.py` | MLB API fallback for Tier 2 |
| `src/game_day.py` | Remove internal `fetch_team_strength()` call from game_day, canonicalize_team usage |
| `src/valuation.py` | `TEAM_CODE_CANONICAL`, `canonicalize_team()`, roster_slots fix (BN=6, IL=4), `WEEKLY_TRANSACTION_LIMIT` |
| `src/database.py` | Schema migrations (refresh_log.tier, league_rosters.is_undroppable) |
| `src/espn_injuries.py` | Injury writeback to `players.is_injured` |
| `src/yahoo_api.py` | Category centralization, extract `is_undroppable` from Yahoo |
| `src/optimizer/daily_optimizer.py` | Remove `_TEAM_EQUIVALENCES`, use `canonicalize_team()`, SF-12 fix |
| `src/optimizer/fa_recommender.py` | Undroppable filter, transaction budget awareness |
| `src/optimizer/shared_data_layer.py` | SF-9 recency filter on opp_pitcher_stats |
| `src/optimizer/advanced_lp.py` | Category centralization |
| `src/league_manager.py` | Category centralization |
| `src/player_databank.py` | Category centralization |
| `src/ui_shared.py` | Category centralization, data tier badges |
| `src/engine/portfolio/copula.py` | Category centralization |
| `src/engine/portfolio/valuation.py` | Category centralization |
| `src/waiver_wire.py` | Undroppable filter |
| `src/trade_finder.py` | Undroppable awareness (allow trade, prevent drop) |
| `pages/6_Line-up_Optimizer.py` | SF-12 display fix, transaction counter, tier badges |
| `pages/4_Free_Agents.py` | Transaction counter display |
| `CLAUDE.md` | Correct roster slots (28 total), phase count, new phases |

**Estimated:** ~23 files modified, 1 new file created.

---

## 5. Testing Strategy

### Existing tests to update
- `tests/test_live_stats.py` — add Two-Way Player test cases
- `tests/test_data_bootstrap.py` — add expected_min validation tests
- `tests/test_optimizer_integration.py` — verify with corrected roster slots

### New tests to add
- `test_data_fetch_utils.py` — `fetch_with_fallback()` 3-tier chain, browser headers
- `test_team_code_canonical.py` — all team code variants canonicalize correctly
- `test_injury_writeback.py` — ESPN + Yahoo injuries update `players.is_injured`
- `test_undroppable.py` — rounds 1-3 flagged, filtered from drop candidates
- `test_transaction_counter.py` — weekly count computation, limit enforcement

### Validation after implementation
```bash
python -m pytest -x -q                    # All ~3401 tests pass
python -m ruff check .                    # No lint errors
streamlit run app.py                      # Manual: verify Data Status panel shows tiers
# After bootstrap completes:
python -c "
import sqlite3
conn = sqlite3.connect('data/draft_tool.db')
print('game_logs:', conn.execute('SELECT COUNT(*) FROM game_logs').fetchone())
print('ohtani:', conn.execute(\"SELECT COUNT(*) FROM season_stats WHERE name LIKE '%Ohtani%' AND season=2026\").fetchall())
print('injured:', conn.execute('SELECT COUNT(*) FROM players WHERE is_injured=1').fetchone())
print('undroppable:', conn.execute('SELECT COUNT(*) FROM league_rosters WHERE is_undroppable=1').fetchone())
print('tiers:', conn.execute('SELECT tier, COUNT(*) FROM refresh_log GROUP BY tier').fetchall())
conn.close()
"
```

---

## 6. Out of Scope

- Full `data_bootstrap.py` rewrite (Approach C — too risky)
- Deployment to Vercel (app is Streamlit, not Next.js)
- Real-time WebSocket updates (Streamlit doesn't support this natively)
- FAAB budget tracking (league uses FCFS, not FAAB)
- Yahoo global undroppable list (league uses custom rounds 1-3 rule)

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| FanGraphs blocks browser headers too | Automatic fallback to MLB API (Tier 2) — no manual intervention |
| mlb_id backfill matches wrong player (name collision) | Match on normalized name + team + position. Require 2/3 match. Log ambiguous matches. |
| Bootstrap takes too long with new phases | Phases 32-33 are lightweight (1 Yahoo API call + DB writes). No measurable impact. |
| Injury writeback overwrites correct data | Yahoo is authoritative for YOUR roster (exact IL status). ESPN supplements for league-wide. Never clear `is_injured` based on ESPN absence alone. |
| pybaseball session header patch breaks other code | Context manager ensures headers are restored after each call. No global state mutation. |
| LeagueConfig roster slot change breaks LP solver | LP solver reads starter slots only (no BN/IL). BN=6 and IL=4 are used by roster display and simulation, not the LP itself. |
