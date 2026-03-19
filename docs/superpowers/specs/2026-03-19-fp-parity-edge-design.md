# FantasyPros Edge: Prospect Rankings, ECR Consensus, News Intelligence

**Date:** 2026-03-19
**Status:** Draft
**Goal:** Transform HEATER's 3 weakest FantasyPros-parity features from "checkbox parity" into genuine competitive advantages using free public data sources and analytical computation.

---

## Constraints

- Free APIs only (no paid subscriptions)
- Undocumented APIs accepted with graceful fallback chains
- All new features follow existing patterns: TDD, SQLite persistence, staleness-based refresh, `try/except ImportError` degradation
- New dependencies: `feedparser` (RSS parsing). No other new deps.

---

## Feature 1: Prospect Rankings Engine

### Problem

Current implementation is a hardcoded 20-row list in `src/ecr.py:94-115`. No live data, no scouting context, stale immediately.

### Solution

Replace with a live prospect engine combining FanGraphs scouting data + MLB Stats API minor league performance stats + a computed "MLB Readiness Score."

### Data Sources

| Source | Endpoint | Data | Staleness |
|--------|----------|------|-----------|
| FanGraphs Board API | `fangraphs.com/api/prospects/board/data?draft=false&season=2026` | Rankings, FV (20-80), scouting tools (pHit/fHit, pGame/fGame, pRaw/fRaw, pSpd, Fld, pCtl/fCtl for pitchers), ETA, risk level, scouting report narratives, TLDR, fantasy relevance flags | 7 days |
| MLB Stats API MiLB Stats | `statsapi.mlb.com/api/v1/people/{id}/stats?stats=yearByYear&leagueListId=milb_all` | Batting: AVG/OBP/SLG/HR/SB/K%/BB% by level. Pitching: ERA/WHIP/K9/BB9/IP by level. | 1 day |
| MLB Pipeline Scrape (custom) | Direct HTTP scrape of `mlb.com/prospects` with `requests` + BeautifulSoup/JSON parsing | Fallback ranking + basic stats. **Note:** pybaseball has no `top_prospects()` function; we scrape directly. | 7 days |

### New File: `src/prospect_engine.py`

**Key functions:**

```
fetch_fangraphs_prospects(season=2026) -> pd.DataFrame
fetch_milb_stats(mlb_ids: list[int]) -> pd.DataFrame
compute_mlb_readiness_score(prospect_row) -> float
refresh_prospect_rankings(force=False) -> pd.DataFrame
get_prospect_rankings(top_n=100, position=None, org=None) -> pd.DataFrame
get_prospect_detail(prospect_id) -> dict
```

### New DB Table: `prospect_rankings`

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
    hit_present INTEGER, hit_future INTEGER,   -- FG fields: pHit, fHit
    game_present INTEGER, game_future INTEGER,  -- FG fields: pGame, fGame (power for hitters)
    raw_present INTEGER, raw_future INTEGER,    -- FG fields: pRaw, fRaw
    speed INTEGER, field INTEGER,               -- FG fields: pSpd, Fld
    ctrl_present INTEGER, ctrl_future INTEGER,  -- FG fields: pCtl, fCtl (pitchers only)
    scouting_report TEXT,
    tldr TEXT,
    milb_level TEXT,
    milb_avg REAL, milb_obp REAL, milb_slg REAL,
    milb_k_pct REAL, milb_bb_pct REAL, milb_hr INTEGER, milb_sb INTEGER,
    milb_ip REAL, milb_era REAL, milb_whip REAL, milb_k9 REAL, milb_bb9 REAL,
    readiness_score REAL,
    fetched_at TEXT
);
```

### MLB Readiness Score (0-100)

```
readiness = (
    0.40 * fv_normalized          # FV 80 -> 100, FV 20 -> 0
  + 0.25 * age_level_performance  # wOBA vs level avg, age-adjusted
  + 0.20 * eta_proximity          # 2026->100, 2027->75, 2028->50, 2029->25
  + 0.15 * risk_factor            # Low->1.0, Med->0.8, High->0.6
)
```

**`fv_normalized`:** `(fv - 20) / 60 * 100`

**`age_level_performance`:** Compute wOBA proxy from MiLB stats: `(obp * 1.2 + slg * 0.8) / 2`. Compare to level average (AAA: .330, AA: .310, High-A: .300, A: .290). Bonus/penalty for age relative to level average age (AAA avg ~25, AA ~23, High-A ~22, A ~21). Scale to 0-100.

**`eta_proximity`:** Computed relative to current season (2026). ETA 2026 or earlier = 100, ETA 2027 = 75, ETA 2028 = 50, ETA 2029 = 25, ETA 2030+ = 0. Prospects with ETA in the past (already MLB-ready) get full score.

**`risk_factor`:** Scales the raw score. Low risk = 1.0, Medium = 0.8, High = 0.6, Extreme = 0.4.

### Fallback Chain

1. FanGraphs Board API (richest)
2. Custom MLB Pipeline scrape (ranking + basic stats, no scouting tools)
3. Existing hardcoded list (last resort)

Each level gracefully degrades: without FG data, no scouting tools or reports, but still has rankings + stats. Without MLB Pipeline, just the static list.

### UI: Enhanced `pages/9_Leaders.py` Prospects Tab

- Sortable table: Rank, Name, Team, Pos, FV, ETA, Risk, Readiness Score, MiLB slash line
- Filter pills: position, organization, ETA year
- Expandable row: full scouting report, tool grades
- Plotly radar chart for selected prospect (hitters: Hit/Game/Raw/Speed/Field; pitchers: Fastball/Offspeed/Control from FG tool grades)
- "Fantasy Relevant" badge from FG's `Fantasy_Redraft` / `Fantasy_Dynasty` flags

### Test File: `tests/test_prospect_engine.py` (~15 tests)

- FG API response parsing (mock JSON)
- MiLB stats aggregation
- Readiness score computation (hand-calculated)
- Score bounds [0, 100]
- Fallback chain (FG fails -> MLB Pipeline scrape -> static)
- Position/org filtering
- DB round-trip
- Empty/malformed API responses

---

## Feature 2: Multi-Platform ECR Consensus

### Problem

Current `src/ecr.py` scrapes FantasyPros aggregate output and fabricates best/worst rank as `ecr_rank * 0.80 / 1.20`. No real multi-source data.

### Solution

Build a genuine multi-platform consensus from 7 independent ranking sources using a Trimmed Borda Count algorithm. Real min/max ranks from actual source disagreement.

### Data Sources

| Source | Endpoint | Data | Auth | Staleness |
|--------|----------|------|------|-----------|
| ESPN Fantasy API | `lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/2026/players?view=kona_player_info` | Draft rank via `draftRanksByRankType.STANDARD.rank` (nested path). Paginated: use `offset` + `limit` query params (default limit=50, max=1000). | None | 24h |
| Yahoo ADP | Existing `YahooFantasyClient.fetch_yahoo_adp()` method (uses `get_draft_results()` to compute average pick per player) | Average pick, average round, percent drafted | OAuth (existing) | 24h |
| CBS Sports | `cbssports.com/fantasy/baseball/rankings/h2h/top300/` | H2H Top 300 ranked table. **Note:** Page is JS-rendered — use `requests` to try fetching HTML, but fallback gracefully if content is empty/JS-only. CBS may return partial data or require headless browser. Mark as "best-effort" source with 0-1 availability. | Scrape | 24h |
| NFBC ADP | Already in `adp_sources.py` | High-stakes ADP | Scrape (existing) | 24h |
| FanGraphs ADP | Already in `data_pipeline.py` | Steamer-based ADP | JSON API (existing) | 7d |
| FantasyPros ECR | Already in `adp_sources.py` | Aggregated ECR | Scrape (existing) | 24h |
| HEATER SGP Rank | Internal `valuation.py` | SGP-derived rank from 7 projection systems | Local compute | Instant |

### Rewritten Module: `src/ecr.py`

**Key functions:**

```
fetch_espn_rankings() -> pd.DataFrame
fetch_yahoo_adp() -> pd.DataFrame
fetch_cbs_rankings() -> pd.DataFrame
fetch_all_ranking_sources() -> dict[str, pd.DataFrame]
resolve_player_ids_across_sources(sources: dict) -> pd.DataFrame
compute_consensus(resolved_df: pd.DataFrame) -> pd.DataFrame
refresh_ecr_consensus(force=False) -> pd.DataFrame
load_ecr_consensus() -> pd.DataFrame

# Kept from existing module:
blend_ecr_with_projections(valued_pool, consensus_df, ecr_weight=0.15) -> pd.DataFrame
compute_ecr_disagreement(consensus_row) -> str | None

# Moved to prospect_engine.py:
fetch_prospect_rankings()  # removed from ecr.py
filter_prospects_by_position()  # removed from ecr.py
```

### New DB Table: `ecr_consensus`

```sql
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
```

### New DB Table: `player_id_map`

Cross-reference player IDs across platforms:

```sql
CREATE TABLE IF NOT EXISTS player_id_map (
    player_id INTEGER PRIMARY KEY,  -- HEATER internal ID
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
```

**Dedup strategy:** When inserting a new mapping, first check if any external ID already maps to a different `player_id`. If conflict found, merge: keep the lower `player_id`, update all references, delete the duplicate row. This prevents the same real-world player from having two HEATER IDs.

Populated via fuzzy name matching (existing `match_player_id()` from `live_stats.py:27`) with team + position as tiebreakers. Can be seeded from Smart Fantasy Baseball cross-reference sheet.

### Consensus Algorithm: Trimmed Borda Count

```python
def compute_consensus(player_sources: dict[str, int | None]) -> dict:
    """Compute per-player consensus from all sources. consensus_rank is NOT computed here —
    it is assigned AFTER all players have consensus_avg, by sorting all players by consensus_avg
    ascending and assigning sequential integer ranks (1-based, no gaps)."""
    ranks = [v for v in player_sources.values() if v is not None]
    n = len(ranks)
    if n == 0:
        return {"consensus_avg": None, "consensus_rank": None}
    if n >= 4:
        # Trim highest and lowest
        trimmed = sorted(ranks)[1:-1]
    else:
        trimmed = ranks
    return {
        "consensus_avg": round(sum(trimmed) / len(trimmed), 1),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "rank_stddev": round(statistics.stdev(ranks), 1) if n >= 2 else 0.0,
        "n_sources": n,
    }

def assign_consensus_ranks(all_player_consensus: list[dict]) -> list[dict]:
    """Sort all players by consensus_avg ascending, assign consensus_rank 1..N.
    Players with consensus_avg=None are unranked (consensus_rank=None)."""
    ranked = sorted([p for p in all_player_consensus if p["consensus_avg"] is not None],
                    key=lambda p: p["consensus_avg"])
    for i, p in enumerate(ranked, 1):
        p["consensus_rank"] = i
    return all_player_consensus
```

### Disagreement Detection

Replaces fabricated ±20% range:

- `rank_stddev > 30` AND `n_sources >= 3` → "High Disagreement" badge
- `rank_stddev > 15` AND `n_sources >= 3` → "Moderate Disagreement"
- Badge text shows the outlier: e.g., "ESPN: 15th vs Yahoo: 52nd"
- `biggest_gap` computed as max absolute deviation from consensus_avg

### Fallback Chain

Minimum viable: FanGraphs ADP + NFBC ADP + HEATER SGP rank (3 sources, all local/existing). Always available even if every external API breaks.

Full: 7 sources when all APIs respond.

### UI Changes

**Draft board (`app.py`):**
- "ECR Rank" column → "Consensus Rank" with range tooltip (hover shows all source ranks)
- Disagreement badge with platform-specific callouts
- "HEATER vs Consensus" column: delta between SGP rank and consensus rank

**Available Players tab:**
- Sort by consensus rank option
- Filter by disagreement level (show only high-disagreement players = potential value)

### Test File: `tests/test_ecr_consensus.py` (~25 tests)

- ESPN API response parsing (mock JSON, nested `draftRanksByRankType.STANDARD.rank` path)
- ESPN pagination handling (offset + limit)
- Yahoo ADP extraction
- CBS HTML parsing (success + JS-rendered fallback)
- Trimmed Borda count (hand-calculated: 3, 4, 5, 6, 7 source cases)
- Trim removes correct outliers
- `assign_consensus_ranks()` produces sequential 1..N from consensus_avg
- rank_min/rank_max from real data
- rank_stddev computation
- Disagreement badge thresholds
- Player ID resolution across platforms
- Player ID dedup/merge on conflict
- Fallback with missing sources (minimum 3 = FG + NFBC + SGP)
- DB round-trip (ecr_consensus table)
- DB round-trip (player_id_map table)
- Consensus with single source = that source's rank
- Empty source handling
- Backward-compat: `blend_ecr_with_projections()` signature unchanged
- Backward-compat: `compute_ecr_disagreement()` accepts new consensus row format

### Backward Compatibility with Existing Tests

**Critical:** Rewriting `src/ecr.py` will break 12 tests in `tests/test_ecr.py` and imports in `tests/test_prospect_rankings.py`. Migration strategy:

1. **`tests/test_ecr.py`** — Rewrite in place. The 12 existing tests tested the FantasyPros-only implementation. New tests replace them with consensus-based tests. Net test count increases from 12 to ~25.
2. **`tests/test_prospect_rankings.py`** — Currently imports `fetch_prospect_rankings()` from `src/ecr.py`. Update imports to `from src.prospect_engine import get_prospect_rankings`. The 6 existing tests are rewritten to test the new engine.
3. **`blend_ecr_with_projections()`** — Keep same function signature. Internally it now uses `consensus_rank` instead of `ecr_rank`, but the column it produces (`blended_rank`, `ecr_badge`) stays the same. Callers in `app.py` don't change.
4. **`compute_ecr_disagreement()`** — Change signature from `(proj_rank, ecr_rank, threshold=20)` to `(consensus_row)` which reads `rank_stddev` and `n_sources`. Internal call within `blend_ecr_with_projections()` updated: after merging `consensus_df` into `valued_pool` by player_id, each row now contains `rank_stddev`, `n_sources`, `rank_min`, `rank_max` columns. The function passes the merged row to `compute_ecr_disagreement()`. Callers in `app.py` also updated to pass the consensus row format.

---

## Feature 3: News/Transaction Intelligence

### Problem

Current `src/news_fetcher.py` fetches raw MLB Stats API transactions only. No injury detail, no context, no analysis. Just "Player X placed on IL."

### Solution

Aggregate structured data from 4 free sources (MLB API enhanced + Yahoo + ESPN + RotoWire RSS), then generate template-based analytical summaries combining injury models, SGP impact, and replacement recommendations.

### Data Sources

| Source | Endpoint | Data | Auth | Staleness |
|--------|----------|------|------|-----------|
| MLB Stats API (enhanced) | `/api/v1/people/{id}?hydrate=rosterEntries,transactions` | IL status codes, dates, status descriptions, transaction history | None | 1h |
| Yahoo yfpy | Player model fields during roster sync | `injury_note` (body part), `status_full`, `has_player_notes`, `percent_owned` | OAuth (existing) | 6h |
| ESPN News API | `site.api.espn.com/apis/site/v2/sports/baseball/mlb/news` | Headlines, descriptions, `categories[].athleteId` | None | 1h |
| RotoWire RSS | Multiple candidate URLs tried in sequence: `rotowire.com/feed/`, `rotowire.com/rss/news.htm?sport=mlb`, `rotowire.com/baseball/news.php` (RSS discovery). **Note:** RotoWire RSS URLs change periodically and may return empty XML. Implementation tries each candidate, uses first that returns valid entries. Marked as "best-effort" source (0-1 availability). | Headline blurbs (top stories) | None (feedparser) | 1h |

### New File: `src/player_news.py`

**Key functions:**

```
# Fetchers
fetch_mlb_enhanced_status(player_ids: list[int]) -> list[dict]
fetch_espn_news() -> list[dict]
fetch_rotowire_rss() -> list[dict]
aggregate_news(player_id: int) -> list[dict]

# Analytical layer
generate_player_intel(player_id, player_pool, config=None) -> dict
generate_injury_context(player_id, il_status, body_part) -> dict
find_replacement_options(player_id, player_pool, config=None, top_n=3) -> list[dict]
compute_ownership_trend(player_id, lookback_days=7) -> dict  # Reads from ownership_trends table (populated by Yahoo sync). Returns empty dict when no Yahoo data available.
generate_intel_summary(intel: dict) -> str  # template-based

# Batch operations
refresh_all_news(yahoo_client=None, force=False) -> int  # returns count of new items
generate_roster_intel(roster_ids, player_pool, config=None) -> list[dict]
```

### New DB Table: `player_news`

```sql
CREATE TABLE IF NOT EXISTS player_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    source TEXT NOT NULL,  -- 'mlb_api', 'yahoo', 'espn', 'rotowire'
    headline TEXT NOT NULL,
    detail TEXT,
    news_type TEXT,  -- 'injury', 'transaction', 'lineup', 'general'
    injury_body_part TEXT,
    il_status TEXT,  -- 'IL10', 'IL15', 'IL60', 'DTD', 'active', null
    sentiment_score REAL,
    published_at TEXT,
    fetched_at TEXT,
    UNIQUE(player_id, source, headline, published_at)  -- prevent duplicate inserts
);
CREATE INDEX IF NOT EXISTS idx_player_news_player ON player_news(player_id);
CREATE INDEX IF NOT EXISTS idx_player_news_type ON player_news(news_type);
```

### New DB Table: `ownership_trends`

```sql
CREATE TABLE IF NOT EXISTS ownership_trends (
    player_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    percent_owned REAL,
    delta_7d REAL,
    PRIMARY KEY (player_id, date)
);
```

### Template-Based Summary Generation

No LLM required. Pattern-matched templates by `news_type`:

**Injury:**
```
"{status} ({body_part}). Expected ~{duration:.1f} weeks. Lost SGP: {lost_sgp:+.1f}. Best FA replacement: {replacement_name} ({replacement_sgp:+.1f} SGP). Ownership {trend_direction} {delta:.0f}% this week."
```

**Trade:**
```
"Traded to {new_team}. Park factor: {park_factor:.2f}. {park_context}. Projected SGP change: {sgp_delta:+.2f}."
```

**Call-up:**
```
"Called up from {level}. MiLB: {avg}/{obp}/{slg}. Projected SGP: {sgp:.2f}. {roster_context}."
```

**Lineup change:**
```
"Moved to {slot} in batting order. Projected PA/week: {pa:.1f} ({pa_delta:+.1f}). SGP adjustment: {sgp_delta:+.2f}."
```

Fields populated from:
- `src/il_manager.py` → IL duration estimates, lost SGP computation (already exists in this branch)
- `src/injury_model.py` → health score, age risk
- `waiver_wire.py` → FA replacement candidates
- `data_bootstrap.py:PARK_FACTORS` → park context
- `prospect_engine.py` → MiLB stats for call-ups
- `ownership_trends` table → ownership direction

### Modifications to Existing Modules

| File | Change |
|------|--------|
| `src/live_stats.py` | Add `fetch_player_enhanced_status(mlb_id)` using `hydrate=rosterEntries,transactions` |
| `src/yahoo_api.py` | Extract `injury_note`, `status_full`, `percent_owned` fields from Player model during `sync_league_data()` |
| `src/data_bootstrap.py` | Add Phase 14: `_refresh_news()` calling `refresh_all_news()` with 1h staleness |
| `src/news_sentiment.py` | No changes needed — existing keyword scorer processes ESPN/RotoWire text |
| `src/news_fetcher.py` | **Unchanged.** Keeps existing MLB transaction logic (`fetch_recent_transactions()`, `fetch_player_news()`). `player_news.py` imports and delegates to it for MLB API data — no duplication. ESPN + RotoWire fetchers live exclusively in `player_news.py`. |

### Fallback Chain

- If ESPN breaks → 3 sources (MLB API + Yahoo + RotoWire)
- If RotoWire RSS changes → 3 sources (MLB API + Yahoo + ESPN)
- If Yahoo not connected → 2 sources (MLB API + ESPN) — no injury body part, no ownership trends
- Analytical layer (Weibull + SGP + FA replacements) works with just IL status from any single source

### UI: News Feed on `pages/1_My_Team.py`

New "News & Alerts" tab:
- Player intel cards for all roster players with recent news
- Each card: headline, source badge (colored dot per source), injury context block (if applicable), ownership trend arrow, sentiment indicator
- Sort options: recency, severity (|sentiment_score|), SGP impact
- IL alert cards integrate with existing `il_manager.py` banner
- "Expand" shows full detail + replacement options + ownership chart

### New Dependency

`feedparser` — add to `requirements.txt`. Lightweight RSS parser, no sub-dependencies. Wrapped in `try/except ImportError` with `FEEDPARSER_AVAILABLE` flag.

### Test File: `tests/test_player_news.py` (~20 tests)

- ESPN news API parsing (mock JSON)
- RotoWire RSS parsing (mock XML)
- MLB enhanced status parsing (mock JSON)
- Yahoo field extraction
- Injury context generation (Weibull integration)
- Replacement option ranking (SGP-based)
- Ownership trend computation
- Template summary generation (all 4 types)
- Sentiment scoring of ESPN headlines
- Player ID cross-reference from ESPN athleteId
- DB round-trip (insert + query by player + query by type)
- Empty API responses
- Fallback with missing sources
- `generate_roster_intel()` batch operation

---

## Cross-Cutting Concerns

### Bootstrap Integration

`src/data_bootstrap.py` adds three new phases after the existing Phase 12 (deduplication):

| Phase | Function | Staleness | Dependencies |
|-------|----------|-----------|-------------|
| Phase 13: Prospects | `_refresh_prospects()` → `prospect_engine.refresh_prospect_rankings()` | 7d | Phase 1 (players) for mlb_id cross-ref |
| Phase 14: News Intelligence | `_refresh_news()` → `player_news.refresh_all_news()` | 1h | Phase 1 (players), Phase 7 (Yahoo, optional) |
| Phase 15: ECR Consensus | `_refresh_ecr_consensus()` → `ecr.refresh_ecr_consensus()` | 24h | Phase 1 (players), Phase 3 (projections — needs SGP ranks), Phase 9 (ADP sources) |

**Important:** ECR consensus is its own phase (not embedded in Phase 3) because it depends on projection data being fully loaded AND ADP sources being refreshed first. It must run after both Phase 3 and Phase 9.

**Relationship to existing Phase 11 (`_bootstrap_news`):** Phase 11 fetches raw MLB transactions via `news_fetcher.fetch_recent_transactions()` and stores them in the `transactions` table. Phase 14 builds on top: it calls `news_fetcher.fetch_recent_transactions()` for MLB data (via import, not duplication), then adds ESPN News API + RotoWire RSS + Yahoo injury fields, stores everything in the new `player_news` table, and runs the analytical layer (templates + SGP impact). Phase 11 is **retained** for backward compatibility — the `transactions` table is still populated for existing consumers. Phase 14 is additive.

`StalenessConfig` dataclass updated with three new fields:
```python
prospects_hours: float = 168  # 7 days
news_hours: float = 1         # 1 hour
ecr_consensus_hours: float = 24  # 24 hours
```

### Database Migration

`src/database.py` `init_db()` adds 5 new tables:
- `prospect_rankings`
- `ecr_consensus`
- `player_id_map`
- `player_news`
- `ownership_trends`

Current count: 14 tables in `init_db()`. Adding 5 new = **19 tables total** in `init_db()`.

### Rate Limiting

All new API calls follow existing `data_pipeline.py` pattern:
- 0.5s between requests
- User-Agent header
- Staleness checks before fetching
- Per-source error counting (3 consecutive failures → skip source this cycle)

### New Files Summary

| File | Responsibility | Tests |
|------|---------------|-------|
| `src/prospect_engine.py` | FG + MLB API prospect data, readiness score | ~15 |
| `src/player_news.py` | Multi-source news aggregation, analytical intel | ~20 |
| `src/ecr.py` (rewrite) | Multi-platform consensus builder | ~25 |

### Modified Files Summary

| File | Changes |
|------|---------|
| `src/database.py` | 5 new tables in `init_db()` |
| `src/data_bootstrap.py` | Phase 13 (prospects) + Phase 14 (news) + Phase 15 (ECR consensus). **Existing Phase 11 (`_bootstrap_news`) retained** for MLB transaction fetching; Phase 14 adds ESPN/RotoWire/Yahoo sources and analytical layer on top. |
| `src/live_stats.py` | Add `fetch_player_enhanced_status()` |
| `src/yahoo_api.py` | Extract injury_note, status_full, percent_owned |
| `src/news_fetcher.py` | **Unchanged** — `player_news.py` imports from it |
| `pages/9_Leaders.py` | Enhanced Prospects tab with radar charts |
| `pages/1_My_Team.py` | News & Alerts tab |
| `app.py` | Consensus Rank column + disagreement badges in draft board |
| `requirements.txt` | Add `feedparser` |

### Total New Tests: ~60

Across 3 test files: `test_prospect_engine.py` (~15), `test_ecr_consensus.py` (~25), `test_player_news.py` (~20). Note: 12 existing tests in `test_ecr.py` and 6 in `test_prospect_rankings.py` are rewritten (not net-new but updated to test new implementations).
