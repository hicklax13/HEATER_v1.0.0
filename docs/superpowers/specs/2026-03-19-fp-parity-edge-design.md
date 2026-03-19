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
| FanGraphs Board API | `fangraphs.com/api/prospects/board/data?draft=false&season=2026` | Rankings, FV (20-80), scouting tools (Hit/Power/Speed/Field/Arm present+future), ETA, risk level, scouting report narratives, TLDR, fantasy relevance flags | 7 days |
| MLB Stats API MiLB Stats | `statsapi.mlb.com/api/v1/people/{id}/stats?stats=yearByYear&leagueListId=milb_all` | Batting: AVG/OBP/SLG/HR/SB/K%/BB% by level. Pitching: ERA/WHIP/K9/BB9/IP by level. | 1 day |
| pybaseball `top_prospects()` | Scrapes MLB Pipeline | Fallback ranking + basic stats | 7 days |

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
    hit_present INTEGER, hit_future INTEGER,
    power_present INTEGER, power_future INTEGER,
    speed INTEGER, field INTEGER, arm INTEGER,
    scouting_report TEXT,
    tldr TEXT,
    milb_level TEXT,
    milb_avg REAL, milb_obp REAL, milb_slg REAL,
    milb_k_pct REAL, milb_bb_pct REAL, milb_hr INTEGER,
    milb_ip REAL, milb_era REAL, milb_whip REAL, milb_k9 REAL,
    readiness_score REAL,
    fetched_at TEXT
);
```

### MLB Readiness Score (0-100)

```
readiness = (
    0.40 * fv_normalized          # FV 80 -> 100, FV 20 -> 0
  + 0.25 * age_level_performance  # wOBA vs level avg, age-adjusted
  + 0.20 * eta_proximity          # 2025->100, 2026->75, 2027->50, 2028->25
  + 0.15 * risk_factor            # Low->1.0, Med->0.8, High->0.6
)
```

**`fv_normalized`:** `(fv - 20) / 60 * 100`

**`age_level_performance`:** Compute wOBA proxy from MiLB stats: `(obp * 1.2 + slg * 0.8) / 2`. Compare to level average (AAA: .330, AA: .310, High-A: .300, A: .290). Bonus/penalty for age relative to level average age (AAA avg ~25, AA ~23, High-A ~22, A ~21). Scale to 0-100.

**`eta_proximity`:** Current year = 100, +1 = 75, +2 = 50, +3 = 25, +4 = 0.

**`risk_factor`:** Scales the raw score. Low risk = 1.0, Medium = 0.8, High = 0.6, Extreme = 0.4.

### Fallback Chain

1. FanGraphs Board API (richest)
2. pybaseball `top_prospects()` (MLB Pipeline scrape)
3. Existing hardcoded list (last resort)

Each level gracefully degrades: without FG data, no scouting tools or reports, but still has rankings + stats. Without pybaseball, just the static list.

### UI: Enhanced `pages/9_Leaders.py` Prospects Tab

- Sortable table: Rank, Name, Team, Pos, FV, ETA, Risk, Readiness Score, MiLB slash line
- Filter pills: position, organization, ETA year
- Expandable row: full scouting report, tool grades
- Plotly radar chart for selected prospect (6 tools: Hit/Power/Speed/Field/Arm/Command)
- "Fantasy Relevant" badge from FG's `Fantasy_Redraft` / `Fantasy_Dynasty` flags

### Test File: `tests/test_prospect_engine.py` (~15 tests)

- FG API response parsing (mock JSON)
- MiLB stats aggregation
- Readiness score computation (hand-calculated)
- Score bounds [0, 100]
- Fallback chain (FG fails -> pybaseball -> static)
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
| ESPN Fantasy API | `lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/2026/players?view=kona_player_info` | `defaultDraftRank` for all players | None | 24h |
| Yahoo ADP | yfpy `/draft_analysis` sub-resource on player objects | Average pick, average round, percent drafted | OAuth (existing) | 24h |
| CBS Sports | `cbssports.com/fantasy/baseball/rankings/h2h/top300/` | H2H Top 300 ranked table | Scrape | 24h |
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
```

Populated via fuzzy name matching (existing `match_player_id()` from `live_stats.py:27`) with team + position as tiebreakers. Can be seeded from Smart Fantasy Baseball cross-reference sheet.

### Consensus Algorithm: Trimmed Borda Count

```python
def compute_consensus(player_sources: dict[str, int | None]) -> dict:
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

### Test File: `tests/test_ecr_consensus.py` (~18 tests)

- ESPN API response parsing (mock JSON)
- Yahoo ADP extraction
- CBS HTML parsing
- Trimmed Borda count (hand-calculated: 3, 4, 5, 6, 7 source cases)
- Trim removes correct outliers
- rank_min/rank_max from real data
- rank_stddev computation
- Disagreement badge thresholds
- Player ID resolution across platforms
- Fallback with missing sources
- DB round-trip
- Consensus with single source = that source's rank
- Empty source handling

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
| RotoWire RSS | `rotowire.com/rss/news.htm?sport=mlb` | Headline blurbs (top stories) | None (feedparser) | 1h |

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
compute_ownership_trend(player_id, lookback_days=7) -> dict
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
    fetched_at TEXT
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
- `il_manager.py` → duration estimates (Weibull), lost SGP
- `injury_model.py` → health score, age risk
- `waiver_wire.py` → FA replacement candidates
- `data_bootstrap.py:PARK_FACTORS` → park context
- `prospect_engine.py` → MiLB stats for call-ups
- `ownership_trends` table → ownership direction

### Modifications to Existing Modules

| File | Change |
|------|--------|
| `src/live_stats.py` | Add `fetch_player_enhanced_status(mlb_id)` using `hydrate=rosterEntries,transactions` |
| `src/yahoo_api.py` | Extract `injury_note`, `status_full`, `percent_owned` fields from Player model during `sync_league_data()` |
| `src/data_bootstrap.py` | Add Phase 8: `_refresh_news()` calling `refresh_all_news()` with 1h staleness |
| `src/news_sentiment.py` | No changes needed — existing keyword scorer processes ESPN/RotoWire text |
| `src/news_fetcher.py` | Keep existing MLB transaction logic, add ESPN + RotoWire fetchers as supplementary sources |

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

`src/data_bootstrap.py` adds two new phases:

| Phase | Function | Staleness | Dependencies |
|-------|----------|-----------|-------------|
| Phase 8: Prospects | `_refresh_prospects()` → `prospect_engine.refresh_prospect_rankings()` | 7d | Phase 1 (players) |
| Phase 9: News | `_refresh_news()` → `player_news.refresh_all_news()` | 1h | Phase 1 (players), Phase 7 (Yahoo) |

ECR consensus refresh runs inside Phase 3 (projections) since it depends on projection data for HEATER SGP rank.

### Database Migration

`src/database.py` `init_db()` adds 5 new tables:
- `prospect_rankings`
- `ecr_consensus`
- `player_id_map`
- `player_news`
- `ownership_trends`

Total: 16 → 21 tables.

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
| `src/ecr.py` (rewrite) | Multi-platform consensus builder | ~18 |

### Modified Files Summary

| File | Changes |
|------|---------|
| `src/database.py` | 5 new tables in `init_db()` |
| `src/data_bootstrap.py` | Phase 8 (prospects) + Phase 9 (news) |
| `src/live_stats.py` | `fetch_player_enhanced_status()` |
| `src/yahoo_api.py` | Extract injury_note, status_full, percent_owned |
| `src/news_fetcher.py` | ESPN + RotoWire source integration |
| `pages/9_Leaders.py` | Enhanced Prospects tab with radar charts |
| `pages/1_My_Team.py` | News & Alerts tab |
| `app.py` | Consensus Rank column + disagreement badges in draft board |
| `requirements.txt` | Add `feedparser` |

### Total New Tests: ~53

Across 3 test files: `test_prospect_engine.py` (~15), `test_ecr_consensus.py` (~18), `test_player_news.py` (~20).
