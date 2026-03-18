# Gap Closure Design Spec — All 37 Remaining Items

**Goal:** Close every gap between the original Fantasy Baseball Draft Engine spec and the current implementation. 37 items across 6 subsystems.

**Architecture:** Additive changes only — no rewrites. Each subsystem extends existing modules with new data sources, schema columns, engine output fields, UI elements, or infrastructure. All changes maintain backward compatibility with the existing 1,108-test suite.

**Tech Stack:** Python 3.11+, Streamlit, SQLite, MLB-StatsAPI, pybaseball, requests, BeautifulSoup4 (new), schedule (new for cron)

---

## Subsystem 0: Config Clarifications (G1-G3)

### 0.1: Roster Rounds — 23 vs 28 (G1-G2)

**Status:** RESOLVED — no code change needed.

The spec says "28 total per team" and "336 total drafted." The actual Yahoo FourzynBurn league uses **23 roster slots** (C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23). The spec's "23 active + 5 bench = 28" double-counted the bench — BN slots ARE part of the 23. The code is correct at 23 rounds / 276 total picks.

**Action:** Document in CLAUDE.md that the spec's "28" was a miscount. No code change.

### 0.2: 14 vs 12 Scoring Categories (G3)

**Status:** RESOLVED — no code change needed.

The spec lists 14 categories including H/AB and IP. Per Yahoo's actual scoring UI (screenshot provided by user), H/AB and IP appear as **display-only paired stats** (asterisked), NOT separate scoring categories. The 12 scored categories (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) are correct.

**Action:** Document in CLAUDE.md. No code change.

---

## Subsystem A: Data Source Expansion (G4-G20, 17 gap numbers across 9 sections)

### A1: Player Pool Expansion (G4)

**Problem:** `fetch_all_mlb_players()` returns only ~750 active roster players. Spec requires 40-man rosters, spring training invitees, and NRIs.

**Design:**
- Add `fetch_extended_roster(season=2026)` to `src/live_stats.py`
- Use MLB Stats API `sports_players` endpoint with `gameType=S` (spring training) in addition to `gameType=R`
- Also query `team_roster` with `rosterType=40Man` for each team (30 teams × 1 call = 30 API calls)
- Deduplicate by `mlb_id`; merge results
- Expected yield: ~1,200-1,500 players (vs current ~750)
- Add `roster_type` column to players table: `"active"`, `"40man"`, `"spring_invite"`, `"nri"`
- Bootstrap phase 1 calls `fetch_extended_roster()` instead of `fetch_all_mlb_players()`
- Graceful fallback: if extended fetch fails, fall back to current active-only fetch

### A2: Additional Projection Systems (G5-G8)

**Problem:** Only 3 of 7 spec'd projection systems are fetched (Steamer, ZiPS, Depth Charts). Missing: ATC, THE BAT/BAT X, Marcel, PECOTA.

**Design:**
- **ATC:** Available on FanGraphs as `type=atc` — add to `SYSTEM_MAP` in `data_pipeline.py`
- **THE BAT / THE BAT X:** FanGraphs API `type=thebat` and `type=thebatx` — add to `SYSTEM_MAP`
- **Marcel:** Not directly available on FanGraphs API. Implement locally using Marcel formula: `3yr weighted avg (5/4/3) + regression to mean + age adjustment`. Add `src/marcel.py` (~80 lines).
- **PECOTA:** Behind Baseball Prospectus paywall. **Cannot implement** with free APIs. Document as "not available" in spec.
- Update `SYSTEM_MAP`:
  ```python
  SYSTEM_MAP = {
      "steamer": "steamer",
      "zips": "zips",
      "fangraphsdc": "depthcharts",
      "atc": "atc",          # NEW
      "thebat": "thebat",    # NEW
      "thebatx": "thebatx",  # NEW
  }
  ```
- Marcel computed locally after historical stats fetched (no API needed)
- Blending: `create_blended_projections()` already handles N systems — no changes needed
- Rate limit: add 0.5s delay between new FG endpoints (existing pattern)

### A3: Multi-Source ADP (G9-G12)

**Problem:** Only FanGraphs Steamer ADP. Missing: Yahoo direct, NFBC, FantasyPros ECR, ESPN.

**Design:**
- **Yahoo ADP (G9):** Extract from Yahoo Fantasy API when connected. Add `fetch_yahoo_adp()` to `src/yahoo_api.py` using `get_all_yahoo_fantasy_game_keys()` + league draft results. Populates `yahoo_adp` column.
- **FantasyPros ECR (G11):** Scrape FantasyPros consensus page (`https://www.fantasypros.com/mlb/rankings/overall.php`). Parse HTML table with BeautifulSoup4. Map player names to `player_id` via fuzzy match. Populates `fantasypros_adp` column.
- **NFBC ADP (G10):** NFBC publishes ADP data. Scrape from `https://nfc.shgn.com/adp/baseball`. Requires BeautifulSoup4. Populates new `nfbc_adp` column.
- **ESPN ADP (G12):** ESPN API is restricted. Use FantasyPros as proxy (they aggregate ESPN). Mark as "covered via FantasyPros consensus."
- Add `src/adp_sources.py` (~200 lines): `fetch_fantasypros_ecr()`, `fetch_nfbc_adp()`, `fetch_yahoo_adp()`
- Schema: add `nfbc_adp REAL` column to `adp` table via ALTER
- Composite ADP: `adp = coalesce(yahoo_adp, fantasypros_adp, nfbc_adp, steamer_adp)`
- Staleness: 24h for all ADP sources

### A4: Expert Rankings (G13)

**Problem:** Zero expert ranking integration.

**Design:**
- FantasyPros consensus rankings already covered by A3 (ECR = Expert Consensus Rankings)
- Additional expert sources (Rotowire, The Athletic, PitcherList) are behind paywalls
- **Decision:** FantasyPros ECR IS the expert consensus. It aggregates 100+ experts including Rotowire, Rotoworld, etc. No need for individual expert scraping.
- Mark G13 as **RESOLVED via A3 (FantasyPros ECR)**.

### A5: Additional Injury Sources (G14-G15)

**Problem:** Only MLB Stats API injury data. Spec wants Rotowire + ESPN feeds.

**Design:**
- **Rotowire:** Behind paywall. Cannot scrape for free.
- **ESPN:** API restricted; no public endpoint for injury data.
- **Alternative:** MLB Stats API already provides IL status + transaction history. Supplement with pybaseball `playerid_reverse_lookup()` for cross-referencing.
- **Enhancement:** Add `fetch_current_injuries()` to `src/live_stats.py` using MLB Stats API `injuries` endpoint (free, returns current IL placements). Currently only derived from GP/GA historical.
- Mark G14-G15 as **PARTIALLY RESOLVED** — MLB Stats API injury endpoint + historical GP/GA covers the functional need. Paid feeds would add timeliness but not accuracy.

### A6: Statcast Stuff+/Location+/Pitching+ (G16)

**Problem:** pybaseball doesn't expose these metrics.

**Design:**
- Stuff+, Location+, Pitching+ are proprietary to Baseball Savant's Pitch Leaderboard
- **Approach:** Scrape from `https://baseballsavant.mlb.com/leaderboard/pitch-arsenals` using requests + BeautifulSoup4
- Add `fetch_stuff_plus()` to `src/engine/signals/statcast.py`
- Returns: `stuff_plus`, `location_plus`, `pitching_plus` per pitcher
- Store in `projections` table via new columns (ALTER)
- Graceful fallback: if scrape fails, these fields are NULL (no impact on existing pipeline)
- Integration: feed into `_apply_statcast_delta()` in draft engine Stage 4

### A7: Roster Resource Depth Charts (G17)

**Problem:** Only FanGraphs depth charts.

**Design:**
- Roster Resource (`https://www.rosterresource.com/mlb-depth-charts/`) provides lineup order + bullpen hierarchy
- Scrape team pages (30 teams) with BeautifulSoup4
- Extract: batting order (1-9), rotation order (1-5), closer/setup hierarchy
- Add `src/depth_charts.py` (~150 lines): `fetch_roster_resource_depth()`
- Store results in new `depth_charts` table: `player_id, team, lineup_slot, rotation_slot, bullpen_role`
- Integration: feed `lineup_slot` into `compute_lineup_protection()` (currently hardcoded PA estimates)
- Staleness: 7 days

### A8: Contract/Salary Data (G18-G19)

**Problem:** Hardcoded `contract_year_boost()` only. No live data.

**Design:**
- Spotrac has no public API. Baseball Reference contracts page is scrapeable.
- **Approach:** Scrape BB-Ref contract page: `https://www.baseball-reference.com/players/{letter}/{player_id}.shtml`
- Too many individual pages. Better: scrape the free agent class list: `https://www.baseball-reference.com/leagues/majors/2027-free-agents.shtml` (2027 = players in contract year 2026)
- Add `src/contract_data.py` (~100 lines): `fetch_contract_year_players()`
- Returns: list of player names in their contract year → set `contract_year=True` in player pool
- Also flag arbitration-eligible players from BB-Ref
- Schema: add `arbitration_eligible INTEGER DEFAULT 0` to players table
- Staleness: 30 days (contracts don't change mid-season)

### A9: News API Integration (G20)

**Problem:** `news_sentiment.py` is built and tested but not wired to any data source.

**Design:**
- **Free option:** MLB Stats API `news` endpoint (if available) or RSS feeds
- **Practical option:** Scrape Rotowire player news blurbs from their free tier (limited to ~50 recent items)
- **Best free option:** Use `pybaseball` + Baseball Savant transaction log + MLB Stats API roster moves
- Add `src/news_fetcher.py` (~120 lines): `fetch_recent_news()` returns `dict[int, list[str]]` mapping `player_id → news items`
- Wire into `DraftRecommendationEngine._apply_news_sentiment()` as new Stage 8.5 (after ML ensemble)
- Uses existing `batch_sentiment()` from `news_sentiment.py`
- Staleness: 6 hours
- Graceful fallback: if no news fetched, sentiment scores = 0.0 (neutral)

---

## Subsystem B: Schema + ID Cross-Referencing (4 gaps)

### B1: FanGraphs + Yahoo Player IDs (G21-G22)

**Problem:** Only `mlb_id` stored. No `fangraphs_id` or `yahoo_id`.

**Design:**
- **FanGraphs ID:** Already present in FG JSON responses as `playerid`. Extract during `fetch_projections()` and store in new `fangraphs_id` column.
- **Yahoo ID:** Available from Yahoo Fantasy API player objects. Extract during Yahoo sync and store in new `yahoo_id` column.
- Schema: `ALTER TABLE players ADD COLUMN fangraphs_id TEXT; ALTER TABLE players ADD COLUMN yahoo_id TEXT;`
- Matching: Use existing name+team fuzzy match as fallback when IDs unavailable
- Populate retroactively on next bootstrap run

### B2: Missing Schema Fields (G23-G25)

**Problem:** No `role_status`, `contract_details`, `spring_training_stats` in schema.

**Design:**
- `role_status TEXT` — populated from depth chart data (Subsystem A7): "starter", "platoon", "closer", "setup", "committee", "bench"
- `contract_details TEXT` — populated from BB-Ref scrape (Subsystem A8): JSON blob with years/dollars
- `spring_training_stats TEXT` — JSON blob from MLB Stats API `gameType=S` stats. Fetch during March only.
- All three columns added via ALTER TABLE on next bootstrap
- Populated lazily: NULL until data source runs

---

## Subsystem C: Engine Output Enrichment (G27-G32, 6 gaps)

All enrichment happens in `src/draft_engine.py` (DraftRecommendationEngine.recommend()), NOT in `src/simulation.py`. The base simulator remains unchanged; enrichment is post-processing on the recommendation output.

### C1: 0-100 Composite Value Score (G27)

**Design:**
- In `DraftRecommendationEngine.recommend()`, after `evaluate_candidates()` returns, normalize `combined_score` to 0-100 scale
- Formula: `composite_100 = (score - min_score) / (max_score - min_score) * 100`
- Add `composite_value` column to recommendation output DataFrame
- Clamp to [0, 100]; handle edge case where all scores are equal (all get 50)

### C2: Position Rank (G28)

**Design:**
- In `DraftRecommendationEngine.recommend()`, after scoring, compute rank within each eligible position
- For multi-position players: rank in each position (e.g., "SS: 3, 2B: 5")
- Add `position_rank` column (string, e.g., "SS:3/2B:5")
- Implementation: explode positions, groupby position, rank by combined_score descending

### C3: Overall Rank Column (G29)

**Design:**
- In `DraftRecommendationEngine.recommend()`, add explicit `overall_rank` integer column: `candidates["overall_rank"] = range(1, len(candidates)+1)`
- Trivial — already sorted by combined_score, rank = row number

### C4: Category Impact Breakdown (G30)

**Design:**
- **NOTE:** `per_category_sgp` does NOT currently exist in `evaluate_candidates()` output. Must be built.
- In `DraftRecommendationEngine.recommend()`, compute per-category SGP contribution for each candidate using `SGPCalculator.player_sgp()` for each of 12 categories
- Add 12 columns: `impact_R`, `impact_HR`, `impact_RBI`, `impact_SB`, `impact_AVG`, `impact_OBP`, `impact_W`, `impact_L`, `impact_SV`, `impact_K`, `impact_ERA`, `impact_WHIP`
- Each = player's raw SGP in that category (from `SGPCalculator`)
- This is a post-processing step: iterate candidates, call `sgp_calc.player_sgp(player, cat)` for each cat
- Display in UI as expandable breakdown under hero card

### C5: Confidence Level in Engine (G31)

**Design:**
- Move confidence computation from UI (app.py) into engine output (`DraftRecommendationEngine.recommend()`)
- Formula: `cv = mc_std / max(mc_mean, 0.01)`
- Classification: `cv < 0.15` → "HIGH", `cv < 0.35` → "MEDIUM", else "LOW"
- Add `confidence_level` column to output DataFrame
- **Non-MC fallback:** When MC is disabled (Quick mode, 0 sims), use projection spread across systems as proxy: `spread = std(steamer, zips, depthcharts) / mean(...)`. Same thresholds apply. If only one system available, confidence = "LOW".

### C6: Spring Training Signal (G32)

**Design:**
- Wire `news_sentiment.py` spring training keywords into engine
- **Pipeline placement:** NOT a separate stage. Integrated into A9's news sentiment flow. When `news_fetcher.py` returns items, `batch_sentiment()` already detects spring training keywords ("impressive", "ahead of schedule", "breakout"). The sentiment score feeds into Stage 8.5 (Subsystem A9 design). Spring training items are NOT separated from general news — they are simply news items that happen to contain ST-related keywords, and the existing scoring handles them.
- **No separate stage needed.** The existing `sentiment_adjustment()` function converts score to a multiplicative factor (±5%). This is sufficient for the small, research-supported spring training signal.

**Test file:** `tests/test_engine_output.py` (~60 lines) covers C1-C5 output fields with edge cases (empty pool, single candidate, all-equal scores, Quick mode confidence fallback).

---

## Subsystem D: UI Polish (2 gaps)

### D1: "LAST CHANCE" Badge (G33)

**Design:**
- In `render_hero_pick()`, after survival gauge, add explicit text badge when `p_survive < 0.20`:
  ```python
  if surv < 0.20:
      last_chance_html = f'<span class="badge badge-last-chance">LAST CHANCE</span>'
  ```
- CSS: red background, white bold text, pulsing animation
- Also add to alternatives cards when their survival < 20%

### D2: Top 10 Instead of Top 8 (G34)

**Design:**
- Change `top_n=8` to `top_n=10` in both `app.py` and `pages/2_Draft_Simulator.py`
- Display top 1 as hero card, next 9 as alternatives grid (3 columns × 3 rows instead of 5 columns × 1 row)
- Alternative layout: keep hero + 5 visible, add "Show More" expander for remaining 4

---

## Subsystem E: Infrastructure (3 gaps)

### E1: Automated Update Pipeline (G26)

**Design:**
- Add `src/scheduler.py` (~80 lines) using Python `schedule` library
- Define jobs:
  - Every 1h: `refresh_live_stats()` (staleness already handles skip)
  - Every 6h: `refresh_yahoo_sync()` (when connected)
  - Every 24h: `refresh_adp_sources()`
  - Every 7d: `refresh_projections()`, `refresh_depth_charts()`
- **Streamlit integration:** Run scheduler in a background thread on app startup
- **Alternative:** Add GitHub Actions scheduled workflow (`.github/workflows/refresh.yml`) that runs `python -c "from src.data_bootstrap import bootstrap_all_data; bootstrap_all_data(force=True)"` on cron
- Both approaches: graceful — if app is closed, next startup catches up via staleness

### E2: Backtesting Framework (G35)

**Design:**
- Add `tests/backtest/` directory with `backtest_2025.py`
- Load 2025 preseason projections (Steamer/ZiPS from FG historical)
- Load 2025 actual season stats (from historical fetch)
- Run `DraftRecommendationEngine.recommend()` with 2025 data
- Compare: recommended draft order vs actual season value (SGP)
- Metrics: rank correlation (Spearman ρ), top-50 hit rate, positional accuracy
- Output: JSON report with accuracy metrics
- **Limitation:** Requires 2025 projection data snapshot (may need manual CSV)

### E3: Timeline + Done Criteria (G36-G37)

**Design:**
- Add `docs/ROADMAP.md` with:
  - Phase 0-4 completion dates (historical)
  - Phase 5 (gap closure) target: March 25, 2026
  - Acceptance criteria per subsystem:
    - A: ≥1,000 players in pool, ≥5 projection systems, ≥2 ADP sources
    - B: All 3 ID types populated for ≥80% of players
    - C: All 8 draft board output fields present
    - D: LAST CHANCE badge visible, top 10 display
    - E: Scheduler running, backtest report generated
  - Test count target: ≥1,200 tests (currently 1,113)

---

## Items Documented as Infeasible (Free API Constraint)

| Gap | Item | Reason |
|-----|------|--------|
| G8 | PECOTA projections | Baseball Prospectus paywall |
| G12 | ESPN ADP | API restricted (covered via FantasyPros) |
| G14 | Rotowire injury feed | Paywall |
| G15 | ESPN injury status | API restricted |
| G13 | Individual expert rankings | Paywall (covered via FantasyPros ECR consensus) |

These 5 items are **resolved by proxy** (FantasyPros aggregates ESPN + Rotowire + 100 other experts) or **infeasible** within the free-API constraint.

**Gap arithmetic:** 37 total gaps (G1-G37). G1-G3 resolved as config clarifications (no code change). G8, G12, G13, G14, G15 resolved as infeasible/proxy. Remaining actionable: **37 - 3 (config) - 5 (infeasible) = 29 actionable gaps** requiring code changes.

**Scraping resilience:** All BeautifulSoup4-based scrapers (A3/FantasyPros, A3/NFBC, A6/Savant Stuff+, A7/Roster Resource, A8/BB-Ref) must implement:
1. `try/except` around all HTTP requests — return empty DataFrame on failure
2. User-Agent header to avoid blocks
3. Staleness-based caching (don't re-scrape if data is fresh)
4. Graceful degradation: if scrape fails, existing data (or defaults) remain in use
5. **No hard dependencies**: every scraping-based feature is additive. Zero scrapers working = app runs identically to current state.

---

## New Files Created

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `src/adp_sources.py` | FantasyPros ECR, NFBC ADP, Yahoo ADP fetchers | ~200 |
| `src/marcel.py` | Marcel projection system (local computation) | ~80 |
| `src/depth_charts.py` | Roster Resource scraper | ~150 |
| `src/contract_data.py` | BB-Ref contract year + arb scraper | ~100 |
| `src/news_fetcher.py` | News aggregation from free sources | ~120 |
| `src/scheduler.py` | Background refresh scheduler | ~80 |
| `tests/test_adp_sources.py` | ADP source tests | ~100 |
| `tests/test_marcel.py` | Marcel projection tests | ~60 |
| `tests/test_depth_charts.py` | Depth chart scraper tests | ~80 |
| `tests/test_contract_data.py` | Contract data tests | ~50 |
| `tests/test_news_fetcher.py` | News fetcher tests | ~60 |
| `tests/test_scheduler.py` | Scheduler tests | ~40 |
| `tests/test_engine_output.py` | Engine output enrichment tests (C1-C5) | ~60 |
| `tests/backtest/backtest_2025.py` | Backtesting framework | ~150 |
| `docs/ROADMAP.md` | Timeline + done criteria | ~50 |

## Modified Files

| File | Changes |
|------|---------|
| `src/live_stats.py` | Add `fetch_extended_roster()`, `fetch_current_injuries()` |
| `src/data_pipeline.py` | Add ATC, THE BAT, THE BAT X to `SYSTEM_MAP` |
| `src/data_bootstrap.py` | Wire new data sources into bootstrap phases |
| `src/database.py` | ALTER TABLE for new columns |
| `src/simulation.py` | Add `category_weights` passthrough to MC output (no structural change) |
| `src/draft_engine.py` | Add `composite_value`, `position_rank`, `overall_rank`, `impact_{cat}`, `confidence_level`, news sentiment stage |
| `src/yahoo_api.py` | Add `fetch_yahoo_adp()` |
| `src/engine/signals/statcast.py` | Add `fetch_stuff_plus()` |
| `src/contextual_factors.py` | Wire depth chart lineup slot data |
| `src/ui_shared.py` | Add LAST CHANCE badge CSS |
| `app.py` | LAST CHANCE badge, top 10, category impact display |
| `pages/2_Draft_Simulator.py` | Same UI updates |
| `.github/workflows/refresh.yml` | Scheduled data refresh workflow |
| `requirements.txt` | Add beautifulsoup4, schedule |
| `CLAUDE.md` | Document all new modules + APIs |
