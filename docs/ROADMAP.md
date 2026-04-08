# HEATER Roadmap

## Completed Phases

| Phase | Description | Date | Key Deliverables |
|-------|-------------|------|-----------------|
| 0 | League Config | Dec 2025 | 12-team H2H, 12 categories, 23-slot roster |
| 1 | Data Pipeline | Jan 2026 | MLB Stats API + FanGraphs auto-fetch, bootstrap |
| 2 | Draft Engine | Mar 2026 | 5 modules, 270 tests, 25-feature pipeline |
| 3 | Live Draft UI | Mar 2026 | Hero card, alternatives, practice mode, opponent intel |
| 4 | Planning + Testing | Mar 2026 | CI/CD, 1,108 tests, math verification suite |
| 5 | Gap Closure | Mar 2026 | 14 new modules, extended roster, 7 projection systems |

## Acceptance Criteria (Phase 5)

- [x] >=1,000 players in pool (extended roster: 40-man + spring training)
- [x] >=5 projection systems blended (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel)
- [x] >=2 ADP sources (FG Steamer + FantasyPros ECR + NFBC)
- [x] All 8 draft board output fields present (composite value, position rank, overall rank, category impact, risk score, urgency, BUY/FAIR/AVOID, confidence level)
- [x] LAST CHANCE badge when survival < 20%
- [x] Top 10 recommendations (was 8)
- [x] Background scheduler running
- [x] >=1,200 tests passing (target: 1,254 actual)

## Infeasible Items (Free API Constraint)

| Item | Reason | Proxy |
|------|--------|-------|
| PECOTA | Baseball Prospectus paywall | Marcel local computation |
| ESPN ADP | API restricted | FantasyPros ECR (aggregates ESPN) |
| Rotowire injury feed | Paywall | MLB Stats API injuries |
| ESPN injury status | API restricted | MLB Stats API injuries |
| Individual expert rankings | Paywalls | FantasyPros ECR consensus (100+ experts) |

## Future Directions

- Real-time Yahoo draft sync (live draft board)
- Waiver wire alerts + weekly auto-optimization
- Standings projections + playoff odds
- Mobile-responsive layout for draft day
- Cloud deployment for remote access

---

## Accuracy & Calibration Backlog

Identified April 7, 2026 via full algorithm audit. All 53 engines reviewed.
Priority: HIGH items affect every downstream calculation. MEDIUM items affect specific features. LOW items are polish.

### HIGH — Cascade Through All Engines

| # | Item | Current State | Fix | Impact |
|---|------|--------------|-----|--------|
| A1 | **Dynamic SGP Denominators** | Hardcoded (R:32, HR:13, SB:14, etc.) — never recomputed | Compute from `league_standings`: `denom = (max - min) / (n_teams - 1)` at bootstrap, update weekly | Propagates to EVERY engine that uses SGP — trade values, draft picks, lineup optimizer, waiver recs, power rankings. Single highest-leverage fix. |
| A2 | **Weighted Projection Ensemble** | Simple average of Steamer/ZiPS/DepthCharts (equal 1/n weight) | Weight by inverse RMSE from prior season. Even crude 40/35/25 beats naive average. FanGraphs publishes annual accuracy comparisons. | More accurate player projections = better everything downstream. |
| A3 | **Bayesian SGP Updating** | Static denominators all season | Start with preseason denoms as prior, update weekly from actual standings. By week 8: 70% actual / 30% prior. | Denominators drift as season progresses — teams that tank, injuries, trade deadline. Static denoms become stale. |
| A4 | **Remove Self-Referential ECR Input** | HEATER SGP rank is source #7 in ECR consensus | Remove source #7 (HEATER SGP). Use only 6 external sources (ESPN, Yahoo, CBS, NFBC, FanGraphs, FantasyPros). | Circular input: model's output feeds back into its own valuation. Removes confirmation bias. |

### MEDIUM — Specific Engine Improvements

| # | Item | Current State | Fix | Impact |
|---|------|--------------|-----|--------|
| B1 | **League-Specific Acceptance Calibration** | Generic behavioral economics (loss aversion 1.8 from Kahneman/Tversky) | Compute empirical acceptance rates from Yahoo `transactions` data — what SGP/ADP gaps led to accepted vs rejected trades in this league | Trade Finder recommendations become league-calibrated, not generic. Even 10-20 past trades would help. |
| B2 | **Position-Specific Health Scoring** | Same age threshold (30 hitters, 28 pitchers) regardless of position | Catchers: threshold 28 with 0.03/yr. DH: threshold 34 with 0.01/yr. OF: threshold 31. | Catchers age faster than DHs. Current model treats them identically. |
| B3 | **Injury-Type Adjustment** | Linear IL days / 162 penalty — all injuries equal | Weight by injury type: TJ surgery = 0.4 health floor for 2 years. Hamstring = 0.7 for 1 year. Concussion = 0.6. | 30 IL days from a bone bruise (heals fully) is NOT the same as 30 IL days from TJ. |
| B4 | **Temporal ECR Weighting** | All ECR sources weighted equally regardless of when fetched | Weight recent rankings higher: exponential decay with 14-day half-life. March rank = 0.25x weight vs April rank = 1.0x. | Stale preseason ranks shouldn't carry same weight as in-season updates. |
| B5 | **Dynamic Park Factors** | Static FanGraphs 2024 values (e.g., Coors always 1.38) | Fetch updated park factors from FanGraphs/pybaseball mid-season. Some parks change significantly year-to-year (new dimensions, humidor). | Matchup Planner and Lineup Optimizer use stale park data all season. |
| B6 | **Category Urgency k-Calibration** | Sigmoid k=2.0 counting, k=3.0 rate — no sensitivity analysis | Run backtests: simulate weekly decisions with k from 1.0 to 5.0, measure end-of-season category wins. Pick k that maximizes weekly W-L. | Current k values control how aggressively you chase losing categories. Could be significantly off. |
| B7 | **Power Rankings Momentum Weight** | Momentum = 10% of power rating | In 12-team H2H, hot streaks dominate weekly outcomes. Consider momentum 20%, reduce roster quality to 30%. | Power rankings may undervalue teams on 4-week winning streaks. |
| B8 | **Validate Trade Finder Composite Weights** | 15/15/15/30/10/15 — acceptance-heavy, not backtested | Simulate: run trade finder on week 1-10 data, check if top-ranked trades would have improved your team by week 24. Tune weights to maximize post-trade improvement. | Currently optimizing for "trades that get accepted" rather than "trades that make you better." |

### LOW — Polish & Edge Cases

| # | Item | Current State | Fix | Impact |
|---|------|--------------|-----|--------|
| C1 | **Waiver Wire Drop Penalties** | Hardcoded: DH -3.0, low SB -1.5, low AVG -1.0 | Derive from league context: DH penalty should scale with how many Util slots you have. AVG threshold should be league-average AVG, not 0.245. | Mostly affects edge-case drop decisions. |
| C2 | **Dual Objective Alpha Thresholds** | Time-based: 0.3/0.5/0.7/0.85 with ±0.1 adjustments | Validate against playoff probability: if you're 95% to make playoffs, alpha should be high (H2H focus). If 20%, pivot to spoiler/roto mode. | Affects lineup optimization strategy, not individual decisions. |
| C3 | **Start/Sit Home Advantage** | Hardcoded 1.02 (2% boost) | MLB home advantage has declined to ~1.5% in recent years. Update to 1.015 and cite source. | Very small per-game effect. |
| C4 | **Trade Readiness Normalization** | `(sgp + 5) * 5` assumes SGP range [-5, +15] | Compute actual SGP range from pool, normalize to [0, 100] dynamically. | Affects Trade Readiness tab scores — cosmetic more than functional. |
| C5 | **Matchup Planner Inverse Park Formula** | `2.0 - park_factor` for pitchers (linear) | Use `1.0 / park_factor` (reciprocal) — theoretically correct since pitcher benefit is inverse of hitter benefit. | Tiny difference for most parks. Only Coors (1.38 → 0.62 vs 0.72) is meaningfully different. |

---

## Machine Learning & Advanced Analytics Backlog

Identified April 7, 2026 via deep research across academic papers, FanGraphs, and industry analysis.
Only items with genuine data support and measurable accuracy improvement are included.

### ML — High Impact (Data Exists, Signal Is Real)

| # | Item | Current State | What to Build | Data Source | Accuracy Gain | Effort |
|---|------|--------------|---------------|-------------|---------------|--------|
| D1 | **Statcast Regression Model** | XGBoost stub in `src/ml_ensemble.py` — no pre-trained model, falls back to zeros, 0.1 weight | Train XGBoost on exit_velocity, barrel_rate, hard_hit_pct, xwOBA, xBA, xSLG, xERA, sprint_speed. Target: (actual_value_next_30d - projected). Retrain daily in bootstrap. | pybaseball `statcast_batter/pitcher_expected_stats()` (2023-2025 leaderboards, ~2100 player-seasons) | ~10-15% on player projections. Identifies regression candidates (high xwOBA but low wOBA = buy, reverse = sell). | Medium — scaffold exists, need training pipeline + daily retrain |
| D2 | **Playing Time Prediction Model** | No explicit model. Static preseason PA/IP from Steamer/ZiPS, never updated. | Ridge/Lasso regression: `remaining_PA = f(recent_PA_rate, depth_chart_slot, health_score, age, platoon_pct)`. Update weekly. Apply as multiplier to all counting stats. | Depth charts (bootstrap), game logs (L7/L14/L30 from game_day.py), IL status (ESPN feed), age (players table) | ~10-15% on counting stats. This is the #1 source of projection error per FantasyPros 2025 accuracy study — OOPSY led accuracy mainly via playing time. | Medium |
| D3 | **Weighted Projection Stacking** | A2 in calibration backlog says "inverse RMSE weighting" | Go further: train a **ridge regression stacking model** where each system's projection is a feature, 2025 actuals are the target. Learned weights > hand-tuned weights. 50 lines of sklearn. | `projections` table (7 systems × ~700 players), `season_stats` 2025 actuals | ~5-8% over naive averaging. Research: composite systems consistently outperform individual ones (FanGraphs 2023 game-theory comparison). | Low — tiny model, fast to train |
| D4 | **Opponent Behavior Learning** | Generic `LOSS_AVERSION=1.8` from behavioral economics, applied uniformly to all 11 opponents | Per-opponent **logistic regression**: features = SGP_delta, ADP_gap, standings_rank, days_since_draft, trade_count_this_season. Target = accepted (1) or rejected (0). Fallback to current sigmoid when <3 data points per opponent. | Yahoo `transactions` table (all completed + rejected trades). Even 20-30 trades over 2-3 seasons gives enough signal. | ~10-20% on acceptance prediction. Replaces lab-derived loss aversion with league-derived behavior. | Medium |

### ML — Medium Impact (Pre-Trained Models, Low Effort)

| # | Item | Current State | What to Build | Data Source | Accuracy Gain | Effort |
|---|------|--------------|---------------|-------------|---------------|--------|
| D5 | **NLP News Sentiment** | `src/news_sentiment.py` — 40 hardcoded keywords, no NLP. "TJ surgery consultation" scores same as "fractured finger." | Replace with pre-trained `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace. 3 lines of code, <100ms per headline, no training. Classify positive/negative/neutral with confidence. | MLB Stats API + Yahoo news headlines (already fetched in `src/player_news.py`) | ~3-5% on trade/waiver timing. Research: transformer sentiment improved Fantasy Premier League prediction by 8-12% (ResearchGate 2024). | Low — pre-trained, no training needed |
| D6 | **Backtesting & Validation Framework** | No backtesting exists. Weights/thresholds are domain guesses with no validation. | Build `src/backtesting_framework.py`: replay past weeks, measure whether engine recommendations (start/sit, waiver adds, trade proposals) would have improved category wins. Score each engine against actual outcomes. | `season_stats` (2025 actuals), `league_standings` (historical), `transactions` (past trades/adds) | Meta-improvement: enables validating ALL other changes. Without this, every weight is a guess. | Medium-High — framework, not a model |

### ML — Future / Experimental

| # | Item | Current State | What to Build | Data Source | Accuracy Gain | Effort |
|---|------|--------------|---------------|-------------|---------------|--------|
| D7 | **Category-Aware Lineup RL** | LP optimizer with H2H weights — optimal for single week but doesn't learn from outcomes | Contextual bandit: state=(category_gaps, opp_strength, weather, day), action=(start A vs B), reward=(won category?). After 8+ weeks, learns patterns LP can't capture. | Weekly lineup decisions + category outcomes (8+ weeks of tracked data needed) | Unknown — promising but needs data accumulation. | High — requires 8+ weeks of tracked decisions before useful |

### Implementation Priority (Combined)

**Phase A — Calibration (no ML, highest leverage):**
1. A1: Dynamic SGP denominators
2. A4: Remove self-referential ECR
3. A2+D3: Weighted projection stacking (upgrade A2 to ML stacking)
4. A3: Bayesian SGP updating

**Phase B — ML Models (train on existing data):**
5. D1: Statcast regression model (XGBoost)
6. D2: Playing time prediction model
7. D5: NLP news sentiment (pre-trained, no training)
8. D4: Opponent behavior learning (logistic regression)

**Phase C — Validation & Learning:**
9. D6: Backtesting framework
10. B6: Category urgency k-calibration (using backtesting framework)
11. B8: Trade finder weight validation (using backtesting framework)

**Phase D — Experimental:**
12. D7: Category-aware lineup RL (after 8+ weeks of data)

**Remaining calibration items (B1-B5, B7, C1-C5):** As time allows, in numbered order.

---

## Data Signal & Intelligence Backlog

Identified April 7, 2026 via deep web research across FanGraphs, Baseball Savant, academic papers,
and industry analysis. These are NEW data signals and intelligence layers that HEATER does not
currently use at all, backed by published research showing quantified accuracy improvements.

### E — New Data Signals (Not Currently in HEATER)

| # | Item | What It Is | Research Basis | Where It Helps | Data Source | Effort |
|---|------|-----------|---------------|---------------|-------------|--------|
| E1 | **BABIP Regression Targets** | Flag hitters whose BABIP is far from expected (based on line-drive rate, sprint speed, fly-ball rate). BABIP stabilizes at ~800 BIP but individual baselines vary by 30+ points. | FanGraphs Sabermetrics Library: hitters with high LD% and sprint speed sustain .320+ BABIP; fly-ball hitters sustain .270-.280. xBABIP models (contact quality + speed) predict future BABIP within 15 points. | Waiver Wire (buy low on unlucky hitters), Trade Finder (sell high on lucky hitters), Start/Sit (trust process over results for low-BABIP contact hitters) | pybaseball `batting_stats()` — LD%, FB%, sprint_speed from Statcast, BIP count from season_stats | Medium |
| E2 | **Pitcher Stuff+ / Location+ / Pitching+** | FanGraphs' pitch-quality models that rate each pitch type on a 80-120 scale (100 = average). Stuff+ measures raw pitch movement/velocity, Location+ measures command, Pitching+ combines both. | FanGraphs research (2022-2025): Stuff+ correlates with future K% at r=0.72, stronger than current-year K% itself. Pitching+ is the single best predictor of pitcher future ERA. Already stored in `season_stats` schema (`stuff_plus`, `location_plus`, `pitching_plus` columns exist but are never populated). | Pitcher projections, Start/Sit (pitcher matchup quality), Streaming recommendations, Trade valuation for pitchers | pybaseball `pitching_stats(qual=0)` with `fangraphs` source — includes stuff_plus, location_plus, pitching_plus columns | Low — DB columns already exist, just need to populate in bootstrap |
| E3 | **Umpire Strike Zone Adjustment** | Home plate umpires vary by ±0.3-0.5 runs per game. "Pitcher's umps" expand the zone (more K, fewer BB, fewer runs); "hitter's umps" squeeze it (more walks, more runs). | Sports Info Solutions 2024: Extra Strikes Per 150 Called Pitches metric. FantasyGuru publishes daily umpire reports with quantified K% and run environment deltas. SABR Umpire Analytics. | Start/Sit (pitcher vs hitter's ump = +5% K value), Matchup Planner (umpire-adjusted run environment), Daily Lineup Optimizer (DCV adjustment per game) | Free: retrosheet.org umpire assignments (historical). Paid: UmpScoreCards.com API or FantasyGuru umpire reports. Alternatively, scrape from Baseball Savant game feeds (umpire listed per game). | Medium — need umpire assignment data source + per-umpire tendency model |
| E4 | **Pitcher Fatigue & Workload Model** | Track pitch count accumulation, days of rest, velocity trends, and innings workload to predict late-season decline. Pitchers on >100 IP by ASB with rising pitch counts show measurable K% decline and ERA increase. | PMC Research (2020): reduced time between appearances predicts UCL injury. FanGraphs Fatigue Units model (2023): workload-adjusted projections outperform static projections by 3-5% in second half. Velocity decline of 0.5+ mph correlates with 15%+ ERA increase within 3 starts. | Pitcher projections (second-half discount), Start/Sit (fade high-workload pitchers in August/September), Trade timing (sell pitchers approaching workload thresholds), IL prediction | MLB Stats API game logs (pitch counts per start), pybaseball Statcast (velocity per start), season_stats (cumulative IP), game_day.py (existing L7/L14/L30 framework) | Medium — data exists, need workload tracking + fatigue multiplier |
| E5 | **Stolen Base Prediction Model** | Current SB projections are static preseason numbers. Modern analytics use sprint speed + baserunning opportunity rate + catcher pop time + pitcher slide step tendency to predict SB attempts and success rate. | Baseball Savant: sprint speed >29 ft/s = 80.3% SB success rate. Logistic regression on sprint speed + catcher pop time + pitcher handedness predicts SB outcomes with ~75% accuracy. FanGraphs: "How Sprint Speed Relates to Stolen Bases" — sprint speed explains 60% of SB variance. | SB projection accuracy (SB is the most volatile counting stat), Waiver Wire (identify speed upside not reflected in projections), Lineup Optimizer (game-level SB opportunity based on opposing catcher) | Statcast sprint speed leaderboard (pybaseball), Baseball Savant catcher pop times, pitcher pickoff/slide step data. Sprint speed already partially used in injury_model but NOT in SB projections. | Medium |
| E6 | **Enhanced Weather Model** | Current weather adjustment is HR-only (Alan Nathan's +0.9%/degree above 72F). Research shows temperature also affects ball carry for all fly balls, wind direction/speed is the strongest weather variable, and humidity is negligible (lighter than dry air). | Home Run Forecast physics model: +0.33 feet per 1°F. Climate Central 2025: 500+ extra HRs since 2010 from warming alone. Research: wind blowing out >10mph increases HR rate by ~15-20%. K rate increases 7.5% in clear vs cloudy conditions. Humidity effect is near-zero (opposite of intuition). | Matchup Planner (wind-adjusted game ratings), DCV optimizer (wind + temp for all hitters, not just HR), Start/Sit (fade fly-ball pitchers in high-wind-out conditions), Streaming picks | Open-Meteo (already in bootstrap) — need to add wind direction relative to outfield orientation per stadium. `STADIUM_COORDS` already has lat/lon for 30 parks. Need outfield orientation angles (static, one-time data entry). | Medium — weather data exists, need wind direction model relative to park orientation |
| E7 | **Pitcher-Batter Matchup History** | Direct pitcher-vs-batter historical stats (e.g., Trout is 8-for-15 lifetime vs Verlander). Small sample but stabilizes after ~60 PA. When combined with handedness and pitch-type splits, adds 2-4% prediction value over generic platoon adjustments. | FanGraphs Splits Library: PvB data has predictive value because "pitch repertoire creates systematic vulnerabilities" — a batter who struggles against high-spin curveballs does so because of a documented mechanical tendency, not random variation. Stabilizes at ~60 PA per matchup. BaseballHQ 2025: handedness splits are the most stable of all matchup drivers. | Start/Sit (specific PvB history for tiebreakers), Matchup Planner (player-level matchup quality vs specific opposing pitcher), Daily Optimizer (game-level PvB adjustment) | pybaseball `statcast_batter(pitcher_id)` or Baseball Savant PvB lookup. MLB Stats API also provides batter-vs-pitcher splits. | High — large data volume, need smart caching + minimum PA threshold |
| E8 | **Punt Strategy Optimizer** | Current punt detection (category_gap_analysis) identifies WHAT to punt but doesn't optimize WHICH categories to punt or how many. In H2H, the optimal punt count depends on league structure and opponent distribution. | FantasyPros 2026 punt strategy research: "Punting one category gives an advantage in 11 others." FanGraphs: in 12-team H2H with 12 categories, optimally punting 1-2 categories and redistributing resources wins 7-5 or 8-4 more often than winning 6.5-5.5 across all 12. The key is punting categories that are CORRELATED (e.g., punt SV + L together since they're closer-related). | Trade strategy (trade away assets in punted categories for strength in contested ones), Draft strategy (skip closers entirely if punting SV), Lineup Optimizer (zero weight punted categories), Weekly matchup planning (identify when opponent is already winning your punted categories = free losses, focus resources on contested categories) | `league_standings` (category ranks), `category_gap_analysis()` (already exists), `CATEGORY_CORRELATIONS` in standings_engine.py | Medium — algorithm design, not data collection |
| E9 | **Schedule-Aware Streaming Intelligence** | Current streaming candidates are scored by raw pitcher quality + park factor. Should also factor: number of games the user's team plays that day (off-day streams are more valuable), whether the stream spot can be used for a two-start pitcher later in the week, and opposing team's recent offensive form (L14 wRC+). | FanGraphs Starting Pitcher Chart methodology: matchup quality = f(pitcher quality, opposing lineup quality, park, weather). Fantasy Six Pack streaming algorithm: two-start pitchers with favorable both-start matchups are 2-3x more valuable than single-start streams. | Lineup Optimizer (Streaming tab), Free Agents page, Daily Optimizer (DCV already factors park but not opponent recent form or schedule slot optimization) | Weekly schedule (already in matchup_adjustments), team_strength (already in game_day.py and MatchupContextService), game_day_weather (already populated) | Low-Medium — data sources exist, need scheduling logic |
| E10 | **Catcher Framing Value** | Elite pitch framers (e.g., Contreras, Raleigh) add 10-15 extra strikes per game, translating to measurable ERA/WHIP benefit for their team's pitchers. This affects pitcher projections when paired with different catchers. | Baseball Savant Catcher Framing leaderboard: top framers save 15+ runs per season. Research: pitcher ERA can differ by 0.20-0.40 depending on catcher framing ability. FanGraphs Shadow Zone data. | Pitcher projections (catcher-adjusted ERA), Start/Sit (pitcher with elite catcher gets ERA boost), Trade valuation (catcher framing value not captured by counting stats) | Baseball Savant catcher framing leaderboard (via pybaseball `statcast_running_value` or scrape), team rosters to identify starting catcher per game | Medium — need framing data + pitcher-catcher pairing logic |

### Implementation Priority for New Signals

**Highest value / lowest effort (do first):**
1. E2: Stuff+/Location+/Pitching+ — DB columns already exist, just populate in bootstrap
2. E9: Schedule-aware streaming — data sources already wired, need scheduling logic
3. E1: BABIP regression targets — simple computation from existing stats

**High value / medium effort:**
4. E6: Enhanced weather model (wind direction relative to park)
5. E4: Pitcher fatigue & workload model
6. E5: Stolen base prediction model (sprint speed + catcher pop time)
7. E8: Punt strategy optimizer

**High value / high effort:**
8. E3: Umpire strike zone adjustment (needs new data source)
9. E7: Pitcher-batter matchup history (large data volume)
10. E10: Catcher framing value (needs new data source)

---

## Complete Priority Matrix (All Backlogs Combined)

**Tier 1 — Do Now (highest leverage, cascade everywhere):**
1. A1: Dynamic SGP denominators
2. A4: Remove self-referential ECR
3. E2: Populate Stuff+/Location+/Pitching+ columns
4. A2+D3: Weighted projection stacking

**Tier 2 — Core ML Models:**
5. D1: Statcast XGBoost regression model
6. D2: Playing time prediction model
7. A3: Bayesian SGP updating
8. E1: BABIP regression targets

**Tier 3 — Intelligence Layer Upgrades:**
9. E9: Schedule-aware streaming
10. E6: Enhanced weather (wind direction)
11. E4: Pitcher fatigue & workload model
12. E5: Stolen base prediction model
13. D5: NLP news sentiment
14. D4: Opponent behavior learning

**Tier 4 — Validation & Strategy:**
15. D6: Backtesting framework
16. E8: Punt strategy optimizer
17. B6+B8: Urgency k-calibration + trade finder weight validation

**Tier 5 — Advanced / Experimental:**
18. E3: Umpire strike zone adjustment
19. E7: Pitcher-batter matchup history
20. E10: Catcher framing value
21. D7: Category-aware lineup RL

**Remaining calibration items (B1-B5, B7, C1-C5):** As time allows, in numbered order.
