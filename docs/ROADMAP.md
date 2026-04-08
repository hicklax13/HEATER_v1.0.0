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

## Trade Finder Accuracy Backlog

Identified April 7, 2026 via deep research across FanGraphs, Baseball Savant, behavioral economics
(Kahneman/Tversky, Hobbs 2022), game theory (Nash Bargaining Solution), and sabermetric literature
(Carleton stabilization thresholds, Alan Nathan physics). All items specific to the Trade Finder
page and its 5 tabs.

### F — Acceptance Model Improvements

| # | Item | Research Basis | Fix | Impact |
|---|------|---------------|-----|--------|
| F1 | **Nash Bargaining Fairness** | Nash Bargaining Solution: fair trade = both sides gain proportional to bargaining power, not equal raw value. | Compute `nash_fairness` where alpha = bargaining power (standings, FA alternatives, weeks remaining). Replace or supplement ADP/ECR fairness. | Theoretically grounded; captures that desperate teams should concede more. |
| F2 | **Draft-Round Anchoring** | Endowment effect 2.3x (Kahneman 1990). Round 1-3 picks have 30% higher perceived value due to sunk cost. | In acceptance model: Rd 1-3 = 1.3x endowment multiplier, Rd 4-8 = 1.15x, Rd 9+ = 1.0x. Apply to opponent's perceived value. | 15-30% more realistic acceptance for early-round players. Uses existing draft data. |
| F3 | **Disposition Effect** | Managers hold losers too long, sell winners 1.5x more readily. | If opponent player YTD > projection: +10-15% acceptance (willing to "lock in gains"). If YTD < projection: -10-15% acceptance. | Captures #1 behavioral trade rejection pattern. Uses existing YTD data. |
| F4 | **Recently-Acquired Penalty** | Hobbs 2022: compounding endowment — recently traded-for players overvalued MORE than drafted ones. | Check transactions: if opponent acquired player via trade within 3 weeks, -10-15% acceptance. | Uses existing Yahoo transaction data. |
| F5 | **Playoff-Odds Acceptance** | Teams at 30-70% playoff probability trade most actively. >85% conservative. <15% disengaged. | Replace raw standings rank with playoff odds from `simulate_season_enhanced()`. Willingness curve: <15% = 0.5x, 30-70% = 1.2x, >85% = 0.7x. | More precise than rank alone. Data already computed in standings_engine. |

### G — Regression Signal Improvements

| # | Item | Research Basis | Fix | Impact |
|---|------|---------------|-----|--------|
| G1 | **xwOBA-wOBA Regression Flags** | FanGraphs/Savant: gap ≥0.030 = 1.5 SD, predicts 40-80 point wOBA surge. Stabilizes at 50-60 batted ball events. | Add `xwoba_delta` to pool. ≥+0.030 = BUY_LOW, ≤-0.030 = SELL_HIGH. +0.05 composite bonus when receiving buy-low / giving sell-high. | Single most actionable regression signal. Free via pybaseball. |
| G2 | **Stuff+/botERA Pitcher Regression** | Stuff+ to ERA r²=0.996. Pitching+ beats ALL projection systems for relievers after 250 pitches. | Populate existing DB columns via pybaseball. `stuff_delta = botERA - actual_ERA`. ≤-0.50 = pitcher BUY_LOW, ≥+0.50 = SELL_HIGH. | Low effort — DB columns exist but are empty. Transforms pitcher trade evaluation. |
| G3 | **BABIP Regression Scoring** | Stabilizes at 820 BIP (~1.6 seasons). Early-season BABIP is 80-90% noise. LD%+speed predicts individual baseline. | `babip_regression = (career_BABIP - current_BABIP) / 0.020`, capped ±2.0. Add career BABIP from historical_stats. | Best early-season regression signal (weeks 3-8). |
| G4 | **Stat Reliability Weighting** | Carleton/FanGraphs: K% at 60 PA, BB% at 120, HR at 170, AVG at 910, BABIP at 820. | `reliability = min(1.0, PA / threshold[stat])`. YTD modifier = `1.0 + (divergence * reliability * clamp)`. At 50 PA, HR reliability = 0.29. | Prevents over-reacting to small samples AND under-reacting late season. No new data. |
| G5 | **Velocity Trend Signal** | 1.0 mph fastball decline ≈ 0.5-0.8 ERA increase, 2-3% K% decrease. Gainers maintain gains 8/9 times. | `velo_delta = current_FF_velo - prior_season_FF_velo`. ≤-1.0 = SELL, ≥+1.0 = BUY. Weight 10-15% of pitcher valuation. | Early warning before ERA/WHIP show decline. Free via pybaseball Statcast. |

### H — Strategic & Structural Improvements

| # | Item | Research Basis | Fix | Impact |
|---|------|---------------|-----|--------|
| H1 | **Differential Time Decay** | Counting stats value ∝ weeks remaining (linear). Rate stats ≈ constant but confidence decreases below 8 weeks. | Split: `counting *= weeks_rem / total_weeks`. Rate stays 1.0 with `confidence = min(1.0, weeks_rem / 8)`. | More accurate ROS valuation. Prevents overvaluing counting stats late. |
| H2 | **Dynamic Roster Spot Value** | FanGraphs: spot = $2-3 in $260 budget. Actual value = f(FA pool quality). | `ROSTER_SPOT_SGP = median_fa_sgp * 0.8`, recomputed weekly from FA pool. | Fixes 2-for-1 valuation in leagues with strong/weak waivers. |
| H3 | **Graduated Positional Scarcity** | C most scarce (24th C = -$9), 2B second, SS loaded top but thin middle, OF deepest. | C=1.20x, 2B=1.15x, SS=1.10x, 3B=1.05x, OF/1B=1.00x. Or compute from replacement-level gap. | More accurate than flat 1.15x for C/SS/2B. Trivial change. |
| H4 | **SB Independence Premium Increase** | Published R²: SB-RBI = 0.0002, SB-HR = 0.048. Mean r=0.12, most isolated category. | Increase from 1.08x to 1.12-1.15x. | SB contributors slightly undervalued currently. |
| H5 | **Trade Timing Multiplier** | Buy-low window peaks weeks 3-6. AVG unreliable before 100 PA. Stats stabilize by week 10+. | Dynamic YTD clamp: weeks 1-4 = ±5%, weeks 5-8 = ±10%, weeks 9+ = ±15%. Add 1.1x bonus for weeks 4-10. | Prevents chasing April hot streaks; allows meaningful June adjustments. |
| H6 | **Consistency/Variance Modifier** | BaseballHQ DOM%/DIS%. Consistent 2 HR/week beats volatile 8 HR in 6 weeks. | Weekly CV from game logs. CV<0.30 = +5-8% SGP. CV>0.50 = -5-8% SGP. Needs 6+ weeks. | H2H-specific dimension currently invisible in trade valuation. |
| H7 | **Proactive Punt Advisor** | FanGraphs MC: SB lowest mean correlation (r=0.12). W lowest pitching (R²≈0.03). Punting correlated pairs is safer. | `recommend_punt_targets()` using mean pairwise correlation. Show in Trade Readiness: "Punt SB to gain +1.5 standings in HR/R/RBI." | Transforms Trade Finder from reactive to strategic. |
| H8 | **Closer Stability Discount** | Full closer = 25-30 SV. Committee = 12-15 SV (50% cut). 3+ BS in 2 weeks = leading indicator. | `sv_adjusted = sv_proj * (confidence / 100)`. Wire closer_monitor job security into trade valuation. | Prevents overvaluing shaky closers. closer_monitor already exists. |
| H9 | **Prospect Call-Up Valuation** | `value = P(call_up) * ROS_SGP * (weeks_avail / weeks_rem) - ROSTER_SPOT_SGP`. Top-25 produce value ~40-50% of time. | Add call-up probability from FanGraphs ETA (already fetched). Compute expected value for incoming prospects. | Better than binary "stash or zero" treatment. |

### Trade Finder Priority Order

**Tier 1 (biggest composite improvement):** G1 (xwOBA), G4 (stat reliability), F5 (playoff-odds acceptance), G2 (Stuff+/botERA)
**Tier 2 (acceptance refinements):** F2 (anchoring), F3 (disposition), H1 (time decay), H2 (dynamic roster spot)
**Tier 3 (strategic intelligence):** H7 (punt advisor), H8 (closer discount), G3 (BABIP), H5 (trade timing)
**Tier 4 (polish):** F1 (Nash), F4 (recently-acquired), H3 (scarcity), H4 (SB premium), H6 (consistency), G5 (velocity), H9 (prospects)

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

**Tier 6 — Trade Finder Specific (F/G/H items):**
22. G1: xwOBA regression flags
23. G4: Stat reliability weighting
24. F5: Playoff-odds acceptance
25. G2: Stuff+/botERA pitcher regression
26. F2: Draft-round anchoring
27. F3: Disposition effect modifier
28. H1: Differential time decay
29. H7: Proactive punt advisor
30. H8: Closer stability discount

**Remaining items:** B1-B5, B7, C1-C5, F1, F4, G3, G5, H2-H6, H9 — as time allows.

---

## Lineup Optimizer Accuracy Backlog

Identified April 7, 2026 via deep research across FanGraphs, Baseball Savant, academic papers
(Management Science, MIT Sloan, American Meteorological Society), and sabermetric literature.
All items specific to the Lineup Optimizer page and its 6 tabs.

### I — Weekly & Daily Optimization Improvements

| # | Item | Current State | Research Basis | Fix | Impact |
|---|------|--------------|---------------|-----|--------|
| I1 | **Mid-Week Pivot Advisor** | No mid-week category state tracking. Optimizer runs once at start of week. | FanGraphs H2H: you only need 7 of 12 categories. Identifying 2-3 lost causes by Wednesday and redirecting resources swings 1-2 extra category wins/week = 24+ category wins/season. | Classify each category as WON (P(win)>85%, protect), LOST (P(win)<15%, concede), or CONTESTED (30-70%, pursue). Bench pitchers to protect WON ratios. Stream for CONTESTED counting stats. | Estimated +1-2 category wins/week. Uses existing Yahoo matchup data. |
| I2 | **Swing-Category Weighting** | H2H weights use Normal PDF proximity to toss-up. Does not distinguish "swing" from "locked". | Haugh & Singal 2019 (Management Science/Columbia): optimal lineup maximizes P(winning 7+ categories) not total SGP. Categories at P(win)=40-60% have highest marginal value. | Replace or supplement current H2H weights: maximize investment in categories where P(win) is 30-70%. De-emphasize categories where P(win) > 80% (already won) or < 20% (already lost). | +0.5-1 category wins/week. standings_engine.py already computes P(win). |
| I3 | **Ratio Protection Calculator** | No ERA/WHIP protection logic. Optimizer treats all starts equally regardless of current ratio lead. | With 30 IP at 2.50 ERA, one 6 IP / 5 ER start raises ERA to 3.50. With 60 IP banked, same disaster = 3.18. Protection decision depends on IP base + lead size + remaining matchup quality. | `marginal_era_risk = (proj_ER * 9) / (banked_IP + proj_IP) - current_ERA`. If risk > lead * threshold: bench the pitcher. Show "Protect Ratios" mode in Daily Optimize. | Protects 1-2 ratio category wins/week when leading ERA/WHIP. |
| I4 | **Category Flip Probability** | No model for which categories might flip on remaining days of the matchup. | SB is most volatile (binary event), AVG/ERA can swing dramatically with low sample. K/RBI are most stable (accumulated). Quantified: SB has highest single-day flip probability. | Compute `flip_prob = f(current_margin, daily_stdev, games_remaining)`. On final day: recommend targeting highest flip-probability LOSING categories and protecting highest flip-probability WINNING categories. | Critical for close H2H matchups. Especially valuable for playoff weeks. |
| I5 | **Batting Order Slot PA Adjustment** | Daily projections use `season_proj / 162` regardless of batting order position. | FanGraphs: leadoff gets 4.65 PA/game, 9th gets 3.77 PA/game — a 19% difference. Each slot costs ~17.8 PA per 600 PA. | When confirmed lineups are available (via `get_todays_lineups()`), adjust daily counting stat projections by `PA_at_slot / PA_at_reference_slot`. Player moved from 6th to 2nd = +8% daily projection. | +5-10% daily projection accuracy. game_day.py already fetches batting orders. |
| I6 | **IP Management Mode** | Fixed `IP_BUDGET` concept exists but no dynamic strategy that adapts to mid-week ratio position. | Expert consensus: if winning ERA by 0.50+ with 40+ IP and winning WHIP by 0.15+, bench borderline starters. Career WHIP > 1.40 = avoid streaming regardless of matchup. | Add IP management mode toggle: "Chase K/W" (stream aggressively), "Protect Ratios" (bench risky arms), "Balanced" (default). Mode auto-recommended from category urgency state. | Prevents ratio destruction from unnecessary starts. |

### J — Projection & Signal Improvements

| # | Item | Current State | Research Basis | Fix | Impact |
|---|------|--------------|---------------|-----|--------|
| J1 | **Per-Stat In-Season Update Rates** | Bayesian updater uses uniform blend weight. Same trust in AVG at 100 PA as K% at 100 PA. | FanGraphs/Carleton: K% stabilizes at 60 PA, AVG at 910 AB, BABIP at 820 BIP, pitcher K% at 70 BF, pitcher BABIP at 2000 BIP. Trust in stats should scale per-stat. | Replace uniform blend with stat-specific: `current_weight[stat] = PA / stabilization_threshold[stat]`. At 100 PA: K% is 62.5% observed (trust it), AVG is 11% observed (don't trust it), HR rate is 59% observed (trust it). | Single highest-impact projection improvement. Different stats should update at drastically different speeds. |
| J2 | **Platoon Split Bayesian Regression** | Uses individual platoon data when available. Falls back to league-average defaults. | The Book (Tango/Lichtman/Dolphin): LHB platoon splits need ~1000 PA to stabilize, RHB need ~2200 PA. Individual single-season splits are almost entirely noise. | Always use Bayesian blend: `adjusted = (individual_PA / stab_PA) * individual + (1 - individual_PA / stab_PA) * league_avg`. For LHB with 200 PA vs LHP: use 80% league-avg, 20% individual. | Eliminates noise injection from unreliable individual splits. Currently matchup_adjustments.py may use raw splits. |
| J3 | **Opposing Pitcher Quality Scalar** | `matchup_adjustments.py` has Log5-based adjustment but calibration unknown. | Research: facing ace vs replacement-level = ~40-60 wOBA points = 15-20% counting stat adjustment. Pitcher quality scalar: `mult = 1.0 + 0.15 * (league_avg_xFIP - opp_xFIP) / league_std_xFIP`. | Calibrate existing opposing pitcher adjustment to these specific magnitudes: ±15% for extreme matchups, centered at 1.0 for league-average. | Currently may be over- or under-adjusted. Data exists in opp_pitcher_stats. |
| J4 | **Comprehensive Weather Model** | HR-only temperature adjustment (Nathan +0.9%/°F). No rain, wind, K/BB effects. | AMS research: rain = +9.6% walks, -10.1% K. Wind out 10+ mph = +20% HR. 90°F vs 50°F = +20% HR, 0.037 ISO gap. Humidity effect near-zero. | Expand weather model: (1) Rain > 40% chance: pitcher BB proj × 1.10, K proj × 0.90. (2) Wind out > 10 mph: HR proj × 1.15-1.20. (3) Temperature: linear power scalar 0.96 at 40F to 1.04 at 90F. | Rain K/BB effects are larger than most realize. Data from Open-Meteo already fetched. |
| J5 | **Catcher Framing Pitcher Adjustment** | No catcher framing data used. | Pitcher List/Baseball Savant: best-to-worst framer gap = ~0.34 ERA and ~0.95 K/9 over full season. Per start: ±0.25 ERA adjustment for elite/poor framers. | Fetch catcher framing leaderboard (Baseball Savant). For each pitcher start: `era_adj = -0.01 * catcher_framing_runs`, `k9_adj = 0.025 * catcher_framing_runs`. Patrick Bailey (+25 runs) = -0.25 ERA, +0.625 K/9 vs baseline. | Meaningful for streaming decisions between similar-tier pitchers on different teams. |
| J6 | **Projection Uncertainty Bands** | Percentile projections use cross-system variance. Not calibrated to empirical year-to-year SD. | Baseball Prospectus: ERA year-to-year SD = 1.20, WHIP SD = 0.20, K/9 SD = 1.82. A 3.50 ERA pitcher's 90% CI is [2.30, 4.70]. AVG SD = ~0.025. | Use empirical SDs for P10/P50/P90 instead of cross-system spread. When two players are within 1 SD, treat as interchangeable — use tiebreakers (matchup, weather, platoon). | Prevents over-optimizing on noise. Focuses optimizer energy on categories with reliable projections (K/9, BB/9). |

### K — H2H Strategic Features

| # | Item | Current State | Research Basis | Fix | Impact |
|---|------|--------------|---------------|-----|--------|
| K1 | **SB Streaming by Catcher/Pitcher** | No catcher pop time or pitcher delivery time in SB projections. | Statcast: pitcher delivery + catcher pop time threshold is ~3.5s total (coin flip). Below 3.3s = runner out. Above 3.7s = runner safe. Success rate: 80.3% when sprint speed > 29 ft/s. | Build `sb_opportunity_score = sprint_speed * (pop_time / 1.95) * (delivery_time / 1.35) * handedness_factor`. Surface highest-scoring runners when losing SB mid-week. | Targeted SB streaming. Uses Statcast pop time (free) + pitcher delivery data. |
| K2 | **Pitcher Fatigue Multiplier** | No workload tracking for in-season projection decay. | PMC research: self-reported fatigue = 36x injury risk. 100+ IP in calendar year = 3.5x risk. ACWR > 1.3 = 15x injury risk. 16/20 Cy Youngs had BETTER second-half ERA (elite pitchers defy trend). | `fatigue_factor = max(0.85, 1.0 - 0.003 * max(0, current_IP - 100))`. 0.3%/IP discount above 100 IP, cap at 15%. Flag ACWR > 1.3 pitchers. Exempt top-10 Stuff+ pitchers (elite arms resist fatigue). | Prevents over-projecting fatigued mid-tier pitchers in August/September. |
| K3 | **Consistency Premium** | No variance tracking. Equal SGP for consistent vs volatile players. | FanGraphs ATC Volatility: higher projection volatility correlates with lower rotisserie earnings. Positive skew = +$8 value at mid-round. H2H simulation: consistent player wins ~5-8% more weekly matchups. | `consistency_penalty = k * (weekly_CV)` where k = 0.05-0.10. Applied as SGP discount for volatile players, SGP bonus for consistent ones. Requires L7/L14/L30 game log data (exists in game_day.py). | H2H-specific value dimension. Currently invisible in optimization. |
| K4 | **Punt Mode Optimizer** | Category urgency sets low weight for losing categories. No explicit punt configuration. | FanGraphs MC: punting 1-2 correlated categories in H2H concentrates resources for 7-5 or 8-4 weekly wins. SB (r=0.12) and W (R²=0.03) are safest punt targets. | Add punt mode toggle: user selects 0-2 categories to punt. Optimizer sets those weights to exactly 0.0 and redistributes freed roster slots to strengthen remaining categories. | Transforms optimizer from balanced-roster-only to strategy-aware. |
| K5 | **Streaming Composite Score** | Current streaming uses counting_sgp + rate_impact. No opponent strength or WHIP safety filter. | FanGraphs SP Chart: composite = K_proj × (1/opp_wOBA) × park_factor × recent_form. Career WHIP < 1.25 = safe stream floor. WHIP > 1.40 = avoid. | `stream_score = K_proj * (1/opp_wOBA) * park_factor * form_L3 * whip_safety`. Filter: career WHIP > 1.40 → excluded unless opp wOBA < .290. Two-start bonus: 1.8x (not 2x, accounts for fatigue). | Better streaming picks. Data mostly exists. |
| K6 | **Dynamic Volume Factor from Confirmed Lineups** | volume_factor = 0.9 when lineup not posted, 1.0 when confirmed. No batting order granularity. | MLB lineups posted 2-4 hours before first pitch. Confirmed lineup gives batting order slot → PA/game rates differ by 19% (slot 1 vs 9). | When lineup confirmed: set volume_factor from batting order slot PA rate (4.65 for leadoff, 3.77 for 9th). When NOT in lineup: volume_factor = 0.0 (not 0.3). When not yet posted: volume_factor = 0.85 (slightly conservative). | More accurate daily DCV scores. game_day.py already fetches lineups. |

### Lineup Optimizer Priority Order

**Tier 1 (biggest wins/week impact):**
1. I1: Mid-week pivot advisor (classify WON/LOST/CONTESTED)
2. J1: Per-stat in-season update rates (K% 15x faster than AVG)
3. I2: Swing-category weighting (maximize 30-70% P(win) categories)
4. I3: Ratio protection calculator (bench risky pitchers when protecting ERA/WHIP)

**Tier 2 (projection accuracy):**
5. J2: Platoon split Bayesian regression (eliminate noise from raw individual splits)
6. J3: Opposing pitcher quality scalar (calibrate to ±15%)
7. J4: Comprehensive weather model (rain K/BB, wind HR, full temperature)
8. I5: Batting order slot PA adjustment (+5-10% daily accuracy)

**Tier 3 (strategic features):**
9. K4: Punt mode optimizer (zero-weight designated categories)
10. K1: SB streaming by catcher/pitcher matchup
11. K2: Pitcher fatigue multiplier (100+ IP decay)
12. I4: Category flip probability (last-day strategy)

**Tier 4 (refinements):**
13. I6: IP management mode (protect/chase/balanced)
14. K5: Streaming composite score (opponent strength + WHIP filter)
15. K3: Consistency premium (CV-based SGP adjustment)
16. J5: Catcher framing pitcher adjustment
17. J6: Projection uncertainty bands (empirical SDs)
18. K6: Dynamic volume factor from confirmed lineups
