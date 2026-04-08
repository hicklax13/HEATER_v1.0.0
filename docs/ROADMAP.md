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
