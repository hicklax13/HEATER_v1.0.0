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
| 6 | Unified Enrichment | Apr 2026 | Enriched player pool, MatchupContextService, page wiring |

## Infeasible Items (Free API Constraint)

| Item | Reason | Proxy |
|------|--------|-------|
| PECOTA | Baseball Prospectus paywall | Marcel local computation |
| ESPN ADP | API restricted | FantasyPros ECR (aggregates ESPN) |
| Rotowire injury feed | Paywall | MLB Stats API injuries |
| Individual expert rankings | Paywalls | FantasyPros ECR consensus (100+ experts) |

---

## Improvement Backlog — By Page

**115 unique item rows** after deduplication and audit.
**67 DONE, 2 PARTIAL, 46 remaining** as of April 8, 2026.
Organized strictly by the page each task improves. Items that affect all pages
are under "Global / Core Engine." Status: (empty)=not started, PARTIAL=infrastructure
exists, DONE=implemented, CUT=removed after audit, MERGED=combined with another item.

---

### 1. GLOBAL / CORE ENGINE — 17 items

Cascade through all 53 engines and all 13 pages. Highest leverage.

| # | Item | Fix | Impact | Effort | Status |
|---|------|-----|--------|--------|--------|
| A1 | **Dynamic SGP Denominators** | `compute_sgp_denominators()` EXISTS at `valuation.py:718` but not wired. Wire into `value_all_players()`. Update weekly. | Propagates to EVERY engine. Single highest-leverage fix. | Low | DONE |
| U1 | **Rate-Stat Baselines from League Data** | Auto-compute AB/IP baselines from `league_standings` instead of hardcoded 5500 AB / 1300 IP. If league AB is 5200, every AVG SGP is off ~5%. | Cascades through all rate-stat SGP. Pairs with A1. | Low | DONE |
| A2+D3 | **Weighted Projection Stacking** | Ridge regression: each system = feature, 2025 actuals = target. Learned weights > naive 1/n average. | ~5-8% projection improvement. | Low | DONE |
| A3 | **Bayesian SGP Updating** | Preseason denoms as prior, update weekly from standings. By week 8: 70% actual / 30% prior. | Prevents stale denominators mid-season. | Medium | |
| A4 | **Remove Self-Referential ECR** | Remove HEATER SGP rank as source #7 in ECR consensus. Use 6 external sources only. | Removes circular confirmation bias. Confirmed still present in `ecr.py:910`. | Trivial | DONE |
| E2+T1 | **Fetch & Populate Stuff+/Location+/Pitching+** | Add `pybaseball.pitching_stats(year, qual=0)` to bootstrap. Write to existing empty DB columns. | Transforms pitcher evaluation across all pages. Unlocks G2, O1, L2. | Trivial | DONE |
| T2 | **DATA: Fetch Detailed Batting Stats** | Add `pybaseball.batting_stats(year, qual=0)` to bootstrap. Populate LD%, FB%, GB%, BABIP, ISO, K%, BB%. | Unlocks E1, G3, O1 breakout components. | Trivial | DONE |
| T3 | **DATA: Fetch Sprint Speed** | Add `pybaseball.statcast_sprint_speed(year)` to bootstrap. | Unlocks K1, E5 stolen base prediction. | Trivial | DONE |
| T4 | **DATA: Dynamic Park Factor Refresh** | Mid-season `pybaseball.team_batting()` + `team_pitching()` park factor derivation. Replace static 2024 values. | Some parks change year-to-year (humidor, dimensions). | Low | |
| B2 | **Position-Specific Health Scoring** | C: threshold 28/0.03yr. DH: 34/0.01yr. OF: 31. | Catchers age faster than DHs. | Low | |
| B3 | **Injury-Type Adjustment** | TJ = 0.4 health floor 2yr. Hamstring = 0.7 1yr. Concussion = 0.6. | 30 IL days from bone bruise ≠ 30 IL days from TJ. | Medium | |
| B4 | **Temporal ECR Weighting** | Exponential decay 14-day half-life. March rank = 0.25x weight vs April = 1.0x. | Stale preseason ranks devalued. | Low | |
| D1 | **Statcast XGBoost Regression** | Train on EV, barrel, xwOBA, xBA, sprint speed. Target: actual − projected. Retrain daily. | ~10-15% projection improvement. | Medium | DONE |
| D2 | **Playing Time Prediction** | Ridge regression: `remaining_PA = f(recent_PA_rate, depth_chart, health, age)`. Update weekly. | ~10-15% counting stat accuracy. #1 projection error source. | Medium | DONE |
| D6 | **Backtesting Framework** | Replay past weeks, score engine recommendations vs actual outcomes. | Meta: validates ALL other changes. Without this, every weight is a guess. | Medium-High | DONE |
| J1 | **Per-Stat In-Season Update Rates** | `STABILIZATION_POINTS` dict exists in `bayesian.py:38`. Verify trade YTD modifier (G4) also uses per-stat thresholds. | Projection blend already uses per-stat rates. Trade YTD modifier may not. | Low | PARTIAL |
| J6 | **Projection Uncertainty Bands** | Use empirical SDs (ERA=1.20, WHIP=0.20, AVG=0.025) for P10/P50/P90. | Prevents over-optimizing on noise. | Low | |
| V1 | **UNIFY: Category Weights** | 4 divergent methods (urgency sigmoid, gap analysis, H2H PDF, simple median). Same roster gets contradictory priorities on different pages. Create `MatchupContextService.get_category_weights(mode)` with 3 modes: "matchup" (H2H urgency), "standings" (gap analysis), "blended" (alpha-weighted). All pages consume from ONE source. | **CRITICAL** — pages currently give contradictory advice for same roster. | Medium | DONE |
| V2 | **UNIFY: SGP Computation** | 5 divergent paths. `_totals_sgp()` in trade_finder treats AVG like HR (no volume adjustment) — mathematically wrong. Remove `_totals_sgp()` and `_weighted_totals_sgp()`. Replace with `SGPCalculator.player_sgp()` which properly volume-weights rate stats. | **CRITICAL** — Trade Finder By Value tab inflates value of low-PA players. | Medium | DONE |
| V3 | **UNIFY: Roster Totals** | 3 implementations (My Team, in_season, League Standings) with different defaults (AVG=0.250 vs 0.0) and formats (string vs float). Create `standings_utils.get_team_totals()` with session cache. | Inconsistent defaults across pages. | Low | |
| V4 | **UNIFY: Opponent Intelligence** | My Team + Free Agents import `opponent_intel.py` directly (no cache). Trade Analyzer uses `opponent_trade_analysis.py` independently. Only Matchup Planner uses `MatchupContextService`. All should use MCS. | Different opponent profiles on different pages. | Low | |
| V5 | **UNIFY: FA Pool Access** | 4 loading paths (Yahoo API, local function, enriched pool filter, DB query). FA pool loaded differently on Free Agents page vs Optimizer. Create single `get_fa_pool()` in session cache. | Different FA lists on different pages. | Low | |
| V6 | **UNIFY: Stat Display Formatting** | AVG shown as `.2f` in some variance displays (wrong, should be `.3f`). SGP sign inconsistent (`+.2f` vs `.2f`). Create `format_stat(value, stat_type)` in `ui_shared.py`. Enforce: AVG/OBP=`.3f`, ERA/WHIP=`.2f`, SGP=`+.2f`. | Formatting inconsistencies confuse users. | Trivial | DONE |

---

### 2. MY TEAM — 10 items

Daily dashboard: War Room, alerts, roster overview, Monday briefing.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| M1 | **Category Flip Analyzer** | Compute mid-week flip probability per category: `flip_prob = f(margin, daily_stdev, games_remaining)`. Show top-priority flip opportunities. Shares engine with I1+I4. | Most tools show scores but don't compute which categories are close enough to flip. | DONE |
| M2 | **Conditional Swap Impact** | "If you bench Player X and start Player Y, you gain +2 K and lose -0.8 ERA delta." Roster-level marginal impact per swap. | Bridges player-level data to roster-level decision. | |
| M3 | **Opponent Lineup Tracking** | Surface opponent's mid-week roster moves: empty spots, pitcher streams, off-day players. | Uses existing Yahoo API (`yds.get_transactions()` filtered to opponent). | DONE |
| M4 | **Regression Alert System** | Compare L30 actual to expected stats (xBA, xwOBA via barrel% + speed). Flag >1.5 SD divergence as sell-high or buy-low. Weight K%/BB% over BA. Require 50 PA min. | Process metrics more predictive than outcome metrics. | DONE |
| M5 | **IL Slot Utilization Alert** | Empty IL slot = "Fill with X." IL stash ranking by `ros_sgp - best_fa_sgp` with days-to-return. Injury-type duration lookup (hamstring 21d, oblique 28d, TJ 14mo). | Empty IL slot = wasted value. | DONE |
| M6 | **Ratio Lock Alert** | "You're winning ERA by 0.80 and WHIP by 0.15 with 45 IP banked — bench Sunday starters to lock 2 categories." | Worth 1-2 category wins/season. | DONE |
| U3 | **War Room Flippable Thresholds from Weekly SDs** | Replace hardcoded `_COUNTING_THRESHOLD=3` for all stats. Use `threshold[cat] = 1.5 * weekly_sd[cat]`. HR has different variance than R. | Prevents false flip alerts for stable categories, catches real flips in volatile ones. | |
| E3 | **Umpire Strike Zone Adjustment** | Per-umpire K%/BB%/run environment. ±0.3-0.5 runs/game. | | |
| T7 | **DATA: Fetch Umpire Assignments** | Scrape Baseball Savant game feed JSON + Retrosheet historical. Build per-umpire tendency table. | Unlocks E3. 1-2 hours. | |
| S4 | **CONSOLIDATE: Merge Alerts + War Room** | My Team shows same injuries in both "News & Alerts" and "War Room." Merge into single surface. | Cleaner page, single source of truth. | |

---

### 3. TRADE FINDER — 23 items

5 tabs: Smart Recommendations, By Value, Target a Player, Browse Partners, Trade Readiness.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| F2 | **Draft-Round Anchoring** | Rd 1-3 = 1.3x endowment, Rd 4-8 = 1.15x, Rd 9+ = 1.0x on opponent perceived value. | 15-30% more realistic acceptance for early-round players. | DONE |
| F3 | **Disposition Effect** | YTD > projection: +10-15% acceptance. YTD < projection: -10-15%. | Captures #1 behavioral rejection pattern. | DONE |
| F4 | **Recently-Acquired Penalty** | Traded-for within 3 weeks: -10-15% acceptance. | Uses existing Yahoo transaction data. | DONE |
| F5 | **Playoff-Odds Acceptance** | Replace raw standings rank with `simulate_season_enhanced()` playoff odds. | More precise than rank alone. Data already computed. | DONE |
| B1 | **League-Specific Acceptance Calibration** | Empirical acceptance rates from Yahoo transactions. | League-calibrated, not generic. Prerequisite for D4. | |
| D4 | **Opponent Behavior Learning** | Per-opponent logistic regression on trade history. Requires B1 first. | ~10-20% acceptance improvement. | |
| G1 | **xwOBA-wOBA Regression Flags** | Gap ≥0.030 = BUY_LOW, ≤-0.030 = SELL_HIGH. +0.05 composite bonus. | Single most actionable regression signal. | DONE |
| G2 | **Stuff+/botERA Pitcher Regression** | `stuff_delta = botERA - actual_ERA`. ≤-0.50 = BUY_LOW, ≥+0.50 = SELL_HIGH. | Requires E2+T1 data. | DONE |
| G3 | **BABIP Regression Scoring** | `(career_BABIP - current) / 0.020`, capped ±2.0. | Best early-season signal (weeks 3-8). E1 supersedes once implemented. | DONE |
| G4 | **Stat Reliability Weighting** | `reliability = min(1.0, PA / threshold[stat])`. Trust K% at 60 PA ≠ AVG at 60 PA. | Uses same stabilization table as J1 but applied to trade YTD modifier. | DONE |
| G5 | **Velocity Trend Signal** | 1.0 mph decline ≈ 0.5-0.8 ERA increase. `velo_delta = current - prior`. | Early warning before ERA shows it. | DONE |
| E1 | **BABIP Regression Targets** | xBABIP from contact quality + sprint speed. Predict future BABIP within 15 points. | Forward-looking model. Supersedes G3 heuristic. | |
| H1 | **Differential Time Decay** | Counting stats × `weeks_rem / total_weeks`. Rate stats stay 1.0 with confidence penalty <8 weeks. | More accurate ROS valuation. | DONE |
| H2 | **Dynamic Roster Spot Value** | `ROSTER_SPOT_SGP = median_fa_sgp * 0.8`, recomputed weekly from FA pool. | Fixes 2-for-1 valuation. | DONE |
| H3 | **Graduated Positional Scarcity** | C=1.20x, 2B=1.15x, SS=1.10x, 3B=1.05x, OF/1B=1.00x. | More accurate than flat 1.15x. | DONE |
| H4 | **SB Independence Premium** | Increase from 1.08x to 1.12-1.15x. | SB contributors undervalued (R²=0.0002 with RBI). | DONE |
| H5 | **Trade Timing Multiplier** | Weeks 1-4 = ±5% YTD clamp, weeks 5-8 = ±10%, weeks 9+ = ±15%. 1.1x bonus weeks 4-10. | Prevents chasing April hot streaks. | DONE |
| H6 | **Consistency/Variance Modifier** | Weekly CV from game logs. CV<0.30 = +5-8% SGP, CV>0.50 = -5-8%. | H2H dimension invisible today. | DONE |
| H7 | **Proactive Punt Advisor** | Same engine as K4. Surface in Trade Readiness: "Punt SB to gain +1.5 in HR/R/RBI." | Transforms Trade Finder from reactive to strategic. | DONE |
| H8 | **Closer Stability Discount** | `sv_adjusted = sv_proj * (confidence / 100)`. Wire closer_monitor into trade valuation. | Prevents overvaluing shaky closers. | DONE |
| H9 | **Prospect Call-Up Valuation** | `value = P(call_up) * ROS_SGP * (weeks_avail / weeks_rem) - ROSTER_SPOT_SGP`. | Better than binary stash/zero. | |
| U2 | **LP-Constrained Totals in By Value Tab** | `scan_1_for_1()` uses raw roster totals counting bench. Wire LP-constrained 18-starter totals. | By Value currently overcounts bench production. | DONE |
| B8 | **Validate Trade Finder Weights** | Backtest top-ranked trades vs actual improvement by week 24. | Requires D6 backtesting framework. | DONE |
| S1 | **CONSOLIDATE: Merge Smart Recs + By Value Tabs** | Same data, different sort. Single tab with sort/filter toggles. | Reduces user confusion. | DONE |

---

### 4. TRADE ANALYZER — 4 items

6-phase evaluation engine for specific trade proposals.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| P1 | **Trade Grade Confidence Interval** | Show grade RANGE ("B+ to A-") not single letter. Based on ±1 SD projection uncertainty. | More honest. Current single grade implies false precision. | DONE |
| P2 | **Antithetic Variate MC Sampling** | For each sim, generate mirror using negated z-scores. 20K pairs for cost of 10K. Cuts SE ~30%. | Free precision improvement. | |
| P3 | **Copula Correlation Calibration** | Use empirical correlations from league weekly totals instead of hardcoded matrix. Share with scenario generator. | Hardcoded correlations underestimate variance by ~15%. | |
| P4 | **Per-Category Replacement Level** | Subtract replacement-level stats per category per position (not aggregate SGP). | Captures positional scarcity in specific categories. | DONE |

---

### 5. LINEUP OPTIMIZER — 17 items

6 tabs: Start/Sit, Optimize, Manual, H2H, Streaming, Daily Optimize.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| I1+I4 | **Mid-Week Pivot + Category Flip Probability** | (1) `flip_prob = f(margin, daily_stdev, games_remaining)`. (2) Classify WON/LOST/CONTESTED. (3) Bench pitchers for WON ratios, stream for CONTESTED. | +1-2 category wins/week. | DONE |
| I2 | **Swing-Category Weighting** | Maximize P(win) for 30-70% categories in LP objective. De-emphasize locked/lost. | +0.5-1 category wins/week. | DONE |
| I3 | **Ratio Protection Calculator** | `marginal_era_risk = (proj_ER * 9) / (banked_IP + proj_IP) - current_ERA`. Bench if risk > lead. | Protects ratio categories. | DONE |
| I5+K6 | **Batting Order Slot PA Adjustment** | Wire `LINEUP_SLOT_PA` dict + confirmed lineups into DCV counting stats AND volume_factor. Single implementation. | +5-10% daily accuracy. | DONE |
| I6 | **IP Management Mode** | "Chase K/W", "Protect Ratios", "Balanced" toggle. Auto-recommend from urgency state. | Prevents ratio destruction. | DONE |
| J2 | **Platoon Split Bayesian Regression** | `adjusted = (PA/stab) * individual + (1-PA/stab) * league_avg`. LHB stab=1000, RHB stab=2200. | Eliminates noise from raw individual splits. | DONE |
| J3 | **Opposing Pitcher Quality Scalar** | Calibrate existing multiplier to ±15% based on research (ace vs replacement = 40-60 wOBA). | Currently wired but calibration unverified. | DONE |
| J4 | **Comprehensive Weather Model** | Add rain→K/BB (+9.6% BB, -10.1% K when rain >40%) and wind direction→HR. | Temp→HR works. Rain/wind currently ignored. | DONE |
| J5 | **Catcher Framing Pitcher Adjustment** | `era_adj = -0.01 * framing_runs`, `k9_adj = 0.025 * framing_runs`. | ±0.25 ERA per start for elite/poor framers. | |
| B6 | **Category Urgency k-Calibration** | Backtest k=1.0 to 5.0, measure category wins. Requires D6 framework. | Current k=2/3 may be off. | DONE |
| C2 | **Dual Objective Alpha Validation** | Tie alpha to playoff probability, not just time remaining. | Affects strategy, not individual decisions. | DONE |
| K1 | **SB Streaming by Catcher/Pitcher** | `sb_score = sprint_speed * (pop_time/1.95) * (delivery_time/1.35) * handedness`. | Targeted SB streaming. Requires T3, T8 data. | |
| K2 | **Pitcher Fatigue Multiplier** | `fatigue = max(0.85, 1.0 - 0.003 * max(0, IP - 100))`. ACWR >1.3 flag. Exempt top Stuff+. | Prevents over-projecting fatigued arms. | DONE |
| K3 | **Consistency Premium** | `penalty = k * weekly_CV`. k=0.05-0.10. Penalize volatile, reward consistent. | H2H-specific dimension. | DONE |
| K4 | **Punt Mode Optimizer** | User selects 0-2 categories to punt → weight 0.0 in LP. Also feeds H7 in Trade Finder. | Strategy-aware optimization. | DONE |
| K5 | **Streaming Composite Score** | `K_proj * (1/opp_wOBA) * park * form_L3 * whip_safety`. Career WHIP >1.40 = avoid. | Better streaming picks. | DONE |
| S2 | **CONSOLIDATE: Remove H2H Analysis Tab** | Duplicate of Matchup Planner Category Probabilities. Link to Matchup Planner instead. | Eliminates duplicate display. | |

---

### 6. MATCHUP PLANNER — 6 items

5 tabs: Category Probabilities, Player Matchups, Per-Game Detail, Hitters Only, Pitchers Only.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| B5 | **Dynamic Park Factors** | Fetch updated park factors mid-season via pybaseball. | Some parks change year-to-year. | |
| C5 | **Inverse Park Formula** | Use `1.0 / park_factor` (reciprocal) instead of `2.0 - park_factor`. | Tiny except Coors. | |
| E6 | **Enhanced Weather (Wind Direction)** | Wind direction relative to outfield orientation. +20% HR wind out >10 mph. | Currently HR-only, no wind direction. | |
| E7 | **Pitcher-Batter Matchup History** | PvB stats stabilize at ~60 PA. +2-4% over generic platoon. | High data volume. | |
| T11 | **DATA: Stadium Outfield Orientations** | Hardcode 30 outfield bearing angles. One-time research. | Unlocks E6. | |
| T12 | **DATA: Pitcher-Batter Splits** | `pybaseball.statcast_batter(batter_id, pitcher_id)`. Smart caching (rostered players only). | Unlocks E7. High data volume. | |

---

### 7. FREE AGENTS / WAIVER WIRE — 3 items

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| C1 | **Waiver Wire Drop Penalties** | DH penalty scales with Util slots. AVG threshold = league-average (not 0.245). | 5-factor drop scoring added (f795ea8). Thresholds still hardcoded. | PARTIAL |
| E9 | **Schedule-Aware Streaming** | Off-day streams more valuable. Two-start detection. Opponent L14 wRC+ for quality. | Data sources already wired. | DONE |
| D7 | **Category-Aware Lineup RL** | Contextual bandit learning from weekly decisions + outcomes. Needs 8+ weeks. | Experimental. | |

---

### 8. CLOSER MONITOR — 7 items

30-team closer depth chart grid with job security scoring.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| L1 | **gmLI Trust Tracking** | 2-week rolling gmLI vs season avg. Drop from 2.0+ to <1.5 = "Usage Downgrade." | 1-2 week earlier warning than blown saves. | DONE |
| L2 | **K% Skill Decay Alert** | Rolling 14-day K% vs season. K% drop ≥8 pts OR K-BB% <10% = "Skill Warning." | Catches decline via process, not outcomes. | DONE |
| L3 | **Committee Risk Score** | # pitchers with saves + max save share % + gmLI variance among top 3 RPs. | ~70% of teams have committee element. | DONE |
| L4 | **Opener Detection Flag** | Reliever pitches 1st inning >30% of appearances = "Opener" not "Closer." | Growing bullpen trend. | DONE |
| T5 | **DATA: Fetch gmLI per Reliever** | `pybaseball.pitching_stats()` includes gmLI. | Unlocks L1, L3. 30 min. | DONE |
| T6 | **DATA: Fetch Reliever Appearance Inning** | MLB Stats API game logs: inning of entry per appearance. | Unlocks L4. 30 min. | DONE |
| S3 | **CONSOLIDATE: Streaming Tab** | Lineup Optimizer Streaming tab duplicates Free Agents pitcher rankings. Convert to quick-view or remove. | Prevents different rankings in two places. | |

---

### 9. LEAGUE STANDINGS — 4 items

2 tabs: Current Standings, Season Projections. Includes power rankings.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| B7 | **Power Rankings Momentum Weight** | Increase momentum 10%→20%. Reduce roster quality 40%→30%. Add L3W trajectory factor. | Better reflects hot-streak value in H2H. | |
| R1 | **MC Simulation Upgrade to 10K** | Default 1,000→10,000 sims. Reduces playoff probability SE from ~1.5% to ~0.5%. | Standard is 10K+. | DONE |
| R2 | **Per-Category Variance Calibration** | Use quantified weekly SDs: K=1.2, R=1.6, W=1.7, SV=1.8, WHIP=2.0, HR=2.1, RBI=2.3, SB=2.3, ERA=2.2, AVG=2.9. | FanGraphs community research (48 leagues). | DONE |
| S5 | **CONSOLIDATE: Shared Standings Utilities** | `_roster_category_totals()` computed 8+ times independently. Extract `standings_utils.py` with session cache. | Eliminates redundant computation. Guarantees consistency. | DONE |

---

### 10. LEADERS — 7 items

3 tabs: Category Leaders, Category Value, Prospects.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| O1 | **Statcast Breakout Score** | Composite: EV50 (30%), barrel rate YoY (25%), xwOBA-wOBA (20%), bat speed (15%), launch angle (10%). Pitchers: SwStr% (30%), Stuff+ (25%), SIERA-ERA (25%), K-BB% (20%). >70th pctl = "Breakout." | Strongest buy-low signal. | DONE |
| O2 | **Prospect Fantasy Relevance Score** | Adjust FV by: ETA proximity, position scarcity in YOUR league, path to playing time, historical FV hit rate (55 FV = 67% regular). | Bridges scouting to fantasy timelines. | |
| O3 | **40-Man + Service Time Call-Up Alerts** | 40-man flag (highest signal). Days until Super Two. MLB IL cross-reference at prospect's position = "imminent." | First-mover advantage on prospect adds. | DONE |
| O4 | **Projection Skew Indicator** | When 5/7 systems above consensus = positive skew. Especially valuable mid-round pitchers (+50-100% ROI). | FanGraphs ATC Volatility research. | |
| O5 | **SGP Contribution Breakdown** | Wire N3 into Leaders table. Same function, different context. | Shows concentrated vs diversified value. | |
| T9 | **DATA: Fetch Bat Speed** | Scrape Baseball Savant bat tracking leaderboard. | Unlocks O1 bat speed component. 45 min. | |
| T10 | **DATA: Fetch 40-Man Roster Status** | MLB Stats API `/api/v1/teams/{id}/roster/40Man`. | Unlocks O3. 30 min. | |

---

### 11. PLAYER COMPARE — 5 items

Head-to-head z-score comparison, radar chart, health/confidence.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| N1 | **Category Fit Indicator** | Show which of YOUR team's weak categories each player helps vs wastes value in. | Transforms generic comparison into team-specific decision. | DONE |
| N2 | **Schedule Strength Comparison** | Next 2-4 weeks of opposing pitchers/matchups side-by-side. | Upcoming matchups matter enormously in H2H. | |
| N3 | **SGP Contribution Breakdown** | Stacked bar: concentrated (3.0 SGP from HR/RBI) vs diversified (0.6-0.8 across all). | Shows value profile instantly. Also wired into O5 for Leaders. | |
| E10 | **Catcher Framing Value** | Pitcher ERA differs 0.20-0.40 by catcher. 10-15 extra strikes/game for elite framers. | | |
| T8 | **DATA: Fetch Catcher Framing + Pop Time** | Scrape Baseball Savant catcher framing + pop time leaderboards. | Unlocks E10, J5, K1. 45 min. | |

---

### 12. DRAFT SIMULATOR — 4 items

3 tabs: Available Players, Draft Board, Pick Log.

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| Q1 | **Marginal SGP-Based AI Opponents** | AI picks highest marginal SGP to roster 70%, ADP-random 30%. | Currently AI picks near ADP. Real opponents pick for need. | |
| Q2 | **Position Run Detection** | 2+ consecutive same-position picks → boost remaining AI probability 1.3x for 3 picks. | Creates realistic draft dynamics. | |
| Q3 | **Per-Player ADP Standard Deviation** | `(max_pick - min_pick) / 4` per player from NFBC instead of global ADP/4. | Injury-risk players have wider distributions. | |
| Q4 | **Draft Value Standings Impact** | AI evaluates each pick's effect on projected end-of-draft standings, not just BPA. | DraftKick methodology. | |

---

### 13. CROSS-PAGE CONSOLIDATION — 3 remaining items

(S1, S2, S3, S4, S5 are listed under their respective pages above.)

| # | Item | Fix | Impact | Status |
|---|------|-----|--------|--------|
| S6 | **Deduplicate Roster Rendering** | Extract `render_roster_table(mode="overview"|"optimizer")` in `ui_shared.py`. | Shared code between My Team + Lineup Optimizer. | |

---

## Implementation Priority

**Tiers 1-3: ALL DONE** (Phases 1-3 + session commits)
All cascade, page-specific, and accuracy items complete except **A3** (Bayesian SGP updating).

**Tier 4 — Strategic features: MOSTLY DONE**
Remaining: **E9** (schedule-aware streaming), **K1** (SB streaming), **B1** (league-specific acceptance calibration), **D4** (opponent behavior learning), **Q1, Q2** (draft simulator AI).

**Tier 5 — Validation & polish: MOSTLY DONE**
Remaining: **N2** (schedule strength comparison), **O2** (prospect relevance), **P2** (antithetic variate MC).

**Remaining 46 items by page (audited April 8, 2026):**

Global (10): A3, B2, B3, B4, J1 (PARTIAL), J6, T4, V3, V4, V5
My Team (3): M2, U3, S4
Trade Finder (4): E1, H9, B1, D4
Trade Analyzer (2): P2, P3
Lineup Optimizer (3): J5, K1, S2
Matchup Planner (6): B5, C5, E6, E7, T11, T12
Free Agents (2): C1 (PARTIAL), D7
Closer Monitor (1): S3
League Standings (1): B7
Leaders (5): O2, O4, O5, T9, T10
Player Compare (4): N2, N3, E10, T8
Draft Simulator (4): Q1, Q2, Q3, Q4
Cross-page (1): S6
