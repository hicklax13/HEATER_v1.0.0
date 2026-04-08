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

## Infeasible Items (Free API Constraint)

| Item | Reason | Proxy |
|------|--------|-------|
| PECOTA | Baseball Prospectus paywall | Marcel local computation |
| ESPN ADP | API restricted | FantasyPros ECR (aggregates ESPN) |
| Rotowire injury feed | Paywall | MLB Stats API injuries |
| Individual expert rankings | Paywalls | FantasyPros ECR consensus (100+ experts) |

---

## Accuracy & Improvement Backlog — Organized by Page

**71 total items** identified via algorithm audit, deep web research (FanGraphs, Baseball Savant,
behavioral economics, game theory, MIT Sloan, American Meteorological Society, PMC), and
sabermetric literature. Each item includes research basis, quantified impact, and data requirements.

Legend: Items that affect ALL pages are listed under "Global / Core Engine" first.

---

### GLOBAL / CORE ENGINE (affects every page downstream) — 14 items

These cascade through all 53 engines. Highest leverage.

| # | Item | Fix | Impact | Effort |
|---|------|-----|--------|--------|
| A1 | **Dynamic SGP Denominators** | Compute from `league_standings`: `denom = (max - min) / (n_teams - 1)`. Update weekly. | Propagates to EVERY engine. Single highest-leverage fix. | Low |
| A2+D3 | **Weighted Projection Stacking** | Ridge regression stacking model (each system = feature, 2025 actuals = target). Learned weights > naive average. | ~5-8% projection improvement. Cascades everywhere. | Low |
| A3 | **Bayesian SGP Updating** | Start with preseason denoms as prior, update weekly from standings. By week 8: 70% actual / 30% prior. | Prevents stale denominators mid-season. | Medium |
| A4 | **Remove Self-Referential ECR** | Remove HEATER SGP rank as source #7 in ECR consensus. Use 6 external sources only. | Removes circular confirmation bias. | Trivial |
| B2 | **Position-Specific Health Scoring** | C: threshold 28/0.03yr. DH: 34/0.01yr. OF: 31. | Catchers age faster than DHs. | Low |
| B3 | **Injury-Type Adjustment** | TJ = 0.4 health floor 2yr. Hamstring = 0.7 1yr. Concussion = 0.6. | 30 IL days from bone bruise ≠ 30 IL days from TJ. | Medium |
| B4 | **Temporal ECR Weighting** | Exponential decay 14-day half-life. March rank = 0.25x weight vs April = 1.0x. | Stale preseason ranks devalued. | Low |
| D1 | **Statcast XGBoost Regression** | Train on EV, barrel, xwOBA, xBA, sprint speed. Target: actual − projected. Retrain daily. | ~10-15% projection improvement. | Medium |
| D2 | **Playing Time Prediction** | Ridge regression: `remaining_PA = f(recent_PA_rate, depth_chart, health, age)`. Update weekly. | ~10-15% counting stat accuracy. #1 projection error source. | Medium |
| D5 | **NLP News Sentiment** | Pre-trained DistilBERT. 3 lines, <100ms/headline. | ~3-5% trade/waiver timing. | Low |
| D6 | **Backtesting Framework** | Replay past weeks, score engine recommendations vs actual outcomes. | Meta: validates ALL other changes. | Medium-High |
| E2 | **Populate Stuff+/Location+/Pitching+** | DB columns exist but empty. Populate via `pybaseball pitching_stats()`. | Transforms pitcher evaluation across all pages. | Low |
| J1 | **Per-Stat In-Season Update Rates** | `current_weight[stat] = PA / stabilization[stat]`. K% at 60, AVG at 910, BABIP at 820. | Highest-impact projection fix. K% updates 15x faster than AVG. | Medium |
| J6 | **Projection Uncertainty Bands** | Use empirical SDs (ERA=1.20, WHIP=0.20, AVG=0.025) for P10/P50/P90. | Prevents over-optimizing on noise. | Low |

---

### TRADE FINDER (5 tabs: Smart Recommendations, By Value, Target a Player, Browse Partners, Trade Readiness) — 24 items

#### Acceptance Model (F1-F5)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| F1 | **Nash Bargaining Fairness** | `nash_fairness` where alpha = bargaining power from standings/FA alternatives/weeks remaining. | Theoretically grounded acceptance metric. |
| F2 | **Draft-Round Anchoring** | Rd 1-3 = 1.3x endowment, Rd 4-8 = 1.15x, Rd 9+ = 1.0x on opponent perceived value. | 15-30% more realistic acceptance for early-round players. |
| F3 | **Disposition Effect** | YTD > projection: +10-15% acceptance. YTD < projection: -10-15%. | Captures #1 behavioral rejection pattern. |
| F4 | **Recently-Acquired Penalty** | Traded-for within 3 weeks: -10-15% acceptance. | Uses existing Yahoo transaction data. |
| F5 | **Playoff-Odds Acceptance** | Replace raw standings rank with `simulate_season_enhanced()` playoff odds. <15% = 0.5x, 30-70% = 1.2x, >85% = 0.7x. | More precise than rank alone. |
| B1 | **League-Specific Acceptance Calibration** | Compute empirical acceptance rates from Yahoo transactions (SGP/ADP gaps vs accepted/rejected). | League-calibrated, not generic. |
| D4 | **Opponent Behavior Learning** | Per-opponent logistic regression on trade history. | ~10-20% acceptance improvement. |

#### Regression Signals (G1-G5)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| G1 | **xwOBA-wOBA Regression Flags** | Gap ≥0.030 = BUY_LOW, ≤-0.030 = SELL_HIGH. +0.05 composite bonus. | Single most actionable regression signal. |
| G2 | **Stuff+/botERA Pitcher Regression** | `stuff_delta = botERA - actual_ERA`. ≤-0.50 = BUY_LOW, ≥+0.50 = SELL_HIGH. | DB columns exist — just populate. |
| G3 | **BABIP Regression Scoring** | `(career_BABIP - current) / 0.020`, capped ±2.0. | Best early-season signal (weeks 3-8). |
| G4 | **Stat Reliability Weighting** | `reliability = min(1.0, PA / threshold[stat])`. Trust K% at 60 PA ≠ AVG at 60 PA. | Prevents over/under-reacting to samples. |
| G5 | **Velocity Trend Signal** | 1.0 mph decline ≈ 0.5-0.8 ERA increase. `velo_delta = current - prior`. | Early warning before ERA shows it. |
| E1 | **BABIP Regression Targets** | xBABIP from contact quality + sprint speed. Predict future BABIP within 15 points. | Buy-low on unlucky hitters. |

#### Strategic & Structural (H1-H9)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| H1 | **Differential Time Decay** | Counting stats × `weeks_rem / total_weeks`. Rate stats stay 1.0 with confidence penalty below 8 weeks. | More accurate ROS valuation. |
| H2 | **Dynamic Roster Spot Value** | `ROSTER_SPOT_SGP = median_fa_sgp * 0.8`, recomputed weekly. | Fixes 2-for-1 valuation. |
| H3 | **Graduated Positional Scarcity** | C=1.20x, 2B=1.15x, SS=1.10x, 3B=1.05x, OF/1B=1.00x. | More accurate than flat 1.15x. |
| H4 | **SB Independence Premium** | Increase from 1.08x to 1.12-1.15x (R²=0.0002 with RBI). | SB contributors undervalued. |
| H5 | **Trade Timing Multiplier** | Weeks 1-4 = ±5% YTD clamp, weeks 5-8 = ±10%, weeks 9+ = ±15%. 1.1x bonus weeks 4-10. | Prevents chasing April hot streaks. |
| H6 | **Consistency/Variance Modifier** | Weekly CV from game logs. CV<0.30 = +5-8% SGP, CV>0.50 = -5-8%. | H2H dimension invisible today. |
| H7 | **Proactive Punt Advisor** | `recommend_punt_targets()` by mean pairwise correlation. SB (r=0.12) and W (R²=0.03) safest. | Transforms Trade Finder from reactive to strategic. |
| H8 | **Closer Stability Discount** | `sv_adjusted = sv_proj * (confidence / 100)`. Wire closer_monitor into trade valuation. | Prevents overvaluing shaky closers. |
| H9 | **Prospect Call-Up Valuation** | `value = P(call_up) * ROS_SGP * (weeks_avail / weeks_rem) - ROSTER_SPOT_SGP`. | Better than binary stash/zero. |
| B8 | **Validate Trade Finder Weights** | Backtest: top-ranked trades vs actual improvement by week 24. | Optimize for improvement, not acceptance. |
| C4 | **Trade Readiness Normalization** | Compute actual SGP range dynamically. | Cosmetic. |

---

### LINEUP OPTIMIZER (6 tabs: Start/Sit, Optimize, Manual, H2H, Streaming, Daily Optimize) — 20 items

#### Weekly & Daily Optimization (I1-I6)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| I1 | **Mid-Week Pivot Advisor** | Classify categories WON (P>85%), LOST (P<15%), CONTESTED (30-70%). Bench pitchers to protect WON ratios, stream for CONTESTED. | +1-2 category wins/week. |
| I2 | **Swing-Category Weighting** | Maximize P(win) for 30-70% categories in LP objective. De-emphasize locked/lost categories. | +0.5-1 category wins/week. |
| I3 | **Ratio Protection Calculator** | `marginal_era_risk = (proj_ER * 9) / (banked_IP + proj_IP) - current_ERA`. Bench if risk > lead. | Protects ratio categories. |
| I4 | **Category Flip Probability** | `flip_prob = f(margin, daily_stdev, games_remaining)`. Target flippable losses on final day. | Critical for close matchups. |
| I5 | **Batting Order Slot PA Adjustment** | PA rates: 4.65 leadoff, 3.77 9th. Wire confirmed lineups into daily projections. | +5-10% daily accuracy. |
| I6 | **IP Management Mode** | "Chase K/W", "Protect Ratios", "Balanced" toggle. Auto-recommend from urgency state. | Prevents ratio destruction. |

#### Projection & Signal (J1-J6)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| J2 | **Platoon Split Bayesian Regression** | `adjusted = (PA/stab) * individual + (1-PA/stab) * league_avg`. LHB stab=1000, RHB stab=2200. | Eliminates noise from raw individual splits. |
| J3 | **Opposing Pitcher Quality Scalar** | `mult = 1.0 + 0.15 * (league_xFIP - opp_xFIP) / std_xFIP`. Calibrate to ±15%. | Better Start/Sit and streaming decisions. |
| J4 | **Comprehensive Weather Model** | Rain >40%: BB ×1.10, K ×0.90. Wind out >10 mph: HR ×1.15-1.20. Full temp scalar. | Rain K/BB effects currently ignored. |
| J5 | **Catcher Framing Pitcher Adjustment** | `era_adj = -0.01 * framing_runs`, `k9_adj = 0.025 * framing_runs`. | ±0.25 ERA per start for elite/poor framers. |
| B6 | **Category Urgency k-Calibration** | Backtest k=1.0 to 5.0, measure category wins. | Current k=2/3 may be off. |
| C2 | **Dual Objective Alpha Validation** | Tie alpha to playoff probability, not just time remaining. | Affects strategy, not individual decisions. |
| C3 | **Start/Sit Home Advantage** | Update 1.02 to 1.015. | Very small effect. |

#### H2H Strategic Features (K1-K6)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| K1 | **SB Streaming by Catcher/Pitcher** | `sb_score = sprint_speed * (pop_time/1.95) * (delivery_time/1.35) * handedness`. | Targeted SB streaming. |
| K2 | **Pitcher Fatigue Multiplier** | `fatigue = max(0.85, 1.0 - 0.003 * max(0, IP - 100))`. ACWR >1.3 flag. Exempt top Stuff+. | Prevents over-projecting fatigued arms. |
| K3 | **Consistency Premium** | `penalty = k * weekly_CV`. k=0.05-0.10. Penalize volatile, reward consistent. | H2H-specific dimension. |
| K4 | **Punt Mode Optimizer** | User selects 0-2 categories to punt → weight 0.0 in LP. | Strategy-aware optimization. |
| K5 | **Streaming Composite Score** | `K_proj * (1/opp_wOBA) * park * form_L3 * whip_safety`. Career WHIP >1.40 = avoid. | Better streaming picks. |
| K6 | **Dynamic Volume from Confirmed Lineups** | Leadoff volume=4.65/4.3, 9th=3.77/4.3. Not in lineup=0.0 (not 0.3). | More accurate DCV. |

---

### MATCHUP PLANNER — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| B5 | **Dynamic Park Factors** | Fetch updated park factors mid-season via pybaseball. | Some parks change year-to-year. |
| C5 | **Inverse Park Formula** | Use `1.0 / park_factor` (reciprocal) instead of `2.0 - park_factor` (linear) for pitchers. | Tiny difference except Coors. |
| E6 | **Enhanced Weather (Wind Direction)** | Wind direction relative to outfield orientation per stadium. +20% HR wind blowing out >10 mph. | Currently HR-only, no wind direction. |
| E7 | **Pitcher-Batter Matchup History** | PvB stats stabilize at ~60 PA. Add 2-4% prediction value over generic platoon. | High data volume. |

---

### FREE AGENTS / WAIVER WIRE — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| C1 | **Waiver Wire Drop Penalties** | DH penalty scales with Util slots. AVG threshold = league-average AVG (not hardcoded 0.245). | Edge-case drop decisions. |
| E9 | **Schedule-Aware Streaming** | Off-day streams more valuable. Two-start pitcher detection. Opponent L14 wRC+ for quality. | Data sources already wired. |
| E5 | **Stolen Base Prediction** | Sprint speed + catcher pop time + pitcher handedness. 75% accuracy. | SB is most volatile counting stat. |
| D7 | **Category-Aware Lineup RL** | Contextual bandit learning from weekly decisions + outcomes. Needs 8+ weeks of data. | Experimental. |

---

### LEAGUE STANDINGS & POWER RANKINGS — 2 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| B7 | **Power Rankings Momentum Weight** | Increase momentum from 10% to 20%. Reduce roster quality from 40% to 30%. | Better reflects hot-streak value in H2H. |
| E8 | **Punt Strategy Optimizer** | Optimal punt selection by category isolation (mean pairwise correlation). SB + W safest punt combo. | Strategic standings intelligence. |

---

### CLOSER MONITOR — 1 item

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E4 | **Pitcher Fatigue & Workload Model** | Velocity trend + pitch count accumulation + ACWR. Predict late-season decline. | Closer role instability prediction. |

---

### MY TEAM — 1 item

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E3 | **Umpire Strike Zone Adjustment** | Per-umpire K%/BB%/run environment. ±0.3-0.5 runs/game. | Needs new data source. |

---

### PLAYER COMPARE — 1 item

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E10 | **Catcher Framing Value** | 10-15 extra strikes/game for elite framers. Pitcher ERA differs 0.20-0.40 by catcher. | Needs new data source. |

---

## Summary by Page

| Page | Items | Top Priority |
|------|-------|-------------|
| **Global / Core Engine** | 14 | A1 (dynamic SGP), A4 (ECR), E2 (Stuff+), A2+D3 (stacking) |
| **Trade Finder** | 24 | G1 (xwOBA), G4 (reliability), F5 (playoff-odds), G2 (Stuff+) |
| **Lineup Optimizer** | 20 | I1 (mid-week pivot), J1 (update rates), I2 (swing-category), I3 (ratio protection) |
| **Matchup Planner** | 4 | B5 (dynamic park), E6 (weather wind) |
| **Free Agents / Waiver** | 4 | E9 (streaming), E5 (SB prediction) |
| **Standings / Rankings** | 2 | B7 (momentum), E8 (punt optimizer) |
| **Closer Monitor** | 1 | E4 (fatigue model) |
| **My Team** | 1 | E3 (umpire zones) |
| **Player Compare** | 1 | E10 (catcher framing) |
| **Total** | **71** | |

---

## Implementation Priority (Cross-Page)

**Do first (cascade everywhere):**
1. A1: Dynamic SGP denominators
2. A4: Remove self-referential ECR
3. E2: Populate Stuff+/Location+/Pitching+
4. A2+D3: Weighted projection stacking
5. J1: Per-stat in-season update rates

**Do second (biggest page-specific wins):**
6. I1: Mid-week pivot advisor (Lineup Optimizer)
7. G1: xwOBA regression flags (Trade Finder)
8. D1: Statcast XGBoost regression (Global)
9. D2: Playing time prediction (Global)
10. I2: Swing-category weighting (Lineup Optimizer)

**Do third (accuracy refinements):**
11-20: A3, G4, F5, I3, J2, J3, J4, E1, G2, I5

**Do fourth (strategic features):**
21-30: K4, H7, E9, K1, H8, I4, E8, E5, B1, D4

**Do fifth (validation):**
31-35: D6, B6, B8, I6, K2

**Remaining:** All other B/C/D/E/F/G/H/I/J/K items — as time allows.
