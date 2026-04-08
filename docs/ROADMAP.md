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

**99 total items** (87 unique after deduplication) identified via algorithm audit, deep web research
(FanGraphs, Baseball Savant, behavioral economics, game theory, MIT Sloan, AMS, PMC), and
sabermetric literature. Updated April 7, 2026 to reflect Line-up Optimizer V3 (f795ea8).

### Status Key
- (empty) = Not started
- **PARTIAL** = Infrastructure exists, full wiring incomplete
- **DONE** = Fully implemented
- **DUPLICATE** = Merged with another item (see note)

### Deduplication Notes
- E4 (Pitcher Fatigue) + K2 (Pitcher Fatigue Multiplier) → merged as **K2** under Lineup Optimizer
- E5 (SB Prediction) + K1 (SB Streaming) → merged as **K1** under Lineup Optimizer
- H6 (Consistency Trade) + K3 (Consistency Lineup) → kept separate (different application contexts)
- G4 (Stat Reliability) + J1 (Per-Stat Update) → kept separate (trade YTD modifier vs projection blend)
- E8 (Punt Strategy) + H7 (Punt Advisor) + K4 (Punt Mode) → merged as **K4** under Lineup Optimizer (single feature, 3 consumers)
- C1 (Drop Penalties) → **PARTIAL** via f795ea8 FA recommender 5-factor drop scoring

### What f795ea8 (Line-up Optimizer V3) Completed
- ✅ Shared data layer loads confirmed lineups, weather, recent form, opposing pitcher, AVIS constraints
- ✅ Category correlation dampening in weight computation
- ✅ FA recommender with 5-factor drop scoring and AVIS hard constraints
- ✅ Recent form 25% blend in DCV daily decisions
- ✅ Unified scope router (Today/This Week/Rest of Season)

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
| H7 | **Proactive Punt Advisor** — see K4 | Same engine as K4, surfaced in Trade Readiness tab. Implement K4 first, then wire into Trade Finder. | Transforms Trade Finder from reactive to strategic. |
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
| I5 | **Batting Order Slot PA Adjustment** — PARTIAL | `LINEUP_SLOT_PA` dict + confirmed lineups loaded (f795ea8). Wire PA rates into DCV counting stats. | +5-10% daily accuracy. |
| I6 | **IP Management Mode** | "Chase K/W", "Protect Ratios", "Balanced" toggle. Auto-recommend from urgency state. | Prevents ratio destruction. |

#### Projection & Signal (J1-J6)

| # | Item | Fix | Impact |
|---|------|-----|--------|
| J2 | **Platoon Split Bayesian Regression** | `adjusted = (PA/stab) * individual + (1-PA/stab) * league_avg`. LHB stab=1000, RHB stab=2200. | Eliminates noise from raw individual splits. |
| J3 | **Opposing Pitcher Quality Scalar** — PARTIAL | Pitcher quality multiplier wired (f795ea8). Calibration to ±15% unverified. | Verify calibration against research magnitudes. |
| J4 | **Comprehensive Weather Model** — PARTIAL | Temp→HR works (f795ea8). Add rain→K/BB and wind direction→HR. | Rain K/BB effects currently ignored. |
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
| K6 | **Dynamic Volume from Confirmed Lineups** — PARTIAL | Confirmed lineups loaded (f795ea8). Wire slot PA rates into volume_factor. | More accurate DCV. |

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
| C1 | **Waiver Wire Drop Penalties** — PARTIAL | FA recommender 5-factor drop scoring added (f795ea8). DH/Util/AVG thresholds still hardcoded. | Derive from league context dynamically. |
| E9 | **Schedule-Aware Streaming** | Off-day streams more valuable. Two-start pitcher detection. Opponent L14 wRC+ for quality. | Data sources already wired. |
| E5 | **Stolen Base Prediction** — DUPLICATE of K1 | See K1 under Lineup Optimizer. | — |
| D7 | **Category-Aware Lineup RL** | Contextual bandit learning from weekly decisions + outcomes. Needs 8+ weeks of data. | Experimental. |

---

### LEAGUE STANDINGS & POWER RANKINGS — 2 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| B7 | **Power Rankings Momentum Weight** | Increase momentum from 10% to 20%. Reduce roster quality from 40% to 30%. | Better reflects hot-streak value in H2H. |
| E8 | **Punt Strategy Optimizer** — DUPLICATE of K4 | See K4 under Lineup Optimizer. Also feeds H7 in Trade Finder. | — |

---

### CLOSER MONITOR — 5 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E4 | **Pitcher Fatigue & Workload Model** — DUPLICATE of K2 | See K2 under Lineup Optimizer. | — |
| L1 | **gmLI Trust Tracking** | Track game-entry Leverage Index trend (2-week rolling vs season avg). gmLI drop from 2.0+ to <1.5 = "Usage Downgrade" alert 1-2 weeks before blown saves pile up. | Earlier role change detection than counting blown saves. FanGraphs LI research. |
| L2 | **K% Skill Decay Alert** | Rolling 14-day K% vs season average. K% drop ≥ 8 pts OR K-BB% < 10% = "Skill Warning" flag. More predictive of demotion than blown saves. | Catches closer decline via process, not outcomes. Athlon Sports 2025 case studies. |
| L3 | **Committee Risk Score** | Combine: (a) number of pitchers with saves on team, (b) max single-closer save share %, (c) gmLI variance among top 3 RPs. No pitcher >60% of saves AND 3+ with saves = committee. | Only 8-9 closers truly "secure" across MLB. ~70% of teams have some committee element. |
| L4 | **Opener Detection Flag** | When a reliever pitches 1st inning in >30% of appearances, flag as "Opener" not "Closer". Prevents false positives in closer identification. | Growing bullpen trend; prevents misidentifying openers as closers. |

---

### MY TEAM — 7 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E3 | **Umpire Strike Zone Adjustment** | Per-umpire K%/BB%/run environment. ±0.3-0.5 runs/game. | Needs new data source. |
| M1 | **Category Flip Analyzer** | Compute mid-week flip probability per category: `flip_prob = f(margin, daily_stdev, games_remaining)`. Show "categories within X% of flipping" as top-priority actions. | Most tools show scores but don't compute which categories are close enough to flip with a single roster move. |
| M2 | **Conditional Swap Impact** | "If you bench Player X and start Player Y, you gain +2 K and lose -0.8 ERA delta." Show roster-level marginal impact per swap, not just player stats. | Bridges player-level data to roster-level decision. Currently invisible. |
| M3 | **Opponent Lineup Tracking** | Surface opponent's mid-week roster moves: empty spots, pitcher streams, off-day players. Know what they're doing. | Almost never surfaced in fantasy tools. Uses existing Yahoo API. |
| M4 | **Regression Alert System** | Compare L30 actual stats to expected stats (xBA, xwOBA via barrel% + speed). Flag >1.5 SD divergence as sell-high or buy-low. Weight K%/BB% changes over BA. Require 50 PA minimum. | Process metrics (barrel%, K%) are more predictive than outcome metrics (BA, HR count). FanGraphs streak research. |
| M5 | **IL Slot Utilization Alert** | Empty IL slot = "Fill your IL slot with X". IL stash ranking sorted by `ros_sgp - best_fa_sgp` with days-to-return. Injury-type duration lookup (hamstring 21d, oblique 28d, TJ 14mo). | An empty IL slot is wasted value equivalent to a bench spot. IL-15 averages 18d actual, IL-60 averages 70d. |
| M6 | **Ratio Lock Alert** | Automated detection: "You are winning ERA by 0.80 and WHIP by 0.15 with 45 IP banked — bench Sunday starters to lock 2 categories." Compute marginal ERA/WHIP risk per remaining start. | Worth 1-2 category wins/season for managers who don't naturally think about this. |

---

### PLAYER COMPARE — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| E10 | **Catcher Framing Value** | 10-15 extra strikes/game for elite framers. Pitcher ERA differs 0.20-0.40 by catcher. | Needs new data source. |
| N1 | **Category Fit Indicator** | For each compared player, show which of YOUR team's weak categories they help and which strong/punt categories they waste value in. | Transforms generic comparison into team-specific decision tool. |
| N2 | **Schedule Strength Comparison** | Show next 2-4 weeks of opposing pitchers/matchups for each player side-by-side. | H2H: upcoming matchups matter enormously. No public comparison tool does this. |
| N3 | **SGP Contribution Breakdown** | Stacked bar showing how many SGP come from each category. Concentrated (3.0 SGP from HR/RBI) vs diversified (0.6-0.8 across all cats). In H2H, diversified is usually more valuable. | Instantly shows concentrated vs balanced value profiles. |

---

### LEADERS — 5 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| O1 | **Statcast Breakout Score** | Composite: EV50 percentile (30%), barrel rate change YoY (25%), xwOBA-wOBA gap (20%), bat speed (15%), launch angle change (10%). For pitchers: SwStr% (30%), Stuff+ (25%), SIERA-ERA gap (25%), K-BB% (20%). Flag >70th percentile as "Breakout Candidate." | xwOBA-wOBA gap predicts 40-80 pt wOBA surge. EV50 is stronger than raw exit velo. Barrel >8% = power threshold. |
| O2 | **Prospect Fantasy Relevance Score** | Adjust raw FV by: (a) ETA proximity (exponential decay), (b) position scarcity in YOUR league, (c) path to playing time (depth chart), (d) historical hit rate for FV tier (55 FV = 67% chance of regular). | Bridges scouting grades to actionable fantasy timelines. FanGraphs FV accuracy data. |
| O3 | **40-Man + Service Time Call-Up Alerts** | 40-man roster status flag (highest signal). Days until Super Two cutoff. MLB team injury cross-reference (IL at prospect's position = "imminent"). AAA performance triggers (OPS .900+, K/9 10+). | First-mover advantage on prospect adds worth 2-5 wins/season. Multiple call-up timing signals. |
| O4 | **Projection Skew Indicator** | When 5 of 7 projection systems are above consensus (positive skew), flag as better bet. Negative skew = cautionary flag. Especially valuable for mid-round pitchers (+50-100% ROI for positive skew). | FanGraphs ATC Volatility research (2019-2024): positive skew = $8 premium. |
| O5 | **SGP Contribution Breakdown** | Same as N3 but for leaders table — show concentrated vs diversified value per leader. | See N3. Applied to leaders context. |

---

### TRADE ANALYZER — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| P1 | **Trade Grade Confidence Interval** | Show grade RANGE (e.g., "B+ to A-") not just single letter. Based on projection uncertainty (+/- 1 SD). Ottoneu data: ~$5-10 surplus value range where trades are coin-flips. | More honest, better calibrated. Current single-letter grade implies false precision. |
| P2 | **Antithetic Variate MC Sampling** | For each of 10K sims, generate mirror sim using negated z-scores. Gives 20K paired observations for cost of 10K. Cuts standard error of MC mean by ~30%. | Free precision improvement. Academic MC research validates O(1/sqrt(N)) convergence. |
| P3 | **Phase 2 Copula Correlation Calibration** | Current copula uses hardcoded category correlations. Should use empirical correlations from your league's actual category data (league_standings weekly totals). | Ignoring correlations underestimates variance in total category wins by ~15%. |
| P4 | **Per-Category Replacement Level** | Subtract replacement-level stats per category per position (not aggregate SGP). Captures positional scarcity in specific categories (e.g., catcher replacement HR much lower than 1B replacement HR). | Smart Fantasy Baseball methodology: more accurate VORP computation. |

---

### DRAFT SIMULATOR — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| Q1 | **Marginal SGP-Based AI Opponents** | AI picks player with highest marginal SGP improvement to their roster 70% of the time, ADP-weighted random 30%. Responds to roster composition. | Currently AI picks near ADP. Real opponents pick for need. DraftKick methodology. |
| Q2 | **Position Run Detection** | When 2+ consecutive picks are same position, boost remaining AI teams' probability by 1.3x for next 3 picks. | Creates realistic draft dynamics. Position runs are a well-documented phenomenon. |
| Q3 | **Per-Player ADP Standard Deviation** | Use `(max_pick - min_pick) / 4` per player from NFBC data instead of global ADP/4. Injury-risk players have wider distributions. Pitchers wider than hitters. | More accurate pick survival probability. Currently uses fixed sigma. |
| Q4 | **Draft Value Standings Impact** | AI evaluates each candidate pick's effect on projected end-of-draft standings, not just BPA. | DraftKick methodology: best simulators model standings impact, not raw value. |

---

### LEAGUE STANDINGS — 4 items

| # | Item | Fix | Impact |
|---|------|-----|--------|
| B7 | **Power Rankings Momentum Weight** | Increase momentum from 10% to 20%. Reduce roster quality from 40% to 30%. Add "trajectory" factor (L3W trend). | Better reflects hot-streak value in H2H. |
| E8 | **Punt Strategy Optimizer** — DUPLICATE of K4 | See K4 under Lineup Optimizer. Also feeds H7 in Trade Finder. | — |
| R1 | **MC Simulation Upgrade to 10K** | Default from 1,000 to 10,000 sims. Reduces playoff probability SE from ~1.5% to ~0.5%. Add "high precision" mode. | At 1K sims, playoff odds have significant sampling error. Standard is 10K+. FanGraphs ZiPS uses 1M. |
| R2 | **Per-Category Variance Calibration** | Use quantified weekly SDs: K=1.2, R=1.6, W=1.7, SV=1.8, WHIP=2.0, HR=2.1, RBI=2.3, SB=2.3, ERA=2.2, AVG=2.9. Widen confidence intervals for ERA/AVG (ratio stats). | FanGraphs community research (48 leagues). Currently uses hardcoded WEEKLY_TAU. |

---

## Summary by Page

| Page | Items | Top Priority |
|------|-------|-------------|
| **Global / Core Engine** | 14 | A1 (dynamic SGP), A4 (ECR), E2 (Stuff+), A2+D3 (stacking) |
| **Trade Finder** | 24 | G1 (xwOBA), G4 (reliability), F5 (playoff-odds), G2 (Stuff+) |
| **Lineup Optimizer** | 20 | I1 (mid-week pivot), J1 (update rates), I2 (swing-category), I3 (ratio protection) |
| **My Team** | 7 | M1 (category flip), M4 (regression alerts), M6 (ratio lock), M2 (swap impact) |
| **Leaders** | 5 | O1 (Statcast breakout score), O3 (call-up alerts), O2 (prospect relevance) |
| **Closer Monitor** | 5 | L1 (gmLI tracking), L2 (K% decay), L3 (committee risk score) |
| **Matchup Planner** | 4 | B5 (dynamic park), E6 (weather wind) |
| **Trade Analyzer** | 4 | P1 (grade confidence interval), P2 (antithetic MC), P4 (per-cat replacement) |
| **Draft Simulator** | 4 | Q1 (marginal SGP AI), Q2 (position run detection), Q3 (per-player ADP SD) |
| **League Standings** | 4 | R1 (10K sims), R2 (per-category variance), B7 (momentum) |
| **Free Agents / Waiver** | 4 | E9 (streaming), E5 (SB prediction) |
| **Player Compare** | 4 | N1 (category fit), N2 (schedule comparison), N3 (SGP breakdown) |
| **Total** | **99** (87 unique after dedup) | |

---

## Implementation Priority (Cross-Page)

**Tier 1 — Cascade everywhere (do first):**
1. A1: Dynamic SGP denominators
2. A4: Remove self-referential ECR
3. E2: Populate Stuff+/Location+/Pitching+
4. A2+D3: Weighted projection stacking
5. J1: Per-stat in-season update rates

**Tier 2 — Biggest page-specific wins:**
6. I1: Mid-week pivot advisor (Lineup Optimizer)
7. G1: xwOBA regression flags (Trade Finder)
8. M1: Category flip analyzer (My Team)
9. O1: Statcast breakout score (Leaders)
10. D1: Statcast XGBoost regression (Global)
11. D2: Playing time prediction (Global)
12. I2: Swing-category weighting (Lineup Optimizer)
13. L1: gmLI closer trust tracking (Closer Monitor)

**Tier 3 — Accuracy refinements:**
14-25: A3, G4, F5, I3, J2, J3, J4, P1, R1, R2, L2, M4

**Tier 4 — Strategic features:**
26-40: K4, H7, E9, K1, H8, I4, E5, B1, D4, Q1, Q2, N1, O3, M6, P4

**Tier 5 — Validation & polish:**
41-50: D6, B6, B8, I6, K2, M5, L3, N2, O2, P2

**Remaining:** All other items — as time allows.
