# Lineup Optimizer V2 — Deep Research Findings

**Date:** 2026-03-31
**Scope:** Comprehensive research across 5 domains for H2H Categories lineup optimization
**League:** 12-team, 6x6 (R/HR/RBI/SB/AVG/OBP + W/L/SV/K/ERA/WHIP), daily lineups

---

## Research Agent 1: Academic Optimization + Projection Blending + Stabilization

### Key Academic Papers
- **arXiv 2411.11012** — LP approach to DFS baseball optimization
- **MIT Sloan (Hunter/Vielma/Zaman)** — Integer programming for DFS, solved in <4 min
- **arXiv 2501.00933** — Rotisserie/category optimization: V = Phi(mu_D / sigma_D), rewards balanced teams
- **PyMC Labs Bayesian Marcel** — Data-estimated weights came out 6/2/1 (stronger recency than 5/4/3)

### Marcel Projection System
- Weights: 5/4/3 for years T, T-1, T-2
- Regression: r = PA / (PA + 1200) for batters
- Age adjustment: Under 29 = +0.6%/year, Over 29 = -0.3%/year (peak at 29)

### In-Season Blending (MGL/FanGraphs Critical Finding)
- Season-to-date stats provide virtually NO additional info beyond preseason projection until last 1-2 months
- At 110 PA (1 month), >40% of hitters hit 40+ points above/below projection — yet projections remain more accurate
- Even at 413 PA (5 months), projections still carry ~44% weight in optimal blend
- ZiPS finding: In September, optimal blend = 56% actual + 44% preseason

### Month-by-Month Optimal Blend Ratios
| Time Point | Preseason Proj | In-Season |
|---|---|---|
| Opening Day (~0 PA) | 100% | 0% |
| End of April (~100 PA) | 80% | 20% |
| End of May (~200 PA) | 65% | 35% |
| End of June (~300 PA) | 55% | 45% |
| End of July (~400 PA) | 48% | 52% |
| End of August (~500 PA) | 44% | 56% |
| September (~550 PA) | 40% | 60% |
| **Current (Week 1, ~20 PA)** | **95%** | **5%** |

### Stabilization Thresholds (Plate Appearances)
| Stat | Stabilization PA | Implication |
|---|---|---|
| K Rate | 60 PA | Trust early — 2-3 weeks |
| Contact Rate | 100 PA | ~1 month |
| BB Rate | 120 PA | ~1 month |
| HR Rate | 170 PA | ~6 weeks |
| SB Rate | 200 PA | ~2 months |
| OBP | 460 PA | ~3 months |
| AVG | 910 AB | NEVER trust in one season |
| BABIP | 820 BIP | 2+ seasons |

### Pitcher Stabilization (Batters Faced)
| Stat | Stabilization BF |
|---|---|
| K Rate | 70 BF (~3 starts) |
| GB/FB Rate | 70 BIP |
| BB Rate | 170 BF (~7 starts) |
| HR Rate | 1320 BF (2+ seasons) |

### H2H Categories Strategy
- Need to win 7/12 categories, not maximize total SGP
- Category correlations: HR-RBI (r=0.86), AVG-OBP (r=0.70), SB nearly independent (r<0.20)
- Best punt candidates: SB (independent), SV (role-dependent), L (uncontrollable)
- H2H marginal value is S-shaped (logistic), not linear like Roto SGP
- Optimal: weight categories by closeness to opponent's score

---

## Research Agent 2: Matchup Factors + Park Effects

### Platoon Splits
- LHB vs RHP: +28 wOBA points average
- RHB vs LHP: +16 wOBA points average
- Pitcher arm angle amplifies: sidearm RHP = 76-point split vs LHB

### Opposing Pitcher Quality
- Elite vs replacement pitcher gap: 60-80 wOBA points
- Pitchers allowing .350+ wOBA = most exploitable
- DFS pros avoid hitters facing pitchers with K rate >25%
- Adjustment formula: Weight = (2 - xFIP-/100)^2 for elite pitchers

### Park Factors (by stat, not single multiplier)
- Coors: +26% HR, +16% singles, +120% triples
- Oracle Park: HR factor 74 but hits factor 103
- Use separate multipliers for HR, R, hits — NOT a single number

### Weather Impact
- Per 1.8F increase: +1.96% HR per game
- Wrigley wind blowing out: +50% HR, +1.7 runs/game
- Home Run Forecast index: 9-10 rating = 2.61 HR/game vs 1-2 rating = 1.40 HR/game
- Weather adjustment: +/- 5-15% on projections for that day

### Batting Order Position
- #3 hitter: ~170 combined R+RBI per season
- #7 hitter: ~123 combined R+RBI per season
- Difference: ~0.29 R+RBI per game between #3 and #7 spot
- Tiebreaker value: prefer hitters batting #1-#4

### Home/Away Splits
- Home: +.006 AVG, +5% HR rate, fewer K, more BB
- Small effect — +/- 2-3% adjustment, mostly captured by park factor

### Recommended Adjustment Magnitudes (ranked)
1. Park factors (HR-specific): +/- 5-30% — HIGH priority
2. Weather (wind + temp): +/- 5-20% — HIGH priority
3. Opposing pitcher quality: +/- 5-25% — HIGH priority
4. Platoon splits (L/R): +/- 5-8% — MEDIUM priority
5. Batting order position: +/- 0.3 R+RBI/game — MEDIUM tiebreaker
6. Home/away: +/- 2-3% — LOW, mostly captured by park factor
7. BvP history: only with 50+ PA — LOW, tiebreaker only

---

## Research Agent 3: Trend Detection + Recency Weighting

### Hot/Cold Streak Predictiveness
- Stanford study: last 25 AB is statistically significant predictor — 25-30 OBP points, 30% more HR likely
- FanGraphs enigma: P(Hot|Hot) = 24.8% vs 21.7% baseline — only 3 percentage points of edge
- Razzball DFS: 3-day and 5-day performance = zero predictive improvement over projections
- Pitcher streaks: stronger evidence (Baris & Losak 2026) — hot pitchers are underpriced in DFS
- **Recommended weight for streaks: 5-15% of total signal**

### Optimal Rolling Windows (stat-specific)
- 7-day window: Plate discipline only (K%, BB%, contact rate)
- 14-day window: Power metrics (barrel rate, exit velo, HR/FB)
- 30-day window: Rate stats (AVG, OBP, ERA, WHIP)
- Season-to-date: BABIP, line drive rate, luck-dependent stats

### Statcast as Leading Indicators
- Barrel rate explains 73% of variance in HR/FB% (r-squared 0.727) — most predictive metric
- EV on fly balls/line drives: r-squared 0.618 for HR/FB
- Year-to-year stickiness: EV on FB/LD has r-squared 0.667
- xwOBA vs actual wOBA converges at ~250 AB
- Below 250 AB: xwOBA is MORE indicative of true talent than actual wOBA

### Statcast Stabilization
| Metric | Stabilization (batted balls) | Games |
|---|---|---|
| Exit Velocity | ~50 BIP | ~18 games |
| Launch Angle | ~50 BIP | ~18 games |
| Barrel Rate | ~100 BIP | ~36 games |
| Hard-Hit Rate | ~100 BIP | ~36 games |

### Injury/Workload Effects
- No universal recovery timeline from IL
- Pitcher short rest (4 vs 5 days): +0.06 ERA penalty
- Fatigue units: back-to-back appearances = 5x fatigue multiplier
- Hitter bat speed after rest day: +1.04 mph (fatigued players) = ~5 wOBA points

---

## Research Agent 4: Pitching Strategy + Streaming

### Start/Sit Decision Factors (ranked)
1. Pitcher xFIP/SIERA (30-35% weight) — better predictor than ERA
2. Opponent wOBA (20-25% weight)
3. Park factor (15% weight)
4. Pitcher K/9 (10-15% weight)
5. Win probability based on team R/G (10% weight)
6. Matchup context modifier (winning/losing ERA/WHIP adjusts threshold)

### Win Probability Formula
```
P(Win) = 0.7 * (0.112 * team_R/G - 0.105 * pitcher_ERA + 0.446)
```
R-squared = 0.827 — very strong model

### ERA Estimator Hierarchy (predictive power)
1. SIERA (RMSE 0.964, best year-to-year predictor)
2. xFIP (RMSE 0.983)
3. ERA (RMSE 1.001)
4. FIP (RMSE 1.010)

### Save Opportunity Model
```
Weekly saves = games/week * team_win% * P(save_situation|win) * conversion_rate
```
Where P(save_situation|win) is approximately 0.58-0.62

### Streaming Decision Rules
- Team ERA < 3.50: Only stream xFIP < 4.00 arms
- Team ERA 3.50-4.00: Stream two-start pitchers with xFIP < 4.50
- Team ERA > 4.00: Stream freely — ratios already lost, maximize K and W

### Rate Stat Protection vs Volume
- Winning ERA/WHIP comfortably: bench marginal starters
- Losing ERA/WHIP badly: start everyone (volume can't hurt further)
- ERA/WHIP close: start only favorable-matchup pitchers

---

## Research Agent 5: Free Agent Evaluation + Opportunity Cost

### Replacement Level by Position (12-team league)
| Position | Scarcity | Replacement Quality |
|---|---|---|
| C | Highest | Must produce >75% of league avg |
| SS | High | Must produce >78% |
| 2B/3B | Medium | Must produce >80% |
| OF/1B/DH | Low | Must produce >85% |

### Category Streamability (easiest to hardest via FA)
SV > K > W > SB > R >> HR > RBI > AVG > OBP > ERA > WHIP

### Key Decision Rules
- Hot/cold streaks have ~5 wOBA points of predictive value (negligible)
- Use xwOBA + barrel rate to identify real decline vs bad luck
- FA pickup signal: ROS projection > rostered player ROS projection (NOT recent stats)
- Ownership velocity (rate of change) > absolute ownership %
- Optimal transactions: ~40/season, 2-4/week, with 1-2 designated churn slots

### Daily Lineup Rules
1. Playing player > not-playing player ALWAYS (zero production is always worse)
2. Check real MLB lineups 30 min before first pitch
3. For rate stats: sometimes FEWER AB/IP is correct if winning the rate category
4. For counting stats: ALWAYS maximize volume
5. Optimize for THIS WEEK against THIS OPPONENT, not season-long

### Volume Advantage Formula
```
Weekly PA Edge = (your_active_hitter_days - opponent_active_hitter_days) * avg_PA_per_game
Category Edge ~= PA_Edge * league_avg_rate
```
14 extra hitter-games = ~56 extra PA = ~3 R, 1 HR, 3 RBI, 0.5 SB per week

---

## Sources (100+)

### Academic Papers
- https://arxiv.org/abs/2411.11012
- https://mitsloan.mit.edu/shared/ods/documents?DocumentID=2858
- https://arxiv.org/html/2501.00933
- https://arxiv.org/html/2412.19215v1
- https://onlinelibrary.wiley.com/doi/10.1111/itor.13344
- http://www.columbia.edu/~mh2078/DFS_Revision_1_May2019.pdf
- https://projecteuclid.org/journals/bayesian-analysis/volume-4/issue-4

### Projection Systems
- https://www.baseball-reference.com/about/marcels.shtml
- https://www.pymc-labs.com/blog-posts/bayesian-marcel
- https://mglbaseball.wordpress.com/2014/06/12/what-can-a-players-season-to-date-performance-tell-us-beyond-his-up-to-date-projection/
- https://tht.fangraphs.com/the-math-of-weighting-past-results/
- https://blogs.fangraphs.com/yes-march-projections-matter-in-september/

### Stabilization + Sample Size
- https://library.fangraphs.com/principles/sample-size/
- https://blogs.fangraphs.com/when-samples-become-reliable/
- https://blogs.fangraphs.com/randomness-stabilization-regression/
- https://blogs.fangraphs.com/a-long-needed-update-on-reliability/

### Matchup Factors
- https://library.fangraphs.com/principles/split/
- https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
- https://library.fangraphs.com/principles/park-factors/
- https://www.homerunforecast.com/

### Statcast
- https://pitcherlist.com/going-deep-the-real-value-of-statcast-data-part-i/
- http://varianceexplained.org/r/hierarchical_bayes_baseball/
- https://baseballsavant.mlb.com/savant-player/

### Strategy
- https://www.smartfantasybaseball.com/2013/03/create-your-own-fantasy-baseball-rankings-part-5-understanding-standings-gain-points/
- https://www.fantasypros.com/2018/02/punting-categories-fantasy-baseball/
- https://razzball.com/heads-up-2020-h2h-categories-strategy-punt-away/
- https://fantasy.fangraphs.com/estimating-wins-using-era-and-run-support/

---

## CORRECTED IMPLEMENTATION FRAMEWORK (User-Approved)

### Master Formula

```
DCV(player, category, day) =
    Blended_Projection(player, category)
    * Matchup_Multiplier(player, today)
    * Health_Factor(player)
    * Volume_Factor(player, today)
```

### Health Factor (CORRECTED)

IL/DTD/NA players are **EXCLUDED entirely** (multiplier = 0). They cannot produce fantasy points. The only players who get a Health Factor are ACTIVE players:

| Status | Multiplier | Action |
|---|---|---|
| Active, healthy | 1.00 | Eligible for lineup |
| DTD | **0.00** | **EXCLUDE from lineup pool** |
| IL10 | **0.00** | **EXCLUDE from lineup pool** |
| IL15 | **0.00** | **EXCLUDE from lineup pool** |
| IL60 | **0.00** | **EXCLUDE from lineup pool** |
| NA/Minors | **0.00** | **EXCLUDE from lineup pool** |
| Returning from IL (first 3 games after activation) | 0.85 | Eligible but discounted |
| High fatigue (pitcher, back-to-back) | 0.80 | Eligible but discounted |

The 3-year injury history health score adjusts the **ROS projection** (upstream), not the daily lineup decision. A player with injury-prone history gets lower projected stats, which naturally reduces their DCV.

### Volume Factor (CORRECTED)

The lineup availability factor transitions from 0.9 to 1.0 when the real MLB lineup card is posted (typically 2-3 hours before game time):

| Situation | Factor | When |
|---|---|---|
| In today's confirmed starting lineup | **1.00** | After lineup card posted |
| Team plays today, lineup NOT yet posted | **0.90** | Before lineup card posted |
| Team plays today, lineup posted, player NOT in it | **0.30** | Pinch-hit possibility only |
| Team has off-day | **0.00** | Zero production guaranteed |

**Rule:** Once lineups are posted, re-run the optimizer with updated Volume Factors. A player confirmed in the lineup jumps from 0.90 to 1.00; a player unexpectedly benched drops to 0.30.

### Blended Projection (stat-specific stabilization)

```
Blended = (Preseason_Rate * Stab_PA + Observed_Rate * Actual_PA) / (Stab_PA + Actual_PA)
```

| Category | Stabilization PA | At 20 PA (week 1): Preseason Weight |
|---|---|---|
| K Rate | 60 PA | 75% |
| SB Rate | 200 PA | 91% |
| HR Rate | 170 PA | 89% |
| OBP | 460 PA | 96% |
| AVG | 910 AB | 98% |
| Pitcher K Rate | 70 BF | 78% |
| Pitcher ERA | 630 BF | 97% |

### Matchup Multiplier

```
Matchup_Mult = pitcher_quality_adj * park_factor * platoon_adj * weather_adj * order_adj
```

| Factor | Formula | Range |
|---|---|---|
| Pitcher quality | (2 - xFIP/4.20)^2, clamped [0.5, 2.5] | +/- 5-25% |
| Park factor | stat-specific (HR factor, R factor) / 100 | +/- 5-30% |
| Platoon | +0.028 wOBA for opposite-hand, -0.016 for same-hand | +/- 5-8% |
| Weather | HomeRunForecast index-based adjustment | +/- 5-15% |
| Batting order | +0.05 per spot above #5, -0.05 per spot below #5 | +/- 2-4% |

### Category Urgency (H2H-specific)

```
Urgency(cat) = 1 / (1 + exp(-k * (opp_total - my_total) / sgp_denom))
```

- k=2.0 for counting stats (R, HR, RBI, SB, K, SV, W)
- k=3.0 for rate stats (AVG, OBP, ERA, WHIP) — steeper because rate stats flip harder
- L uses inverse: k=2.0, gap = my_total - opp_total (lower is better)

### Rate Stat Protection Logic

```python
if winning_rate_comfortably(gap > 0.30 ERA or > 0.05 WHIP or > .020 AVG):
    # PROTECT: sit marginal players, fewer AB/IP is better
    rate_protection_mode = "protect"
elif losing_rate_badly(gap < -0.50 ERA or < -0.08 WHIP or < -.030 AVG):
    # ABANDON: start everyone, volume for counting stats
    rate_protection_mode = "abandon"
else:
    # COMPETE: start only high-rate players
    rate_protection_mode = "compete"
```

### Win Probability (Pitchers)

```
P(Win) = 0.7 * (0.112 * team_R/G - 0.105 * pitcher_xFIP + 0.446)
```

### Save Opportunity Estimate

```
Weekly_SV = games_per_week * team_win_pct * 0.60 * conversion_rate
```

### Complete Variable Inventory (50 variables)

**Hitter (26 variables):** ROS projected R/HR/RBI/SB/AVG/OBP, 2025 per-game rates, 2024 per-game rates, 2026 YTD stats, 7-day K%/BB%, 14-day barrel rate/exit velo, YTD xwOBA, opposing pitcher xFIP, opposing pitcher K rate, opposing pitcher hand, park HR factor, park R factor, weather adjustment, batting order position, home/away, health score (3-year), Yahoo status, days since last game, BvP (50+ PA only)

**Pitcher (14 variables):** ROS projected IP/W/K/SV/ERA/WHIP, xFIP/SIERA, opponent wOBA (14-day), opponent K%, pitcher K/9, park factor (pitching), team R/G, days rest/fatigue, probable starter indicator, two-start week flag, closer role confidence, save opportunity probability

**Matchup Context (6 variables):** Current matchup category scores, category urgency weights, opponent probable starters, opponent closer situation, categories to attack/protect, rate stat protection mode

**Free Agent (4 variables):** Best FA at each position (ROS projection), FA ownership velocity, FA games remaining this week, FA matchup quality

**TOTAL: 52 variables across 5 domains** (50 original + doubleheader flag + transaction budget)

### Critical Review Findings (Verified 2026-03-31)

**10 issues identified and resolved:**

1. **ADDED: Roster-slot LP assignment layer.** DCV scores feed into the existing `advanced_lp.py` LP solver for positional slot assignment. DCV is the scoring input, LP is the assignment output.

2. **FIXED: Rate stat matchup multiplier.** Apply matchup multipliers to component stats (H, AB, BB, ER, IP), then recompute rates. Do NOT multiply AVG/ERA/WHIP directly — overstates the effect.

3. **ADDED: Doubleheader multiplier.** Volume_Factor = 2.0 for confirmed doubleheader starters (counting stats double). Pitchers stay at 1.0 (typically one start per doubleheader).

4. **ADDED: Weekly IP minimum constraint.** If team IP < 20 heading into the last day, MUST start a pitcher regardless of matchup quality — or forfeit all pitching categories. Uses existing `ip_tracker.py`.

5. **ADDED: Transaction budget tracking.** Streaming FA recommendations discounted by remaining weekly adds. If 1 add left and it's Wednesday, save it for emergencies.

6. **ADDED: Pitcher blow-up risk / variance penalty.** Pitchers with high game-to-game ERA variance get a penalty in rate-stat protection mode. Consistent 3.50 xFIP pitcher preferred over volatile 3.50 xFIP pitcher when protecting ERA lead.

7. **FIXED: Win probability clamped to [0.05, 0.75].** Formula can produce values outside [0,1] with extreme inputs. Clamped.

8. **REMOVED: BvP history.** Research consensus: near-zero predictive value even at 200 PA. Adds complexity, no signal.

9. **REMOVED: Month-by-month blend schedule.** Redundant with Bayesian stabilization formula. The Bayesian formula already handles the transition via stat-specific stabilization points. Using both double-counts sample size adjustment.

10. **ADDED: Stud floor.** Players in the top 30 by ROS projection get a DCV floor that prevents benching except on off days or IL. Trout doesn't get benched for a bad matchup.

**Additional edge cases noted (lower priority):**
- Postponement/rainout risk: P(game_played) factor for outdoor parks with rain
- Lock-time awareness: early-game slots must be set before late-game info available
- Closer committee handling: split save opportunities between committee members
- Opponent lineup prediction: factor in known opponent SP starts when setting own pitching strategy
- September roster expansion: detect playing time changes for prospects
- Empty slot option: sometimes leaving a pitching slot empty is optimal (protecting rate stats)

### Math Verification

1. **Bayesian blend at PA=0:** (Prior * Stab + 0) / (Stab + 0) = Prior. Correct — 100% preseason at start.
2. **Bayesian blend at PA=Stab:** (Prior * Stab + Obs * Stab) / (2 * Stab) = (Prior + Obs) / 2. Correct — 50/50 at stabilization point.
3. **Urgency at gap=0:** 1 / (1 + exp(0)) = 0.5. Correct — tied category gets neutral urgency.
4. **Urgency when losing big (gap >> 0):** approaches 1.0. Correct — max urgency for losing categories.
5. **Urgency when winning big (gap << 0):** approaches 0.0. Correct — no urgency for safely winning categories.
6. **Volume factor for off-day player:** 0.00. Correct — zero production guaranteed.
7. **Volume factor transitions from 0.90 to 1.00:** only when lineup card confirmed. Correct.
8. **IL/DTD/NA = 0.00 (excluded):** Correct — cannot produce fantasy points while injured/inactive.
9. **Pitcher win prob range check:** At xFIP=3.0, team_RPG=5.0: 0.7*(0.56-0.315+0.446) = 0.7*0.691 = 0.484 (48%). At xFIP=5.0, team_RPG=3.5: 0.7*(0.392-0.525+0.446) = 0.7*0.313 = 0.219 (22%). Reasonable range.
10. **Park factor application:** If HR park factor = 120 (Coors), multiplier = 1.20. For a projected 0.5 HR/game, adjusted = 0.60 HR/game. This is for home games only — away games use the away park's factor. Correct.
11. **Rate stat protection thresholds:** 0.30 ERA gap = meaningful lead (e.g., 2.50 vs 2.80). 0.05 WHIP gap = meaningful (e.g., 1.10 vs 1.15). 0.020 AVG gap = meaningful (e.g., .265 vs .245). These are reasonable competitive thresholds for weekly H2H.
