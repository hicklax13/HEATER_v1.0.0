# Fantasy Baseball Draft Tool: Implementation Plan

## 1. Executive Summary

This report presents a comprehensive implementation plan for a live fantasy baseball draft assistant designed for use on a laptop during a Yahoo Sports 12-team snake draft (23 rounds, 5x5 roto scoring). The tool will recommend the optimal player to draft each time the user is on the clock, accounting for league settings, draft state, category scarcity, positional needs, opponent behavior, and risk.

**Best overall concept:** A Streamlit-based Python application that runs locally on the user's laptop alongside the Yahoo draft room in a split-screen layout. The tool maintains a live model of the draft board — updated via Yahoo Fantasy API polling supplemented by manual entry — and recommends picks by combining Standings Gain Points (SGP) valuation, positional scarcity analysis, Monte Carlo draft simulation, and roster-conditional category optimization. Recommendations are displayed in a novice-friendly "big green button" interface with progressive disclosure of supporting analysis.

**Single best recommended build approach:** A Python/Streamlit local web application backed by a SQLite database of pre-loaded projections (blended from FanGraphs Steamer, ZiPS, and Depth Charts), with a NumPy-vectorized Monte Carlo simulation engine running 300-500 draft simulations per recommendation cycle. The tool polls the Yahoo Fantasy API for draft state updates and falls back to one-click manual entry when API access is unavailable.

**Most important advantage over simpler tools:** Unlike static cheat sheets or rank-based drafters, this tool dynamically revalues every available player after every pick based on the user's current roster composition, category gaps, positional needs, and the probability each player survives to the user's next pick. This roster-conditional, simulation-informed approach captures the compounding value of category balance and positional scarcity that static rankings miss entirely — the difference between drafting the nominally "best available" player and drafting the player who maximizes expected standings points given everything that has happened in the draft so far.

---

## 2. League Interpretation and Design Implications

### Roster Construction Analysis

The league uses the following roster structure (23 slots total, matching 23 draft rounds):

| Slot | Count | Category | Notes |
|------|-------|----------|-------|
| C | 1 | Hitting | Most scarce position; steep talent drop-off after top 5 |
| 1B | 1 | Hitting | Deep position; many power hitters eligible |
| 2B | 1 | Hitting | Moderate scarcity |
| 3B | 1 | Hitting | Generally deep with power |
| SS | 1 | Hitting | Historically scarce; has deepened recently (Witt Jr., De La Cruz, etc.) |
| OF | 3 | Hitting | Very deep pool (36 starters league-wide) |
| Util | 2 | Hitting | Flexible — any hitter; acts as overflow valve |
| SP | 2 | Pitching | Dedicated SP-only slots |
| RP | 2 | Pitching | Dedicated RP-only slots |
| P | 4 | Pitching | Flexible — any pitcher (SP or RP) |
| BN | 5 | Any | Bench slots; no positional restriction |

**Total hitters rostered:** 13 per team × 12 teams = 156 hitters league-wide (including bench hitters)
**Total pitchers rostered:** 8 per team × 12 teams = 96 pitchers league-wide (including bench pitchers)
**Grand total drafted:** 23 × 12 = 276 players

### How the Roster and 5x5 Format Shape Draft Strategy

**Verified Fact:** The standard Yahoo 5x5 roto categories are:
- **Batting:** Runs (R), Home Runs (HR), Runs Batted In (RBI), Stolen Bases (SB), Batting Average (AVG)
- **Pitching:** Wins (W), Saves (SV), Strikeouts (K), Earned Run Average (ERA), Walks+Hits per Inning Pitched (WHIP)

(Yahoo Help, "Default league settings in Fantasy Baseball," accessed March 2026; Yahoo Help, "Rotisserie scoring," accessed March 2026)

**Assumption:** The FourzynBurn league uses these standard 5x5 categories. This should be verified from the league settings page before draft day. If the league uses OBP instead of AVG, or Holds instead of Saves, the tool's category weights and SGP denominators must be recalibrated accordingly.

**Key strategic implications of this roster structure:**

1. **The 2 Util slots reduce hitting positional scarcity — except at C and SS.** Because any hitter can fill Util, surplus value at 1B, 3B, or OF flows into Util. But C and SS must be filled by position-eligible players. This makes C and SS relatively more valuable in this format.

2. **The 4 P slots create pitcher flexibility.** Managers can run 6 SP + 2 RP, or 4 SP + 4 RP, or any mix. This flexibility means the tool must model pitcher allocation dynamically rather than assuming a fixed SP/RP split.

3. **The 5 BN slots are significant.** With 5 bench spots, managers can stash prospects, hedge injuries, or carry extra pitchers. In a skilled league, bench slots are used strategically for upside plays.

4. **Rate stats (AVG, ERA, WHIP) require volume-weighted modeling.** Adding a low-AVG slugger actively hurts your team's AVG. The tool must compute marginal rate-stat impact using at-bats/innings, not just the player's individual rate.

5. **Category correlations matter.** R, HR, and RBI are positively correlated (power hitters in good lineups contribute to all three). SB is largely independent, coming from a different player archetype. This means SB contributors carry a scarcity premium. Similarly, SV is concentrated among a small pool of closers, creating extreme scarcity dynamics.

### How to Model P, SP, and RP Slots

The pitcher slot structure creates a strategic puzzle:

- **SP slots (2):** Must be filled by starting pitchers. These contribute primarily to W, K, ERA, WHIP.
- **RP slots (2):** Must be filled by relief pitchers. These contribute primarily to SV (for closers), ERA, WHIP, and some K.
- **P slots (4):** Can be filled by either SP or RP. The optimal allocation depends on available talent and category needs.

**Strong Inference:** The dominant strategy in competitive leagues is to lean SP-heavy in P slots (e.g., 5-6 SP, 2-3 RP total) because elite SP contribute to 3+ pitching categories while closers contribute primarily to SV alone. However, completely ignoring saves is risky in roto because it concedes an entire category (maximum 11 standings points lost).

The tool should model pitcher allocation as a continuous optimization decision, evaluating each pitcher candidate by their marginal SGP contribution across all pitching categories given the current pitcher roster composition.

### Why This League Is Strategically Difficult

**Verified Fact (from PROMPT.md):** The opposing managers are described as "extremely high-skilled, experienced." The user is described as "extremely low and inexperienced."

This creates several challenges:

1. **Efficient market:** Skilled managers have already internalized most valuation theory. Obvious value plays (undervalued players by ADP) are exploited quickly. Edge must come from execution, not from basic valuation.

2. **Position runs are faster and earlier.** Skilled managers recognize tier breaks and scarcity triggers. Runs on SS, C, and closers will happen earlier than ADP suggests.

3. **Contrarian strategies are already priced in.** "Punt saves" and other advanced strategies are not novel to this field — skilled opponents may already be employing them, limiting their edge.

4. **In-season management advantage is asymmetric.** Skilled managers excel at waiver-wire pickups, trades, and streaming. A novice user should draft for floor over ceiling — a stable, balanced roster that requires minimal active management to remain competitive.

5. **The tool is the great equalizer.** The primary value proposition is that a well-built tool can give the novice user access to the same (or better) analytical capabilities that skilled managers use intuitively. The tool should execute sound fundamentals flawlessly, which alone puts the novice on competitive footing.

---

## 3. Specialist Agent Team and Research Workflow

### Team Size Justification

**Optimal team size: 6 agents.**

This count is justified by the natural decomposition of the problem domain into six largely independent areas of expertise, each requiring distinct knowledge and producing distinct deliverables that integrate at well-defined interfaces. Fewer agents would overload individual mandates and risk shallow coverage; more agents would create coordination overhead without proportional benefit.

### Agent Roster

#### Agent 1: Rules & Platform Mechanics Agent
- **Mission:** Map the complete Yahoo Fantasy Baseball draft environment — settings, interface, API capabilities, timing, and constraints — so the tool is designed around verified platform realities.
- **Research Questions:**
  1. What are the exact Yahoo 5x5 roto default categories? What variations exist?
  2. How does the Yahoo live draft room work (timer, queue, board, visibility)?
  3. What does the Yahoo Fantasy Sports API provide? Can it return live draft picks during an in-progress draft?
  4. What are Yahoo's roster eligibility rules (multi-position players, DL/IL handling)?
  5. What input methods (API polling, browser extension, manual entry) are feasible for real-time draft tracking?
- **Output Format:** Platform specification document with verified capabilities, unverified assumptions, and recommended integration approach.
- **Why Necessary:** Every other agent's output depends on platform constraints. False assumptions about Yahoo's capabilities (e.g., assuming live API streaming that doesn't exist) would invalidate the entire build plan.

#### Agent 2: Player Projections & Baseball Modeling Agent
- **Mission:** Identify the best available projection data for 2026, design the projection blending methodology, and define the risk-adjustment framework.
- **Research Questions:**
  1. Which 2026 projection systems are currently published and accessible?
  2. What is the evidence-based best practice for blending projections?
  3. How should injury risk, playing time uncertainty, and role volatility be modeled?
  4. How should aging curves and regression inform projections?
  5. What data sources provide the necessary inputs (projections, ADP, injuries, depth charts)?
- **Output Format:** Data source catalog with access methods, a projection blending formula, and a risk-adjustment framework with explicit formulas.
- **Why Necessary:** The entire valuation engine depends on accurate, well-calibrated player projections. Garbage projections make all downstream optimization worthless.

#### Agent 3: Optimization & Simulation Agent
- **Mission:** Design the mathematical framework for draft pick optimization, including the objective function, Monte Carlo simulation engine, and opponent modeling system.
- **Research Questions:**
  1. What is the correct objective function for 5x5 roto draft optimization?
  2. How should Monte Carlo draft simulation be structured for computational feasibility?
  3. How should opponent pick probabilities be modeled, especially for skilled opponents?
  4. What is the optimal "reach vs. wait" decision framework?
  5. How should category balance and positional scarcity enter the optimization?
  6. What computational budget is realistic for live-draft recommendations?
- **Output Format:** Mathematical specification with explicit formulas, algorithm pseudocode, and computational complexity analysis.
- **Why Necessary:** This is the core intellectual contribution — the difference between a ranked list and a true decision engine. Without rigorous optimization, the tool is just another cheat sheet.

#### Agent 4: UX & Live Draft Workflow Agent
- **Mission:** Design the user interface and operational workflow for a novice user drafting under time pressure on a laptop.
- **Research Questions:**
  1. What is the optimal split-screen layout for Yahoo draft room + assistant tool?
  2. How should recommendations be displayed to minimize cognitive load?
  3. What interaction model (clicks, typing, keyboard shortcuts) is fastest under pressure?
  4. How should the tool handle panic situations (short clock, unexpected picks)?
  5. What failover and recovery mechanisms are needed?
  6. What existing draft tools (FantasyPros Draft Wizard, etc.) offer, and where can a custom tool improve?
- **Output Format:** UX specification with screen layouts, interaction flows, and failure mode handling.
- **Why Necessary:** A technically brilliant tool that is unusable under draft pressure is worthless. UX is a first-class requirement, not an afterthought.

#### Agent 5: Data Architecture & Systems Agent
- **Mission:** Design the data pipeline, tech stack, storage layer, and deployment architecture.
- **Research Questions:**
  1. What is the best framework for a local laptop-based draft tool?
  2. How should pre-draft data (projections, ADP) be ingested and stored?
  3. What caching strategy optimizes computation latency during the draft?
  4. Can Monte Carlo simulations run fast enough on a laptop in Python?
  5. What is the backup architecture if the primary tool fails?
- **Output Format:** Architecture diagram, tech stack recommendation with justification, and deployment/setup instructions.
- **Why Necessary:** Architectural decisions (framework choice, data layer, caching) constrain everything else. The wrong tech stack can make a correct algorithm too slow or too fragile for live use.

#### Agent 6: Validation & Backtesting Agent
- **Mission:** Design the testing framework to verify the tool produces good recommendations before draft day.
- **Research Questions:**
  1. How should historical draft replays be structured for backtesting?
  2. What metrics quantify "good" draft recommendations?
  3. How should the tool be calibrated against known-good strategies?
  4. What ablation tests verify that each component adds value?
  5. How should sensitivity analysis be conducted on key parameters (SGP denominators, risk aversion, simulation count)?
- **Output Format:** Test plan with specific test cases, success criteria, and a pre-draft validation checklist.
- **Why Necessary:** Without validation, we cannot know whether the tool helps or hurts. Backtesting is the only way to build confidence before the irreversible live draft.

### Synthesis Workflow

```
Agent 1 (Platform)  ──→ Constraints feed into all other agents
Agent 2 (Projections) ──→ Data inputs feed into Agent 3
Agent 3 (Optimization) ──→ Algorithm specification feeds into Agent 5
Agent 4 (UX) ──→ Interface requirements feed into Agent 5
Agent 5 (Architecture) ──→ Implementation plan integrates all outputs
Agent 6 (Validation) ──→ Tests verify integrated system

Synthesis Order:
1. Agent 1 produces platform constraints (first, because all agents need this)
2. Agents 2, 3, 4 work in parallel (independent research domains)
3. Agent 5 integrates architecture after 2, 3, 4 complete
4. Agent 6 designs validation after the system design is finalized
5. Lead architect synthesizes all outputs into this master plan
```

**Conflict Resolution Protocol:** Where agent outputs conflict (e.g., Agent 3 recommends a computationally expensive method that Agent 5 deems infeasible), the resolution prioritizes: (1) verified platform constraints, (2) computational feasibility for live use, (3) analytical rigor, (4) development simplicity.

---

## 4. Research Synthesis

### Well-Established Methods and Best Practices

The following methods are well-established in fantasy baseball analytics and should form the foundation of the tool:

**Valuation Theory (Verified Fact — established literature since 1990s):**
- **Standings Gain Points (SGP)** is the gold-standard valuation framework for roto leagues, directly mapping raw stats to standings movement (Patton, "Patton$ Dollar$"; McGee, "How to Value Players for Rotisserie Baseball"; Smart Fantasy Baseball, various articles 2015-2024).
- **Replacement-level baselines** (VORP) correctly capture positional scarcity by measuring each player against the best freely available alternative at their position.
- **Rate-stat adjustment** using volume weighting (at-bats for AVG, innings for ERA/WHIP) is required to avoid systematic errors in player valuation.

**Projection Blending (Verified Fact — meta-analytic evidence):**
- Simple equal-weight averaging of 3-4 major projection systems (Steamer, ZiPS, THE BAT) performs nearly as well as complex optimization-based weighting (Carty, various articles; "wisdom of crowds" effect in forecasting).
- Playing time is the largest source of projection disagreement and error; using FanGraphs Depth Charts (which blend projections with RosterResource playing time) is the most practical approach (FanGraphs, "All the 2026 Projections Are In!", accessed March 2026).
- **Strong Inference:** Adding PECOTA (paid) or THE BAT X (paid) provides marginal improvement; the free tier of FanGraphs is sufficient for a strong projection foundation.

**Draft Optimization (Strong Inference — applied research, 2016-2025):**
- Monte Carlo draft simulation with ADP-based opponent modeling is the dominant practical approach for snake draft optimization (Hunter, Vielma, Zaman, "Picking Winners," MIT Sloan Sports Analytics, 2016; Jensen, "Simulating the Snake," Medium, 2024).
- 300-500 simulations provide adequate convergence for distinguishing top candidates while remaining computationally feasible on a laptop (10-15 seconds with NumPy vectorization).
- Exact dynamic programming is intractable for realistic draft sizes but approximate DP via rollout policies (implemented as Monte Carlo simulation with heuristic future play) is computationally practical and theoretically sound (Bertsekas, "Dynamic Programming and Optimal Control," multiple editions).

**Category Balance (Verified Fact — mathematical property of roto scoring):**
- Roto scoring has inherent diminishing returns within categories (maximum 12 points per category). Balanced teams mathematically outperform unbalanced teams at the same total production level.
- "Punt saves" is the only commonly defensible category-punt strategy, and even this is risky in skilled leagues.

### Speculative or Unproven Methods

**Methods to use cautiously or reject:**

| Method | Assessment | Recommendation |
|--------|-----------|---------------|
| Deep reinforcement learning for draft policy | Academically interesting but impractical — requires massive training infrastructure and provides marginal improvement over Monte Carlo + heuristic rollout | **Reject** for this project |
| Complex Bayesian opponent modeling with MCMC | Theoretically elegant but computationally expensive and overkill for a 23-round draft with limited observed data | **Simplify** to discrete strategy classification |
| Game-theoretic Nash equilibrium computation | Not computationally tractable for 12-player sequential games with this state space | **Use conceptually** to inform strategy but do not attempt to compute |
| Machine learning on historical draft data | Requires large training datasets that are hard to obtain for specific league formats; prone to overfitting | **Reject** in favor of projection-based valuation |
| Full mixed-integer optimization of remaining draft | Useful as a subroutine (roster completion solver) but not as the primary decision engine because it cannot model opponent behavior | **Use as component** within Monte Carlo framework |

### Framework Comparison and Recommendations

| Framework | Strengths | Weaknesses | Verdict |
|-----------|-----------|------------|---------|
| **SGP + Replacement Level** | Directly optimizes standings points; well-calibrated | Requires historical denominators; static without marginal adjustment | **Use as primary valuation** with marginal extension |
| **Z-scores** | Simple; no historical data needed | Does not map to standings gains; pool-definition-sensitive | **Use as secondary cross-check** |
| **Auction dollar values** | Standard in industry | Not designed for snake drafts; ignores pick-timing | **Reject** for this context |
| **Tier-based ranking** | Intuitive; good for manual fallback | No mathematical optimization | **Use for cheat-sheet backup** |
| **Monte Carlo + SGP rollout** | Handles uncertainty; models opponent behavior; computationally feasible | Requires careful implementation; convergence depends on simulation count | **Use as primary recommendation engine** |

---

## 5. Mathematical Objective Function

### The Optimization Target

The tool maximizes the user's **expected total standings points** across all 10 categories over the full season:

```
Maximize:  E[U] = E[ Sum_{c=1}^{10} Rank_c(Team) ]
```

where `Rank_c(Team)` is the team's rank (1 = worst, 12 = best) in category `c` among all 12 teams, and the expectation is over projection uncertainty and draft outcome uncertainty.

### Decomposed Objective

Since computing expected ranks directly requires full-league projection, we decompose using SGP as a proxy:

```
U(roster) = Sum_{c=1}^{10} SGP_c(roster)
```

where for **counting stats** (R, HR, RBI, SB, W, SV, K):

```
SGP_c(roster) = [ Sum_{p in roster} proj_c(p) ] / D_c
```

and for **rate stats** (AVG, ERA, WHIP):

```
SGP_AVG(roster) = [ ( Sum_{p} H_p ) / ( Sum_{p} AB_p ) - AVG_replacement ] / D_AVG

SGP_ERA(roster) = [ ERA_replacement - ( Sum_{p} ER_p * 9 ) / ( Sum_{p} IP_p ) ] / D_ERA

SGP_WHIP(roster) = [ WHIP_replacement - ( Sum_{p} (BB_p + H_allowed_p) ) / ( Sum_{p} IP_p ) ] / D_WHIP
```

Note: ERA and WHIP are inverted because lower is better.

**Variables:**
- `proj_c(p)` = player `p`'s projected stat in category `c`
- `D_c` = SGP denominator for category `c` (stat units per standings point)
- `H_p`, `AB_p` = projected hits and at-bats for player `p`
- `ER_p`, `IP_p` = projected earned runs and innings pitched for player `p`
- `AVG_replacement`, `ERA_replacement`, `WHIP_replacement` = league-average replacement baselines

### SGP Denominators

**Strong Inference:** Typical SGP denominators for a competitive 12-team 5x5 roto league (BaseballHQ, "2025 Standings Gain Points," accessed March 2026; Smart Fantasy Baseball, "SGP Slope Calculator," accessed March 2026):

| Category | SGP Denominator | Interpretation |
|----------|----------------|----------------|
| R | ~32 | 32 runs = 1 standings point |
| HR | ~13 | 13 HR = 1 standings point |
| RBI | ~32 | 32 RBI = 1 standings point |
| SB | ~14 | 14 SB = 1 standings point |
| AVG | ~.004 | .004 AVG = 1 standings point |
| W | ~3.5 | 3.5 wins = 1 standings point |
| SV | ~9 | 9 saves = 1 standings point |
| K | ~45 | 45 K = 1 standings point |
| ERA | ~0.20 | 0.20 ERA = 1 standings point |
| WHIP | ~0.020 | 0.020 WHIP = 1 standings point |

**Assumption:** These are approximate. The tool should allow the user to input league-specific historical denominators or use published defaults.

### Marginal Value of a Player Given Current Roster

The key innovation over static rankings: player value depends on the current roster state.

**Marginal SGP for counting stat `c`:**

```
Delta_SGP_c(p, roster) = proj_c(p) / D_c
```

This is additive and roster-independent for counting stats.

**Marginal SGP for rate stat AVG:**

```
New_AVG = (roster_H + p_H) / (roster_AB + p_AB)
Old_AVG = roster_H / roster_AB
Delta_SGP_AVG(p, roster) = (New_AVG - Old_AVG) / D_AVG
```

This is roster-dependent: a .300 hitter adds more AVG SGP to a .260 roster than to a .290 roster, and the impact scales with the player's AB relative to the roster's total AB.

### Category Scarcity in the Formula

To prevent overinvesting in already-strong categories, apply a **diminishing returns multiplier**:

```
w_c(roster) = max(0, target_rank_c - projected_rank_c(roster)) / target_rank_c
```

where `target_rank_c` is the target standings rank in category `c` (e.g., 4th place = 9 standings points). When the roster already projects to exceed the target, the weight approaches zero, redirecting value toward weaker categories.

**Adjusted marginal value:**

```
Adjusted_Delta_SGP(p, roster) = Sum_{c} [ w_c(roster) * Delta_SGP_c(p, roster) ]
```

### Positional Scarcity in the Formula

**Replacement-level adjustment:**

```
VORP(p) = Adjusted_Delta_SGP(p, roster) - Adjusted_Delta_SGP(replacement_at_pos(p), roster)
```

where `replacement_at_pos(p)` is the best available player at `p`'s position who would be available in a later round (approximated by the player ranked just below the roster-filling threshold at that position).

For multi-position-eligible players, use the position with the highest replacement level (i.e., the position where their scarcity premium is greatest).

### Uncertainty in the Formula

**Risk-adjusted value:**

```
Risk_Adj_Value(p) = E[VORP(p)] - lambda * Var[VORP(p)]
```

where:
- `E[VORP(p)]` = expected VORP from blended projections
- `Var[VORP(p)]` = variance estimated from inter-system projection disagreement + injury probability
- `lambda` = risk-aversion parameter (recommended: 0.1-0.3 for a novice user; higher = more conservative)

**Injury discount:**

```
E_adjusted(stat) = proj_stat * (1 - P_injury) + replacement_stat * P_injury
```

where `P_injury` is estimated from injury history and health projections.

### Expected Availability at Next Pick

**Survival probability:**

```
P_survive(p, t_next) = Product_{k=t_current+1}^{t_next-1} (1 - P_drafted(p, k))
```

where `P_drafted(p, k)` is the probability player `p` is selected at pick `k`, estimated from ADP-based opponent modeling.

**Simplified approximation:**

```
P_survive(p, t_next) ≈ Phi( (ADP_p - t_next) / sigma_ADP )
```

where `Phi` is the standard normal CDF and `sigma_ADP` ≈ 0.15 * ADP_p.

### Combined Pick Score

The final pick recommendation score integrates all components:

```
Pick_Score(p) = Risk_Adj_VORP(p) + alpha * Urgency(p)

Urgency(p) = (1 - P_survive(p)) * [Risk_Adj_VORP(p) - Risk_Adj_VORP(next_best_at_position(p))]
```

The urgency term captures: "If I don't draft this player now, they'll be gone, and the drop-off at their position is steep." The parameter `alpha` (recommended: 0.3-0.5) controls how much weight to give urgency versus pure value.

---

## 6. Core Player Valuation Framework

### Component Architecture

The valuation engine has 8 components, each contributing to the final player value:

### Component 1: Projection Blending

**Formula:** Simple equal-weight average of available projection systems.

```
blended_stat_c(p) = (1/N) * Sum_{sys=1}^{N} proj_c_sys(p)
```

**Recommended systems (all accessible free via FanGraphs CSV export):**
- Steamer
- ZiPS
- FanGraphs Depth Charts (50/50 Steamer/ZiPS with RosterResource playing time)

(FanGraphs, "All the 2026 Projections Are In!", accessed March 2026; FanGraphs, "2026 Projections - Steamer," accessed March 2026)

**Why it belongs:** Blending reduces individual system biases and exploits the wisdom-of-crowds effect. Simple averaging performs within 1-2% of optimally weighted blends (Strong Inference — consistent finding in forecasting literature).

**Data needed:** CSV exports from FanGraphs for each projection system.

**Calibration:** Compare blended projections against prior-year actuals to verify no systematic bias. Adjust if needed.

### Component 2: Category Standardization via SGP

**Formula:** Convert raw projected stats to SGP values (see Section 5).

**Why it belongs:** SGP is the correct unit of value for roto leagues because it directly measures standings impact. Raw stats are incomparable across categories (1 HR ≠ 1 SB in standings value).

**Data needed:** SGP denominators from historical league data or published defaults (BaseballHQ, Smart Fantasy Baseball).

**Calibration:** If league-specific historical standings data is available, compute custom denominators. Otherwise, use published 12-team 5x5 defaults and adjust based on league competitiveness.

### Component 3: Replacement Level Baselines

**Formula:**

```
VORP(p) = Total_SGP(p) - Total_SGP(replacement_at_best_position(p))
```

Replacement level for each position = the (N+1)th best player, where N = league-wide starters at that position:

| Position | N (league-wide starters) | Replacement = rank... |
|----------|--------------------------|----------------------|
| C | 12 | 13th |
| 1B | 12 | 13th |
| 2B | 12 | 13th |
| 3B | 12 | 13th |
| SS | 12 | 13th |
| OF | 36 | 37th |
| SP | ~48-60 (depends on P allocation) | ~50th |
| RP | ~36-48 (depends on P allocation) | ~38th |

**Why it belongs:** Without replacement baselines, the tool cannot distinguish between "good player at a deep position" and "mediocre player at a scarce position." VORP captures the opportunity cost of positional allocation.

**Data needed:** Player positional eligibility from FanGraphs or Yahoo.

**Calibration:** Recompute replacement levels dynamically after each draft pick, as the available player pool shrinks non-uniformly across positions.

### Component 4: Role-Adjusted Playing Time

**Formula:**

```
adj_stat(p) = rate_stat(p) * adj_PA(p)  [for hitters]
adj_stat(p) = rate_stat(p) * adj_IP(p)  [for pitchers]
```

where `adj_PA` and `adj_IP` account for platoon splits, injury history, depth chart position, and spring training developments.

**Why it belongs:** Playing time is the largest source of projection error. A player projected for .290/25HR is worth far less if they only get 350 PA instead of 600 PA.

**Data needed:** FanGraphs Depth Charts (which incorporate RosterResource playing time estimates).

**Calibration:** Cross-reference with THE BAT X playing time estimates if available.

### Component 5: Injury Probability Discount

**Formula:**

```
effective_value(p) = value(p) * (1 - P_injury(p)) + value(replacement(p)) * P_injury(p)
```

**Why it belongs:** Injury-prone players have lower expected value because their projected stats assume health. The discount accounts for the expected replacement-level production during missed time.

**Data needed:** Injury history (available from FanGraphs, Roster Resource, MLB.com); for 2026, spring training health reports.

**Calibration:** Use historical games-played percentages as a proxy for `P_injury`. Players averaging <130 games/season over the past 3 years should have progressively higher injury discounts.

### Component 6: Volatility / Variance Penalty

**Formula:**

```
variance(p) = Var_across_systems(Total_SGP(p))  [inter-system disagreement]
risk_penalty(p) = lambda * variance(p)
risk_adj_value(p) = E[VORP(p)] - risk_penalty(p)
```

**Why it belongs:** For a novice user who cannot optimally manage the waiver wire, high-variance players are less desirable. A floor-oriented strategy is more robust.

**Data needed:** Multiple projection systems (variance = disagreement between Steamer, ZiPS, Depth Charts).

**Calibration:** Set `lambda` to 0.15 as default. Higher values (0.25+) for extremely risk-averse drafting.

### Component 7: Roster Fit / Marginal Category Value

**Formula:**

```
marginal_value(p, roster) = Sum_{c} [ w_c(roster) * Delta_SGP_c(p, roster) ]
```

where `w_c(roster)` is the category weight based on current roster strength (see Section 5).

**Why it belongs:** This is the critical roster-conditional adjustment. A player who contributes SB is worth more if the roster is weak in SB. This component turns a static ranking into a dynamic, context-sensitive recommendation.

**Data needed:** Running category totals for the user's roster (maintained in the draft state).

**Calibration:** The weighting function should approach zero for categories where the roster projects to finish 1st-2nd (diminishing returns) and peak for categories where the roster projects to finish 8th-12th (most room for improvement).

### Component 8: Pick-Timing / Urgency

**Formula:**

```
urgency(p) = (1 - P_survive(p, t_next)) * positional_dropoff(p)

positional_dropoff(p) = VORP(p) - VORP(next_available_at_pos(p))
```

**Why it belongs:** In a snake draft, the cost of a player being taken before your next pick depends on how steep the talent drop-off is at their position. High urgency means "draft now or suffer a large value loss."

**Data needed:** ADP data (for survival probability), positional rankings (for drop-off).

**Calibration:** The `alpha` parameter controlling urgency weight should be set via backtesting against historical drafts.

---

## 7. Draft Recommendation Engine

### Step-by-Step Algorithm for One Draft Turn

When the user goes on the clock at pick number `t`:

**Step 1: Update Draft State (< 1 second)**
- Ingest all picks made since the last update (via API polling or manual entry)
- Update: available player pool, opponent rosters, user roster, category totals
- Recalculate replacement levels at each position

**Step 2: Compute Marginal Values (< 2 seconds)**
- For each available player `p`:
  - Compute `marginal_value(p, roster)` using current category weights
  - Compute `risk_adj_value(p)` with injury discount and variance penalty
  - Compute `urgency(p)` using survival probability to next pick
  - Compute `pick_score(p) = risk_adj_value(p) + alpha * urgency(p)`
- Rank by `pick_score`

**Step 3: Monte Carlo Validation (5-15 seconds, pre-computed when possible)**
- For the top 5-10 candidates:
  - Run 300-500 draft simulations for each candidate
  - In each simulation: simulate remaining opponent picks, use greedy SGP policy for user's future picks
  - Compute `E[total_SGP]` for each candidate
- Re-rank candidates by Monte Carlo expected outcome
- Flag any candidate where Monte Carlo ranking differs significantly from greedy ranking (indicates the greedy ranking may be myopic)

**Step 4: Generate Recommendation (instant)**
- **Primary pick:** Highest Monte Carlo expected value
- **Fallback picks:** 2nd and 3rd highest
- **Explanation:**
  - "Best value: [Player] fills [position] and helps [weak categories]"
  - "Urgency: [X]% chance they're gone by pick [next_pick]"
  - "Alternative: [Player B] if you prefer [category focus]"

**Step 5: Display to User**
- Show recommendation panel (see Section 13: UX/UI Specification)
- Await user's actual pick entry
- When user confirms their pick, update roster and return to Step 1

### "Must Draft Now" vs. "Likely Available Later" Logic

```
IF P_survive(p, t_next) < 0.30:
    Label: "MUST DRAFT NOW — 70%+ chance they're taken"
ELIF P_survive(p, t_next) < 0.60:
    Label: "Moderate risk — consider drafting now"
ELSE:
    Label: "Likely available later — safe to wait"
```

### Opponent Pick Risk

After each pick, update the estimated ADP model based on observed behavior:
- If an opponent drafted a player earlier than ADP predicted, increase that opponent's "aggressiveness" parameter
- If a position run is detected (3+ picks at the same position in 8 picks), increase urgency for remaining players at that position

### Category Impact After Selection

Display a before/after comparison:

```
Category  | Before    | After     | Change | Projected Rank
----------|-----------|-----------|--------|---------------
R         | 782       | 812       | +30    | 6th → 5th
HR        | 198       | 213       | +15    | 7th → 6th
SB        | 89        | 89        | +0     | 10th (no change)
AVG       | .264      | .266      | +.002  | 5th (no change)
...
```

### Positional Opportunity Cost

For each recommendation, show what the user would give up:

```
"By drafting [SS] now, you use a pick that could have gone to [best available OF].
  SS drop-off if you wait: -2.3 SGP (steep — 13th-best SS is much worse)
  OF drop-off if you wait: -0.8 SGP (shallow — plenty of good OF remain)"
```

---

## 8. Opponent and Draft-Room Modeling

### Modeling Skilled Opposing Managers

**Baseline model:** Each opponent selects from the available pool with probability proportional to an ADP-proximity kernel:

```
P(opponent drafts p at pick k) = exp( -|ADP_p - k| / sigma ) / Z
```

where `Z` is the normalization constant over available players, and `sigma` controls adherence to consensus. For skilled leagues, `sigma ≈ 8-12` (tighter than casual leagues where `sigma ≈ 15-25`).

**Positional need adjustment:** Multiply the base probability by a positional need factor:

```
P_adjusted(p, k, opponent_i) = P_base(p, k) * need_factor(pos(p), roster(opponent_i))
```

where `need_factor > 1` if the opponent has an unfilled required position and `< 1` if they are already set at that position.

### Position Runs

**Detection heuristic:**

```
IF (picks_at_position_X in last 8 picks) >= 3:
    position_run_active(X) = True
    FOR all remaining opponents in current round:
        need_factor(X) *= 1.5  [increased probability of continuing the run]
```

**Common run triggers:**
- **Closer run:** After the 3rd-4th closer is taken in quick succession, remaining closers are drafted rapidly. Urgency for SV contributors spikes.
- **Catcher run:** After 3-4 catchers go, the remaining catchers drop off severely. Anyone still needing C must act.
- **SS run:** Similar tier-break dynamics. When the last elite SS is taken, a cascade follows.
- **Ace collapse:** When multiple SP1-tier pitchers go in a cluster, mid-tier SP become overvalued.

### Prospect Hype and Injury Discounting

**Strong Inference:** Skilled managers in 2026 will:
- Target high-upside prospects (particularly those with early-season call-up potential) in middle-to-late rounds
- Discount injured players more than ADP suggests (injured players' ADP often lags behind news)
- The tool should flag known injured players and track prospect ETA dates

### Estimating Survival Probability

The Monte Carlo simulation framework provides the most accurate survival estimates:

1. Run N=300 draft simulations from the current state
2. For each simulation, record which players survive to the user's next pick
3. `P_survive(p) = (count of simulations where p survives) / N`

This automatically accounts for opponent positional needs, position runs, and non-uniform ADP distributions.

**Simplified fallback (no simulation required):**

```
picks_between = t_next - t_current - 1  [number of opponent picks before user's next turn]
P_survive(p) ≈ max(0, 1 - picks_between / (ADP_p - t_current))  [linear approximation]
```

Clamped to [0, 1]. This is a rough heuristic but runs in constant time.

---

## 9. Simulation and Optimization Layer

### Monte Carlo Draft Simulation Framework

**Architecture:**

```
SimulationEngine:
  Input: current_state (available_players, all_rosters, pick_order, current_pick)
  Input: candidate_pick (player the user is considering)
  Output: expected_total_SGP for the user's final roster

  For each of N simulations:
    1. Clone current_state
    2. Assign candidate_pick to user's roster
    3. For each remaining pick in the draft:
       a. If user's pick: select player by greedy max(marginal_SGP) from available
       b. If opponent's pick: sample player from ADP_probability_model
    4. Compute total_SGP for user's final roster
    5. Store result

  Return: mean(results), std(results), percentile_25(results)
```

### Opponent Sampling Model

For each opponent pick, sample using the positional-need-adjusted ADP kernel:

```python
# Pseudocode
weights = []
for p in available_players:
    adp_weight = exp(-abs(p.adp - current_pick) / sigma)
    pos_need = 1.5 if opponent needs p.position else 0.7 if opponent has p.position filled else 1.0
    weights.append(adp_weight * pos_need)

selected = random.choices(available_players, weights=weights, k=1)
```

### Scenario Trees vs. Full Simulation

**Recommendation: Full Monte Carlo simulation, not explicit scenario trees.**

Scenario trees grow exponentially (each pick has ~50+ viable options, 276 total picks). Even pruned trees are impractical. Monte Carlo sampling implicitly explores the tree probabilistically without enumerating it.

### Dynamic Re-Ranking

After every pick (including opponent picks), the tool should:
1. Update available player pool
2. Recalculate positional replacement levels
3. Recalculate category weights based on updated roster
4. If it's between the user's picks: pre-compute recommendations for likely board states

### Recommended Optimization Framework

| Framework | Appropriate? | Role |
|-----------|-------------|------|
| **Monte Carlo simulation** | Yes — primary | Draft outcome evaluation |
| **Greedy SGP rollout** | Yes — within MC sims | Future-pick policy for user |
| **ILP (PuLP/OR-Tools)** | Yes — supplementary | Roster completion optimality check |
| **Approximate DP** | Yes — via MC + rollout | Implicitly implemented by simulation |
| **Full exact DP** | No — intractable | N/A |
| **Bayesian updating** | Yes — lightweight | Opponent model refinement |
| **Deep RL** | No — impractical timeline | N/A |

### Sophistication vs. Live-Draft Speed Tradeoffs

| Approach | Accuracy | Latency | Verdict |
|----------|----------|---------|---------|
| Greedy marginal SGP (no simulation) | Medium | <500ms | Always available as Tier 1 fallback |
| ILP roster completion per candidate | Medium-High | <2s | Good supplement to greedy |
| MC simulation, 300 sims | High | 5-10s | Sweet spot for live use |
| MC simulation, 1000 sims | Very High | 15-30s | Use between picks (pre-compute) |
| MC simulation, 5000+ sims | Diminishing returns | 1-3 min | Offline analysis only |

**Recommended tiered approach:**
1. **Instant (Tier 1):** Greedy marginal SGP ranking — always available, displayed immediately
2. **Fast (Tier 2):** ILP roster completion check — confirms greedy ranking within 2 seconds
3. **Full (Tier 3):** Monte Carlo simulation — pre-computed between picks, refined when on the clock if time permits

---

## 10. Data Stack and Source Plan

### Ranked Data Sources

#### Tier 1: Essential (Must Have Before Draft Day)

**1. FanGraphs Projections (Steamer, ZiPS, Depth Charts)**
- **What it provides:** Full-season stat projections for all MLB players across all standard categories
- **Why it matters:** The foundation of all player valuations. Without projections, the tool has nothing to value.
- **Reliability:** High — FanGraphs is the industry standard
- **Update cadence:** Pre-season release; Depth Charts updated daily
- **Cost/access:** Free. CSV export available via "Export Data" button on projection leaderboards
- **Verification:** **Verified Fact** — confirmed available for 2026 season (FanGraphs, "All the 2026 Projections Are In!", accessed March 2026)
- **Fallback:** If FanGraphs is unavailable, use Baseball Reference Marcel projections or FantasyPros consensus projections

**2. ADP Data (FantasyPros Yahoo-filtered)**
- **What it provides:** Average Draft Position data specific to Yahoo leagues, reflecting where players are actually being drafted
- **Why it matters:** Required for opponent modeling, survival probability, and reach-vs-wait decisions
- **Reliability:** High — aggregated from thousands of drafts
- **Update cadence:** Updated multiple times per week during draft season
- **Cost/access:** Basic ADP free; detailed export may require FantasyPros premium (~$30/year)
- **Verification:** **Strong Inference** — FantasyPros has provided Yahoo-specific ADP for many years
- **Fallback:** Use NFBC ADP (free, but based on 15-team expert leagues — adjust for 12-team) or Yahoo's own ADP from mock drafts

**3. Player Positional Eligibility**
- **What it provides:** Which positions each player qualifies for on Yahoo (e.g., a player eligible at SS and OF)
- **Why it matters:** Critical for roster construction constraints and positional scarcity calculations
- **Reliability:** High — determined by Yahoo based on prior-season games played at each position
- **Update cadence:** Set before the season; rarely changes
- **Cost/access:** Available from FanGraphs player pages; also from Yahoo Fantasy API league/players endpoint
- **Verification:** **Verified Fact** — standard fantasy baseball data
- **Fallback:** Manual lookup on Yahoo's player pages

**4. SGP Denominators**
- **What it provides:** Historical stat gaps per standings point for each category in 12-team 5x5 leagues
- **Why it matters:** Converts raw stats to standings value — the heart of the valuation engine
- **Reliability:** Medium-High — depends on using appropriate historical comparisons
- **Update cadence:** Published annually by BaseballHQ and Smart Fantasy Baseball
- **Cost/access:** BaseballHQ (subscription, ~$50/year), Smart Fantasy Baseball (free tools and articles)
- **Verification:** **Strong Inference** — SGP denominators for 2025 confirmed published (BaseballHQ Forums, accessed March 2026)
- **Fallback:** Use published 12-team 5x5 defaults. Or calculate from simulated leagues using the tool's own projection data.

#### Tier 2: Important (Strongly Recommended)

**5. Injury/IL Status**
- **What it provides:** Current injured list status, expected return dates
- **Why it matters:** Injured players should be flagged or discounted. Drafting an IL player without knowing wastes a pick.
- **Reliability:** High from MLB official sources
- **Update cadence:** Daily during spring training
- **Cost/access:** Free from MLB.com/injuries; also reflected in FanGraphs projections (players on IL get reduced playing time)
- **Verification:** **Verified Fact**
- **Fallback:** Manual check on MLB.com on draft day

**6. Yahoo Fantasy API (League Settings & Draft Results)**
- **What it provides:** League configuration (categories, roster slots, team count); during draft, returns picks made so far
- **Why it matters:** Automates league setup and can provide semi-automated draft tracking
- **Reliability:** Medium — API is functional but minimally maintained
- **Update cadence:** Real-time for league data; draft results available during active draft
- **Cost/access:** Free with OAuth 2.0 authentication via Yahoo Developer account
- **Verification:** **Verified Fact** — API documentation live at developer.yahoo.com; third-party wrappers (yfpy, yahoo-fantasy-api) actively maintained (GitHub, accessed March 2026). Draft results endpoint returns players drafted so far during an in-progress draft.
- **Fallback:** Manual entry of league settings; manual pick tracking during draft

**7. Closer/Bullpen Role Data**
- **What it provides:** Current closer designations, saves committee situations, bullpen hierarchies
- **Why it matters:** Saves are concentrated among ~25-30 closers; role changes dramatically affect RP value
- **Reliability:** Medium — closer roles are inherently volatile
- **Update cadence:** Changes frequently; check 1-2 days before draft
- **Cost/access:** Free from FantasyPros closer rankings, Roster Resource
- **Verification:** **Strong Inference**
- **Fallback:** Use FanGraphs saves projections as a proxy for closer role probability

#### Tier 3: Nice to Have

**8. Park Factors**
- **What it provides:** Adjustments for how each ballpark affects stats (HR, runs, etc.)
- **Why it matters:** Minor adjustment to player values based on home park
- **Reliability:** High from multi-year averages
- **Update cadence:** Annual; rarely changes dramatically
- **Cost/access:** Free from FanGraphs (fangraphs.com/guts.aspx?type=pf)
- **Verification:** **Verified Fact**
- **Note:** Most projection systems already incorporate park factors internally. Only apply separately if using raw rate stats, not blended projections.
- **Fallback:** Skip — double-counting risk if applied on top of projections that already include park factors

**9. Schedule/Platoon Data**
- **What it provides:** Projected platoon splits, 162-game schedule implications
- **Why it matters:** Minor for draft purposes; more relevant for in-season management
- **Reliability:** Low — schedules and lineups are not fixed pre-season
- **Cost/access:** Free from Baseball Reference, FanGraphs
- **Verification:** **Strong Inference**
- **Fallback:** Skip for draft tool; projections already incorporate expected playing time

---

## 11. Best Tech Stack and Build Medium

### Primary Architecture: Streamlit (Python)

**Streamlit is the recommended framework** based on evaluation of 8 alternatives across 6 criteria (development speed, reliability, UX quality, Python ecosystem access, deployment simplicity, live-draft suitability).

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Frontend** | Streamlit (Python) | Fastest development; declarative UI; built-in widgets; runs in browser tab for easy split-screen |
| **Backend** | Streamlit + Python standard library | Single-process; no frontend/backend split; native pandas/numpy/scipy |
| **Local database** | SQLite | Zero-config; Python standard library; stores projections, ADP, player data |
| **In-memory data** | pandas DataFrames | Fast filtering/sorting; native Streamlit integration via `st.dataframe` |
| **Computation** | NumPy + (optional) Numba JIT | Vectorized Monte Carlo; 300-500 sims in 5-15 seconds |
| **Caching** | `st.cache_data`, `st.cache_resource` | Prevents recomputation on UI reruns; supports session-scoped caching |
| **State management** | `st.session_state` | Persists draft board, roster, pick history across interactions |
| **Draft log backup** | JSON file (auto-saved after each pick) | Crash recovery; can reload mid-draft |
| **Partial reruns** | `st.fragment` decorator | Only re-renders changed UI sections; reduces latency on interaction |
| **Yahoo API client** | `yfpy` Python package | OAuth 2.0 Yahoo Fantasy API wrapper; league settings and draft results |

(Streamlit Docs, "2026 release notes," accessed March 2026; Streamlit Docs, "Working with fragments," accessed March 2026)

### Why Streamlit Over Alternatives

| Alternative | Why Rejected |
|-------------|-------------|
| **Electron + React** | 10x development time; no native Python data science stack; 150MB+ bundle |
| **Tauri + Svelte** | Rust learning curve; Python integration friction |
| **Next.js** | Wrong ecosystem; would need separate Python API server |
| **Flask/FastAPI + HTML** | 2-3x development time; requires frontend skill |
| **Gradio** | Designed for ML demos, not stateful dashboards |
| **Jupyter/Voila** | Poor UX for live use; fragile kernels |
| **Excel** | Cannot run Monte Carlo simulations; limited optimization |

### Backup Architecture: Flask + Jinja2

If Streamlit proves insufficient (e.g., rerun latency is unacceptable), the fallback is:
- Flask backend with Jinja2 server-rendered templates
- HTMX for dynamic partial updates (no full-page reloads)
- Same Python computation engine underneath
- More development work but full control over rendering

### Emergency Backup: Pre-Computed Cheat Sheet

Before draft day, the tool exports a static Excel/PDF cheat sheet containing:
- Tiered player rankings by position
- Category balance guidance
- "If you need X position, draft Y player" lookup tables
- Print this as the ultimate fallback

### Update Loop and Draft Board State Management

```
[Pre-Draft]
  1. Load SQLite database with projections, ADP, positional eligibility
  2. Compute base SGP values, replacement levels, tier breaks
  3. Cache all static computations

[During Draft — Event Loop]
  1. Poll Yahoo API every 10 seconds for new draft picks (if configured)
  2. On new pick detected (or manual entry):
     a. Update available_players, opponent_rosters
     b. If user's pick: update user_roster, category_totals
     c. Recalculate replacement levels
     d. If approaching user's next pick: trigger MC simulation
  3. On user's turn:
     a. Display cached recommendation (from pre-computation)
     b. If pre-computation is stale: run fast Tier 1 (greedy) ranking
     c. Show recommendation with category impact and alternatives
  4. After user picks:
     a. Save draft state to JSON backup
     b. Start pre-computing for next pick
```

### Export and Logging

- **Draft log:** Every pick (all 276) logged with timestamp, pick number, player, position, team
- **Recommendation log:** Every recommendation the tool made, with the actual pick taken
- **State snapshots:** JSON snapshot after each of the user's 23 picks
- **Post-draft export:** Full draft board as CSV for review

---

## 12. Live Yahoo Draft Workflow

### Pre-Draft Setup (Day Before or Morning Of)

1. **Install and verify the tool:**
   - Install Python 3.10+ (if not present)
   - `pip install streamlit pandas numpy scipy yfpy`
   - Run `streamlit run app.py` to verify it launches
   - Open in browser, confirm player data displays correctly

2. **Load data:**
   - Download latest FanGraphs projection CSVs (Steamer, ZiPS, Depth Charts — hitters and pitchers)
   - Run the data import script to load projections into SQLite
   - Download latest FantasyPros Yahoo ADP data
   - Check MLB.com/injuries for current IL status
   - Run the data import for ADP and injury status

3. **Configure league settings:**
   - If using Yahoo API: authenticate and pull league settings automatically
   - If manual: enter roster positions, categories, number of teams
   - Verify the tool displays correct roster structure

4. **Practice run:**
   - Enter 5-10 mock picks to practice the workflow
   - Verify recommendations appear within 5-15 seconds
   - Practice the split-screen layout

5. **Print backup cheat sheet:**
   - Export tiered rankings to Excel/PDF
   - Print a paper copy

### Draft Day Workflow (1 Hour Before)

1. **Refresh data:**
   - Re-run data loader to catch any last-minute injury news or ADP changes
   - Verify no key players have been placed on IL since last data load

2. **Set up screens:**
   - Open Yahoo draft room in Chrome/Firefox (left side of screen)
   - Open the Streamlit tool in another browser tab/window (right side)
   - Use Windows key + Left/Right arrow to snap side-by-side
   - Verify both are readable and accessible

3. **Enter draft position:**
   - When the draft room opens, enter your draft position (e.g., "Picking 7th overall") in the tool
   - The tool computes pick intervals and survival probabilities

4. **Wait for draft to start**

### During the Draft

**When opponents are picking:**
- **If using Yahoo API sync:** Picks auto-populate. Confirm each with a glance.
- **If using manual entry:** When a pick is announced in the Yahoo draft room, quickly enter it in the tool:
  - Type first few letters of player name → autocomplete dropdown → click to confirm
  - Target time: 3-5 seconds per pick entry
  - If you fall behind: skip entries temporarily; batch-enter missed picks during long opponent stretches

**When you are on the clock:**
1. The tool immediately shows the recommendation panel:
   - **Big green box:** "#1 Pick: [Player Name] ([Position])"
   - **Why:** "Fills your [position] need. Boosts [weak category] by [amount]. 75% chance they're taken before your next pick."
   - **Alternatives:** "#2: [Player B], #3: [Player C]"
2. **Decide:** Accept the recommendation or choose an alternative
3. **Draft in Yahoo:** Navigate to the Yahoo draft room, search for the player, click "Draft"
4. **Confirm in tool:** Click "I drafted [Player Name]" in the tool (or let API sync detect it)
5. The tool immediately begins pre-computing for your next pick

**If the clock is running low (<15 seconds):**
- The tool displays a **PANIC MODE** overlay:
  - Full-screen, large text: "DRAFT [PLAYER NAME] NOW"
  - All other UI hidden
  - One-click dismiss after drafting

**If the tool crashes:**
- Relaunch: `streamlit run app.py`
- The tool detects the saved draft state (JSON backup) and offers to resume
- Click "Resume Draft" — restores full state within seconds
- If relaunching takes too long: use the printed cheat sheet

### Handling Short Clocks

**Assumption:** Yahoo's default pick timer is approximately 90 seconds, but may be configured differently by the commissioner.

- Most opponent picks take 10-30 seconds each
- Between your picks (11 opponent picks in most rounds), you have 2-5 minutes
- Use this time to review the tool's pre-computed recommendation and alternatives
- When you go on the clock, you should already know your pick
- The tool is a confirmation/refinement device, not a "figure it out from scratch" device

### Keeping the Tool Reliable

- **No internet dependency during draft:** All projections and computations run locally
- **Auto-save after every pick:** JSON state file written synchronously
- **Minimal UI:** No unnecessary animations, charts, or decorations that could cause lag
- **Session state persistence:** Streamlit session state survives page reloads
- **Draft pause capability:** **Verified Fact** — Yahoo allows commissioners to pause the draft up to 20 times for up to 60 minutes each (Yahoo Help, "Manage your Private League's draft," accessed March 2026). If the tool has issues, the user can request a pause.

---

## 13. UX/UI Specification

### Screen Layout (Split-Screen Mode)

The tool is designed for a 700-900px wide viewport (right half of a laptop screen).

```
+----------------------------------------------------------+
|  [ROUND 5 / PICK 55]     [YOUR PICK: 7th in Round 5]    |
|  [Status: Waiting for Pick 52...]                         |
+----------------------------------------------------------+
|                                                           |
|  +-----------------------------------------------------+ |
|  |                                                       | |
|  |  ★ RECOMMENDED PICK                                  | |
|  |                                                       | |
|  |  BOBBY WITT JR.  (SS | KC)                           | |
|  |                                                       | |
|  |  "Best available SS. Your only open infield spot.     | |
|  |   65% chance he's gone by pick 62."                   | |
|  |                                                       | |
|  |  [DRAFT THIS PLAYER]  ← big green button              | |
|  +-----------------------------------------------------+ |
|                                                           |
|  Alternatives:                                            |
|  2. Trea Turner (SS | PHI) — similar value, more SB      |
|  3. Rafael Devers (3B | BOS) — fills 3B, boosts HR       |
|                                                           |
+----------------------------------------------------------+
|  CATEGORY BALANCE          | MY ROSTER (8/23 filled)     |
|  R:  ████████░░ 6th        | C: [empty]                  |
|  HR: ██████████ 3rd        | 1B: Freddie Freeman          |
|  RBI:████████░░ 5th        | 2B: [empty]                  |
|  SB: ███░░░░░░░ 10th ⚠️    | 3B: [empty]                  |
|  AVG:███████░░░ 5th        | SS: [empty] ← NEED           |
|  W:  ██████░░░░ 7th        | OF: Aaron Judge, [+2 empty]  |
|  SV: █████░░░░░ 8th        | Util: [empty] [empty]        |
|  K:  ████████░░ 4th        | SP: Zack Wheeler, [+1 empty] |
|  ERA:███████░░░ 5th        | RP: [empty] [empty]          |
|  WHIP:██████░░░ 6th        | P: [empty] x4                |
+----------------------------------------------------------+
|  QUICK ENTRY: [Type player name...] [Enter Opponent Pick]|
+----------------------------------------------------------+
```

### Panel Descriptions

**1. Recommended Pick Panel (Always Visible, Top of Screen)**
- Player name in large text (20px+)
- Position and team
- 1-2 sentence plain-English explanation (no jargon)
- Green "DRAFT THIS PLAYER" action button
- Urgency indicator (% chance they survive to next pick)

**2. Alternatives Panel**
- 2-3 fallback picks with brief justification
- Click to expand for full comparison

**3. Category Balance Dashboard**
- 10 horizontal bars showing projected standings rank per category
- Color coded: green (top 4), yellow (5-8), red (9-12)
- Warning icon on categories where the team is weak

**4. Roster Grid**
- Shows filled and empty positions
- Highlights positions that need to be filled soon (based on scarcity timing)
- "NEED" label on the most urgent unfilled position

**5. Quick Entry Bar (Bottom)**
- Text input with autocomplete for entering opponent picks
- Also serves as the entry point for the user's own pick if not using the recommendation button

### Additional Panels (Available via Tabs)

**6. Positional Scarcity Board (Tab)**
- For each position: how many viable players remain, tier break analysis
- "C: 3 startable options left — draft soon"
- "OF: 18 options — no rush"

**7. Expected Next-Turn Availability (Tab)**
- Top 20 available players ranked by survival probability
- Highlights players likely to be gone ("85% taken") vs. safe to wait ("20% taken")

**8. Queue / Watchlist (Tab)**
- User can pre-flag players they like
- Watchlist items highlighted when they appear in recommendations

**9. Danger Alerts**
- "Position run detected: 3 SS taken in last 5 picks"
- "Your weakest category (SB) has only 4 viable contributors remaining"
- "Key player you targeted was just drafted by another team"

**10. "Do Not Draft" Flags**
- Players on IL, suspended, or in minor leagues
- Players the user has manually flagged to avoid

**11. Emergency / Panic Mode**
- Triggered when clock < 15 seconds or user presses panic button
- Full-screen overlay: "[PLAYER NAME]" in 32px+ text, nothing else
- Auto-dismisses when pick is entered

---

## 14. Validation and Backtesting Plan

### Historical Replay Testing

**Method:** Simulate past drafts using historical projection data and ADP, then evaluate the tool's recommendations against what actually happened in the season.

**Data needed:**
- Historical projections from prior years (FanGraphs archives 2023-2025)
- Historical ADP data from prior years
- Actual season results (stats achieved by each player)
- Historical draft results from comparable leagues (if available)

**Process:**
1. Load a prior year's projections as if it were draft day
2. Simulate a 12-team draft using the tool's recommendation engine for the user's team
3. Simulate opponents using historical ADP
4. After the simulated draft, score the user's roster using actual season results
5. Compare the tool's draft against: (a) ADP-based drafting, (b) expert mock draft results, (c) actual winning teams from that year's roto leagues

**Success criterion:** The tool's drafted roster should score in the top 3 of the simulated 12-team league in >50% of replay simulations.

### Season-Level Retrospective Testing

Using actual season stats from 2023, 2024, and 2025:
1. Compute what the tool's valuation framework would have recommended pre-season
2. Build the optimal roster using the tool's algorithm
3. Compute the roster's standings points using actual season stats
4. Compare to the average fantasy team and to the championship-winning team

**Success criterion:** The tool's roster should achieve >85 standings points (out of 120 max) in retrospective analysis.

### Simulation Benchmarks

Run large-scale Monte Carlo validation:
1. Simulate 10,000 complete 12-team drafts
2. In each: one team uses the tool, 11 teams use ADP-based drafting
3. Measure: average standings finish, win rate, consistency

**Success criterion:** The tool-assisted team should finish in the top 3 in >40% of simulations and in the top 6 in >75%.

### Ablation Tests

Systematically remove each component and measure impact:

| Component Removed | Expected Impact |
|-------------------|----------------|
| Category balance weighting | Should worsen standings by 3-5 points (unbalanced roster) |
| Positional scarcity adjustment | Should worsen by 2-4 points (wrong position timing) |
| Monte Carlo simulation (use greedy only) | Should worsen by 1-3 points (misses reach/wait value) |
| Injury discount | Should worsen by 1-2 points (drafts injury-prone players) |
| Urgency/survival probability | Should worsen by 1-3 points (misses time-sensitive value) |

If removing a component does not worsen performance, it adds complexity without value and should be simplified.

### Calibration Tests

1. **SGP denominator sensitivity:** Vary denominators by ±20% and measure impact on recommendations. If recommendations change dramatically, the tool is fragile and denominators need better calibration.
2. **Risk aversion parameter:** Test lambda values from 0 to 0.5. Verify that higher lambda produces more conservative (lower ceiling, higher floor) rosters.
3. **Simulation count:** Compare recommendations at 100, 300, 500, 1000 simulations. Verify convergence (recommendations should stabilize by 300-500).

### Recommendation Quality Metrics

1. **Rank correlation:** Spearman correlation between the tool's draft-day player rankings and end-of-season actual rankings. Target: ρ > 0.65.
2. **Category balance score:** Standard deviation of projected standings ranks across 10 categories. Lower is better. Target: σ < 2.5.
3. **Position timing accuracy:** Correlation between when the tool recommends drafting each position and the optimal timing (determined retrospectively). Target: ρ > 0.60.

### Pre-Draft Validation Checklist

Before draft day, verify:

- [ ] All 2026 projection data loaded correctly (spot-check 10 players)
- [ ] ADP data is current (within 3 days of draft)
- [ ] SGP denominators produce reasonable valuations (top-10 players match expert consensus)
- [ ] Positional scarcity rankings match intuition (C and SS are scarce)
- [ ] Monte Carlo simulations complete within 15 seconds
- [ ] Category balance tracking updates correctly after adding players
- [ ] Panic mode triggers and displays correctly
- [ ] Draft state saves and restores from JSON backup
- [ ] Cheat sheet PDF exported and printed
- [ ] Yahoo API connection tested (if using API sync)
- [ ] Split-screen layout works on the user's laptop at the user's screen resolution

---

## 15. Failure Modes and Risk Controls

| # | Failure Mode | Severity | Mitigation |
|---|-------------|----------|------------|
| 1 | **Bad projections** — projection systems are systematically wrong about key players | Medium | Blend 3+ systems to reduce individual system risk; flag players with high inter-system variance |
| 2 | **Stale injury data** — a player goes on IL after data was loaded but before the draft | High | Check MLB.com injuries within 1 hour of draft; add manual "flag as injured" button in tool |
| 3 | **False assumptions about Yahoo mechanics** — e.g., assuming API provides live draft data when it doesn't | High | Verified via web search that API returns in-progress draft picks. Still maintain manual entry as primary fallback |
| 4 | **Too much complexity** — tool is slow, confusing, or buggy under pressure | High | Progressive disclosure UX; Tier 1 (greedy) recommendation always available in <500ms; panic mode for emergencies |
| 5 | **Fragile integrations** — Yahoo API breaks or requires re-authentication mid-draft | Medium | Manual entry as primary mode; API is a supplement, not a dependency |
| 6 | **Latency** — Monte Carlo simulations take too long | Medium | Pre-compute between picks; Tier 1 greedy always available instantly; limit to 300 sims if laptop is slow |
| 7 | **Overfitting** — tool over-optimizes for projected scenarios that don't match real draft | Medium | Sensitivity analysis on key parameters; use broad confidence intervals in simulations |
| 8 | **Hidden category imbalance** — tool produces a roster strong in 7 categories but terrible in 3 | Medium | Category balance dashboard with visual warnings; quadratic penalty for weak categories in objective function |
| 9 | **Incorrect positional logic** — misassigning player eligibility or miscounting roster slots | High | Validate positional eligibility against Yahoo data; unit tests for roster slot logic; manual override |
| 10 | **User input errors** — entering the wrong player name or missing picks | Medium | Autocomplete with fuzzy matching; "undo last entry" button; API sync as cross-check |
| 11 | **Tool crash mid-draft** | High | Auto-save after every pick; crash recovery from JSON state; printed cheat sheet as ultimate backup |
| 12 | **Opponent picks invalidate pre-computed recommendation** | Low | Detect when pre-computed state is stale; fall back to Tier 1 greedy if full recalculation hasn't finished |
| 13 | **Network issues** (if using API sync) | Low | Tool runs entirely locally except for optional API calls; all data pre-loaded |
| 14 | **SGP denominators miscalibrated** | Medium | Sensitivity test before draft; use published defaults with ±15% confidence band |
| 15 | **User overrides tool recommendation repeatedly** | None (feature) | Tool gracefully re-optimizes after any user choice; never penalizes the user for deviating |

---

## 16. Phased Build Plan

### Phase 1: Data Foundation
- **Objective:** Establish the data pipeline and storage layer
- **Deliverables:**
  - SQLite database schema (players, projections, ADP, positional eligibility, injury status)
  - Data import scripts for FanGraphs CSV files
  - ADP data import from FantasyPros
  - Projection blending logic (average of Steamer, ZiPS, Depth Charts)
- **Dependencies:** FanGraphs 2026 projections must be published (confirmed available)
- **If time is short:** Use FanGraphs Depth Charts only (already a blend) — skip manual multi-system blending

### Phase 2: Valuation Engine
- **Objective:** Implement the core SGP-based valuation with replacement levels and positional scarcity
- **Deliverables:**
  - SGP calculation module with configurable denominators
  - Replacement level computation per position
  - VORP calculation
  - Category marginal value computation (roster-conditional)
  - Rate stat volume-weighted adjustment
  - Injury discount application
- **Dependencies:** Phase 1 (data)
- **If time is short:** Skip injury discounts and variance penalties — use raw VORP only

### Phase 3: Simulation Engine
- **Objective:** Build the Monte Carlo draft simulation
- **Deliverables:**
  - Opponent pick probability model (ADP + noise + positional need)
  - Draft simulation loop (simulate full remaining draft)
  - Greedy SGP rollout policy for user's future picks
  - Survival probability calculator
  - Reach-vs-wait scoring
- **Dependencies:** Phase 2 (valuation engine provides scoring function)
- **If time is short:** Skip Monte Carlo entirely — use greedy marginal SGP with ADP-based survival estimates. This alone outperforms static rankings.

### Phase 4: Streamlit UI
- **Objective:** Build the user-facing interface
- **Deliverables:**
  - Recommendation panel (primary pick + alternatives)
  - Category balance dashboard
  - Roster grid with positional status
  - Quick entry bar with autocomplete
  - Draft state management (session state + JSON backup)
  - Panic mode overlay
- **Dependencies:** Phase 2 (valuation engine powers recommendations)
- **If time is short:** Minimal UI — text-based recommendations in a single Streamlit page with a text input. No charts, no tabs, no visual dashboard.

### Phase 5: Yahoo Integration
- **Objective:** Connect to Yahoo Fantasy API for league settings and draft tracking
- **Deliverables:**
  - OAuth 2.0 authentication flow using `yfpy`
  - League settings auto-import (categories, roster slots, teams)
  - Draft results polling (every 10 seconds during draft)
  - Automatic pick ingestion and state update
- **Dependencies:** Phase 4 (UI must exist to display auto-imported picks)
- **If time is short:** Skip entirely — use manual configuration and manual pick entry. The tool works fully without Yahoo API.

### Phase 6: Validation and Testing
- **Objective:** Verify the tool produces good recommendations
- **Deliverables:**
  - Historical replay test using 2025 projection data vs. actual results
  - Monte Carlo simulation benchmark (10,000 draft simulations)
  - Ablation tests for each valuation component
  - SGP denominator sensitivity analysis
  - Performance profiling (ensure <15 second recommendation latency)
- **Dependencies:** Phase 3 (simulation engine required for full testing; Phase 2 sufficient for basic testing)
- **If time is short:** Run 100 simulated drafts and verify the tool's team finishes top-4 more often than random. Spot-check 10 recommendations against expert consensus.

### Phase 7: Polish and Pre-Draft Prep
- **Objective:** Final preparation for draft day
- **Deliverables:**
  - Cheat sheet export (Excel/PDF)
  - Pre-draft checklist verification
  - Practice draft run-through
  - Bug fixes from testing
  - Documentation: 1-page "how to use" guide
- **Dependencies:** All prior phases
- **If time is short:** Export a tiered ranking cheat sheet and do one practice run. Fix only critical bugs.

### Minimum Viable Product (Phases 1-2 + Minimal Phase 4)

If only 2-3 days are available, the MVP is:
- Data loaded from FanGraphs CSVs
- SGP-based valuation with positional scarcity
- A simple Streamlit page showing:
  - Top 10 recommended players (sorted by marginal SGP)
  - Manual entry for each pick
  - Updated recommendations after each entry
- No Monte Carlo, no API integration, no fancy UI

This MVP alone substantially outperforms drafting by Yahoo's default rankings or gut feel, because it incorporates category-specific valuation, replacement-level adjustments, and positional scarcity — concepts that most novice drafters do not apply.

---

## 17. Final Recommendation

### Valuation Framework
**Use SGP-based valuation with dynamic replacement levels, roster-conditional category weights, and rate-stat volume adjustment.** This is the theoretically correct and empirically validated approach for 5x5 roto optimization. Blend projections from Steamer, ZiPS, and FanGraphs Depth Charts using simple equal-weight averaging.

### Simulation Framework
**Use Monte Carlo draft simulation (300-500 iterations) with ADP-based opponent modeling and greedy SGP rollout for the user's future picks.** This provides the reach-vs-wait intelligence that separates a true draft engine from a static ranking. NumPy vectorization keeps computation within the live-draft time budget (5-15 seconds on a modern laptop).

### Integration Approach
**Primary: Yahoo Fantasy API polling for draft state updates** (confirmed functional for in-progress drafts), **supplemented by manual entry with autocomplete** as the always-available fallback. Do not depend on the API — design the tool to work fully with manual entry alone.

### Tech Stack
**Streamlit (Python)** for the UI and application framework. **SQLite** for pre-draft data storage. **pandas/NumPy** for computation. **JSON** for draft state backup. This stack maximizes development speed, computation capability, and deployment simplicity.

### Live Workflow
Split-screen on a laptop: Yahoo draft room on the left, Streamlit tool on the right. Recommendations pre-computed between picks. "Big green button" for primary pick, alternatives listed below. Panic mode for short clocks. Printed cheat sheet as emergency backup.

### Minimum Viable Version
Phases 1-2 + minimal Phase 4: Data import, SGP valuation engine, simple Streamlit UI with manual entry and top-10 recommendations. No Monte Carlo, no API. Deliverable in 2-3 days. Still provides substantial edge over default rankings.

### Best Full-Feature Version
All 7 phases complete: Blended projections, full SGP/VORP/marginal valuation, Monte Carlo simulation with opponent modeling, category balance dashboard, Yahoo API sync, positional scarcity tracking, panic mode, crash recovery, backtested and validated. Deliverable in 7-10 days of focused development.

---

**This plan is complete and awaiting approval to begin implementation.**

All sections have been produced as specified. The plan is grounded in verified data sources, explicit mathematical formulas, and practical considerations for a novice user operating under live draft pressure. No code has been written. Upon approval, implementation will proceed according to the phased build plan in Section 16.

---

## Sources Referenced

- [Yahoo Help, "Default league settings in Fantasy Baseball"](https://help.yahoo.com/kb/default-league-settings-fantasy-baseball-sln6785.html), accessed March 2026
- [Yahoo Help, "Rotisserie scoring in Yahoo Fantasy"](https://help.yahoo.com/kb/rotisserie-scoring-sln6187.html), accessed March 2026
- [Yahoo Help, "Manage your Private League's draft"](https://help.yahoo.com/kb/SLN6086.html), accessed March 2026
- [Yahoo Sports, "Yahoo Fantasy Baseball 2026: New features"](https://sports.yahoo.com/fantasy/article/yahoo-fantasy-baseball-2026-play-new-features-162909923.html), accessed March 2026
- [Yahoo Developer Network, "Fantasy Sports API"](https://developer.yahoo.com/fantasysports/guide/), accessed March 2026
- [FanGraphs, "All the 2026 Projections Are In!"](https://blogs.fangraphs.com/all-the-2026-projections-are-in/), accessed March 2026
- [FanGraphs, "2026 Projections - Steamer"](https://www.fangraphs.com/projections), accessed March 2026
- [FanGraphs, "2026 Projections - ZiPS"](https://www.fangraphs.com/projections?pos=all&stats=bat&type=zips), accessed March 2026
- [BaseballHQ, "2025 Standings Gain Points"](https://stage.baseballhq.com/articles/gaming/rotisserie/2025-standings-gain-points), accessed March 2026
- [Smart Fantasy Baseball, "SGP Slope Calculator"](https://www.smartfantasybaseball.com/2015/02/excel-tool-sgp-slope-calculator/), accessed March 2026
- [Smart Fantasy Baseball, "Standings Gain Points" tag](https://www.smartfantasybaseball.com/tag/standings-gain-points/), accessed March 2026
- [FantasyPros, "2026 Fantasy Baseball Draft Assistant"](https://draftwizard.fantasypros.com/baseball/draft-assistant/), accessed March 2026
- [FantasyPros, "How does the Browser Extension help during my draft?"](https://support.fantasypros.com/hc/en-us/articles/360040970794), accessed March 2026
- [GitHub/uberfastman, "yfpy: Python API wrapper for Yahoo Fantasy Sports"](https://github.com/uberfastman/yfpy), accessed March 2026
- [Streamlit Docs, "2026 release notes"](https://docs.streamlit.io/develop/quick-reference/release-notes/2026), accessed March 2026
- [Streamlit Docs, "Working with fragments"](https://docs.streamlit.io/develop/concepts/architecture/fragments), accessed March 2026
- [Streamlit Docs, "st.fragment"](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), accessed March 2026
- [RotoWire, "What is 5x5 fantasy baseball?"](https://www.rotowire.com/betting/faq/what-is-5x5-fantasy-baseball-e5569ab1), accessed March 2026
- Alex Patton, "Patton$ Dollar$" — originated SGP framework for roto valuation
- Art McGee, "How to Value Players for Rotisserie Baseball" — definitive work on replacement level and SGP
- Tanner Bell and Jeff Zimmerman, "The Process" (2017) — projection blending and draft strategy
- Tom Tango, "Marcel the Monkey" projection system — baseline forecasting methodology
- Bertsekas, "Dynamic Programming and Optimal Control" (multiple editions) — approximate DP and rollout algorithms
- Hunter, Vielma, Zaman, "Picking Winners: A Framework for Fantasy Football Draft Optimization," MIT Sloan Sports Analytics Conference (2016)
- Jensen, "Simulating the Snake: An AI-Assisted Fantasy Football Draft Strategy," Medium (2024)
