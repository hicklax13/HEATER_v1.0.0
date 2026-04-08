# HEATER Postmortem: Why Recommendations Failed (April 8, 2026)

## Context

- **Team:** Team Hickey (12th place, 0-2-0, likely 0-3)
- **League:** 12-team H2H Categories, snake draft, money league
- **Season start:** March 25, 2026
- **User followed ALL HEATER recommendations** since season start: trades, waiver pickups, drops, daily lineups, start/sit decisions

## The Core Problem (1 sentence)

HEATER was built as a season-long draft tool. Every recommendation it gives optimizes for "be better over 22 weeks" instead of "win 7 categories THIS WEEK against THIS OPPONENT."

That single design flaw poisons everything downstream — trades, waivers, lineups, start/sit, all of it.

---

## System-by-System Diagnosis

### 1. Valuation Engine (SGP/VORP) — Season-Long, Not Weekly

The entire app is built on SGP — Standings Gained Points. SGP asks: "How much does this player move me in the overall standings?" That's a great question for a Roto league or for draft day. It's the **wrong question** for weekly H2H. The right question is: "Does this player help me win SB against my opponent this week?" SGP never asks that.

- **File:** `src/valuation.py` (lines 118-166)
- `SGPCalculator.player_sgp()` and `total_sgp()` compute season-long standings gain points
- This is the foundation every other module builds on

### 2. Projections Are 95% Preseason — Actual 2026 Performance Barely Matters

The Bayesian stabilization thresholds require 910 PA to stabilize batting average and 170 PA for HR rate. After 2 weeks, hitters have ~50 PA. The system is still running on ~95% Steamer/ZiPS preseason projections.

- **File:** `src/bayesian.py` (lines 40-57)
- AVG: 910 PA to stabilize
- HR rate: 170 PA to stabilize
- At 50 PA, blend is approximately 95% preseason / 5% observed

The "recent form" adjustment added April 7 is capped at +/-3% effective change:

- **File:** `src/optimizer/projections.py` (lines 401-455)
- `_RECENT_FORM_BLEND = 0.20` (20% weight)
- Ratio clamped to +/-15%
- Effective max adjustment: `1 + 0.20 * 0.15 = 1.03` (3%)
- Hot streaks and cold streaks are almost completely invisible to the optimizer

### 3. Lineup Optimizer Barely Cares About Weekly Matchup

The `alpha` blend parameter starts at 0.3 H2H / 0.7 season-long early in the season. 70% of the decision weight is "who's the better player this season" and only 30% is "who helps me win categories this week."

- **File:** `src/optimizer/dual_objective.py` (line 163)
- Alpha = 0.3 when `weeks_remaining > 12`
- For a 0-2 team fighting for survival, this should be 0.8+ (nearly all weekly focus)

The H2H weighting layer silently falls back to uniform `{cat: 1.0}` weights when Yahoo matchup data doesn't load:

- **File:** `src/optimizer/pipeline.py` (lines 633-634)
- H2H weights only compute "if _H2H_AVAILABLE and h2h_opponent_totals and my_totals"
- If Yahoo data doesn't load cleanly, every category is treated equally

Category urgency (sigmoid scoring) only fires in "daily" mode (Stage 10), not in the standard weekly optimizer:

- **File:** `src/optimizer/category_urgency.py`
- Standard weekly optimizer falls back to alpha-blended roto/H2H weights

### 4. Trade Finder Too Conservative For a Last-Place Team

- **File:** `src/trade_finder.py`
- `ELITE_RETURN_FLOOR = 0.75` (line 672): Top-20% players require 75% SGP return. A 12th-place team should be willing to sell high to fill holes.
- `MAX_WEIGHT_RATIO = 1.5` (line 657): Category weight cap prevents the system from identifying biggest weaknesses and finding targeted trades.
- `LOSS_AVERSION = 1.8` (line 30): Self-rejects trades it predicts opponents won't accept. Too polite when the team needs to be desperate.
- `MAX_OPP_LOSS = -0.5` (line 29): Only surfaces "fair" trades. A last-place team needs to find lopsided value.
- Composite score is 25% SGP, 12% ADP, 12% ECR — mostly season-long metrics. No component asks "does this trade help me win this week?"

### 5. Waiver Wire Doesn't Know Who You're Playing

- **File:** `src/waiver_wire.py` (lines 483-637)
- `compute_add_drop_recommendations()` ranks FAs by net season-long SGP delta
- Never asks: "I'm losing HR 1-4 and SB 2-3. Which FA gives me the best shot at closing those gaps in the next 5 days?"
- `classify_category_priority()` uses season-long standings gaps, not weekly matchup context
- The only matchup-aware function (`recommend_streams()`) requires `opponent_profile` to be passed in, and even then only checks static `opponent_profile.get("weaknesses")`

### 6. `weeks_remaining` Hardcoded Wrong

Across 20+ function calls, `weeks_remaining` defaults to 16. In week 3 of a 24-week season, it should be ~21. This makes every urgency, decay, and threshold calculation think the season is a third over when it just started.

- Found in: `src/trade_finder.py`, `src/trade_intelligence.py`, `src/optimizer/pipeline.py`, `src/waiver_wire.py`, `src/start_sit.py`, `src/engine/output/trade_evaluator.py`, and many more

### 7. No "Desperation Mode"

There is no concept of team record context. A 0-2 team in 12th place gets the same conservative, season-long-optimized advice as a 10-2 team in 1st. No system adjusts behavior based on:

- Current W-L-T record
- League standings position
- How many weeks until playoffs
- Whether the team can afford to lose another week

### 8. Start/Sit Advisor — Actually Well-Designed But Under-Utilized

- **File:** `src/start_sit.py`
- Uses H2H category weights from actual matchup (line 375)
- Classifies matchup state (winning/losing/close) and adjusts risk tolerance (line 378)
- When losing, chases upside (alpha=0.2, weight toward P90 ceiling) (lines 155-157)
- BUT: only compares 2-4 players per slot, not roster-wide decisions
- Still relies on season projections converted to per-game rates

### 9. Alerts System Is Reactive, Not Strategic

- **File:** `src/alerts.py`, `src/war_room_actions.py`
- Checks for: empty spots, injuries, closer count, IL stash, league trades
- Does NOT: alert that you're losing SB 2-8 and should pick up a speed specialist, alert to bench risky SP to protect an ERA lead, alert to stream SPs when losing K, recommend specific lineup changes based on matchup state

---

## What Was Actually Working

The math in several modules is sound:

- H2H win probability engine (`src/optimizer/h2h_engine.py`)
- Category urgency sigmoid (`src/optimizer/category_urgency.py`)
- LP solver formulation (`src/lineup_optimizer.py`)
- Start/sit 3-layer decision model (`src/start_sit.py`)
- Trade intelligence correlation adjustments (`src/trade_intelligence.py`)

The problems aren't math bugs. The problem is that the right math is applied to the wrong objective.

---

## Required Fixes

### Critical (Must Fix)

1. **Weekly H2H mode** — Every recommendation engine needs a "weekly mode" that optimizes for category wins this week, not season-long value
2. **Actual performance weighting** — Raise recent form blend from 3% effective to 20-30% with unclamped ratios
3. **Matchup-aware waivers** — FA rankings driven by "which categories am I losing and who on the wire helps those in the next 5 days"
4. **Desperation mode for losing teams** — Aggressive alpha (0.8+ H2H), relaxed trade protections, streaming-heavy recommendations
5. **Dynamic `weeks_remaining`** — Compute from actual current week, not hardcoded 16

### High Priority

6. **Daily matchup state driving everything** — Winning ERA? Protect it (bench risky SP). Losing K by 3? Stream a starter.
7. **Trade finder aggression scaling** — Lower elite protection floor, raise category weight cap, relax acceptance filtering for losing teams
8. **Category-targeted FA pickups** — "You need SB this week, here are the 5 fastest FAs with games remaining"

### Medium Priority

9. **Performance feedback loop** — Track what was recommended vs what actually happened
10. **Roster-wide start/sit** — Optimize all positions simultaneously, not slot-by-slot

---

## Timeline

- **Season started:** March 25, 2026
- **Week 1 result:** Loss (record: 0-1-0)
- **Week 2 result:** Loss (record: 0-2-0)
- **Week 3 (current):** Losing 1-10 vs Jonny Jockstrap after Tuesday games
- **Postmortem date:** April 8, 2026
