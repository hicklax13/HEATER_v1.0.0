# Phase 3: Accuracy Refinements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 12 accuracy refinements from ROADMAP Tier 3. These are calibration fixes, signal improvements, and precision upgrades that build on Phase 1-2 infrastructure.

**Architecture:** 12 tasks in 3 parallel waves. Tasks target different pages/engines so agents don't conflict.

**Prerequisite:** Phase 1+2 complete. Full suite green.

**Total: 12 tasks, 3 waves, estimated 5-7 hours with parallel agents.**

---

## Wave 9: Trade Finder + Trade Analyzer Refinements — PARALLEL (4 agents)

| Agent | Task | ROADMAP | Files | Goal |
|-------|------|---------|-------|------|
| `trade-accept-odds` | Playoff-Odds Acceptance | F5 | `src/trade_finder.py` | Replace raw standings rank with playoff odds from `simulate_season_enhanced()`. Acceptance curve: <15% odds = 0.5x, 30-70% = 1.2x, >85% = 0.7x. |
| `trade-reliability` | Stat Reliability Weighting | G4 | `src/trade_finder.py` | `reliability = min(1.0, PA / threshold[stat])`. Apply to trade YTD modifier so K% at 60 PA is trusted more than AVG at 60 PA. Uses STABILIZATION_POINTS from bayesian.py. |
| `trade-grade-ci` | Trade Grade Confidence Interval | P1 | `src/engine/output/trade_evaluator.py` | Show grade RANGE ("B+ to A-") not single letter. Compute ±1 SD based on J6 empirical SDs (ERA=1.20, AVG=0.025). |
| `trade-time-decay` | Differential Time Decay | H1 | `src/trade_finder.py`, `src/trade_intelligence.py` | Counting stats × `weeks_rem / total_weeks`. Rate stats stay 1.0 with confidence penalty below 8 weeks remaining. |

---

## Wave 10: Lineup Optimizer Refinements — PARALLEL (4 agents)

| Agent | Task | ROADMAP | Files | Goal |
|-------|------|---------|-------|------|
| `lineup-ratio` | Ratio Protection Calculator | I3 | `src/optimizer/pivot_advisor.py`, `src/optimizer/daily_optimizer.py` | `marginal_era_risk = (proj_ER*9)/(banked_IP+proj_IP) - current_ERA`. If risk > ERA lead, recommend benching pitcher. Wire into Daily Optimize tab. |
| `lineup-platoon` | Platoon Split Bayesian Regression | J2 | `src/optimizer/matchup_adjustments.py` | Bayesian blend: `adjusted = (PA/stab)*individual + (1-PA/stab)*league_avg`. LHB stab=1000 PA, RHB stab=2200 PA. Never trust single-season individual splits. |
| `lineup-pitcher` | Opposing Pitcher Quality Calibration | J3 | `src/optimizer/matchup_adjustments.py` | Calibrate existing multiplier to ±15%: `mult = 1.0 + 0.15 * (league_avg_xFIP - opp_xFIP) / std_xFIP`. Verify against research magnitudes. |
| `lineup-weather` | Comprehensive Weather Model | J4 | `src/optimizer/matchup_adjustments.py`, `src/optimizer/daily_optimizer.py` | Temp→HR works. ADD: rain >40% chance → BB proj ×1.10, K proj ×0.90. Wind out >10mph → HR proj ×1.15. Open-Meteo data already fetched. |

---

## Wave 11: Standings + Closer + My Team — PARALLEL (4 agents)

| Agent | Task | ROADMAP | Files | Goal |
|-------|------|---------|-------|------|
| `standings-mc` | MC Simulation 10K + Bayesian SGP | R1, A3 | `src/standings_engine.py`, `src/valuation.py` | Default 1K→10K sims for Season Projections. Also wire Bayesian SGP updating: start with preseason denoms as prior, update weekly from standings. By week 8: 70% actual / 30% prior. |
| `standings-variance` | Per-Category Variance Calibration | R2 | `src/standings_engine.py` | Replace hardcoded WEEKLY_TAU with quantified weekly SDs from FanGraphs research: K=1.2, R=1.6, W=1.7, SV=1.8, WHIP=2.0, HR=2.1, RBI=2.3, SB=2.3, ERA=2.2, AVG=2.9. |
| `closer-decay` | K% Skill Decay Alert | L2 | `src/closer_monitor.py`, `pages/7_Closer_Monitor.py` | Rolling 14-day K% vs season average. K% drop ≥8 pts OR K-BB% <10% = "Skill Warning" flag. Display in closer cards. |
| `myteam-regression` | Regression Alert System | M4 | `pages/1_My_Team.py`, `src/alerts.py` | Compare L30 actual stats to expected (xBA, xwOBA from enriched pool). Flag >1.5 SD divergence as sell-high or buy-low. Weight K%/BB% changes over BA. Require 50 PA minimum. |

---

## Verification

After all waves:
- [ ] `python -m pytest tests/ -x -q` — all pass (2600+ expected)
- [ ] `python -m ruff check src/ tests/` — clean
- [ ] Trade Finder uses playoff-odds acceptance + stat reliability + time decay
- [ ] Lineup Optimizer has ratio protection + calibrated platoon/pitcher/weather
- [ ] Standings uses 10K sims + calibrated per-category variance
- [ ] Closer Monitor shows K% decay alerts
- [ ] My Team shows regression alerts (sell-high/buy-low)

---

## Agent Dispatch Summary

| Wave | Agent | Task | Time Est |
|------|-------|------|----------|
| **9** | `trade-accept-odds` | F5: Playoff-odds acceptance | 30 min |
| **9** | `trade-reliability` | G4: Stat reliability weighting | 25 min |
| **9** | `trade-grade-ci` | P1: Trade grade confidence interval | 30 min |
| **9** | `trade-time-decay` | H1: Differential time decay | 25 min |
| **10** | `lineup-ratio` | I3: Ratio protection calculator | 35 min |
| **10** | `lineup-platoon` | J2: Platoon Bayesian regression | 30 min |
| **10** | `lineup-pitcher` | J3: Opposing pitcher calibration | 20 min |
| **10** | `lineup-weather` | J4: Comprehensive weather model | 30 min |
| **11** | `standings-mc` | R1+A3: 10K sims + Bayesian SGP | 35 min |
| **11** | `standings-variance` | R2: Per-category variance | 20 min |
| **11** | `closer-decay` | L2: K% skill decay alert | 25 min |
| **11** | `myteam-regression` | M4: Regression alert system | 30 min |
