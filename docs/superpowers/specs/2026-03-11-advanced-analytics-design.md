# Design Spec: Advanced Analytics & Yahoo Integration (Plan 3)

**Date:** 2026-03-11
**Status:** Pending Approval
**Scope:** 6 major features across 8 implementation phases

---

## 1. Problem Statement

The Fantasy Baseball Draft Tool currently has several capability gaps:

1. **No live league connection** — All data must be manually CSV-imported; no Yahoo Fantasy API integration
2. **Static projections** — Preseason projections never update with actual performance
3. **No injury risk modeling** — Healthy and injury-prone players valued equally
4. **No uncertainty quantification** — Single-point projections with no floor/ceiling
5. **No lineup optimization** — No automated start/sit decisions
6. **Basic opponent modeling** — Draft simulation uses ADP only, ignoring team tendencies

## 2. Solution Architecture

### 2.1 Yahoo Fantasy API Integration
- **Package:** `yfpy>=17.0` with `streamlit-oauth>=0.1.14` for browser-based OAuth 2.0
- **Flow:** User enters Yahoo Developer credentials → OAuth redirect → token exchange → auto-sync league data
- **Endpoints:** Rosters, standings, transactions, draft results, free agents, league settings
- **Fallback:** Manual CSV import remains fully functional when Yahoo not connected
- **Token management:** 1-hour expiry with auto-refresh, credentials stored in `yahoo_credentials.json` (gitignored)

### 2.2 Injury Risk Modeling
- **Health score:** 3-year average of `games_played / games_available` (0.0–1.0)
- **Age curves:** Position players +2%/yr risk after 30; pitchers +3%/yr after 28
- **Workload flags:** >40 IP increase year-over-year triggers overwork warning
- **Integration:** Counting stats multiplied by `health_score × age_risk`; rate stats get reduced PA/IP denominators
- **Data source:** MLB Stats API (games played per season, IL stints)

### 2.3 Bayesian Projection Updater
- **Primary:** PyMC 5 hierarchical beta-binomial model for rate stats
- **Fallback:** Marcel-style regression formula: `(observed × n + prior × stabilization) / (n + stabilization)`
- **Stabilization thresholds** (FanGraphs research):
  - K rate: 60 PA, BB rate: 120 PA, HR rate: 170 PA, AVG: 910 AB, ERA/WHIP: ~70 IP
- **Age adjustment:** Applied on logit scale to avoid boundary artifacts
- **Update trigger:** After daily MLB Stats API refresh, automatically re-compute ROS projections

### 2.4 Percentile Forecasts
- **Inter-projection volatility:** StdDev across Steamer, ZiPS, Depth Charts projections
- **Process risk:** Year-over-year stat correlations (HR r²=0.72, SB r²=0.55, AVG r²=0.41) widen intervals for volatile stats
- **Output:** P10 (floor), P50 (median), P90 (ceiling) for each player
- **MC integration:** Simulation optionally samples from percentile distributions per player

### 2.5 Lineup Optimization
- **Solver:** PuLP linear programming
- **Objective:** Maximize total marginal SGP across all 10 roto categories
- **Constraints:** Position eligibility, one player per slot, roster slot limits
- **Category targeting:** Weight categories by standings gap (small gap = high weight)
- **Two-start SP bonus:** Cross-reference MLB schedule for pitchers with 2 starts in scoring period

### 2.6 Enhanced Opponent Modeling
- **Current:** ADP-based pick probability only
- **Enhanced:** `P(pick) = 0.5 × ADP_rank + 0.3 × team_need + 0.2 × historical_preference`
- **History source:** Yahoo draft results API (when connected) or manual tracking
- **Positional bias:** Track each opponent's historical positional draft tendencies

## 3. Database Changes

### New Tables
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `injury_history` | 3-year health data per player | player_id, season, games_played, games_available, il_stints, il_days |
| `transactions` | League transaction log | transaction_id, league_id, player_id, type, team_from, team_to, timestamp |

### Modified Tables
| Table | Change |
|-------|--------|
| `projections` | Add `birth_date TEXT`, `mlb_id INTEGER` |
| `player_pool` | Add `health_score REAL DEFAULT 1.0`, `p10 REAL`, `p50 REAL`, `p90 REAL` |

## 4. New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/bayesian.py` | PyMC hierarchical models, Marcel fallback, age curves | ~250 |
| `src/injury_model.py` | Health scores, age risk, workload flags | ~150 |
| `src/lineup_optimizer.py` | PuLP LP model, category targeting, 2-start SP | ~200 |
| `pages/5_Lineup_Optimizer.py` | Streamlit UI for lineup optimization | ~120 |
| `tests/test_yahoo_api.py` | Yahoo API mock tests | ~120 |
| `tests/test_injury_model.py` | Injury model tests | ~80 |
| `tests/test_bayesian.py` | Bayesian updater tests | ~100 |
| `tests/test_percentiles.py` | Percentile forecast tests | ~70 |
| `tests/test_lineup_optimizer.py` | Lineup optimizer tests | ~90 |
| `tests/test_integration.py` | End-to-end pipeline tests | ~80 |

## 5. Modified Files

| File | Changes |
|------|---------|
| `requirements.txt` | Add pymc, arviz, yfpy, streamlit-oauth, PuLP |
| `src/database.py` | New tables, column additions |
| `src/yahoo_api.py` | Complete rewrite with yfpy v17 |
| `src/valuation.py` | Percentile computation functions, health_scores parameter |
| `src/simulation.py` | Percentile sampling, enhanced opponent modeling |
| `src/live_stats.py` | Bayesian update after stat refresh |
| `src/in_season.py` | load_best_projections() helper |
| `src/draft_state.py` | Draft history tracking, positional needs |
| `app.py` | Yahoo OAuth tab, injury badges, floor/ceiling badges |
| `.github/workflows/ci.yml` | New dependencies, test files |
| `.gitignore` | Yahoo credential exclusions |
| `load_sample_data.py` | Sample injury/draft history data |
| `CLAUDE.md` | Updated docs |

## 6. Dependencies & New Packages

| Package | Version | Purpose | Size Impact |
|---------|---------|---------|-------------|
| pymc | >=5.0 | Hierarchical Bayesian models | ~500MB (heavy) |
| arviz | >=0.15.0 | Bayesian model diagnostics/visualization | ~50MB |
| yfpy | >=17.0 | Yahoo Fantasy API wrapper | ~5MB |
| streamlit-oauth | >=0.1.14 | OAuth 2.0 flow in Streamlit | ~2MB |
| PuLP | >=2.7 | Linear programming solver | ~10MB |

**Note:** PyMC is the heaviest addition. The Marcel fallback ensures core functionality works without it (important for CI and lightweight deployments).

## 7. Validated Decisions

### Included (Research-Validated)
- **Bayesian updating with PyMC** — Gold standard for projection systems; Marcel fallback covers edge cases
- **FanGraphs stabilization thresholds** — Well-researched, widely cited benchmarks
- **PuLP for lineup optimization** — Lightweight, pure Python, fast solve times for roster-sized problems
- **yfpy for Yahoo API** — Most maintained Python Yahoo Fantasy wrapper, v17+ supports OAuth 2.0

### Excluded (Research-Validated as Overkill)
- **Copula correlation modeling** — Captures inter-stat correlations but LOW impact for 5x5 roto; simple independent modeling is 95% as effective
- **Dynamic programming for draft optimization** — State space is intractable (>10^15 states for 12-team, 23-round draft); existing Monte Carlo simulation is near-optimal

## 8. Testing Strategy

**Test count:** 22 existing → ~54 total (+32 new)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_yahoo_api.py` | ~8 | OAuth, sync, parsing, fallback |
| `test_injury_model.py` | ~6 | Health scores, age curves, integration |
| `test_bayesian.py` | ~8 | Regression, stabilization, aging, PyMC |
| `test_percentiles.py` | ~5 | Volatility, bounds, MC sampling |
| `test_lineup_optimizer.py` | ~7 | LP solver, constraints, targeting |
| `test_integration.py` | ~6 | End-to-end pipelines |

All tests must pass across Python 3.11, 3.12, 3.13 in CI.

## 9. Performance Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Full valuation pipeline | <10s | ~5s |
| MC simulation (100 sims) | <6s | ~4.5s |
| Lineup optimization | <2s | N/A |
| Bayesian update (full pool) | <30s | N/A |
| Yahoo API sync | <15s | N/A |

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PyMC 5 heavy (~500MB) | CI slow, deployment bloated | Marcel fallback; optional CI job for PyMC |
| Yahoo API rate limits | Sync failures | 15-min cache, graceful degradation |
| pybaseball data gaps | Missing projections | Multi-source fallback, handle NaN |
| PuLP infeasible solutions | Optimizer crashes | Feasibility check → greedy fallback |
| Token expiry mid-session | Auth errors | Auto-refresh with retry logic |
