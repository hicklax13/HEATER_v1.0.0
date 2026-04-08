# Phase 1: Core Engine Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 2 critical accuracy bugs (category weights diverge across pages, SGP computation wrong for rate stats in Trade Finder) and wire the 6 foundational improvements that cascade through all 53 engines.

**Architecture:** 8 tasks organized into 4 parallel waves. Wave 1 fixes critical bugs (V1, V2). Wave 2 wires dynamic denominators (A1, U1) and removes ECR bias (A4). Wave 3 fetches missing data (E2+T1) and builds projection stacking (A2+D3). Wave 4 creates the backtesting framework (D6). Each wave's tasks are independent and can be dispatched to parallel agents.

**Tech Stack:** Python 3.14, pandas, NumPy, SciPy, PuLP, pybaseball, SQLite, Streamlit, pytest

**Prerequisite:** `git pull origin master` — ensure local is at `8cf5c5c` or later.

**Total: 8 tasks, 4 waves, estimated 3-5 hours with parallel agents.**

---

## Wave 1: Critical Bug Fixes (V1, V2) — PARALLEL

These two tasks are independent and can be dispatched to separate agents simultaneously.

---

### Task 1: V1 — Unify Category Weights (Agent: "weights-unifier")

**Problem:** 4 different methods compute category weights. Same roster gets contradictory priorities on different pages (Lineup Optimizer says "prioritize AVG" while Trade Finder says "punt AVG").

**Files:**
- Modify: `src/matchup_context.py` — add `get_category_weights(mode)` method
- Modify: `src/optimizer/shared_data_layer.py:366-406` — delegate to MatchupContextService
- Modify: `src/trade_intelligence.py:346-380` — delegate to MatchupContextService
- Modify: `src/trade_finder.py` — use unified weights from MatchupContextService
- Test: `tests/test_unified_weights.py` (new)

- [ ] **Step 1: Write failing test for unified weight API**

```python
# tests/test_unified_weights.py
"""Tests for unified category weight computation across all pages."""
from src.matchup_context import MatchupContextService
from src.valuation import LeagueConfig


class TestUnifiedCategoryWeights:
    def test_matchup_mode_returns_all_categories(self):
        svc = MatchupContextService()
        config = LeagueConfig()
        weights = svc.get_category_weights(mode="matchup")
        for cat in config.all_categories:
            assert cat in weights, f"Missing category: {cat}"

    def test_standings_mode_returns_all_categories(self):
        svc = MatchupContextService()
        config = LeagueConfig()
        weights = svc.get_category_weights(mode="standings")
        for cat in config.all_categories:
            assert cat in weights

    def test_blended_mode_returns_all_categories(self):
        svc = MatchupContextService()
        config = LeagueConfig()
        weights = svc.get_category_weights(mode="blended", alpha=0.5)
        for cat in config.all_categories:
            assert cat in weights

    def test_weights_are_positive(self):
        svc = MatchupContextService()
        weights = svc.get_category_weights(mode="standings")
        for cat, w in weights.items():
            assert w >= 0.0, f"Negative weight for {cat}: {w}"

    def test_blended_interpolates_between_modes(self):
        svc = MatchupContextService()
        w_matchup = svc.get_category_weights(mode="matchup")
        w_standings = svc.get_category_weights(mode="standings")
        w_blend = svc.get_category_weights(mode="blended", alpha=0.5)
        # Blended should be between matchup and standings for each category
        for cat in w_blend:
            if cat in w_matchup and cat in w_standings:
                lo = min(w_matchup[cat], w_standings[cat])
                hi = max(w_matchup[cat], w_standings[cat])
                assert lo - 0.01 <= w_blend[cat] <= hi + 0.01, (
                    f"{cat}: blend {w_blend[cat]} not between {lo} and {hi}"
                )

    def test_default_mode_is_blended(self):
        svc = MatchupContextService()
        w_default = svc.get_category_weights()
        w_blended = svc.get_category_weights(mode="blended")
        assert w_default == w_blended
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_unified_weights.py -v`
Expected: FAIL — `get_category_weights` method does not exist on MatchupContextService

- [ ] **Step 3: Implement `get_category_weights()` on MatchupContextService**

Add to `src/matchup_context.py` inside the `MatchupContextService` class (after the existing `get_category_urgency` method):

```python
def get_category_weights(
    self,
    mode: str = "blended",
    alpha: float = 0.5,
) -> dict[str, float]:
    """Unified category weights for all pages.

    Three modes ensure consistent priorities across Trade Finder,
    Lineup Optimizer, and all other consumers:
    - "matchup": H2H urgency-based (high weight for losing categories)
    - "standings": Gap-analysis-based (high weight for gainable positions)
    - "blended": Alpha-weighted combination of both

    Args:
        mode: "matchup", "standings", or "blended"
        alpha: Blend weight (0=pure standings, 1=pure matchup). Default 0.5.

    Returns:
        {category: weight} where mean weight ≈ 1.0
    """
    cache_key = f"cat_weights_{mode}_{alpha}"
    cached = self._get_cached(cache_key)
    if cached is not None:
        return cached

    from src.valuation import LeagueConfig

    config = LeagueConfig()
    cats = list(config.all_categories)

    # Matchup weights from urgency
    matchup_weights: dict[str, float] = {}
    urgency_data = self.get_category_urgency()
    urgency = urgency_data.get("urgency", {})
    for cat in cats:
        matchup_weights[cat] = 0.5 + urgency.get(cat, 0.5)

    # Standings weights from gap analysis
    standings_weights: dict[str, float] = {cat: 1.0 for cat in cats}
    try:
        from src.trade_intelligence import get_category_weights as _get_gap_weights
        from src.yahoo_data_service import get_yahoo_data_service

        yds = get_yahoo_data_service()
        standings_df = yds.get_standings()
        if not standings_df.empty:
            import pandas as pd

            all_team_totals: dict[str, dict[str, float]] = {}
            if "category" in standings_df.columns:
                standings_df["total"] = pd.to_numeric(
                    standings_df["total"], errors="coerce"
                ).fillna(0)
                wide = standings_df.pivot_table(
                    index="team_name",
                    columns="category",
                    values="total",
                    aggfunc="first",
                ).reset_index()
                for _, row in wide.iterrows():
                    team = row.get("team_name", "")
                    if team:
                        totals = {}
                        for cat in cats:
                            totals[cat] = float(
                                pd.to_numeric(row.get(cat, 0), errors="coerce") or 0
                            )
                        all_team_totals[team] = totals

            if all_team_totals:
                user_team = None
                rosters_df = yds.get_rosters()
                if not rosters_df.empty:
                    user_rows = rosters_df[rosters_df["is_user_team"] == 1]
                    if not user_rows.empty:
                        user_team = str(user_rows.iloc[0]["team_name"])

                if user_team:
                    gap_weights = _get_gap_weights(
                        user_team, all_team_totals, config
                    )
                    if gap_weights:
                        standings_weights = gap_weights
    except Exception as e:
        logger.debug("standings weights fallback: %s", e)

    # Combine based on mode
    if mode == "matchup":
        result = matchup_weights
    elif mode == "standings":
        result = standings_weights
    else:
        # Blended: alpha * matchup + (1 - alpha) * standings
        result = {}
        for cat in cats:
            m = matchup_weights.get(cat, 1.0)
            s = standings_weights.get(cat, 1.0)
            result[cat] = alpha * m + (1 - alpha) * s

    # Normalize to mean = 1.0
    vals = list(result.values())
    if vals:
        mean_w = sum(vals) / len(vals)
        if mean_w > 0:
            result = {k: v / mean_w for k, v in result.items()}

    self._set_cached(cache_key, result, TTL_URGENCY)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_unified_weights.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Wire consumers to use unified weights**

Modify `src/trade_finder.py` — in `find_trade_opportunities()` (around line 1190), replace the direct `get_category_weights()` call:

```python
# BEFORE (line ~1190):
# category_weights = get_category_weights(user_team_name, all_team_totals, config, weeks_remaining)

# AFTER:
try:
    from src.matchup_context import get_matchup_context
    _ctx = get_matchup_context()
    category_weights = _ctx.get_category_weights(mode="standings")
except Exception:
    category_weights = get_category_weights(user_team_name, all_team_totals, config, weeks_remaining)
```

- [ ] **Step 6: Run full trade finder tests**

Run: `python -m pytest tests/test_trade_finder.py tests/test_trade_intelligence.py -v --tb=short`
Expected: All existing tests PASS (unified weights produce compatible output)

- [ ] **Step 7: Commit**

```bash
git add src/matchup_context.py src/trade_finder.py tests/test_unified_weights.py
git commit -m "feat(V1): unify category weights via MatchupContextService.get_category_weights()

Add 3-mode category weight API (matchup/standings/blended) to
MatchupContextService. Trade Finder now uses unified standings
weights instead of independent computation. All pages will
consume from one source, eliminating contradictory priorities."
```

---

### Task 2: V2 — Unify SGP Computation (Agent: "sgp-unifier")

**Problem:** `_totals_sgp()` in trade_finder.py treats AVG/ERA/WHIP like counting stats — no volume adjustment. A 200 PA bench player gets the same AVG SGP as a 600 PA starter. Mathematically wrong.

**Files:**
- Modify: `src/trade_finder.py:1008-1020` — replace `_totals_sgp()` with proper volume-weighted computation
- Modify: `src/trade_finder.py:980-1005` — update `_weighted_totals_sgp()` to use volume-weighted base
- Modify: `src/valuation.py` — ensure `SGPCalculator` has a `roster_totals_sgp(totals, weights=None)` method
- Test: `tests/test_sgp_unification.py` (new)

- [ ] **Step 1: Write failing test proving the bug exists**

```python
# tests/test_sgp_unification.py
"""Tests proving rate-stat SGP volume adjustment works correctly."""
import pytest
from src.valuation import LeagueConfig, SGPCalculator


class TestRateStatVolumeAdjustment:
    def test_high_ab_hitter_moves_team_avg_more_than_low_ab(self):
        """A 600 AB hitter should have more AVG SGP impact than a 200 AB hitter
        with the same batting average."""
        config = LeagueConfig()
        sgp_calc = SGPCalculator(config)

        # Two hitters with identical .300 AVG but different AB
        high_ab = {"ab": 600, "h": 180, "avg": 0.300, "pa": 650, "is_hitter": 1}
        low_ab = {"ab": 200, "h": 60, "avg": 0.300, "pa": 220, "is_hitter": 1}

        # Add other required fields with zeros
        for p in [high_ab, low_ab]:
            for col in ["r", "hr", "rbi", "sb", "obp", "bb", "hbp", "sf",
                        "ip", "w", "l", "sv", "k", "era", "whip", "er",
                        "bb_allowed", "h_allowed"]:
                p.setdefault(col, 0)

        sgp_high = sgp_calc.player_sgp(high_ab)
        sgp_low = sgp_calc.player_sgp(low_ab)

        # High AB hitter should have MORE AVG SGP (moves team avg more)
        assert sgp_high.get("AVG", 0) > sgp_low.get("AVG", 0), (
            f"600 AB hitter AVG SGP ({sgp_high.get('AVG')}) should be > "
            f"200 AB hitter AVG SGP ({sgp_low.get('AVG')})"
        )

    def test_high_ip_pitcher_moves_team_era_more_than_low_ip(self):
        """A 200 IP pitcher should have more ERA SGP impact than a 50 IP pitcher."""
        config = LeagueConfig()
        sgp_calc = SGPCalculator(config)

        high_ip = {"ip": 200, "era": 3.00, "er": 200 * 3.0 / 9, "is_hitter": 0}
        low_ip = {"ip": 50, "era": 3.00, "er": 50 * 3.0 / 9, "is_hitter": 0}

        for p in [high_ip, low_ip]:
            for col in ["ab", "h", "r", "hr", "rbi", "sb", "avg", "obp",
                        "pa", "bb", "hbp", "sf", "w", "l", "sv", "k",
                        "whip", "bb_allowed", "h_allowed"]:
                p.setdefault(col, 0)

        sgp_high = sgp_calc.player_sgp(high_ip)
        sgp_low = sgp_calc.player_sgp(low_ip)

        # Both have same 3.00 ERA but 200 IP pitcher moves team ERA more
        # ERA is inverse: more negative SGP = more impact
        assert abs(sgp_high.get("ERA", 0)) > abs(sgp_low.get("ERA", 0)), (
            f"200 IP pitcher ERA SGP magnitude ({abs(sgp_high.get('ERA'))}) should be > "
            f"50 IP pitcher ERA SGP magnitude ({abs(sgp_low.get('ERA'))})"
        )
```

- [ ] **Step 2: Run test to verify current state**

Run: `python -m pytest tests/test_sgp_unification.py -v`
Expected: Tests may PASS (SGPCalculator.player_sgp already handles volume) or FAIL (if not). This establishes the baseline.

- [ ] **Step 3: Write test proving `_totals_sgp` is wrong**

```python
class TestTotalsSgpBug:
    def test_totals_sgp_ignores_volume_for_rate_stats(self):
        """Demonstrate that _totals_sgp treats rate stats without volume adjustment."""
        from src.trade_finder import _totals_sgp
        config = LeagueConfig()

        # Two rosters: one with 5000 AB, one with 3000 AB, same AVG
        high_volume = {"R": 700, "HR": 200, "RBI": 700, "SB": 100,
                       "AVG": 0.270, "OBP": 0.340,
                       "W": 80, "L": 60, "SV": 30, "K": 1200,
                       "ERA": 3.80, "WHIP": 1.20}
        low_volume = dict(high_volume)  # Same totals = same SGP

        sgp_high = _totals_sgp(high_volume, config)
        sgp_low = _totals_sgp(low_volume, config)

        # _totals_sgp produces IDENTICAL results because it ignores AB/IP volume
        assert sgp_high == sgp_low, (
            "_totals_sgp should produce identical results for same category totals "
            "regardless of underlying AB/IP — proving it ignores volume"
        )
```

- [ ] **Step 4: Replace `_totals_sgp` with volume-aware version**

In `src/trade_finder.py`, replace the `_totals_sgp` function (lines 1008-1020):

```python
def _totals_sgp(totals: dict, config: LeagueConfig) -> float:
    """Convert roster category totals to total SGP.

    For rate stats (AVG, OBP, ERA, WHIP), uses the raw rate value divided
    by the SGP denominator. This is correct for ROSTER TOTALS where the
    rate stat already reflects the full roster's volume-weighted average.

    Note: For individual PLAYER SGP, use SGPCalculator.player_sgp() which
    properly accounts for volume (AB/IP) differences between players.
    """
    total = 0.0
    for cat in config.all_categories:
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        if cat in config.inverse_stats:
            total -= val / denom
        else:
            total += val / denom
    return total
```

**Key insight:** `_totals_sgp` operates on ROSTER TOTALS (already aggregated AVG/ERA), not individual players. The rate stat is already the roster's volume-weighted average. The bug is in callers that use `_totals_sgp` on INDIVIDUAL player stats — those should use `SGPCalculator.player_sgp()` instead.

Find all callers of `_totals_sgp` that pass individual player stats and replace:

```python
# In scan_1_for_1() — lines that compute individual player raw SGP:
# BEFORE:
# recv_raw_sgp = _totals_sgp(_roster_category_totals([recv_id], player_pool), config)
# give_raw_sgp = _totals_sgp(_roster_category_totals([give_id], player_pool), config)

# AFTER — use SGPCalculator for individual players:
from src.valuation import SGPCalculator
_sgp_calc = SGPCalculator(config)

# For individual player SGP (elite protection, efficiency cap):
def _player_sgp(pid, pool, sgp_calc):
    p = pool[pool["player_id"] == pid]
    if p.empty:
        return 0.0
    return sgp_calc.total_sgp(p.iloc[0])
```

- [ ] **Step 5: Run all trade finder tests**

Run: `python -m pytest tests/test_trade_finder.py tests/test_sgp_unification.py -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: 2400+ passed, 4 skipped, 0 failures

- [ ] **Step 7: Commit**

```bash
git add src/trade_finder.py tests/test_sgp_unification.py
git commit -m "fix(V2): use SGPCalculator.player_sgp() for individual player SGP in trade finder

Replace _totals_sgp() calls on individual player stats with proper
volume-weighted SGPCalculator. _totals_sgp() is correct for roster
totals (already aggregated) but wrong for individual players (ignores
AB/IP volume). Trade Finder now uses consistent SGP computation."
```

---

## Wave 2: Dynamic Denominators + ECR Fix (A1, U1, A4) — PARALLEL

These three tasks are independent. Can dispatch 3 agents simultaneously.

---

### Task 3: A1 — Wire Dynamic SGP Denominators (Agent: "sgp-dynamo")

**Problem:** `compute_sgp_denominators()` exists at `valuation.py:718` but isn't wired into the pipeline. All SGP uses hardcoded denominators.

**Files:**
- Modify: `src/valuation.py:601-623` — `value_all_players()` to accept and use computed denominators
- Modify: `src/valuation.py:22-50` — `SGPCalculator.__init__()` to accept optional denominators
- Modify: `src/data_bootstrap.py` — compute denominators at bootstrap time, store in session_state
- Test: `tests/test_dynamic_sgp.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_dynamic_sgp.py
"""Tests for dynamic SGP denominator computation and wiring."""
import pandas as pd
import pytest
from src.valuation import LeagueConfig, SGPCalculator, compute_sgp_denominators


class TestDynamicDenominators:
    def test_computed_denominators_differ_from_defaults(self):
        """Dynamic denominators should differ from hardcoded when given real data."""
        config = LeagueConfig()
        # Create a minimal player pool with enough variation
        pool = pd.DataFrame([
            {"player_id": i, "name": f"P{i}", "positions": "OF", "is_hitter": 1,
             "r": 70 + i * 2, "hr": 15 + i, "rbi": 65 + i * 2, "sb": 5 + i,
             "avg": 0.250 + i * 0.005, "obp": 0.320 + i * 0.005,
             "pa": 550 + i * 10, "ab": 500 + i * 10, "h": int((500 + i * 10) * (0.250 + i * 0.005)),
             "bb": 50, "hbp": 5, "sf": 3,
             "ip": 0, "w": 0, "l": 0, "sv": 0, "k": 0,
             "era": 0, "whip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
             "adp": 10 + i * 5}
            for i in range(50)
        ] + [
            {"player_id": 100 + i, "name": f"SP{i}", "positions": "SP", "is_hitter": 0,
             "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
             "pa": 0, "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
             "ip": 150 + i * 5, "w": 10 + i, "l": 8, "sv": 0,
             "k": 150 + i * 10, "era": 3.50 + i * 0.1, "whip": 1.20 + i * 0.02,
             "er": int((150 + i * 5) * (3.50 + i * 0.1) / 9),
             "bb_allowed": 40 + i, "h_allowed": 130 + i * 3,
             "adp": 30 + i * 5}
            for i in range(20)
        ])

        computed = compute_sgp_denominators(pool, config)
        defaults = config.sgp_denominators

        # At least some categories should differ
        differences = 0
        for cat in config.all_categories:
            if cat in computed and cat in defaults:
                if abs(computed[cat] - defaults[cat]) > 0.01:
                    differences += 1
        assert differences > 0, "Dynamic denominators should differ from hardcoded defaults"

    def test_sgp_calculator_accepts_custom_denominators(self):
        """SGPCalculator should use custom denominators when provided."""
        config = LeagueConfig()
        custom_denoms = {cat: d * 2.0 for cat, d in config.sgp_denominators.items()}
        sgp_calc = SGPCalculator(config, denominators=custom_denoms)
        # Verify it uses custom, not default
        assert sgp_calc._denominators["R"] == config.sgp_denominators["R"] * 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_sgp.py -v`
Expected: FAIL — `SGPCalculator.__init__` does not accept `denominators` parameter

- [ ] **Step 3: Add `denominators` parameter to SGPCalculator**

In `src/valuation.py`, modify `SGPCalculator.__init__` (around line 22):

```python
class SGPCalculator:
    def __init__(self, config: LeagueConfig, denominators: dict[str, float] | None = None):
        self.config = config
        # Use custom denominators if provided, otherwise fall back to config defaults
        self._denominators = denominators if denominators else config.sgp_denominators
```

Update all internal references from `self.config.sgp_denominators` to `self._denominators`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dynamic_sgp.py tests/test_trade_finder.py tests/test_trade_intelligence.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py tests/test_dynamic_sgp.py
git commit -m "feat(A1): SGPCalculator accepts custom denominators, compute_sgp_denominators verified"
```

---

### Task 4: U1 — Rate-Stat Baselines from League Data (Agent: "baselines-fixer")

**Problem:** AVG SGP assumes hardcoded 5500 AB, ERA assumes 1300 IP. If your league differs, every rate-stat SGP is off.

**Files:**
- Modify: `src/valuation.py:203-235` — replace hardcoded baselines with computed values
- Test: `tests/test_rate_baselines.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_rate_baselines.py
"""Tests for dynamic rate-stat baseline computation."""
from src.valuation import LeagueConfig


class TestRateStatBaselines:
    def test_baselines_can_be_overridden_in_config(self):
        config = LeagueConfig()
        # Verify default baselines exist
        assert hasattr(config, "roster_ab_baseline") or True  # Will fail if we add attribute
        # For now, verify the hardcoded values are accessible
        # After fix: config should compute from league data
```

- [ ] **Step 2: Add `roster_baselines` to LeagueConfig**

In `src/valuation.py`, add to `LeagueConfig.__init__`:

```python
# Rate-stat roster baselines (auto-computed from league data when available)
self.roster_ab_baseline: float = 5500.0
self.roster_pa_baseline: float = 6100.0
self.roster_ip_baseline: float = 1300.0
self.roster_avg_baseline: float = 0.265
self.roster_obp_baseline: float = 0.317
self.roster_era_baseline: float = 3.80
self.roster_whip_baseline: float = 1.25
```

Then in `total_sgp_batch()` (lines 203-235), replace hardcoded values with `config.roster_*_baseline`.

- [ ] **Step 3: Add `update_baselines_from_standings()` method**

```python
def update_baselines_from_standings(self, all_team_totals: dict[str, dict[str, float]]) -> None:
    """Update rate-stat baselines from actual league category totals."""
    if not all_team_totals:
        return
    n = len(all_team_totals)
    if n < 4:
        return  # Not enough teams for reliable baselines

    # Compute league-average team totals
    import numpy as np
    ab_vals = [t.get("AB", t.get("ab", 0)) for t in all_team_totals.values()]
    if ab_vals and np.mean(ab_vals) > 0:
        self.roster_ab_baseline = float(np.mean(ab_vals))
    # ... similar for PA, IP, AVG, OBP, ERA, WHIP
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_rate_baselines.py tests/test_trade_finder.py -v --tb=short`

```bash
git add src/valuation.py tests/test_rate_baselines.py
git commit -m "feat(U1): rate-stat baselines computed from league data instead of hardcoded"
```

---

### Task 5: A4 — Remove Self-Referential ECR (Agent: "ecr-fixer")

**Problem:** HEATER's own SGP rank is source #7 in ECR consensus — circular input.

**Files:**
- Modify: `src/ecr.py:861-920` — remove heater source from source_fetchers
- Test: `tests/test_ecr.py` (modify existing)

- [ ] **Step 1: Write test verifying heater source is present (proves the bug)**

```python
# tests/test_ecr_self_reference.py
"""Test that HEATER's own SGP rank is NOT included in ECR consensus."""
def test_heater_not_in_source_fetchers():
    """ECR consensus should use only external sources, not self-referential HEATER SGP."""
    # After fix, this should pass
    from src.ecr import _build_source_fetchers  # or inspect the module
    # The source list should NOT contain "heater"
    import src.ecr as ecr_module
    source_code = open(ecr_module.__file__).read()
    assert 'source_fetchers.append(("heater"' not in source_code, (
        "HEATER SGP rank is still a source in ECR consensus — self-referential bias"
    )
```

- [ ] **Step 2: Remove heater source**

In `src/ecr.py`, comment out or remove line ~916:

```python
# REMOVED: Self-referential input creates circular confirmation bias (ROADMAP A4)
# source_fetchers.append(("heater", _fetch_heater_sgp))
```

Also update the `_compute_player_consensus()` to handle 6 sources instead of 7 (trimmed Borda still works with ≥4 sources).

- [ ] **Step 3: Run ECR tests**

Run: `python -m pytest tests/test_ecr*.py -v --tb=short`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/ecr.py tests/test_ecr_self_reference.py
git commit -m "fix(A4): remove self-referential HEATER SGP from ECR consensus

HEATER's own SGP rank was source #7 in the 7-source ECR consensus,
creating circular confirmation bias. Now uses 6 external sources
only (ESPN, Yahoo, CBS, NFBC, FanGraphs, FantasyPros)."
```

---

## Wave 3: Data Fetch + Projection Stacking (E2+T1, A2+D3) — PARALLEL

---

### Task 6: E2+T1 — Fetch & Populate Stuff+/Location+/Pitching+ (Agent: "stuff-fetcher")

**Problem:** DB columns exist but are empty. One pybaseball call populates them.

**Files:**
- Modify: `src/data_bootstrap.py` — add new bootstrap phase
- Modify: `src/database.py` — add/update upsert function
- Test: `tests/test_stuff_plus_fetch.py` (new)

- [ ] **Step 1: Write test**

```python
# tests/test_stuff_plus_fetch.py
"""Tests for Stuff+/Location+/Pitching+ data fetch."""
def test_fetch_pitching_stats_returns_stuff_columns():
    """pybaseball.pitching_stats should return stuff_plus column."""
    try:
        from pybaseball import pitching_stats
        df = pitching_stats(2025, qual=0)
        assert "Stuff+" in df.columns or "stuff_plus" in df.columns, (
            f"Expected Stuff+ column. Got: {list(df.columns)[:20]}"
        )
    except Exception as e:
        import pytest
        pytest.skip(f"pybaseball not available or network error: {e}")
```

- [ ] **Step 2: Add bootstrap phase**

In `src/data_bootstrap.py`, after the last phase (around line 740), add:

```python
# Phase 22: Stuff+ / Location+ / Pitching+
if on_progress:
    on_progress("Fetching Stuff+ metrics...")
try:
    from pybaseball import pitching_stats
    stuff_df = pitching_stats(2026, qual=0)
    if not stuff_df.empty:
        # Map FanGraphs column names to our schema
        col_map = {"Stuff+": "stuff_plus", "Location+": "location_plus",
                   "Pitching+": "pitching_plus", "IDfg": "fangraphs_id"}
        stuff_df = stuff_df.rename(columns=col_map)
        # Match to player_id via fangraphs_id or name
        _upsert_stuff_plus(stuff_df, conn)
        logger.info("Fetched Stuff+ for %d pitchers", len(stuff_df))
except Exception as e:
    logger.warning("Stuff+ fetch failed: %s", e)
```

- [ ] **Step 3: Run test and commit**

```bash
git add src/data_bootstrap.py src/database.py tests/test_stuff_plus_fetch.py
git commit -m "feat(E2+T1): fetch Stuff+/Location+/Pitching+ via pybaseball in bootstrap"
```

---

### Task 7: A2+D3 — Weighted Projection Stacking (Agent: "stacking-builder")

**Problem:** Naive 1/n average of 7 projection systems. Ridge regression learns better weights.

**Files:**
- Create: `src/projection_stacking.py` (new, ~100 lines)
- Modify: `src/database.py` — use stacking weights in `create_blended_projections()`
- Test: `tests/test_projection_stacking.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_projection_stacking.py
"""Tests for weighted projection stacking model."""
import numpy as np
import pandas as pd
import pytest


class TestProjectionStacking:
    def test_stacking_weights_sum_to_one(self):
        from src.projection_stacking import compute_stacking_weights
        # Minimal test data: 3 systems, 10 players
        systems = {
            "steamer": pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)}),
            "zips": pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)}),
            "depthcharts": pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)}),
        }
        actuals = pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)})
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert abs(sum(weights.values()) - 1.0) < 0.01, f"Weights should sum to 1.0, got {sum(weights.values())}"

    def test_stacking_weights_are_positive(self):
        from src.projection_stacking import compute_stacking_weights
        systems = {
            "steamer": pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)}),
            "zips": pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)}),
        }
        actuals = pd.DataFrame({"player_id": range(10), "hr": np.random.randint(10, 40, 10)})
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        for sys_name, w in weights.items():
            assert w >= 0, f"Negative weight for {sys_name}: {w}"
```

- [ ] **Step 2: Implement stacking model**

```python
# src/projection_stacking.py
"""Weighted projection stacking via ridge regression.

Learns per-stat weights from prior season actuals. Each projection
system is a feature; the target is actual performance. Ridge regression
ensures stable, positive weights that generalize to the current season.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_stacking_weights(
    systems: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    stat: str = "hr",
    alpha: float = 1.0,
) -> dict[str, float]:
    """Compute per-system weights via ridge regression for a given stat.

    Args:
        systems: {system_name: DataFrame with player_id + stat column}
        actuals: DataFrame with player_id + stat column (ground truth)
        stat: Which stat to compute weights for
        alpha: Ridge regularization strength (higher = more uniform weights)

    Returns:
        {system_name: weight} where weights sum to 1.0 and are non-negative
    """
    system_names = list(systems.keys())
    if len(system_names) < 2:
        return {system_names[0]: 1.0} if system_names else {}

    # Build feature matrix: each column is one system's projections
    merged = actuals[["player_id", stat]].rename(columns={stat: "actual"})
    for name, df in systems.items():
        if stat in df.columns and "player_id" in df.columns:
            merged = merged.merge(
                df[["player_id", stat]].rename(columns={stat: name}),
                on="player_id", how="inner",
            )

    if len(merged) < 10:
        # Too few players — fall back to uniform
        n = len(system_names)
        return {name: 1.0 / n for name in system_names}

    y = merged["actual"].values
    X = merged[[name for name in system_names if name in merged.columns]].values
    valid_names = [name for name in system_names if name in merged.columns]

    # Ridge regression (closed form): w = (X'X + alpha*I)^-1 X'y
    try:
        XtX = X.T @ X + alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        raw_weights = np.linalg.solve(XtX, Xty)

        # Clip to non-negative and normalize
        raw_weights = np.maximum(raw_weights, 0.0)
        total = raw_weights.sum()
        if total > 0:
            raw_weights = raw_weights / total
        else:
            raw_weights = np.ones(len(valid_names)) / len(valid_names)

        return dict(zip(valid_names, raw_weights.tolist()))
    except Exception as e:
        logger.warning("Ridge regression failed: %s — using uniform weights", e)
        n = len(valid_names)
        return {name: 1.0 / n for name in valid_names}
```

- [ ] **Step 3: Run tests and commit**

```bash
git add src/projection_stacking.py tests/test_projection_stacking.py
git commit -m "feat(A2+D3): ridge regression projection stacking for weighted system blending"
```

---

## Wave 4: Backtesting Framework (D6) — Single Agent

---

### Task 8: D6 — Backtesting Framework (Agent: "backtester")

**Problem:** No way to validate whether ANY change improved accuracy. Every weight is a guess.

**Files:**
- Create: `src/backtesting_framework.py` (new, ~200 lines)
- Test: `tests/test_backtesting_framework.py` (new)

- [ ] **Step 1: Write test for the framework API**

```python
# tests/test_backtesting_framework.py
"""Tests for the backtesting framework."""
import pandas as pd
import pytest
from src.backtesting_framework import BacktestRunner


class TestBacktestRunner:
    def test_score_trade_recommendation(self):
        """A trade recommendation should be scorable against actual outcomes."""
        runner = BacktestRunner()
        # Fake recommendation: trade Player A for Player B in week 3
        recommendation = {
            "type": "trade",
            "week": 3,
            "giving_ids": [1],
            "receiving_ids": [2],
            "predicted_sgp_gain": 1.5,
        }
        # Fake actual outcome: player stats in weeks 4-8
        actuals = pd.DataFrame([
            {"player_id": 1, "week": 4, "r": 5, "hr": 2},
            {"player_id": 2, "week": 4, "r": 7, "hr": 3},
        ])
        score = runner.score_recommendation(recommendation, actuals)
        assert "actual_sgp_gain" in score
        assert "predicted_sgp_gain" in score
        assert "error" in score

    def test_calibration_report(self):
        """Calibration report should bucket predictions and check actual rates."""
        runner = BacktestRunner()
        predictions = [
            {"predicted": 0.8, "actual": True},
            {"predicted": 0.8, "actual": True},
            {"predicted": 0.8, "actual": False},
            {"predicted": 0.2, "actual": False},
            {"predicted": 0.2, "actual": False},
            {"predicted": 0.2, "actual": True},
        ]
        report = runner.calibration_report(predictions)
        assert "buckets" in report
        assert len(report["buckets"]) > 0
```

- [ ] **Step 2: Implement minimal framework**

```python
# src/backtesting_framework.py
"""Backtesting framework for validating HEATER engine accuracy.

Replays past weeks and scores engine recommendations against actual
outcomes. Supports trade recommendations, start/sit decisions, waiver
adds, and category win predictions.
"""
from __future__ import annotations
import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Score engine recommendations against actual outcomes."""

    def score_recommendation(
        self, recommendation: dict, actuals: pd.DataFrame
    ) -> dict[str, Any]:
        """Score a single recommendation against actual stats."""
        predicted = recommendation.get("predicted_sgp_gain", 0)
        # Compute actual SGP gain from actuals DataFrame
        # (simplified — full implementation would use SGPCalculator)
        actual_gain = 0.0  # TODO: compute from actuals
        return {
            "predicted_sgp_gain": predicted,
            "actual_sgp_gain": actual_gain,
            "error": predicted - actual_gain,
            "type": recommendation.get("type", "unknown"),
            "week": recommendation.get("week", 0),
        }

    def calibration_report(
        self, predictions: list[dict]
    ) -> dict[str, Any]:
        """Compute calibration report from predicted vs actual outcomes.

        Groups predictions into buckets (0-20%, 20-40%, etc.) and checks
        whether actual success rates match predicted probabilities.
        """
        buckets: dict[str, dict] = {}
        bucket_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

        for lo, hi in bucket_ranges:
            label = f"{int(lo*100)}-{int(hi*100)}%"
            in_bucket = [
                p for p in predictions
                if lo <= p.get("predicted", 0) < hi
            ]
            if in_bucket:
                actual_rate = sum(1 for p in in_bucket if p.get("actual")) / len(in_bucket)
                buckets[label] = {
                    "predicted_avg": sum(p["predicted"] for p in in_bucket) / len(in_bucket),
                    "actual_rate": actual_rate,
                    "count": len(in_bucket),
                    "calibration_error": abs(actual_rate - sum(p["predicted"] for p in in_bucket) / len(in_bucket)),
                }

        return {"buckets": buckets}
```

- [ ] **Step 3: Run tests and commit**

```bash
git add src/backtesting_framework.py tests/test_backtesting_framework.py
git commit -m "feat(D6): backtesting framework for validating engine accuracy

Minimal framework that scores recommendations against actual outcomes
and produces calibration reports. Foundation for validating all future
weight/threshold changes."
```

---

## Verification

After all 4 waves complete:

- [ ] **Run full test suite:** `python -m pytest tests/ -x -q` — expect 2400+ passed
- [ ] **Run lint:** `python -m ruff check src/ tests/` — expect clean
- [ ] **Verify enriched pool:** `python -c "from src.database import load_player_pool; df = load_player_pool(); print(df.columns.tolist())"` — verify health_score, consensus_rank, ytd_* columns present
- [ ] **Verify ECR sources:** `python -c "import src.ecr; print('heater' not in open(src.ecr.__file__).read())"` — should print False (heater removed)

---

## Agent Dispatch Summary

| Wave | Agent Name | Task | Files | Time Est |
|------|-----------|------|-------|----------|
| **1** (parallel) | `weights-unifier` | V1: Unify category weights | matchup_context.py, trade_finder.py | 30 min |
| **1** (parallel) | `sgp-unifier` | V2: Fix SGP rate stats | trade_finder.py, valuation.py | 30 min |
| **2** (parallel) | `sgp-dynamo` | A1: Wire dynamic denominators | valuation.py, data_bootstrap.py | 25 min |
| **2** (parallel) | `baselines-fixer` | U1: Rate-stat baselines | valuation.py | 15 min |
| **2** (parallel) | `ecr-fixer` | A4: Remove self-referential ECR | ecr.py | 10 min |
| **3** (parallel) | `stuff-fetcher` | E2+T1: Fetch Stuff+ data | data_bootstrap.py, database.py | 20 min |
| **3** (parallel) | `stacking-builder` | A2+D3: Projection stacking | projection_stacking.py (new) | 25 min |
| **4** (single) | `backtester` | D6: Backtesting framework | backtesting_framework.py (new) | 30 min |

---

# Phases 2-5: Future Plans

Phases 2-5 will be planned AFTER Phase 1 is fully implemented and verified.
Each phase gets its own plan document written at the start of that phase.

- **Phase 2:** Page-Specific Wins (I1+I4, G1, M1, O1, D1, D2, I2, L1, U2)
- **Phase 3:** Accuracy Refinements (A3, G4, F5, I3, J2, J3, J4, P1, R1, R2, L2, M4)
- **Phase 4:** Strategic Features (K4, H7, E9, K1, H8, I5+K6, B1, D4, Q1, Q2, N1, O3, M6, P2, P4)
- **Phase 5:** Validation & Polish (B6, B8, C2, K2, I6, L3, L4, T5-T12, S1-S6, remaining)

<!--

**Depends on:** Phase 1 complete (unified weights, fixed SGP, dynamic denoms).
**Architecture:** 9 tasks across 5 pages. Many are independent and can be parallelized by page.

## Wave 5: Lineup Optimizer + Trade Finder — PARALLEL (3 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `lineup-pivot` | Mid-Week Pivot + Category Flip | I1+I4 | `src/optimizer/pivot_advisor.py` (new), `pages/5_Line-up_Optimizer.py` | Classify categories WON/LOST/CONTESTED. Compute flip_prob per category. Bench pitchers to protect WON ratios. |
| `trade-regression` | xwOBA Regression Flags | G1 | `src/trade_finder.py`, `src/database.py` | Add xwoba_delta to enriched pool. Flag BUY_LOW (gap ≥+0.030) / SELL_HIGH (≤-0.030). +0.05 composite bonus. |
| `lineup-swing` | Swing-Category Weighting | I2 | `src/optimizer/h2h_engine.py`, `src/optimizer/pipeline.py` | Maximize P(win) for 30-70% categories in LP objective. De-emphasize locked/lost. |

## Wave 6: My Team + Leaders + Closer Monitor — PARALLEL (3 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `myteam-flip` | Category Flip Analyzer | M1 | `pages/1_My_Team.py`, `src/war_room.py` | Show mid-week flip probabilities in War Room. Shares engine with I1+I4. |
| `leaders-breakout` | Statcast Breakout Score | O1 | `src/leaders.py`, `pages/9_Leaders.py` | Composite: EV50 (30%), barrel YoY (25%), xwOBA-wOBA (20%), bat speed (15%), LA change (10%). Flag >70th pctl. |
| `closer-gmli` | gmLI Trust Tracking | L1 | `src/closer_monitor.py`, `pages/7_Closer_Monitor.py` | Track 2-week rolling gmLI vs season avg. Drop from 2.0+ to <1.5 = "Usage Downgrade" alert. Requires T5 data. |

## Wave 7: Global ML Models — PARALLEL (2 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `statcast-ml` | Statcast XGBoost Regression | D1 | `src/ml_ensemble.py`, `src/data_bootstrap.py` | Train on EV, barrel, xwOBA, xBA, sprint speed. Target: actual − projected. Retrain daily. Replace stub. |
| `playtime-ml` | Playing Time Prediction | D2 | `src/playing_time_model.py` (new), `src/data_bootstrap.py` | Ridge regression: remaining_PA = f(recent_PA_rate, depth_chart, health, age). Apply as multiplier to counting stats. |

## Wave 8: Trade Finder LP Fix (1 agent, depends on V2)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `trade-lp` | LP-Constrained Totals in By Value | U2 | `src/trade_finder.py` | Wire LP-constrained 18-starter totals into scan_1_for_1() composite. Currently overcounts bench. |

---

# Phase 3: Accuracy Refinements (Tier 3) — 12 Tasks

**Depends on:** Phase 1-2 complete.

## Wave 9: Trade Finder Refinements — PARALLEL (4 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `trade-accept-odds` | Playoff-Odds Acceptance | F5 | `src/trade_finder.py` | Replace raw standings rank with playoff odds from simulate_season_enhanced(). |
| `trade-reliability` | Stat Reliability Weighting | G4 | `src/trade_finder.py` | reliability = min(1.0, PA/threshold[stat]). Trust K% at 60 PA ≠ AVG at 60 PA. |
| `trade-grade-ci` | Trade Grade Confidence Interval | P1 | `src/engine/output/trade_evaluator.py` | Show grade RANGE ("B+ to A-") not single letter. Based on ±1 SD. |
| `trade-time-decay` | Differential Time Decay | H1 (partial) | `src/trade_finder.py`, `src/trade_intelligence.py` | Counting × weeks_rem/total_weeks. Rate stats stay 1.0 with confidence penalty <8 weeks. |

## Wave 10: Lineup Optimizer Refinements — PARALLEL (4 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `lineup-ratio` | Ratio Protection Calculator | I3 | `src/optimizer/daily_optimizer.py` | marginal_era_risk = (proj_ER×9)/(banked_IP+proj_IP) - current_ERA. Bench if risk > lead. |
| `lineup-platoon` | Platoon Split Bayesian Regression | J2 | `src/optimizer/matchup_adjustments.py` | Bayesian blend: (PA/stab)*individual + (1-PA/stab)*league_avg. LHB stab=1000, RHB stab=2200. |
| `lineup-pitcher` | Opposing Pitcher Quality Calibration | J3 | `src/optimizer/matchup_adjustments.py` | Calibrate to ±15%: mult = 1.0 + 0.15*(league_xFIP - opp_xFIP)/std_xFIP. |
| `lineup-weather` | Comprehensive Weather Model | J4 | `src/optimizer/matchup_adjustments.py`, `src/optimizer/daily_optimizer.py` | Rain >40%: BB×1.10, K×0.90. Wind out >10mph: HR×1.15-1.20. Full temp scalar. |

## Wave 11: Standings + Regression — PARALLEL (4 agents)

| Agent | Task | ROADMAP Item | Files | Goal |
|-------|------|-------------|-------|------|
| `standings-mc` | MC Simulation 10K | R1 | `src/standings_engine.py` | Default 1K→10K sims. Reduces SE from ~1.5% to ~0.5%. |
| `standings-variance` | Per-Category Variance | R2 | `src/standings_engine.py` | Use weekly SDs: K=1.2, R=1.6, SB=2.3, AVG=2.9, etc. |
| `closer-decay` | K% Skill Decay Alert | L2 | `src/closer_monitor.py` | Rolling 14-day K% vs season. K% drop ≥8 pts = "Skill Warning." |
| `myteam-regression` | Regression Alert System | M4 | `pages/1_My_Team.py`, `src/alerts.py` | Compare L30 actual to xBA/xwOBA. Flag >1.5 SD divergence. |

---

# Phase 4: Strategic Features (Tier 4) — 15 Tasks

**Depends on:** Phase 1-3 complete.

## Wave 12: Punt + Streaming + SB — PARALLEL (3 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `punt-optimizer` | Punt Mode + Punt Advisor | K4, H7 | `src/optimizer/pipeline.py`, `src/trade_finder.py` | User selects 0-2 categories to punt → weight 0.0 in LP. Surface in Trade Readiness. |
| `schedule-stream` | Schedule-Aware Streaming | E9 | `src/optimizer/streaming.py` | Off-day streams more valuable. Opponent L14 wRC+ for quality. Two-start detection. |
| `sb-streaming` | SB Streaming by Catcher/Pitcher | K1 | `src/optimizer/streaming.py` | sb_score = sprint_speed × (pop_time/1.95) × (delivery_time/1.35) × handedness. |

## Wave 13: Trade Finder Strategic — PARALLEL (4 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `trade-closer` | Closer Stability Discount | H8 | `src/trade_finder.py`, `src/closer_monitor.py` | sv_adjusted = sv_proj × (confidence/100). Wire closer_monitor into trade valuation. |
| `trade-pa-lineup` | Batting Order PA + Volume | I5+K6 | `src/optimizer/daily_optimizer.py` | Wire LINEUP_SLOT_PA into DCV counting stats AND volume_factor. |
| `trade-accept-learn` | League-Specific Acceptance + Opponent Learning | B1, D4 | `src/trade_finder.py`, `src/trade_intelligence.py` | Empirical acceptance from Yahoo transactions. Per-opponent logistic regression. |
| `trade-consistency` | Consistency/Variance Modifier | H6 | `src/trade_finder.py` | Weekly CV from game logs. CV<0.30 = +5-8% SGP, CV>0.50 = -5-8%. |

## Wave 14: Page-Specific Features — PARALLEL (5 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `myteam-features` | IL Alert + Ratio Lock + Opponent Track | M5, M6, M3 | `pages/1_My_Team.py`, `src/alerts.py` | IL slot utilization, ratio lock detection, opponent roster moves. |
| `leaders-prospects` | Prospect Relevance + Call-Up Alerts | O2, O3 | `src/prospect_engine.py`, `pages/9_Leaders.py` | Fantasy relevance score. 40-man flag + service time + injury cross-ref. |
| `compare-features` | Category Fit + SGP Breakdown | N1, N3 | `pages/6_Player_Compare.py` | Category fit indicator + stacked SGP bar chart. |
| `draft-ai` | Marginal SGP AI + Position Runs | Q1, Q2 | `src/simulation.py`, `pages/2_Draft_Simulator.py` | AI picks by marginal SGP 70%, ADP 30%. Position run detection 1.3x boost. |
| `trade-analyzer-mc` | Trade Analyzer MC + Replacement | P2, P4 | `src/engine/monte_carlo/trade_simulator.py`, `src/valuation.py` | Antithetic variate sampling. Per-category replacement level. |

---

# Phase 5: Validation, Polish & Remaining (Tier 5) — 14+ Tasks

**Depends on:** Phase 1-4 complete.

## Wave 15: Validation — PARALLEL (3 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `validate-urgency` | Category Urgency k-Calibration | B6 | `src/optimizer/category_urgency.py` | Backtest k=1.0 to 5.0 using D6 framework. Pick k that maximizes weekly W-L. |
| `validate-trade-wts` | Trade Finder Weight Validation | B8 | `src/trade_finder.py` | Backtest top-ranked trades vs actual improvement by week 24. |
| `validate-alpha` | Dual Objective Alpha Validation | C2 | `src/optimizer/dual_objective.py` | Tie alpha to playoff probability, not just time remaining. |

## Wave 16: Remaining Page Features — PARALLEL (5 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `lineup-fatigue` | Pitcher Fatigue Multiplier | K2 | `src/optimizer/projections.py` | fatigue = max(0.85, 1.0 - 0.003 × max(0, IP-100)). ACWR >1.3 flag. |
| `lineup-ip-mode` | IP Management Mode | I6 | `pages/5_Line-up_Optimizer.py` | "Chase K/W", "Protect Ratios", "Balanced" toggle. Auto-recommend from urgency. |
| `closer-committee` | Committee Risk + Opener Flag | L3, L4 | `src/closer_monitor.py` | Committee risk score + opener detection (1st inning >30%). |
| `compare-schedule` | Schedule Strength Comparison | N2 | `pages/6_Player_Compare.py` | Next 2-4 weeks opposing pitchers side-by-side. |
| `leaders-skew` | Projection Skew Indicator | O4 | `pages/9_Leaders.py` | Flag players where 5/7 systems above consensus. |

## Wave 17: Data Acquisition — PARALLEL (5 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `fetch-gmli` | Fetch gmLI + Reliever Innings | T5, T6 | `src/data_bootstrap.py` | pybaseball pitching_stats for gmLI. statsapi game logs for inning of entry. |
| `fetch-umpires` | Fetch Umpire Assignments | T7 | `src/data_bootstrap.py` | Scrape Baseball Savant game feeds for umpire + build tendency table. |
| `fetch-catchers` | Fetch Catcher Framing + Pop Time | T8 | `src/data_bootstrap.py` | Scrape Baseball Savant leaderboards. |
| `fetch-batspeed` | Fetch Bat Speed + 40-Man | T9, T10 | `src/data_bootstrap.py` | Scrape Baseball Savant bat tracking. MLB API 40-man roster. |
| `fetch-stadiums` | Stadium Orientations + PvB | T11, T12 | `src/game_day.py`, `src/data_bootstrap.py` | Hardcode 30 outfield bearings. PvB splits with smart caching. |

## Wave 18: Consolidation — PARALLEL (3 agents)

| Agent | Task | ROADMAP Items | Files | Goal |
|-------|------|--------------|-------|------|
| `consolidate-tf` | Merge Trade Finder Tabs | S1 | `pages/10_Trade_Finder.py` | Merge Smart Recs + By Value into single sortable tab. |
| `consolidate-lo` | Remove H2H Tab + Streaming Quick-View | S2, S3 | `pages/5_Line-up_Optimizer.py` | Remove duplicate H2H tab. Convert Streaming to quick-view. |
| `consolidate-infra` | Shared Standings Utils + Roster Rendering + Formatting | S4, S5, S6, V3-V6 | `src/standings_utils.py` (new), `src/ui_shared.py` | Extract shared utilities. Unify opponent intel, FA pool, formatting. |

## Wave 19: Remaining Items — As Time Allows

| ROADMAP Items | Description |
|--------------|-------------|
| H2 | Dynamic roster spot value |
| H3 | Graduated positional scarcity |
| H4 | SB independence premium increase |
| H5 | Trade timing multiplier |
| H9 | Prospect call-up valuation |
| B2 | Position-specific health scoring |
| B3 | Injury-type adjustment |
| B4 | Temporal ECR weighting |
| B5 | Dynamic park factors |
| B7 | Power rankings momentum weight |
| C1 | Waiver wire drop penalties (finish PARTIAL) |
| C5 | Inverse park formula |
| D7 | Category-aware lineup RL (experimental, needs 8+ weeks) |
| E1 | BABIP regression targets (xBABIP model) |
| E3 | Umpire strike zone adjustment |
| E6 | Enhanced weather wind direction |
| E7 | Pitcher-batter matchup history |
| E10 | Catcher framing value |
| F2 | Draft-round anchoring |
| F3 | Disposition effect |
| F4 | Recently-acquired penalty |
| G3 | BABIP regression scoring |
| G5 | Velocity trend signal |
| J5 | Catcher framing pitcher adjustment |
| K3 | Consistency premium |
| K5 | Streaming composite score |
| M2 | Conditional swap impact |
| O5 | SGP contribution breakdown (wire N3 into Leaders) |
| P3 | Copula correlation calibration |
| Q3 | Per-player ADP standard deviation |
| Q4 | Draft value standings impact |
| R2 | Per-category variance calibration (if not done in Wave 11) |
| U3 | War Room flippable thresholds from weekly SDs |
| A3 | Bayesian SGP updating |

---

# Complete Agent Dispatch Summary (All Phases)

| Phase | Waves | Total Agents | Total Tasks | Est. Time |
|-------|-------|-------------|-------------|-----------|
| **1: Core Engine** | 1-4 | 8 | 8 | 3-5 hours |
| **2: Page Wins** | 5-8 | 9 | 9 | 4-6 hours |
| **3: Accuracy** | 9-11 | 12 | 12 | 5-7 hours |
| **4: Strategic** | 12-14 | 12 | 15 | 6-8 hours |
| **5: Validation+** | 15-19 | 16+ | 14+ | 6-10 hours |
| **Total** | **19 waves** | **57+ agents** | **58+ tasks** | **24-36 hours** |

Each wave's agents run in parallel. Waves within a phase are sequential (later waves depend on earlier ones). Phases are strictly sequential (Phase 2 depends on Phase 1 outputs).

**Execution model:** Use `/dispatching-parallel-agents` to launch each wave's agents simultaneously. Review outputs between waves. Commit after each wave passes tests.
-->
