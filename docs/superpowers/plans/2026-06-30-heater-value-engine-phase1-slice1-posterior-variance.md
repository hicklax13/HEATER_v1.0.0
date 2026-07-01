# Phase 1 Slice 1 — Layer-0 Posterior Variance Core (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the keystone of the Layer-0 shared player model — a per-player, per-category posterior that surfaces the projection MEAN plus two SEPARATE variances: `σ²` (between-player / true-talent epistemic uncertainty, shrinks with the player's own sample size but never vanishes) and `τ²` (week-to-week aleatory outcome variance, with a distributional margin descriptor: NB for counting cats, beta-binomial for AVG/OBP, ratio-normal for ERA/WHIP).

**Architecture:** A new focused module `src/player_model/posterior.py` that GENERALIZES the existing per-*team* `build_team_kalman_variances` (`src/engine/output/weekly_matrix.py`) to per-*player* level, and ADDS (a) an irreducible projection-error floor so bands are never dishonestly tight (spec gap G3), and (b) explicit per-category week-to-week margins (NB / beta-binomial / ratio-normal) so Layer 1 (cheap proxy) and Layer 2 (deep MC) can sample the SAME margins (spec Tier-1 correction #2). It composes — not reinvents — the pool's blended mean (`load_player_pool`), `LeagueConfig`'s category taxonomy, and the Kalman stabilization-shrinkage pattern. Pure functions over a pool row; no DB or network. Every per-category seed constant is documented as harness-calibratable (slice 5 / a follow-on tunes them via SURE on the backtest).

**Tech Stack:** Python 3.12/3.14, NumPy, dataclasses, pandas (a pool row is a `pd.Series`), pytest. Reuses `src/valuation.py` (`LeagueConfig`).

**Spec:** `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` §3 (Layer 0 — "heteroscedastic true-talent posterior … σ² (player-to-player) and τ² (week-to-week) separately; NB margins for counting cats, beta-binomial for rate cats — variance unequal across players") + §6 (module boundary `player_model`) + research-validation Tier-1 #2/#5, Tier-2 #6, gap G3/G5.

**Scope note:** This slice delivers the variance PARAMETERIZATION only (mean + σ² + τ² + margin descriptor). Drawing samples from the margins is Layer 1/2 (Phase 2/3). The availability survival layer is slice 2, the display G-score is slice 3, the unified `(player_id, league_ctx) → PlayerPosterior` facade is slice 4, and the backtest-gate validation is slice 5. Constants are SEEDED (sensible defaults), not yet calibrated — calibration is slice 5.

---

## File Structure

- **Create `src/player_model/__init__.py`** — make `player_model` a package. Empty placeholder this slice; slice 4 fills it with the facade.
- **Create `src/player_model/posterior.py`** — the `CategoryPosterior` dataclass + the seeded constants + `category_posterior()` / `player_posteriors()`. One responsibility: turn a pool row into per-category posterior variance parameters.
- **Create `tests/test_player_model_posterior.py`** — TDD tests for every behavior.

---

### Task 1: Package + `CategoryPosterior` dataclass + category classification & mean extraction

The posterior carries the mean and two variances plus a `margin` descriptor (how Layer 1/2 reconstruct the sampling distribution). Category `kind` is one of `"counting"` (R/HR/RBI/SB/W/L/SV/K), `"rate_prop"` (AVG/OBP — bounded [0,1] proportions → beta-binomial), `"rate_ratio"` (ERA/WHIP — events-per-IP, NOT proportions → ratio-normal). The per-week MEAN is the season projection ÷ `weeks` for counting cats, and the rate value directly for rate cats.

**Files:**
- Create: `src/player_model/__init__.py`
- Create: `src/player_model/posterior.py`
- Test: `tests/test_player_model_posterior.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_model_posterior.py
"""Tests for the Layer-0 posterior variance core (Phase 1 slice 1)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.valuation import LeagueConfig


def _hitter_row(**over):
    base = dict(
        player_id=1, name="Test Hitter", is_hitter=1,
        r=90, hr=30, rbi=95, sb=12, avg=0.270, obp=0.340,
        ab=560, pa=620, ip=0.0, w=0, l=0, sv=0, k=0, era=0.0, whip=0.0,
        ytd_pa=0, ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def _pitcher_row(**over):
    base = dict(
        player_id=2, name="Test Pitcher", is_hitter=0,
        r=0, hr=0, rbi=0, sb=0, avg=0.0, obp=0.0,
        ab=0, pa=0, ip=180.0, w=14, l=8, sv=0, k=200, era=3.50, whip=1.15,
        ytd_pa=0, ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def test_category_kind_classification():
    from src.player_model.posterior import classify_kind
    assert classify_kind("HR") == "counting"
    assert classify_kind("SB") == "counting"
    assert classify_kind("K") == "counting"
    assert classify_kind("L") == "counting"      # inverse but still a counting total
    assert classify_kind("AVG") == "rate_prop"
    assert classify_kind("OBP") == "rate_prop"
    assert classify_kind("ERA") == "rate_ratio"  # events/IP, not a [0,1] proportion
    assert classify_kind("WHIP") == "rate_ratio"


def test_counting_mean_is_per_week():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(hr=30), "HR", cfg, weeks=26)
    assert p.mean == pytest.approx(30 / 26)   # season total spread over weeks


def test_rate_mean_is_the_rate_directly():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(avg=0.270), "AVG", cfg, weeks=26)
    assert p.mean == pytest.approx(0.270)     # rate cats are not divided by weeks


def test_posterior_fields_present():
    from src.player_model.posterior import CategoryPosterior, category_posterior
    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(), "HR", cfg)
    assert isinstance(p, CategoryPosterior)
    assert p.category == "HR"
    assert p.kind == "counting"
    assert p.mean > 0 and p.sigma2 > 0 and p.tau2 > 0
    assert isinstance(p.margin, dict) and p.margin["dist"] == "nb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_posterior.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.player_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/player_model/__init__.py
"""HEATER Layer-0 shared player model (Advanced Value Engine, Phase 1).

Produces, per player per scoring category, a posterior {mean, sigma2, tau2} with a
distributional margin descriptor, plus (later slices) an availability survival
distribution and a display-only G-score. Single source of truth feeding every surface.
"""
```

```python
# src/player_model/posterior.py
"""Layer-0 posterior variance core — per-player, per-category {mean, sigma2, tau2}.

Generalizes the per-team Kalman stabilization-shrinkage in
src/engine/output/weekly_matrix.build_team_kalman_variances to PER-PLAYER, and adds:
  * an irreducible projection-error floor so true-talent bands never falsely vanish
    (research-validation gap G3 — "champ/playoff bands are dishonestly narrow" without it);
  * explicit week-to-week MARGIN descriptors (NB for counting cats, beta-binomial for
    AVG/OBP, ratio-normal for ERA/WHIP) so the Layer-1 proxy and the Layer-2 deep MC
    sample the SAME margins (research-validation Tier-1 correction #2).

sigma2 = between-player TRUE-TALENT (epistemic) variance: how unsure we are of the
         player's true rate. Shrinks with the player's own YTD sample, never to zero.
tau2   = week-to-week ALEATORY variance: how much a single week's outcome scatters
         around the true rate. Counting -> NB; AVG/OBP -> beta-binomial; ERA/WHIP -> ratio-normal.

All per-category SEED constants below are documented harness-calibratable (slice 5 tunes
them via SURE on the game-log backtest). Pure functions over a pool row (pd.Series);
no DB / network. Never raises on missing columns or NaN — degrades to max-uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.valuation import LeagueConfig

# ── Seeded constants (HARNESS-CALIBRATABLE — slice 5) ───────────────────────────
# Per-team analog: weekly_matrix._KALMAN_TALENT_CV / _KALMAN_STABILIZE_PA/IP. Per-PLAYER
# stabilization scales are smaller than per-team (one player's season ~600 PA vs a roster's
# ~6000). Seeds below are deliberate, sensible, and replaced by slice-5 SURE calibration.
_TALENT_CV: float = 0.40       # reducible true-talent spread coefficient (counting cats)
_PROJ_FLOOR_CV: float = 0.10   # irreducible projection-error CV — NEVER vanishes (gap G3)
_STABILIZE_PA: float = 1200.0  # ytd_pa at which true-talent shrink == 0.5 (hitter cats)
_STABILIZE_IP: float = 400.0   # ytd_ip at which true-talent shrink == 0.5 (pitcher cats)

# Week-to-week overdispersion for counting cats: tau2 = phi * mu_week (phi=1 => Poisson).
# SB/SV are role-/opportunity-driven and most overdispersed; K is steadiest. (FanGraphs
# NB run-distribution work: counting outcomes are ~1.3-2x overdispersed vs Poisson.)
_COUNTING_OVERDISPERSION: dict[str, float] = {
    "R": 1.5, "HR": 1.6, "RBI": 1.5, "SB": 1.8,
    "W": 1.4, "L": 1.4, "SV": 2.0, "K": 1.3,
}

# Rate cats: per-week ABSOLUTE std at a reference weekly volume (seeded from
# scenario_generator._RATE_STD). tau2 scales inversely with the player's weekly volume.
_RATE_TALENT_STD: dict[str, float] = {"AVG": 0.020, "OBP": 0.022, "ERA": 0.60, "WHIP": 0.09}
_RATE_FLOOR_STD: dict[str, float] = {"AVG": 0.012, "OBP": 0.013, "ERA": 0.35, "WHIP": 0.05}
_RATE_WEEK_STD: dict[str, float] = {"AVG": 0.022, "OBP": 0.025, "ERA": 0.70, "WHIP": 0.10}
_RATE_REF_AB: float = 20.0     # reference weekly AB the AVG/OBP week-std is quoted at
_RATE_REF_IP: float = 20.0     # reference weekly IP the ERA/WHIP week-std is quoted at
_MIN_WEEK_VOL: float = 1.0     # floor on weekly volume to avoid division blow-ups

_HITTER_VOL_COL = "ab"         # rate_prop weekly-volume source column (season AB)
_PITCHER_VOL_COL = "ip"        # rate_ratio weekly-volume source column (season IP)


@dataclass(frozen=True)
class CategoryPosterior:
    """Per-player, per-category posterior. `mean` is per-WEEK for counting cats and the
    rate value for rate cats. `sigma2` = between-player true-talent variance; `tau2` =
    week-to-week outcome variance. `margin` reconstructs the sampling distribution:
      counting  -> {"dist": "nb", "mean": mu_week, "r": dispersion}      (r=inf => Poisson)
      rate_prop -> {"dist": "beta_binomial", "theta": rate, "n": v_week, "rho": icc}
      rate_ratio-> {"dist": "ratio_normal", "mean": rate, "std_week": sqrt(tau2)}
    """

    category: str
    kind: str
    mean: float
    sigma2: float
    tau2: float
    margin: dict = field(default_factory=dict)


def classify_kind(category: str, config: LeagueConfig | None = None) -> str:
    """Map an UPPERCASE category to its variance kind: 'counting' | 'rate_prop' | 'rate_ratio'.
    AVG/OBP are bounded proportions (beta-binomial); ERA/WHIP are events-per-IP ratios."""
    cfg = config or LeagueConfig()
    if category not in cfg.rate_stats:
        return "counting"
    return "rate_prop" if category in {"AVG", "OBP"} else "rate_ratio"


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _mean_for(row, category: str, kind: str, config: LeagueConfig, weeks: float) -> float:
    col = config.STAT_MAP[category]
    raw = _f(row.get(col)) if hasattr(row, "get") else _f(row[col])
    if kind == "counting":
        return raw / weeks if weeks > 0 else 0.0
    return raw  # rate cats: the rate itself
```

- [ ] **Step 4: Run test to verify it passes**

Note: `category_posterior` is referenced by the tests but defined in Task 4. Run only the Task-1 tests now:

Run: `python -m pytest tests/test_player_model_posterior.py -k "kind or mean or fields" -q`
Expected: `test_category_kind_classification` PASSES; the three `category_posterior`-dependent tests ERROR with `ImportError: cannot import name 'category_posterior'` (defined in Task 4 — expected).

- [ ] **Step 5: Commit**

```bash
git add src/player_model/__init__.py src/player_model/posterior.py tests/test_player_model_posterior.py
git commit -m "feat(player_model): CategoryPosterior dataclass + kind classification (P1 slice1)"
```

---

### Task 2: `between_player_sigma2` — heteroscedastic true-talent variance with a never-vanishing floor

`σ²` = a reducible part that shrinks with the player's own YTD sample (`shrink = S0/(S0+n)`, generalizing the per-team Kalman model to per-player) PLUS an irreducible projection-error floor that never vanishes (gap G3). A rookie (tiny sample) carries far more true-talent uncertainty than a veteran with a full season — *unequal across players*, which is the whole point.

**Files:**
- Modify: `src/player_model/posterior.py`
- Test: `tests/test_player_model_posterior.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_posterior.py
def test_sigma2_heteroscedastic_low_sample_is_more_uncertain():
    from src.player_model.posterior import between_player_sigma2
    # Same mean, different YTD sample size -> low-sample player has LARGER true-talent variance.
    lo = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=50, is_hitter=True)
    hi = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=1200, is_hitter=True)
    assert lo > hi


def test_sigma2_floor_never_vanishes():
    from src.player_model.posterior import between_player_sigma2
    # Even an "infinitely sampled" player keeps the irreducible projection-error floor (gap G3).
    huge = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=10_000_000, is_hitter=True)
    assert huge > 0.0
    # The floor equals (|mean| * _PROJ_FLOOR_CV)^2 in the large-sample limit.
    from src.player_model.posterior import _PROJ_FLOOR_CV
    assert huge == pytest.approx((1.15 * _PROJ_FLOOR_CV) ** 2, rel=1e-6)


def test_sigma2_rate_uses_absolute_std_not_cv():
    from src.player_model.posterior import between_player_sigma2, _RATE_FLOOR_STD
    # Rate cats use an ABSOLUTE std floor (a 0.40 CV on a 0.270 AVG would be absurd).
    s = between_player_sigma2(mean=0.270, kind="rate_prop", category="AVG", n=10_000_000, is_hitter=True)
    assert s == pytest.approx(_RATE_FLOOR_STD["AVG"] ** 2, rel=1e-6)


def test_sigma2_zero_sample_is_max_uncertainty():
    from src.player_model.posterior import between_player_sigma2
    # n=0 -> shrink=1 -> full talent spread + floor.
    s0 = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=0, is_hitter=True)
    s_big = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=5000, is_hitter=True)
    assert s0 > s_big
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_posterior.py -k sigma2 -q`
Expected: FAIL — `ImportError: cannot import name 'between_player_sigma2'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/posterior.py
def _shrink(n: float, is_hitter: bool) -> float:
    """True-talent shrink factor S0/(S0+n): 1.0 at n=0 (no info), -> 0 with large samples.
    Generalizes weekly_matrix.build_team_kalman_variances' per-team shrink to per-player."""
    s0 = _STABILIZE_PA if is_hitter else _STABILIZE_IP
    n = max(0.0, _f(n))
    return s0 / (s0 + n) if (s0 + n) > 0 else 1.0


def between_player_sigma2(mean: float, kind: str, category: str, n: float, is_hitter: bool) -> float:
    """Between-player (epistemic) true-talent variance. Reducible part shrinks with the
    player's own sample `n` (ytd_pa for hitters / ytd_ip for pitchers); the floor never
    vanishes (gap G3). Counting cats use a CV on the per-week mean; rate cats use absolute
    std seeds. Returns a variance (>= floor^2 > 0)."""
    mean = _f(mean)
    shrink = _shrink(n, is_hitter)
    if kind == "counting":
        reducible = abs(mean) * _TALENT_CV * shrink
        floor = abs(mean) * _PROJ_FLOOR_CV
    else:  # rate_prop / rate_ratio
        reducible = _RATE_TALENT_STD.get(category, 0.02) * shrink
        floor = _RATE_FLOOR_STD.get(category, 0.01)
    return float(reducible * reducible + floor * floor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_posterior.py -k sigma2 -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/posterior.py tests/test_player_model_posterior.py
git commit -m "feat(player_model): heteroscedastic between-player sigma2 with G3 floor (P1 slice1)"
```

---

### Task 3: `week_to_week_tau2` + margin — NB (counting) / beta-binomial (AVG/OBP) / ratio-normal (ERA/WHIP)

`τ²` is the aleatory week-to-week scatter, with a margin descriptor Layer 1/2 sample from. Counting → NB with `τ² = φ·μ_week` and dispersion `r = μ_week/(φ−1)` (`φ=1` ⇒ Poisson, `r=inf`). AVG/OBP → beta-binomial over the player's weekly AB; ERA/WHIP → ratio-normal whose week-std scales `∝ √(ref_IP / IP_week)` (more innings ⇒ tighter weekly rate).

**Files:**
- Modify: `src/player_model/posterior.py`
- Test: `tests/test_player_model_posterior.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_posterior.py
def test_tau2_counting_is_phi_times_mean_and_nb_margin_consistent():
    from src.player_model.posterior import week_to_week_tau2, _COUNTING_OVERDISPERSION
    mu = 1.15
    tau2, margin = week_to_week_tau2(mean=mu, kind="counting", category="HR", weekly_vol=0.0)
    phi = _COUNTING_OVERDISPERSION["HR"]
    assert tau2 == pytest.approx(phi * mu)
    assert margin["dist"] == "nb"
    # NB variance mu + mu^2/r must equal tau2 (the margin reconstructs the same variance).
    r = margin["r"]
    assert mu + mu * mu / r == pytest.approx(tau2, rel=1e-6)


def test_tau2_counting_poisson_limit_when_phi_one(monkeypatch):
    import src.player_model.posterior as pm
    monkeypatch.setitem(pm._COUNTING_OVERDISPERSION, "HR", 1.0)
    tau2, margin = pm.week_to_week_tau2(mean=2.0, kind="counting", category="HR", weekly_vol=0.0)
    assert tau2 == pytest.approx(2.0)                 # Poisson: variance == mean
    assert math.isinf(margin["r"])                    # phi=1 -> r = inf (Poisson)


def test_tau2_rate_prop_decreases_with_weekly_volume():
    from src.player_model.posterior import week_to_week_tau2
    lo_vol, _ = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=10.0)
    hi_vol, _ = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=40.0)
    assert lo_vol > hi_vol                            # more AB/week -> tighter weekly AVG
    _, margin = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=20.0)
    assert margin["dist"] == "beta_binomial"
    assert margin["theta"] == pytest.approx(0.270)


def test_tau2_rate_ratio_margin_is_ratio_normal():
    from src.player_model.posterior import week_to_week_tau2
    tau2, margin = week_to_week_tau2(mean=3.50, kind="rate_ratio", category="ERA", weekly_vol=20.0)
    assert tau2 > 0
    assert margin["dist"] == "ratio_normal"
    assert margin["std_week"] == pytest.approx(math.sqrt(tau2), rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_posterior.py -k tau2 -q`
Expected: FAIL — `ImportError: cannot import name 'week_to_week_tau2'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/posterior.py
def week_to_week_tau2(mean: float, kind: str, category: str, weekly_vol: float) -> tuple[float, dict]:
    """Week-to-week (aleatory) variance + sampling margin. Counting -> NB (tau2 = phi*mu,
    r = mu/(phi-1); phi<=1 -> Poisson, r=inf). rate_prop (AVG/OBP) -> beta-binomial over
    `weekly_vol` AB. rate_ratio (ERA/WHIP) -> ratio-normal with week-std scaled by volume.
    Returns (tau2, margin_dict). Never raises."""
    mean = _f(mean)
    if kind == "counting":
        phi = _COUNTING_OVERDISPERSION.get(category, 1.4)
        mu = max(0.0, mean)
        tau2 = phi * mu
        r = mu / (phi - 1.0) if (phi > 1.0 and mu > 0.0) else math.inf
        return float(tau2), {"dist": "nb", "mean": mu, "r": r}

    if kind == "rate_prop":
        ref = _RATE_REF_AB
        vol = max(_MIN_WEEK_VOL, _f(weekly_vol))
        base = _RATE_WEEK_STD.get(category, 0.022)
        tau2 = (base * base) * (ref / vol)
        # Beta-binomial intra-class correlation implied by the inflated week variance over a
        # binomial baseline theta(1-theta)/vol; clamped to [0, 0.5] for stability.
        theta = min(max(mean, 1e-3), 1 - 1e-3)
        binom_var = theta * (1 - theta) / vol
        rho = 0.0 if binom_var <= 0 else min(0.5, max(0.0, (tau2 - binom_var) / max(binom_var, 1e-9)))
        return float(tau2), {"dist": "beta_binomial", "theta": theta, "n": vol, "rho": rho}

    # rate_ratio (ERA/WHIP): events-per-IP, not a proportion.
    ref = _RATE_REF_IP
    vol = max(_MIN_WEEK_VOL, _f(weekly_vol))
    base = _RATE_WEEK_STD.get(category, 0.70)
    tau2 = (base * base) * (ref / vol)
    return float(tau2), {"dist": "ratio_normal", "mean": mean, "std_week": math.sqrt(max(tau2, 0.0))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_posterior.py -k tau2 -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/posterior.py tests/test_player_model_posterior.py
git commit -m "feat(player_model): week-to-week tau2 + NB/beta-binomial/ratio-normal margins (P1 slice1)"
```

---

### Task 4: `category_posterior` + `player_posteriors` orchestrators

Tie mean + σ² + τ² + margin into a `CategoryPosterior` for one (player, cat), and `player_posteriors` for all cats relevant to the player (hitting cats for hitters, pitching cats for pitchers). Weekly volume for rate cats comes from the pool's season AB (hitters) / IP (pitchers) ÷ weeks; the YTD sample for σ² comes from `ytd_pa` / `ytd_ip`.

**Files:**
- Modify: `src/player_model/posterior.py`
- Test: `tests/test_player_model_posterior.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_posterior.py
def test_category_posterior_full_object_hitter_counting():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(hr=30, ytd_pa=300), "HR", cfg, weeks=26)
    assert p.category == "HR" and p.kind == "counting"
    assert p.mean == pytest.approx(30 / 26)
    assert p.sigma2 > 0 and p.tau2 > 0
    assert p.margin["dist"] == "nb"


def test_category_posterior_rate_uses_weekly_volume_from_pool():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    # 520 AB over 26 weeks -> 20 AB/week feeds the beta-binomial n.
    p = category_posterior(_hitter_row(avg=0.300, ab=520), "AVG", cfg, weeks=26)
    assert p.kind == "rate_prop"
    assert p.margin["n"] == pytest.approx(20.0)
    assert p.margin["theta"] == pytest.approx(0.300)


def test_player_posteriors_covers_only_relevant_cats():
    from src.player_model.posterior import player_posteriors
    cfg = LeagueConfig()
    hit = player_posteriors(_hitter_row(), cfg)
    assert set(hit.keys()) == set(cfg.hitting_categories)     # hitter -> 6 hitting cats only
    pit = player_posteriors(_pitcher_row(), cfg)
    assert set(pit.keys()) == set(cfg.pitching_categories)    # pitcher -> 6 pitching cats only


def test_sigma2_wired_to_ytd_sample_through_posterior():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    rookie = category_posterior(_hitter_row(hr=30, ytd_pa=40), "HR", cfg)
    veteran = category_posterior(_hitter_row(hr=30, ytd_pa=1500), "HR", cfg)
    assert rookie.sigma2 > veteran.sigma2     # low YTD sample -> wider true-talent band
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_posterior.py -k "category_posterior or player_posteriors or wired" -q`
Expected: FAIL — `ImportError: cannot import name 'category_posterior'` (and `player_posteriors`)

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/posterior.py
def _is_hitter(row) -> bool:
    raw = row.get("is_hitter") if hasattr(row, "get") else row["is_hitter"]
    return _f(raw, default=1.0) >= 0.5


def _weekly_volume(row, kind: str, weeks: float) -> float:
    """Per-week AB (rate_prop) or IP (rate_ratio) from the pool's season projection."""
    col = _HITTER_VOL_COL if kind == "rate_prop" else _PITCHER_VOL_COL
    season = _f(row.get(col)) if hasattr(row, "get") else _f(row[col])
    return season / weeks if weeks > 0 else 0.0


def category_posterior(row, category: str, config: LeagueConfig | None = None,
                       weeks: float | None = None) -> CategoryPosterior:
    """Build the per-category posterior for one pool row. `weeks` defaults to
    config.season_weeks. Never raises on missing columns / NaN (degrades gracefully)."""
    cfg = config or LeagueConfig()
    weeks = float(weeks) if weeks else float(cfg.season_weeks)
    kind = classify_kind(category, cfg)
    is_hit = _is_hitter(row)
    mean = _mean_for(row, category, kind, cfg, weeks)
    n = _f(row.get("ytd_pa")) if is_hit else _f(row.get("ytd_ip"))
    sigma2 = between_player_sigma2(mean, kind, category, n, is_hit)
    weekly_vol = _weekly_volume(row, kind, weeks) if kind != "counting" else 0.0
    tau2, margin = week_to_week_tau2(mean, kind, category, weekly_vol)
    return CategoryPosterior(category=category, kind=kind, mean=mean,
                             sigma2=sigma2, tau2=tau2, margin=margin)


def player_posteriors(row, config: LeagueConfig | None = None,
                      weeks: float | None = None) -> dict[str, CategoryPosterior]:
    """All posteriors for the cats relevant to this player (hitting cats for hitters,
    pitching cats for pitchers, by the pool `is_hitter` flag)."""
    cfg = config or LeagueConfig()
    cats = cfg.hitting_categories if _is_hitter(row) else cfg.pitching_categories
    return {c: category_posterior(row, c, cfg, weeks) for c in cats}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_posterior.py -q`
Expected: PASS (all tests in the file, including the Task-1 tests previously erroring)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/posterior.py tests/test_player_model_posterior.py
git commit -m "feat(player_model): category_posterior + player_posteriors orchestrators (P1 slice1)"
```

---

### Task 5: Robustness — never-raise on degenerate rows + LeagueConfig category lock

The posterior must degrade gracefully (the live pool has NaN/missing columns for partial players), and its category taxonomy must stay locked to `LeagueConfig` (HEATER's single-source-of-truth invariant — no hardcoded category lists).

**Files:**
- Modify: `src/player_model/posterior.py` (only if a guard is missing)
- Test: `tests/test_player_model_posterior.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_posterior.py
def test_never_raises_on_nan_and_missing_columns():
    from src.player_model.posterior import category_posterior, player_posteriors
    cfg = LeagueConfig()
    sparse = pd.Series({"player_id": 9, "is_hitter": 1, "hr": np.nan})  # missing most columns
    p = category_posterior(sparse, "HR", cfg)
    assert math.isfinite(p.mean) and math.isfinite(p.sigma2) and math.isfinite(p.tau2)
    assert p.sigma2 > 0  # floor still applies even with a NaN mean (degrades to 0 mean -> floor only path)
    out = player_posteriors(sparse, cfg)
    assert set(out.keys()) == set(cfg.hitting_categories)


def test_pitcher_rate_ratio_uses_ip_volume_and_ytd_ip_sample():
    from src.player_model.posterior import category_posterior
    cfg = LeagueConfig()
    p = category_posterior(_pitcher_row(era=3.20, ip=180.0, ytd_ip=60.0), "ERA", cfg, weeks=26)
    assert p.kind == "rate_ratio"
    assert p.margin["dist"] == "ratio_normal"
    assert p.mean == pytest.approx(3.20)
    assert math.isfinite(p.sigma2) and p.sigma2 > 0


def test_categories_locked_to_league_config():
    # No hardcoded category list — every produced category is a LeagueConfig category.
    from src.player_model.posterior import player_posteriors
    cfg = LeagueConfig()
    keys = set(player_posteriors(_hitter_row(), cfg)) | set(player_posteriors(_pitcher_row(), cfg))
    assert keys == set(cfg.all_categories)
```

- [ ] **Step 2: Run test to verify it fails (or passes if guards already hold)**

Run: `python -m pytest tests/test_player_model_posterior.py -k "never_raises or rate_ratio or locked" -q`
Expected: the guards added in Tasks 1–4 (`_f`, `.get` fallbacks, `_MIN_WEEK_VOL`) should already make these PASS. If `test_never_raises_on_nan_and_missing_columns` fails because `row.get` is unavailable on a plain dict path or a missing column raises, add the fix in Step 3.

- [ ] **Step 3: Write minimal implementation (only if a test failed)**

```python
# If a missing-column access raised, ensure every row access uses .get with a default.
# `_mean_for`, `_is_hitter`, `_weekly_volume`, and `category_posterior` already route through
# `_f(row.get(col))`. If pd.Series.get on an absent key surfaced a non-default, harden _f's
# call sites — no change expected; this step is a safety net, not a placeholder.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_posterior.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/posterior.py tests/test_player_model_posterior.py
git commit -m "test(player_model): degenerate-row safety + LeagueConfig category lock (P1 slice1)"
```

---

## Self-Review

**Spec coverage (§3 Layer 0 posterior + §6 module boundary):**
- "heteroscedastic true-talent posterior per category … σ² (player-to-player) and τ² (week-to-week) separately" → `between_player_sigma2` (Task 2) + `week_to_week_tau2` (Task 3), separate fields on `CategoryPosterior`. ✅
- "NB margins for counting cats, beta-binomial for rate cats — variance unequal across players" → NB margin (counting) + beta-binomial margin (AVG/OBP); ERA/WHIP correctly routed to ratio-normal since they are NOT [0,1] proportions (an honest correction beyond the spec's loose "rate cats" wording, flagged in the module docstring). Unequal-across-players via the sample-dependent shrink (Task 2). ✅
- Gap G3 (no falsely-tight bands) → the never-vanishing projection-error floor in `between_player_sigma2`. ✅
- Tier-1 #2 (same margins both tiers) → the `margin` descriptor is the single shared parameterization Layer 1/2 consume. ✅
- §6 module boundary `player_model` → this slice delivers the per-cat posterior half; availability + g-score + facade are slices 2/3/4 (scope note). ✅
- **Deferred (named follow-ons, out of this slice):** SURE-optimal λ tuning + per-category constant calibration (slice 5), availability survival (slice 2), G-score (slice 3), the unified facade (slice 4), Layer-1/2 sampling FROM these margins (Phase 2/3).

**Placeholder scan:** Task 5 Step 3 is a conditional safety-net (explicitly "no change expected"), not a placeholder — every other step has complete runnable code + exact commands. No "TODO"/"handle edge cases"/"similar to Task N".

**Type consistency:** `CategoryPosterior(category, kind, mean, sigma2, tau2, margin)` used identically across Tasks 1/4/5. `classify_kind(category, config)`, `between_player_sigma2(mean, kind, category, n, is_hitter)`, `week_to_week_tau2(mean, kind, category, weekly_vol) -> (tau2, margin)`, `category_posterior(row, category, config, weeks)`, `player_posteriors(row, config, weeks)` — signatures match every call site in tests and orchestrators. `margin` dist keys (`"nb"`/`"beta_binomial"`/`"ratio_normal"`) and their fields (`mean`/`r`; `theta`/`n`/`rho`; `mean`/`std_week`) are consistent between Task 3's impl and Tasks 3/4's assertions. Constants (`_TALENT_CV`, `_PROJ_FLOOR_CV`, `_STABILIZE_PA/IP`, `_COUNTING_OVERDISPERSION`, `_RATE_*`) are defined once in Task 1 and read in Tasks 2/3.

**Reuse (DRY):** generalizes `weekly_matrix.build_team_kalman_variances`' shrink pattern (cited, not duplicated); `LeagueConfig` for all category/STAT_MAP/rate/inverse taxonomy (no hardcoded lists); seeds traceable to `scenario_generator._RATE_STD`/`DEFAULT_CV`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase1-slice1-posterior-variance.md`.
