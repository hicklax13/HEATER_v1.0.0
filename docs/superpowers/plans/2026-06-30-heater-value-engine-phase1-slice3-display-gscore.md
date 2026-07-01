# Phase 1 Slice 3 — Layer-0 Display G-Score (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rosenof **G-score** as a per-player, per-category interpretable DISPLAY/RANKING scalar — and a player-level aggregate — computed from the slice-1 posterior's mean + week-to-week variance against league context. It is **display-only**: it gives UI a readable "how valuable is this player in this category" number, but it is **never** the probability engine (the proxy/deep tiers compute win/tie/loss from heteroscedastic NB/Skellam — gap G5). It correctly reduces to the classic z-score in the limit.

**Architecture:** A new focused module `src/player_model/gscore.py` implementing `G = (μ − μ_league) / √(σ²_league + κ·τ²)`, κ = 2N/(2N−1) (Rosenof, arXiv 2307.02188v4), where μ/τ² come from a slice-1 `CategoryPosterior` and μ_league/σ²_league/N are league context. Inverse cats (L/ERA/WHIP) flip the numerator so "good" is always positive. A player-level aggregate sums the signed per-category G across the player's relevant cats (SGP-like). Pure functions; no DB / network; never raises (degenerate denominators → 0.0).

**Tech Stack:** Python 3.12/3.14, `math`, dataclasses, pytest. Reuses `src/player_model/posterior.py` (`CategoryPosterior`) + `src/valuation.py` (`LeagueConfig` for inverse-stat + category taxonomy).

**Spec:** `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` §3 (Layer 1 "G-score = DISPLAY-ONLY interpretable per-player scalar — never the probability engine") + §6 (`player_model` surfaces `display_g_score`) + research-validation gap G5 ("use G-score only as an interpretable scalar… drop its equal-variance + normal assumptions; Z-scores are equivalent to G-scores with |W|=1").

**Scope note:** This slice computes G from a posterior + league context passed in. It does NOT compute the league context from the pool (that is the slice-4 facade's job) and does NOT feed any probability calculation. League context (`LeagueContext`) is a small input dataclass with sensible fallbacks so the module is testable in isolation.

---

## File Structure

- **Create `src/player_model/gscore.py`** — `LeagueContext` input dataclass + `category_gscore()` + `player_gscore()`. One responsibility: turn (posterior, league context) into a display G-score.
- **Create `tests/test_player_model_gscore.py`** — TDD tests.

(`src/player_model/__init__.py` and `posterior.py` already exist from slice 1.)

---

### Task 1: `LeagueContext` + `kappa` + `category_gscore`

`category_gscore` standardizes a category mean against the league's per-player spread. κ = 2N/(2N−1) is the small-sample correction for a head-to-head difference of two N-player sums (κ → 1 as N grows). Inverse cats flip the numerator sign so good play is always positive. A zero denominator → 0.0 (no signal, never a divide error).

**Files:**
- Create: `src/player_model/gscore.py`
- Test: `tests/test_player_model_gscore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_model_gscore.py
"""Tests for the Layer-0 display G-score (Phase 1 slice 3)."""

import math

import pytest

from src.valuation import LeagueConfig


def test_kappa_small_sample_correction():
    from src.player_model.gscore import kappa
    # kappa = 2N/(2N-1): >1, decreasing toward 1 as N grows.
    assert kappa(1) == pytest.approx(2 / 1)          # 2.0
    assert kappa(10) == pytest.approx(20 / 19)
    assert kappa(10) < kappa(2)
    assert kappa(10_000) == pytest.approx(1.0, abs=1e-3)


def test_category_gscore_above_league_is_positive():
    from src.player_model.gscore import category_gscore
    g = category_gscore(mean=1.5, tau2=1.8, league_mean=1.0, league_sd=0.4, n_slots=10, inverse=False)
    assert g > 0   # above-league HR rate -> positive value


def test_category_gscore_below_league_is_negative():
    from src.player_model.gscore import category_gscore
    g = category_gscore(mean=0.6, tau2=1.8, league_mean=1.0, league_sd=0.4, n_slots=10, inverse=False)
    assert g < 0


def test_category_gscore_inverse_lower_is_better():
    from src.player_model.gscore import category_gscore
    # ERA: a 3.00 ERA vs a 4.00 league mean is GOOD -> positive G (numerator flipped).
    g = category_gscore(mean=3.00, tau2=0.5, league_mean=4.00, league_sd=0.7, n_slots=8, inverse=True)
    assert g > 0
    worse = category_gscore(mean=5.00, tau2=0.5, league_mean=4.00, league_sd=0.7, n_slots=8, inverse=True)
    assert worse < 0


def test_category_gscore_reduces_to_zscore_when_tau2_zero():
    from src.player_model.gscore import category_gscore
    # tau2=0 -> denominator = league_sd -> classic z-score (Rosenof: |W|=1 special case).
    g = category_gscore(mean=1.4, tau2=0.0, league_mean=1.0, league_sd=0.5, n_slots=10, inverse=False)
    assert g == pytest.approx((1.4 - 1.0) / 0.5)


def test_category_gscore_zero_denominator_is_zero_not_error():
    from src.player_model.gscore import category_gscore
    g = category_gscore(mean=1.4, tau2=0.0, league_mean=1.0, league_sd=0.0, n_slots=10, inverse=False)
    assert g == 0.0   # no spread + no noise -> no defined signal, return 0 (never divide-by-zero)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_gscore.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.player_model.gscore'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/player_model/gscore.py
"""Layer-0 display G-score (Rosenof, arXiv 2307.02188v4) — a per-player, per-category
interpretable VALUE/RANKING scalar. DISPLAY ONLY: it gives the UI a readable "how good is
this player in this category" number; it is NEVER the win/tie/loss probability engine (that
is the heteroscedastic NB/Skellam in the proxy/deep tiers — research-validation gap G5).

    G = (mu - mu_league) / sqrt(sigma2_league + kappa * tau2),   kappa = 2N / (2N - 1)

mu/tau2 come from a slice-1 CategoryPosterior; mu_league/sigma2_league/N are league context.
Inverse cats (L/ERA/WHIP) flip the numerator so good play is always positive. Reduces to the
classic z-score when tau2 -> 0 (the paper's |W|=1 special case). Pure; never raises; a zero
denominator returns 0.0 (no defined signal, not a divide error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.valuation import LeagueConfig


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass(frozen=True)
class LeagueContext:
    """Per-category league context for the G-score denominator/numerator.
    `means`/`sds` are per-category {CAT: value}; `n_slots` is the number of roster slots
    contributing to a category total (hitters vs pitchers), driving the kappa correction."""

    means: dict = field(default_factory=dict)
    sds: dict = field(default_factory=dict)
    hitter_slots: int = 10   # FourzynBurn active hitters: C/1B/2B/3B/SS/3OF/2Util
    pitcher_slots: int = 8   # FourzynBurn active pitchers: 2SP/2RP/4P


def kappa(n_slots: int) -> float:
    """Small-sample correction 2N/(2N-1) for a head-to-head difference of two N-player sums.
    >1, decreasing to 1 as N grows. N<=0 -> 1.0 (no correction)."""
    n = int(n_slots)
    if n <= 0:
        return 1.0
    return (2.0 * n) / (2.0 * n - 1.0)


def category_gscore(mean: float, tau2: float, league_mean: float, league_sd: float,
                    n_slots: int, inverse: bool = False) -> float:
    """Rosenof G for one category: (mu - mu_league)/sqrt(sigma2_league + kappa*tau2).
    Inverse cats flip the numerator (lower is better -> positive G). Zero denominator -> 0.0."""
    mu = _f(mean)
    t2 = max(0.0, _f(tau2))
    lm = _f(league_mean)
    lsd = _f(league_sd)
    denom2 = lsd * lsd + kappa(n_slots) * t2
    if denom2 <= 0.0:
        return 0.0
    num = (lm - mu) if inverse else (mu - lm)
    return float(num / math.sqrt(denom2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_gscore.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/gscore.py tests/test_player_model_gscore.py
git commit -m "feat(player_model): Rosenof category G-score (display-only) + kappa (P1 slice3)"
```

---

### Task 2: `player_gscore` — aggregate signed per-category G from a posterior set

Compute the player's overall display value by summing the signed per-category G (inverse cats already flipped to positive-is-good) across the player's relevant cats — the SGP-like aggregate. Consumes the slice-1 `player_posteriors` output + a `LeagueContext`.

**Files:**
- Modify: `src/player_model/gscore.py`
- Test: `tests/test_player_model_gscore.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_gscore.py
import pandas as pd  # noqa: E402

from src.player_model.posterior import player_posteriors  # noqa: E402


def _elite_hitter():
    return pd.Series(dict(
        player_id=1, name="Elite", is_hitter=1,
        r=110, hr=42, rbi=115, sb=20, avg=0.310, obp=0.390,
        ab=560, pa=640, ytd_pa=600, ytd_ip=0.0,
    ))


def _replacement_hitter():
    return pd.Series(dict(
        player_id=2, name="Repl", is_hitter=1,
        r=55, hr=10, rbi=50, sb=3, avg=0.235, obp=0.290,
        ab=480, pa=520, ytd_pa=500, ytd_ip=0.0,
    ))


def _league_ctx():
    from src.player_model.gscore import LeagueContext
    # Per-week league means + per-player league spreads (rough mid-pack values).
    return LeagueContext(
        means={"R": 80 / 26, "HR": 22 / 26, "RBI": 80 / 26, "SB": 10 / 26, "AVG": 0.255, "OBP": 0.320},
        sds={"R": 0.6, "HR": 0.35, "RBI": 0.6, "SB": 0.4, "AVG": 0.022, "OBP": 0.025},
    )


def test_player_gscore_elite_beats_replacement():
    from src.player_model.gscore import player_gscore
    cfg = LeagueConfig()
    ctx = _league_ctx()
    elite = player_gscore(player_posteriors(_elite_hitter(), cfg), ctx, cfg)
    repl = player_gscore(player_posteriors(_replacement_hitter(), cfg), ctx, cfg)
    assert elite > repl
    assert elite > 0   # an above-average hitter has positive aggregate value


def test_player_gscore_returns_total_and_per_cat():
    from src.player_model.gscore import player_gscore
    cfg = LeagueConfig()
    out = player_gscore(player_posteriors(_elite_hitter(), cfg), _league_ctx(), cfg, detail=True)
    assert "total" in out and "per_category" in out
    assert set(out["per_category"]) == set(cfg.hitting_categories)
    assert out["total"] == pytest.approx(sum(out["per_category"].values()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_gscore.py -k player_gscore -q`
Expected: FAIL — `ImportError: cannot import name 'player_gscore'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/gscore.py
def player_gscore(posteriors: dict, context: LeagueContext,
                  config: LeagueConfig | None = None, detail: bool = False):
    """Aggregate display value: sum of signed per-category G across the player's posteriors
    (inverse cats flipped to positive-is-good). `posteriors` is the slice-1 player_posteriors
    output {CAT: CategoryPosterior}. Returns a float total, or {'total', 'per_category'} when
    detail=True. Never raises."""
    cfg = config or LeagueConfig()
    inverse = cfg.inverse_stats
    hitting = set(cfg.hitting_categories)
    per_cat: dict[str, float] = {}
    for cat, post in posteriors.items():
        n_slots = context.hitter_slots if cat in hitting else context.pitcher_slots
        per_cat[cat] = category_gscore(
            mean=post.mean, tau2=post.tau2,
            league_mean=context.means.get(cat, 0.0),
            league_sd=context.sds.get(cat, 0.0),
            n_slots=n_slots, inverse=cat in inverse,
        )
    total = float(sum(per_cat.values()))
    return {"total": total, "per_category": per_cat} if detail else total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_gscore.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/gscore.py tests/test_player_model_gscore.py
git commit -m "feat(player_model): player_gscore aggregate display value (P1 slice3)"
```

---

### Task 3: Robustness — degenerate inputs + lint

The facade may pass an empty `LeagueContext` (missing cats → fallback 0 spread) or a posterior with NaN fields. The module must never raise and must pass ruff.

**Files:**
- Modify: `src/player_model/gscore.py` (only if a guard is missing)
- Test: `tests/test_player_model_gscore.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_gscore.py
def test_player_gscore_empty_context_no_raise():
    from src.player_model.gscore import player_gscore, LeagueContext
    cfg = LeagueConfig()
    out = player_gscore(player_posteriors(_elite_hitter(), cfg), LeagueContext(), cfg, detail=True)
    # No league spreads -> every denominator collapses -> all-zero, but no error.
    assert math.isfinite(out["total"])
    assert all(math.isfinite(v) for v in out["per_category"].values())


def test_category_gscore_nan_inputs_safe():
    from src.player_model.gscore import category_gscore
    g = category_gscore(mean=float("nan"), tau2=float("nan"), league_mean=float("nan"),
                        league_sd=float("nan"), n_slots=10, inverse=False)
    assert g == 0.0   # all-NaN -> coerced to 0 -> zero denominator -> 0.0


def test_pitcher_gscore_uses_pitcher_slots_and_inverse():
    from src.player_model.gscore import player_gscore, LeagueContext
    cfg = LeagueConfig()
    pitcher = pd.Series(dict(
        player_id=3, name="Ace", is_hitter=0,
        w=16, l=6, sv=0, k=220, era=2.90, whip=1.02, ip=190.0, ytd_ip=70.0, ytd_pa=0,
    ))
    ctx = LeagueContext(
        means={"W": 10 / 26, "L": 8 / 26, "SV": 5 / 26, "K": 180 / 26, "ERA": 3.95, "WHIP": 1.25},
        sds={"W": 0.2, "L": 0.2, "SV": 0.6, "K": 1.5, "ERA": 0.7, "WHIP": 0.12},
    )
    out = player_gscore(player_posteriors(pitcher, cfg), ctx, cfg, detail=True)
    # A 2.90-ERA ace beats a 3.95 league ERA -> positive ERA G (inverse handled).
    assert out["per_category"]["ERA"] > 0
    assert out["per_category"]["WHIP"] > 0
    assert set(out["per_category"]) == set(cfg.pitching_categories)
```

- [ ] **Step 2: Run test to verify it fails (or passes if guards already hold)**

Run: `python -m pytest tests/test_player_model_gscore.py -k "empty_context or nan_inputs or pitcher" -q`
Expected: the `_f` coercion + zero-denominator guard from Task 1 should already make these PASS. If a NaN leaked, harden in Step 3.

- [ ] **Step 3: Write minimal implementation (only if a test failed)**

```python
# Safety net: category_gscore already routes every numeric input through _f and returns 0.0
# on a non-positive denominator, so NaN/empty-context paths cannot raise or leak NaN. No
# change expected; this step is a guard, not a placeholder.
```

- [ ] **Step 4: Run test to verify it passes + lint**

Run: `python -m pytest tests/test_player_model_gscore.py -q`
Expected: PASS (all tests)

Run: `python -m ruff check src/player_model/gscore.py tests/test_player_model_gscore.py && python -m ruff format src/player_model/gscore.py tests/test_player_model_gscore.py`
Expected: no lint errors; format clean.

- [ ] **Step 5: Commit**

```bash
git add src/player_model/gscore.py tests/test_player_model_gscore.py
git commit -m "test(player_model): G-score degenerate-input safety + lint (P1 slice3)"
```

---

## Self-Review

**Spec coverage (§3 G-score display-only + gap G5):**
- "G-score = DISPLAY-ONLY interpretable per-player scalar — never the probability engine" → `category_gscore`/`player_gscore` compute a value scalar; no probability is produced; the docstring states the boundary. ✅
- Rosenof formula `G=(μ−μ_league)/√(σ²+κτ²)`, κ=2N/(2N−1) → `category_gscore` + `kappa` (Task 1). ✅
- "Z-scores are equivalent to G-scores with |W|=1" (drop equal-variance/normal assumptions, keep the mean-vs-league intuition) → `test_category_gscore_reduces_to_zscore_when_tau2_zero`. ✅
- Inverse-stat correctness (L/ERA/WHIP) → numerator flip; `test_category_gscore_inverse_lower_is_better` + the pitcher aggregate test. ✅
- §6 `display_g_score` surfaced → `player_gscore` aggregate (the facade attaches it). ✅
- **Deferred (named follow-ons):** computing `LeagueContext` from the pool (slice-4 facade); any use of G in ranking UI (consumer surfaces, later phases).

**Placeholder scan:** Task 3 Step 3 is a conditional safety-net ("no change expected"), not a placeholder. Every other step has complete runnable code + exact commands.

**Type consistency:** `LeagueContext(means, sds, hitter_slots, pitcher_slots)`, `kappa(n_slots)`, `category_gscore(mean, tau2, league_mean, league_sd, n_slots, inverse)`, `player_gscore(posteriors, context, config, detail)` are used identically across tasks/tests. `player_gscore` consumes the slice-1 `player_posteriors` dict ({CAT: CategoryPosterior}) and reads `.mean`/`.tau2` — fields confirmed present on `CategoryPosterior` (slice 1).

**Reuse (DRY):** `LeagueConfig` for inverse-stat + hitting/pitching taxonomy (no hardcoded category lists); `CategoryPosterior` from slice 1 (no re-derivation of mean/variance). `_f` mirrors the slice-1/slice-2 NaN-safe helper (a future cleanup may hoist a shared `player_model` util).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase1-slice3-display-gscore.md`.
