# Phase 1 Slice 4 — Unified `player_model` Facade (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tie slices 1–3 into the spec's single Layer-0 contract: `(player_row, league_context) → PlayerModel{per-category posterior, availability_survival, display_g_score}`. Add the one missing piece — `build_league_context`, which derives the per-category league means + spreads from the pool (the context the G-score needs) at the same per-week/rate scale as the posteriors. This is the single object every later layer (Layer-1 proxy, Layer-2 deep MC) and every surface consumes — no surface re-derives a value.

**Architecture:** A new `src/player_model/model.py` holding `PlayerModel` + `build_league_context()` + `build_player_model()` + `build_player_models()` (batch), and `src/player_model/__init__.py` re-exports the package's public API. The facade COMPOSES the locked slice APIs — `posterior.player_posteriors`, `availability.availability_survival`, `gscore.build_league_context`-inputs + `gscore.player_gscore` — and adds no new statistics beyond the league-context aggregation. Pure; no DB / network; never raises.

**Tech Stack:** Python 3.12/3.14, pandas, NumPy, dataclasses, pytest. Reuses `src/player_model/{posterior,availability,gscore}.py` + `src/valuation.py` (`LeagueConfig`).

**Spec:** `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` §6 (module boundary: `player_model` (Layer 0): `(player_id, league_ctx) → {per-cat posterior (mean, σ², τ²), availability_survival, display_g_score}`) + §5 (single source of truth — "Every surface consumes Layer 0/1… never its own valuation").

**Scope note:** This slice assembles the Layer-0 output and computes league context from the pool. It does NOT wire any consumer surface (Phase 2+ wires Trades/Optimizer/etc.) and does NOT add Layer-1 win-prob (Phase 2). Current status / return-date per player are accepted via optional maps (the caller sources them from `league_rosters` / news); absent → treated active. Slice 5 validates the whole Layer-0 against the backtest harness.

**Depends on (locked public interfaces):**
- slice 1 `posterior.py`: `player_posteriors(row, config=None, weeks=None) -> dict[str, CategoryPosterior]`; `CategoryPosterior{category, kind, mean, sigma2, tau2, margin}`.
- slice 2 `availability.py`: `availability_survival(row, status=None, expected_return_days=None, weeks_remaining=None) -> AvailabilitySurvival`.
- slice 3 `gscore.py`: `LeagueContext{means, sds, hitter_slots, pitcher_slots}`; `player_gscore(posteriors, context, config=None, detail=False) -> float | {total, per_category}`.

---

## File Structure

- **Create `src/player_model/model.py`** — `PlayerModel` dataclass + `build_league_context()` + `build_player_model()` + `build_player_models()`. One responsibility: assemble the Layer-0 output.
- **Modify `src/player_model/__init__.py`** — re-export the public API (`PlayerModel`, `build_player_model`, `build_player_models`, `build_league_context`, and convenience re-exports `CategoryPosterior`, `AvailabilitySurvival`, `LeagueContext`).
- **Create `tests/test_player_model_facade.py`** — TDD tests.

---

### Task 1: `build_league_context` — per-category league means + spreads from the pool

The G-score needs per-category league baselines at the SAME scale as the posterior means (per-week for counting cats: season ÷ weeks; the rate itself for rate cats). Compute mean + std across fantasy-relevant players only (a volume filter excludes AAA scrubs that would drag the baseline), per hitting/pitching cat.

**Files:**
- Create: `src/player_model/model.py`
- Test: `tests/test_player_model_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_model_facade.py
"""Tests for the unified player_model facade (Phase 1 slice 4)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.valuation import LeagueConfig


def _pool(n_hitters=8, n_pitchers=6, seed=0):
    """A small synthetic pool: fantasy-relevant hitters + pitchers + a couple scrubs."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_hitters):
        rows.append(dict(
            player_id=100 + i, name=f"H{i}", is_hitter=1, age=27, positions="OF",
            health_score=0.9, r=70 + i * 4, hr=18 + i * 2, rbi=70 + i * 3, sb=8 + i,
            avg=0.250 + i * 0.005, obp=0.320 + i * 0.005, ab=520, pa=580,
            ytd_pa=300, ytd_ip=0.0, w=0, l=0, sv=0, k=0, era=0.0, whip=0.0, ip=0.0,
        ))
    for i in range(n_pitchers):
        rows.append(dict(
            player_id=200 + i, name=f"P{i}", is_hitter=0, age=28, positions="SP",
            health_score=0.9, r=0, hr=0, rbi=0, sb=0, avg=0.0, obp=0.0, ab=0, pa=0,
            w=8 + i, l=7, sv=0, k=160 + i * 10, era=4.2 - i * 0.2, whip=1.30 - i * 0.03,
            ip=150, ytd_pa=0, ytd_ip=60.0,
        ))
    # two zero-volume scrubs that must be filtered out of league baselines
    rows.append(dict(player_id=900, name="ScrubH", is_hitter=1, age=24, positions="OF",
                     health_score=0.85, r=0, hr=0, rbi=0, sb=0, avg=0.0, obp=0.0, ab=0, pa=0,
                     ytd_pa=0, ytd_ip=0.0, w=0, l=0, sv=0, k=0, era=0.0, whip=0.0, ip=0.0))
    rows.append(dict(player_id=901, name="ScrubP", is_hitter=0, age=24, positions="SP",
                     health_score=0.85, r=0, hr=0, rbi=0, sb=0, avg=0.0, obp=0.0, ab=0, pa=0,
                     w=0, l=0, sv=0, k=0, era=0.0, whip=0.0, ip=0.0, ytd_pa=0, ytd_ip=0.0))
    return pd.DataFrame(rows)


def test_league_context_has_all_categories():
    from src.player_model.model import build_league_context
    cfg = LeagueConfig()
    ctx = build_league_context(_pool(), cfg)
    assert set(ctx.means) == set(cfg.all_categories)
    assert set(ctx.sds) == set(cfg.all_categories)
    assert all(math.isfinite(v) for v in ctx.means.values())
    assert all(v >= 0 for v in ctx.sds.values())


def test_league_context_counting_mean_is_per_week():
    from src.player_model.model import build_league_context
    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg, weeks=26)
    # HR league mean should equal mean(season HR of relevant hitters) / 26.
    relevant = pool[(pool["is_hitter"] == 1) & (pool["pa"] >= 1)]
    expected = float(relevant["hr"].mean()) / 26
    assert ctx.means["HR"] == pytest.approx(expected, rel=1e-6)


def test_league_context_excludes_zero_volume_scrubs():
    from src.player_model.model import build_league_context
    cfg = LeagueConfig()
    ctx_full = build_league_context(_pool(), cfg)
    # The two 0-PA/0-IP scrubs must not drag the AVG/ERA baselines toward 0.
    assert ctx_full.means["AVG"] > 0.20
    assert ctx_full.means["ERA"] > 2.0


def test_league_context_empty_pool_no_raise():
    from src.player_model.model import build_league_context
    cfg = LeagueConfig()
    ctx = build_league_context(pd.DataFrame(), cfg)
    assert set(ctx.means) == set(cfg.all_categories)
    assert all(v == 0.0 for v in ctx.sds.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_facade.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.player_model.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/player_model/model.py
"""Unified Layer-0 player model facade (Advanced Value Engine, Phase 1).

Assembles the single source-of-truth Layer-0 output per the spec §6 contract:
    (player_row, league_context) -> PlayerModel{posteriors, availability, display_g_score}

Composes the locked slice APIs (posterior.player_posteriors, availability.availability_survival,
gscore.player_gscore) and adds the one missing piece — build_league_context, the per-category
league baseline the G-score standardizes against, computed at the posterior's per-week/rate scale.
Pure; no DB / network; never raises. Every later layer and surface consumes a PlayerModel; none
re-derives a value (spec §5 single source of truth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.player_model.availability import AvailabilitySurvival, availability_survival
from src.player_model.gscore import LeagueContext, player_gscore
from src.player_model.posterior import CategoryPosterior, player_posteriors
from src.valuation import LeagueConfig

# Volume floors that define "fantasy-relevant" for the league baseline (excludes AAA scrubs).
# Calibratable (slice 5). A hitter needs projected PA, a pitcher projected IP, to count.
_MIN_CONTEXT_HITTER_PA: float = 200.0
_MIN_CONTEXT_PITCHER_IP: float = 30.0


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _col_per_week(frame: pd.DataFrame, col: str, weeks: float) -> pd.Series:
    vals = pd.to_numeric(frame.get(col), errors="coerce") if col in frame else pd.Series(dtype=float)
    return vals.dropna() / weeks if weeks > 0 else vals.dropna()


def build_league_context(
    pool: pd.DataFrame, config: LeagueConfig | None = None, weeks: float | None = None
) -> LeagueContext:
    """Per-category league mean + spread from the fantasy-relevant slice of the pool, at the
    posterior's scale (per-week for counting cats, the rate for rate cats). Empty/degenerate ->
    zero means/sds (never raises).

    NOTE (display-only G-score scale): for RATE cats the resulting `sds` are season-talent
    spreads, while the slice-1 posterior `tau2` is the weekly beta-binomial outcome variance,
    which is the larger term — so the rate-cat G-score denominator is tau2-dominated and the
    rate-cat G is a per-week RANKING signal, not a calibrated season z-score. This is intentional
    and honest (a 20-AB week is genuinely noisier than the cross-player talent spread); calibrated
    win/tie/loss probability is the NB/Skellam tiers' job, not the display G (gap G5)."""
    cfg = config or LeagueConfig()
    weeks = float(weeks) if (weeks is not None and weeks > 0) else float(cfg.season_weeks)
    means: dict[str, float] = {}
    sds: dict[str, float] = {}

    if pool is None or len(pool) == 0:
        return LeagueContext(means={c: 0.0 for c in cfg.all_categories},
                             sds={c: 0.0 for c in cfg.all_categories})

    is_hit = pd.to_numeric(pool.get("is_hitter", 1), errors="coerce").fillna(1) >= 0.5
    pa = pd.to_numeric(pool.get("pa", 0), errors="coerce").fillna(0)
    ip = pd.to_numeric(pool.get("ip", 0), errors="coerce").fillna(0)
    hitters = pool[is_hit & (pa >= _MIN_CONTEXT_HITTER_PA)]
    pitchers = pool[(~is_hit) & (ip >= _MIN_CONTEXT_PITCHER_IP)]

    hitting = set(cfg.hitting_categories)
    for cat in cfg.all_categories:
        frame = hitters if cat in hitting else pitchers
        col = cfg.STAT_MAP.get(cat, cat.lower())
        if cat in cfg.rate_stats:
            series = pd.to_numeric(frame.get(col), errors="coerce").dropna() if col in frame else pd.Series(dtype=float)
            series = series[series > 0]   # rate of 0 means "no data", not a true 0.000 talent
        else:
            series = _col_per_week(frame, col, weeks)
        means[cat] = float(series.mean()) if len(series) else 0.0
        sds[cat] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

    return LeagueContext(means=means, sds=sds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_facade.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/model.py tests/test_player_model_facade.py
git commit -m "feat(player_model): build_league_context pool aggregation (P1 slice4)"
```

---

### Task 2: `PlayerModel` + `build_player_model` — assemble the Layer-0 contract for one player

Tie posteriors (slice 1) + availability (slice 2) + display G-score (slice 3) into the single `PlayerModel`. Status / return-date are optional inputs.

**Files:**
- Modify: `src/player_model/model.py`
- Test: `tests/test_player_model_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_facade.py
def test_build_player_model_assembles_all_three_layers():
    from src.player_model.model import build_league_context, build_player_model
    from src.player_model.availability import AvailabilitySurvival
    from src.player_model.posterior import CategoryPosterior
    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    elite = pool.iloc[7]  # highest-stat hitter (H7)
    pm = build_player_model(elite, ctx, cfg)
    assert pm.player_id == int(elite["player_id"])
    assert pm.is_hitter is True
    assert set(pm.posteriors) == set(cfg.hitting_categories)
    assert all(isinstance(v, CategoryPosterior) for v in pm.posteriors.values())
    assert isinstance(pm.availability, AvailabilitySurvival)
    assert math.isfinite(pm.g_score)
    assert set(pm.g_by_category) == set(cfg.hitting_categories)


def test_build_player_model_gscore_orders_players():
    from src.player_model.model import build_league_context, build_player_model
    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    elite = build_player_model(pool.iloc[7], ctx, cfg)   # best hitter
    weak = build_player_model(pool.iloc[0], ctx, cfg)    # weakest hitter
    assert elite.g_score > weak.g_score


def test_build_player_model_availability_reflects_status():
    from src.player_model.model import build_league_context, build_player_model
    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    healthy = build_player_model(pool.iloc[3], ctx, cfg, status=None)
    injured = build_player_model(pool.iloc[3], ctx, cfg, status="IL60")
    assert injured.availability.expected_active_fraction < healthy.availability.expected_active_fraction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_facade.py -k build_player_model -q`
Expected: FAIL — `ImportError: cannot import name 'build_player_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/model.py
@dataclass(frozen=True)
class PlayerModel:
    """The single Layer-0 output every later layer / surface consumes (spec §6). `posteriors`
    is {CAT: CategoryPosterior} for the player's relevant cats; `availability` is the survival
    distribution; `g_score`/`g_by_category` are the display-only Rosenof value scalars."""

    player_id: int
    name: str
    is_hitter: bool
    posteriors: dict = field(default_factory=dict)
    availability: AvailabilitySurvival | None = None
    g_score: float = 0.0
    g_by_category: dict = field(default_factory=dict)


def build_player_model(
    row, league_context: LeagueContext, config: LeagueConfig | None = None,
    weeks: float | None = None, status=None, expected_return_days=None,
) -> PlayerModel:
    """Assemble the Layer-0 PlayerModel for one pool row: posteriors (slice 1) + availability
    (slice 2) + display G-score (slice 3). Never raises."""
    cfg = config or LeagueConfig()
    posteriors = player_posteriors(row, cfg, weeks)
    avail = availability_survival(row, status=status, expected_return_days=expected_return_days,
                                  weeks_remaining=weeks)
    g = player_gscore(posteriors, league_context, cfg, detail=True)
    is_hit = _f(row.get("is_hitter") if hasattr(row, "get") else 1, default=1.0) >= 0.5
    name = row.get("name") if hasattr(row, "get") else ""
    return PlayerModel(
        player_id=int(_f(row.get("player_id") if hasattr(row, "get") else 0)),
        name=str(name) if name is not None else "",
        is_hitter=bool(is_hit), posteriors=posteriors, availability=avail,
        g_score=float(g["total"]), g_by_category=g["per_category"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_facade.py -q`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/model.py tests/test_player_model_facade.py
git commit -m "feat(player_model): PlayerModel + build_player_model facade (P1 slice4)"
```

---

### Task 3: `build_player_models` batch + `__init__` public API

A batch helper builds the league context once, then a `PlayerModel` per row (keyed by player_id). Optional `{player_id: status}` / `{player_id: return_days}` maps inject current injury info. `__init__.py` re-exports the public surface.

**Files:**
- Modify: `src/player_model/model.py`
- Modify: `src/player_model/__init__.py`
- Test: `tests/test_player_model_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_facade.py
def test_build_player_models_batch_keyed_by_id():
    from src.player_model.model import build_player_models
    cfg = LeagueConfig()
    pool = _pool()
    models = build_player_models(pool, cfg)
    assert len(models) == len(pool)
    assert all(pid == pm.player_id for pid, pm in models.items())
    # hitters get hitting cats, pitchers get pitching cats
    h = models[107]
    assert set(h.posteriors) == set(cfg.hitting_categories)
    p = models[205]
    assert set(p.posteriors) == set(cfg.pitching_categories)


def test_build_player_models_injects_status_map():
    from src.player_model.model import build_player_models
    cfg = LeagueConfig()
    pool = _pool()
    models = build_player_models(pool, cfg, status_map={103: "IL60"})
    assert models[103].availability.status == "IL60"
    assert models[104].availability.status == "ACTIVE"   # untouched


def test_public_api_reexports():
    import src.player_model as pmpkg
    for name in ("PlayerModel", "build_player_model", "build_player_models",
                 "build_league_context", "CategoryPosterior", "AvailabilitySurvival",
                 "LeagueContext"):
        assert hasattr(pmpkg, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_facade.py -k "batch or status_map or reexport" -q`
Expected: FAIL — `ImportError: cannot import name 'build_player_models'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/model.py
def build_player_models(
    pool: pd.DataFrame, config: LeagueConfig | None = None, weeks: float | None = None,
    status_map: dict | None = None, return_days_map: dict | None = None,
) -> dict[int, PlayerModel]:
    """Batch: build the league context once, then a PlayerModel per pool row (keyed by
    player_id). status_map/return_days_map are optional {player_id: value} overrides
    (default: active). Never raises; a row that fails is skipped."""
    cfg = config or LeagueConfig()
    status_map = status_map or {}
    return_days_map = return_days_map or {}
    ctx = build_league_context(pool, cfg, weeks)
    out: dict[int, PlayerModel] = {}
    if pool is None or len(pool) == 0:
        return out
    for i in range(len(pool)):
        row = pool.iloc[i]
        pid = int(_f(row.get("player_id")))
        try:
            out[pid] = build_player_model(
                row, ctx, cfg, weeks=weeks,
                status=status_map.get(pid), expected_return_days=return_days_map.get(pid),
            )
        except Exception:
            continue
    return out
```

```python
# src/player_model/__init__.py  — REPLACE the file contents with:
"""HEATER Layer-0 shared player model (Advanced Value Engine, Phase 1).

Produces, per player per scoring category, a posterior {mean, sigma2, tau2} with a
distributional margin descriptor, plus an availability survival distribution and a
display-only G-score. The single source of truth feeding every surface.
"""

from src.player_model.availability import AvailabilitySurvival, availability_survival, sample_active_weeks
from src.player_model.gscore import LeagueContext, category_gscore, player_gscore
from src.player_model.model import (
    PlayerModel,
    build_league_context,
    build_player_model,
    build_player_models,
)
from src.player_model.posterior import CategoryPosterior, category_posterior, player_posteriors

__all__ = [
    "AvailabilitySurvival",
    "CategoryPosterior",
    "LeagueContext",
    "PlayerModel",
    "availability_survival",
    "build_league_context",
    "build_player_model",
    "build_player_models",
    "category_gscore",
    "category_posterior",
    "player_gscore",
    "player_posteriors",
    "sample_active_weeks",
]
```

- [ ] **Step 4: Run test to verify it passes + lint**

Run: `python -m pytest tests/test_player_model_facade.py -q`
Expected: PASS (all tests)

Run: `python -m ruff check src/player_model/ tests/test_player_model_facade.py && python -m ruff format src/player_model/model.py src/player_model/__init__.py tests/test_player_model_facade.py`
Expected: no lint errors; format clean.

- [ ] **Step 5: Commit**

```bash
git add src/player_model/model.py src/player_model/__init__.py tests/test_player_model_facade.py
git commit -m "feat(player_model): build_player_models batch + public API re-exports (P1 slice4)"
```

---

### Task 4: Full package smoke + never-raise on a realistic sparse pool

The live pool has NaN/missing columns and mixed player types. Assert the whole package imports and `build_player_models` survives a sparse pool without raising, and the full `src/player_model/` test set is green together.

**Files:**
- Test: `tests/test_player_model_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_facade.py
def test_facade_never_raises_on_sparse_pool():
    from src.player_model.model import build_player_models
    cfg = LeagueConfig()
    sparse = pd.DataFrame([
        {"player_id": 1, "is_hitter": 1},                       # missing all stats
        {"player_id": 2, "is_hitter": 0, "k": np.nan, "ip": 150},
        {"player_id": 3},                                        # missing is_hitter
    ])
    models = build_player_models(sparse, cfg)
    assert len(models) >= 1
    for pm in models.values():
        assert math.isfinite(pm.g_score)
        assert pm.availability is not None


def test_full_player_model_suite_green_together():
    # Sanity: the per-cat posterior + availability + gscore + facade all coexist.
    from src.player_model import build_player_models, PlayerModel
    cfg = LeagueConfig()
    models = build_player_models(_pool(), cfg)
    assert all(isinstance(pm, PlayerModel) for pm in models.values())
```

- [ ] **Step 2: Run test to verify it fails (or passes if guards already hold)**

Run: `python -m pytest tests/test_player_model_facade.py -k "sparse_pool or suite_green" -q`
Expected: the `_f` guards + per-row try/except in `build_player_models` + the slice-1/2/3 never-raise contracts should already make these PASS. If a sparse row raised, the try/except in `build_player_models` skips it — confirm at least the valid rows produce models.

- [ ] **Step 3: Write minimal implementation (only if a test failed)**

```python
# Safety net: build_player_models already wraps each row in try/except and skips failures,
# and the slice-1/2/3 functions never raise. No change expected; this is a guard.
```

- [ ] **Step 4: Run the whole player_model test set + lint**

Run: `python -m pytest tests/test_player_model_posterior.py tests/test_player_model_availability.py tests/test_player_model_gscore.py tests/test_player_model_facade.py -q`
Expected: PASS (all four files together)

Run: `python -m ruff check src/player_model/`
Expected: no lint errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_player_model_facade.py
git commit -m "test(player_model): facade sparse-pool safety + package smoke (P1 slice4)"
```

---

## Self-Review

**Spec coverage (§6 module boundary + §5 single source of truth):**
- `(player_id, league_ctx) → {per-cat posterior, availability_survival, display_g_score}` → `PlayerModel{posteriors, availability, g_score, g_by_category}` via `build_player_model`. ✅
- The single object every surface consumes (no re-derivation) → `build_player_models` is the batch entry; `__init__` re-exports it as the package's public API. ✅
- League context the G-score standardizes against, at the posterior scale → `build_league_context` (Task 1). ✅
- **Deferred (named follow-ons):** wiring consumer surfaces (Phase 2+); Layer-1 win-prob (Phase 2); the slice-5 harness validation (the Phase-1 gate); a single shared margins/copula config object for Layer-1/2 — the `margin` descriptors already live on each `CategoryPosterior` (the de-facto shared parameterization), so Phase 2/3 read them directly from the `PlayerModel`.

**Placeholder scan:** Task 4 Step 3 is a conditional safety-net ("no change expected"), not a placeholder. Every other step has complete runnable code + exact commands.

**Type consistency:** `PlayerModel(player_id, name, is_hitter, posteriors, availability, g_score, g_by_category)`; `build_league_context(pool, config, weeks) -> LeagueContext`; `build_player_model(row, league_context, config, weeks, status, expected_return_days) -> PlayerModel`; `build_player_models(pool, config, weeks, status_map, return_days_map) -> dict[int, PlayerModel]` — used identically across tasks/tests. Consumes the locked slice APIs (`player_posteriors`, `availability_survival`, `player_gscore`, `LeagueContext`) with their exact signatures.

**Reuse (DRY):** composes slices 1–3, no re-implementation; `LeagueConfig` for taxonomy/STAT_MAP (no hardcoded lists); `_f` mirrors the slice-1/2/3 helper (future hoist candidate, noted).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase1-slice4-facade.md`.
