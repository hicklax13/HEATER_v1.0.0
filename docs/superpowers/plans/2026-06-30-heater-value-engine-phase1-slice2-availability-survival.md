# Phase 1 Slice 2 — Layer-0 Availability Survival (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Layer-0 availability survival layer — a per-player distribution over *games/IP remaining this season* that the season Monte-Carlo samples PER REPLICATE (not a single deterministic playing-time scalar), so counting-stat totals and the IL-stash/streaming policy react to simulated availability. This is the spec's single largest unmodeled rest-of-season value swing (research-validation gap G1) and has no academic prior — it is built in-house from HEATER's existing injury signals.

**Architecture:** A new focused module `src/player_model/availability.py` that COMPOSES — not reinvents — HEATER's injury machinery: chronic durability from the pool's `health_score` × `injury_model.age_risk_adjustment`, acute status from `il_manager.classify_il_type` / `estimate_il_duration` and `in_season._il_weight_from_status` / `_return_date_weight`. It produces an `AvailabilitySurvival` dataclass whose `sample_active_weeks(rng, n)` draws integer active-week realizations (Binomial after the current IL window) so the Layer-2 MC propagates availability uncertainty. Pure functions over a pool row (`pd.Series`) + optional status/return-date inputs; no DB / network; never raises.

**Tech Stack:** Python 3.12/3.14, NumPy, dataclasses, pandas, pytest. Reuses `src/injury_model.py`, `src/il_manager.py`, `src/in_season.py`, `src/valuation.py` (`LeagueConfig.season_weeks`).

**Spec:** `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` §3 (Layer 0 — "Availability survival distribution (NEW — frontier gap)… per-player games-/IP-remaining + current-injury hazard (age + position + injury-history + status), sampled per replicate. Built in-house from injury_model.py / il_manager.py / return-date curves") + research-validation gap G1.

**Scope note:** This slice delivers the per-player availability distribution + a sampler. It does NOT wire availability INTO a season sim (that is Phase 3) and does NOT model the streaming/FA replacement policy (Phase 3 G7). The current-status inputs (`status`, `expected_return_days`) are accepted as parameters; the unified facade (slice 4) sources them from `league_rosters` / news. Hazard/duration seeds are in-house first guesses, documented as calibratable against HEATER's own injury logs (a follow-on).

---

## File Structure

- **Create `src/player_model/availability.py`** — the `AvailabilitySurvival` dataclass + `chronic_availability()` + `availability_survival()` + `sample_active_weeks()`. One responsibility: turn a pool row + current status into a sample-able games/IP-remaining distribution.
- **Create `tests/test_player_model_availability.py`** — TDD tests for every behavior.

(`src/player_model/__init__.py` already exists from slice 1.)

---

### Task 1: `AvailabilitySurvival` dataclass + `chronic_availability`

Chronic (long-run) durability = the pool's `health_score` (0–1, already computed by `injury_model`) tempered by an age/position risk multiplier (`injury_model.age_risk_adjustment`, 0.5–1.0). This is the per-week probability the player is available ABSENT a current injury.

**Files:**
- Create: `src/player_model/availability.py`
- Test: `tests/test_player_model_availability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_model_availability.py
"""Tests for the Layer-0 availability survival layer (Phase 1 slice 2)."""

import math

import numpy as np
import pandas as pd
import pytest


def _row(**over):
    base = dict(
        player_id=1, name="P", is_hitter=1, age=28, positions="OF",
        health_score=0.90, ytd_pa=300, ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def test_chronic_availability_monotonic_in_health():
    from src.player_model.availability import chronic_availability
    lo = chronic_availability(health_score=0.60, age=28, is_pitcher=False, position="OF")
    hi = chronic_availability(health_score=0.95, age=28, is_pitcher=False, position="OF")
    assert hi > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_chronic_availability_older_pitcher_lower():
    from src.player_model.availability import chronic_availability
    young = chronic_availability(health_score=0.90, age=25, is_pitcher=True, position="SP")
    old = chronic_availability(health_score=0.90, age=37, is_pitcher=True, position="SP")
    assert old < young   # age/position risk drags an older SP down


def test_chronic_availability_nan_safe():
    from src.player_model.availability import chronic_availability
    v = chronic_availability(health_score=float("nan"), age=None, is_pitcher=False, position=None)
    assert math.isfinite(v) and 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_availability.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.player_model.availability'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/player_model/availability.py
"""Layer-0 availability survival — per-player games/IP-remaining distribution (gap G1).

The single largest unmodeled rest-of-season value swing, with no academic prior — built
in-house by COMPOSING HEATER's injury signals:
  * chronic durability: pool health_score x injury_model.age_risk_adjustment;
  * acute status: il_manager.classify_il_type / estimate_il_duration +
    in_season._il_weight_from_status / _return_date_weight.

Produces an AvailabilitySurvival whose sample_active_weeks(rng, n) draws integer active-week
realizations so the Layer-2 MC propagates availability uncertainty per replicate (rather than
applying a single deterministic playing-time scalar). Pure; no DB / network; never raises.

Hazard/duration seeds are in-house first guesses, calibratable against HEATER injury logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.injury_model import age_risk_adjustment


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass(frozen=True)
class AvailabilitySurvival:
    """Per-player rest-of-season availability. `chronic_avail` is the per-week available
    probability absent a current injury; `status_weight` is the near-term availability from
    the current status/return-date; `il_weeks_out` is the expected weeks fully sidelined now;
    `expected_active_weeks` = E[active weeks over `weeks_remaining`]; `weekly_hazard` is the
    per-week probability of a NEW availability loss. `sample_active_weeks` draws realizations."""

    player_id: int
    is_hitter: bool
    weeks_remaining: float
    status: str
    status_weight: float
    chronic_avail: float
    il_weeks_out: float
    expected_active_weeks: float
    expected_active_fraction: float
    weekly_hazard: float


def chronic_availability(health_score, age, is_pitcher: bool, position) -> float:
    """Long-run per-week availability absent a current injury: pool health_score tempered
    by age/position risk. Clamped to [0, 1]. NaN-safe (missing health -> 0.85 league avg)."""
    hs = _f(health_score, default=0.85)
    hs = min(max(hs, 0.0), 1.0)
    pos = position if isinstance(position, str) else ""
    try:
        risk = age_risk_adjustment(int(_f(age, default=28)), bool(is_pitcher), pos or None)
    except Exception:
        risk = 1.0
    risk = min(max(_f(risk, default=1.0), 0.0), 1.0)
    return float(min(max(hs * risk, 0.0), 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_availability.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/availability.py tests/test_player_model_availability.py
git commit -m "feat(player_model): AvailabilitySurvival dataclass + chronic_availability (P1 slice2)"
```

---

### Task 2: `availability_survival` orchestrator — fold status + return-date + IL duration

Combine chronic durability with the current acute status. An active player has `status_weight=1`, `il_weeks_out=0`. An IL player is fully OUT for `estimate_il_duration(il_type)` weeks, then returns at the chronic rate; a return-date (if known) overrides via the `_return_date_weight` curve. `expected_active_weeks = max(0, weeks_remaining − il_weeks_out) × chronic_avail`.

**Files:**
- Modify: `src/player_model/availability.py`
- Test: `tests/test_player_model_availability.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_availability.py
def test_active_player_full_availability():
    from src.player_model.availability import availability_survival
    s = availability_survival(_row(health_score=1.0, age=27), status=None, weeks_remaining=20)
    assert s.status == "ACTIVE"
    assert s.status_weight == pytest.approx(1.0)
    assert s.il_weeks_out == pytest.approx(0.0)
    assert s.expected_active_fraction == pytest.approx(s.chronic_avail)


def test_il60_player_mostly_out():
    from src.player_model.availability import availability_survival
    healthy = availability_survival(_row(), status=None, weeks_remaining=20)
    il60 = availability_survival(_row(), status="IL60", weeks_remaining=20)
    assert il60.il_weeks_out >= 8.0           # estimate_il_duration("IL60") ~ 10 weeks
    assert il60.expected_active_weeks < healthy.expected_active_weeks
    assert 0.0 <= il60.expected_active_fraction <= 1.0


def test_return_date_overrides_status_curve():
    from src.player_model.availability import availability_survival
    # 14-days-to-return -> _return_date_weight ~ 0.70 near-term weight (more specific than raw IL10).
    s = availability_survival(_row(), status="IL10", expected_return_days=14, weeks_remaining=20)
    assert s.status_weight == pytest.approx(0.70, abs=0.05)


def test_expected_active_weeks_bounded():
    from src.player_model.availability import availability_survival
    s = availability_survival(_row(health_score=0.8), status="IL15", weeks_remaining=12)
    assert 0.0 <= s.expected_active_weeks <= 12.0
    assert 0.0 <= s.weekly_hazard <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_availability.py -k "active or il60 or return_date or bounded" -q`
Expected: FAIL — `ImportError: cannot import name 'availability_survival'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/availability.py
from src.il_manager import classify_il_type, estimate_il_duration
from src.in_season import _il_weight_from_status, _return_date_weight

# Per-week probability of a NEW availability loss, seeded from chronic durability:
# a fully-healthy player (chronic=1) carries a small residual hazard; a fragile player more.
_BASE_WEEKLY_HAZARD: float = 0.02   # calibratable against HEATER injury logs (follow-on)


def _normalize_status(status) -> tuple[str, str | None]:
    """Return (normalized_status, il_type). il_type is None when the player is active.
    Normalized status is one of ACTIVE/DTD/IL10/IL15/IL60."""
    if status is None or (isinstance(status, float) and math.isnan(status)):
        return "ACTIVE", None
    raw = str(status).strip()
    if not raw:
        return "ACTIVE", None
    il_type = classify_il_type(raw)            # -> IL10/IL15/IL60/DTD or None
    if il_type is None:
        return "ACTIVE", None
    if il_type == "DTD":
        return "DTD", None
    return il_type, il_type


def _is_hitter(row) -> bool:
    raw = row.get("is_hitter") if hasattr(row, "get") else row["is_hitter"]
    return _f(raw, default=1.0) >= 0.5


def availability_survival(row, status=None, expected_return_days=None,
                          weeks_remaining: float | None = None) -> AvailabilitySurvival:
    """Build the rest-of-season availability distribution for one pool row + current status.
    `weeks_remaining` defaults to LeagueConfig.season_weeks. Never raises."""
    from src.valuation import LeagueConfig

    weeks = _f(weeks_remaining) if weeks_remaining else float(LeagueConfig().season_weeks)
    weeks = max(0.0, weeks)
    is_hit = _is_hitter(row)
    position = row.get("positions") if hasattr(row, "get") else None
    chronic = chronic_availability(
        row.get("health_score") if hasattr(row, "get") else None,
        row.get("age") if hasattr(row, "get") else None,
        is_pitcher=not is_hit, position=position,
    )

    norm_status, il_type = _normalize_status(status)

    # Near-term status weight: the return-date curve wins when known, else status default.
    rd = _return_date_weight(expected_return_days) if expected_return_days is not None else None
    status_weight = rd if rd is not None else _il_weight_from_status(norm_status, expected_return_days)
    status_weight = min(max(_f(status_weight, default=1.0), 0.0), 1.0)

    # Expected weeks fully sidelined now (IL only; DTD/active -> 0).
    pos_str = position if isinstance(position, str) else ""
    il_weeks_out = estimate_il_duration(il_type, pos_str) if il_type in {"IL10", "IL15", "IL60"} else 0.0
    il_weeks_out = min(max(_f(il_weeks_out), 0.0), weeks)

    active_window = max(0.0, weeks - il_weeks_out)
    expected_active_weeks = active_window * chronic
    expected_active_fraction = (expected_active_weeks / weeks) if weeks > 0 else 0.0
    weekly_hazard = min(max(_BASE_WEEKLY_HAZARD * (1.0 - chronic) / max(1.0 - 0.85, 1e-6), 0.0), 1.0)

    return AvailabilitySurvival(
        player_id=int(_f(row.get("player_id") if hasattr(row, "get") else 0)),
        is_hitter=is_hit, weeks_remaining=weeks, status=norm_status,
        status_weight=status_weight, chronic_avail=chronic, il_weeks_out=il_weeks_out,
        expected_active_weeks=expected_active_weeks,
        expected_active_fraction=expected_active_fraction, weekly_hazard=weekly_hazard,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_availability.py -q`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/availability.py tests/test_player_model_availability.py
git commit -m "feat(player_model): availability_survival orchestrator (status+return-date+IL) (P1 slice2)"
```

---

### Task 3: `sample_active_weeks` — draw availability realizations for the MC

The Layer-2 MC must SAMPLE availability per replicate (gap G1 — not a deterministic scalar). After the current IL window, each remaining week is active independently with probability `chronic_avail` (Binomial). The mean of the draws matches `expected_active_weeks`.

**Files:**
- Modify: `src/player_model/availability.py`
- Test: `tests/test_player_model_availability.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_availability.py
def test_sample_active_weeks_mean_matches_expectation():
    from src.player_model.availability import availability_survival, sample_active_weeks
    s = availability_survival(_row(health_score=0.9, age=27), status=None, weeks_remaining=20)
    rng = np.random.default_rng(0)
    draws = sample_active_weeks(s, rng, n_samples=5000)
    assert draws.shape == (5000,)
    assert draws.min() >= 0 and draws.max() <= 20
    assert abs(draws.mean() - s.expected_active_weeks) < 0.5   # Monte-Carlo mean ~ expectation


def test_sample_active_weeks_reproducible():
    from src.player_model.availability import availability_survival, sample_active_weeks
    s = availability_survival(_row(), status="IL15", weeks_remaining=15)
    a = sample_active_weeks(s, np.random.default_rng(7), n_samples=100)
    b = sample_active_weeks(s, np.random.default_rng(7), n_samples=100)
    assert np.array_equal(a, b)


def test_sample_active_weeks_zero_when_out_all_season():
    from src.player_model.availability import availability_survival, sample_active_weeks
    s = availability_survival(_row(), status="IL60", weeks_remaining=4)  # IL ~10wk > 4wk left
    draws = sample_active_weeks(s, np.random.default_rng(1), n_samples=50)
    assert draws.max() == 0   # sidelined the whole remaining window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_availability.py -k sample -q`
Expected: FAIL — `ImportError: cannot import name 'sample_active_weeks'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/availability.py
def sample_active_weeks(survival: AvailabilitySurvival, rng, n_samples: int = 1) -> np.ndarray:
    """Draw `n_samples` integer active-week realizations for the season MC. The current IL
    window is fully out; each remaining week is active ~ Bernoulli(chronic_avail). Returns a
    length-n_samples int array in [0, floor(active_window)]. Deterministic given `rng`."""
    weeks = max(0.0, survival.weeks_remaining)
    active_window = int(math.floor(max(0.0, weeks - survival.il_weeks_out)))
    n = max(1, int(n_samples))
    if active_window <= 0:
        return np.zeros(n, dtype=int)
    p = min(max(survival.chronic_avail, 0.0), 1.0)
    return rng.binomial(active_window, p, size=n).astype(int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_availability.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/availability.py tests/test_player_model_availability.py
git commit -m "feat(player_model): sample_active_weeks MC availability sampler (P1 slice2)"
```

---

### Task 4: Robustness — never-raise on degenerate rows + lint

The live pool has NaN/missing `health_score`, `age`, `positions`, and `status` for partial players. The layer must degrade gracefully (never raise), and the module must pass ruff.

**Files:**
- Modify: `src/player_model/availability.py` (only if a guard is missing)
- Test: `tests/test_player_model_availability.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_availability.py
def test_never_raises_on_sparse_row():
    from src.player_model.availability import availability_survival, sample_active_weeks
    sparse = pd.Series({"player_id": 9})   # no is_hitter/health/age/positions
    s = availability_survival(sparse, status=None, weeks_remaining=None)
    assert math.isfinite(s.chronic_avail) and 0.0 <= s.chronic_avail <= 1.0
    assert math.isfinite(s.expected_active_weeks)
    draws = sample_active_weeks(s, np.random.default_rng(0), n_samples=10)
    assert draws.shape == (10,)


def test_garbage_status_treated_active():
    from src.player_model.availability import availability_survival
    s = availability_survival(_row(), status="not-a-real-status", weeks_remaining=20)
    assert s.status == "ACTIVE"          # unrecognized -> classify_il_type None -> active
    assert s.il_weeks_out == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails (or passes if guards already hold)**

Run: `python -m pytest tests/test_player_model_availability.py -k "sparse or garbage" -q`
Expected: the `_f` guards + `.get` fallbacks + the `_normalize_status` None path should already make these PASS. If `test_never_raises_on_sparse_row` fails because a missing `is_hitter` raised, harden `_is_hitter`/`.get` calls in Step 3.

- [ ] **Step 3: Write minimal implementation (only if a test failed)**

```python
# Safety net: ensure every row access uses .get with a default (pd.Series.get returns None
# for absent keys, which _f coerces to its default). No change expected — _is_hitter,
# chronic_availability, and availability_survival already route through _f(row.get(...)).
```

- [ ] **Step 4: Run test to verify it passes + lint**

Run: `python -m pytest tests/test_player_model_availability.py -q`
Expected: PASS (all tests)

Run: `python -m ruff check src/player_model/availability.py tests/test_player_model_availability.py && python -m ruff format src/player_model/availability.py tests/test_player_model_availability.py`
Expected: no lint errors; format clean.

- [ ] **Step 5: Commit**

```bash
git add src/player_model/availability.py tests/test_player_model_availability.py
git commit -m "test(player_model): availability degenerate-row safety + lint (P1 slice2)"
```

---

## Self-Review

**Spec coverage (§3 Layer 0 availability + gap G1):**
- "per-player games-/IP-remaining + current-injury hazard (age + position + injury-history + status)" → `chronic_availability` (health_score=injury-history + age/position) × acute status (`availability_survival`); `weekly_hazard` field. ✅
- "sampled per replicate" (NOT a deterministic scalar) → `sample_active_weeks` draws Binomial realizations (Task 3). ✅
- "Built in-house from injury_model.py / il_manager.py / return-date curves" → composes `age_risk_adjustment`, `classify_il_type`, `estimate_il_duration`, `_il_weight_from_status`, `_return_date_weight` (no reinvention). ✅
- **Deferred (named follow-ons, out of this slice):** wiring availability into the season sim + the streaming/FA policy (Phase 3, gap G7); hazard/duration calibration against HEATER injury logs.

**Placeholder scan:** Task 4 Step 3 is a conditional safety-net (explicitly "no change expected"), not a placeholder. Every other step has complete runnable code + exact commands.

**Type consistency:** `AvailabilitySurvival(player_id, is_hitter, weeks_remaining, status, status_weight, chronic_avail, il_weeks_out, expected_active_weeks, expected_active_fraction, weekly_hazard)` used identically across Tasks 1–4. `chronic_availability(health_score, age, is_pitcher, position)`, `availability_survival(row, status, expected_return_days, weeks_remaining)`, `sample_active_weeks(survival, rng, n_samples)` match every call site. `_normalize_status -> (status, il_type)` consumed in `availability_survival`.

**Reuse (DRY):** `injury_model.age_risk_adjustment`, the pool's `health_score`, `il_manager.classify_il_type`/`estimate_il_duration`, `in_season._il_weight_from_status`/`_return_date_weight` — all composed, none duplicated. `_f` is a small module-private NaN-safe coercion mirroring slice 1's helper (a future cleanup may hoist a shared `player_model` util if it proliferates — noted, not blocking).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase1-slice2-availability-survival.md`.
