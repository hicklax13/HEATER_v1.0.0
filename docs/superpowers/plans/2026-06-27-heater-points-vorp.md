# Points VORP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone `src/points_vorp.py` that computes per-position replacement-level points and points value-over-replacement for points leagues, ranking players by scarcity-adjusted value — mirroring HEATER's category VORP but over points.

**Architecture:** Pure functions mirroring `src.valuation.compute_replacement_levels`/`compute_vorp`, but scoring with slice-1 `src.points_scoring.project_player_points` instead of SGP. Reuses the format-agnostic `src.valuation.compute_positional_scarcity_factor`. Imports NO category scoring (no `SGPCalculator`). `LeagueConfig` is used read-only for roster structure (`num_teams`, `roster_slots`, `hitter_starters_at`, `pitcher_starters`). Never raises; NaN-safe via slice-1's `_num`.

**Tech Stack:** Python 3.12/3.14, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-06-27-heater-points-vorp-design.md`. **Depends on:** slice 1 (`src/points_scoring.py`, present on master) and `src/valuation.py` (`compute_positional_scarcity_factor`, `LeagueConfig`). **Mirror reference:** `src/valuation.py:648` `compute_replacement_levels` + `:765` `compute_vorp`.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/points_vorp.py` | replacement levels, points VORP, scarcity multiplier, VORP ranking | Create |
| `tests/test_points_vorp.py` | TDD coverage | Create |

No existing file is modified. (`points_scoring`, `compute_positional_scarcity_factor`, `LeagueConfig` imported read-only.)

---

## Task 1: Per-position replacement levels

**Files:**
- Create: `src/points_vorp.py`
- Test: `tests/test_points_vorp.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_points_vorp.py`:
```python
import pandas as pd

from src.points_scoring import PointsScoringConfig
from src.points_vorp import compute_points_replacement_levels
from src.valuation import LeagueConfig

# Points come only from HR (hitters) / K (pitchers) so test arithmetic is obvious.
_CFG = PointsScoringConfig(hitter_weights={"HR": 1.0}, pitcher_weights={"K": 1.0})
# Tiny league so n_starters is small: 2 teams x 1 starter per slot.
_LC = LeagueConfig(
    num_teams=2,
    roster_slots={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 1,
                  "Util": 1, "SP": 1, "RP": 1, "P": 0, "BN": 0, "IL": 0},
)


def _h(pid, pos, hr):
    return {"player_id": pid, "is_hitter": 1, "positions": pos, "hr": hr, "ip": 0}


def _p(pid, pos, k):
    return {"player_id": pid, "is_hitter": 0, "positions": pos, "k": k, "ip": 150}


def _pool():
    return pd.DataFrame([
        _h(1, "C", 100), _h(2, "C", 50), _h(3, "C", 10),
        _h(4, "OF", 100), _h(5, "OF", 90), _h(6, "OF", 80), _h(7, "OF", 70),
        _p(8, "SP", 200), _p(9, "SP", 150), _p(10, "SP", 100),
    ])


def test_replacement_is_first_non_starter_by_points():
    repl = compute_points_replacement_levels(_pool(), _CFG, _LC)
    assert repl["C"] == 10.0    # 3 catchers, 2 starters -> iloc[2] = 10
    assert repl["OF"] == 80.0   # 4 OF, 2 starters -> iloc[2] = 80
    assert repl["SP"] == 100.0  # 3 SP, 2 starters -> iloc[2] = 100


def test_shallow_position_uses_half_last_fallback():
    pool = pd.DataFrame([_h(1, "C", 40), _h(4, "OF", 100), _h(5, "OF", 90), _h(6, "OF", 80)])
    repl = compute_points_replacement_levels(pool, _CFG, _LC)
    assert repl["C"] == 20.0   # 1 catcher, 2 starters needed -> 40 * 0.5


def test_util_is_min_hitting_replacement():
    repl = compute_points_replacement_levels(_pool(), _CFG, _LC)
    assert repl["Util"] == min(repl[p] for p in ["C", "1B", "2B", "3B", "SS", "OF"])


def test_empty_pool_all_zero():
    repl = compute_points_replacement_levels(pd.DataFrame(), _CFG, _LC)
    assert repl["C"] == 0.0 and repl["SP"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.points_vorp'`.

- [ ] **Step 3: Write the implementation**

`src/points_vorp.py`:
```python
"""Points-league value over replacement (Phase 4 slice 3).

Standalone + additive: per-position replacement baselines and points-VORP for
points leagues, mirroring the category VORP (src.valuation.compute_replacement_levels
/ compute_vorp) but over points. Reuses slice-1 scoring (src.points_scoring) and
the format-agnostic src.valuation.compute_positional_scarcity_factor. Imports NO
category scoring (no SGP). LeagueConfig is used read-only for roster structure.
Never raises; NaN-safe via slice-1's _num.
"""

from __future__ import annotations

import pandas as pd

from src.points_scoring import STANDARD_POINTS, PointsScoringConfig, project_player_points
from src.valuation import LeagueConfig, compute_positional_scarcity_factor

_HITTING_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF"]
_SCARCE_POSITIONS = {"C", "SS", "2B"}


def _get(row, key):
    try:
        return row.get(key)
    except AttributeError:
        return row[key] if key in row else None


def _eligible(pool: pd.DataFrame, pos: str, is_hitter: bool) -> pd.DataFrame:
    pos_match = pool["positions"].apply(lambda x: pos in [p.strip() for p in str(x).split(",")])
    type_match = pd.to_numeric(pool.get("is_hitter"), errors="coerce").fillna(0) == (1 if is_hitter else 0)
    return pool[pos_match & type_match].copy()


def _replacement_for(pool, pos, is_hitter, n_starters, points_config):
    elig = _eligible(pool, pos, is_hitter)
    if elig.empty:
        return 0.0
    elig["_pts"] = [project_player_points(r, points_config).points for _, r in elig.iterrows()]
    elig = elig.sort_values("_pts", ascending=False).reset_index(drop=True)
    if len(elig) > n_starters:
        return float(elig.iloc[n_starters]["_pts"])
    return float(elig.iloc[-1]["_pts"]) * 0.5


def compute_points_replacement_levels(
    pool: pd.DataFrame,
    points_config: PointsScoringConfig | None = None,
    league_config: LeagueConfig | None = None,
) -> dict:
    """Replacement-level POINTS per position (mirror of compute_replacement_levels).
    For each position: the points of the first non-starter (index n_starters) by
    points, or last*0.5 when the eligible pool is shallower than n_starters.
    Util = the deepest (min) hitting replacement."""
    points_config = points_config or STANDARD_POINTS
    league_config = league_config or LeagueConfig()
    if pool is None or len(pool) == 0:
        return {p: 0.0 for p in _HITTING_POSITIONS + ["Util", "SP", "RP"]}

    replacement = {}
    for pos in _HITTING_POSITIONS:
        replacement[pos] = _replacement_for(
            pool, pos, True, league_config.hitter_starters_at(pos), points_config
        )
    pcounts = league_config.pitcher_starters()
    for pos in ["SP", "RP"]:
        replacement[pos] = _replacement_for(pool, pos, False, pcounts[pos], points_config)
    replacement["Util"] = min(replacement.get(p, 0.0) for p in _HITTING_POSITIONS)
    return replacement
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/points_vorp.py tests/test_points_vorp.py
python -m ruff check src/points_vorp.py tests/test_points_vorp.py
git add src/points_vorp.py tests/test_points_vorp.py
git commit -m "feat(points-vorp): per-position replacement levels (Phase 4 slice 3)"
```

---

## Task 2: points_vorp + scarcity multiplier

**Files:**
- Modify: `src/points_vorp.py`
- Test: `tests/test_points_vorp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_points_vorp.py`:
```python
from src.points_vorp import points_vorp, scarcity_multiplier  # noqa: E402
from src.valuation import compute_positional_scarcity_factor  # noqa: E402


def test_scarce_position_player_outranks_deep_position_equal_points():
    repl = compute_points_replacement_levels(_pool(), _CFG, _LC)  # C=10, OF=80
    catcher = _h(1, "C", 100)      # vorp = 100 - 10 = 90
    outfielder = _h(4, "OF", 100)  # vorp = 100 - 80 = 20
    assert points_vorp(catcher, _CFG, repl) > points_vorp(outfielder, _CFG, repl)


def test_points_vorp_is_points_minus_best_eligible_replacement():
    repl = {"C": 10.0, "OF": 80.0}
    assert points_vorp(_h(1, "C", 100), _CFG, repl) == 90.0


def test_multi_position_flexibility_bonus():
    repl = {"C": 10.0, "SS": 5.0}
    flex = _h(1, "C,SS", 50)   # base 50 - max(10,5)=10 = 40, + bonus (2 eligible, 2 scarce)
    single = _h(2, "C", 50)    # base 40, no bonus
    assert points_vorp(flex, _CFG, repl) > points_vorp(single, _CFG, repl)


def test_no_valid_position_vorp_equals_points():
    repl = {"C": 10.0}
    assert points_vorp(_h(1, "Util", 50), _CFG, repl) == 50.0  # "Util" not in repl


def test_nan_points_safe():
    assert points_vorp(_h(1, "C", float("nan")), _CFG, {"C": 10.0}) == -10.0  # 0 - 10


def test_scarcity_multiplier_reuses_valuation_util():
    repl = {"C": 5.0, "OF": 80.0, "1B": 70.0, "SS": 10.0}
    p = _h(1, "C", 50)
    assert scarcity_multiplier(p, repl) == compute_positional_scarcity_factor("C", repl)
    assert scarcity_multiplier(p, repl) >= 1.0  # scarce C -> boost >= neutral
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: FAIL — `ImportError` (`points_vorp`/`scarcity_multiplier` not defined).

- [ ] **Step 3: Append the implementation**

Append to `src/points_vorp.py`:
```python
def points_vorp(
    player_row,
    points_config: PointsScoringConfig | None = None,
    replacement_levels: dict | None = None,
) -> float:
    """Points value over replacement (mirror of compute_vorp): player points minus
    the BEST (max) replacement among eligible positions, plus a multi-position
    flexibility bonus. best_repl = 0 when no eligible position is in replacement_levels."""
    points_config = points_config or STANDARD_POINTS
    replacement_levels = replacement_levels or {}
    pts = project_player_points(player_row, points_config).points
    positions = [p.strip() for p in str(_get(player_row, "positions") or "Util").split(",") if p.strip()]
    valid = [p for p in positions if p in replacement_levels]
    best_repl = max((replacement_levels[p] for p in valid), default=0.0)
    vorp = pts - best_repl
    if len(valid) > 1:
        scarce_count = sum(1 for p in valid if p in _SCARCE_POSITIONS)
        vorp += 0.12 * (len(valid) - 1) + 0.08 * scarce_count
    return vorp


def scarcity_multiplier(player_row, replacement_levels: dict) -> float:
    """Format-agnostic positional-scarcity multiplier, reusing
    src.valuation.compute_positional_scarcity_factor. Kept separate from
    points_vorp (mirrors the category side); callers may multiply a VORP by it
    for a scarcity-weighted value."""
    positions = str(_get(player_row, "positions") or "")
    return float(compute_positional_scarcity_factor(positions, replacement_levels))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: PASS (all tests so far). Do NOT change the test arithmetic; if a test fails, fix the implementation to mirror `compute_vorp` (best = max replacement among eligible; bonus 0.12·(n−1)+0.08·scarce).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/points_vorp.py tests/test_points_vorp.py
python -m ruff check src/points_vorp.py tests/test_points_vorp.py
git add src/points_vorp.py tests/test_points_vorp.py
git commit -m "feat(points-vorp): points VORP + reusable scarcity multiplier (Phase 4 slice 3)"
```

---

## Task 3: Pool ranking by points VORP

**Files:**
- Modify: `src/points_vorp.py`
- Test: `tests/test_points_vorp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_points_vorp.py`:
```python
from src.points_vorp import rank_players_by_points_vorp  # noqa: E402


def test_rank_by_points_vorp_sorts_and_adds_columns_without_mutation():
    pool = _pool()
    before = pool.copy(deep=True)
    ranked = rank_players_by_points_vorp(pool, _CFG, _LC)
    assert "points" in ranked.columns and "points_vorp" in ranked.columns
    assert "points_vorp" not in pool.columns
    pd.testing.assert_frame_equal(pool, before)
    order = list(ranked["player_id"])
    assert order.index(1) < order.index(4)  # 100-pt catcher (vorp 90) before 100-pt OF (vorp 20)


def test_rank_empty_pool_is_safe():
    ranked = rank_players_by_points_vorp(pd.DataFrame(), _CFG, _LC)
    assert list(ranked["points_vorp"]) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: FAIL — `ImportError` (`rank_players_by_points_vorp` not defined).

- [ ] **Step 3: Append the implementation**

Append to `src/points_vorp.py`:
```python
def rank_players_by_points_vorp(
    pool: pd.DataFrame,
    points_config: PointsScoringConfig | None = None,
    league_config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Return a copy of `pool` with `points` and `points_vorp` columns, sorted by
    points_vorp descending. Never mutates the input."""
    points_config = points_config or STANDARD_POINTS
    league_config = league_config or LeagueConfig()
    out = pool.copy() if pool is not None else pd.DataFrame()
    if len(out) == 0:
        out["points"] = []
        out["points_vorp"] = []
        return out
    repl = compute_points_replacement_levels(out, points_config, league_config)
    out["points"] = [project_player_points(r, points_config).points for _, r in out.iterrows()]
    out["points_vorp"] = [points_vorp(r, points_config, repl) for _, r in out.iterrows()]
    return out.sort_values("points_vorp", ascending=False, kind="mergesort").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_vorp.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/points_vorp.py tests/test_points_vorp.py
python -m ruff check src/points_vorp.py tests/test_points_vorp.py
git add src/points_vorp.py tests/test_points_vorp.py
git commit -m "feat(points-vorp): rank pool by points VORP (Phase 4 slice 3)"
```

---

## Task 4: Registry row + verification + push

**Files:**
- Modify: `docs/launch/evidence_registry.yaml`

- [ ] **Step 1: Add the registry row**

In `docs/launch/evidence_registry.yaml`, insert immediately AFTER the `P4-ROTO-ENGINE` row (or after `P4-POINTS-ENGINE` if Roto slice not yet landed) and BEFORE `P4-FORMAT-GENERALIZATION`:
```yaml
  - id: P4-POINTS-VORP
    category: analytical_value
    phase: 4
    description: Standalone points-league value over replacement (per-position replacement points, points VORP, reusable positional-scarcity multiplier, VORP ranking) reusing slice-1 scoring; no category-scoring dependency.
    status: passing
    subsystem: engine
    verify: "python -m pytest tests/test_points_vorp.py -q"
    metric: "scarce-position players boosted; VORP = points - best replacement; category engine untouched"
    evidence: "src/points_vorp.py, tests/test_points_vorp.py, docs/superpowers/specs/2026-06-27-heater-points-vorp-design.md"
    external_review: none
    blocking_ring: invite_beta
    last_verified: "2026-06-27"
    score_contribution: analytical_value
```

- [ ] **Step 2: Validate registry + run VORP suite + structural guards**

Run:
```bash
python -m scripts.launch.evidence_registry --summary
python -m pytest tests/test_points_vorp.py tests/test_points_scoring.py -q
python -m pytest tests/ -k "no_hardcoded_categories or no_lc_singletons or no_direct_sqlite" --ignore=tests/test_cheat_sheet.py -q
```
Expected: registry valid (P4-POINTS-VORP passing); points-VORP + slice-1 points tests green; structural guards green. The module hardcodes no category lists (only roster-position labels + the shared scarcity util) and creates no module-level singleton.

- [ ] **Step 3: Real-pool smoke check (manual, non-test — DB may be empty in CI/worktrees)**

Run:
```bash
python -c "
from src.database import load_player_pool
from src.points_vorp import rank_players_by_points_vorp
from src.points_scoring import STANDARD_POINTS
from src.valuation import LeagueConfig
pool = load_player_pool()
ranked = rank_players_by_points_vorp(pool, STANDARD_POINTS, LeagueConfig())
cols = [c for c in ['name','positions','points','points_vorp'] if c in ranked.columns] or ['player_id','points','points_vorp']
print('top 10 by points VORP:')
print(ranked[cols].head(10).to_string())
"
```
Expected: scarce-position players (C/SS) rank higher relative to raw points than they would by `rank_players_by_points`. Sanity check only (needs the local DB).

- [ ] **Step 4: Commit + push**

```bash
git add docs/launch/evidence_registry.yaml
git commit -m "docs(launch): register P4-POINTS-VORP (passing) (Phase 4 slice 3)"
git pull --no-rebase --no-edit origin master
git push origin master
```
Expected: pre-push structural suite passes; push succeeds. Never `--no-verify`. (Reconcile the parallel CMO track via `git pull --no-rebase` — do not disturb their branch.)

---

## Self-review notes

- **Spec coverage:** `compute_points_replacement_levels` (per-position, n-th non-starter, last*0.5 fallback, Util=min) → Task 1; `points_vorp` (points − best replacement + flexibility bonus) + `scarcity_multiplier` (reuse `compute_positional_scarcity_factor`) → Task 2; `rank_players_by_points_vorp` → Task 3; registry + verification → Task 4. Out-of-scope items (optimizer/FA/trade wiring, unified cross-format VORP, connector-sourced roster structure, extra-stat projections) are correctly absent.
- **No placeholders:** every step has exact code/commands; the LeagueConfig roster-structure reuse is concrete (`hitter_starters_at`/`pitcher_starters`).
- **Type/name consistency:** `compute_points_replacement_levels`, `points_vorp`, `scarcity_multiplier`, `rank_players_by_points_vorp`, `_eligible`, `_replacement_for`, `_get`, `_HITTING_POSITIONS`/`_SCARCE_POSITIONS` used identically across tasks and tests. Columns `points`/`points_vorp` and the `points_config`/`league_config`/`replacement_levels` parameter names consistent throughout. Mirrors `compute_vorp`'s `best = max(replacement)` + `0.12·(n−1)+0.08·scarce` bonus exactly.
```
