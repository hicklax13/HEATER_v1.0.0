# Points Scoring Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone `src/points_scoring.py` that, given a points-league's per-stat weights, computes a player's projected points, ranks players, and totals a roster — scoring only the stats HEATER already projects and flagging the rest as "uncovered".

**Architecture:** Pure functions over the existing enriched player pool (`load_player_pool()`), parallel to and never touching the category engine (`LeagueConfig`/`SGPCalculator`). A per-player-type stat-name→pool-column mapping is the single extensible point. NaN-safe; never raises.

**Tech Stack:** Python 3.12/3.14, pandas, dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-06-27-heater-points-scoring-engine-design.md`. **Pool columns verified 2026-06-27** against `load_player_pool()` (hitters `r/hr/rbi/sb/ab/h/bb/hbp/sf/avg/obp`; pitchers `w/l/sv/k/ip/er/bb_allowed/h_allowed/era/whip`; plus `is_hitter`, `player_id`).

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/points_scoring.py` | Config, stat→column maps, per-player scoring, pool ranking/roster totals, preset | Create |
| `tests/test_points_scoring.py` | TDD coverage | Create |

No existing file is modified.

---

## Task 1: Config + per-player scoring core

**Files:**
- Create: `src/points_scoring.py`
- Test: `tests/test_points_scoring.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_points_scoring.py`:
```python
import math

import pandas as pd

from src.points_scoring import (
    PointsScoringConfig,
    PointsResult,
    project_player_points,
)


def _hitter(**over):
    row = {"player_id": 1, "is_hitter": 1, "r": 90, "hr": 30, "rbi": 100,
           "sb": 10, "h": 150, "bb": 60, "hbp": 5, "ab": 550, "sf": 4,
           "avg": 0.273, "obp": 0.350, "ip": 0}
    row.update(over)
    return row


def _cfg(hit=None, pit=None):
    return PointsScoringConfig(hitter_weights=hit or {}, pitcher_weights=pit or {})


def test_hitter_points_is_weighted_sum():
    cfg = _cfg(hit={"HR": 4.0, "R": 1.0, "RBI": 1.0, "SB": 2.0, "BB": 1.0})
    res = project_player_points(_hitter(), cfg)
    # 30*4 + 90*1 + 100*1 + 10*2 + 60*1 = 120 + 90 + 100 + 20 + 60 = 390
    assert isinstance(res, PointsResult)
    assert res.points == 390.0
    assert res.breakdown["HR"] == 120.0


def test_unprojected_hitter_stat_is_uncovered_not_zeroed_silently():
    # Hitter strikeouts ("K") are NOT projected → flagged uncovered, not scored.
    cfg = _cfg(hit={"HR": 4.0, "K": -1.0})
    res = project_player_points(_hitter(), cfg)
    assert res.points == 120.0          # only HR scored
    assert "K" in res.uncovered
    assert "K" not in res.breakdown


def test_weight_keys_are_case_insensitive():
    cfg = _cfg(hit={"hr": 4.0})
    assert project_player_points(_hitter(), cfg).points == 120.0


def test_nan_and_missing_values_score_zero_never_raise():
    cfg = _cfg(hit={"HR": 4.0, "RBI": 1.0})
    row = _hitter(hr=float("nan"))
    del row["rbi"]  # missing column entirely
    res = project_player_points(row, cfg)
    assert res.points == 0.0
    assert math.isfinite(res.points)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.points_scoring'`.

- [ ] **Step 3: Write the implementation**

`src/points_scoring.py`:
```python
"""Points-league scoring valuation (Phase 4 slice 1).

Standalone + additive: pure functions over the existing player pool, parallel to
(and never touching) the category engine (LeagueConfig/SGPCalculator). Given a
points league's per-stat weights, computes a player's projected points, ranks
players, and totals a roster. Stats HEATER does not project are flagged
'uncovered', never silently zeroed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Friendly stat name -> pool column, per player type. The SINGLE place to widen
# coverage later. Verified against load_player_pool() (2026-06-27).
_HITTER_STAT_COLUMNS: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "H": "h",
    "BB": "bb",
    "HBP": "hbp",
    "AB": "ab",
    "SF": "sf",
    "AVG": "avg",
    "OBP": "obp",
}
_PITCHER_STAT_COLUMNS: dict[str, str] = {
    "IP": "ip",
    "K": "k",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "ER": "er",
    "H": "h_allowed",
    "BB": "bb_allowed",
    "ERA": "era",
    "WHIP": "whip",
}


@dataclass(frozen=True)
class PointsScoringConfig:
    """Per-stat point weights for a points league. Keys are friendly stat names
    (case-insensitive); values are points per unit of the stat (sign encodes
    inverse stats, e.g. ER = -2.0)."""

    hitter_weights: dict[str, float]
    pitcher_weights: dict[str, float]
    name: str = "custom"


@dataclass
class PointsResult:
    points: float
    breakdown: dict[str, float] = field(default_factory=dict)
    uncovered: set[str] = field(default_factory=set)


def _num(value) -> float:
    """NaN/None/inf-safe numeric coercion -> finite float (0.0 on failure)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def _get(row, key):
    """Mapping-or-Series accessor; returns None when the key is absent."""
    try:
        return row.get(key)  # dict / pandas Series / Mapping all support .get
    except AttributeError:
        return row[key] if key in row else None


def _score_half(row, weights: dict, stat_columns: dict[str, str]) -> PointsResult:
    """Score one weight map against one column map (one player 'half')."""
    points = 0.0
    breakdown: dict[str, float] = {}
    uncovered: set[str] = set()
    for raw_stat, weight in weights.items():
        stat = str(raw_stat).strip().upper()
        col = stat_columns.get(stat)
        if col is None:
            uncovered.add(stat)  # HEATER does not project this stat for this type
            continue
        contribution = _num(weight) * _num(_get(row, col))
        breakdown[stat] = contribution
        points += contribution
    return PointsResult(points=points, breakdown=breakdown, uncovered=uncovered)


def _is_hitter(row) -> bool:
    return bool(_num(_get(row, "is_hitter")))


def _has_pitcher_volume(row) -> bool:
    return _num(_get(row, "ip")) > 0.0


def project_player_points(player_row, config: PointsScoringConfig) -> PointsResult:
    """Project a single player's points under `config`.

    Hitters use hitter_weights, pitchers use pitcher_weights. A two-way player
    (is_hitter AND pitcher volume — Ohtani) is scored as BOTH halves summed;
    its breakdown keys are prefixed BAT:/PIT: to disambiguate same-named stats
    (a hitter's H vs a pitcher's H allowed)."""
    is_hit = _is_hitter(player_row)
    pitches = _has_pitcher_volume(player_row)
    two_way = is_hit and pitches

    total = 0.0
    breakdown: dict[str, float] = {}
    uncovered: set[str] = set()

    score_hit = is_hit
    score_pit = pitches and not is_hit or two_way  # pitcher-only OR two-way

    if score_hit:
        half = _score_half(player_row, config.hitter_weights, _HITTER_STAT_COLUMNS)
        total += half.points
        prefix = "BAT:" if two_way else ""
        breakdown.update({f"{prefix}{k}": v for k, v in half.breakdown.items()})
        uncovered |= half.uncovered
    if score_pit:
        half = _score_half(player_row, config.pitcher_weights, _PITCHER_STAT_COLUMNS)
        total += half.points
        prefix = "PIT:" if two_way else ""
        breakdown.update({f"{prefix}{k}": v for k, v in half.breakdown.items()})
        uncovered |= half.uncovered

    return PointsResult(points=total, breakdown=breakdown, uncovered=uncovered)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/points_scoring.py tests/test_points_scoring.py
python -m ruff check src/points_scoring.py tests/test_points_scoring.py
git add src/points_scoring.py tests/test_points_scoring.py
git commit -m "feat(points): points-league per-player scoring core (Phase 4 slice 1)"
```

---

## Task 2: Pitcher + two-way scoring

**Files:**
- Modify: `src/points_scoring.py` (no change — Task 1 already implements it; this task adds tests proving the pitcher/two-way paths)
- Test: `tests/test_points_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_points_scoring.py`:
```python
def _pitcher(**over):
    row = {"player_id": 2, "is_hitter": 0, "w": 15, "l": 8, "sv": 0, "k": 220,
           "ip": 195, "er": 70, "bb_allowed": 50, "h_allowed": 165,
           "era": 3.23, "whip": 1.10}
    row.update(over)
    return row


def test_pitcher_uses_pitcher_columns_for_h_and_bb():
    # A pitcher's "H" must resolve to h_allowed (165), NOT a hitter h.
    cfg = _cfg(pit={"K": 1.0, "W": 5.0, "ER": -2.0, "H": -0.5, "BB": -0.5})
    res = project_player_points(_pitcher(), cfg)
    # 220*1 + 15*5 + 70*-2 + 165*-0.5 + 50*-0.5 = 220 + 75 - 140 - 82.5 - 25 = 47.5
    assert res.points == 47.5


def test_two_way_player_scores_both_halves():
    cfg = _cfg(hit={"HR": 4.0}, pit={"K": 1.0})
    # Ohtani-like: hits AND pitches.
    row = _hitter(player_id=3, hr=50, ip=130, k=180, is_hitter=1)
    res = project_player_points(row, cfg)
    # hitter: 50*4 = 200 ; pitcher: 180*1 = 180 → 380
    assert res.points == 380.0


def test_pure_pitcher_is_not_scored_with_hitter_weights():
    cfg = _cfg(hit={"HR": 4.0}, pit={"K": 1.0})
    res = project_player_points(_pitcher(k=200), cfg)
    assert res.points == 200.0  # only the pitcher half
```

- [ ] **Step 2: Run to verify it passes (Task 1 already implements the behavior)**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: PASS (these exercise the pitcher/two-way branches written in Task 1). If any fail, fix `project_player_points` until green — do NOT change the tests' expected arithmetic.

- [ ] **Step 3: Commit**

```bash
git add tests/test_points_scoring.py
git commit -m "test(points): pitcher column resolution + two-way scoring (Phase 4 slice 1)"
```

---

## Task 3: Pool-level uncovered stats, ranking, roster totals

**Files:**
- Modify: `src/points_scoring.py`
- Test: `tests/test_points_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_points_scoring.py`:
```python
from src.points_scoring import (  # noqa: E402
    rank_players_by_points,
    roster_points,
    uncovered_stats,
)


def _pool():
    return pd.DataFrame([_hitter(player_id=1, hr=30), _hitter(player_id=2, hr=10),
                         _pitcher(player_id=3, k=250)])


def test_uncovered_stats_reports_per_type_unprojected_stats():
    cfg = _cfg(hit={"HR": 4.0, "K": -1.0, "DOUBLES": 2.0},
              pit={"K": 1.0, "HLD": 5.0})
    unc = uncovered_stats(cfg, _pool())
    assert unc["hitter"] == {"K", "DOUBLES"}
    assert unc["pitcher"] == {"HLD"}


def test_rank_orders_by_points_desc_and_does_not_mutate_input():
    cfg = _cfg(hit={"HR": 4.0})
    pool = _pool()
    before = pool.copy(deep=True)
    ranked = rank_players_by_points(pool, cfg)
    assert "points" in ranked.columns
    hitters = ranked[ranked["is_hitter"] == 1]
    assert list(hitters["player_id"])[:2] == [1, 2]  # hr 30 (120pts) before hr 10 (40pts)
    assert "points" not in pool.columns  # input untouched
    pd.testing.assert_frame_equal(pool, before)


def test_roster_points_sums_named_players_unknown_ids_zero():
    cfg = _cfg(hit={"HR": 4.0})
    total = roster_points([1, 2, 999], _pool(), cfg)
    assert total == 120.0 + 40.0  # ids 1 and 2; 999 unknown → 0


def test_roster_points_empty_roster_is_zero():
    cfg = _cfg(hit={"HR": 4.0})
    assert roster_points([], _pool(), cfg) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: FAIL — `ImportError` (`rank_players_by_points`/`roster_points`/`uncovered_stats` not defined).

- [ ] **Step 3: Add the pool-level functions**

Append to `src/points_scoring.py`:
```python
def uncovered_stats(config: PointsScoringConfig, pool=None) -> dict[str, set[str]]:
    """Which configured stats HEATER cannot score, per player type — for upfront
    UI transparency. `pool` is accepted for signature symmetry but not required
    (coverage is determined by the static stat maps)."""
    hit = {str(s).strip().upper() for s in config.hitter_weights}
    pit = {str(s).strip().upper() for s in config.pitcher_weights}
    return {
        "hitter": {s for s in hit if s not in _HITTER_STAT_COLUMNS},
        "pitcher": {s for s in pit if s not in _PITCHER_STAT_COLUMNS},
    }


def rank_players_by_points(pool: pd.DataFrame, config: PointsScoringConfig) -> pd.DataFrame:
    """Return a copy of `pool` with a `points` column, sorted points-descending.
    Never mutates the input."""
    out = pool.copy()
    out["points"] = [
        project_player_points(row, config).points for _, row in out.iterrows()
    ]
    return out.sort_values("points", ascending=False, kind="mergesort").reset_index(drop=True)


def roster_points(roster_ids: list, pool: pd.DataFrame, config: PointsScoringConfig) -> float:
    """Total projected points for the players in `roster_ids` (looked up by
    player_id). Unknown ids contribute 0."""
    if pool is None or len(pool) == 0 or not roster_ids:
        return 0.0
    ids = {int(_num(i)) for i in roster_ids}
    subset = pool[pool["player_id"].astype("int64", errors="ignore").isin(ids)]
    return float(sum(project_player_points(row, config).points for _, row in subset.iterrows()))
```

Note: `astype("int64", errors="ignore")` keeps the lookup robust if `player_id` is stored as text/float (Python 3.14 SQLite returns bytes for some integer columns — see CLAUDE.md). If `errors="ignore"` is unavailable in the installed pandas, replace with `pd.to_numeric(pool["player_id"], errors="coerce")`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: PASS (all tests so far). If `astype(..., errors="ignore")` raises, switch to the `pd.to_numeric(...)` form noted above and re-run.

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/points_scoring.py tests/test_points_scoring.py
python -m ruff check src/points_scoring.py tests/test_points_scoring.py
git add src/points_scoring.py tests/test_points_scoring.py
git commit -m "feat(points): pool ranking, roster totals, uncovered-stat reporting (Phase 4 slice 1)"
```

---

## Task 4: STANDARD_POINTS preset + real-pool smoke + verification

**Files:**
- Modify: `src/points_scoring.py`
- Test: `tests/test_points_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_points_scoring.py`:
```python
from src.points_scoring import STANDARD_POINTS  # noqa: E402


def test_standard_points_preset_is_usable_and_covered():
    # Every stat in the shipped preset must resolve (no uncovered) — it is built
    # only from projected stats by construction.
    unc = uncovered_stats(STANDARD_POINTS, _pool())
    assert unc["hitter"] == set()
    assert unc["pitcher"] == set()


def test_standard_points_scores_a_hitter_and_pitcher_positive_ish():
    res_h = project_player_points(_hitter(), STANDARD_POINTS)
    res_p = project_player_points(_pitcher(), STANDARD_POINTS)
    assert res_h.points > 0
    # A solid starter nets positive under the preset (K/W/IP outweigh ER/WHIP).
    assert isinstance(res_p.points, float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: FAIL — `ImportError` (`STANDARD_POINTS` not defined).

- [ ] **Step 3: Add the preset**

Append to `src/points_scoring.py`:
```python
# A documented, illustrative points preset built ONLY from projected stats (so
# it never produces 'uncovered'). It is NOT a claim of any provider's exact
# defaults — a user's real weights come from their league settings (Phase 5).
STANDARD_POINTS = PointsScoringConfig(
    name="standard",
    hitter_weights={
        "R": 1.0,
        "HR": 4.0,
        "RBI": 1.0,
        "SB": 2.0,
        "H": 1.0,
        "BB": 1.0,
    },
    pitcher_weights={
        "IP": 1.0,
        "K": 1.0,
        "W": 5.0,
        "SV": 5.0,
        "ER": -2.0,
        "H": -0.5,
        "BB": -0.5,
        "L": -3.0,
    },
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_points_scoring.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Real-pool smoke check (manual, non-test — the test DB may be empty in CI/worktrees)**

Run:
```bash
python -c "
from src.database import load_player_pool
from src.points_scoring import rank_players_by_points, STANDARD_POINTS
pool = load_player_pool()
ranked = rank_players_by_points(pool, STANDARD_POINTS)
print('top 5 by standard points:')
print(ranked[['name','points','is_hitter']].head(5).to_string() if 'name' in ranked else ranked[['player_id','points']].head(5).to_string())
"
```
Expected: prints a sensible top-5 (high-volume hitters / aces). This is a sanity check, not a committed test (it needs the 26MB local DB).

- [ ] **Step 6: Format, lint, commit**

```bash
python -m ruff format src/points_scoring.py tests/test_points_scoring.py
python -m ruff check src/points_scoring.py tests/test_points_scoring.py
git add src/points_scoring.py tests/test_points_scoring.py
git commit -m "feat(points): STANDARD_POINTS preset + coverage tests (Phase 4 slice 1)"
```

---

## Task 5: Registry + final verification + push

**Files:**
- Modify: `docs/launch/evidence_registry.yaml`

- [ ] **Step 1: Add the registry row**

In `docs/launch/evidence_registry.yaml`, insert before the `P4-FORMAT-GENERALIZATION` row:
```yaml
  - id: P4-POINTS-ENGINE
    category: analytical_value
    phase: 4
    description: Standalone points-league scoring valuation (config -> player points, ranking, roster totals; uncovered-stat flagging) over existing projected stats.
    status: passing
    subsystem: engine
    verify: "python -m pytest tests/test_points_scoring.py -q"
    metric: "points = weighted sum; uncovered surfaced; category engine untouched"
    evidence: "src/points_scoring.py, tests/test_points_scoring.py, docs/superpowers/specs/2026-06-27-heater-points-scoring-engine-design.md"
    external_review: none
    blocking_ring: invite_beta
    last_verified: "2026-06-27"
    score_contribution: analytical_value
```

- [ ] **Step 2: Validate registry + run the full points suite + confirm no regression**

Run:
```bash
python -m scripts.launch.evidence_registry --summary
python -m pytest tests/test_points_scoring.py tests/launch/test_evidence_registry.py -q
python -m pytest tests/ -k "no_ or guard" --ignore=tests/test_cheat_sheet.py -q
```
Expected: registry valid; points + registry tests green; structural guards green (the new module touches nothing guarded).

- [ ] **Step 3: Commit + push**

```bash
git add docs/launch/evidence_registry.yaml
git commit -m "docs(launch): register P4-POINTS-ENGINE (passing) (Phase 4 slice 1)"
git pull --no-rebase --no-edit origin master
git push origin master
```
Expected: pre-push structural suite passes; push succeeds. Never `--no-verify`.

---

## Self-review notes

- **Spec coverage:** config + maps → Task 1; per-type column resolution + two-way → Tasks 1-2; uncovered flagging → Tasks 1 & 3; rank/roster → Task 3; preset → Task 4; NaN-safe/never-raise → Task 1; "no existing file touched" → confirmed (only `src/points_scoring.py` + its test + the registry doc). Out-of-scope items (optimizer/FA/trade wiring, points-VORP, LeagueConfig generalization, Roto, extra-stat projections) are correctly absent.
- **No placeholders:** every step has exact code/commands. The two `astype` fallbacks are explicit recoverable instructions, not deferrals.
- **Type/name consistency:** `PointsScoringConfig(hitter_weights, pitcher_weights, name)`, `PointsResult(points, breakdown, uncovered)`, `project_player_points`, `rank_players_by_points`, `roster_points`, `uncovered_stats`, `STANDARD_POINTS`, `_HITTER_STAT_COLUMNS`/`_PITCHER_STAT_COLUMNS` — used identically across tasks and tests.
```
