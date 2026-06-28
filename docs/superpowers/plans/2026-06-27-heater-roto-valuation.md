# Roto Valuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone `src/roto_scoring.py` that computes rank-sum Rotisserie standings from team category totals, values a player in roto terms (SGP), and computes a player's standings-aware marginal roto contribution to a team.

**Architecture:** Pure functions that reuse the proven category primitives read-only — `SGPCalculator.total_sgp`/`total_sgp_batch` is the per-player roto value, `LeagueConfig` supplies categories + inverse stats. The net-new piece is `compute_roto_standings` (rank-sum, inverse-aware, tie-averaged). `marginal_roto_value` re-ranks after adding a player, recomputing rate categories from component totals (built from the pool). Never touches the category engine's internals; never raises; NaN-safe.

**Tech Stack:** Python 3.12/3.14, pandas, numpy, scipy (`scipy.stats.rankdata`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-27-heater-roto-valuation-design.md`. **Verify at impl time:** `standings_utils.get_all_team_totals()` returns `{team: {category: value}}` (adapt with a tiny normalizer if the live shape differs); pool component columns (hitters `r/hr/rbi/sb/h/bb/hbp/ab/sf`; pitchers `w/l/sv/k/ip/er/bb_allowed/h_allowed`; plus `is_hitter`, `player_id`) per `_build_player_pool`.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/roto_scoring.py` | roto standings, per-player roto value, ranking, marginal roto value | Create |
| `tests/test_roto_scoring.py` | TDD coverage | Create |

No existing file is modified. (`SGPCalculator`/`LeagueConfig` imported read-only.)

---

## Task 1: Roto standings core

**Files:**
- Create: `src/roto_scoring.py`
- Test: `tests/test_roto_scoring.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_roto_scoring.py`:
```python
import pandas as pd

from src.roto_scoring import compute_roto_standings
from src.valuation import LeagueConfig


def _totals():
    """3 teams; equal in every cat except HR (A>B>C) and ERA (A<B<C = A best)."""
    base = {c: 0.0 for c in LeagueConfig().all_categories}
    return {
        "A": dict(base, HR=30, ERA=3.00),
        "B": dict(base, HR=20, ERA=4.00),
        "C": dict(base, HR=10, ERA=5.00),
    }


def _row(df, team):
    return df[df["team"] == team].iloc[0]


def test_normal_cat_best_gets_n_worst_gets_one():
    df = compute_roto_standings(_totals(), LeagueConfig())
    assert _row(df, "A")["pts_HR"] == 3.0  # most HR in a 3-team field
    assert _row(df, "C")["pts_HR"] == 1.0  # fewest


def test_inverse_cat_lowest_value_is_best():
    df = compute_roto_standings(_totals(), LeagueConfig())
    assert _row(df, "A")["pts_ERA"] == 3.0  # lowest ERA -> best
    assert _row(df, "C")["pts_ERA"] == 1.0


def test_ties_split_points():
    cfg = LeagueConfig()
    base = {c: 0.0 for c in cfg.all_categories}
    tt = {"A": dict(base, HR=30), "B": dict(base, HR=30), "C": dict(base, HR=10)}
    df = compute_roto_standings(tt, cfg)
    # A and B tie for the top two slots (3,2) -> 2.5 each; C gets 1
    assert _row(df, "A")["pts_HR"] == 2.5
    assert _row(df, "B")["pts_HR"] == 2.5
    assert _row(df, "C")["pts_HR"] == 1.0


def test_roto_score_is_sum_and_sorted_desc_with_rank():
    df = compute_roto_standings(_totals(), LeagueConfig())
    assert list(df["team"]) == ["A", "B", "C"]
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["roto_score"] >= df.iloc[1]["roto_score"] >= df.iloc[2]["roto_score"]


def test_empty_team_totals_returns_empty():
    assert compute_roto_standings({}, LeagueConfig()).empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.roto_scoring'`.

- [ ] **Step 3: Write the implementation**

`src/roto_scoring.py`:
```python
"""Rotisserie scoring valuation (Phase 4 slice 2).

Standalone + additive: pure functions for roto-league valuation. Roto is
categories scored by RANK, so this module reuses the proven category primitives
read-only (SGPCalculator = per-player roto value; LeagueConfig = categories +
inverse stats). The net-new piece is rank-sum standings. Never touches the
category engine's internals; never raises; NaN-safe.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.valuation import LeagueConfig, SGPCalculator


def _num(value) -> float:
    """NaN/None/inf-safe numeric coercion -> finite float (0.0 on failure)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def compute_roto_standings(team_totals: dict, config: LeagueConfig | None = None) -> pd.DataFrame:
    """Rank-sum roto standings from each team's per-category totals.

    team_totals: {team_name: {category: value}} (counting cats = season/projected
    totals; rate cats = aggregated team rate). Best team in a category scores N
    (= num_teams), worst scores 1; inverse cats (L/ERA/WHIP) flip direction; ties
    split the slot points (standard roto). Returns DataFrame
    [team, roto_score, rank, pts_<cat>...] sorted desc.
    """
    config = config or LeagueConfig()
    teams = list(team_totals.keys())
    cats = list(config.all_categories)
    if not teams:
        return pd.DataFrame(columns=["team", "roto_score", "rank"])
    inverse = config.inverse_stats
    pts: dict[str, dict[str, float]] = {t: {} for t in teams}
    for cat in cats:
        vals = np.array([_num(team_totals[t].get(cat)) for t in teams], dtype=float)
        ordered = -vals if cat in inverse else vals  # higher ordered = better
        ranks = rankdata(ordered, method="average")  # 1=worst .. N=best, ties averaged
        for i, t in enumerate(teams):
            pts[t][cat] = float(ranks[i])
    rows = []
    for t in teams:
        row = {"team": t, "roto_score": float(sum(pts[t].values()))}
        row.update({f"pts_{c}": pts[t][c] for c in cats})
        rows.append(row)
    df = (
        pd.DataFrame(rows)
        .sort_values(["roto_score", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df["rank"] = range(1, len(df) + 1)
    return df
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/roto_scoring.py tests/test_roto_scoring.py
python -m ruff check src/roto_scoring.py tests/test_roto_scoring.py
git add src/roto_scoring.py tests/test_roto_scoring.py
git commit -m "feat(roto): rank-sum roto standings (Phase 4 slice 2)"
```

---

## Task 2: Per-player roto value + ranking

**Files:**
- Modify: `src/roto_scoring.py`
- Test: `tests/test_roto_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roto_scoring.py`:
```python
from src.roto_scoring import player_roto_value, rank_players_by_roto_value  # noqa: E402
from src.valuation import SGPCalculator  # noqa: E402


def _player(**over):
    row = {"player_id": 1, "is_hitter": 1, "r": 90, "hr": 30, "rbi": 100, "sb": 10,
           "ab": 550, "h": 150, "bb": 60, "hbp": 5, "sf": 4, "avg": 0.273, "obp": 0.350}
    row.update(over)
    return pd.Series(row)


def test_player_roto_value_equals_total_sgp():
    cfg = LeagueConfig()
    p = _player()
    assert player_roto_value(p, cfg) == float(SGPCalculator(cfg).total_sgp(p))


def test_rank_orders_desc_and_does_not_mutate():
    cfg = LeagueConfig()
    pool = pd.DataFrame([_player(player_id=1, hr=40).to_dict(),
                         _player(player_id=2, hr=5).to_dict()])
    before = pool.copy(deep=True)
    ranked = rank_players_by_roto_value(pool, cfg)
    assert "roto_value" in ranked.columns
    assert list(ranked["player_id"])[0] == 1  # more HR -> more roto value
    assert "roto_value" not in pool.columns
    pd.testing.assert_frame_equal(pool, before)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: FAIL — `ImportError` (`player_roto_value`/`rank_players_by_roto_value` not defined).

- [ ] **Step 3: Append the implementation**

Append to `src/roto_scoring.py`:
```python
def player_roto_value(player_row, config: LeagueConfig | None = None) -> float:
    """Per-player roto value = Standings Gain Points (canonical roto valuation).
    Context-free (league-average); use marginal_roto_value for a roster-aware,
    standings-aware figure."""
    config = config or LeagueConfig()
    return float(SGPCalculator(config).total_sgp(player_row))


def rank_players_by_roto_value(pool: pd.DataFrame, config: LeagueConfig | None = None) -> pd.DataFrame:
    """Return a copy of `pool` with a `roto_value` (SGP) column, sorted desc.
    Never mutates the input."""
    config = config or LeagueConfig()
    out = pool.copy() if pool is not None else pd.DataFrame()
    if len(out) == 0:
        out["roto_value"] = []
        return out
    out["roto_value"] = SGPCalculator(config).total_sgp_batch(out)
    return out.sort_values("roto_value", ascending=False, kind="mergesort").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: PASS (all tests so far).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/roto_scoring.py tests/test_roto_scoring.py
python -m ruff check src/roto_scoring.py tests/test_roto_scoring.py
git add src/roto_scoring.py tests/test_roto_scoring.py
git commit -m "feat(roto): per-player roto value + ranking via SGP (Phase 4 slice 2)"
```

---

## Task 3: Standings-aware marginal roto value

**Files:**
- Modify: `src/roto_scoring.py`
- Test: `tests/test_roto_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roto_scoring.py`:
```python
from src.roto_scoring import marginal_roto_value  # noqa: E402


def _pool_for_marginal():
    # player 10 = my weak hitter (5 HR); player 20 = a 45-HR bat (FA candidate)
    return pd.DataFrame([
        {"player_id": 10, "is_hitter": 1, "r": 50, "hr": 5, "rbi": 40, "sb": 2,
         "h": 120, "bb": 30, "hbp": 2, "ab": 500, "sf": 3},
        {"player_id": 20, "is_hitter": 1, "r": 100, "hr": 45, "rbi": 110, "sb": 5,
         "h": 170, "bb": 70, "hbp": 6, "ab": 560, "sf": 5},
    ])


def _all_totals(my_hr_rivals):
    cfg = LeagueConfig()
    base = {c: 0.0 for c in cfg.all_categories}
    tt = {"MINE": dict(base)}  # MINE value is overridden by pool-derived totals
    tt.update(my_hr_rivals)
    return tt


def test_marginal_value_positive_when_player_helps():
    cfg = LeagueConfig()
    tt = _all_totals({"B": {**{c: 0.0 for c in cfg.all_categories}, "HR": 20},
                      "C": {**{c: 0.0 for c in cfg.all_categories}, "HR": 1}})
    d = marginal_roto_value(20, "MINE", [10], tt, _pool_for_marginal(), cfg)
    assert d > 0


def test_marginal_value_zero_for_unknown_player():
    cfg = LeagueConfig()
    tt = _all_totals({"B": {**{c: 0.0 for c in cfg.all_categories}, "HR": 20}})
    assert marginal_roto_value(999, "MINE", [10], tt, _pool_for_marginal(), cfg) == 0.0


def test_marginal_value_zero_when_team_absent():
    cfg = LeagueConfig()
    tt = _all_totals({"B": {**{c: 0.0 for c in cfg.all_categories}, "HR": 20}})
    assert marginal_roto_value(20, "NOPE", [10], tt, _pool_for_marginal(), cfg) == 0.0


def test_marginal_value_higher_when_category_is_contested():
    cfg = LeagueConfig()
    base = {c: 0.0 for c in cfg.all_categories}
    pool = _pool_for_marginal()  # my roster HR (from player 10) = 5
    # Contested: a rival just above my 5 HR -> adding 45 jumps me a rank.
    contested = _all_totals({"B": dict(base, HR=20), "C": dict(base, HR=1)})
    # Locked: rivals below my 5 HR -> I'm already 1st, adding HR gains no rank.
    locked = _all_totals({"B": dict(base, HR=3), "C": dict(base, HR=1)})
    dc = marginal_roto_value(20, "MINE", [10], contested, pool, cfg)
    dl = marginal_roto_value(20, "MINE", [10], locked, pool, cfg)
    assert dc > dl
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: FAIL — `ImportError` (`marginal_roto_value` not defined).

- [ ] **Step 3: Append the implementation**

Append to `src/roto_scoring.py`:
```python
# Pool COLUMN names for rate-stat component aggregation (not category lists).
_HIT_COMPONENTS = ["r", "hr", "rbi", "sb", "h", "bb", "hbp", "ab", "sf"]
_PIT_COMPONENTS = ["w", "l", "sv", "k", "ip", "er", "bb_allowed", "h_allowed"]


def _player_get(row, key):
    try:
        return row.get(key)
    except AttributeError:
        return row[key] if key in row else None


def _counting_col_map(config: LeagueConfig) -> dict:
    """Display category -> pool column, derived from LeagueConfig (no hardcoded
    category list — satisfies the category-hardcoding structural guard)."""
    return {cat: config.STAT_MAP[cat] for cat in config.counting_stats if cat in config.STAT_MAP}


def _sum_components(rows: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in rows.columns:
            out[c] = float(pd.to_numeric(rows[c], errors="coerce").fillna(0.0).sum())
        else:
            out[c] = 0.0
    return out


def _component_totals(roster_ids, pool: pd.DataFrame) -> dict:
    ids = {int(_num(i)) for i in (roster_ids or [])}
    pid = pd.to_numeric(pool["player_id"], errors="coerce")
    sub = pool[pid.isin(ids)]
    is_hit = pd.to_numeric(sub.get("is_hitter"), errors="coerce").fillna(0)
    comp = _sum_components(sub[is_hit == 1], _HIT_COMPONENTS)
    comp.update(_sum_components(sub[is_hit == 0], _PIT_COMPONENTS))
    return comp


def _values_from_components(comp: dict, config: LeagueConfig) -> dict:
    v = {cat: comp.get(col, 0.0) for cat, col in _counting_col_map(config).items()}
    ab, h, bb, hbp, sf = (comp.get(k, 0.0) for k in ("ab", "h", "bb", "hbp", "sf"))
    v["AVG"] = h / ab if ab > 0 else 0.0
    v["OBP"] = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0.0
    ip, er, bba, ha = (comp.get(k, 0.0) for k in ("ip", "er", "bb_allowed", "h_allowed"))
    v["ERA"] = er * 9.0 / ip if ip > 0 else 0.0
    v["WHIP"] = (bba + ha) / ip if ip > 0 else 0.0
    return v


def _add_player_components(comp: dict, player_row) -> dict:
    out = dict(comp)
    is_hit = bool(_num(_player_get(player_row, "is_hitter")))
    has_ip = _num(_player_get(player_row, "ip")) > 0.0
    if is_hit:
        for c in _HIT_COMPONENTS:
            out[c] = out.get(c, 0.0) + _num(_player_get(player_row, c))
    if has_ip or not is_hit:  # pitcher OR two-way
        for c in _PIT_COMPONENTS:
            out[c] = out.get(c, 0.0) + _num(_player_get(player_row, c))
    return out


def marginal_roto_value(player_id, my_team_name, my_roster_ids, all_team_totals: dict,
                        pool: pd.DataFrame, config: LeagueConfig | None = None) -> float:
    """Delta in my roto score from adding the player to my team. Standings-aware
    (re-ranks) and rate-correct (recomputes AVG/OBP/ERA/WHIP from components).
    Returns 0.0 for unknown player / absent team / empty inputs."""
    config = config or LeagueConfig()
    if (not all_team_totals or my_team_name not in all_team_totals
            or pool is None or len(pool) == 0):
        return 0.0
    pid = pd.to_numeric(pool["player_id"], errors="coerce")
    prow = pool[pid == int(_num(player_id))]
    if prow.empty:
        return 0.0
    prow = prow.iloc[0]

    comp = _component_totals(my_roster_ids, pool)
    cur_vals = _values_from_components(comp, config)
    new_vals = _values_from_components(_add_player_components(comp, prow), config)

    def _my_score(my_vals):
        tt = dict(all_team_totals)
        tt[my_team_name] = my_vals
        df = compute_roto_standings(tt, config)
        sel = df[df["team"] == my_team_name]
        return float(sel.iloc[0]["roto_score"]) if not sel.empty else 0.0

    return _my_score(new_vals) - _my_score(cur_vals)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_roto_scoring.py -q`
Expected: PASS (all tests). If `test_marginal_value_higher_when_category_is_contested` fails, do NOT change the test arithmetic — re-check that `marginal_roto_value` derives MY team values from the pool (so the rivals' HR is what differs between scenarios) and that inverse handling is unaffected.

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/roto_scoring.py tests/test_roto_scoring.py
python -m ruff check src/roto_scoring.py tests/test_roto_scoring.py
git add src/roto_scoring.py tests/test_roto_scoring.py
git commit -m "feat(roto): standings-aware, rate-correct marginal roto value (Phase 4 slice 2)"
```

---

## Task 4: Registry row + verification + push

**Files:**
- Modify: `docs/launch/evidence_registry.yaml`

- [ ] **Step 1: Add the registry row**

In `docs/launch/evidence_registry.yaml`, insert immediately AFTER the `P4-POINTS-ENGINE` row and BEFORE `P4-FORMAT-GENERALIZATION`:
```yaml
  - id: P4-ROTO-ENGINE
    category: analytical_value
    phase: 4
    description: Standalone Rotisserie valuation (rank-sum roto standings; per-player roto value via SGP; standings-aware, rate-correct marginal roto value) over existing projected stats. Category engine untouched.
    status: passing
    subsystem: engine
    verify: "python -m pytest tests/test_roto_scoring.py -q"
    metric: "rank-sum standings correct (inverse + ties); marginal value standings-aware + rate-correct"
    evidence: "src/roto_scoring.py, tests/test_roto_scoring.py, docs/superpowers/specs/2026-06-27-heater-roto-valuation-design.md"
    external_review: none
    blocking_ring: invite_beta
    last_verified: "2026-06-27"
    score_contribution: analytical_value
```

- [ ] **Step 2: Validate registry + run roto suite + structural guards**

Run:
```bash
python -m scripts.launch.evidence_registry --summary
python -m pytest tests/test_roto_scoring.py -q
python -m pytest tests/ -k "no_hardcoded_categories or no_lc_singletons or no_direct_sqlite" --ignore=tests/test_cheat_sheet.py -q
```
Expected: registry valid (P4-ROTO-ENGINE passing); roto tests green; structural guards green. The module derives category↔column from `LeagueConfig.STAT_MAP`/`counting_stats` and instantiates `SGPCalculator` inside functions (no module-level `_LC` singleton), so the category-hardcoding and singleton guards stay green. If a guard flags the new file, fix by deriving from `config` (do NOT add hardcoded category literals).

- [ ] **Step 3: Real-pool smoke check (manual, non-test — the test DB may be empty in CI/worktrees)**

Run:
```bash
python -c "
from src.database import load_player_pool
from src.standings_utils import get_all_team_totals
from src.roto_scoring import rank_players_by_roto_value, compute_roto_standings
from src.valuation import LeagueConfig
pool = load_player_pool()
ranked = rank_players_by_roto_value(pool, LeagueConfig())
print('top 5 by roto value (SGP):')
print(ranked[['player_id','roto_value']].head().to_string())
tt = get_all_team_totals()
if tt:
    print(compute_roto_standings(tt, LeagueConfig())[['team','roto_score','rank']].to_string())
"
```
Expected: a sensible top-5 (aces / high-volume bats) and, if league totals are present, a roto standings table. Sanity check only (needs the local DB) — verify the `get_all_team_totals()` shape here and add a normalizer in `compute_roto_standings`/`marginal_roto_value` callers if it isn't `{team: {cat: value}}`.

- [ ] **Step 4: Commit + push**

```bash
git add docs/launch/evidence_registry.yaml
git commit -m "docs(launch): register P4-ROTO-ENGINE (passing) (Phase 4 slice 2)"
git pull --no-rebase --no-edit origin master
git push origin master
```
Expected: pre-push structural suite passes; push succeeds. Never `--no-verify`. (Reconcile the parallel CMO track via the `git pull --no-rebase` — do not disturb their branch.)

---

## Self-review notes

- **Spec coverage:** `compute_roto_standings` (inverse + ties + rank-sum) → Task 1; `player_roto_value`/`rank_players_by_roto_value` → Task 2; `marginal_roto_value` (standings-aware + rate-correct from components) → Task 3; registry row + verification → Task 4. Out-of-scope items (optimizer/FA/trade/standings-page wiring, roto-VORP, connector-sourced settings, LeagueConfig generalization) are correctly absent.
- **No placeholders:** every step has exact code/commands. The `get_all_team_totals()` shape and pool-column verifications are explicit recoverable instructions, not deferrals.
- **Type/name consistency:** `compute_roto_standings`, `player_roto_value`, `rank_players_by_roto_value`, `marginal_roto_value`, `_num`, `_component_totals`, `_values_from_components`, `_add_player_components`, `_counting_col_map`, `_HIT_COMPONENTS`/`_PIT_COMPONENTS` used identically across tasks and tests. DataFrame columns `team`/`roto_score`/`rank`/`pts_<cat>`/`roto_value` consistent throughout.
