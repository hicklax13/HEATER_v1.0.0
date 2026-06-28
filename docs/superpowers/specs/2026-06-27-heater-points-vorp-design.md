# HEATER Points VORP — Design (Phase 4, slice 3)

**Date:** 2026-06-27
**Status:** Approved design (brainstorming complete 2026-06-27) — pending owner review of this written spec, then writing-plans.
**Owner:** Connor
**Program:** Public commercial launch, Phase 4 (League-format generalization). Third standalone slice — **value over replacement** for **points** leagues. Builds on slice 1 (`src/points_scoring.py`); buildable now; no Phase 2/3 dependency.

---

## 1. Purpose

Slice 1 ranks players by **raw projected points**, but raw points ignore **positional scarcity** — a 400-point catcher is worth more than a 400-point outfielder because replacement-level catchers are far weaker. This slice adds **value over replacement (VORP)** for points leagues: per-position replacement baselines and a player's points above the replaceable alternative at their position — making the points engine decision-useful, exactly as HEATER's category VORP does for categories.

It mirrors the existing category VORP (`compute_replacement_levels` + `compute_vorp` in `src/valuation.py`) but over **points**, and **does not touch the category scoring** (no SGP) — consistent with slice-1's standalone ethos.

## 2. Locked decisions (brainstorming, 2026-06-27)

1. **Mirror the proven category VORP shape** (per-position replacement levels → points − best-eligible replacement → multi-position flexibility bonus) so behavior is familiar across formats.
2. **Reuse `compute_positional_scarcity_factor`** from `src/valuation.py` — a **format-agnostic** utility (positions + a replacement dict → multiplier; no SGP knowledge), already hoisted for cross-module reuse. Points VORP imports it; it imports **no** category scoring.
3. **Roster structure comes from `LeagueConfig`, read-only** (owner choice 2026-06-27, option i): `num_teams`, `roster_slots`, `hitter_starters_at`, `pitcher_starters` — all format-neutral roster modeling. Points VORP uses **none** of LeagueConfig's categories / SGP.
4. **New module** `src/points_vorp.py` (imports `points_scoring`), keeping slice 1 frozen.
5. **Standalone valuation layer only** — no optimizer/FA/trade wiring.

## 3. Architecture

A single new module `src/points_vorp.py`. Pure functions over the pool + a `PointsScoringConfig` (slice 1) + a `LeagueConfig` (roster structure only). No global state; no DB writes; never raises.

```
compute_points_replacement_levels(pool, points_config, league_config) ──► {position -> replacement_points}
points_vorp(player_row, points_config, replacement_levels)            ──► float
rank_players_by_points_vorp(pool, points_config, league_config)       ──► DataFrame (+ "points", "points_vorp", sorted)
```

### 3.1 Replacement levels (mirror of `compute_replacement_levels`)

For each hitting position (C, 1B, 2B, 3B, SS, OF) and pitching position (SP, RP):

- Filter the pool to players **eligible** at that position (`positions` contains the slot) and of the right type (`is_hitter`).
- Score each with `project_player_points(row, points_config).points` (slice 1).
- Sort descending; `n_starters = league_config.hitter_starters_at(pos)` (or `league_config.pitcher_starters()[pos]`).
- `replacement[pos]` = the points of the player at index `n_starters` (the best **non**-startable, i.e., waiver-level), or `last_eligible_points × 0.5` if the eligible pool is shallower than `n_starters` (same fallback as the category version).
- `replacement["Util"]` = min hitting replacement (deepest hitting baseline), mirroring the category code.

**Scarcity falls out naturally:** scarce positions (e.g., C) have a weaker n-th player → a **lower** replacement baseline → a good player there clears it by more → **higher** VORP. No extra scarcity term is needed in the core (the category VORP relies on this same mechanism).

### 3.2 `points_vorp` (mirror of `compute_vorp`)

`vorp = project_player_points(player, points_config).points − best_repl`, where `best_repl = max(replacement[p] for p in the player's eligible positions)` (the **best** replacement among eligible spots — the conservative baseline the player must beat; `0` if no valid position). Then the same **multi-position flexibility bonus** as `compute_vorp`: `+0.12·(num_eligible − 1) + 0.08·(scarce_count)` with `scarce_positions = {C, SS, 2B}`. Two-way players (Ohtani) already carry both halves' points from slice 1 and are eligible at hitting slots + SP, so they flow through unchanged.

`compute_positional_scarcity_factor(positions, replacement_levels)` is **available for callers** that want the additional scarcity *multiplier* — kept separate from `points_vorp`, exactly as the category side keeps it out of `compute_vorp` (scarcity is applied by consumers like the FA engine, not baked into raw VORP).

### 3.3 `rank_players_by_points_vorp`

Compute replacement levels once, then `points` + `points_vorp` per row; return a copy sorted by `points_vorp` descending (never mutates input).

### 3.4 Edge cases

- Position with no eligible players → `replacement[pos] = 0`. Player with no mapped position → `best_repl = 0` (VORP = raw points). Eligible pool shallower than starters → `last × 0.5` fallback. NaN-safe + never-raise inherited from `project_player_points`. `league_config` / `points_config` default to `LeagueConfig()` / `STANDARD_POINTS`.

## 4. Error handling

Never raises; NaN/None → finite via slice-1's `_num`. Empty pool → all replacements `0` / empty ranking. No replacement data → neutral (`best_repl = 0`), mirroring slice 1 + the category VORP's fail-safe defaults.

## 5. Testing (TDD)

- **Scarcity demotion (headline):** two players with equal raw points, one eligible only at a scarce position (C, low replacement) and one only at a deep position (OF, high replacement) → the catcher's `points_vorp` is **higher**.
- `compute_points_replacement_levels`: replacement = the (n_starters)-th best eligible by points; shallow pool → `last × 0.5`; `Util` = min hitting replacement; a `LeagueConfig` with different `roster_slots`/`num_teams` changes the starter counts → changes replacement levels.
- `points_vorp`: equals `points − best-eligible replacement`; multi-position flexibility bonus applied (a C/SS player ranks above a single-position player with equal points − replacement).
- `rank_players_by_points_vorp`: sorted by `points_vorp`; `points` + `points_vorp` columns present; input unmutated.
- NaN / missing / empty → `0.0` / empty, no raise.

## 6. File structure

| File | Responsibility | Action |
|---|---|---|
| `src/points_vorp.py` | replacement levels, points VORP, VORP ranking | Create |
| `tests/test_points_vorp.py` | TDD coverage of §5 | Create |

No existing file is modified. (`points_scoring`, `compute_positional_scarcity_factor`, and `LeagueConfig` are imported read-only.)

## 7. Out of scope

- Optimizer / FA / trade / draft wiring of points-VORP (later slices; parallel track owns integration).
- A **unified cross-format VORP** abstraction (points + SGP + category value through one replacement layer) — a sensible later refactor once both concrete versions exist.
- Sourcing roster structure from a **connected league** (Phase 5).
- Extra-stat projections to widen points coverage (a slice-1 out-of-scope item).

## 8. Success criteria

- For any points config + league roster structure, replacement levels and `points_vorp` rank players by scarcity-adjusted value (scarce-position players correctly boosted), reusing slice-1 scoring + the shared scarcity utility, with **no** category-scoring dependency.
- Zero impact on the category engine / live app (no existing file touched; suite green).
- A `P4-POINTS-VORP` evidence-registry row (added at build time by the plan) tracks it.
