# HEATER Points Scoring Engine — Design (Phase 4, slice 1)

**Date:** 2026-06-27
**Status:** Approved design (brainstorming complete) — pending owner review of this written spec, then writing-plans.
**Owner:** Connor
**Program:** Public commercial launch, Phase 4 (League-format generalization). This is the first, standalone slice of Phase 4 — the Points scoring paradigm — buildable now with no dependency on Phase 2 (Postgres) or Phase 3 (tenant model).

---

## 1. Purpose

HEATER's engines are hardwired to **categories** scoring (head-to-head category wins, valued via SGP). The public "all formats" goal requires also supporting **points** leagues, where every stat is worth a configurable number of points and a player/roster's value is a single weighted sum. This slice delivers the **standalone Points valuation core**: given a points league's scoring, compute each player's projected points, rank players, and total a roster — the foundation the optimizer / FA / trade / draft will later build on for points leagues.

It does **not** touch the proven category engine (`LeagueConfig` / `SGPCalculator`). It is pure, additive, and independently testable.

## 2. Locked decisions (from brainstorming, 2026-06-27)

1. **Points first** (the bigger net-new paradigm; Roto reuses categories and comes later).
2. **v1 scores only the stats HEATER already projects.** Any league-config stat HEATER doesn't project is **skipped and flagged** ("uncovered"), never silently zeroed-and-hidden. Extra stats (1B/2B/3B split, holds, quality starts) are a later projection slice.
3. **Approach A — standalone module** parallel to the category engine. The unified scoring abstraction (and the `LeagueConfig` generalization) is deferred to later Phase 4 work, once a second paradigm exists to abstract over.

## 3. Architecture

A single new module `src/points_scoring.py`, pure functions over the existing enriched player pool (`load_player_pool()` output). No global state; no DB writes; never raises.

```
PointsScoringConfig (hitter_weights, pitcher_weights, name)
        │
        ▼
project_player_points(player_row, config) ──► PointsResult(points, breakdown, uncovered)
rank_players_by_points(pool, config)      ──► DataFrame (+ "points" column, sorted)
roster_points(roster_ids, pool, config)   ──► float
uncovered_stats(config, pool)             ──► {hitter: set, pitcher: set}
```

### 3.1 The stat-name → pool-column mapping (the crux)

A points league's settings name stats in friendly terms ("BB", "K", "H"), but those mean **different pool columns for hitters vs pitchers** (a hitter's "BB" is `bb`; a pitcher's "BB allowed" is `bb_allowed`). The engine resolves a config stat name → a pool column **per player type**. A config stat with no mapping for a given type is **uncovered** for that type.

**Hitter map** (config name → pool column; pool columns per `_build_player_pool`):
`R→r, HR→hr, RBI→rbi, SB→sb, H→h, BB→bb, HBP→hbp, AB→ab, SF→sf, AVG→avg, OBP→obp`
Uncovered for hitters (HEATER doesn't project them): `1B, 2B, 3B, CS, K (hitter strikeouts), SO, TB, GIDP, …`

**Pitcher map:**
`IP→ip, K→k, W→w, L→l, SV→sv, ER→er, H→h_allowed, BB→bb_allowed, ERA→era, WHIP→whip` (FIP/xFIP/SIERA also available)
Uncovered for pitchers: `HLD, QS, BS, HBP_allowed, CG, SHO, …`

The exact pool column names are verified against `load_player_pool()` at implementation time; the mapping table lives in `points_scoring.py` as two dicts (`_HITTER_STAT_COLUMNS`, `_PITCHER_STAT_COLUMNS`) so it's the single place to extend coverage later.

### 3.2 Config

```python
@dataclass(frozen=True)
class PointsScoringConfig:
    hitter_weights: dict[str, float]    # friendly stat name -> points, e.g. {"HR": 4.0, "R": 1.0, ...}
    pitcher_weights: dict[str, float]   # e.g. {"IP": 3.0, "K": 1.0, "W": 5.0, "ER": -2.0, ...}
    name: str = "custom"
```

- Weights are arbitrary (real leagues are fully custom) — keys are friendly stat names (case-insensitive, normalized upper).
- One documented **`STANDARD_POINTS`** preset ships for convenience/tests (a common points scoring). It is illustrative — **provider-exact presets (Yahoo/ESPN/CBS) are NOT claimed in v1**; the real source of a user's weights is their league settings, which arrives via the connectors in Phase 5. Presets can be refined/added later.

### 3.3 Core functions

- `project_player_points(player_row: Mapping, config) -> PointsResult`
  Determines the player's type from the pool's `is_hitter` flag. Applies the matching weight map: for each `(stat, weight)`, resolves the pool column; if present and numeric, adds `weight * value`; if the stat has no column for this type, records it in `uncovered`. A **two-way player** (`is_hitter` true AND has pitcher stats — Ohtani) scores **both** halves (hitter points + pitcher points). Returns `PointsResult(points: float, breakdown: dict[str, float], uncovered: set[str])`.
- `rank_players_by_points(pool: DataFrame, config) -> DataFrame`
  Adds a `points` column (via `project_player_points` per row) and returns the pool sorted descending by `points`. Pure (returns a copy; never mutates input).
- `roster_points(roster_ids: list[int], pool: DataFrame, config) -> float`
  Sum of `project_player_points(...).points` over the roster's players (looked up by `player_id`).
- `uncovered_stats(config, pool) -> dict[str, set[str]]`
  Returns `{"hitter": {...}, "pitcher": {...}}` — which configured stats HEATER can't score, for upfront UI transparency ("your league scores Holds and Doubles, which HEATER doesn't project yet").

### 3.4 Two-way players, rate stats, edge cases

- **Two-way (Ohtani):** scored as hitter + pitcher (both maps applied, points summed). Detected by `is_hitter == True` AND a non-null pitcher stat (e.g. `ip > 0`).
- **Rate stats** (AVG/OBP/ERA/WHIP): supported uniformly if a league weights them (`weight * value`) — unusual but not rejected.
- **Inverse semantics are the *league's* job**, encoded in the sign of the weight (e.g. ER is `-2.0`). The engine never flips signs itself — it's a pure weighted sum. This is intentionally simpler than the category engine's inverse-stat handling.

## 4. Error handling

- **NaN/None-safe:** a missing or non-numeric stat value contributes `0.0` (not NaN); the weighted sum is always a finite float.
- **Never raises** into callers: bad rows, missing columns, empty pools, unknown `player_id`s degrade to `0.0` / empty results.
- **Uncovered ≠ error:** an unscoreable configured stat is reported in `uncovered`, not raised — the engine still returns a valid (partial) score. This is the load-bearing honesty property (per the locked decision).

## 5. Testing (TDD)

- A fixture hitter row + a config → `points` equals the exact hand-computed weighted sum; `breakdown` matches per-stat; `uncovered` lists the configured-but-unprojected stats (e.g. a config with hitter `K` flags `K` uncovered).
- A fixture pitcher row + config → correct sum using `bb_allowed`/`h_allowed` (proves the per-type column resolution).
- A two-way (Ohtani-like) row → hitter + pitcher points summed.
- `rank_players_by_points` → descending order, input unmutated, `points` column present.
- `roster_points` → sum over a roster; unknown `player_id` contributes 0.
- NaN/None/missing-column rows → `0.0`, no raise; empty pool/roster → empty/`0.0`.
- `STANDARD_POINTS` preset validates (all keys resolve or are documented-uncovered).
- `uncovered_stats` → returns the expected per-type unscoreable sets for a config mixing covered + uncovered stats.

## 6. File structure

| File | Responsibility | Action |
|---|---|---|
| `src/points_scoring.py` | Config, stat→column maps, the 4 core functions, `STANDARD_POINTS` preset | Create |
| `tests/test_points_scoring.py` | TDD coverage of §5 | Create |

No existing file is modified. (`LeagueConfig`, the optimizer, and the pool builder are untouched.)

## 7. Out of scope (later Phase 4 / other phases)

- Wiring points into the **optimizer** (points-maximizing start/sit), **FA** value, **trade** value, **draft** value, and **standings** (points leagues are total-points standings).
- **Value over replacement** in points (positional scarcity — the points analog of VORP). Important for decision usefulness; its own slice (needs replacement baselines).
- The `LeagueConfig` **generalization** to carry `scoring_type` + weights, and the unified scoring abstraction (later Phase 4).
- Sourcing point values from a **connected league's settings** (Phase 5 connectors).
- **Roto** scoring (separate Phase 4 slice — reuses categories with rank-sum standings).
- **Extra-stat projections** (1B/2B/3B, holds, quality starts) to widen coverage.

## 8. Success criteria

- For any supplied points config, the engine returns each player's projected points + roster totals over HEATER's projected stats, with unscoreable stats explicitly surfaced.
- Zero impact on the category engine / live app (no existing file touched; full suite stays green).
- The mapping is the single extensible point for widening stat coverage later.
