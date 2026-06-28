# HEATER Roto Valuation — Design (Phase 4, slice 2)

**Date:** 2026-06-27
**Status:** Approved design (brainstorming complete 2026-06-27) — pending owner review of this written spec, then writing-plans.
**Owner:** Connor
**Program:** Public commercial launch, Phase 4 (League-format generalization). Second standalone slice of Phase 4 — the **Rotisserie** scoring paradigm. Buildable now; no dependency on Phase 2 (Postgres) or Phase 3 (tenant model). Companion to slice 1 (Points, `src/points_scoring.py`).

---

## 1. Purpose

HEATER fully supports H2H **categories** and now **points** (slice 1). The "all formats" goal (owner-locked decision #3) also requires **Rotisserie** ("roto"), where teams are ranked 1..N in each category and a team's score is the sum of those category ranks. This slice delivers the **standalone Roto valuation core**: project roto standings from team category totals, value a player in roto terms, and compute a player's **marginal** roto contribution to a specific team. It is the foundation later slices wire into the optimizer / FA / trade / standings for roto leagues.

Roto is the one format with **zero** valuation today. Unlike points (a brand-new paradigm), roto is *categories scored by rank*, so this slice legitimately **reuses** the proven category primitives (`SGPCalculator`, `LeagueConfig`) rather than avoiding them — SGP (Standings Gain Points) is literally the per-player roto value.

## 2. Locked decisions (brainstorming, 2026-06-27)

1. **Roto after Points** (points was the bigger net-new paradigm; roto reuses categories).
2. **Reuse the category engine, don't reinvent it.** `SGPCalculator.total_sgp` is the per-player roto value; `LeagueConfig` supplies categories + inverse stats. The net-new piece is rank-sum **standings** — no such function exists today (`standings_engine` only does H2H simulation).
3. **`compute_roto_standings` is exact** over each team's per-category value (counting sums and pre-computed rates), with standard roto **tie-averaging** and **inverse-aware** ranking.
4. **`marginal_roto_value` is standings-aware and exact** — it re-ranks after adding the player, recomputing rate categories from component totals (built from the pool), so a contested category is rewarded and a locked one isn't.
5. **Standalone valuation layer only** — no optimizer/FA/trade/standings-page wiring (later slices; the parallel CMO track owns the integration layer).

## 3. Architecture

A single new module `src/roto_scoring.py`. Pure functions over the enriched player pool (`load_player_pool()`) and team category totals (`standings_utils.get_all_team_totals()` shape). No global state; no DB writes; never raises.

```
compute_roto_standings(team_totals, config)            ──► DataFrame [team, roto_score, rank, pts_<cat>...], sorted
player_roto_value(player_row, config)                  ──► float   (= SGPCalculator.total_sgp)
rank_players_by_roto_value(pool, config)               ──► DataFrame (+ "roto_value" column, sorted)
marginal_roto_value(player_id, my_team, my_roster_ids,
                    all_team_totals, pool, config)      ──► float   (Δ my roto_score from adding the player)
```

### 3.1 Roto standings — the rank-sum algorithm (the crux)

Input `team_totals` = mapping `team_name -> {category -> value}` for all `num_teams` teams (counting categories as season/projected totals; rate categories as the already-aggregated team rate). For each category in `config.all_categories`:

- **Direction:** normal categories — higher value is better; **inverse** categories (`config.inverse_stats` = {L, ERA, WHIP}) — lower value is better.
- **Award points:** best team gets `N` (= `config.num_teams`), worst gets `1`.
- **Ties** (standard roto): tied teams each receive the **mean** of the point-slots they jointly occupy (e.g., two teams tied for the top two slots in a 12-team league each get (12+11)/2 = 11.5). Implemented via average-rank (`scipy.stats.rankdata(method="average")` on the values, negated for inverse cats so "higher rank = better" holds uniformly).

A team's **roto score** = sum of its category points across all categories (max = `N × len(all_categories)`). Standings = teams sorted by roto score descending; total-ties broken deterministically (by team name) for stable output. The returned DataFrame carries the per-category points (`pts_<cat>`) so the breakdown is inspectable.

### 3.2 Per-player roto value

`player_roto_value(player_row, config)` returns `SGPCalculator(config).total_sgp(player_row)` — the league-average standings-gain value, the canonical context-free roto valuation. `rank_players_by_roto_value(pool, config)` adds a `roto_value` column (via `SGPCalculator.total_sgp_batch` for speed) and returns the pool sorted descending. Pure (returns a copy; never mutates input).

### 3.3 Marginal roto value (standings-aware, rate-correct)

`marginal_roto_value(player_id, my_team_name, my_roster_ids, all_team_totals, pool, config)`:

1. Build **my component totals** by summing my roster's component columns from `pool` (hitters: `r/hr/rbi/sb/h/bb/hbp/ab/sf`; pitchers: `w/l/sv/k/ip/er/bb_allowed/h_allowed`). Derive my per-cat values: counting = sum; rate = canonical aggregation (AVG = Σh/Σab, OBP = Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf), ERA = Σer·9/Σip, WHIP = Σ(bb+h)/Σip).
2. `base` = my roto score from `compute_roto_standings(all_team_totals with my team = my pool-derived values, config)`.
3. Add the player's component columns → recompute my per-cat values (counting added; **rates recomputed from components — never naive rate addition**).
4. `after` = my roto score from `compute_roto_standings(all_team_totals with my team = the new values, config)`.
5. Return `after − base`.

This is exact: only my team's totals change, other teams' values are used as-is for ranking, and `base`/`after` share the same pool-derived construction so the delta is purely the player's effect. It captures the contested-vs-locked effect a flat SGP can't. (For a two-way player, both the hitting and pitching component sets are added.)

### 3.4 Inputs / adapters

- `team_totals` shape is **verified against `standings_utils.get_all_team_totals()` at implementation time** (slice-1's "verify columns at impl" practice). A small internal adapter normalizes it to `{team: {cat: value}}` if the live shape differs.
- `config` defaults to `LeagueConfig()`; any league's categories / inverse stats / num_teams flow through it.

### 3.5 Edge cases

- **Ties** → average points (3.1). **Inverse cats** → direction flipped. **Rate cats** → ranked on the team rate; recomputed from components in `marginal_roto_value`.
- **Missing team / fewer than N teams present** → ranks over whatever teams exist (points scale to the count); documented, never raises.
- **Unknown `player_id` / empty roster / empty pool** → marginal value `0.0`; standings over empty input → empty DataFrame.
- **NaN/None** values coerce to a finite `0.0` (slice-1 `_num` style). Rate recomputation guards divide-by-zero (0 AB/IP → 0.0).

## 4. Error handling

NaN/None-safe; never raises into callers (bad rows, missing columns, empty inputs degrade to `0.0` / empty). Mirrors slice-1's honesty: nothing is silently wrong; degenerate inputs produce explainable neutral outputs.

## 5. Testing (TDD)

- `compute_roto_standings`: a 3–4 team fixture → correct per-cat points and totals; best team in a normal cat gets `N`, worst gets `1`; **inverse cat** (ERA) gives the lowest-ERA team `N`; **ties** split points (two tied teams get the averaged slot points); total = sum of category points; sorted descending with a `rank` column.
- `player_roto_value` equals `SGPCalculator.total_sgp` for a fixture player; `rank_players_by_roto_value` is descending, input unmutated, `roto_value` present.
- `marginal_roto_value`: adding a strong player to my team **raises** my roto score; a zero-stat player → `0.0` delta; a HR-heavy player gains **more** roto points when I'm **contested** in HR (close to the team above me) than when I'm **locked** (far ahead) — the standings-aware property; rate cats recompute correctly (a high-OBP, low-AB player moves OBP via components, not naively).
- NaN/missing rows → `0.0`, no raise; empty pool/roster/team_totals → empty / `0.0`.

## 6. File structure

| File | Responsibility | Action |
|---|---|---|
| `src/roto_scoring.py` | roto standings, per-player roto value, ranking, marginal roto value | Create |
| `tests/test_roto_scoring.py` | TDD coverage of §5 | Create |

No existing file is modified. (`SGPCalculator` / `LeagueConfig` are imported read-only; `standings_engine` / `standings_utils` untouched.)

## 7. Out of scope (later Phase 4 / other phases)

- Wiring roto into the **optimizer** (roto-maximizing start/sit), **FA**, **trade**, **draft**, and the **standings page** (roto standings UI).
- **Roto value over replacement** (positional scarcity in roto terms) — its own concern; SGP-based category VORP already exists and could generalize later.
- Sourcing roto category lists / settings from a **connected league** (Phase 5 connectors).
- The `LeagueConfig` **`scoring_format` generalization** + a unified standings abstraction across categories/points/roto (later Phase 4).
- A point-in-time roto **projection backtest** (Phase 7 validation).

## 8. Success criteria

- For any league config + team totals, `compute_roto_standings` returns correct rank-sum standings (inverse-aware, tie-averaged); `player_roto_value` / `rank_players_by_roto_value` value the pool in roto terms; `marginal_roto_value` gives the exact standings-aware, rate-correct delta of adding a player to a team.
- Zero impact on the category engine / live app (no existing file touched; full suite green).
- A `P4-ROTO-ENGINE` evidence-registry row (added at build time by the plan) tracks it.
