# Team dashboard Slice 2 — `lever` + structured stats (M0)

**Goal:** Add the Team dashboard `lever` (biggest category weakness + suggested FA pickups) to
`/api/me/team`, and convert stat lists to structured `StatItem{label,value}` per the CMO's contract ask.

**Scope:** additive contract change (lever) + a stat-shape change to the not-yet-wired `movers[].stats`
(safe — CMO is holding the Team wire). `src/` engines unchanged. Disjoint from the CMO's Standings/Trades wiring.

## Contract changes (`api/contracts/`)
1. **Move `StatItem{label,value}` to `api/contracts/common.py`** (shared) and re-import it in
   `free_agents.py` (`from api.contracts.common import StatItem`) so existing imports + the openapi
   schema name stay identical (no break to the CMO's free-agents-pool wiring).
2. `my_team.py`:
   - `Mover.stats: list[StatItem]` (was `list[str]`).
   - New `LeverPickup{player: PlayerRef, proj_stat: StatItem}`.
   - New `Lever{category_key: str, headline: str, behind_by: float, pickups: list[LeverPickup]}`.
   - `MyTeamResponse += lever: Lever | None = None`.

## Service (`api/services/team_service.py`)
- `_mover_stats` now returns `list[StatItem]` — `[StatItem(label="HR", value="18"), StatItem(label="AVG", value=".310")]` (hitters) / K·ERA (pitchers). Reuse `_fmt_avg` for AVG value.
- `_lever(team_name, cfg) -> Lever | None` — mirrors `fa_pool_service`:
  - `ctx = build_optimizer_context(scope="rest_of_season", yds=get_yahoo_data_service(), config=cfg, user_team_name=team_name, level_filter="MLB only")`.
  - `cat = min(ctx.category_gaps, key=ctx.category_gaps.get)` (most-negative gap = weakest); None if no gaps.
  - `behind_by = round(abs(gap), 1)`.
  - pickups: `rank_free_agents(ctx.user_roster_ids, ctx.free_agents, ctx.player_pool, cfg)` filtered to
    `best_category == cat`, top 3 → `LeverPickup(player=player_ref_from_pool(pid, ctx.player_pool, …),
    proj_stat=_cat_stat(pool_row, cat))`.
  - never-raise → None.
- `_cat_stat(pool_row, cat) -> StatItem`: `col = "ytd_"+cat.lower()`; rate cats (AVG/OBP/ERA/WHIP) via
  `format_stat`, counting via int; `StatItem(label=cat, value=…)`.
- Wire `lever=self._lever(team_name, cfg)` into `get_my_team`.
- Perf note: `_lever` adds one `build_optimizer_context` call (~2-4s, same as the Players page). Acceptable
  for a dashboard; a future lazy/async split is possible if needed.

## Tests (`tests/api/test_me_team.py`, DB-free)
- Update the slice-1 movers tests: `movers[].stats` is now `list[StatItem]` (assert `.label`/`.value`).
- `_lever` maps weakest cat + pickups (monkeypatch `build_optimizer_context` with a fake ctx
  [category_gaps, free_agents, player_pool, user_roster_ids] + `rank_free_agents` returning rows with
  `best_category`); assert `category_key`, `behind_by`, pickups filtered to the cat, `proj_stat` StatItem.
- `_lever` None on no gaps / cold env (build_optimizer_context raises).
- `_cat_stat` formats rate vs counting correctly.
- contract round-trip: endpoint returns `lever.category_key` + `lever.pickups[0].player.mlb_id` + structured `movers[].stats[0].label`.

## Verify
- `pytest tests/api/ -q` green; ruff; regen `api/openapi.json` (StatItem relocated [same name], Mover.stats
  type change, new Lever/LeverPickup + lever field); live smoke with HEATER_DB_PATH.
