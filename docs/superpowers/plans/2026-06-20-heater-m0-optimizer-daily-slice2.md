# Optimizer enrich slice 2 — daily mode (M0, the last M0 backend item)

**Goal:** Complete `POST /api/lineup/optimize` with the DAILY (today's DCV start/sit) path the
CMO's Optimizer page needs. Slice 1 fixed the standard path + added bench/optimal/impact; slice 2
adds the daily fields deferred there: per-slot DCV `value`, `matchup` string, start/sit decision,
day-level urgency/rate-modes/IP-pace/swaps. `src/` UNCHANGED (engine is frozen).

## Why this couldn't be verified end-to-end locally
Daily mode needs live Yahoo (rosters/matchup) + today's MLB schedule. The worktree DB is empty and
there's no Yahoo, so the real `daily_dcv` is empty locally. Mitigation: **all mapping logic is unit-
tested DB-free with synthetic `daily_dcv`/`daily_lineup`** (the part the API owns), and the daily path
is proven to run end-to-end without crashing (graceful empty). The ENGINE is proven (the Streamlit
Daily tab uses it). Live field-population is verified on Railway when the CMO wires the page — same
deferral pattern as Matchup live-stats / Team ops.

## Request
`LineupOptimizeRequest.mode: str = "standard"` — `"daily"` switches to the daily path. Default
unchanged ⇒ standard mode is byte-for-byte backward-compatible.

## Contract additions (`api/contracts/lineup.py`, all additive)
- `LineupSlot`: `value` (0-100 heat = normalized DCV, best play today = 100), `matchup` ("vs SF" /
  "@ COL"), `current_slot` (player's current Yahoo slot → frontend diffs swaps).
- `IpPace{projected, target, pace_pct, status, message}`.
- `Swap{player: PlayerRef, slot, value}`.
- `DailyMeta{urgency{cat:0-1}, rate_modes{ERA/WHIP→protect|compete|abandon}, winning/losing/tied[],
  ip_pace, recommendations[], swaps[]}`.
- `LineupOptimizeResponse`: `mode` (echo) + `daily: DailyMeta | None` (daily mode only).

## Service (`api/services/lineup_service.py`)
- `optimize(..., mode="standard")` branches → `_optimize_standard` (unchanged) | `_optimize_daily`.
- `_optimize_daily`: yds rosters + matchup + `statsapi.schedule(date)` → `LineupOptimizerPipeline(
  roster, mode="daily").optimize(matchup=, schedule_today=)` → maps `daily_dcv` + `daily_lineup`.
  - **Robust join (the key design point):** PlayerRef comes from `daily_dcv` (has `player_id`); the LP
    start/slot decision comes from `daily_lineup` (name-keyed ONLY — the engine's return drops
    player_id and I can't add it). Joined by `(normalized_name, round(total_dcv, 4))` — total_dcv is a
    per-player near-unique float, so same-name players don't collide (the Muncy-DNA lesson). player_id
    is never sourced from the name.
  - `value` normalized to the best active DCV (×100). `forced_start` when started with matchup_mult <
    0.70 or total_dcv ≤ 0. `reason` (LOCKED/IL/OFF_DAY) passed through from the engine.
- `_daily_meta`: urgency (NaN-dropped via a nan sentinel), rate_modes, winning/losing/tied,
  recommendations, swaps (started players whose current_slot is benchy), ip_pace.
- `_ip_pace`: roster pitchers (positions ∋ SP/RP/P) + season IP merged from the pool →
  `ip_tracker.compute_weekly_ip_projection`; None when no pitchers/pool.
- helpers: `_matchup_str`, `_schedule_today`, `_current_slots`, `_days_remaining` (Mon-Sun week),
  `_norm`, `_round_dcv`, `_f`. Every path never-raises → empty/None.

## DEFERRED (documented degradation, follow-ups — NOT bugs)
- Richer daily inputs NOT yet assembled: `confirmed_lineups` (PA-order volume), `recent_form` (L14
  blend), `team_strength` (opp wRC+), `park_factors`. Absent ⇒ the engine degrades gracefully
  (hitters 0.9 volume, neutral matchup). v1 passes matchup + schedule_today (gives the SP probable-
  gate + locked games + opponent strings). The Streamlit page is the full-fidelity reference.
- `moves_left` lives on `/api/me/team` (ops) — not duplicated here.

## Verify
220 api tests (+7 daily, DB-free synthetic); ruff; openapi regenerated; live smoke: daily path runs
end-to-end with no Yahoo → graceful empty (mode=daily, no crash). Code-review the join + NaN/never-
raise. Railway field-population check when the CMO wires the Optimizer page.
