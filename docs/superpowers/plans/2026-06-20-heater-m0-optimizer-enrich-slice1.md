# Optimizer contract enrich — slice 1: FIX + structural fields (M0)

**Goal:** Fix the broken `/api/lineup/optimize` and add the safe, standard-mode-derivable
structural fields. Defer the daily-mode rich fields (value/matchup/proj) + new-logic fields
(swaps/optimal-via-Yahoo) to slice 2. `src/` engines unchanged. Disjoint from the CMO's
Team/Trades wiring (Optimizer is held).

## CRITICAL BUG (Explore-found) — the endpoint returns empty even on Railway
`api/services/lineup_service.py` calls `pipeline.run()` — the real method is
`pipeline.optimize()`. `hasattr(pipeline,"run")` is False → `result=None` → empty slots ALWAYS.
AND it maps `result.lineup` as a LIST; the real shape is `result["lineup"]` =
`{assignments:[{slot,player_name,player_id}], bench:[names], projected_stats:{cat:float}, status}`
(from `lineup_optimizer.optimize_lineup`, confirmed src/lineup_optimizer.py:446; pipeline.py:621).

## Contract (`api/contracts/lineup.py`) — additive, backward-compatible
```python
class LineupSlot(BaseModel):          # existing + status
    slot: str
    player: PlayerRef
    action: str                       # "START" | "SIT" (kept)
    projected: float
    forced_start: bool = False
    reason: str | None = None
    status: str = "start"             # "start" | "bench"  (sit/off → daily-mode slice 2)

class CatImpact(BaseModel):           # new
    key: str
    proj: str
    trend: str = "flat"

class LineupOptimizeResponse(BaseModel):   # + bench / optimal / impact
    team_name: str
    date: str
    slots: list[LineupSlot]           # STARTERS (assignments)
    summary: str = ""
    bench: list[LineupSlot] = Field(default_factory=list)   # roster − starters
    optimal: bool = False
    impact: list[CatImpact] = Field(default_factory=list)   # projected lineup category totals
```

## Service (`api/services/lineup_service.py`)
- FIX: `result = pipeline.optimize()` (not `.run()`); read `lineup = result["lineup"]` (dict).
- `_to_slots(lineup, pool, roster)` → `(starters, bench)`:
  - starters: `lineup["assignments"]` → LineupSlot(status="start", action="START"); enrich via `player_ref_from_pool`.
  - bench: roster rows whose `player_id` NOT in the assignment id-set → LineupSlot(slot=selected_position or "BN", status="bench", action="SIT"). (player_id from roster → clean, no name lookup; the engine's `bench` list is names-only.)
- `_impact(projected_stats)` → `[CatImpact(key=CAT, proj=formatted, trend="flat")]` for cats present (rate cats 2-dp, counting ints). trend flat — no baseline to compute up/down (documented).
- `_optimal(roster, assignment_ids)` → bool: current starters = roster players whose `selected_position` is a real lineup slot (not BN/IL/NA); optimal = `set(current_starter_ids) == assignment_ids`. Roster missing selected_position → False (can't claim optimal).
- never-raise → empty response (existing behavior).

## DEFERRED → "Optimizer slice 2 (daily mode)" (documented)
per-slot `value` (0-100 DCV), `matchup` string, `proj` human line — all need
`mode="daily"` + `build_daily_dcv_table` which requires live Yahoo (confirmed lineups +
schedule) and can't be exercised in this env (same reason as the matchup live-stats
deferral). Plus `swaps` (no engine — needs a new bench-vs-starter diff module) +
`ip_pace`/`moves_left` ops counters (reuse the Team-ops helpers in slice 2).

## Tests (`tests/api/test_api_lineup.py`, DB-free)
- endpoint contract via fake-service dependency-override (round-trips slots + bench + optimal + impact).
- `_to_slots` with a synthetic `lineup` dict + roster df → starters from assignments, bench = roster−starters, statuses.
- `_impact` formats counting vs rate.
- `_optimal` True when current==optimized starter sets, False otherwise / no selected_position.
- service never-raises when the pipeline import/`optimize()` fails → empty response.

## Verify
- `pytest tests/api/ -q` green; ruff; regen openapi; live smoke (HEATER_DB_PATH) — empty locally
  (no roster) but the FIX means it WILL populate on Railway (verify the shape mapping via the synthetic test).
