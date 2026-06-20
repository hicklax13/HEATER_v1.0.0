# Team dashboard Slice 3 — `ops` cards (M0, completes the Team contract)

**Goal:** Add the 3 operational cards (IP pace, moves-left, roster-health) to `/api/me/team`,
completing the Team dashboard contract. Additive; `src/` unchanged. Disjoint from the CMO's
Compare/Trades wiring.

**Key refactor (perf):** lever (slice 2) builds `build_optimizer_context`; ops needs the same ctx
(`adds_remaining_this_week`, `roster`, `player_pool`). Hoist it to ONE `_build_ctx(team_name, cfg)`
call shared by both → one ~2-4s call instead of two. `_lever` becomes `_lever(ctx, cfg)`.

## Contract (`api/contracts/my_team.py`)
```python
class OpsCard(BaseModel):
    key: str        # "ip_pace" | "moves_left" | "roster_health"
    label: str
    value: float
    total: float
    verdict: str    # human-readable
    status: str = "ok"  # "ok" | "warn" | "danger"

# MyTeamResponse += ops: list[OpsCard] = Field(default_factory=list)
```

## Service (`api/services/team_service.py`)
- `_build_ctx(team_name, cfg) -> ctx | None` — hoisted build_optimizer_context (wrapped, logs on fail).
- `_lever(ctx, cfg)` — now takes ctx (guard `if ctx is None: return None`); no longer builds it.
- `_ops(ctx) -> list[OpsCard]` — assembles 3 cards (each wrapped; skip a card that fails):
  - **IP pace** `_ip_pace_card(ctx)`: roster_pitchers from `ctx.roster` ∩ `ctx.player_pool` (pitchers =
    is_hitter==0 / SP|RP in positions) as `{name, positions, ip, gs, status, is_starter}` →
    `compute_weekly_ip_projection(pitchers, get_days_remaining_in_week())`. OpsCard(value=projected_ip,
    total=ip_target, verdict=message, status= safe→ok/warning→warn/danger→danger).
  - **moves-left** `_moves_left_card(ctx)`: `value=ctx.adds_remaining_this_week`, total=10,
    verdict="{n} of 10 adds left this week", status= >2 ok / >0 warn / 0 danger.
  - **roster-health** `_roster_health_card(ctx)`: `il=_il_count(ctx.roster)` (reuse slice-1 helper),
    total=len(roster), value=total-il, verdict="{il} on IL"/"All active", status= il→warn else ok.
- `get_my_team`: `ctx=self._build_ctx(team_name,cfg)`; `lever=self._lever(ctx,cfg)`; `ops=self._ops(ctx)`.

## Tests (`tests/api/test_me_team.py`, DB-free)
- Refactor lever tests to pass `_FakeCtx` directly to `_lever(ctx, cfg)` (no longer monkeypatch
  build_optimizer_context for lever); add `_build_ctx` failure → None test; `_lever(None, cfg)` → None.
- ops: `_ops(_FakeCtx with roster+pool+adds_remaining)` → 3 cards; assert moves-left value/total,
  roster-health il count, ip-pace card present + status mapped; `_ops(None)` → [].
- contract round-trip: endpoint returns `ops[0].key` etc.
- `_FakeCtx` gains `.roster` + `.adds_remaining_this_week`.

## Verify
- `pytest tests/api/ -q` green; ruff; regen openapi (new OpsCard + ops field); live smoke (HEATER_DB_PATH)
  — ops degrades to thin/empty locally (roster/FA Yahoo-dependent), full on Railway, like movers/lever.

## After this
Team contract is COMPLETE (slices 1+2+3) → signal the CMO to wire the full flagship. Then the shared
playoff-odds endpoint (Standings + Team), per the CMO's recommendation.
