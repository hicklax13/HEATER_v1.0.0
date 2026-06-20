# Team dashboard — `GET /api/me/team` extension (M0)

> REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Steps use `- [ ]`.

**Goal:** Extend the minimal `MyTeamResponse` toward the frontend Team-page dashboard
(gap spec §1), building only what's ENGINE-DERIVABLE, deferring fields with no backing data.

**Architecture:** Additive fields on `MyTeamResponse` (all `Optional`/defaulted → backward-
compatible; openapi regenerated). All engine calls stay in `api/services/team_service.py`
(the ONE place importing `src/`). Resilient: any missing live data → empty/None, never raises.

**Engine grounding (Explore, 2026-06-20):** movers=`src.trend_tracker.compute_player_trends`;
lever cat=existing matchup categories (lowest win_prob) + pickups=`src.in_season.rank_free_agents`
(returns `best_category`); ops=`src.ip_tracker.compute_weekly_ip_projection` +
`src.league_rules` txn helpers + `src.alerts.get_il_stash_names`; freshness=`get_refresh_log_snapshot`;
playoff_cut_rank=`_PLAYOFF_SPOTS`=4. **trajectory + win_prob_trend = DEFERRED** (no per-week
snapshot table exists — needs a cron-written history table, out of M0 lean-infra scope).

---

## Slicing
- **Slice 1 (THIS plan): movers + dashboard scalars + status_chips.** Clean, no FA-engine dependency.
- **Slice 2: lever** (weakest category + FA pickups via rank_free_agents). Isolates FA complexity.
- **Slice 3: ops** (IP pace, moves left, IL/roster health). Isolates current-week txn-boundary logic.
- **Deferred (documented): trajectory, win_prob_trend, lineup-status chip** — need a per-week
  snapshot table + cron (future B3-workers slice). Matchup-style honest deferral.

---

## Slice 1 — movers + scalars + chips

### New contract models (`api/contracts/my_team.py`)

```python
class Mover(BaseModel):
    player: PlayerRef           # from api.contracts.common
    stats: list[str]            # up to 2 display stats, e.g. ["18 HR", ".322 AVG"]
    trend: str                  # "up" | "down"
    tag: str                    # "hot" | "cold"
    context: str                # short note, e.g. "Trending hot vs projection"
    rostered_by_you: bool = True


class StatusChip(BaseModel):
    label: str                  # "IL", "News"
    value: int                  # count
    status: str                 # "ok" | "warn" | "info"


class MyTeamResponse(BaseModel):
    team_name: str
    record: str
    rank: int
    matchup: MatchupHero | None
    categories: list[CategoryLine]
    # ── Slice 1 additions (all defaulted → backward-compatible) ──
    eyebrow: str = ""
    subline: str = ""
    freshness_minutes: float | None = None
    playoff_cut_rank: int = 4
    status_chips: list[StatusChip] = Field(default_factory=list)
    movers: list[Mover] = Field(default_factory=list)
    movers_scope: str = "mine"
```

### Service additions (`api/services/team_service.py`)

- `_roster_ids(team_name)` → user player_ids via `load_league_rosters()` filtered to team_name.
- `_movers(team_name, cfg)` → `compute_player_trends(pool, season_stats)`, filter to roster_ids,
  keep HOT+COLD, sort by abs(trend_delta) desc, top 4. Each → `Mover` via `player_ref_from_pool`
  + 2 ytd stats (reuse a small stat formatter) + trend/tag from trend_label.
- `_status_chips(team_name)` → IL count (`get_il_stash_names` ∩ roster, or league_rosters status
  filter) + news count (`player_news` query for roster ids). Each → `StatusChip`.
- `_freshness_minutes()` → **max** (stalest) age in minutes across core sources from
  `get_refresh_log_snapshot()` (`yahoo_standings`/`season_stats`/`yahoo_rosters`); clamped ≥0;
  None if unavailable. (Stalest, not freshest — the dashboard is only as fresh as its oldest input.)
- `_eyebrow_subline(team_name, rank, record, week, n_teams)` → display strings.
- `playoff_cut_rank` = `_PLAYOFF_SPOTS` (import from playoff_sim) with a literal `4` fallback.
- All wrapped try/except → degrade to empty/None.

### Tasks

- [ ] **T1: contract models** — add `Mover`, `StatusChip`, extend `MyTeamResponse`. Import `PlayerRef`.
- [ ] **T2: contract test** — `tests/api/test_api_my_team.py`: fake service returns the new fields;
  assert the endpoint round-trips `movers[].player.mlb_id`, `status_chips`, `eyebrow`, `freshness_minutes`.
- [ ] **T3: service helpers** — implement `_movers`, `_status_chips`, `_freshness_minutes`,
  `_eyebrow_subline`, playoff_cut_rank; wire into `get_my_team`. Each helper static + resilient.
- [ ] **T4: service unit tests (DB-free)** — monkeypatch `src.trend_tracker.compute_player_trends`,
  `src.database.load_league_rosters`/`load_player_pool`/`get_refresh_log_snapshot` with synthetic
  frames; assert `_movers` filters to roster + caps at 4 + maps trend→tag; `_freshness_minutes`
  picks min age; `_status_chips` counts IL. **No live DB** (worktree/CI DB is empty — see
  reference_worktree_empty_db memory; set HEATER_DB_PATH only for manual live checks).
- [ ] **T5: regen openapi** — `python scripts/export_openapi.py` in the SHARED venv (fastapi
  0.137.1); the contract changed so openapi.json WILL diff (new schemas) — commit it.
- [ ] **T6: verify** — `pytest tests/api/ -q` green; ruff clean; live smoke with HEATER_DB_PATH.

### Acceptance
- `MyTeamResponse` carries movers + scalars + chips; existing fields unchanged.
- Backward-compatible (all new fields defaulted) — existing frontend wiring unaffected.
- DB-free tests pass in worktree + CI; live smoke shows ≥0 movers without raising.

---

## Slice 2 — lever (next)
`lever: Lever | None` where `Lever{category_key, headline, behind_by, pickups: list[LeverPickup]}`,
`LeverPickup{player: PlayerRef, proj_stat: str}`. Category = weakest matchup category (lowest
win_prob, inverse-aware). Pickups = `rank_free_agents(roster_ids, fa_pool, pool, cfg)` filtered to
`best_category == category_key`, top 3. Graceful None when no live matchup/FA pool.

## Slice 3 — ops (next)
`ops: list[OpsCard]` (3): IP pace (`compute_weekly_ip_projection`), moves left
(`league_rules` txn helpers + current-week boundary from schedule), roster health (IL/active counts).
Each `OpsCard{key, value, total, verdict, status}`.

## Deferred (documented — NOT built)
- `trajectory` (week-by-week rank history) + `win_prob_trend` — require a new per-week snapshot
  table + a cron writer (no such storage exists). Future B3-workers slice. Like the matchup
  live-stats deferral, the contract can add these later without breaking changes.
- `lineup_status` chip — no engine signal for "lineup set/not set".
