# Playoff-odds endpoint — `GET /api/playoff-odds` (M0, shared: Standings panel + Team)

**Goal:** A per-team playoff-odds + projected-standings endpoint (the CMO's Standings odds
panel + a forward-looking element on the Team page). Composes the existing, structurally-guarded
`simulate_season_enhanced` engine — `src/` unchanged. Disjoint from the CMO's Team/Trades wiring
(brand-new endpoint; touches no existing one).

**Engine (proven path, mirrors the Streamlit Season Projections tab):**
`standings_utils.get_all_team_totals(league_rosters, pool)` → divide counting stats by 26 →
`standings_engine.simulate_season_enhanced(current_standings, team_weekly_totals, full_schedule,
current_week, n_sims)` → per-team `playoff_probability` + `projected_records`. ONE MC run.

**Scope honesty:** per-team PLAYOFF odds ARE engine-derivable (this). Per-team **CHAMP odds are
NOT** — no engine computes them (only the Hickey-centric `engine/output/playoff_sim.simulate_playoff_outcomes`
gives the *user's* champ%, and mixing a 2nd heavy sim into a read endpoint is a perf cost). **CHAMP
is DEFERRED** (documented follow-up: user-champ via the Hickey sim, or an all-teams bracket sim).

## Contract (`api/contracts/playoff.py`)
```python
class PlayoffTeam(BaseModel):
    team: str
    playoff_odds: float = 0.0     # 0-100
    projected_wins: float = 0.0
    projected_record: str = ""    # "W-L-T"
    current_wins: int = 0
    rank: int = 0                 # projected finish (1 = best playoff odds)
    in_cut: bool = False          # projected top-`playoff_spots`
    is_user: bool = False

class PlayoffOddsResponse(BaseModel):
    team_name: str
    playoff_spots: int = 4
    you: PlayoffTeam | None = None
    league: list[PlayoffTeam] = Field(default_factory=list)  # sorted by playoff_odds desc
    n_sims: int = 0
```
(No `manager`/`record` beyond projected — the frontend joins by `team` name with its standings rows.)

## Service (`api/services/playoff_service.py`) — the ONE place importing src/
- `get_playoff_odds(team_name, n_sims=4000) -> PlayoffOddsResponse`:
  - `_team_weekly_totals()`: rosters via `load_league_rosters()` grouped → `{team:[pid]}`; `pool=load_player_pool()` (rename name→player_name); `get_all_team_totals(rosters, pool)`; counting cats /26, rate cats passthrough (cats from `LeagueConfig`). `{}` if empty.
  - `current_standings` from `load_league_records()` → `{team:{W,L,T}}`; `current_wins` map.
  - `full_schedule = load_league_schedule_full()`.
  - `current_week` from `league_rules.weeks_remaining` proxy (or `1`).
  - if no totals/standings → return empty response (cold env).
  - `sim = simulate_season_enhanced(current_standings, team_weekly_totals, full_schedule, current_week, n_sims=n_sims, playoff_spots=4)`.
  - map: for each team → `PlayoffTeam(team, playoff_odds=round(prob*100,1), projected_wins=rec.W, projected_record="W-L-T", current_wins, is_user=norm-match)`; sort by playoff_odds desc; assign `rank=i+1`, `in_cut=rank<=4`. `you` = the is_user row.
  - never-raise → `PlayoffOddsResponse(team_name=team_name)`.
  - `n_sims` lower than the Streamlit 10k default for API responsiveness (4000 ≈ ~0.8% SE — fine for a dashboard); tune after a timing check.
- Reuse `src.auth._normalize_team_name` for the is_user / "you" match (emoji-prefixed Yahoo names).

## Router / wiring
- `api/routers/playoff.py`: thin `GET /api/playoff-odds?team_name=...` → `service.get_playoff_odds(team_name)`. Logic-free (AST-guarded).
- `api/deps.py`: `get_playoff_service()` provider.
- `tests/api/test_api_playoff.py`: fake-service dependency-override (round-trips you + league + sorted + in_cut).
- Mount in `api/main.py`; regen `api/openapi.json`.

## Tests (DB-free)
- Service unit (monkeypatch `get_all_team_totals` / `load_league_records` / `load_league_schedule_full` / `simulate_season_enhanced` at source): assert weekly /26 conversion, sort-by-odds + rank + in_cut, is_user/you match (incl. emoji-name normalization), cold-env → empty.
- Contract round-trip via the fake-service endpoint test.

## Verify
- `pytest tests/api/ -q` green; ruff; regen openapi; live smoke (HEATER_DB_PATH) — empty locally (league_rosters unpopulated), real on Railway (documented degradation). Timing check → tune n_sims.

## Follow-up (documented, not built)
Per-team CHAMP odds — needs an all-teams bracket sim, or expose the user's champ% via the
Hickey-centric `simulate_playoff_outcomes` as a 2nd call/endpoint. Flag to the CMO: render
`playoff_odds` now; `champOdds` deferred (show "—"/hide).
