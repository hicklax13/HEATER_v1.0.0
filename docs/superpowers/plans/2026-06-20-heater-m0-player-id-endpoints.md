# Player-id endpoints — search + league-rosters (M0)

**Goal:** Give the frontend a HEATER-player-id source for the Trades build-a-trade evaluator
(`/api/trade/evaluate` needs `giving_ids`/`receiving_ids`) + compare-any-player. Two brand-new
read endpoints, both reusing `PlayerRef`. `src/` unchanged. Disjoint from the CMO's Team/Trades
wiring (the CMO formally flagged these).

## Endpoints (22nd + 23rd)
1. `GET /api/players/search?q=<str>&limit=25` → `PlayerSearchResponse{query, results: list[PlayerRef]}`.
   Pool-wide name search (DB-backed → REAL locally). Powers compare-any-player + "add any player"
   in the trade builder. Returns ANY player's id (rostered or FA).
2. `GET /api/league/rosters` → `LeagueRostersResponse{teams: list[LeagueRosterTeam{team_name, manager, players: list[PlayerRef]}]}`.
   All teams' rosters with ids. Powers the trade builder's "receive from team X" picker.
   Yahoo-dependent (empty locally, real on Railway).

## Contract (`api/contracts/players.py`)
```python
class PlayerSearchResponse(BaseModel):
    query: str
    results: list[PlayerRef] = Field(default_factory=list)

class LeagueRosterTeam(BaseModel):
    team_name: str
    manager: str = ""
    players: list[PlayerRef] = Field(default_factory=list)

class LeagueRostersResponse(BaseModel):
    teams: list[LeagueRosterTeam] = Field(default_factory=list)
```

## Service (`api/services/roster_query_service.py`) — ONE place importing src/
- `search(q, limit=25)`:
  - `q.strip()` empty → `PlayerSearchResponse(query=q, results=[])`.
  - `load_player_pool()`; filter rows whose `name` contains q (case-insensitive); sort by
    `consensus_rank` asc (NaN last → most fantasy-relevant matches first); head(limit).
  - map each row → `make_player_ref(id=player_id, name, positions, mlb_id, team_abbr=team)`.
  - never-raise → empty.
- `league_rosters()`:
  - `load_league_rosters()` → group by `team_name`; `load_player_pool()` for enrichment;
    manager per team from `league_teams` (one SELECT, like matchup `_team_side`).
  - per team: `players = [player_ref_from_pool(pid, pool, name=row.name, positions=row.positions)]`.
  - never-raise → empty.

## Wiring
- `api/routers/players.py`: thin `GET /api/players/search` + `GET /api/league/rosters` (logic-free, AST-guarded).
- `api/deps.py`: `get_roster_query_service()`.
- mount in `api/main.py`; `tests/api/test_api_players.py` fake-service + service unit tests; regen openapi.

## Tests (DB-free)
- endpoint contract (fake-service dependency-override) for both routes.
- `search`: monkeypatch `load_player_pool` → name filter + consensus_rank sort + limit + enrichment; empty q → [].
- `league_rosters`: monkeypatch `load_league_rosters`/`load_player_pool`/`get_connection` → grouping + PlayerRef per team; empty → [].

## Verify
- `pytest tests/api/ -q` green; ruff; regen openapi; live smoke (HEATER_DB_PATH): search is REAL
  locally (pool DB-backed — e.g. search "trout" → Mike Trout's id+mlb_id); league-rosters empty
  locally (Yahoo), real on Railway.
