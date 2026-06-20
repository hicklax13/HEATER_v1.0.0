# Matchup-C — league scoreboard (`league[]` on `/api/matchup`)

**Goal:** Add the league scoreboard (all of the week's matchups) to `/api/matchup` so the CMO can
wire the Matchup-page sidebar. Additive, backward-compatible; `src/` engines unchanged.

**Scope:** small. Reuses existing service helpers (`_team_side`, `_build_categories`, `_cat_win`).
Coordination: the CMO finished wiring the current `/api/matchup` contract — this only ADDS `league[]`
(defaulted `[]`), so their shipped wiring can't break. Frontend shape (from `web/src/lib/matchup-data.ts`):
`league: { a: LeagueTeam; b: LeagueTeam }[]`, `LeagueTeam = {name, manager, record, score}` ≡ `TeamSide`.

## Contract (`api/contracts/matchup.py`)
```python
class LeagueMatchup(BaseModel):
    a: TeamSide = Field(default_factory=TeamSide)
    b: TeamSide = Field(default_factory=TeamSide)

# MatchupResponse += :
    league: list[LeagueMatchup] = Field(default_factory=list)
```
Reuse `TeamSide` (identical fields) — DRY; the CMO adapter maps `TeamSide`→their `LeagueTeam` by field name.

## Service (`api/services/matchup_service.py`)
- `_league(team_name, opponent, week, you_score, opp_score) -> list[LeagueMatchup]`:
  - `if not week: return []`
  - `pairings = load_league_schedule_full().get(week, [])` → `[(a, b), ...]` (all 6 for the week).
  - For each `(a, b)`: scores via `_pairing_scores(...)`, then `LeagueMatchup(a=_team_side(a, a_s), b=_team_side(b, b_s))`.
  - Includes the user's OWN pairing (CMO mock shows it as entry 0). never-raise → `[]`.
- `_pairing_scores(a, b, team_name, opponent, you_score, opp_score, week) -> (int, int)`:
  - If `{norm(a),norm(b)} == {norm(team_name),norm(opp)}` (use `src.auth._normalize_team_name`): return
    `(you_score,opp_score)` or `(opp_score,you_score)` mapped so the user's side aligns with a/b.
  - Else best-effort: `_score_from_cache(a, week)` → if found `(a_wins, a_losses)`; else try `b` swapped;
    else `(0, 0)`.
- `_score_from_cache(name, week) -> tuple[int,int] | None`:
  - `m = load_matchup_cache(name, week)`; if None or no `categories` → None.
  - `cats = _build_categories(m["categories"], name)`; set `c.win = _cat_win(c.you, c.opp, c.inverse)`;
    return `(count win=="you", count win=="opp")`. Wrap try/except → None.
- Wire into `get_matchup`: `league=self._league(team_name, opponent, week, you_score, opp_score)`.

## Tests (`tests/api/test_api_matchup_league.py`, DB-free)
- `_league` maps pairings → LeagueMatchup with name/manager/record (monkeypatch `load_league_schedule_full`,
  `load_league_records`, `get_connection` for manager, `load_matchup_cache`); user pairing uses you/opp scores;
  other pairing uses cache wins; cache-miss → 0/0.
- cold env (`load_league_schedule_full` raises / empty / week 0) → `league == []` (never raises).
- contract round-trip: endpoint returns `league[0].a.name/manager/record/score`.

## Verify
- `pytest tests/api/ -q` green; ruff; regen `api/openapi.json` (new `LeagueMatchup` schema + `league` field);
  live smoke with HEATER_DB_PATH (pairings empty locally if `league_schedule_full` unpopulated — document).

## Known limitation (documented, not a bug)
Other-team weekly scores depend on `sync_all_team_matchups` having cached each team's matchup; locally (and
until that scheduler phase runs) other pairings show score 0 with real pairings+records. Same env-dependent
degradation as the matchup roster tables / live-stats follow-up.
