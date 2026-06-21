# HEATER Probable Pitcher + Hitter Matchup Schedule — Design Spec

**Date:** 2026-06-20
**Lane:** CEO — **full-stack on this feature** (backend `api/` + `src/` AND the frontend `web/`), owner-directed 2026-06-20. The grid lands as a **new tab on the Pitcher Streaming page** — shared with the CMO's streaming page, reconciled at build time.
**Status:** design. ~85% composes existing engine; one new scorer. Two phases.

## Vision

A FantasyPros-style **7-day matchup grid** (model: `https://www.fantasypros.com/mlb/probable-pitchers.php`) as a new **Pitcher Streaming tab**, in two linked views:
1. **Probable Pitchers** — TEAM rows × next-7-days columns; each cell = the probable SP (name, opponent `@SF`/`vs`, home/away, **matchup difficulty**, **two-start** flag), **league-connected** (rostered-by-your-team / taken-by-another / available) with filters (Home / Away / Easy / Tough / Two-start / availability).
2. **Hitter Matchups** (the inverse) — TEAM rows showing each team's **batting squad's** weekly matchup: the opposing probable SP per day (+ L/R handedness), a per-team totals strip (games, vs RHP, vs LHP) and a **matchups-rank**, color-coded easy/tough, same league-connection + filters.

## Architecture — compose the existing engine

From the engine map (2026-06-20 exploration), the plumbing already exists:

| Capability | Reuse |
|---|---|
| 7-day probables + per-pitcher data | loop `src/optimizer/stream_analyzer.py::build_stream_board(ctx, target_date, schedule, include_rostered=True)` over the 7 dates (rows carry name, mlb_id, team, opponent, is_home, **stream_score** 0-100, **num_starts**, status, confidence) |
| matchup difficulty | the `stream_score` already blends `two_start.compute_pitcher_matchup_score` (opp wRC+ / K% / park / home-away) → easy/tough |
| two-start | `num_starts` (from `ctx.two_start_pitchers`) |
| league roster tagging | `src/database.load_league_rosters()` (player_id → team_name) + `ctx.league_rostered_ids` / `ctx.user_roster_ids` |
| free-agent = available | a probable's mlb_id NOT in `league_rostered_ids` |
| hitter / pitcher handedness | the pool's `bats` / `throws` columns |
| opposing SP per game | `statsapi.schedule` + `src/game_day.fetch_opposing_pitchers` (resolves SP + L/R hand) |

**The ONE genuinely new engine piece:** `compute_hitter_matchup_score(team_hitters, opp_sp_stats, is_home, park_factor)` — the inverse lens of the pitcher scorer (a lineup's offensive rating + platoon edge vs the opposing SP's quality). Lives beside the existing scorer (`src/optimizer/stream_analyzer.py` or a sibling), reusing `get_opponent_offense_context` + the pool's `bats`/`throws`. Everything else is composition + a **league-tag layer** added in the service.

### Endpoints (read; **free / ungated**, consistent with the existing `/api/streaming`)
- `GET /api/schedule/probables?days=7` → `ProbableGridResponse` — `teams[]` × `days[]`, each cell `{pitcher: PlayerRef|None, opponent, is_home, difficulty (0-100 + easy/tough band), two_start, availability: "yours"|"taken"|"available", rostered_by?}`. NaN-safe, never-raise → empty.
- `GET /api/schedule/hitter-matchups?days=7` → `HitterMatchupGridResponse` — `teams[]` × `days[]`, each cell `{opp_sp: PlayerRef|None, opp_sp_throws: "L"|"R", difficulty, is_home}` + per-team `totals {games, vs_rhp, vs_lhp}` + `matchups_rank`.
- Service composes the engine + the league-tag layer; the ONE src-importing seam. Thin logic-free router (AST-guarded). Contracts in `api/contracts/schedule.py`, reusing `PlayerRef`.

### Frontend (my lane for this feature)
A new **Pitcher Streaming tab** rendering the grid (TEAM rows × day columns) with the filter chips (Home / Away / Easy / Tough / Two-start) + the availability toggles (Roster / Taken / Available), color-coded difficulty (the `heatColor` ramp), two-start highlights, and the linked Pitcher↔Hitter view switch. Player cells open the canonical `PlayerDialog`. Built in `web/` (Combustion design); coordinate the tab seam with the CMO's streaming page.

## Phasing
- **Phase P1 — Probable Pitcher grid:** the `/api/schedule/probables` endpoint (compose `build_stream_board` ×7 + league-tag layer) + the React grid tab + filters. *Outcome: the pitcher grid, league-connected + filterable.*
- **Phase P2 — Hitter Matchup grid:** the new `compute_hitter_matchup_score` + the `/api/schedule/hitter-matchups` endpoint + the inverse grid view + the totals/rank strip.

## Error handling / testing
Never-raise → empty grid (cold env). DB-free service tests with synthetic `build_stream_board` output + a fake league-roster map (verify the availability tagging + the difficulty band + two-start). The new hitter scorer gets unit tests against the REAL `get_opponent_offense_context` shape (the live-only mapping must be verified against the engine, per the M0 lesson). Frontend verified via preview at build time.

## Risks
- **CMO tab seam** — the Pitcher Streaming page is the CMO's; the new tab integration is the coordination point.
- **W-L record / SP-rank cells** (the FantasyPros "(5-3) SP 126") are best-effort from the pool (`ytd_w/l`, a derived SP rank among pitchers) — graceful defaults; the core value (difficulty + league tag + two-start) doesn't depend on them.
- **Cost** — a 7-day × all-teams board is heavier than the single-date streaming board; cache per (date-range) with a short TTL (the streaming board is already cacheable).

## Index
- Engine: `src/optimizer/stream_analyzer.py` (board/week-plan), `src/two_start.py` (matchup score), `src/game_day.py` (opposing pitchers, team strength).
- Existing streaming API (contract pieces to reuse): `api/contracts/streaming.py`, `api/services/streaming_service.py`.
- Roadmap entry: `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md` (Planned features).
