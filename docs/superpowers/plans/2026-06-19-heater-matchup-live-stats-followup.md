# Matchup live-stats integration — follow-up plan (DEFERRED)

**Date:** 2026-06-19
**Status:** PLAN ONLY — not yet built. Owner-aware (chosen "build core + plan live-stats"). A follow-up to the Matchup-core slice (`2026-06-19-heater-m0-matchup-rosters.md`).
**Lane:** CEO / backend (`src/` + `api/`).

## Why this exists

The Matchup-core slice builds the roster-comparison tables with **projected** stat lines + game **state** (sched/live/final from `statsapi.schedule()`). The frontend Matchup page's mock shows **live** per-player lines ("0-5, Top 3rd", "Final (L) 2-4 @ HOU") and per-batter opposing pitcher. Those require a **live box-score feed the backend does not have today**. This plan adds it.

## What's missing (the deferred fields on `MatchPlayer`)

- `stats[]` — currently **projected** season line; should be **today's actual** line (H/AB, R, HR, RBI, SB for hitters; IP, K, ER… for pitchers) while a game is live/final.
- `status` — currently a basic game status ("In Progress" / "Final" / scheduled time); should be a play-by-play string ("Top 3rd · 0-5 vs SF", "Final (L) 2-4 @ HOU").
- opposing pitcher per batter (a nice-to-have for the status string).

## The data source

**MLB Stats API live game feed** (already used elsewhere via `statsapi`):
- `statsapi.schedule(date=…)` → game_ids + status (already used for game-state in the core).
- **NEW:** `statsapi.boxscore_data(game_id)` / `statsapi.get("game", {...})` `/api/v1/game/{game_id}/boxscore` + `/linescore` → per-player batting/pitching lines + the current inning/half + score. This is the missing piece.

## Proposed shape (a future slice)

1. **New engine helper** `src/game_day.py::fetch_live_player_lines(date) -> dict[mlb_id, {stats, status, state}]` (or a `src/live_boxscore.py` module): for each in-progress/final game on the date, pull the boxscore, build a per-`mlb_id` map of today's actual stat line + a status string (inning/half + score + home/away + W/L for finals). 3-tier-resilient (live API → empty on failure; the core's projected line is the fallback). **Short TTL cache** (~60-120s — live data) keyed by date, to avoid hammering the API across the 12 users / per-request.
2. **Matchup service** consumes the map: when a player's `mlb_id` is in the live map and the game is live/final, override `MatchPlayer.stats` + `status` with the live values; else keep projected + game-state (the core behavior). `state` already comes from the schedule.
3. **Totals** recompute from live lines for in-progress/final games (mixed projected/live is acceptable; document).

## Gates / cautions

- **Live data + the sole-writer model:** the live box-score fetch is READ-only (no DB write needed if computed per-request + cached in-process), so it does NOT touch the SQLite sole-writer invariant. If a cache table is added, it must respect the scheduler-sole-writer rule (write only from the scheduler) — prefer in-process TTL cache to avoid that.
- **Cost:** boxscore-per-game per request is heavy; the TTL cache + only fetching live/final games (skip scheduled) bounds it.
- **Owner-gated** like other live-path work; its own brainstorm → plan → TDD slice when prioritized.

## Not in scope here

Play-by-play "At Bat / Fielding" micro-status, win-probability-added, and per-batter opposing-pitcher name are further enhancements beyond the first live-line pass.
