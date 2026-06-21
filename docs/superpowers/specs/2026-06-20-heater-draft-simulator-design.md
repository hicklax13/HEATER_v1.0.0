# Draft Simulator (React) — Design Spec

**Date:** 2026-06-20
**Track:** CMO / frontend (web/ only). Part of M1 (14 Streamlit surfaces → React, wired to live API).
**Mirrors:** `pages/20_Draft_Simulator.py` (Streamlit).
**Endpoints:** `POST /api/draft/recommend`, `POST /api/draft/simulate-picks` (both stateless — client holds `config`+`pick_log`, server replays).

## Goal

A `/draft` React page that reproduces the Streamlit mock-draft flow: set up a snake draft, get Monte-Carlo recommendations on your pick, draft a player, watch the AI opponents auto-advance to your next pick, repeat to the end, then see your roster. Wired to the two live endpoints, with a **playable offline mock** (owner-approved 2026-06-20) so the page fully works/demos/verifies without the Python backend.

## Stateless model (the core)

The server holds **zero** draft state. The client owns:
- `config: { numTeams, numRounds, userTeamIndex, rosterConfig? }` — `userTeamIndex` is **0-based** (UI input is 1-based → subtract 1).
- `pickLog: DraftPick[]` where `DraftPick = { pick, teamIndex, playerId, playerName, positions }`.

Every API call sends `{config, pick_log}`; the server replays the log to rebuild `DraftState` and returns a fresh `clock`. Gotchas (from the engine trace):
- `pick` field = `pickLog.length` at the moment of that pick (0-indexed running count).
- `teamIndex` is **authoritative** — for AI picks, echo back what `simulate-picks` returned; for the user's pick, it's `config.userTeamIndex` (it's the user's turn).
- Snake order is pure + local: `pickingTeam(pick) = round%2==0 ? pos : numTeams-1-pos` where `round = pick // numTeams`, `pos = pick % numTeams`.
- The server **never 500s** — on a bad state it returns an empty `clock`/recs with a summary string. The UI must tolerate empty recommendations (offer manual pick from the available list).

## Two-call turn loop

1. **Advance AI → your turn:** `POST /api/draft/simulate-picks {config, pick_log, seed:null}` → `{clock, picks[]}`. Append `picks` to `pickLog`. (Called at `start()` to reach your first pick, and after each user pick.) No-op + skip when it's already your turn.
2. **Recommend (your turn only):** `POST /api/draft/recommend {config, pick_log, top_n, n_simulations}` → `{clock, recommendations[], summary}`. Render the rec cards.
3. **User drafts:** append `{pick: pickLog.length, teamIndex: userTeamIndex, playerId, playerName, positions}` → go to step 1.

## Components & files

| File | Responsibility |
|------|----------------|
| `web/src/app/draft/page.tsx` | Page shell; switches on phase (`setup` / `drafting` / `complete`). |
| `web/src/lib/draft-data.ts` | View-model types (`DraftConfig`, `DraftPick`, `DraftClock`, `DraftRec`, `DraftPlayer`); live fetchers (`draftRecommend`, `draftSimulate`) flag-gated with mock fallback; `pickingTeam()` snake helper. |
| `web/src/lib/draft-mock.ts` | Offline mock: `MOCK_POOL` (validated-mlbId players + generic fillers), `mockSimulate(config, log)` (AI picks best-available by adp to next user turn), `mockRecommend(config, log)` (top-N available by rank → tag/score/reason). Leaf-ish (no adapters import). |
| `web/src/lib/use-draft.ts` | `useDraft()` state machine: `{phase, config, pickLog, clock, recs, busy, start, pick, reset}`. Owns the two-call loop. |
| `web/src/components/draft/SetupForm.tsx` | numTeams (12) / numRounds (23) / your position (1-based, 1) / sim depth; "Start Draft". |
| `web/src/components/draft/RecCard.tsx` | One recommendation: headshot, `PlayerLink`, BUY/FAIR/AVOID tag, score heat bar, reason, "Draft" button. |
| `web/src/components/draft/RosterRail.tsx` | My roster (slot+player) + recent-picks feed (last ~10, user highlighted) + Reset. |
| `web/src/components/draft/AvailableList.tsx` | Searchable available-players list (reuses the search-filter pattern) with "Draft" buttons (manual pick / fallback when recs empty). |
| `web/src/lib/api/types.ts` | `ApiDraftRecommend*` / `ApiDraftSimulate*` aliases over generated schemas. |
| `web/src/lib/api/adapters.ts` | `apiDraftRecommendToData`, `apiDraftSimulateToData` (snake→camel, reuse `toPlayerRef`). |
| `web/src/components/chrome/TopBar.tsx` | "Draft" nav item. |

## Offline mock

`MOCK_POOL`: ~60 real players (reuse the validated mlbIds already audited across the other mock files) each `{id, name, positions, teamAbbr, teamId, mlbId, adp}`, plus generic `Prospect N` fillers (`mlbId: 0`, generic positions) so a full 12×23 draft can complete. `mlbId 0` → `PlayerLink` fallback avatar (audit script only checks 6-digit ids, so fillers are exempt). `mockSimulate` advances AI by picking the lowest-adp available (light top-of-board weighting); `mockRecommend` returns the top-N available by adp with a derived tag/score/reason. Browser runtime → `Math.random` permitted (seed by pick index for stability if needed). The mock never throws.

## States & error handling

- **Phase machine** (not `usePageData` — this page is interactive, not a single fetch): `setup → drafting → complete` (`complete` when `clock.current_pick >= numTeams*numRounds`).
- **Busy:** disable pick buttons + show a subtle spinner during `recommend`/`simulate`.
- **Empty recs:** show "recommendations unavailable — pick from the board" and keep `AvailableList` actionable (the engine may return empty; the user can still draft).
- **Live error:** a failed live call falls back to the mock for that call (so the page degrades to a working mock rather than breaking) — consistent with the other pages' mock fallback, but per-action.

## Out of scope (parity-lite)

- The Streamlit "Undo last pick" (nice-to-have; can add later). 
- The full post-draft **category grade** (Streamlit computes 12-cat totals + A–F). v1 shows the completed roster + a short summary; the grade is a documented follow-up (needs SGP math the API doesn't return).
- Engine-mode radio (Quick/Standard/Full) — default `n_simulations` server-side; expose only a light "sim depth" if trivial.

## Verification

`pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build`, then preview: run a full mock draft (setup → several user picks with AI auto-advance → reach complete), `preview_console_logs` zero errors, `preview_screenshot` of setup + an active pick. `node scripts/audit-mock-ids.mjs` (add `draft-mock.ts`) → 0 mismatches. Live path verified if a local FastAPI `:8000` is runnable; otherwise the mock exercises the full interaction and `tsc` proves the live contract against `generated.ts`.
