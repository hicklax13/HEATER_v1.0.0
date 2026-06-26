# HEATER — Comprehensive User-Testing Campaign: Plan & Coverage Matrix

**Date:** 2026-06-26 · **Branch:** master (HEAD `0444ad2`) · **Tester:** autonomous (Opus)
**Stack under test:** LOCAL React (`web/`, :3000 `heater-web-live`) → LOCAL FastAPI (`api/`, :8000, Clerk OFF = reads open) → real `draft_tool.db` (refreshed from live Yahoo league `469.l.109662` FourzynBurn this session: 317 rosters, 12 records, 12 matchups, 608 FAs). Live spot-check: Vercel `heater-v1-0-1.vercel.app` + Railway API `celebrated-respect-production.up.railway.app`.

## Goal
Find every bug, broken control, wrong/stale number, fabricated-data leak, broken empty/error state, a11y gap, perf problem, and cross-page inconsistency before the 12-friend beta. Re-verify (not trust) the `Outstanding_June26` / `Bugs_June26` / Codex backlogs and go beyond them.

## Test architecture (3 tiers)
- **Tier 1 — Parallel background agents (stateless: HTTP + code, no browser).** Deep API/data-correctness + finding re-verification + security. 7 agents.
- **Tier 2 — Serial browser pass (orchestrator drives Claude_Preview).** All 14 surfaces × every control/state + cross-cutting (nav, a11y, responsive, dark mode, console, perf). One browser.
- **Tier 3 — Live spot-check (orchestrator).** Public live API correctness + unauth-leak probe + logged-out Vercel render (must not show mock-as-real).

## Ground-truth anchors (Yahoo, this session)
| Team | W-L-T | Rank |
|---|---|---|
| Over the Rembow | 11-2-0 | 1 |
| My Precious | 9-3-1 | 2 |
| 🏆 Team Hickey (= viewer) | 5-7-1 | 8 |
| Cyrus The Greats | 3-8-2 | 12 |

Open anomaly to chase: `/api/standings` returns **13 teams** (league_standings has 13 distinct names; league_records/league_teams have 12) → ghost-team leak?

## Surface × Check coverage matrix
Surfaces (14 pages + cross-cutting). Checks: **R**=renders, **D**=data correct vs source, **I**=interactions/controls, **S**=loading/empty/error/unlinked states, **A**=accessibility, **P**=perf/console, **E**=edge/negative.

| # | Surface | API endpoints | R | D | I | S | A | P | E |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Team dashboard `/` | me/team, playoff-odds | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 2 | Lineup Optimizer `/optimizer` | lineup/optimize (std+daily), lineup/set | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 3 | Matchup `/matchup` | matchup | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 4 | Standings `/standings` | standings, playoff-odds | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 5 | Trades `/trades` (Analyzer+Compare tabs) | trade/evaluate, trade-finder, compare, players/search | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 6 | Players/Free Agents `/players` | free-agents, free-agents/pool, transactions/add-drop | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 7 | Pitcher Streaming `/streaming` | streaming, streaming/analyze | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 8 | Probables `/probables` | schedule/probables | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 9 | Hitter Matchups `/hitter-matchups` | schedule/hitter-matchups | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 10 | Closer Monitor `/closers` | closers | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 11 | Research/Leaders `/research` | leaders, leaders/overall | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 12 | Player Databank `/databank` | databank | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 13 | Punt Analyzer `/punt` | punt | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 14 | Draft Simulator `/draft` | draft/recommend, draft/simulate-picks, draft/grade | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| X1 | Bubba AI (every page) | chat/send(-stream), saved-prompts | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| X2 | PlayerDialog (every row) | players/{mlb_id} | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| X3 | Nav + Command palette + TopBar | — | ✔ | — | ✔ | ✔ | ✔ | ✔ | ✔ |
| X4 | Auth: login/signup/unlinked 409/paywall 402 | me/team, admin | — | — | ✔ | ✔ | ✔ | — | ✔ |
| X5 | Responsive (375/768/1280) + dark mode | — | ✔ | — | — | — | ✔ | — | — |

## Tier-1 parallel agent assignments (background)
- **A1** Team + Standings + Playoff-Odds API & data correctness (13-team ghost, records, lever/ops/movers, NaN).
- **A2** Matchup + Optimizer API (pitcher classification HIGH-2, totals, win/tie/loss sim, daily mode, lineup/set).
- **A3** Trades + FA + Punt + Compare + player endpoints (eval correctness, FA pool top-need, player-detail).
- **A4** Streaming + Probables + Hitter-Matchups + Closers + Leaders + Databank + Draft (NaN→max streaming, leader saturation, closer NaN, draft NaN).
- **B1** Re-verify ALL `Bugs_June26.md` findings vs current master code + venv NaN repros.
- **B2** Re-verify ALL `Outstanding_June26.md` HIGH/MED/LOW vs current master code.
- **B3** Security review (Bubba SQL tool, Pro gate, write-auth, CORS, secret hygiene, fail-closed) + live unauth-leak probe.

## Tier-2 browser pass dimensions (orchestrator, serial)
For each surface: snapshot (content/structure) → console errors → network (failed/contracts) → interactions (every control) → states (empty/error/unlinked via API-down or bad params) → screenshot (visual) → a11y (keyboard, focus, labels, color-only cues, table semantics) → responsive (375/1280) → dark mode (sample).

## Output
`docs/testing/2026-06-26-comprehensive-user-test-report.md` — exec summary + beta-readiness verdict; findings by Severity × Surface (problem, repro, evidence, root cause file/endpoint, fix rec); coverage matrix (tested/pass/fail); data-correctness section; reconciliation vs the 3 backlogs.
For Critical/High in-scope+low-risk+verifiable: fix via TDD + merge (standing rule). Else `spawn_task` with full repro.
