# HEATER "Combustion Index" Redesign — Implementation Plan

> **Approved design (2026-06-08).** Pixel-exact targets: `docs/design/mockup-myteam-v3.png`,
> `docs/design/mockup-player-popup.png`, `docs/design/heater-design-canvas.png`. Philosophy:
> `docs/design/2026-06-08-combustion-index-philosophy.md`. Branch: `ui/combustion-redesign-2026-06-08`.
> REQUIRED SUB-SKILL: superpowers:executing-plans (or subagent-driven-development). Big build →
> best run as a focused session (like the v1 revamp). Verify every page with real screenshots;
> deploy is OWNER-GATED.

## Locked palette + type (from all owner feedback)
- **Canvas (content bg):** neutral off-white `#f4f3f1`; **panels** white `#ffffff`.
- **Chrome (sidebar/logo/avatar):** deep **navy** `#112744`→`#0e2244` (the "original cool" blue), bone text `#eef1f6`.
- **Accent:** hot **orange `#ff6d00`** (+ `#ff9a3c` flame, `#e8480a` ember). **No red** anywhere (brand). Negative/COLD = muted steel `#5f7d9c`; good = `#1f9d6b`.
- **Text on light:** `#1b1c20` / muted `#646a78` / subtle `#9aa0ac`.
- **Type:** **Archivo** (700–900 display/headers) + **Inter** (400–600 body/UI) + **IBM Plex Mono** (figures/tables only). Retire Figtree + Bebas.
- **Detail vocabulary:** corner registration ticks on panels, `FIG.NN`/index micro-labels (mono), orange accent rule under headers, heat bars, sparklines, stat-readout chips, tick scales. Full-width content (remove the 1180px clamp). No emoji (inline SVG).

## Phase A — Design system (`src/ui_shared.py`, `.streamlit/config.toml`)
- THEME tokens → the locked palette above (orange primary, navy chrome, neutral light, steel COLD). Update every `T["..."]` consumer; keep all keys (BR-4 guard).
- `inject_custom_css`: swap `@import` to Archivo+Inter+IBM Plex Mono; rewrite `.sec-head`/h1–h3 to Archivo; body Inter; `.stApp` bg `#f4f3f1`; remove the 1180px max-width (full-width with comfortable padding, content never under the rail); orange buttons (solid primary / outline secondary); underline tabs; **instrument-panel** card styling (corner ticks, soft shadow, accent rule); light detailed tables; sidebar = deep navy rail, bone text, orange active.
- `get_plotly_layout`/`polar` → light bg, orange colorway, fine gridlines.
- `config.toml` → primaryColor `#ff6d00`, backgroundColor `#f4f3f1`, base light.

## Phase B — Reusable detail renderers (`src/ui_shared.py`)
- `render_panel(title, fig_label, body)` instrument card (corner ticks + Archivo header + accent + fig).
- `render_stat_readout`, `render_heatbar`, `render_sparkline`, `render_eyebrow` helpers.
- Page-header renderer: Archivo title + eyebrow + `FIG.NN` + orange accent rule + tick scale (replaces the navy pill).

## Phase C — My Team feature (`pages/1_My_Team.py` + helpers)
- **Active Roster panel:** full roster, **headshots** (`midfield.mlbstatic.com/v1/people/{mlb_id}/spots/120`), Yahoo-style columns (hitters: AB R H HR RBI SB AVG OBP; pitchers: IP W L SV K ERA WHIP), status dots. **Timeframe** segmented control (Season/L30/L14/L7/Today → `player_databank.compute_rolling_stats`) + **Hitters/Pitchers** toggle.
- **Player dossier popup** (`st.dialog`): header (headshot + season line) + **Game Log** (last 10 from `player_databank.load_game_logs`, with W/L outcome + per-game line + form bar) + **Upcoming · Projections** (next games from `game_day`/schedule + per-game projection from the projection engine + matchup confidence).
- Keep existing My Team content above it (War Room, matchup pulse, actions, streaks) restyled to the new look.

## Phase D — Per-page sweep (other 12 pages)
- Apply the design system + detail renderers; replace page-local light/cream/red styling with the new palette; instrument-panel cards; detailed light tables; orange accents. One commit per page, screenshot-verified.

## Phase E — Guards + suite
- Update `tests/test_fp_revamp_lock.py` → assert orange primary `#ff6d00` (not red), Archivo/Inter (not Figtree), navy chrome token, no-emoji. Update theme/font guards. Full suite green + structural gate.

## Phase F — Review + deploy (OWNER-GATED)
- code-reviewer + coderabbit; before/after deck; **confirm with owner**; merge → Railway redeploy; live walkthrough.

## Notes / gotchas
- The data exists: `src/player_databank.py` (game logs, rolling stats), projection engine, `game_day` (upcoming/probables); headshots by `mlb_id`. Mostly wiring into the new UI.
- Streamlit `st.dialog` for the popup (≥1.37, pinned). Mobile: keep the drawer (header-hide `@media min-width:768px`).
- Local verify: `HEATER_SCHEDULER_BOOT=1` + park `data/yahoo_token.json` for a stable read-only app (restore before pytest). Login `qa_admin`/`qa-local-only-2026`. Live URL: `https://heaterv100-production.up.railway.app`.

## Final approved refinements (2026-06-08) — `mockup-player-popup.png` is the GOLD STANDARD; match it everywhere
- **"In your face" data type (app-wide):** ALL data tables use **Archivo bold (700) with tabular figures** for values (scoreboard feel), **bold underlined Archivo headers**, and **brightened/bolder eyebrow + stat labels** (no more quiet muted mono). The player popup is the reference for density + weight.
- **Team identity theming (everywhere a team appears):**
  - **Team logos** via `https://www.mlbstatic.com/team-logos/{teamId}.svg`; **player headshots** via `https://midfield.mlbstatic.com/v1/people/{mlbId}/spots/120`. Need an `mlb_team_id` (and team primary/secondary colors) lookup — add a `TEAM_BRAND` map (id → {abbr, primary, secondary, logo}) in `src/ui_shared.py` (or derive from MLB API `teams` endpoint, cache in DB).
  - **Player popup header bg = that player's TEAM primary color** (e.g. Astros navy `#002D62`), with the team logo (large faint watermark + small inline by the team name) and the opponent's logo in each **game-log OPP** row.
  - **Roster rows:** show the player's **team logo** + a subtle **team-color accent** (left border / chip) so the roster reads as a colorful branded grid.
  - Keep text readable on any team color (auto-pick light/dark text by luminance).
- These apply in Phase C (My Team) and propagate via the shared renderers (Phase B) to every page that lists players/teams (Standings, Leaders, Free Agents, Trade pages, Closer Monitor, etc.).
