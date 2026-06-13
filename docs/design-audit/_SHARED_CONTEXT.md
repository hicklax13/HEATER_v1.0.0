# HEATER Beta — UI/UX Design Audit — SHARED CONTEXT & RULES

> **Read this file in full before you start.** Every page-auditor agent and the
> master-report compiler agent works from these rules so all 15 page reports are
> consistent enough to merge. You own exactly ONE page. Tear it apart — but be
> concrete, evidence-based, and fair.

---

## 1. The mission

HEATER is a fantasy-baseball in-season manager (Streamlit web app) about to go to
**Beta**. The owner wants the UI/frontend to feel **professional, modern,
high-quality, and high-tech**. Your job is to act as a **test user**, exercise
**every** feature/tab/button on your assigned page, **record the real outputs**
(all text and numbers), find **every** error / rough edge / point of confusion,
**criticize** the design and the outputs hard, and deliver **at least 10
high-level recommendations** to make the page better.

### Your persona while testing
You are the actual user: **a NOVICE fantasy-baseball manager** — *Connor*, owner
of **Team Hickey** in the 12-team **FourzynBurn** Yahoo H2H-categories league.
He is sharp but **not a data scientist and not a CLI user**. If a label, number,
or control would confuse a smart beginner, that is a finding. "What does this
number mean? what do I do with it? what happens if I click this?" is the lens.

---

## 2. The 15 pages (and who owns what)

Sidebar order. Each agent owns ONE and writes ONE report file in
`docs/design-audit/`.

| # | Page (sidebar label) | Source file | Your output file |
|---|----------------------|-------------|------------------|
| 01 | Draft Tool (Home) | `app.py` (the `render_single_user_app` / draft page) | `page-01-draft-tool.md` |
| 02 | My Team | `pages/1_My_Team.py` | `page-02-my-team.md` |
| 03 | Lineup Optimizer | `pages/2_Line-up_Optimizer.py` | `page-03-lineup-optimizer.md` |
| 04 | Closer Monitor | `pages/3_Closer_Monitor.py` | `page-04-closer-monitor.md` |
| 05 | Pitcher Streaming | `pages/4_Pitcher_Streaming.py` | `page-05-pitcher-streaming.md` |
| 06 | Matchup Planner | `pages/5_Matchup_Planner.py` | `page-06-matchup-planner.md` |
| 07 | League Standings | `pages/6_League_Standings.py` | `page-07-league-standings.md` |
| 08 | Punt Analyzer | `pages/10_Punt_Analyzer.py` | `page-08-punt-analyzer.md` |
| 09 | Trade Analyzer | `pages/11_Trade_Analyzer.py` | `page-09-trade-analyzer.md` |
| 10 | Trade Finder | `pages/12_Trade_Finder.py` | `page-10-trade-finder.md` |
| 11 | Free Agents | `pages/14_Free_Agents.py` | `page-11-free-agents.md` |
| 12 | Player Compare | `pages/16_Player_Compare.py` | `page-12-player-compare.md` |
| 13 | Leaders | `pages/17_Leaders.py` | `page-13-leaders.md` |
| 14 | Player Databank | `pages/19_Player_Databank.py` | `page-14-player-databank.md` |
| 15 | Draft Simulator | `pages/20_Draft_Simulator.py` | `page-15-draft-simulator.md` |

### HARD EXCLUSIONS (do not touch, do not open, do not critique, do not recommend)
- **Admin Console** (`pages/_admin_console.py`), **Usage Analytics**
  (`pages/_admin_analytics.py`), **Admin Controls** (`pages/_admin_controls.py`).
- **The "HEATER AI" chat panel** (everything under `src/ai/` —
  `src/ai/chat.py`, `src/ai/chat_shell.py`). It auto-opens and floats over the
  right side of every page. **It is 100% out of scope.** Do not test it, do not
  count its buttons, do not critique it, do not recommend anything about it.
  Mentally subtract it from your page.

---

## 3. The app is LIVE and you have real data

A real instance is running at `http://localhost:8501` (MULTI_USER mode, logged in
as the QA admin) against the **live league SQLite DB**. Real facts you can rely on
(captured by the orchestrator during the live pass):

- Player pool: **9,888** players. League: 12-team, 23-round snake, H2H categories.
- User team: **Team Hickey**, record **3-7-1** (~.318), sitting **10th of 12**,
  "GB from 1st: 7, 2 GB from 4th". Roster: **27 players (13 hitters, 14 pitchers)**,
  "23 active + 4 IL".
- This week = **Week 12 vs "The Good The Vlad The Ugly"**, projected win prob
  **46% / tie 19% / loss 35%**, projected **6-6**.
- Hitting cats: R, HR, RBI, SB, AVG, OBP. Pitching cats: W, L, SV, K, ERA, WHIP.
  Inverse: L, ERA, WHIP.
- **Data freshness is currently degraded:** the home page shows
  **"Yahoo: Warming up"** / "Data last refreshed 1219 min ago" and the server log
  says *"Matchup served from SQLite cache (Yahoo offline)"*. So you are looking at
  **cached** data, which is fine for an audit — but note anywhere the UI fails to
  communicate staleness to the user.

### DO NOT drive the browser
The orchestrator already captured the live rendered state. There is ONE shared
browser; 15 agents fighting over it would corrupt each other. **Do not use any
`preview_*` / browser tools.** Work from **source code + read-only DB queries**.
(The orchestrator's live observations for your page, if any, are in §7.)

---

## 4. The design system ("Combustion Index") — what "good" looks like here

Defined in `src/ui_shared.py` (the `THEME`/`T` dict + `inject_custom_css()` +
`render_page_header()` + `build_compact_table_html()` + `render_empty_state()` +
`PAGE_ICONS`), locked by `tests/test_combustion_lock.py`, with native theming in
`.streamlit/config.toml`. **Read `src/ui_shared.py` — it is the authority on every
color, font, and component your page uses.** Key tokens:

- **Primary / accent:** hot orange `#ff6d00` (CTAs, active tabs, the "." in the
  wordmark, left-border rails). NO brand red; ember `#e63946` only for
  functional-negative icons.
- **Chrome:** deep navy `#112744` sidebar rail + top bar. White text on navy.
- **Canvas:** pure white. **Surface/cards:** pale `#f5f6f8` with darker bold text.
- **Fonts:** Archivo (display/headings, `--font-display`), Inter (body,
  `--font-body`), IBM Plex Mono (figures/numbers, `--font-mono`).
- **Categorical chart colors:** `THEME["tiers"]`.
- **Page header pattern** (`render_page_header`): a small eyebrow
  `SECTION · / FIG.NN — TITLE`, then a big `Title.` wordmark with an orange period,
  then an orange underline rule. Every page uses it.
- **Formatting authority:** `format_stat(value, stat_type)` — AVG/OBP `.3f`,
  ERA/WHIP `.2f`, SGP `+.2f`. Flag any stat rendered with wrong precision.

When you critique visuals, judge against this system: Is the hierarchy clear? Is
spacing/density comfortable? Is color used meaningfully (orange = action/positive,
not decoration)? Are numbers monospaced & aligned? Is it consistent with sibling
pages? Does it actually look *modern/high-tech*, or generic-dashboard?

---

## 5. How to capture REAL outputs safely (read-only)

You must record actual text/numbers, not guesses. Two reliable sources:

**(a) Read the page source** to enumerate EVERY tab/button/widget/expander/input
and trace exactly what each renders and what function it calls.

**(b) Query the live DB read-only** to capture the real values behind each feature.
Use the project's sanctioned access. Example pattern you can run via `python -c`
or a temp script in the repo root (delete it after):

```python
# Prepend this network guard so any engine/data call falls back to cache/seed
# instead of hanging on a live Yahoo/MLB/FanGraphs fetch (Yahoo is offline now):
import socket
_orig = socket.socket.connect
def _guard(self, addr):
    host = addr[0] if isinstance(addr, (tuple, list)) else addr
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise OSError("network blocked for audit")
    return _orig(self, addr)
socket.socket.connect = _guard

from src.database import get_connection, load_player_pool
conn = get_connection()
# ... read-only SELECTs ...  e.g. league_rosters, standings, closer data,
# season_stats, players, refresh_log, ecr_rankings, etc.
conn.close()
pool = load_player_pool()   # the enriched 9,888-row pool many pages consume
```

Rules for (b):
- **READ-ONLY.** No INSERT/UPDATE/DELETE. No `force_refresh`. No writes of any
  kind to the DB. The live server is the only sanctioned writer.
- Keep any Monte-Carlo / simulation calls **tiny** (low sim counts) or skip them
  and trace the logic instead — don't burn minutes on heavy sims.
- If a function needs the Streamlit runtime (`st.session_state`, widgets) and
  won't run standalone, **reconstruct** the output from the DB data + the source
  logic and clearly label it **"(reconstructed)"**.
- Delete any temp script you create. Do not leave junk in the repo. The ONLY file
  you create/keep is your one report `.md`.

---

## 6. Your report — required structure (write to your `page-NN-*.md`)

Be exhaustive; the master report is built from your file, so richer = better.
Use this skeleton:

1. **`# Page NN — <Name> — Test-User Report`**
2. **Page purpose & first impression** — what is this page for; what a novice
   thinks in the first 5 seconds.
3. **Methodology** — what you read/queried to test it.
4. **Feature & control inventory** — a table of EVERY tab, button, input,
   expander, toggle, table, chart on the page (Control | Type | What it does |
   Tested? ).
5. **Feature-by-feature test log WITH REAL OUTPUTS** — for each control: what you
   did, and the **actual output captured** (paste real numbers/text/labels; mark
   "(reconstructed)" where applicable). This is mandatory — every button's output
   must be recorded.
6. **Errors, issues & difficulties** — anything broken, slow, confusing, mislabeled,
   inconsistent, empty-state-ugly, or novice-hostile. Include perf (slow renders).
7. **UI/UX & visual-design critique** — layout, hierarchy, spacing/density, color
   use, typography, number formatting, consistency w/ the design system, mobile,
   accessibility, empty/error states, microcopy. Be harsh and specific.
8. **≥10 high-level recommendations** — numbered, each: a clear title, the problem
   it fixes, and the suggested change. Order by impact. (More than 10 is welcome.)
9. **Severity-tagged issue list** — bullet list, each tagged
   `[BLOCKER] / [HIGH] / [MEDIUM] / [LOW] / [POLISH]`.

Write real Markdown. Quote real values. No placeholders like "TODO" or "e.g. some
number" — capture the actual data.

---

## 7. Cross-cutting findings already observed (confirm/expand for YOUR page)

The orchestrator already saw these during the live pass. Check whether each
applies to your page and fold it into your report where relevant:

- **`st.components.v1.html` is deprecated and overdue for removal.** The server
  logs spew *"Please replace `st.components.v1.html` with `st.iframe`. … will be
  removed after 2026-06-01"* on many page loads. If your page renders custom HTML
  tables/cards via that API, it is on borrowed time — flag it.
- **Data-freshness comms:** Yahoo is "Warming up"/offline; data is cached and
  ~1219 min old. Note whether YOUR page tells the user how stale its data is.
- **Performance:** a *"12s budget exceeded … slow/degraded MLB Stats API"* warning
  appears; **Trade Analyzer is heavy enough that its render dropped the browser's
  WebSocket** (frozen page). If your page is compute-heavy, assess load time and
  whether the user gets a spinner / progress / partial-results.
- **Figure-number inconsistency:** headers use `FIG.NN`, but Pitcher Streaming
  shows `FIG.4` (not `FIG.04`), and Matchup Planner has a section caption
  `FIG.02 · …` on a `FIG.05` page. Check your page's `FIG` numbering & captions.
- **Nav-label vs page-title drift:** e.g. sidebar "Punt Analyzer" vs H1 "Punt
  Strategy Simulator." Check your page's label-vs-title consistency.
- **No emoji / inline-SVG icon system** (`PAGE_ICONS`); injury badges are CSS dots.
  Flag any stray emoji or off-palette hex (guarded by
  `tests/test_no_offpalette_hex_in_pages.py`).

---

## 8. Constraints (all agents)
- Read-only repo: do **not** edit any source file, do **not** run git, do **not**
  write the DB. The only artifact you produce is your one report `.md`.
- Do not use browser/`preview_*` tools. Do not touch the excluded pages or the AI
  chat (§2).
- Stay on YOUR page. If a control navigates to another page, note it and stop there.
- When done, your final message back to the orchestrator = a ~10-line summary:
  page name, # controls tested, # errors found, your top 3 recommendations, and
  confirmation the report file was written.
