# Design — Pitcher Streaming Analyzer page (`pages/4_Pitcher_Streaming.py`)

> Status: **draft design, pre-implementation**
> Date: 2026-06-09
> Companion implementation plan: `docs/superpowers/plans/2026-06-09-pitcher-streaming-analyzer.md`

## 1. Problem

Streaming starting pitchers is the highest-leverage weekly action in FourzynBurn: 10
adds/week, a 20-IP weekly floor, a ~54-IP competitive target, and 5 of the 6 pitching
categories (W, L, K, ERA, WHIP) move materially with every streamed start. HEATER's
streaming intelligence exists but is **fragmented across four places with no
matchup-specific deep dive and no date control**:

| Where | What it does | What it lacks |
|-------|--------------|----------------|
| `recommend_streaming_moves` (`src/optimizer/fa_recommender.py:1750`) | Today-only add/drop swaps with net SGP, in-play/protected cats, IP-minimum + relaxed-threshold logic | Today only; buried as one section of the Lineup Optimizer's Streaming tab; no per-matchup evidence |
| Streaming tab in `pages/2_Line-up_Optimizer.py:3119-3640` | 7-day FA pitcher table + two-start list + calendar | **Page-local ad-hoc scoring formula** (lines 3333-3415) that duplicates engine logic outside `CONSTANTS_REGISTRY` — the same class of duplication the SGP consolidation eliminated elsewhere; fixed window; no add-budget awareness; no live-game lock detection |
| `src/optimizer/streaming.py` | Canonical marginal-SGP math (`compute_streaming_value`, `compute_bayesian_stream_score`, `optimal_streaming_schedule`), two-start fatigue (0.93×) | Pure math, no UI; `WEEKLY_ADDS_BUDGET=7` is stale (league is 10) |
| `src/two_start.py` | `compute_pitcher_matchup_score` (0-10: K-BB%/xFIP/CSW vs opp wRC+/K% × park × home/away), confidence tiers by days-ahead | Roster-centric two-start lens only |

There is **no page** that answers the actual decision: *"For date X (or this matchup
week), which free-agent pitcher should I stream, against which opposing lineup, and what
is the evidence — opponent offense, park, weather, his recent form, his history vs this
team, and what actually happened when the engine made similar calls?"*

## 2. Goals / non-goals

**Goals**
1. **Date-driven stream finder** — pick any date from today through +7 days (default
   `get_target_game_date()`, which auto-advances to tomorrow when today's slate is final);
   ranked board of streamable probable starters for that date.
2. **Matchup-specific scoring** — opposing-team offense (wRC+, K%, BB%, ISO, L14 trend,
   vs-handedness splits where available), park factor, home/away, weather, confirmed
   lineups + pitcher-vs-batter (PvB) splits when day-of.
3. **Live awareness** — in-progress/final games mark a start LOCKED (visible but
   non-actionable); the board never recommends adding a pitcher whose game has started.
4. **Scheduled-game awareness** — probables with confidence tiers (HIGH 0-2 days /
   MEDIUM 3-5 / LOW 6+), two-start detection, and a Mon-Sun week planner that sequences
   streams under the remaining add budget and IP pacing.
5. **Historical evidence** — per-candidate game-log deep dive (last N starts, vs this
   opponent, at this venue, L14 vs season form) and a track-record tab replaying what the
   scorer would have ranked on past dates vs the actual box-score lines.
6. **Personalized + engine-canonical** — drop candidate and net SGP come from the
   existing `recommend_streaming_moves` engine (today scope); all scoring composes
   canonical services (`SGPCalculator`, `LeagueConfig`, `MatchupContextService` weights,
   `CONSTANTS_REGISTRY`) — zero new inline formulas in the page.

**Non-goals**
- **Not** replacing the Lineup Optimizer's Streaming tab in this change. The tab stays;
  a follow-up phase DRYs its ad-hoc formula to delegate to the new engine module (the
  page-local formula must not survive in two places long-term).
- **No automated Yahoo transactions** — the page recommends; the user executes on Yahoo.
- **No new bootstrap phases** required for v1. Team vs-handedness splits are an optional
  Tier-1 live fetch with a documented fallback to overall wRC+ (see §6).
- **No second SQLite writer** under MULTI_USER — the page is read-only; all page-level
  caching is `st.cache_data`, never DB writes from member sessions.
- Closer/RP streaming for saves stays on the Closer Monitor page; this page is SP-focused
  (an SP/RP hybrid with a scheduled start qualifies; a pure reliever does not).

## 3. Page identity & placement

- **File:** `pages/4_Pitcher_Streaming.py` — slot 4 has been free since the Bullpen page
  was removed in the Section 5 consolidation, and it lands in the "daily action" group of
  the workflow ordering (after Lineup Optimizer / Closer Monitor, before Matchup Planner).
- **Title:** "Pitcher Streaming" | **Nav key / flag:** `page:4_Pitcher_Streaming`
- **Registry:** new `PAGE_REGISTRY` entry in `src/nav.py` between `3_Closer_Monitor` and
  `5_Matchup_Planner`; new SVG icon in `PAGE_ICONS` (`src/ui_shared.py`) — no emoji.
- **Personalized page:** resolves "my team" via `resolve_viewer_team_name(rosters)`
  (WITH a frame — the frame guard test applies) and joins
  `tests/test_pages_use_viewer_team_resolver.py::_PERSONALIZED_PAGES`.

## 4. UX — four tabs

### Tab 1 — Stream Finder (default)
Controls: date selector (default `get_target_game_date()`, range today → +7); candidate
scope (Free agents only / FAs + my SPs — the latter for "should I even start mine over a
stream?"); minimum projected-IP filter; "show locked starts" toggle (default on,
rendered greyed with a LOCKED chip — transparency over hiding).

Main board — one row per (pitcher × scheduled start) on the selected date:

| Column | Source |
|--------|--------|
| Pitcher, Team, Throws | player pool |
| Opp (with @/vs), Park factor | schedule + `ctx.park_factors` |
| Opp wRC+, Opp K% (vs-hand when available, else overall + footnote) | `ctx.team_strength` / §6 |
| Status | PROBABLE-HIGH / -MED / -LOW confidence tier, or LOCKED / FINAL |
| Proj line: IP, K, ER, W% | `compute_bayesian_stream_score` |
| 2-start flag | `identify_two_start_pitchers` |
| **Stream Score** (0-100) | new `score_stream_candidate` (§5) |
| Net SGP + suggested drop | `recommend_streaming_moves` (today only; future dates show score-only with a caption explaining swap recs need same-day matchup state) |
| Own% | `percent_owned` |
| Risk | flags (§5) |

Header strip: adds remaining this week (`ctx.adds_remaining_this_week` / 10), projected
weekly IP vs 54 target and the 20-IP floor (`src/ip_tracker.py`), in-play vs protected
categories for the live matchup (so a K-heavy stream reads differently when K is already
locked up). Each row gets a "why" expander with the full component breakdown (mirroring
the Free Agents page why-expander pattern) — every number traceable to a component.

### Tab 2 — Matchup Microscope
Select any board candidate (or any rostered SP) → single-matchup deep dive:
- **Opposing lineup:** confirmed lineup when posted (`get_todays_lineups`), else projected
  from depth charts; per-batter handedness vs the pitcher's throws with platoon exposure
  summary (`platoon_adjustment` / `bayesian_platoon_adjustment`).
- **PvB history:** top batters by regressed wOBA from `get_pvb_matchup_data`
  (pa ≥ 5, regression to 60-PA stabilization via `pvb_matchup_adjustment`).
- **Pitcher history:** last 10 starts game log; career/season splits vs this opponent and
  at this venue (statsapi game logs, lazily fetched + cached, §6); L14 form vs season
  (`get_player_recent_form_cached`).
- **Environment card:** park factor, weather (temp/wind/precip with dome detection,
  wind-out vs wind-in via `OUTFIELD_BEARING`), home/away.
- **Expected line distribution:** K / IP / ER / win-prob from
  `compute_bayesian_stream_score` with the matchup grade.

### Tab 3 — Week Planner
Mon-Sun matchup-week calendar of streamable probables (reusing the `build_schedule_grid`
tier palette), with a greedy plan from `optimal_streaming_schedule` constrained by
`ctx.adds_remaining_this_week` and IP pacing: "Tue: add A (drop X) → Thu: add B (drop A)
→ Sun: add C (drop B). Projected: +18 IP, +21 K, ~2 W; ERA exposure +0.07." Two-start
candidates highlighted with second-start fatigue (0.93×) already applied. The plan is
advisory and re-greedied on each render — FCFS waivers mean it cannot be reserved.

### Tab 4 — Track Record
- **Replay:** pick a past date (or "last 14 days") → what the scorer would have ranked
  top-N, side-by-side with the actual stat lines from statsapi game logs (IP/K/ER/W/QS),
  plus summary hit-rate metrics (top-3 mean K, ERA, quality-start rate vs the
  all-streamable baseline).
- **My streams:** season-to-date adds of pitchers from the Yahoo `transactions` cache,
  each scored against what the pitcher actually did in the following week.
- **Honesty caveat (rendered on the tab):** HEATER does not store point-in-time
  projections, so replay scores use current YTD/ROS data as a proxy — good for validating
  the *matchup* components (opponent, park, schedule are historical facts), weaker for
  the *form* component. A nightly score-snapshot table (scheduler-written, keeping the
  sole-writer invariant) is the documented Phase-2 path to exact replay.

## 5. Engine — new module `src/optimizer/stream_analyzer.py`

A composition layer over existing canonical pieces; `streaming.py` stays pure math. The
page calls only this module + `recommend_streaming_moves` — no scoring in the page.

```python
build_stream_board(ctx, target_date: str, include_rostered: bool = False) -> pd.DataFrame
    # One row per (pitcher, start) on target_date; all columns in §4 Tab 1.

score_stream_candidate(pitcher_row, start_info, opp_context, config,
                       category_weights=None) -> dict
    # {"stream_score": 0-100, "components": {...}, "risk_flags": [...],
    #  "expected_line": {...}, "net_sgp": float}

get_opponent_offense_context(team_abbr: str, vs_throws: str | None,
                             team_strength: dict) -> dict
    # {"wrc_plus", "k_pct", "bb_pct", "iso", "l14_wrc_plus", "split_source"}
    # split_source ∈ {"vs_hand", "overall"} — UI footnotes the fallback.

get_pitcher_vs_team_history(mlb_id: int, opp_team: str | None = None,
                            venue: str | None = None, last_n: int = 10) -> pd.DataFrame
    # statsapi gameLog rows, aggregated with the backtest_runner pitching aggregator.

replay_stream_date(target_date: str, pool, config, top_n: int = 5) -> dict
    # {"board_then": df, "actuals": df, "summary": {...}, "proxy_caveat": True}
```

**Stream Score** = `100 × sigmoid(weighted z)` over six components, weights registered in
`CONSTANTS_REGISTRY` (names `stream_score_w_*`, each with citation, bounds, sensitivity):

| Component | Weight (initial) | Source |
|-----------|------------------|--------|
| Matchup score | 0.35 | `compute_pitcher_matchup_score` (two_start.py: K-BB%/xFIP/CSW vs opp wRC+/K%, park, home/away) |
| Marginal SGP | 0.25 | `compute_streaming_value` with `MatchupContextService` category weights and `LeagueConfig.sgp_denominators` — inverse stats (ERA/WHIP/L) signed by `SGPCalculator` semantics, never inline |
| Recent form | 0.15 | L14 vs season ERA/K% delta, bounded by existing `_FORM_MULT_LO/HI` (±20%) |
| Lineup/PvB exposure | 0.10 | regressed wOBA of projected opposing lineup vs this pitcher (day-of only; neutral 0 when unavailable) |
| Environment | 0.10 | park × weather multiplier (dome ⇒ park-only) |
| Win probability | 0.05 | team-strength differential proxy from `compute_bayesian_stream_score` |

**Risk flags** (each a registry constant where thresholded): `HIGH_WHIP` (> 1.40 ceiling,
reusing `_WHIP_SAFETY_CEILING`), `SHORT_LEASH` (proj IP/start < 4.5), `ELITE_OFFENSE`
(opp wRC+ ≥ 110), `HITTER_PARK` (factor ≥ 1.08), `WIND_OUT` (≥ 12 mph out, non-dome),
`BULLPEN_GAME_RISK` (no qualified SP confirmed), `LOW_CONFIDENCE` (LOW probable tier).

**Cleanup folded in:** `streaming.py`'s stale `WEEKLY_ADDS_BUDGET = 7` is replaced by the
canonical `src/league_rules.py` value (10) at its call sites — the last-add penalty
currently fires three adds too early.

## 6. Data flows, tiers, and performance

| Data | Primary | Fallback | Notes |
|------|---------|----------|-------|
| Probables / schedule | `ctx.todays_schedule` + `get_weekly_schedule(days_ahead=7)` (statsapi) | — | confidence tier by days-ahead |
| Live/locked status | `LOCKED_GAME_STATUSES` (game_day.py) over schedule rows | — | same convention as DCV locked_teams |
| Opponent offense | `ctx.team_strength` (pybaseball wRC+/K%) | statsapi team stats | already 2-tier in bootstrap |
| vs-hand team splits | live pybaseball team splits fetch (lazy, cached) | overall wRC+ + UI footnote | optional enhancement; board works without it |
| Confirmed lineups | `get_todays_lineups` (day-of) | depth-chart projection | PvB/lineup component neutral when absent |
| PvB splits | `pvb_splits` table via `get_pvb_matchup_data` | neutral | pa ≥ 5, regress to 60 |
| Weather / park | `ctx.weather`, `ctx.park_factors` | dome/neutral | existing tiers |
| Pitcher game logs | statsapi `player_stat_data(gameLog)` | — | `st.cache_data(ttl=3600)`; fetched lazily inside Microscope/Track Record, never during board render |
| FA pool + ownership | `ctx.free_agents` / pool `percent_owned` | SQLite | via `build_optimizer_context` |
| Add budget / IP pacing | `ctx.adds_remaining_this_week`, `src/ip_tracker.py` | — | |

**Context strategy:** one `build_optimizer_context(scope="today", ...)` call per render
(same pattern and `user_roster` / `level_filter="MLB only"` discipline as the Free Agents
page). Future-date boards are schedule-driven off the same ctx; only the swap
recommendation requires `scope == "today"` and is suppressed (with caption) otherwise.
Heavy per-pitcher history fetches are click-triggered and cached, keeping the initial
board render inside the existing gameday-enrichment budget philosophy (~25 s).

## 7. MULTI_USER / permissions

- Page performs **zero DB writes** from user sessions (preserves the scheduler-as-sole-
  SQLite-writer invariant). Page caches are `st.cache_data` only.
- If any data-refresh button is added it must be gated by `viewer_can_write()` and join
  `tests/test_refresh_buttons_admin_gated.py`. v1 ships without one (freshness comes from
  the scheduler under MULTI_USER, per the standard caption).
- Per-page feature flag `page:4_Pitcher_Streaming` honors "absence = enabled".
- Flag-off (v1 single-user) behavior: standard auto-discovered page, byte-for-byte normal.

## 8. Structural-invariant impact (machine-checked)

New page boilerplate, in the exact order the guards enforce: guarded
`st.set_page_config` behind `if not multi_user_enabled():` → `init_db()` →
`inject_custom_css()` → `require_auth()` → `require_page_enabled("page:4_Pitcher_Streaming")`
→ `log_page_view("Pitcher Streaming")` → `page_timer_start()` → … →
`render_feedback_widget()` at EOF.

Existing tests that MUST be updated (page count 13 → 14):
- `tests/test_pages_have_auth_guard.py` (count assertion)
- `tests/test_pages_have_feedback_and_usage.py` (count assertion)
- `tests/test_admin_pages_flag_enforced.py` (count assertion)
- `tests/test_pages_guard_set_page_config.py` (count assertion)
- `tests/test_nav.py` (`groups["Season"] == 13` → 14; PAGE_REGISTRY-matches-disk check
  passes once the registry entry is added)
- `tests/test_pages_use_viewer_team_resolver.py` (add to `_PERSONALIZED_PAGES`)

Standing rules the new code must satisfy: no hardcoded category lists (derive from
`LeagueConfig`), all SGP through `SGPCalculator`, `format_stat` for every rate-stat
display (no inline `f"{x:.2f}"` near ERA/WHIP), no `sqlite3.connect` (use
`get_connection`), no emoji (SVG `PAGE_ICONS`), no module-level `_LC` singleton in the
new engine module, all tunable thresholds in `CONSTANTS_REGISTRY`.

New guard tests shipped with the feature (exact cases in the implementation plan):
- `test_stream_analyzer_constants_registered.py` — every `stream_score_*` weight + risk
  threshold lives in `CONSTANTS_REGISTRY` with value/citation/bounds/sensitivity.
- `test_stream_analyzer_score_components.py` — component breakdown is complete and
  reproducible; weights read from the registry at call time (calibration-compatible).
- `test_stream_analyzer_locked_games.py` — in-progress/final starts are flagged LOCKED
  and excluded from actionable recommendations for today.
- `test_stream_analyzer_inverse_stats.py` — ERA/WHIP/L enter the score with the correct
  sign via canonical SGP paths (the FA-engine PR #99 lesson, locked here too).
- `test_stream_analyzer_adds_budget_canonical.py` — module reads the 10-add budget from
  `src/league_rules.py`, not `streaming.py`'s stale 7.
- `test_stream_page_no_inline_scoring.py` — the page contains no arithmetic scoring
  expressions; all scores arrive from `stream_analyzer` / `recommend_streaming_moves`
  (prevents regrowing the Lineup-Optimizer-tab formula).

## 9. Decisions taken (and alternatives rejected)

1. **New page, not a Lineup Optimizer tab.** The tab is one of six on an already-heavy
   page, today-scoped, and its formula is page-local. A dedicated daily-action page gets
   date control, deep dives, and a track record. (Also the explicit request.)
2. **Keep the existing Streaming tab for now; DRY it in the final phase** by delegating
   its Section-1 scoring to `build_stream_board`, guarded so the inline formula cannot
   persist alongside the engine. Removing the tab outright is a separate product call.
3. **New module `stream_analyzer.py` rather than growing `streaming.py`** — keeps the
   pure-math module dependency-light (no statsapi/streamlit imports) and gives the
   composition layer a single owner, mirroring `fa_recommender.py` over `waiver_wire.py`.
4. **Replay uses YTD-proxy scoring, disclosed in-UI** — exact point-in-time replay needs
   nightly snapshots (scheduler-written table); deferred to Phase 2 with the design noted
   so the sole-writer invariant is preserved when it lands.
5. **vs-handedness splits are best-effort** — fallback to overall wRC+ with a visible
   footnote rather than blocking on a new bootstrap phase.

## 10. Verification (the real bar — not green tests)

Done only when, in a browser session against live league data (logged in
`docs/VERIFICATION_LOG.md`):
1. Board for today matches MLB.com probables; a game that goes live mid-session shows
   LOCKED on rerun and drops out of the actionable swap recs.
2. The date selector at +3 days shows MEDIUM-confidence probables that match team
   beat-writer expectations; two-start flags match the Lineup Optimizer's list.
3. The suggested swap for today agrees with `recommend_streaming_moves` output on the
   Lineup Optimizer Streaming tab (same engine, same answer).
4. Microscope for one candidate shows a confirmed lineup after it posts, real PvB rows,
   and a game log that matches Baseball-Reference for the last 10 starts.
5. Track Record replay over last week's dates shows real box-score lines and a sane
   hit-rate summary.
6. Full suite + structural guards green; ruff clean; page renders for a read-only member
   under MULTI_USER with zero DB writes.
