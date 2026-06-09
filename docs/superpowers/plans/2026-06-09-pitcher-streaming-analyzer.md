# Implementation Plan — Pitcher Streaming Analyzer page

> Status: **draft, pending design approval**
> Date: 2026-06-09
> Design: `docs/superpowers/specs/2026-06-09-pitcher-streaming-analyzer-design.md`
> Branch: `claude/pitcher-streaming-analyzer-gnimrm`

Discipline for every phase: write the failing test first, implement to green, run
`python -m ruff format . && python -m ruff check .`, run the touched test files plus the
structural-invariant suite, commit. Phases are sized to be independently shippable
commits; the page becomes user-visible only in Phase 2.

---

## Phase 0 — Constants + module skeleton

**Goal:** every tunable lands in `CONSTANTS_REGISTRY` before any scoring code exists.

1. `tests/test_stream_analyzer_constants_registered.py` (NEW, failing):
   - Each of these keys exists in `CONSTANTS_REGISTRY` with `value`, `citation`,
     `lower_bound`, `upper_bound`, `sensitivity`:
     `stream_score_w_matchup` (0.35), `stream_score_w_sgp` (0.25),
     `stream_score_w_form` (0.15), `stream_score_w_lineup` (0.10),
     `stream_score_w_env` (0.10), `stream_score_w_winprob` (0.05),
     `stream_risk_short_leash_ip` (4.5), `stream_risk_elite_offense_wrc` (110),
     `stream_risk_hitter_park` (1.08), `stream_risk_wind_out_mph` (12.0).
   - Weights sum to 1.0 within 1e-9.
2. `tests/test_stream_analyzer_adds_budget_canonical.py` (NEW, failing):
   - `src/optimizer/stream_analyzer.py` (and its callers) source the weekly add budget
     from `src.league_rules.WEEKLY_ADDS_BUDGET == 10`; AST check that the module does not
     import `WEEKLY_ADDS_BUDGET` from `src.optimizer.streaming` (stale 7).
   - `streaming.py` call sites that apply the last-add penalty accept the budget as a
     parameter (default to league_rules value) — fixes the penalty firing 3 adds early.
3. Implement: register constants in `src/optimizer/constants_registry.py`; create
   `src/optimizer/stream_analyzer.py` with dataclass-free public stubs raising
   `NotImplementedError` is NOT acceptable — implement `get_opponent_offense_context`
   here (pure dict math over `team_strength`, `split_source` fallback tagging) since it
   has no heavy deps, stub the rest behind Phase 1.
4. Files: `src/optimizer/constants_registry.py`, `src/optimizer/stream_analyzer.py`,
   `src/optimizer/streaming.py` (budget param), 2 new test files.

**Commit:** `feat(stream): registry constants + stream_analyzer skeleton + canonical add budget`

---

## Phase 1 — Scoring engine

**Goal:** `score_stream_candidate` + `build_stream_board` fully tested, no UI.

1. `tests/test_stream_analyzer_score_components.py` (NEW, failing):
   - `score_stream_candidate` returns `stream_score ∈ [0, 100]`, `components` dict with
     exactly the six component keys, `risk_flags` list, `expected_line` dict.
   - Weights are read from `CONSTANTS_REGISTRY` at call time (patch the registry, score
     changes — the sigmoid-calibrator pattern from `test_sigmoid_calibrator_patches_registry.py`).
   - A soft-tossing FA vs the league's best offense in a hitter park scores below the
     same pitcher vs the worst offense in a pitcher park (monotonicity).
   - Missing lineup/PvB data ⇒ lineup component contributes neutrally, not NaN.
   - `HIGH_WHIP`, `ELITE_OFFENSE`, `HITTER_PARK`, `WIND_OUT`, `SHORT_LEASH`,
     `LOW_CONFIDENCE` flags each fire on a constructed fixture and not on a clean one.
2. `tests/test_stream_analyzer_inverse_stats.py` (NEW, failing):
   - The SGP component for a pitcher with ERA 9.0 / WHIP 2.0 is NEGATIVE; for an ace it
     is positive (locks the FA PR #99 lesson). AST guard: no inline
     `(stat / denom) * weight` arithmetic over ERA/WHIP/L in `stream_analyzer.py` — SGP
     flows through `compute_streaming_value` / `SGPCalculator` paths.
   - No hardcoded category list (derive from `LeagueConfig`) — keeps
     `test_no_hardcoded_categories_in_src.py` extendable to this module.
3. `tests/test_stream_analyzer_locked_games.py` (NEW, failing):
   - `build_stream_board(ctx, target_date=today)` marks rows whose game status ∈
     `LOCKED_GAME_STATUSES` with `status="LOCKED"` and `actionable=False`; rows are
     retained (transparency), and the actionable subset excludes them.
   - Future-date boards never mark LOCKED; rows carry the two_start.py confidence tier.
   - Two-start pitchers carry `num_starts=2` and the second start's fatigue-adjusted
     rates (0.93×) in `expected_line`.
4. Implement in `src/optimizer/stream_analyzer.py`, composing:
   `compute_pitcher_matchup_score` (two_start), `compute_streaming_value` +
   `compute_bayesian_stream_score` (streaming), `park_factor_adjustment` +
   `get_pvb_matchup_data` + `pvb_matchup_adjustment` (matchup_adjustments),
   `get_player_recent_form_cached` form delta, `LOCKED_GAME_STATUSES` (game_day).
   No streamlit import in this module (engine purity, mirrors fa_recommender).

**Commit:** `feat(stream): score_stream_candidate + build_stream_board engine`

---

## Phase 2 — Page skeleton + Stream Finder tab (page goes live)

**Goal:** `pages/4_Pitcher_Streaming.py` with full boilerplate + Tab 1; all page-count
guards updated in the SAME commit (they fail otherwise).

1. Update existing structural tests FIRST (they will fail red until the page exists —
   that is the failing-test step):
   - `tests/test_pages_have_auth_guard.py`, `tests/test_pages_have_feedback_and_usage.py`,
     `tests/test_admin_pages_flag_enforced.py`, `tests/test_pages_guard_set_page_config.py`:
     count 13 → 14.
   - `tests/test_nav.py`: `groups["Season"]` 13 → 14.
   - `tests/test_pages_use_viewer_team_resolver.py`: add `4_Pitcher_Streaming.py` to
     `_PERSONALIZED_PAGES`.
2. `tests/test_stream_page_no_inline_scoring.py` (NEW, failing): AST check that
   `pages/4_Pitcher_Streaming.py` contains no arithmetic over ERA/WHIP/K columns used to
   build a score/rank (allowlist: display formatting via `format_stat`); all scoring
   symbols imported from `src.optimizer.stream_analyzer` / `src.optimizer.fa_recommender`.
3. Implement the page:
   - Boilerplate in guard order: guarded `st.set_page_config(page_title="Heater | Pitcher
     Streaming", ...)` → `init_db()` → `inject_custom_css()` → `require_auth()` →
     `require_page_enabled("page:4_Pitcher_Streaming")` → `log_page_view("Pitcher
     Streaming")` → `page_timer_start()`; `render_feedback_widget()` at EOF.
   - `src/nav.py`: PAGE_REGISTRY entry `{"key": "4_Pitcher_Streaming", "title": "Pitcher
     Streaming", "path": "pages/4_Pitcher_Streaming.py"}` between 3 and 5.
   - `src/ui_shared.py`: new `PAGE_ICONS` SVG entry.
   - Viewer resolution + ctx: `resolve_viewer_team_name(rosters)` (with frame),
     `get_team_roster`, `build_optimizer_context(scope="today", user_team_name=...,
     roster=user_roster, level_filter="MLB only")` — the Free Agents page pattern.
   - Tab 1 per design §4: date selector, board via `build_stream_board`, header strip
     (adds remaining / IP pacing / in-play cats), swap recs via
     `recommend_streaming_moves` when date == today, why-expanders, LOCKED chips,
     `format_stat` everywhere.
4. Run the FULL structural-invariant suite (pre-push hook set) — the new page must pass
   `test_pages_format_compliance`, `test_no_merge_conflict_markers`, etc. by construction.

**Commit:** `feat(stream): Pitcher Streaming page + Stream Finder tab (14th page)`

---

## Phase 3 — Matchup Microscope tab

1. Tests (extend `tests/test_stream_analyzer_score_components.py` or NEW
   `tests/test_stream_analyzer_history.py`, failing):
   - `get_pitcher_vs_team_history` filters game-log rows to the opponent / venue and
     aggregates via the pitching aggregation used by `backtest_runner`
     (`era = er*9/ip`, `whip = (bb+h)/ip` — weighted, never averaged).
   - Empty/failed statsapi fetch ⇒ empty DataFrame, no raise (network guard in conftest
     makes this deterministic — the fetch must catch `NetworkBlockedError`-class errors
     through the existing fallback idiom).
2. Implement Microscope UI: candidate selector, opposing-lineup card (confirmed via
   `get_todays_lineups` else depth-chart projection, platoon exposure), PvB table
   (regressed wOBA), game-log table behind an explicit "Load game logs" interaction with
   `st.cache_data(ttl=3600)`, environment card, expected-line card.

**Commit:** `feat(stream): Matchup Microscope deep-dive tab`

---

## Phase 4 — Week Planner tab

1. Tests (NEW `tests/test_stream_week_planner.py`, failing):
   - Planner consumes `optimal_streaming_schedule` output and never plans more adds than
     `ctx.adds_remaining_this_week`.
   - Plan summary reports projected IP added vs the 54-IP target and flags if the week
     still projects under the 20-IP floor (`src/ip_tracker.py` constants, not literals).
   - Two-start entries appear at most once in the sequence.
2. Implement: Mon-Sun grid (reuse `build_schedule_grid` tier colors), greedy sequence
   rendering, IP/adds budget strip shared with Tab 1.

**Commit:** `feat(stream): week planner under add-budget + IP pacing`

---

## Phase 5 — Track Record tab + DRY the Lineup Optimizer tab

1. Tests (NEW `tests/test_stream_replay.py`, failing):
   - `replay_stream_date` returns `proxy_caveat=True` and the UI renders the caveat (AST:
     page references `proxy_caveat`).
   - Actuals aggregation matches the Phase-3 weighted-rate idiom; dates with no games ⇒
     empty result, no raise.
2. Tests for the DRY refactor (extend `test_stream_page_no_inline_scoring.py` with a
   second target): the Streaming tab section of `pages/2_Line-up_Optimizer.py` no longer
   contains its lines-3333-3415 inline score formula; it imports `build_stream_board`.
   (Behavioral parity spot-check: top-3 ordering on a fixture is sane, not identical —
   the formulas intentionally differ.)
3. Implement Tab 4 (replay + "my streams" from `yds.get_transactions()`), then refactor
   the Lineup Optimizer Streaming tab's Section 1 to delegate to `build_stream_board`
   (sections 2-3 of that tab untouched).

**Commit (2):** `feat(stream): track-record replay tab` /
`refactor(lineup): streaming tab delegates scoring to stream_analyzer`

---

## Phase 6 — Docs + verification

1. `CLAUDE.md`: pages list 13 → 14 (add `4_Pitcher_Streaming.py` line + update the
   Section-5 consolidation note), File Structure, Structural Invariants table rows for
   the 6 new guard tests, Key API Signatures (`build_stream_board`,
   `score_stream_candidate`), Gotchas if any surfaced.
2. Full suite locally (`-n auto --dist loadfile`), ruff format + check.
3. Browser verification session against live data per design §10; log as V-018 in
   `docs/VERIFICATION_LOG.md`.
4. PR with the design + plan docs linked; reviews: `code-review` + silent-failure pass
   (the engine touches statsapi fetch paths — every fetch needs the 3-tier/fallback
   idiom and refresh-log-free page-level caching only).

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Board render blows the gameday budget (statsapi schedule + probables for +7 days) | One `get_weekly_schedule` call cached per session; game logs strictly lazy + cached; no per-row network calls during board build |
| Probables churn (LOW tier) misleads users | Confidence chips + tier-based row styling; LOW rows sort below equal-score MED/HIGH |
| Replay proxy scoring oversells accuracy | `proxy_caveat` rendered on-tab; matchup components (schedule/opponent/park) are historically exact and labeled as such |
| Duplicate scoring drifts back into pages | `test_stream_page_no_inline_scoring.py` covers both this page and the Lineup Optimizer streaming section after Phase 5 |
| MULTI_USER member triggers writes | Page has no write paths; if a refresh button is ever added it joins `test_refresh_buttons_admin_gated.py` |
| Page-count guards forgotten | Phase 2 updates all five count assertions in the same commit as the page file |

## Estimated scope

~1 new engine module (~450-600 lines), 1 new page (~700-900 lines), 8 new test files
(~6 engine/page guards + planner + replay), 6 existing test files touched, 3 src files
touched (`nav.py`, `ui_shared.py`, `constants_registry.py`, plus `streaming.py` budget
param), CLAUDE.md. Roughly 6 commits across 6 phases.
