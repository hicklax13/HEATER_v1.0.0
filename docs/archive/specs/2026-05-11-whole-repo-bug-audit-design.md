# Whole-Repo Bug & Data Audit Design

**Date:** 2026-05-11
**Author:** Brainstorming session with Claude Opus 4.7
**Status:** Approved by user, ready for plan generation

## Goal

Comprehensive audit of HEATER's entire codebase + every consumer-facing data point + the live-refresh infrastructure. Produce a single structured findings report that drives a follow-up fix plan.

## Scope

### In scope
- All Python code in the repo (~155 files): `app.py`, `pages/`, `src/`, `scripts/`, root-level utility scripts
- Config + data-asset files: `pyproject.toml`, `requirements.txt`, `.streamlit/config.toml`, `.github/workflows/*.yml`, `data/seed/*.json`, `scripts/pre-commit`
- All ~1200-1300 unique players that the app actually uses (rostered + top ADP + top FAs)
- All 16+ data sources tracked in `refresh_log`

### Out of scope
- Test files in `tests/` (verification, not behavior under review)
- The SQLite database content itself (`data/draft_tool.db`) — that's content, not code
- Backup files, log files
- Pre-existing 28 SF findings already resolved (SF-1..SF-28) — guarded by structural tests

### Bug categories targeted

| Priority | Category | Caught by |
|---|---|---|
| **PRIMARY** | A — Logic & correctness (wrong math, off-by-one, state-management, control-flow) | `feature-dev:code-reviewer` |
| **PRIMARY** | B — Silent failures (success messages while data is wrong/missing) | `pr-review-toolkit:silent-failure-hunter` |
| **PRIMARY** | D — Type & API design (leaky abstractions, signature inconsistency, fragile interfaces) | `pr-review-toolkit:type-design-analyzer` |
| Secondary | C — Security (SQL injection, secrets in logs, auth bypasses) | Side-effect of A reviewer |
| Secondary | E — Performance (N+1, hot-loop allocations, sync-when-async) | Side-effect of A reviewer |
| Secondary | F — Maintainability (dead code, DRY violations, deprecated APIs) | Side-effect of A + D reviewers |
| Secondary | G — Test coverage gaps | Out-of-scope this pass |

## Three Parallel Streams

### Stream A: Code Review (8 domains × 3 reviewers = 24 agents)

| # | Domain | Files (~) | Contents |
|---|--------|-----------|---|
| 1 | Data Pipeline | 26 | `database.py`, `data_bootstrap.py`, `data_pipeline.py`, `data_fetch_utils.py`, `live_stats.py`, `depth_charts.py`, `contract_data.py`, `marcel.py`, `data_2026.py`, `adp_sources.py`, `player_databank.py`, `espn_injuries.py`, news/prospect/league files, `scheduler.py`, `points_league.py`, `analytics_context.py`, `projection_blending.py`, `trend_tracker.py`, `benchmark_load_times.py`, `debug_yahoo.py` |
| 2 | Yahoo + Valuation Core | 17 | `yahoo_api.py`, `yahoo_data_service.py`, `live_draft_sync.py`, `valuation.py`, `ecr.py`, `projection_stacking.py`, `playing_time_model.py`, `ml_ensemble.py`, `bayesian.py`, `injury_model.py`, `validation.py`, `src/validation/` (5 files), `calibrate_constants.py` |
| 3 | Optimizer | 24 | All 21 `src/optimizer/` modules + `lineup_optimizer.py` + `lineup_rl.py` + `load_sample_data.py` |
| 4 | Trade Engine | 26 | All `src/engine/` sub-packages + `trade_finder.py`, `trade_intelligence.py`, `trade_value.py`, `trade_signals.py` |
| 5 | In-Season Strategy | 23 | `in_season.py`, `opponent_trade_analysis.py`, `waiver_wire.py`, matchup_*, standings_*, war_room_*, start_sit_*, weekly_*, alerts.py, opponent_intel.py, ip_tracker.py, il_manager.py, two_start.py, power_rankings.py, leaders.py, playoff_sim.py |
| 6 | Game Day + Draft + Backtest | 14 | `game_day.py`, `closer_monitor.py`, `schedule_grid.py`, `contextual_factors.py`, all draft_*, `simulation.py`, `pick_predictor.py`, `backtesting.py`, `backtesting_framework.py` |
| 7 | UI / Pages / Scripts | 31 | `app.py` (Connect League), all 21 pages, `ui_shared.py`, `ui_analytics_badge.py`, `cheat_sheet.py`, `player_card.py`, `player_tags.py`, all 8 scripts |
| 8 | Config & Data Assets | 7 | `pyproject.toml`, `requirements.txt`, `.streamlit/config.toml`, `.github/workflows/ci.yml`, `.github/workflows/refresh.yml`, `data/seed/catcher_framing_2024.json`, `data/seed/umpire_tendencies_2024.json` (1 reviewer instead of 3) |

**Per Domain 1-7: 3 parallel agents:**
- `pr-review-toolkit:silent-failure-hunter`
- `feature-dev:code-reviewer`
- `pr-review-toolkit:type-design-analyzer`

**Domain 8: 1 reviewer** (`feature-dev:code-reviewer` — only one needed for non-Python content; no silent-failure or type-design analysis applies to YAML/JSON/TOML).

**Total Stream A agents:** 7 × 3 + 1 = **22 agents**.

### Stream B: Player Data Correctness (1 agent)

**Player set (deduped union ~1200-1300):**
- All ~315 league-rostered players (12 teams × ~26 active+bench, dropping IL)
- Top 1000 by ADP
- Top 100 available batters (Yahoo FAs filtered to is_hitter=1, ranked by SGP descending)
- Top 100 available pitchers (Yahoo FAs filtered to is_hitter=0, ranked by SGP descending)

**Per player, verify against 2 sources:**

**Source 1 — MLB Stats API (canonical truth):**
- name (allowing accent/suffix normalization)
- mlb_id (must match)
- current team (vs `players.team`)
- position(s) (vs `players.positions`)
- injury status (vs `players.is_injured`)
- YTD season stats (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP, GP, PA, IP)

**Source 2 — Yahoo Fantasy API:**
- eligible_positions (vs roster_slot column)
- current roster status (rostered/FA/IL — vs `league_rosters` + `yahoo_free_agents`)
- team_key (vs `league_rosters.team_key`)

**Discrepancy thresholds:**
- **HARD** (auto-flag, severity HIGH): wrong team, wrong mlb_id, position mismatch beyond known equivalence, name mismatch beyond accent/suffix normalization, IL/active status mismatch
- **SOFT** (flag if >2% delta, severity MEDIUM): YTD counting stats off by >1, YTD rate stats off by >0.005

**Date context:** Today is 2026-05-11 (early MLB 2026 season; ~5-6 weeks of game data, ~30 games per active player).

**Estimated runtime:** ~15-30 min wall clock (batched, rate-limited).

### Stream C: Live-Refresh Infrastructure (1 agent)

For each of the 30+ data sources in `refresh_log`:
1. Read current `last_refresh`, `status`, `tier`, `rows_written`, `message`
2. Trigger force-refresh of that source only
3. Verify target table row count > 0 if API returned data; verify increment if cached
4. Verify `refresh_log` updated correctly (status valid, tier accurate, message informative)
5. Verify UI Data Status badge reflects current state (via splash-screen / refresh button)
6. Test stale-trigger: manipulate `refresh_log.last_refresh` backward → verify next bootstrap re-fetches that source

**Sources audited:** players, projections, ros_projections, adp_sources, season_stats, historical_stats, park_factors, yahoo_rosters, yahoo_standings, yahoo_data, yahoo_transactions, yahoo_free_agents, news, news_intelligence, prospect_rankings, ecr_consensus, game_day, team_strength, sprint_speed, batting_stats, stuff_plus, dynamic_park_factors, bat_speed, depth_charts, pvb_splits, catcher_framing, umpire_tendencies, contracts, forty_man, game_logs, draft_results, injury_writeback, extended_roster, injury_data.

**Estimated runtime:** ~2 hours wall clock (most phases serialize; some 60-180s timeouts).

## Dispatch Strategy

### Wave 1 (parallel — ~45 min wall clock for code, ~2h for streams B+C)
- 22 Stream A code-review agents (one parallel batch)
- 1 Stream B player-verification agent
- 1 Stream C infrastructure audit agent

All run simultaneously. Each is read-only — no code changes during this phase.

### Wave 2 (sequential — ~15 min)
- 1 aggregation agent: consolidates all 24 reports into a single structured findings report

### Output deliverable

**Path:** `docs/superpowers/specs/2026-05-11-bug-audit-findings.md`

**Structure:** Sortable findings list. Each entry:
- **ID:** BUG-N (code), DATA-N (data correctness), INFRA-N (refresh)
- **Severity:** HIGH / MEDIUM / LOW
- **Category:** logic / silent-fail / type-design / security / performance / maintainability / data-correctness / freshness / config
- **Domain:** which review domain caught it
- **File:line** (or data point — e.g., "player_id=12345 mlb_id mismatch")
- **Evidence:** specific code excerpt or sample data showing the issue
- **Suggested fix:** one-sentence repair direction
- **Confidence:** reviewer's HIGH/MEDIUM/LOW confidence the finding is real

After Wave 2, the report is the input to a writing-plans skill invocation that produces an executable fix plan.

## Constraints

- All Stream A agents are READ-ONLY — no code changes during audit phase
- Stream B agent uses MLB Stats API at safe rate (~50 req/sec max with retry); does not trigger live Yahoo fetches beyond a single rosters/FA fetch at start
- Stream C agent CAN trigger refreshes (operational, not destructive); does not delete or corrupt data
- Each agent is scoped to its assigned files only — won't drift to other domains
- Each agent worktree-isolated — no merge conflicts during audit since no edits

## Acceptance criteria

The audit is complete when:
1. All 24 reviewer agents return structured reports
2. Stream B returns a discrepancy report covering all ~1200-1300 players
3. Stream C returns infrastructure validation per source
4. Wave 2 aggregator produces the consolidated findings file
5. Findings file is committed to repo
6. User reviews findings and approves moving to fix plan
