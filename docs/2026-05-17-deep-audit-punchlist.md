# Deep Audit Punchlist — 2026-05-17

Working punchlist from 6-agent deep audit. Sections are executed sequentially; each gets its own PR. Items checked off as merged.

---

## SECTION 1 — CRITICAL: BROKEN DATA PIPELINE — DONE in PR #32

Live DB shows only ECR data is populated (1,666 rows). Every other Tier-1 source empty. Bootstrap silently failing most phases.

- [x] **C1** — `_safe_add_column gmli` + `CREATE UNIQUE INDEX IF NOT EXISTS idx_statcast_archive_player_season` added to `init_db` migration block.
- [x] **C2** — Added `gmli REAL` column to `statcast_archive` CREATE TABLE for fresh DBs + migration for legacy.
- [x] **C3** — Removed `fip`, `xfip`, `siera` from `live_stats.save_season_stats_to_db` `cols` list. Stuff+ phase is now sole owner of those 3 columns.
- [x] **C4** — Pool SELECT Path 2 (blended fallback) gained `p.bats`, `p.throws`, `ecr.rank_stddev`. Pool SELECT Path 3 (AVG fallback) gained `sa.stuff_plus`, `sa.babip`. All 3 SELECT paths now return identical column sets.
- [x] **C5** — `_bootstrap_yahoo_free_agents` now writes through to `ownership_trends` for every FA (mirroring the existing rostered-player pattern in `yahoo_api.py:1632`).
- [x] **C6** — `_bootstrap_injury_writeback` now guards `UPDATE players SET is_injured=0` behind a check that `league_rosters` has ≥1 IL-status row. If empty (Yahoo disconnected), skip the reset to preserve existing flags.
- [x] **C7** — Diagnosed `data/logs/bootstrap.log`. Most warnings are test mocks from `test_wave8b_silent_failures.py`. Production state: 1 row in players table, 0 in projections/season_stats/statcast_archive/etc, 1666 in ecr_consensus. This is a dev environment without Yahoo OAuth + API access. C1-C6 fixes ensure that when bootstrap runs in production, data flows correctly.

## SECTION 2 — HIGH-SEVERITY LOGIC / MATH BUGS

- [ ] **L1** — `src/engine/output/trade_evaluator.py:677` `weeks_remaining = max(1, 24 - _weeks_elapsed)` → use 26.
- [ ] **L2** — `src/standings_projection.py:57` `n_weeks=22` → 26.
- [ ] **L3** — `src/standings_engine.py:188` projection scaling uses `weeks_remaining` instead of canonical 26.
- [ ] **L4** — `src/start_sit_widget.py:56, 67, 71, 80, 94, 98` — six `/22.0` → `/26.0`.
- [ ] **L5** — `src/start_sit.py:906-907` pitcher rate-stat double-inverse park bug.
- [ ] **L6** — `src/optimizer/daily_optimizer.py:447-450` missing-xFIP defaults to 0 → halves hitter DCV.
- [ ] **L7** — `src/optimizer/matchup_adjustments.py:427` vs `src/start_sit.py:756-759` — reconcile pitcher park inversion conventions.
- [ ] **L8** — `src/valuation.py:446, 531` `sf` computed but missing from OBP marginal denominator.
- [ ] **L9** — `src/bayesian.py:744` `obs_component` discarded.
- [ ] **L10** — `pages/5_Matchup_Planner.py:169` hardcoded 24-week season.
- [ ] **L11** — `src/trade_intelligence.py:1023, 1584`, `src/il_manager.py:64` `weeks_remaining=22` defaults.
- [ ] **L12** — `src/start_sit.py:826` `pitcher_hand="R"` hardcoded default.
- [ ] **L13** — `src/leaders.py:185` `inverse_stats = {"era","whip"}` drops L.
- [ ] **L14** — `src/engine/portfolio/category_analysis.py:33-46` `WEEKLY_RATE_DEFAULTS` based on 22 weeks.

## SECTION 3 — DUPLICATION TO CONSOLIDATE

- [ ] **D1** — `_LEAGUE_AVG_ERA`/`_LEAGUE_AVG_WHIP` 11 sites, 4 values → `CONSTANTS_REGISTRY`.
- [ ] **D2** — Player-name normalization 4 impls → single `normalize_player_name()` in `valuation.py`.
- [ ] **D3** — Team-name → abbrev map 5 inline copies → `valuation.team_name_to_abbr()`.
- [ ] **D4** — `INVERSE_CATS` literal in 6 sites → `LeagueConfig().inverse_stats`.
- [ ] **D5** — `WEEKS_IN_SEASON=26` 3 named copies + ~10 inline → `LeagueConfig.season_weeks`.
- [ ] **D6** — `RATE_STATS` set in 9+ sites → `LeagueConfig.rate_stats`.
- [ ] **D7** — `DEFAULT_HEALTH_SCORE=0.85` 4 sites → `injury_model.py`.
- [ ] **D8** — `_LEAGUE_AVG_WOBA` 6 sites → registry.
- [ ] **D9** — `_safe_float()` 4 impls → `data_fetch_utils.py`.
- [ ] **D10** — `_CAT_DISPLAY` 4 identical dicts → `ui_shared.py`.

## SECTION 4 — DEAD CODE TO DELETE (~2,000 LOC)

35 verified-orphan functions. Highest-LOC items:

- [ ] `src/data_2026.py` — `get_hitter_projections`, `get_pitcher_projections` (327 LOC)
- [ ] `src/engine/game_theory/sensitivity.py` — `player_sensitivity`, `suggest_counter_offers` (205 LOC)
- [ ] `src/optimizer/streaming.py` — `compute_streaming_composite`, `compute_sb_streaming_score` (120 LOC)
- [ ] `src/alerts.py` — `generate_opponent_move_alerts`, `render_alerts_html` (116 LOC)
- [ ] `src/waiver_wire.py:193` — `compute_schedule_aware_streams` (104 LOC)
- [ ] `src/game_day.py:1126` — `get_pitcher_projections`, `fetch_pitcher_pitch_mix` (84 LOC)
- [ ] `src/engine/projections/projection_client.py` — `resolve_trade_players`, `get_ros_projections`, `refresh_projections_if_needed` (84 LOC)
- [ ] `src/engine/portfolio/copula.py` — `compute_empirical_correlation`, `fit_copula_from_data` (79 LOC)
- [ ] `src/database.py:1146` — `import_adp_csv` (77 LOC)
- [ ] `src/validation.py` — `ablation_test`, `generate_cheat_sheet` (73 LOC)
- [ ] `src/ecr.py` — `fetch_prospect_rankings` + 3 (58 LOC)
- [ ] `src/valuation.py:1013` — `bayesian_sgp_update` (48 LOC)
- [ ] 23 smaller orphans (~750 LOC)
- [ ] `src/ecr.py:976-997` — 20-line commented-out block
- [ ] `src/ui_shared.py:528` — `render_theme_toggle` pure-pass stub
- [ ] ~50 dead local-var assignments (some indicate latent bugs — see L8/L9)

## SECTION 5 — UI / PAGE CONSOLIDATION

20 pages → 13 pages, ~2,200 LOC removed:

- [ ] Delete `Weekly_Dashboard` (#8) — 5 tabs duplicate Optimizer + Matchup_Planner (~605 LOC)
- [ ] Fold `Bullpen` (#4) into `Closer_Monitor` (#3) (~182 LOC)
- [ ] Fold `Waiver_Wire` (#15) into `Free_Agents` (#14) (~339 LOC)
- [ ] Fold `Playoff_Odds` (#7) into `League_Standings` (#6) (~376 LOC)
- [ ] Fold `Trade_Values` (#13) into `Trade_Finder` (#12) (~269 LOC)
- [ ] Fold `Trends` (#18) into `Leaders` (#17) (~263 LOC)
- [ ] Fold `Weekly_Recap` (#9) into `Matchup_Planner` (#5) (~218 LOC)
- [ ] Extract `render_position_filter()` → `ui_shared.py` (6 page duplicates)
- [ ] Canonicalize `weeks_remaining()` helper (3 pages, 3 different formulas)

## SECTION 6 — NON-CODE FILES

- [ ] Delete `League Group Chat Screenshots/*.png` (~31 MB)
- [ ] Delete `.pytest_cache/README.md` + add to .gitignore
- [ ] Delete `data/logs/bootstrap.log` from git + .gitignore `data/logs/`
- [ ] Delete `data/draft_tool.db` from git + .gitignore `data/*.db*`
- [ ] Update `README.md` — 20 pages, ~3,900 tests, V4 trade engine, CI 3.12 only
- [ ] Update `docs/architecture.md` — refresh page table, 33 phases, test counts
- [ ] Update `docs/Research.md` — move shipped items to "Already Has"
- [ ] Archive `docs/VERIFICATION_LOG.md` to `docs/archive/`
- [ ] Archive ~15 shipped specs in `docs/superpowers/specs/` to `docs/archive/specs/`

## SECTION 7 — CLAUDE.md OPTIMIZATION

After all sections merged:
- [ ] Compress audit history (Waves 1-9) into a single paragraph
- [ ] Remove "Recent PRs #7-#11" etc. (rotted commit references)
- [ ] Remove "Session totals" line (one-off)
- [ ] Keep canonical sections: Overview, Commands, League Context, Tech Stack, File Structure, Architecture, Key API Signatures, Gotchas, Data Sources, Structural Invariants
- [ ] Reduce length while preserving every truth that prevents bugs
