# The Last Plan — HEATER v2.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform HEATER from a draft tool with data gaps into a championship-winning in-season fantasy baseball operations platform, fully aligned with the AVIS Operations Manual, with complete Yahoo league integration and provably correct analytics.

**Context:** Connor's league mates are trash-talking him for using AI. The only response is winning. The AVIS manual is the bible. Every feature must serve the mission: finish top 4, win the championship.

**Architecture:** 4-phase plan. Phase 1 fixes the Yahoo FA pipeline (the critical data gap). Phase 2 integrates AVIS rules as hard constraints into every algorithm. Phase 3 builds the matchup intelligence layer. Phase 4 delivers the weekly automation cadence.

---

## Phase 1: Complete Yahoo Free Agent Pipeline (Critical Path)

### Context
Yahoo's `get_league_players()` via yfpy doesn't support `status=FA` filtering. The app currently shows FAs computed from the DB (9,214 players minus 292 rostered = 8,922 "FAs"), but most of those are minor leaguers nobody can add. Yahoo knows which players are actually available in the league. We need to call the Yahoo API directly with `status=FA` to get the real FA pool.

### Proof: HEATER FA Ranking IS Roster-Aware (Not Patronizing)
The marginal SGP ranking in `rank_free_agents()` (src/in_season.py:236) is **mathematically proven** to be roster-aware:
1. It calls `_roster_category_totals(user_roster_ids, full_pool)` to get **Team Hickey's actual aggregate stats**
2. It calls `compute_category_weights(roster_totals, config)` which gives **higher weight to weak categories** (SB gets ~1.5x weight because Team Hickey is weak there per AVIS Section 2.2)
3. Rate stats (AVG, OBP, ERA, WHIP) compute **marginal impact on Team Hickey's aggregate**, not generic value
4. Yahoo's sort is a static preseason ranking — same for all 12 teams, no roster context

**However, limitations exist:** No opponent-awareness, no positional scarcity premium, no schedule context. Phase 3 addresses these.

### Task 1.1: Direct Yahoo API Free Agent Fetch

**Files:**
- Modify: `src/yahoo_api.py`

- [ ] Add `fetch_yahoo_free_agents_direct(count=25, start=0, position=None)` method that calls the Yahoo API directly:
  ```
  URL: https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;status=FA;count=25;start={start};sort=OR
  Headers: Authorization: Bearer {access_token}, Accept: application/json
  ```
- [ ] Add `get_all_yahoo_free_agents(max_players=500)` that paginates the above in batches of 25 with 1.5s delay
- [ ] Parse the Yahoo JSON response to extract: player_name, position, percent_owned, player_key, team
- [ ] Handle 429 rate limiting with exponential backoff

### Task 1.2: Wire Yahoo FAs into Bootstrap + Free Agents Page

**Files:**
- Modify: `src/data_bootstrap.py` (Phase 18 — Yahoo FAs)
- Modify: `pages/5_Free_Agents.py`
- Modify: `src/in_season.py` (rank_free_agents to use Yahoo FA pool)

- [ ] Replace the current "DB minus rostered" FA computation with Yahoo's actual FA list
- [ ] Store Yahoo FAs in a new `yahoo_free_agents` table (player_key, name, position, percent_owned, fetched_at)
- [ ] `rank_free_agents()` should filter to ONLY players that appear in both the Yahoo FA list AND the projections DB
- [ ] Free Agents page shows Yahoo-confirmed FAs ranked by marginal SGP
- [ ] Add "Last synced" timestamp to Free Agents page header

### Task 1.3: Fix Remaining Name Matching Issues

**Files:**
- Modify: `src/live_stats.py` (match_player_id)

- [ ] Add Unicode accent normalization (unicodedata.normalize('NFD') + strip accents) to match "Iván" = "Ivan", "José" = "Jose", etc.
- [ ] Add suffix stripping for "(Pitcher)", "(Batter)" in match_player_id itself (not just in yahoo_api.py sync)

### Verification
- [ ] `python -c "from src.yahoo_api import ...; print(len(get_all_yahoo_free_agents()))"` returns 200+
- [ ] Free Agents page shows only Yahoo-confirmed FAs with marginal values
- [ ] `python -m pytest tests/ -q` — 0 failures

---

## Phase 2: AVIS Hard Constraints (Decision Rules Engine)

### Context
The AVIS manual defines 6 hard decision rules. Currently 0 are enforced in code. These must become guardrails that the app can never violate.

### Task 2.1: 20 IP Weekly Minimum Tracking (AVIS Rule #1)

**Files:**
- Create: `src/ip_tracker.py`
- Modify: `pages/1_My_Team.py`
- Modify: `pages/11_Start_Sit.py`

- [ ] Create `compute_weekly_ip_projection(roster, week_schedule)` that projects total IP for the week based on probable pitcher starts
- [ ] Add red banner to My Team page: "IP Watch: X.X projected IP this week (need 20.0)" with color coding (green/yellow/red)
- [ ] If projected IP < 20 by Wednesday, auto-surface "Emergency SP Stream" recommendations
- [ ] Start/Sit page must show IP budget remaining in the week header

### Task 2.2: Never Carry 0 Closers (AVIS Rule #2)

**Files:**
- Modify: `pages/1_My_Team.py`
- Modify: `src/waiver_wire.py`

- [ ] On My Team page, count rostered players with SV > 5 in projections. If < 2, show red alert: "Closer Alert: Only X closers rostered. AVIS requires minimum 2."
- [ ] Waiver Wire page: if closers < 2, override priority to show available closers first

### Task 2.3: Never Worsen 3+ Categories (AVIS Rule #3)

**Files:**
- Modify: `src/engine/output/trade_evaluator.py`

- [ ] After computing `category_impact`, count categories where delta < -0.1 SGP (meaningful worsening)
- [ ] If 3+ categories worsen, override grade to "F" with message: "AVIS RULE VIOLATION: Trade worsens {N} categories ({list}). Auto-rejected."
- [ ] Add `avis_compliant: bool` field to trade result dict

### Task 2.4: Always Fill Empty Roster Spots (AVIS Rule #4)

**Files:**
- Modify: `pages/1_My_Team.py`

- [ ] On page load, count empty roster slots (compare roster size vs max slots from LeagueConfig)
- [ ] If empty slots > 0, show persistent red banner: "EMPTY ROSTER SPOT: {N} open slots = free value being left on the table. Top recommended fills: [list]"
- [ ] Auto-compute top 3 fills from Yahoo FA pool by marginal SGP

### Task 2.5: Floor Over Ceiling in Regular Season (AVIS Rule #5)

**Files:**
- Modify: `src/valuation.py` (value_all_players)

- [ ] Add `season_phase` parameter: "regular" or "playoff"
- [ ] In regular season: weight consistency (low projection volatility) higher. In playoffs: weight ceiling (high upside) higher
- [ ] This affects trade evaluator, waiver wire, and lineup optimizer recommendations

### Task 2.6: IL Stash Protection (AVIS Rule #6)

**Files:**
- Modify: `pages/10_Waiver_Wire.py`

- [ ] Before showing drop candidates, check if player is on IL with return date within 2 weeks
- [ ] If so, flag as "IL STASH — PROTECTED (returns ~{date})" and exclude from drop suggestions
- [ ] Specifically protect Bieber and Strider per AVIS Section 7

### Verification
- [ ] Trade a player that worsens 3+ categories → grade auto-set to F
- [ ] My Team shows IP projection and closer count
- [ ] Empty roster spots trigger fill recommendations
- [ ] `python -m pytest tests/ -q` — 0 failures

---

## Phase 3: Matchup Intelligence Layer

### Context
AVIS Section 3 defines opponent profiles (Tier 1-4) and Section 4 defines the full schedule. The app currently has ZERO opponent awareness. This phase builds it.

### Task 3.1: Opponent Profile Database

**Files:**
- Create: `src/opponent_intel.py`
- Modify: `src/database.py` (add tables)

- [ ] Add `opponent_profiles` table: team_name, tier (1-4), threat_level, strengths (JSON), weaknesses (JSON), notes
- [ ] Add `league_schedule` table: week, team_a, team_b
- [ ] Populate from AVIS Section 3 (opponent profiles) and Section 4 (schedule)
- [ ] Create `get_current_opponent(week)` → returns opponent name, tier, strengths, weaknesses
- [ ] Create `get_week_number()` → computes current MLB week from date

### Task 3.2: Weekly Matchup Analysis (AVIS Section 2.1)

**Files:**
- Create: `src/matchup_analyzer.py`
- Modify: `pages/12_Matchup_Planner.py`

- [ ] `analyze_weekly_matchup(user_roster, opponent_roster, week)` computes:
  1. Category-by-category projection (both teams)
  2. Likely wins, likely losses, toss-ups
  3. Opponent weaknesses to exploit
  4. Team Hickey vulnerabilities to protect
  5. Streaming recommendations specific to this matchup
- [ ] Matchup Planner page shows this analysis with category breakdown table
- [ ] Include opponent tier badge and threat level

### Task 3.3: Streaming Recommendations (AVIS Section 2.4)

**Files:**
- Modify: `src/waiver_wire.py`
- Create: `src/streaming.py`

- [ ] `recommend_streams(opponent, week, adds_remaining)` returns:
  1. SP with 2 starts this week in favorable matchups
  2. RP closers with high-save-opportunity weeks
  3. Speed hitters if SB is a toss-up category
- [ ] Factor in 10 adds/week budget (5 streaming + 3 injury + 2 reserve per AVIS)
- [ ] Wire into Waiver Wire page as "This Week's Streams" section

### Verification
- [ ] My Team shows "Week 1 vs The Good The Vlad The Ugly (Tier 3 — Medium-Low threat)"
- [ ] Matchup Planner shows category-by-category projections
- [ ] Streaming recommendations appear on Waiver Wire
- [ ] `python -m pytest tests/ -q` — 0 failures

---

## Phase 4: Weekly Automation Cadence

### Context
AVIS Section 5 defines a daily/weekly operating cadence. This phase automates the Monday report, daily lineup checks, Thursday checkpoints, and proactive alerts.

### Task 4.1: Monday Matchup Report Generator

**Files:**
- Create: `src/weekly_report.py`
- Modify: `pages/1_My_Team.py`

- [ ] `generate_monday_report(week)` produces:
  - Prior week results (W/L, category breakdown)
  - Current standings position
  - New week opponent analysis (from Phase 3)
  - Top 5 streaming targets for the week
  - Lineup recommendations for Monday games
- [ ] Show as expandable section on My Team page when it's Monday (or first visit of the week)

### Task 4.2: Daily Lineup Checker

**Files:**
- Modify: `pages/1_My_Team.py`

- [ ] On page load, check today's MLB schedule
- [ ] Flag any rostered player on an off-day who is in a starting lineup slot
- [ ] Flag any bench player who IS playing today and could be swapped in
- [ ] Show as "Today's Lineup Check" banner

### Task 4.3: Thursday Mid-Week Checkpoint

**Files:**
- Modify: `pages/1_My_Team.py`

- [ ] On Thursday, auto-compute:
  - Current matchup category standings (from Yahoo scoreboard)
  - IP total so far this week
  - Categories at risk (close margins)
  - Recommendation: bench risky SP starts if winning ERA/WHIP, or add streaming SP if losing K/W

### Task 4.4: Proactive Alert System

**Files:**
- Create: `src/alerts.py`
- Modify: `pages/1_My_Team.py`

- [ ] Monitor for roster-impacting events on each sync:
  - Closer role changes (check closer_monitor data)
  - Player injuries (from Yahoo injury_note)
  - Trades affecting league rosters
- [ ] Surface as alert cards on My Team page: "ALERT: {player} placed on IL. Recommended action: {pickup}"

### Task 4.5: AVIS Communication Style

**Files:**
- All pages that show recommendations

- [ ] Trade Analyzer: Show category impact math, flag risks first, then upside
- [ ] Waiver Wire: Ranked list with reasoning tied to category impact per AVIS Section 6
- [ ] Start/Sit: Data-driven, no cheerleading. "Start X because matchup projects +0.3 SGP in K" not "X is hot!"

### Verification
- [ ] Monday visit shows matchup report
- [ ] Thursday visit shows mid-week checkpoint
- [ ] Alerts surface for injuries detected in Yahoo sync
- [ ] All recommendation text follows AVIS analyst tone
- [ ] `python -m pytest tests/ -q` — 0 failures
- [ ] Full E2E: restart app, navigate all 13 pages, verify no errors

---

## Critical Files Reference

| File | Phase | Changes |
|------|-------|---------|
| `src/yahoo_api.py` | 1 | Direct API FA fetch with pagination |
| `src/live_stats.py` | 1 | Unicode accent normalization in name matching |
| `src/in_season.py` | 1 | rank_free_agents uses Yahoo FA pool |
| `src/data_bootstrap.py` | 1 | Yahoo FA sync phase |
| `pages/5_Free_Agents.py` | 1 | Show Yahoo-confirmed FAs only |
| `src/engine/output/trade_evaluator.py` | 2 | AVIS Rule #3 hard constraint |
| `src/ip_tracker.py` (new) | 2 | 20 IP weekly minimum tracking |
| `src/waiver_wire.py` | 2, 3 | AVIS priority order + streaming |
| `pages/1_My_Team.py` | 2, 3, 4 | IP watch, closer alert, empty spots, daily check, reports |
| `pages/11_Start_Sit.py` | 2 | IP budget display |
| `pages/10_Waiver_Wire.py` | 2 | IL stash protection |
| `src/opponent_intel.py` (new) | 3 | Opponent profiles + schedule |
| `src/matchup_analyzer.py` (new) | 3 | Weekly matchup analysis |
| `src/streaming.py` (new) | 3 | Streaming SP/RP recommendations |
| `src/weekly_report.py` (new) | 4 | Monday report generator |
| `src/alerts.py` (new) | 4 | Proactive alert system |
| `src/database.py` | 1, 3 | New tables (yahoo_free_agents, opponent_profiles, league_schedule) |
| `AVIS_FANTASY_BASEBALL_OPS_MANUAL_2026.md` | ALL | The bible — every feature references this |

---

## Success Criteria

- [ ] Yahoo FA fetch returns 200+ real league free agents
- [ ] Free Agents page shows ONLY Yahoo-confirmed FAs, ranked by roster-aware marginal SGP
- [ ] All 6 AVIS decision rules are enforced in code
- [ ] Trade that worsens 3+ categories auto-grades to F
- [ ] 20 IP minimum tracked and surfaced with streaming emergency recommendations
- [ ] Opponent profiles and schedule loaded for all 24 weeks
- [ ] Weekly matchup analysis available for current week
- [ ] Streaming recommendations factor in 10 adds/week budget
- [ ] Monday report, daily lineup check, Thursday checkpoint all functional
- [ ] All 2022+ tests passing, 0 failures
- [ ] Full E2E: all 13 pages load without errors
- [ ] Connor's friends stop laughing by Week 4
