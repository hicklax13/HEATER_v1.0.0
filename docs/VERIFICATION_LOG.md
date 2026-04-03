# HEATER Full Verification Log

> Started: 2026-04-03 ~4:30 PM EDT
> Method: Claude in Chrome + Playwright MCP (real + headless browser)
> App: http://localhost:8501
> Status: PARTIAL — 2 pages deep-verified, 9 pages surface-verified, browser tools died mid-session
> Resume: Pick up interactive testing in fresh session

## Known Issues (from prior testing)

| ID | Page | Severity | Description | Status |
|----|------|----------|-------------|--------|
| V-001 | Closer Monitor | Cosmetic | Raw `<div style="f` HTML tags leak in 3 closer cards (ARI, ATH, ATL) | OPEN |
| V-002 | Leaders | Data | "No data available" for Category Leaders — live stats >24h stale | OPEN |
| V-003 | My Team | Data | Matchup score differs between page (4-5-3) and direct Yahoo API call (6-4-2) — caching/timing | OPEN |
| V-004 | My Team | Data | Closer Alert says "0 closers" but roster has Devin Williams, Robert Suarez, Raisel Iglesias (all RP with SV projections) | NEEDS INVESTIGATION |

## Page-by-Page Deep Verification

### Connect League (app.py)
- [ ] Bootstrap splash screen completes
- [ ] Yahoo Fantasy Connected badge shows
- [ ] Data Status expander opens and shows all sources
- [ ] League Format card: 12 Teams, 23 Rounds, Snake, H2H
- [ ] Player Pool card: shows count + Yahoo Connected
- [ ] SGP auto-compute toggle works
- [ ] Risk Tolerance slider moves
- [ ] Next button navigates to Step 2
- [ ] Force Refresh Data button visible in sidebar

### My Team (1_My_Team.py)
- [ ] Team name "Team Hickey" displays
- [ ] Roster count: 26 players (15H/11P)
- [ ] Refresh Stats button works
- [ ] Sync Yahoo button works
- [ ] Matchup banner shows correct Week 2 opponent
- [ ] Category Breakdown expander opens
- [ ] Opponent analysis banner shows tier + strengths/weaknesses
- [ ] IP Watch banner shows projected IP
- [ ] Closer Alert shows (verify closer count accuracy)
- [ ] IL Stash alerts for Strider and Bieber
- [ ] Roster table renders with player photos + stats
- [ ] Category totals (hitting + pitching) show real numbers
- [ ] Category Gaps section shows priority targets

### Draft Simulator (2_Draft_Simulator.py)
- [ ] Draft Settings: teams=12, rounds=23, position=1
- [ ] Simulation Depth radio buttons (50/100/200)
- [ ] Engine Mode radio buttons (Quick/Standard/Full)
- [ ] Start Mock Draft button initiates draft
- [ ] AI opponents pick players
- [ ] Recommendations appear for user's pick
- [ ] Draft board updates after each pick

### Trade Analyzer (3_Trade_Analyzer.py)
- [ ] You Give dropdown populates with team roster
- [ ] You Receive dropdown populates with other teams' players
- [ ] Analyze Trade button produces grade
- [ ] Category impact table displays
- [ ] Acceptance Analysis panel shows ADP/ECR fairness
- [ ] Data Freshness panel shows source status

### Free Agents (4_Free_Agents.py)
- [ ] Top pickup banner shows recommendation
- [ ] Recommended Adds/Drops table has real data
- [ ] Net SGP Delta values are reasonable (positive)
- [ ] Sustainability % shows for each recommendation
- [ ] Position filter buttons (ALL, C, 1B, 2B, etc.) filter the table
- [ ] "Why" expanders open with reasoning
- [ ] Analysis Quality indicator shows

### Lineup Optimizer (5_Lineup.py)
- [ ] OPTIMIZE tab: Optimize Lineup button runs and produces result
- [ ] START/SIT tab: Shows advisor recommendations
- [ ] CATEGORY ANALYSIS tab: Shows SGP weights
- [ ] HEAD-TO-HEAD tab: Shows H2H win probabilities
- [ ] STREAMING tab: Shows streaming targets
- [ ] Optimization Mode radio buttons work (Quick/Standard/Full)
- [ ] H2H vs Season-Long slider works

### Player Compare (6_Player_Compare.py)
- [ ] Search Player A text input works
- [ ] Search Player B text input works
- [ ] Selecting two players shows comparison
- [ ] Radar chart renders
- [ ] Category breakdown table shows

### Closer Monitor (7_Closer_Monitor.py)
- [ ] 30-team grid renders
- [ ] Job security % displays per team
- [ ] SV/ERA/WHIP stats show
- [ ] 2026 actual stats display where available
- [ ] Check HTML leak in ARI/ATH/ATL cards

### Standings (8_Standings.py)
- [ ] PROJECTED STANDINGS tab: 12 teams with projected stats
- [ ] Number of Simulations slider works
- [ ] POWER RANKINGS tab: Shows rankings with confidence
- [ ] Data Freshness panel shows

### Leaders (9_Leaders.py)
- [ ] CATEGORY LEADERS tab: Dropdown changes category
- [ ] Data displays for selected category (or "No data" if stale)
- [ ] POINTS LEADERS tab: Shows points-based rankings
- [ ] PROSPECTS tab: Shows prospect rankings
- [ ] Scoring Preset dropdown works

### Trade Finder (10_Trade_Finder.py)
- [ ] BY PARTNER tab: Teams listed with trade suggestions
- [ ] BY CATEGORY NEED tab: Trades grouped by category
- [ ] BY VALUE tab: All trades ranked by composite score
- [ ] TRADE READINESS tab: Per-player readiness scores
- [ ] Category Needs sidebar shows weak categories
- [ ] Best Trade Partners sidebar shows ranked teams
- [ ] Scan Summary shows trade count + teams scanned
- [ ] Trade table shows You Give/Receive/SGP gains

### Matchup Planner (12_Matchup_Planner.py)
- [ ] Days to look ahead dropdown works (7/14)
- [ ] Player type radio (All/Hitters/Pitchers) filters
- [ ] Team dropdown filters to selected team
- [ ] SUMMARY tab: Shows all players with Rating + Tier
- [ ] PER-GAME DETAIL tab: Shows per-game matchup data
- [ ] HITTERS ONLY tab: Filters to hitters
- [ ] PITCHERS ONLY tab: Filters to pitchers
- [ ] Smash/Avoid counts in header are accurate
- [ ] Rating values are reasonable (0-10 scale)

## Deep Verification Results (Claude in Chrome)

### Connect League (app.py) — PASS
- [x] Bootstrap splash screen completes (21 phases, all green checkmarks)
- [x] Yahoo Fantasy Connected badge shows (green checkmark)
- [x] Data Status expander opens — all 21 sources shown with status
- [x] Live Stats: "Saved 791 player stats" (freshly fetched)
- [x] ROS Projections: "Updated 9206 ROS projections"
- [x] Game Day: Fresh (new Phase 20)
- [x] Team Strength: Fresh (new Phase 21)
- [x] League Format card: 12 Teams, 23 Rounds, Snake, H2H Categories
- [x] Player Pool card: 12,958 (up from 12,957 — new player from refresh)
- [x] SGP auto-compute toggle: ON (clickable)
- [x] Risk Tolerance slider: 0.5 (Moderate label shows)
- [x] Draft Engine Mode radio: Quick/Standard/Full all clickable (tested Full, confirmed switch)
- [x] Next button visible
- [x] Depth Charts: "no data" — FanGraphs scrape returned empty (known FG issue)
- [x] Force Refresh Data button visible in sidebar

### My Team (1_My_Team.py) — PARTIAL (Chrome connection lost mid-test)
- [x] Team name "Team Hickey" displays
- [x] Roster count: 26 players (15H/11P) — correct
- [x] Refresh Stats button visible
- [x] Sync Yahoo button visible
- [x] Matchup banner: Week 2 vs Baty Babies (Tier 2 — Medium threat) | Strengths: HR, AVG, SB | Weaknesses: W, K, ERA
- [x] IP Watch: 5.22/20 (26% pace) — DANGER warning correctly shown
- [x] Closer Alert: "Only 0 closer(s) rostered" — **SEE V-004**
- [x] Injury alerts: Kirby Yates (IL15, Knee), Nick Lodolo (IL15, Finger), Bryson Stott (IL15)
- [x] IL Stash alerts: Spencer Strider, Shane Bieber — "Do NOT drop within 2 weeks"
- [x] Roster table renders with player photos + live 2026 stats
- [x] Yordan Alvarez: .417 AVG, 8R, 3HR, 6RBI — matches Yahoo API data
- [x] Hitting totals: R=39, HR=12, RBI=38, SB=7, AVG=.233, OBP=.311
- [x] Pitching totals: W=4, L=2, SV=1, K=63, ERA=1.67, WHIP=1.00
- [x] Category Gaps: Priority Targets = SV, W (makes sense given SV=1, W=4)
- [x] 5 season tabs: 2026 Live, 2026 Projected, 2025, 2024, 2023
- [x] Weekly Report expander: "Week 2 vs Baty Babies"
- [ ] Refresh Stats button click — NOT TESTED (Chrome connection lost)
- [ ] Sync Yahoo button click — NOT TESTED
- [ ] Season tab switching (2025, 2024, 2023) — NOT TESTED

## New Issues Found During Deep Verification

| ID | Page | Severity | Description | Status |
|----|------|----------|-------------|--------|
| V-005 | Connect League | Data | Depth Charts: "no data" — FanGraphs scrape returned empty | OPEN (known FG issue) |
| V-006 | My Team | Logic | Injury alerts show Kirby Yates and Nick Lodolo — these are NOT on Team Hickey's roster. Alert system may be showing league-wide injuries instead of roster-only | NEEDS INVESTIGATION |
| V-007 | My Team | Logic | Closer Alert says "0 closers" but roster has 3 RPs (Williams, Suarez, Iglesias). Detection checks SV > 5 in projections — relievers may not have enough projected saves to trigger. Actual 2026 SV = 1 (low). May be working correctly if projections show low SV. | NEEDS INVESTIGATION |
| V-008 | Chrome | Infra | Claude in Chrome connection timed out after ~15 minutes of continuous use on My Team page. May need reconnection for remaining pages. | INTERMITTENT |

