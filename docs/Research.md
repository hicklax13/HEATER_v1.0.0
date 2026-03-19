# FantasyPros vs. HEATER — Gap Analysis

> **Date:** 2026-03-19
> **Purpose:** Identify FantasyPros features NOT currently built into the HEATER app
> **Scope:** Fantasy Baseball only (MLB)

---

## FantasyPros Overview

FantasyPros (fantasypros.com) is the industry-leading fantasy sports platform, offering tools across NFL, MLB, and NBA. For baseball, they provide draft preparation, in-season management, research, and DFS tools across four pricing tiers: Free, PRO ($3.99/mo annual), MVP ($5.99/mo annual), and HOF ($8.99/mo annual). Their core differentiator is **Expert Consensus Rankings (ECR)** — aggregated rankings from 50-100+ human fantasy experts, plus deep platform integrations with Yahoo, ESPN, CBS, Sleeper, and Fantrax.

---

## What HEATER Already Has (Parity or Better)

These FantasyPros features are already implemented in HEATER, often with more sophisticated analytics:

| FantasyPros Feature | HEATER Equivalent | HEATER Advantage |
|---------------------|-------------------|------------------|
| Mock Draft Simulator | Draft Simulator page + Practice Mode | Monte Carlo simulation, opponent modeling with historical preferences |
| Consensus Projections (12 systems) | 7-system blending (Steamer/ZiPS/DC/ATC/BAT/BATX/Marcel) | Bayesian updater, percentile forecasting (P10/P50/P90), process risk modeling |
| ADP Data | Multi-source ADP (FanGraphs, FantasyPros ECR, NFBC, Yahoo) | Fuzzy name matching, staleness-based refresh |
| Trade Analyzer | 6-Phase Trade Engine | Gaussian copula MC, game theory (Vickrey/adverse selection/DP), signal intelligence (Statcast/Kalman/regime), far beyond FantasyPros |
| Trade Finder | Trade Finder (cosine dissimilarity) | 3-tier evaluation, loss-aversion modeling (Kahneman), LP-verified swaps |
| Waiver Wire Assistant | 7-stage Waiver Wire pipeline | LP removal cost, sustainability filtering (BABIP/xStats), multi-move greedy optimization |
| Lineup Optimizer | 11-module Advanced Optimizer | Stochastic MIP, Gaussian copula scenarios, multi-period planning, dual H2H/Roto objective, maximin robust LP |
| Start/Sit | 3-layer Start/Sit Advisor | Category-level impact, matchup-state adaptation (winning/losing strategy shift), risk tolerance adjustment |
| Two-Start Pitchers | Two-Start Planner | Matchup scoring (K-BB%/xFIP/CSW%), rate damage quantification, confidence tiers |
| Player Comparison | Player Compare page | Z-score normalization, radar charts, marginal SGP team impact, health badges |
| Injury Reports | Injury Model + Health Badges | 3-year health scores, age-risk curves, workload flags, Weibull duration modeling |
| Park Factors | Park factors in DB + contextual adjustments | Integrated into draft engine, trade engine, and lineup optimizer pipelines |
| Yahoo League Sync | Yahoo API (yfpy v17+) | Full roster/standings/FA/draft results sync with auto-reconnect |
| Post-Draft Analysis | Draft Grader | 3-component grading (value/efficiency/balance), steal/reach detection, position-dependent thresholds |
| Player News | News Fetcher + Sentiment | MLB transaction feed + sentiment scoring (positive/neutral/negative) |
| Team Overview | My Team page | Health badges, Bayesian projection indicators, Yahoo sync button |

---

## FEATURES FANTASYPROS HAS THAT HEATER DOES NOT

### Priority 1: High-Impact Features (Would Significantly Improve HEATER)

#### 1. Expert Consensus Rankings (ECR) Integration
- **What it is:** Aggregated rankings from 50-100+ human fantasy experts (not projection systems). ECR is FantasyPros' signature product — the "wisdom of crowds" applied to fantasy baseball rankings.
- **Why it matters:** Projection systems capture statistical models; ECR captures expert intuition about role changes, coaching decisions, and subjective factors that models miss. FantasyPros' "Most Accurate Expert" tracking adds accountability.
- **Free/Paid:** Free (basic view), PRO+ (custom filtering)
- **Implementation idea:** Scrape or API-fetch FantasyPros ECR data and blend with existing projection-based rankings as an additional signal. Could weight by expert accuracy scores.

#### 2. Multi-Platform League Sync (ESPN, CBS, Sleeper, Fantrax)
- **What it is:** One-click league import from ESPN, CBS, Sleeper, Fantrax, RTSports, and NFBC — not just Yahoo.
- **Why it matters:** Yahoo is only ~25% of the fantasy baseball market. ESPN and CBS together represent another ~40%. Without multi-platform support, HEATER is inaccessible to most fantasy players.
- **Free/Paid:** Free (league import)
- **Implementation idea:** ESPN has an undocumented API (espn-api Python package). CBS and Sleeper have public APIs. Fantrax has a developer API. Add platform-specific sync modules alongside yahoo_api.py.

#### 3. Live Draft Assistant with Real-Time Sync
- **What it is:** Syncs with your actual live draft on Yahoo/ESPN/CBS/Sleeper in real-time. Auto-detects when picks are made and provides instant recommendations without manual entry.
- **Why it matters:** HEATER's draft tool requires manual pick entry. During a real draft with time pressure, the friction of entering each pick manually is a significant UX disadvantage. FantasyPros charges $5.99/mo+ for this — it's their #1 premium upsell.
- **Free/Paid:** MVP+ ($5.99/mo)
- **Implementation idea:** Yahoo Draft API supports real-time draft tracking. Poll every 5-10 seconds during active draft, auto-update DraftState when new picks detected.

#### 4. Projected Season Standings
- **What it is:** After a draft (mock or real), projects full-season standings for all 12 teams based on roster composition and projections.
- **Why it matters:** Answers the #1 question after any draft: "Where does my team rank?" HEATER's Draft Grader grades individual picks but doesn't project team-level outcomes across the full season.
- **Free/Paid:** Free (basic), PRO+ (advanced)
- **Implementation idea:** Use existing blended projections + roster assignments to compute per-team season totals → H2H win probability matrix → projected W-L records and standings.

#### 5. Keeper / Dynasty League Support
- **What it is:** Rankings, valuations, and draft tools tailored for keeper and dynasty formats — accounting for player age, contract status, prospect value, and multi-year roster planning.
- **Why it matters:** Keeper and dynasty leagues are the fastest-growing segment of fantasy baseball. These formats require fundamentally different valuation (age curves, prospect rankings, surplus value over keeper cost). HEATER currently supports single-season redraft only.
- **Free/Paid:** Rankings free, draft tools MVP+ ($5.99/mo)
- **Implementation idea:** Add keeper_cost and years_remaining columns to player_pool. Modify valuation.py to compute surplus value (projected_value - keeper_cost). Integrate prospect rankings from existing sources.

#### 6. Auction / Salary Cap Calculator + Simulator
- **What it is:** Custom auction dollar values tailored to league settings (team count, budget, roster). Plus a full auction mock draft simulator with realistic AI bidding.
- **Why it matters:** ~20% of competitive fantasy baseball leagues use auction format. HEATER's snake-draft-only architecture excludes this entire segment. FantasyPros charges MVP+ for the simulator.
- **Free/Paid:** Calculator free, Simulator MVP+ ($5.99/mo)
- **Implementation idea:** Convert SGP-based player values to auction dollar values using formula: `auction_$ = (player_SGP / total_SGP) * total_budget`. Add auction DraftState variant with budget tracking.

---

### Priority 2: Medium-Impact Features (Would Improve UX and Completeness)

#### 7. Printable/Exportable Cheat Sheets
- **What it is:** Custom cheat sheets with tiers, player tags (sleeper/target/avoid), personal notes, and print/PDF export.
- **Why it matters:** Many drafters still print cheat sheets or use them on a second screen. The ability to annotate and customize before draft day is a key prep ritual. "Cheat sheet" is the #1 Google search term for fantasy baseball tools.
- **Free/Paid:** Basic free, customization PRO+ ($3.99/mo)
- **Implementation idea:** Generate a styled HTML/PDF cheat sheet from player_pool data with user-defined tiers and tags. Use Streamlit's download button for PDF export.

#### 8. Player Tags (Sleeper / Target / Avoid)
- **What it is:** User-assignable labels on players for quick reference during drafts. Tags like "Sleeper", "Target", "Avoid", "Breakout", "Bust" with optional notes.
- **Why it matters:** Pre-draft research produces insights that need to be captured and surfaced during the draft. Without tags, that research is lost in the moment.
- **Free/Paid:** Free (basic), PRO+ (persistent)
- **Implementation idea:** Add a `player_tags` table (player_id, tag, note, created_at). Surface tags as badges on hero card and player search results.

#### 9. Closer Depth Chart Visualization
- **What it is:** All 30 MLB teams' closer situations displayed in a single view: current closer, job security rating (1-10), next-in-line relievers, committee situations.
- **Why it matters:** Saves (RP) is the most volatile and situationally-dependent fantasy category. Closer changes happen weekly. A dedicated view for closer situations is essential for RP management. HEATER has `contextual_factors.py` with closer hierarchy logic but no dedicated UI page.
- **Free/Paid:** Free
- **Implementation idea:** New page or tab using existing depth_charts.py closer hierarchy data. Add job_security_score to RP entries. Display as a 30-team grid with color-coded security levels.

#### 10. Streaming Pitcher Daily Recommendations
- **What it is:** Daily "best streaming pitcher" recommendations based on matchup quality, park factors, and opponent batting stats.
- **Why it matters:** Streaming pitchers is the #1 in-season strategy for maximizing pitching categories in H2H. HEATER's optimizer has streaming detection but doesn't surface daily recommendations as a standalone feature.
- **Free/Paid:** Free
- **Implementation idea:** New tab in Lineup Optimizer page using existing `src/optimizer/streaming.py` and `src/two_start.py` logic. Show top 5 streamers per day with matchup grades.

#### 11. Schedule Grid / Weekly Planner View
- **What it is:** Visual 7-day schedule showing all your players' games, off days, and matchups in a calendar grid format.
- **Why it matters:** Knowing which days your players have games (and which don't) is critical for lineup optimization and streaming decisions. HEATER's matchup_planner.py has per-game ratings but no visual calendar grid.
- **Free/Paid:** Free
- **Implementation idea:** Build an HTML table grid (7 columns = days, rows = roster players) using MLB schedule data from live_stats.py. Color-code by matchup quality from matchup_planner.py ratings.

#### 12. League Power Rankings
- **What it is:** Analysis of every team in the league with power rankings, strengths/weaknesses breakdown, VORP ratings, and projected category leaders.
- **Why it matters:** Understanding where your team ranks relative to the whole league — and which teams are strongest in which categories — informs trade strategy and weekly lineup decisions.
- **Free/Paid:** HOF ($8.99/mo)
- **Implementation idea:** Use existing league_standings + league_rosters + blended projections to compute per-team SGP totals → rank teams → identify per-team strengths (top-3 categories) and weaknesses (bottom-3). Display as a sortable leaderboard with drill-down.

#### 13. Pick Predictor (Survival Probability UI)
- **What it is:** Shows the % chance a specific player will still be available at your next pick, your pick after that, etc.
- **Why it matters:** HEATER already computes survival probability in simulation.py, but it's embedded in the MC recommendation engine. A dedicated "will this player be available?" lookup would be a killer UX feature during drafts.
- **Free/Paid:** PRO+ ($3.99/mo)
- **Implementation idea:** Surface existing `survival_probability()` calculations as a player-searchable lookup. Show a decay curve: "Available at pick 45: 72%, pick 57: 31%, pick 69: 8%".

---

### Priority 3: Nice-to-Have Features (Completeness / Market Parity)

#### 14. Points League Projections
- **What it is:** Projected fantasy points per player for specific scoring formats (Yahoo default points, ESPN default points, CBS, custom).
- **Why it matters:** HEATER uses SGP-based valuation which is roto/H2H-category-native. Points leagues use a single aggregate score. Converting raw stat projections to fantasy points per scoring system is a different calculation.
- **Free/Paid:** Free
- **Implementation idea:** Add scoring_format presets (Yahoo/ESPN/CBS points weights) and a `compute_fantasy_points(projections_df, weights)` function. Display alongside SGP-based value.

#### 15. Prospect Rankings
- **What it is:** Minor league prospect rankings for dynasty and keeper leagues.
- **Why it matters:** Only relevant for dynasty/keeper leagues, but those formats are growing. Prospects represent speculative long-term value not captured by MLB projection systems.
- **Free/Paid:** Free
- **Implementation idea:** Fetch prospect rankings from an external source (e.g., FanGraphs "The Board" prospect list). Store in a `prospects` table. Display in a dedicated tab or page.

#### 16. Draft Order Generator
- **What it is:** Randomized draft order creation and email distribution for leagues.
- **Why it matters:** Minor utility feature. Most leagues handle this through their platform (Yahoo, ESPN).
- **Free/Paid:** Free
- **Implementation idea:** Simple random shuffle of team names with seed-based reproducibility. Not a priority.

#### 17. Auto-Swap Inactive/IL Players
- **What it is:** Automatically detects when a rostered player goes to the IL and swaps in the best available bench replacement.
- **Why it matters:** Prevents losing production due to unnoticed IL placements. Requires live roster monitoring.
- **Free/Paid:** MVP+ ($5.99/mo)
- **Implementation idea:** Poll Yahoo API for roster status changes. When IL detected, run lineup optimizer to find optimal swap. Surface as a notification/recommendation (auto-executing roster moves requires API write access).

#### 18. Multiple League Support
- **What it is:** Manage 2-50 fantasy leagues simultaneously from one dashboard.
- **Why it matters:** Serious fantasy players often play in 3-5 leagues. HEATER currently supports only one league context.
- **Free/Paid:** PRO (2 leagues), MVP (10), HOF (50)
- **Implementation idea:** Add a league_id column to all in-season tables. League selector in sidebar. Separate DraftState per league.

#### 19. "Who Should I Start?" Quick Compare
- **What it is:** Ultra-fast 2-4 player comparison specifically for start/sit decisions, powered by expert consensus.
- **Why it matters:** HEATER's start/sit advisor runs the full optimization pipeline. A quick "Player A vs. Player B — who do I start today?" widget would reduce friction for simple decisions.
- **Free/Paid:** Free (2 players), HOF (4 players)
- **Implementation idea:** Lightweight version of start_sit.py that compares N players head-to-head using today's matchup data. Single-click UI widget.

#### 20. Scoring / Points Leaders
- **What it is:** Real-time leaderboard showing who is leading in actual fantasy points across positions.
- **Why it matters:** Helps identify breakout performers and validate projection accuracy.
- **Free/Paid:** Free
- **Implementation idea:** Use season_stats data to compute current fantasy points/SGP leaders. Display as a sortable leaderboard.

---

### Priority 4: Out of Scope (Not Relevant to HEATER's Mission)

These FantasyPros features are intentionally **excluded** from the gap analysis because they don't align with HEATER's design goals:

| Feature | Why Excluded |
|---------|-------------|
| **DFS Tools** (DraftKings/FanDuel optimizers, multi-lineup generator, projected ownership, ROI analyzer) | HEATER is a season-long fantasy tool, not a DFS platform. DFS requires completely different architecture. |
| **Mobile Apps** (iOS/Android) | HEATER is a Streamlit web app. Mobile would require a complete rewrite in React Native / Flutter. |
| **Research Assistant Browser Extension** | Chrome extension development is outside Streamlit's capabilities. |
| **Podcast / Editorial Content** | HEATER is a tool, not a media company. Content creation is a human endeavor. |
| **Multi-Sport Support** (NFL, NBA) | HEATER is baseball-specific by design. |
| **Expert Network** (100+ human contributors) | HEATER uses algorithmic analysis, not human expert aggregation. |

---

## Gap Summary

| Priority | # Features | Est. Effort | Impact |
|----------|-----------|-------------|--------|
| **P1: High-Impact** | 6 | Large (weeks each) | Transformative — opens new markets (auction, dynasty, multi-platform) |
| **P2: Medium-Impact** | 7 | Medium (days each) | Significant UX improvement, surfaces existing analytics better |
| **P3: Nice-to-Have** | 7 | Small-Medium (hours-days) | Completeness, parity with free FantasyPros features |
| **P4: Out of Scope** | 6 | N/A | Not aligned with HEATER's mission |

### Recommended Implementation Order

**Phase 1 — Quick Wins (leverage existing code):**
1. Pick Predictor UI (survival probability already computed)
2. Streaming Pitcher Daily Recommendations (optimizer/streaming.py exists)
3. Closer Depth Chart page (depth_charts.py + contextual_factors.py exist)
4. Schedule Grid / Weekly Planner (matchup_planner.py + live_stats.py exist)
5. Player Tags (simple DB table + UI badges)

**Phase 2 — High-Value New Features:**
6. Projected Season Standings (blended projections + roster data exist)
7. League Power Rankings (standings + projections exist)
8. Cheat Sheet Export (player_pool data + PDF generation)
9. ECR Integration (new data source, blend into rankings)
10. Points League Projections (new computation, straightforward)

**Phase 3 — Market Expansion:**
11. ESPN League Sync (espn-api package)
12. Sleeper League Sync (public API)
13. Live Draft Assistant with Yahoo real-time sync
14. Auction Values Calculator
15. Keeper League Support

**Phase 4 — Full Parity:**
16. Auction Draft Simulator
17. Dynasty/Prospect Rankings
18. Multiple League Support
19. Auto-Swap IL Players
20. Remaining minor features

---

## Sources

- FantasyPros MLB Home: fantasypros.com/mlb/
- FantasyPros Draft Tools: draftwizard.fantasypros.com/baseball/draft-tools/
- FantasyPros Pricing: fantasypros.com/premium/plans/wp/
- FantasyPros Support (PRO tier): support.fantasypros.com/hc/en-us/articles/25998259440795
- FantasyPros Support (MVP tier): support.fantasypros.com/hc/en-us/articles/25998821331995
- FantasyPros Support (HOF tier): support.fantasypros.com/hc/en-us/articles/25998848787739
- FantasyPros Mock Draft Simulator: draftwizard.fantasypros.com/baseball/mock-draft-simulator/
- FantasyPros My Playbook: fantasypros.com/mlb/myplaybook/intro.php
- FantasyPros DFS: fantasypros.com/daily-fantasy/mlb/tools.php
