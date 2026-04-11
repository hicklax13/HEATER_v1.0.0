# HEATER Football Business Plan

**Date:** April 11, 2026
**Author:** Connor Hickey
**Status:** Draft v1.0
**Parent Plan:** `2026-04-11-heater-business-plan.md` (HEATER Baseball)

---

## Executive Summary

HEATER Football is a fantasy football analytics product expanding from the proven HEATER Baseball platform. It targets serious redraft PPR league players (Standard, Half-PPR, Full PPR) in 6-14 team formats with Monte Carlo simulation, LP-constrained lineup optimization, Bayesian projection updating, game theory trade analysis, and — uniquely — an algorithmic FAAB bid optimizer that no competitor offers.

The fantasy football market is 3-4x larger than baseball (~40-42M US players, $1.87B market). No competitor combines MC simulation + LP optimization + Bayesian updating + game theory for season-long fantasy football. The FAAB bid optimization engine fills the single largest unmet need in the market.

HEATER Football reuses 55-65% of the baseball codebase (~25,000 lines of shared infrastructure), requiring ~35,500 lines of new football-specific code over ~19 weeks of development. It launches for NFL 2027 (alpha July, beta August, GA September).

The combined baseball+football business reaches $380K total revenue in Year 3 with 70% margins and ~$860K ARR run-rate. Baseball (Apr-Oct) + Football (Sep-Jan) covers 10 of 12 months, eliminating the seasonal revenue gap that is single-sport competitors' biggest weakness.

---

## 1. Market Analysis

### 1.1 Market Size

| Metric | Football | Baseball | Ratio |
|--------|----------|----------|-------|
| US players | 40-42M | 8-10M | **4-5x** |
| Market value (2026) | $1.87B | $1.5B | 1.2x |
| Market share of fantasy | ~75% | ~15% | 5x |
| Premium tool TAM | $100-250M | $37-105M | 2-3x |
| r/subreddit members | 3.4M | 200K | **17x** |

### 1.2 Target Market

- **Format:** Redraft snake draft, Standard/Half-PPR/Full PPR (95%+ of leagues)
- **League size:** 10-12 teams (70-80% of leagues)
- **SAM:** ~350K-500K serious PPR players who'd pay for premium tools
- **Cross-sell base:** 65-75% of HEATER Baseball subscribers also play fantasy football

### 1.3 Demographics

- Core: 25-44 years old (58%), male (64%), $75K+ income, college-educated
- Mobile-first: 72% of fantasy football engagement is mobile (vs desktop-heavy for baseball)
- More casual on average than baseball, but the serious segment is larger in absolute terms

### 1.4 Key Differences from Baseball

| Dimension | Baseball | Football |
|-----------|----------|----------|
| Games/season | 162 | 17 |
| Projection accuracy ceiling | ~70% correlation | ~45-55% |
| Dominant format | H2H Categories | H2H Points (PPR) |
| Core weekly decision | Daily lineup x 7 days | Single binary start/sit |
| Critical in-season feature | Trade analyzer | FAAB/waiver wire |
| Lineup cadence | Daily | Weekly |
| Injury impact | 0.6% of season per game | 5.9% of season per game |
| Game flow dependency | Low (independent ABs) | Very high (correlated by game script) |

---

## 2. Competitive Analysis

### 2.1 Landscape

| Competitor | Price (Annual) | Strengths | Key Weakness vs HEATER |
|------------|---------------|-----------|----------------------|
| **FantasyPros** | $48-108 (all sports) | ECR industry standard, multi-platform sync | No MC simulation, no LP optimization, no FAAB algorithm |
| **Sleeper** | Free (DFS monetization) | Best social/chat, modern UI, 4M+ users | No premium analytics layer at all |
| **PFF** | $40-200/yr | Proprietary player grades, NFL credibility | Grades are inputs, not decision engine; weak on optimization |
| **4for4** | $29-59/season | Best projection accuracy (verified) | No simulation, no trade analyzer, no FAAB algorithm |
| **RotoWire** | $50-120 | Fastest news, multi-sport | DFS-focused, shallow season-long tools |
| **ESPN/Yahoo** | Free | Massive user bases | Minimal analytics depth |
| **RotoBot AI** | $80-130/yr | Conversational AI interface | LLM wrapper, no MC/LP mathematical backbone |
| **NFL Pro Fantasy AI** | NFL+ subscription | Official NFL product, NextGen Stats access | Locked behind NFL+ paywall, early stage |

### 2.2 Competitive Whitespace

**No competitor combines MC simulation + LP optimization + Bayesian updating + game theory for season-long fantasy football.** This is the same whitespace HEATER occupies in baseball.

Additionally, **FAAB bid optimization is a wide-open gap.** Every existing tool gives qualitative "high/medium/low priority" labels. Nobody provides algorithmic, budget-aware, game-theory-optimal bid recommendations.

### 2.3 Positioning

**"HEATER Football: The only fantasy football tool that uses Monte Carlo simulation and LP optimization to turn projections into probability-weighted, roster-aware decisions — not just rankings."**

For marketing messaging, lead with outcomes: "73% win probability" rather than methods: "Monte Carlo simulation." Football players care about results more than methodology.

---

## 3. SWOT Analysis

### Strengths
- **MC simulation engine transfers** — 10K paired simulations, game theory, Bayesian updating all apply
- **LP optimization is simpler for football** — weekly single-objective (points) vs daily multi-objective (categories)
- **Bayesian updating is MORE valuable** — 17-game small samples make priors critical
- **Yahoo API already built** — same yfpy library works for football
- **Existing brand and subscriber base** — warm cross-sell pipeline
- **3,180+ tests, backtesting framework** — engineering quality transfers
- **Year-round engagement** — baseball + football covers 10 of 12 months

### Weaknesses
- **No football domain expertise yet** — zero football-specific models built
- **SGP framework doesn't transfer** — football needs VBD (Value-Based Drafting), not SGP
- **Game script correlation** — MC engine assumes independent outcomes; football has deep game-flow dependencies
- **Football data ecosystem is less open** — NextGen Stats and PFF grades are paywalled
- **Solo developer** — second sport doubles maintenance burden
- **Mobile-first is mandatory** — football users are overwhelmingly mobile
- **Crowded market** — 50+ competitors vs ~10 in baseball

### Opportunities
- **FAAB bid optimization** — #1 unmet need, no competitor does it algorithmically
- **Floor/ceiling probability distributions** — MC simulation provides real distributions, not arbitrary labels
- **Game script simulation** — model how Vegas lines affect player usage and variance
- **Cross-sport retention** — bundle reduces churn 15-25 percentage points vs single-sport
- **r/fantasyfootball (3.4M members)** — 17x larger organic channel than baseball
- **Sleeper analytics gap** — Sleeper has millions of users but zero premium analytics; HEATER can be the analytics companion
- **Commissioner league reports** — free branded tool that exposes HEATER to 12 league-mates per commissioner

### Threats
- **FantasyPros market dominance** — largest brand, ECR is the standard
- **NFL's own AI play** — AWS-powered NextGen Stats AI assistant with proprietary data
- **PFF's data moat** — unique player grades, NFL team credibility
- **AI commoditization** — ChatGPT/Claude providing "good enough" generic advice
- **NFL data licensing** — Genius Sports holds exclusive rights through 2030
- **Football's inherent variance** — even great models can't overcome 17-game noise; users may expect baseball-level accuracy

---

## 4. Product Strategy

### 4.1 What Transfers (55-65% of codebase)

- Yahoo Data Service + OAuth + 3-tier caching
- Bootstrap pipeline architecture (staleness-based refresh)
- Monte Carlo simulation engine core
- LP solver framework (PuLP)
- Bayesian projection framework (PyMC + Marcel fallback)
- Game theory trade modules (opponent modeling, adverse selection, Bellman DP)
- Scenario generator (Gaussian copula, CVaR)
- UI framework (theme, CSS, layout, formatting)
- CI/CD pipeline, constants registry, sensitivity analysis
- ~25,000 lines of directly reusable infrastructure

### 4.2 What's New (35-45%)

| Module | Lines | Weeks | Transfer % |
|--------|-------|-------|-----------|
| Core library extraction (refactor) | ~5,000 | 2 | 100% |
| Football LeagueConfig + VBD | ~1,500 | 1 | 30% |
| nflverse data pipeline | ~3,000 | 2 | 20% |
| Projection blending (football) | ~2,000 | 1.5 | 50% |
| Lineup optimizer (football) | ~2,500 | 1.5 | 60% |
| Start/sit advisor | ~2,000 | 1.5 | 25% |
| **FAAB bid engine (flagship)** | ~3,000 | 2 | 15% |
| Draft engine (VBD) | ~2,000 | 1 | 55% |
| Trade analyzer (football) | ~2,500 | 1.5 | 65% |
| Matchup/defense adjustments | ~2,000 | 1.5 | 35% |
| Streamlit/Next.js pages | ~4,000 | 2 | 40% |
| Tests | ~6,000 | 2 | 30% |
| **Total** | **~35,500** | **~19 weeks** | ~40% avg |

### 4.3 Football MVP (Ship by NFL Week 1, Sep 2027)

**P0 (Day 1):**
1. Start/Sit Advisor — floor/ceiling probability distributions, matchup grades, game script adjustment
2. FAAB Bid Optimizer — game-theory optimal bids, budget allocation, opponent modeling
3. Draft Tool — VBD calculator, strategy simulation (Zero-RB/Hero-RB/Robust-RB)
4. Trade Analyzer — MC simulation + game theory, schedule-strength adjusted
5. Lineup Optimizer — LP-constrained, risk-adjusted (safe floor vs high ceiling modes)

**P1 (Month 1 post-launch):**
6. Matchup Planner (weekly win probability)
7. League Standings + Playoff Projections
8. Free Agent Rankings (marginal value)
9. ESPN + Sleeper league integration

**P2 (Mid-season):**
10. Playoff Scenario Modeler
11. Dynasty/Keeper support
12. Injury Impact Model

### 4.4 FAAB Bid Engine Design (Flagship Feature)

The #1 differentiator, confirmed across all 5 research streams:

1. **Marginal roster value:** Score each FA by `delta_wins = E[season_wins_with] - E[season_wins_without]` using MC simulation
2. **Opponent bid modeling:** Simulate opponent bids using league history, roster needs, and behavioral patterns (clusters at $5/$10/$15 increments)
3. **Monte Carlo simulation:** 10K scenarios of waiver process, varying opponent bids stochastically
4. **Optimal bid recommendation:** Minimum bid winning in 70%+ of simulations, saving budget for future weeks
5. **Season-long budget allocation:** Dynamic programming optimizing total FAAB across remaining weeks

**Free tier hook:** Show top 3 waiver adds free. Hide the bid recommendation behind paywall.

### 4.5 Data Stack (All Free for v1)

| Source | Data | Library |
|--------|------|---------|
| nflreadpy (nflverse) | Play-by-play since 1999, EPA, WPA, rosters, schedules | `nflreadpy` (MIT) |
| Yahoo Fantasy API | League data, rosters, matchups | `yfpy` (already built) |
| ESPN API | Projections, injuries | `espn-api` |
| FantasyPros | ECR consensus | Custom scraper |
| Odds API | Vegas lines for game script modeling | REST API (free tier) |
| Open-Meteo | Weather (already built) | Existing `game_day.py` |
| Fantasy Football Data Pros | Historical weekly data | REST API |

### 4.6 Architecture: Monorepo

```
HEATER/
  core/           # Shared: MC engine, LP solver, Bayesian, game theory, UI, auth, billing
  baseball/       # HEATER Baseball (existing product)
  football/       # HEATER Football (new)
    config.py     # LeagueConfig (PPR scoring, VBD)
    valuation.py  # VBD calculator
    projections.py # ECR + nflverse blending
    data_sources/  # nflreadpy, PFR, ESPN
    optimizer/     # Weekly LP optimizer
    faab/          # FAAB bid engine (flagship)
    start_sit/     # Start/sit advisor
    draft/         # VBD draft + strategy sim
    engine/        # Trade analyzer (VBD-based)
    pages/         # 8-10 Streamlit/Next.js pages
    tests/
```

---

## 5. Go-to-Market Strategy

### 5.1 Timeline

| Phase | Timing | Milestone |
|-------|--------|-----------|
| Engine development | Apr-Jul 2027 | Build football-specific modules |
| Closed alpha | Jul 2027 | Baseball subscribers test football |
| Open beta | Aug 2027 | Draft tools live, preseason |
| GA launch | NFL Week 1, Sep 2027 | Full in-season features |
| Growth push | Oct-Nov 2027 | Content marketing, paid ads, podcasts |
| Retention | Dec 2027-Aug 2028 | Cross-sell, off-season features |

### 5.2 MVP Launch Features

Day 1: Start/Sit, FAAB Engine, Draft Tool, Trade Analyzer, Lineup Optimizer, Yahoo + ESPN sync

### 5.3 Marketing Channels

| Channel | Strategy | Budget | Expected CAC |
|---------|----------|--------|-------------|
| r/fantasyfootball | 6+ months of value-first engagement before product links | $0 | $5-15 |
| Twitter/X | Real-time injury reactions powered by HEATER models | $0 | $10-25 |
| SEO/Content | Weekly trade value chart (free), FAAB guides, start/sit | $0 | $15-40 |
| Discord | HEATER subscriber community | $0 | $0 (retention) |
| Podcast sponsorship | Tier 2: PFF, Late Round, Establish the Run | $200-500/ep | $30-60 |
| Reddit Ads | Targeted at r/fantasyfootball | $500-1K/mo | $15-35 |
| Google Ads | High-intent keywords (retargeting) | $500-1K/mo | $40-80 |
| Cross-sell from baseball | In-app promotion, email, bundle pricing | $0 | ~$0 |

**Year 1 marketing budget:** $15-25K total. Blended CAC: $25-45.

### 5.4 Content Strategy

In-season weekly cadence:
- **Monday:** Weekly recap, accuracy report
- **Tuesday night:** FAAB bid recommendations (time-sensitive, drives urgency)
- **Wednesday:** Trade value chart update
- **Thursday:** Start/sit rankings + Thursday Night preview
- **Friday:** Matchup previews
- **Sunday morning:** Final lineup recommendations, weather updates

**Free trade value chart** is the #1 SEO marketing asset (same as baseball). "The only trade value chart powered by 10,000 simulations."

---

## 6. Branding

- **Name:** HEATER Football (unified brand, not separate)
- **Tagline:** "10,000 simulations. One decision." (carries from baseball)
- **Alt:** "Stop guessing. Start simulating."
- **Voice for football:** Lead with outcomes ("73% win probability"), not methods ("Monte Carlo simulation")
- **Visual:** Same thermal/fire palette, football-specific icons
- **Domain:** Same domain with `/football` route

---

## 7. Pricing Strategy

### 7.1 Football Standalone

| Tier | Monthly | Season (5 mo) | Annual |
|------|---------|---------------|--------|
| Free | $0 | $0 | $0 |
| Pro | $9.99 | $39.99 | $79.99 |
| Elite | $19.99 | $79.99 | $119.99 |

### 7.2 Multi-Sport Bundle

| Plan | Baseball Only | Football Only | All-Sports |
|------|-------------|--------------|-----------|
| Annual | $119.99 | $119.99 | $149.99 (37% savings) |
| Monthly | $14.99 | $14.99 | $19.99 |

**Cross-sell offer:** Existing baseball annual subscribers get first-year football at $29.99 (total = $149.99, matching bundle price).

---

## 8. Financial Projections

### 8.1 Incremental Startup Costs

| Category | Low | Base | High |
|----------|-----|------|------|
| Development (790 dev-hours) | $0 (side project) | $0 (self-built) | $30,000 (contractor) |
| Data licensing | $0 (free sources) | $1,200/yr | $6,000/yr |
| Marketing (Year 1) | $5,000 | $15,000 | $25,000 |
| Incremental infrastructure | $240/yr | $540/yr | $1,200/yr |
| **Total incremental** | **$5,240** | **$16,740** | **$62,200** |

### 8.2 Football Revenue Projections

| Metric | Year 1 (NFL 2027) | Year 2 (NFL 2028) | Year 3 (NFL 2029) |
|--------|-------------------|-------------------|-------------------|
| Peak subscribers | 300 | 1,000 | 3,200 |
| Revenue | $16,325 | $54,600 | $170,500 |
| Incremental costs | $4,200 | $14,200 | $34,300 |
| **Net from football** | **$12,125** | **$40,400** | **$136,200** |
| Margin | 74% | 74% | 80% |

Football breaks even at just **34 paying users** (shared infrastructure absorbs fixed costs).

### 8.3 Combined Business (Baseball + Football)

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Baseball revenue | $4,500 | $21,600 | $90,000 |
| Football revenue | $0 | $16,325 | $170,500 |
| **Total revenue** | **$4,500** | **$37,925** | **$260,500** |
| Total costs | $1,700 | $17,925 | $72,700 |
| **Net income** | **$2,800** | **$20,000** | **$187,800** |
| Combined ARR run-rate | $13K | $74K | **$860K** |

### 8.4 Unit Economics

| Metric | Baseball Only | Football Only | Bundle |
|--------|-------------|--------------|--------|
| LTV | $255 | $180 | $400 |
| CAC | $35 | $35 | $35 (cross-sell: ~$0) |
| LTV:CAC | 7.3:1 | 5.1:1 | 11.4:1 |
| Annual retention | 55-65% | 50-60% | **75-85%** |

Bundle subscribers are the most valuable: highest LTV, highest retention, lowest CAC (cross-sell).

---

## 9. Growth Strategy

### 9.1 Cross-Sport Flywheel

```
Baseball subscribers (Apr-Oct)
    ↓ 30-45% cross-sell (Aug)
Football subscribers (Sep-Jan)
    ↓ 25-35% cross-sell (Mar)
Baseball subscribers (Apr-Oct)
    ↓ Bundle retention: 75-85% YoY
```

### 9.2 Viral Features

1. **"HEATER Says" shareable cards** — branded start/sit recommendations for league chats
2. **Trade analyzer receipts** — MC simulation results as bragging-rights images
3. **FAAB bid results** — post-waiver performance vs HEATER's recommendations
4. **Commissioner league report** — free branded analytics shared with entire league
5. **Season-end report card** — "What if you followed HEATER 100%?"

### 9.3 League Adoption Flywheel

One user wins → league-mates investigate → compound adoption. Accelerated by:
- League-wide dashboards (3+ subscribers unlock)
- Commissioner tools (free, branded)
- League invite discount (6+ members → 20% off)

### 9.4 Year-Round Calendar

- **Feb (only dead month):** Combine analysis, dynasty rankings
- **Mar-Jul:** Baseball carries engagement; football preview content ramps
- **Aug:** Draft prep peak — prime cross-sell window
- **Sep-Oct:** **Dual sport golden window** — maximum engagement
- **Nov-Jan:** Football carries through baseball off-season
- **Result:** 10-11 months of active engagement vs 5-7 for single-sport competitors

### 9.5 Long-Term Expansion

| Year | Milestone | Combined ARR Target |
|------|-----------|-------------------|
| Y1 (2026-27) | Baseball launch | $20-50K |
| Y2 (2027-28) | Baseball growth + Football launch | $250-500K |
| Y3 (2028-29) | Multi-sport platform | $500K-1M |
| Y4 (2029-30) | Add basketball, consider acquisition | $1-2M |
| Y5 (2030-31) | Full platform or exit | $2-5M |

**Exit precedents:** RotoWire ~$30M acquisition. PrizePicks $2.5B. Fantasy sports M&A is active.

---

## 10. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| NFL data licensing (Genius Sports exclusivity) | HIGH | HIGH | Use nfl_data_py (MIT library) + public box scores; graduate to licensed data with revenue |
| Crowded market (~50 competitors) | CERTAIN | MEDIUM | FAAB optimizer is unique; league-specific context is the moat; don't compete on generic rankings |
| Football's inherent variance disappoints users | MEDIUM | MEDIUM | Set expectations: show probability distributions, not certainties; publish accuracy reports transparently |
| FantasyPros adds MC/LP features | MEDIUM | MEDIUM | Speed of iteration; H2H-specific depth is hard to replicate; community switching costs |
| ESPN API changes/breaks | MEDIUM | HIGH | Support Yahoo + Sleeper + manual import as alternatives |
| Solo developer burnout from 2 sports | HIGH | MEDIUM | First hire: football data engineer (Q2 2027, $30-50K contract); automate pipelines |
| Cross-sell assumption wrong (baseball users don't want football) | LOW | MEDIUM | Even at 10% cross-sell, organic football acquisition from r/fantasyfootball (3.4M) covers demand |
| AI commoditization | HIGH | MEDIUM | League-specific context is the moat; "HEATER AI" uses engines as LLM context, not the reverse |

---

## Appendix A: Football Data Source Costs

| Source | Free for Commercial? | Estimated Cost | Notes |
|--------|---------------------|---------------|-------|
| nflreadpy/nflverse | MIT library (data rights ambiguous) | $0 | Best free option; play-by-play since 1999 |
| Yahoo Fantasy API | Personal use OAuth | $0 (negotiate for commercial) | Already built |
| ESPN API | Undocumented, gray area | $0 | Use carefully |
| Sleeper API | Documented, developer-friendly | $0 | Best football platform API |
| FantasyPros | Scraping gray area | $0 (negotiate license for scale) | Commercial requires agreement |
| PFF grades | Enterprise licensing | $50K-200K/yr | Out of budget — build own from public data |
| Genius Sports (official NFL) | Exclusive rights through 2030 | $unknown (enterprise) | Not accessible for startup |
| SportsRadar | Licensed data | $6K-12K/yr | Graduate to this at scale |
| MySportsFeeds | Developer-friendly | $100-300/mo | Good middle option |

## Appendix B: Supporting Research Documents

- `2026-04-11-heater-football-financial-model.md` — Detailed 3-year financial model
- `2026-04-11-heater-football-gtm.md` — Detailed go-to-market strategy

## Appendix C: Sources

- FSGA 2025 Industry Research
- Global Growth Insights: Fantasy Football Market ($1.87B, 2026)
- DemandSage: Fantasy Sports Market Size
- SubredditStats: r/fantasybaseball user overlaps (49,329x with r/fantasyfootball)
- FantasyPros Premium Plans
- PFF Membership & Pricing
- 4for4 Plans & Pricing
- RotoWire Subscription Pricing
- Sleeper Fantasy Football Review
- nflreadpy Documentation (nflverse)
- Fantasy Football Data Pros API
- FAABLAB (Crowd-Sourced FAAB Data)
- KeepTradeCut Dynasty Resources
- Fantasy Footballers Podcast (140M+ downloads)
- Teamworks PFF Acquisition ($130-140M, March 2026)
- Fantasy Life $7M Seed Round (Matthew Berry)
