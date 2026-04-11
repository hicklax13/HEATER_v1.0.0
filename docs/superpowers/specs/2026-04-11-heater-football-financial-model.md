# HEATER Football -- Incremental Financial Model

**Date:** April 11, 2026
**Author:** Connor Hickey
**Parent Document:** HEATER Business Plan (2026-04-11)
**Status:** Draft v1.0

---

## 1. Incremental Startup Costs (Football Expansion)

Since HEATER Baseball will already have a fully operational Next.js frontend, FastAPI backend, PostgreSQL database, Clerk auth, and Stripe billing by the time football launches (target: NFL 2027), we estimate ONLY the net-new costs.

### 1.1 Football Engine Development

The existing baseball codebase has ~58,000 lines. Football reuses 60-70% of the infrastructure (auth, billing, UI framework, database schema, API layer, caching, deployment). The 30-40% net-new work is sport-specific engine logic.

| Component | Reusable from Baseball | New Work Required | Dev Hours | Cost (@$65/hr) |
|-----------|----------------------|-------------------|-----------|-----------------|
| **Projection engine** (NFL player projections, weekly scoring) | Blending framework, Bayesian updating, Marcel template | NFL stat categories (passing/rushing/receiving/IDP), weekly vs daily cadence, bye week handling | 120-160 hrs | $7,800-$10,400 |
| **Trade analyzer** (MC simulation, game theory) | Full MC pipeline, game theory module, convergence diagnostics | NFL roster construction (QB/RB/WR/TE/FLEX/DST/K), PPR scoring variants, positional scarcity recalibration | 80-120 hrs | $5,200-$7,800 |
| **Lineup optimizer** (LP solver, start/sit) | PuLP LP framework, sigmoid urgency, DCV scoring | NFL roster slots, Thursday/Sunday/Monday splits, bye week LP constraints, injury designation handling (Q/D/O/IR) | 100-140 hrs | $6,500-$9,100 |
| **Matchup planner** (H2H categories/points) | Matchup framework, probability engine, color tiers | NFL scoring types (PPR/Standard/Half-PPR), weekly matchup structure (not daily), DST matchup analysis | 40-60 hrs | $2,600-$3,900 |
| **Draft engine** (snake/auction) | Full draft framework, AI opponents, MC recs | NFL ADP, positional value (Zero-RB, Hero-RB, Robust-RB strategies), NFL-specific tiers | 60-80 hrs | $3,900-$5,200 |
| **Waiver/FA engine** | FA ranking framework, LP-verified add/drop | FAAB bidding logic, waiver priority modeling, NFL weekly waivers (not daily) | 40-60 hrs | $2,600-$3,900 |
| **Data pipeline** (NFL sources) | Pipeline architecture, caching, staleness framework | NFL Stats API integration, nfl_data_py parsing, injury feed (ESPN), snap counts, target share | 60-80 hrs | $3,900-$5,200 |
| **Football-specific UI** | All Streamlit/Next.js pages, layout, theme | NFL-specific visualizations, scoring breakdown charts, snap count displays | 80-100 hrs | $5,200-$6,500 |
| **Testing** | Test framework, CI pipeline | ~1,500-2,000 new football-specific tests | 80-120 hrs | $5,200-$7,800 |
| **TOTAL** | | | **660-920 hrs** | **$42,900-$59,800** |

**Midpoint estimate: 790 hours, $51,350** (solo dev at opportunity cost; no cash outlay if Connor builds it).

### 1.2 Incremental Infrastructure

| Item | Monthly Cost | Annual Cost | Notes |
|------|-------------|-------------|-------|
| Additional Railway compute | $15-30 | $180-360 | NFL workloads run Sep-Jan (5 months active) |
| Additional Neon storage | $5-10 | $60-120 | NFL player pool (~2,500 players vs MLB's 9,000+) |
| Additional Redis cache | $0-5 | $0-60 | Shared instance, marginal increase |
| **Total incremental infra** | **$20-45** | **$240-540** | |

Infrastructure is nearly negligible because the baseball platform already bears the fixed costs. Football adds marginal load only during NFL season.

### 1.3 Football Data Licensing

See Section 2 for detailed source-by-source analysis.

| Scenario | Annual Data Cost | Notes |
|----------|-----------------|-------|
| **Bootstrap (open-source only)** | $0 | nfl_data_py + ESPN/Sleeper scraping + own models. Legal gray area. |
| **Base (selective licensing)** | $3,000-8,000 | MySportsFeeds commercial license + one projection partner |
| **Full (enterprise data)** | $15,000-50,000 | SportsRadar/Genius Sports API + PFF grades |

### 1.4 Marketing & Launch

| Item | Cost | Notes |
|------|------|-------|
| Football-specific content creation | $0 (time) | Weekly rankings, waiver columns, trade charts -- same as baseball |
| Reddit/Twitter organic push | $0 (time) | r/fantasyfootball (2.2M members vs r/fantasybaseball's 225K) |
| Podcast sponsorship (2-3 football pods) | $1,000-3,000 | Fantasy Footballers, Late Round Podcast, etc. |
| Reddit Ads (r/fantasyfootball) | $500-1,500 | 3-month pre-season/early season burst |
| Cross-sell email to baseball subscribers | $0 | Highest-conversion channel |
| **Total marketing (Year 1 football)** | **$1,500-4,500** | |

### 1.5 Total Incremental Startup Costs

| Scenario | Development | Data | Infra | Marketing | Total |
|----------|-------------|------|-------|-----------|-------|
| **Optimistic** | $0 (Connor builds) | $0 | $240 | $1,500 | **$1,740** |
| **Base** | $0 (Connor builds) | $5,000 | $400 | $3,000 | **$8,400** |
| **Pessimistic** | $51,350 (contractor) | $30,000 | $540 | $4,500 | **$86,390** |

If Connor builds the football engine himself (most likely), incremental cash cost is $1,740-$8,400. The real cost is ~790 hours of development time (roughly 4-5 months part-time, Oct 2026 - Feb 2027).

---

## 2. Football Data Source Costs

### 2.1 Source-by-Source Analysis

| Source | Cost | Commercial OK? | Risk Level | Notes |
|--------|------|---------------|-----------|-------|
| **NFL Stats API** (via nfl.com) | Free (public) | Ambiguous -- no official developer program | MEDIUM | No formal commercial API. Data is public but NFL has no developer portal like MLB. Genius Sports holds official rights. |
| **nfl_data_py / nflfastR** | $0 (MIT license) | Code: Yes. Data: Ambiguous. | LOW-MEDIUM | The Python/R packages are MIT-licensed. Underlying NFL play-by-play data is technically NFL property but has been treated as publicly available. FTN charting data within it is CC-BY-SA 4.0 (requires attribution). |
| **PFF Data (B2B)** | $50,000-200,000+/yr | Yes (enterprise license) | HIGH COST | PFF's B2B data arm (now under Teamworks) licenses player grades, advanced metrics. Consumer PFF+ is $99-120/yr but that license is personal use only. Enterprise API pricing is not public and requires sales negotiation. |
| **SportsRadar NFL** | $6,000-12,000+/yr | Yes (enterprise license) | MEDIUM-HIGH COST | Official NFL data distribution partner. Pricing starts ~$500-1,000/month for basic feeds. Full real-time data significantly more. 30-day free trial available. |
| **Genius Sports** | $10,000-50,000+/yr | Yes (exclusive NFL partner) | HIGH COST | The ONLY official NFL data API partner (extended through 2030). Includes NextGen Stats tracking data. Enterprise pricing is custom and not publicly listed. Way out of budget for a startup. |
| **MySportsFeeds** | $50-200/mo (commercial) | Yes | LOW | Budget-friendly option for startups. NFL data including schedules, scores, boxscores, standings, play-by-play, lineups, injuries. Pricing is per-league, in CAD. |
| **ESPN Fantasy API** | $0 (undocumented) | No official terms | HIGH | Undocumented API reverse-engineered by community. No official developer program. ESPN has historically been lenient but could block commercial use at any time. No new developer keys being issued. |
| **Sleeper API** | $0 (free, read-only) | Partially -- no SLA | MEDIUM | Free read-only API, no auth token needed. Explicitly not a commercial API -- no uptime guarantees. Fine for supplementary data, not as primary source. Best used as part of "hybrid data strategy." |
| **Pro Football Reference** | $0 (scraping) | Restricted | HIGH | Terms explicitly prohibit creating competing databases from scraped data. Rate-limited to 20 req/min. Attribution required for shared data. Automated scraping tools violate ToS. |
| **NextGen Stats** | N/A (no public API) | Only via Genius Sports | HIGH | AWS-powered tracking data (speed, acceleration, location). No direct developer access. Only available through Genius Sports' exclusive NFL partnership. |
| **FTN Data** | $599 (CSV access) | Yes | LOW | NFL base stats from last 3 seasons, plus play-by-play feeds. Affordable option for historical data. |

### 2.2 Recommended Football Data Strategy

**Phase 1 (Beta/MVP -- $0):**
- nfl_data_py for historical data, play-by-play, and player stats (MIT license)
- ESPN undocumented API for league sync (risk-accepted for beta, same approach as community tools)
- Sleeper API for supplementary league data (free, read-only)
- Build own projection models from public box scores and play-by-play (like Marcel for football)
- Open-Meteo for weather (already licensed, Apache 2.0)

**Phase 2 (Lean SaaS -- $1,200-3,600/yr):**
- MySportsFeeds commercial license (~$100-300/mo for NFL)
- Continue nfl_data_py for historical/analytical data
- Build ESPN/Yahoo/Sleeper platform integrations

**Phase 3 (Full Platform -- $10,000-25,000/yr):**
- SportsRadar NFL data feed ($6,000-12,000/yr)
- FTN Data for charting/advanced stats ($599/yr)
- Consider PFF B2B data if unit economics support it
- Formal ESPN/Yahoo partnership agreements

### 2.3 NFL vs MLB Data Licensing: Key Differences

| Factor | MLB | NFL |
|--------|-----|-----|
| Official API | MLB Stats API (semi-public, MLBAM controls) | No official public API |
| Official data partner | None (distributed) | Genius Sports (exclusive through 2030) |
| Open-source ecosystem | pybaseball (mature) | nfl_data_py/nflfastR (mature) |
| Commercial licensing attitude | Moderate -- MLBAM enforces selectively | Aggressive -- NFL is historically more protective |
| Data litigation risk | Medium | High |
| Free data availability | High (MLB Stats API) | Medium (no official equivalent) |

**Key takeaway:** NFL data is more restricted and potentially more expensive than MLB. The open-source nfl_data_py ecosystem is the critical enabler for a bootstrapped launch, but commercial scaling requires a licensed data source (MySportsFeeds or SportsRadar).

---

## 3. Pricing Strategy for Football

### 3.1 Competitor Pricing Landscape

| Competitor | Football Price | Multi-Sport? | Model | Key Features |
|------------|---------------|-------------|-------|-------------|
| **FantasyPros** | $5.99-11.99/mo (6-mo packages) | Yes (NFL+MLB+NBA included) | Tiered: PRO/MVP/HOF | Expert consensus rankings, draft simulator, trade analyzer, league sync |
| **PFF+** | ~$99-120/yr (annual) | NFL + College | Single tier (consolidated from old Edge/Elite) | Player grades, rankings, fantasy tools, matchup charts |
| **4for4** | $29-59/season | NFL only | Classic ($29) / Pro ($59) | Rankings, projections, LeagueSync, articles |
| **RotoWire** | $6.99-12/mo | Yes (all sports) | Tiered | Player news (fastest), draft kit, projections, DFS |
| **Fantasy Alarm** | $17.99-39.97/mo | Yes (NFL+MLB+NBA+NHL) | All-Pro monthly/annual | Expert analysis, DFS tools, draft guide |
| **FantasyLife+** | $39.99-99.99/yr | NFL primarily | Tier 1 ($40/yr) / Tier 2 ($100/yr) | Matthew Berry content, tools, data (NBC Sports partnership) |
| **The Athletic** | $8.99/mo or $71.99/yr | Content only, all sports | Single tier | Journalism/analysis, no fantasy tools |
| **Establish the Run** | Est. $50-100/season | NFL + Best Ball | Seasonal | Expert projections, rankings, draft strategy |

### 3.2 Football Pricing Recommendation

| Tier | Monthly | Season (5 mo) | Annual | Features |
|------|---------|---------------|--------|----------|
| **Free** | $0 | $0 | $0 | Basic rankings, 1 league, 2 trades/day, weekly waiver article, mock draft (1/day) |
| **Pro** | $9.99 | $39.99 | $69.99 | Full lineup optimizer, deterministic trade analyzer, FA/waiver engine, matchup planner, multi-league |
| **Elite** | $19.99 | $79.99 | $99.99 | Everything in Pro + Monte Carlo simulation (10K sims), game theory trade analysis, FAAB optimizer, streaming DST/K, advanced injury modeling |

**Rationale:**

1. **Football should be priced the same as baseball per-month ($9.99/$19.99)** but cheaper per-season because the NFL season is only 5 months (Sep-Jan) vs baseball's 7 months (Apr-Oct). Season pass is ~20% less than baseball.

2. **Annual pricing is lower than baseball ($69.99/$99.99 vs $79.99/$119.99)** because football's off-season value is lower -- no daily games, no waiver runs, no daily lineup decisions from Feb-Aug.

3. **Pro at $9.99/mo is competitive:** More than 4for4 Classic ($29/season = $5.80/mo) but less than Fantasy Alarm ($18-40/mo). Positioned as "4for4 depth + FantasyPros ease at FantasyPros price."

4. **Elite at $79.99/season = $16/mo effective** -- cheaper than RotoWire's All-In ($22.99/mo) and Fantasy Alarm ($40/mo), but with genuinely unique MC/game theory features they lack.

### 3.3 Bundle Pricing

| Plan | Baseball Only | Football Only | All-Sports Bundle | Bundle Savings |
|------|-------------|--------------|-------------------|---------------|
| **Pro Monthly** | $9.99 | $9.99 | $14.99 | 25% off |
| **Pro Annual** | $79.99 | $69.99 | $99.99 | 33% off |
| **Elite Monthly** | $19.99 | $19.99 | $29.99 | 25% off |
| **Elite Annual** | $119.99 | $99.99 | $149.99 | 32% off |

**Bundle psychology:** "Get football FREE when you subscribe to baseball annually" framing -- baseball annual is $119.99, bundle is $149.99, so football is effectively $30/year in the bundle. This makes the cross-sell extremely compelling.

### 3.4 Free Trial Strategy

- **14-day full Elite access** (same as baseball)
- **No credit card required** (same as baseball)
- **Football trial timing:** Available starting August 1 each year (pre-draft season), converts during draft season (highest engagement + urgency)
- **Cross-sell trial:** Baseball subscribers get 30-day extended football trial (vs 14 for cold traffic) to maximize conversion

---

## 4. Demand Estimation for Football

### 4.1 Market Sizing

| Metric | Value | Source / Calculation |
|--------|-------|---------------------|
| **US fantasy football players** | ~47.4 million (2025) | FSGA 2025 |
| **Average leagues per player** | 2.8 | FSGA 2025 |
| **Global fantasy football market** | $1.76-1.87B (2025-2026) | Global Growth Insights |
| **Players who subscribe to premium tools** | ~29% of players | FSGA / Industry surveys |
| **Premium fantasy football subscribers (US)** | ~13.7M | 47.4M x 29% |
| **Serious redraft PPR players** | ~4.7-7.1M (10-15% of total) | Estimated from FSGA segments |

### 4.2 TAM / SAM / SOM

| Level | Segment | Size | Annual Value |
|-------|---------|------|-------------|
| **TAM** | All premium fantasy football tool spending (US) | ~$1.87B | Total fantasy football market |
| **SAM** | Serious redraft players willing to pay $50+/yr for premium tools | ~2.0-3.0M players | $100-300M |
| **SOM Year 1** | Realistic capture (beta + cross-sell + organic) | 200-500 paying users | $16,000-$50,000 |
| **SOM Year 2** | Growth + cross-sell maturity | 800-2,000 paying users | $64,000-$200,000 |
| **SOM Year 3** | Established product | 2,500-6,000 paying users | $200,000-$600,000 |

### 4.3 Cross-Sell from Baseball

| Assumption | Estimate | Rationale |
|------------|----------|-----------|
| Baseball subscribers at football launch (Aug 2027) | 300-500 | Business plan projects 400 paying customers by end of Year 2 |
| % who also play fantasy football | 55-65% | Industry data: ~60% of fantasy baseball players also play football |
| % who would try football product (trial) | 70-80% | High -- existing trust, bundle discount, extended trial |
| % who convert to paid football | 40-55% | Higher than cold traffic because product familiarity |
| **Cross-sell football subscribers (Year 1)** | **75-175** | 400 x 60% x 75% x 47% midpoint |

**Cross-sell is the highest-margin channel.** CAC for cross-sell is near $0 (email to existing customers).

### 4.4 Incremental Market Beyond Cross-Sell

| Channel | Year 1 Football Subscribers | CAC |
|---------|---------------------------|-----|
| Cross-sell from baseball | 75-175 | ~$0 |
| Organic (Reddit r/fantasyfootball, SEO, Twitter) | 50-150 | $10-25 |
| Paid acquisition (Reddit Ads, Google Ads) | 25-75 | $30-60 |
| Podcast/influencer referrals | 25-50 | $15-35 |
| Viral/referral from football users | 25-50 | $5-15 |
| **Total Year 1 football subscribers** | **200-500** | **Blended: $15-30** |

### 4.5 Football vs Baseball Market Comparison

| Factor | Baseball | Football | Implication |
|--------|----------|----------|-------------|
| US player base | 12-15M | 47.4M | 3-4x larger audience |
| Premium willingness | 10-15% | 29% | Football players spend more |
| Season length | 7 months | 5 months | Less time to monetize |
| Competition intensity | Moderate | Intense | Higher CAC, harder to differentiate |
| Data availability | Excellent (MLB Stats API) | Restricted (no public API) | Higher data costs |
| r/subreddit size | 225K | 2.2M | 10x organic reach potential |
| Games per week | 7 (daily) | 1 (weekly) | Simpler lineup decisions, less daily engagement |

---

## 5. Revenue Projections -- Football Incremental

### 5.1 Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| NFL active season | 5 months (Sep-Jan) | Weeks 1-18 + playoffs |
| Off-season retention | 25% (at $4.99/mo) | Lower than baseball -- football has less off-season activity |
| In-season monthly churn | 4% | Slightly worse than baseball (5-month window, less habit formation) |
| Trial conversion | 12-15% | Lower than baseball (more competition, less differentiation) |
| Blended ARPU | $11/mo (in-season), $4.99/mo (off-season) | Weighted across tiers |
| Cross-sell conversion | 47% of eligible baseball subs | See section 4.3 |
| Organic growth rate | 15% MoM during NFL season | Pre-season spike, in-season growth, playoff drop |

### 5.2 Year 1 (NFL 2027 Season -- Football Beta + First Paying Users)

| Quarter | Period | Football Subscribers | Football MRR | Football Revenue |
|---------|--------|---------------------|-------------|-----------------|
| Q1 (Sep-Nov 2027) | NFL Weeks 1-12 | 100 -> 250 | $1,100 -> $2,750 | $5,775 |
| Q2 (Dec 2027 - Jan 2028) | NFL Weeks 13-playoffs | 250 -> 300 | $2,750 -> $3,300 | $6,050 |
| Q3 (Feb-May 2028) | Off-season | 300 -> 75 (25% retention) | $375 -> $375 | $1,500 |
| Q4 (Jun-Aug 2028) | Pre-draft, re-activation | 75 -> 150 (re-subs + new) | $375 -> $1,650 | $3,000 |
| **Year 1 Football Total** | | **Peak: 300** | **Peak: $3,300** | **$16,325** |

### 5.3 Year 2 (NFL 2028 Season -- Growth + Cross-Sell Maturity)

| Quarter | Period | Football Subscribers | Football MRR | Football Revenue |
|---------|--------|---------------------|-------------|-----------------|
| Q1 (Sep-Nov 2028) | NFL Weeks 1-12 | 400 -> 800 | $4,400 -> $8,800 | $19,800 |
| Q2 (Dec 2028 - Jan 2029) | NFL Weeks 13-playoffs | 800 -> 1,000 | $8,800 -> $11,000 | $19,800 |
| Q3 (Feb-May 2029) | Off-season | 1,000 -> 250 | $1,250 -> $1,250 | $5,000 |
| Q4 (Jun-Aug 2029) | Pre-draft, re-activation | 250 -> 500 | $1,250 -> $5,500 | $10,000 |
| **Year 2 Football Total** | | **Peak: 1,000** | **Peak: $11,000** | **$54,600** |

### 5.4 Year 3 (NFL 2029 Season -- Established Product)

| Quarter | Period | Football Subscribers | Football MRR | Football Revenue |
|---------|--------|---------------------|-------------|-----------------|
| Q1 (Sep-Nov 2029) | NFL Weeks 1-12 | 1,200 -> 2,500 | $13,200 -> $27,500 | $61,050 |
| Q2 (Dec 2029 - Jan 2030) | NFL Weeks 13-playoffs | 2,500 -> 3,200 | $27,500 -> $35,200 | $62,700 |
| Q3 (Feb-May 2030) | Off-season | 3,200 -> 800 | $4,000 -> $4,000 | $16,000 |
| Q4 (Jun-Aug 2030) | Pre-draft, re-activation | 800 -> 1,500 | $4,000 -> $16,500 | $30,750 |
| **Year 3 Football Total** | | **Peak: 3,200** | **Peak: $35,200** | **$170,500** |

### 5.5 Football P&L Summary (Incremental)

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| **Football Revenue** | $16,325 | $54,600 | $170,500 |
| Football subscribers (peak) | 300 | 1,000 | 3,200 |
| Football MRR (peak) | $3,300 | $11,000 | $35,200 |
| Football ARR run-rate (peak) | $39,600 | $132,000 | $422,400 |
| | | | |
| **Incremental Costs** | | | |
| Infrastructure (incremental) | $400 | $600 | $1,200 |
| Data licensing (football) | $0 | $3,000 | $10,000 |
| Marketing (football) | $3,000 | $8,000 | $15,000 |
| Services (Stripe 2.9% + $0.30) | $800 | $2,600 | $8,100 |
| **Total incremental costs** | **$4,200** | **$14,200** | **$34,300** |
| | | | |
| **Net incremental revenue** | **$12,125** | **$40,400** | **$136,200** |
| **Incremental margin** | **74%** | **74%** | **80%** |

---

## 6. Combined Business Model (Baseball + Football)

### 6.1 Combined 3-Year P&L

| Metric | Year 1 (2027) | Year 2 (2028) | Year 3 (2029) |
|--------|--------------|--------------|--------------|
| | | | |
| **REVENUE** | | | |
| Baseball revenue | $21,600 | $90,000 | $192,000 |
| Football revenue | $16,325 | $54,600 | $170,500 |
| Bundle revenue uplift | $0 | $5,400 | $18,000 |
| **Total revenue** | **$37,925** | **$150,000** | **$380,500** |
| | | | |
| **COSTS** | | | |
| Infrastructure (shared) | $3,600 | $8,400 | $14,400 |
| Infrastructure (football incremental) | $400 | $600 | $1,200 |
| Data licensing (baseball) | $5,000 | $15,000 | $25,000 |
| Data licensing (football) | $0 | $3,000 | $10,000 |
| Marketing (baseball) | $6,000 | $18,000 | $30,000 |
| Marketing (football) | $3,000 | $8,000 | $15,000 |
| Services (Stripe, monitoring, etc.) | $1,800 | $6,600 | $17,100 |
| **Total costs** | **$19,800** | **$59,600** | **$112,700** |
| | | | |
| **NET INCOME** | **$18,125** | **$90,400** | **$267,800** |
| **Net margin** | **47.8%** | **60.3%** | **70.4%** |
| | | | |
| **SUBSCRIBERS** | | | |
| Baseball (year-end) | 400 | 1,300 | 2,500 |
| Football (peak in-season) | 300 | 1,000 | 3,200 |
| Bundle subscribers | 0 | 200 | 800 |
| Unique paying users (deduplicated) | 600 | 1,800 | 4,500 |
| | | | |
| **UNIT ECONOMICS** | | | |
| Blended ARPU (monthly, in-season) | $11.50 | $12.00 | $13.00 |
| Blended CAC | $25 | $22 | $18 |
| LTV (combined) | $180 | $250 | $320 |
| LTV:CAC | 7.2:1 | 11.4:1 | 17.8:1 |

### 6.2 How Football Changes Business Economics

**Positive effects:**

1. **Seasonal smoothing.** Baseball runs Apr-Oct, football runs Sep-Jan. Together they cover 10 of 12 months. The revenue dead zone shrinks from 5 months (Nov-Mar) to 2 months (Feb-Mar). This is the single biggest structural improvement.

2. **Bundle economics improve retention.** Multi-sport subscribers churn ~40% less than single-sport (industry benchmark). A $149.99/yr bundle user who plays both sports has LTV of ~$400+ vs ~$255 for baseball-only.

3. **Shared infrastructure amortization.** The ~$15,000/yr infrastructure cost is shared across 2 products. Football's marginal cost is <$1,200/yr -- essentially free compute.

4. **Cross-sell is the cheapest acquisition channel.** Converting a baseball subscriber to football costs ~$0-5 (email campaign) vs $25-50 for cold acquisition. This dramatically improves blended CAC.

5. **10x larger organic reach.** r/fantasyfootball has 2.2M members vs r/fantasybaseball's 225K. Football SEO keywords have 5-10x the search volume. The content marketing flywheel spins faster.

**Negative effects:**

1. **Development distraction.** Building football diverts ~790 hours from baseball improvements. Must carefully time to baseball off-season (Oct-Feb).

2. **Support complexity doubles.** Two sports means two sets of bugs, data issues, and user questions. Solo developer risk increases.

3. **Brand dilution risk.** HEATER's brand is built on "H2H Categories baseball." Expanding to football could confuse the positioning if not handled carefully.

4. **Football market is far more competitive.** FantasyPros, PFF, 4for4, and others have mature football products. HEATER's "only H2H tool" differentiation is weaker in football (though still relevant -- no competitor does MC simulation for fantasy football either).

### 6.3 Bundle Impact on Unit Economics

| Scenario | Baseball-Only User | Football-Only User | Bundle User |
|----------|-------------------|-------------------|-------------|
| Annual price | $119.99 | $99.99 | $149.99 |
| Active months | 7 (Apr-Oct) | 5 (Sep-Jan) | 10 (Apr-Jan) |
| Effective monthly | $17.14 | $20.00 | $15.00 |
| Estimated retention | 70% YoY | 60% YoY | 82% YoY |
| LTV (3-year) | $255 | $180 | $400 |
| Marginal cost/user/yr | $8.50 | $6.00 | $12.00 |
| Contribution margin | 93% | 94% | 92% |

**Key insight:** Bundle users generate 57% more LTV than baseball-only and 122% more than football-only, while costing only 41% more in marginal costs. The bundle is the highest-value customer segment.

### 6.4 Combined ARR Trajectory

| Timepoint | Baseball ARR | Football ARR | Bundle Uplift | Total ARR |
|-----------|-------------|-------------|--------------|-----------|
| End Year 1 | $57,600 | $39,600 | $0 | $97,200 |
| End Year 2 | $192,000 | $132,000 | $14,400 | $338,400 |
| End Year 3 | $390,000 | $422,400 | $48,000 | $860,400 |

**Year 3 run-rate exceeds $850K ARR** -- within striking distance of the $1-2M ARR target from the business plan, now accelerated by football.

---

## 7. Financial Risks Specific to Football

### 7.1 NFL Data Litigation Risk

| Risk | Assessment | Mitigation |
|------|-----------|-----------|
| **NFL is more aggressive about data rights than MLB** | HIGH probability. The NFL's exclusive deal with Genius Sports (through 2030) signals tight data control. The NFL has historically sent cease-and-desist letters to unauthorized data users. | Use only open-source data (nfl_data_py, which aggregates publicly available data) for beta. Budget for MySportsFeeds commercial license ($1,200-3,600/yr) before charging users. Never scrape NFL.com directly. |
| **PFF grades cannot be used without license** | CERTAIN. PFF grades are proprietary. Even displaying them requires a B2B agreement. | Do not display PFF grades. Build own player grading system from public stats (snap counts, target share, efficiency metrics). Avoid any PFF-derived data. |
| **ESPN/Yahoo API access revoked** | MEDIUM probability. Both APIs are undocumented and could be restricted. ESPN has already stopped issuing new developer keys. | Build manual import as Day 1 feature. Support Sleeper (open API) and Fantrax. Multiple platform support reduces single-platform dependency. |
| **Play-by-play data restricted** | LOW-MEDIUM. nflverse/nfl_data_py has operated for years without issues, but NFL could assert rights. | Cache aggressively. Build fallback projections from box scores only (like Marcel). Maintain 3-year historical data locally. |

**Overall data risk: HIGHER than baseball.** Budget an additional $3,000-10,000/yr for data licensing that baseball does not require. The NFL's centralized data control (Genius Sports exclusive) means there is no equivalent to MLB's generous Stats API.

### 7.2 Competitive Intensity

| Factor | Baseball | Football | Risk |
|--------|----------|----------|------|
| Number of premium competitors | 5-8 | 15-20+ | Much harder to stand out |
| Largest competitor (FantasyPros) | Moderate focus | Primary focus | More resources allocated to football |
| PFF presence | Minimal | Dominant | PFF grades are considered essential by many |
| Free tool quality | Moderate | High (Sleeper, Yahoo, ESPN) | Harder to justify paid when free is good |
| Content creators | Moderate | Saturated | Podcast/YouTube/newsletter space is crowded |

**CAC impact:** Football CAC will be 30-50% higher than baseball. Baseball blended CAC of $35 suggests football blended CAC of $45-55. This is offset by the larger market (10x organic reach) and cross-sell channel ($0 CAC).

**Differentiation challenge:** HEATER's "only MC simulation tool" claim is equally valid in football (no competitor runs 10K sims for fantasy football either), but the gap between HEATER and competitors is perceived as smaller because football analysis is less stat-intensive than baseball. Football is more about "watch the tape" culture.

### 7.3 Seasonality Risk

| Factor | Baseball | Football | Combined |
|--------|----------|----------|----------|
| Active season | 7 months (Apr-Oct) | 5 months (Sep-Jan) | 10 months (Apr-Jan) |
| Revenue months | 7 + 2 off-season | 5 + 2 off-season | 10 + 2 off-season |
| Overlap months | -- | -- | Sep-Oct (both active, peak engagement) |
| Dead months | Nov-Mar (5 months) | Feb-Aug (7 months) | Feb-Mar only (2 months) |
| Churn timing | Oct -> Nov | Jan -> Feb | Staggered, reduces cliff |

**Football's 5-month season is a real weakness individually** -- 28% less revenue time than baseball. But as a **complement to baseball, football is transformative:** it eliminates 3 months of dead revenue and creates a Sep-Oct overlap period where both products are active simultaneously (maximizing bundle value and engagement).

**The seasonality risk is INVERTED when combined.** Football alone has worse seasonality than baseball. Football + baseball together have dramatically better seasonality than either alone.

### 7.4 Cross-Sell Risk

**What if baseball users don't want football?**

| Scenario | Cross-Sell Rate | Football Year 1 Impact | Mitigation |
|----------|----------------|----------------------|-----------|
| **Bull case** | 55% of eligible convert | 175 cross-sell subs, total 500 | None needed |
| **Base case** | 47% of eligible convert | 125 cross-sell subs, total 350 | As modeled |
| **Bear case** | 25% of eligible convert | 60 cross-sell subs, total 250 | Increase paid acquisition budget by $2,000 |
| **Worst case** | 10% of eligible convert | 25 cross-sell subs, total 175 | Football must stand on its own; delay marketing spend until product-market fit proven |

Even in the worst case (10% cross-sell), football still acquires 150+ users from organic/paid channels because the market is so much larger than baseball. The cross-sell is gravy, not the foundation.

**Survey validation:** Before building, survey baseball subscribers: "Would you use a HEATER football product?" and "How much would you pay?" This is a $0 risk-reduction step.

### 7.5 Risk Summary Matrix

| Risk | Probability | Financial Impact | Severity | Primary Mitigation |
|------|------------|-----------------|----------|-------------------|
| NFL data access restricted | Medium (30%) | $10-50K additional licensing | HIGH | Use open-source data; budget for licenses |
| Higher-than-expected CAC | High (50%) | $5-15K additional Year 1 spend | MEDIUM | Lean on cross-sell and organic channels |
| Low cross-sell conversion | Medium (25%) | $5-10K revenue shortfall Year 1 | MEDIUM | Survey first; football can stand alone |
| Football market too competitive | Medium (35%) | Lower conversion rates, slower growth | MEDIUM | MC simulation differentiation still unique |
| Solo developer burnout | High (40%) | 3-6 month delay in football launch | HIGH | Begin football in baseball off-season; consider part-time contractor |
| 5-month season insufficient for retention | Medium (30%) | Higher churn, lower LTV | MEDIUM | Strong off-season features; bundle pricing |

---

## 8. Sensitivity Analysis

### 8.1 Football Revenue Sensitivity (Year 2)

| Variable | Bear (-30%) | Base | Bull (+30%) |
|----------|------------|------|-------------|
| Football subscribers (peak) | 700 | 1,000 | 1,300 |
| Football ARPU | $8/mo | $11/mo | $14/mo |
| Football annual revenue | $29,400 | $54,600 | $82,000 |
| Incremental net revenue | $18,200 | $40,400 | $64,800 |

### 8.2 Combined Revenue Sensitivity (Year 3)

| Scenario | Baseball | Football | Bundle | Total | Net Income |
|----------|----------|----------|--------|-------|-----------|
| **Bear** | $134,000 | $119,000 | $9,000 | $262,000 | $165,000 |
| **Base** | $192,000 | $170,500 | $18,000 | $380,500 | $267,800 |
| **Bull** | $250,000 | $222,000 | $36,000 | $508,000 | $385,000 |

### 8.3 Break-Even Analysis (Football Only)

| Metric | Value |
|--------|-------|
| Monthly fixed costs (football incremental) | ~$350/mo |
| Variable cost/user | ~$0.50/mo |
| Contribution margin | $10.50/user/mo at $11 ARPU |
| **Break-even (football-only)** | **~34 paying users** |
| Time to break-even | Month 2 of first NFL season (Oct 2027) |

Football breaks even almost immediately because the fixed costs are so low (all infrastructure is shared with baseball). If HEATER has 34 football subscribers by October 2027, football is cash-flow positive.

---

## 9. Implementation Timeline

| Phase | Timing | Milestone | Investment |
|-------|--------|-----------|-----------|
| **Design** | Oct-Nov 2026 | NFL stat model architecture, data source evaluation, survey baseball users | 80 hrs, $0 |
| **Core Engine** | Dec 2026 - Feb 2027 | Projection engine, trade analyzer, lineup optimizer (football-specific) | 300 hrs, $0 |
| **Data Pipeline** | Jan-Feb 2027 | nfl_data_py integration, ESPN/Sleeper league sync, injury feeds | 80 hrs, $0 |
| **UI + Testing** | Mar-Apr 2027 | Football pages, 1,500+ tests, integration testing | 200 hrs, $0 |
| **Closed Alpha** | May-Jun 2027 | 20-50 testers from football communities + baseball subscribers | 40 hrs, $0 |
| **Open Beta** | Jul-Aug 2027 | Pre-draft tools live, draft simulator, mock drafts | 60 hrs, $500 (hosting) |
| **Paid Launch** | Sep 1, 2027 | NFL Week 1, full paid tiers live | 30 hrs, $3,000 (marketing) |
| **Growth** | Sep 2027 - Jan 2028 | In-season iteration, content marketing, cross-sell campaigns | ongoing, $1,500 (ads) |

**Total pre-revenue investment: ~790 hours + ~$5,000 cash.**

---

## Appendix A: Football Competitor Feature Comparison

| Feature | FantasyPros | PFF+ | 4for4 | HEATER Football |
|---------|-------------|------|-------|----------------|
| Expert consensus rankings | Yes | No (own grades) | Yes | Yes + MC simulation |
| Monte Carlo trade analysis | No | No | No | **Yes (10K sims)** |
| Game theory trade valuation | No | No | No | **Yes** |
| LP-constrained lineup optimization | No | No | No | **Yes** |
| FAAB bid optimizer | No | No | No | **Yes** |
| Multi-platform sync | Yes (6 platforms) | No | Yes (LeagueSync) | Yes (Yahoo/ESPN/Sleeper) |
| Player grades | No (expert consensus) | Yes (proprietary) | No | Own model (public stats) |
| Bayesian projection updating | No | No | No | **Yes** |
| Weekly matchup probabilities | Basic | Basic | Basic | **Category-level simulation** |
| Streaming DST/K optimizer | No | No | Limited | **Yes (LP-optimized)** |
| Price (season) | $36-72 | $99-120 | $29-59 | $40-80 |

## Appendix B: Sources

- FantasyPros Pricing: https://www.fantasypros.com/premium/plans/bp/
- PFF Membership: https://www.pff.com/lp/membership
- 4for4 Plans: https://www.4for4.com/plans
- RotoWire Pricing: https://www.rotowire.com/subscribe/pricing/
- Fantasy Alarm Pricing: https://www.fantasyalarm.com/pricing
- FantasyLife+ Pricing: https://www.fantasylife.com/pricing
- The Athletic Pricing: https://awfulannouncing.com/athletic/the-athletic-increasing-annual-subscription-rate-71-99.html
- nfl_data_py (MIT): https://github.com/nflverse/nfl_data_py
- nflverse: https://github.com/nflverse
- Sleeper API: https://docs.sleeper.com/
- Sports Reference Data Use: https://www.sports-reference.com/data_use.html
- SportsRadar NFL API: https://marketplace.sportradar.com/products/64d179bb0a92ec119620d9d5
- Genius Sports NFL Partnership: https://www.espn.com/nfl/story/_/id/45493514/nfl-extends-expands-exclusive-data-deal-genius-sports
- PFF B2B Data: https://b2b.pff.com/data
- MySportsFeeds Pricing: https://www.mysportsfeeds.com/feed-pricing/
- FSGA Industry Demographics: https://thefsga.org/industry-demographics/
- Fantasy Football Market Size: https://www.globalgrowthinsights.com/market-reports/fantasy-football-market-102793
- Fantasy Sports Market: https://www.mordorintelligence.com/industry-reports/fantasy-sports-market
- CAC Benchmarks: https://firstpagesage.com/reports/average-cac-for-startups-benchmarks/
- SaaS Churn Benchmarks: https://www.wearefounders.uk/saas-churn-rates-and-customer-acquisition-costs-by-industry-2025-data/
- FTN Data: https://ftnfantasy.com/stats/sports-data
