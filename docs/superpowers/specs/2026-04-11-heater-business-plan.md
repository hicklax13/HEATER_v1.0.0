# HEATER Business Plan

**Date:** April 11, 2026
**Author:** Connor Hickey
**Status:** Draft v1.0

---

## Executive Summary

HEATER is a fantasy baseball in-season management platform built on Monte Carlo simulation, Bayesian projection updating, LP-constrained lineup optimization, and game theory trade analysis. It is the most analytically deep fantasy baseball tool in existence -- purpose-built for H2H Categories leagues, a format no competitor specifically serves.

The fantasy baseball market is $1.5B (US) and growing at 9.8% CAGR. HEATER targets the ~350,000 serious H2H Categories players who currently spend $100-300/year across 3-5 fragmented tools. By consolidating draft, in-season management, trades, lineups, and war room strategy into a single integrated platform powered by 10,000-simulation Monte Carlo analysis, HEATER fills a gap no competitor occupies: **high analytical depth + high ease of use for H2H Categories**.

The business launches as a bootstrapped side project (Phase A: beta), graduates to a lean SaaS (Phase B), and scales to a multi-sport platform (Phase C) with football expansion in Year 2. Three-year revenue projection: $4.5K (Y1) -> $21.6K (Y2) -> $90K (Y3), reaching $192K ARR run-rate by end of Year 3. Unit economics are strong: 95% gross margins, 7.3:1 LTV:CAC ratio, break-even at 26 paying users.

---

## 1. Company Overview

### Mission

Give serious fantasy baseball managers a quantitative edge their league-mates don't have -- replacing gut-feel decisions with simulation-backed recommendations.

### Product

HEATER is a Streamlit-based (migrating to Next.js + FastAPI) fantasy baseball platform with:

- **11 pages:** My Team (War Room), Draft Simulator, Trade Analyzer, Free Agents, Lineup Optimizer, Player Compare, Closer Monitor, League Standings, Leaders/Prospects, Trade Finder, Matchup Planner
- **Core engines:** 7-system Ridge regression projection blending, 10K-sim paired Monte Carlo trade analyzer with game theory, 21-module LP-constrained lineup optimizer with sigmoid-calibrated H2H urgency, Bayesian SGP updating
- **Data sources:** MLB Stats API, FanGraphs (7 projection systems), pybaseball (Statcast), Yahoo Fantasy API, Open-Meteo (weather), plus 10 additional sources
- **Quality:** 3,180+ passing tests, backtesting framework, sensitivity analysis, 115/115 roadmap items complete
- **Tech:** Python, Streamlit, SQLite, pandas/NumPy/SciPy/PuLP/Plotly, GitHub Actions CI

### Corporate Structure

Umbrella company (e.g., "Heater Analytics LLC") holds:
- **HEATER Baseball** -- the initial product (MLB 2026+ seasons)
- **HEATER Football** -- Year 2 expansion (NFL 2027+ seasons)
- Future sport expansions under the HEATER brand

---

## 2. Market Analysis

### 2.1 Market Size

| Metric | Value | Source |
|--------|-------|--------|
| US fantasy sports players | 53M traditional + 31M betting | FSGA 2025 |
| Fantasy baseball players (US/CA) | 12-15M | FSGA / Deep Market Insights |
| US fantasy baseball market | $1.5B (2024), $3.8B projected (2030) | Deep Market Insights |
| CAGR | 9.8% | Deep Market Insights |
| Global fantasy sports market | $31-33B (2025), $67-97B by 2033 | IMARC / SNS Insider |

### 2.2 Target Market Segmentation

| Segment | Size | Willingness to Pay | HEATER Fit |
|---------|------|--------------------|-----------| 
| Casual (free tools only) | ~8-9M (65%) | $0/yr | Not target |
| Engaged (light tool usage) | ~2.5-3M (20%) | $0-50/yr | Secondary |
| Serious (multiple leagues, some paid) | ~1.2-1.5M (10%) | $50-150/yr | Primary |
| Hardcore (high-stakes, multi-tool) | ~500-750K (5%) | $150-500+/yr | Core |

**HEATER's Serviceable Addressable Market (SAM):** ~350,000 serious H2H Categories players
**Niche TAM:** $37M-$105M at $100-200/year per user

### 2.3 Demographics

- **Core age:** 25-44 (58% of participants)
- **Gender:** ~64% male, trending more balanced
- **Income:** 43% earn $50-100K HHI; 23% earn $100K+; median $80K+
- **Profile:** Smartphone-first, high digital engagement, above-average tech savviness
- **Ideal persona ("Analytical Andy"):** Male, 28-42, $80K+ HHI, 2-4 leagues, currently spends $100-300/yr on fragmented tools, works in tech/finance/engineering/data science, Yahoo Fantasy user

### 2.4 Growth Trends

Fantasy baseball is growing, driven by:
- MLB's Statcast data ecosystem making analytics accessible
- Legalized sports betting cross-pollination
- Near-universal smartphone connectivity
- AI/ML integration attracting analytically-minded players
- Rising interest from younger demographics

---

## 3. Competitive Analysis

### 3.1 Competitive Landscape

| Competitor | Price (Annual) | Strengths | Key Weakness |
|------------|---------------|-----------|-------------|
| **FantasyPros** | $48-108 | Expert consensus, multi-platform sync, best UI | No quantitative modeling; trade analyzer is simple value comparison |
| **RotoWire** | $50-120 | Fastest player news (industry gold standard) | DFS-focused; shallow season-long tools |
| **FanGraphs** | $100-180 | Best projections/stats database | Not a fantasy management platform; no lineup/trade tools |
| **BaseballHQ** | $99 | 30+ year track record, NFBC credibility | Aging platform; content-first, not tool-first |
| **Baseball Prospectus** | $40-55 | PECOTA projections | Fantasy tools limited to draft prep |
| **RotoBaller** | Freemium | Best free trade analyzer | Trade analyzer is basic side-by-side comparison |
| **Yahoo Built-in** | $8/yr (Plus) | Clean UI, 81% win-rate claim | AI recommendations poor quality; shallow analytics |
| **ESPN/CBS** | Free-$$ | Large user bases | Broken rankings; cluttered UI; limited format support |

### 3.2 Positioning Map

```
                        ANALYTICS DEPTH
                    Low ──────────────── High
              ┌─────────────────────────────────┐
         High │  FantasyPros     │              │
              │  Yahoo Plus      │   HEATER     │
    EASE OF   │  RotoBaller      │   (target)   │
     USE      │  ESPN/CBS        │              │
              ├──────────────────┼──────────────┤
         Low  │  Razzball        │  FanGraphs   │
              │  Underdog        │  BaseballHQ  │
              │                  │  BProspectus │
              └─────────────────────────────────┘
```

HEATER's strategic position: **High analytics + High ease of use** -- a quadrant nobody occupies.

### 3.3 Competitive Moat

1. **No competitor serves H2H Categories specifically.** Every tool is format-agnostic or Roto/Points-focused.
2. **Trade analysis is the biggest gap.** Every existing analyzer is simple value comparison. HEATER's Monte Carlo + game theory approach is genuinely novel.
3. **21-module optimizer pipeline** with sigmoid-calibrated urgency, Bayesian updating, and LP constraints exceeds anything commercially available.
4. **Backtesting framework** provides transparency about model accuracy -- competitors don't publish accuracy metrics.
5. **League-specific context** (via Yahoo API) makes recommendations personalized, not generic.

---

## 4. SWOT Analysis

### Strengths
- **Quantitative depth unmatched in market:** 7-system projection blending, 10K MC sims, LP optimization, Bayesian SGP, game theory
- **All-in-one integration:** Draft + in-season + trades + lineup + war room in one platform
- **Engineering quality:** 3,180+ tests, backtesting, sensitivity analysis, CI pipeline
- **H2H Categories specialization:** Purpose-built for underserved format
- **Yahoo API live integration:** 3-tier caching, real-time matchup data
- **115/115 roadmap items complete:** Mature, feature-complete product

### Weaknesses
- **Single-tenant Streamlit architecture:** Cannot scale to multi-user SaaS
- **No user authentication or multi-tenancy:** One instance per user
- **Data licensing risk:** FanGraphs/FantasyPros scraping may violate ToS for commercial use
- **SQLite database:** Not production-grade for concurrent users
- **Desktop-only:** No mobile experience
- **Solo developer:** Bus factor of 1
- **No deployment infrastructure:** No Docker, cloud hosting, or load balancing

### Opportunities
- **H2H Categories is unserved:** Zero competitors built for this specific format
- **Trade analysis gap:** Market-wide weakness; HEATER's MC+game theory is 10x better
- **Tool fragmentation:** Users juggle 3-5 subscriptions; HEATER consolidates
- **AI integration:** Add conversational AI layer using HEATER's engines as context
- **Multi-sport expansion:** Football (4-6x larger market), complementary seasonal calendar
- **B2B licensing:** White-label trade engine or projections API to content platforms
- **Serious player willingness to pay:** $150-500+/year across fragmented tools

### Threats
- **Yahoo API dependency:** Yahoo could restrict access or change terms
- **FantasyPros could deepen analytics:** Largest user base + resources to build MC/LP
- **AI commoditization:** ChatGPT/Claude providing "good enough" generic fantasy advice
- **Data licensing crackdown:** FanGraphs, Statcast, or other providers restrict access
- **Fantasy baseball participation ceiling:** Lags far behind football (3-4x smaller)
- **Seasonality:** 7-month active season creates revenue volatility

---

## 5. Product Strategy

### 5.1 Current State

58,000+ lines of Python, 150 test files, fully functional for single-user use. Analytical engines are production-quality. Infrastructure (auth, database, deployment, scalability) requires rebuild for SaaS.

### 5.2 P0 Infrastructure Requirements (Must-Have for Launch)

| Gap | Current State | Required State | Effort |
|-----|--------------|----------------|--------|
| **Authentication** | Yahoo OAuth only, no user system | Clerk/Auth0 with user registration, JWT sessions | 2-3 weeks |
| **Database** | Single SQLite file, no user isolation | PostgreSQL with tenant-scoped tables, Alembic migrations | 3-4 weeks |
| **Frontend** | Streamlit (desktop-only, full-page rerenders) | Next.js/React with responsive design | 8-12 weeks |
| **API layer** | None (UI and logic coupled) | FastAPI backend serving REST API | 4-6 weeks |
| **Deployment** | localhost only | Vercel (frontend) + Railway (Python backend) + Neon (PostgreSQL) | 2-3 weeks |
| **Multi-platform** | Yahoo only | + ESPN, Fantrax integration | 4-6 weeks |
| **Mobile** | None | Responsive PWA via Next.js | Included in frontend rebuild |
| **Payments** | None | Stripe subscription billing | 1-2 weeks |

### 5.3 P1 Features (Ship in First Month Post-Launch)

- Push notifications / alerts (injury, lineup lock, waiver deadline)
- AI chat assistant leveraging existing engines
- Background job queue (Celery + Redis) for heavy computation
- Proper OAuth 2.0 redirect flow (replace Yahoo OOB)

### 5.4 P2 Features (Ship During Season)

- Social features (league chat, trade discussion workspace)
- AI-generated analysis narratives
- Dynasty/keeper league support
- Commissioner tools (custom scoring)
- Redis caching layer

### 5.5 Missing Features vs Market Demand

| Feature | User Demand | HEATER Has It? | Priority |
|---------|------------|----------------|----------|
| Multi-platform (ESPN/Fantrax/Sleeper) | High | No (Yahoo only) | P0 |
| Mobile experience | High | No | P0 |
| Push notifications | High | No | P1 |
| AI chat assistant | High | No (engines exist, needs UI) | P1 |
| Shareable analysis cards | Medium | No | P1 |
| Social/community features | Medium | No | P2 |
| Dynasty/keeper support | Medium | No | P2 |
| Content + tools integration | Medium | Partial (news/sentiment) | P2 |

---

## 6. Go-to-Market Strategy

### 6.1 Launch Timeline

| Phase | Timing | Milestone |
|-------|--------|-----------|
| Closed Alpha | Oct-Nov 2026 | 20-50 testers from serious H2H communities |
| Open Beta (Free) | Jan-Feb 2027 | Draft tools + trade analyzer, collect feedback |
| Paid Launch | Late Feb - Opening Day 2027 | Convert beta users, full in-season features live |
| Growth Push | May-Jun 2027 | Content marketing, partnerships, paid ads |
| Trade Deadline Peak | Jul 2027 | Trade analyzer showcase, highest engagement |

### 6.2 MVP for Launch

**Must-have (Day 1):**
1. Yahoo API integration (roster sync, standings, matchups)
2. Trade Analyzer (Monte Carlo + game theory -- the marquee differentiator)
3. Lineup Optimizer (daily start/sit with LP optimization)
4. Player Rankings / Projections (blended 7-system)
5. Matchup Planner (weekly H2H category probability)

**Ship in first month:** Free Agent recommender, War Room, Trade Finder

### 6.3 Positioning Statement

**For serious H2H Categories fantasy baseball players** who are frustrated by generic tools that don't understand weekly matchup dynamics, **HEATER is the only integrated in-season management platform** that combines Monte Carlo trade simulation, LP-constrained lineup optimization, and Bayesian projection updating -- all purpose-built for head-to-head category scoring. **Unlike FantasyPros and other expert-consensus tools**, HEATER uses quantitative modeling with 10,000 simulations per trade, 21-module lineup optimization, and real-time Yahoo league integration to make decisions grounded in math, not opinions.

---

## 7. Marketing Strategy

### 7.1 Brand

- **Name:** HEATER (all-caps treatment)
- **Tagline:** "10,000 simulations. One decision."
- **Visual identity:** Thermal/fire palette (amber/flame), glassmorphic design, data visualization as brand element
- **Voice:** Confident, data-first, accessible technical. "Your league-mates are guessing. You're simulating."
- **Domain:** Secure heaterfantasy.com, getheater.app, or similar (heater.com is generic/taken; Heater Sports is an equipment brand)

### 7.2 Channel Strategy

| Channel | Strategy | Budget | Expected CAC |
|---------|----------|--------|-------------|
| **Reddit** (r/fantasybaseball) | Daily engagement, free analysis, AMAs | $0 (time) | $5-15 |
| **Twitter/X** | Hot takes backed by HEATER data, engage analysts | $0 (time) | $10-25 |
| **SEO/Content** | Weekly trade value chart (free), blog posts, guides | $0 (time) | $15-40 |
| **Discord** | HEATER subscriber community, daily advice | $0 (time) | $0 (retention) |
| **YouTube** | Screen recordings of real trade analysis | $0 (time) | $20-40 |
| **Podcast sponsorship** | Fantasy baseball shows (Rates & Barrels, etc.) | $200-500/ep | $30-60 |
| **Reddit Ads** | Targeted at r/fantasybaseball | $500-1K/mo | $15-35 |
| **Google Ads** | High-intent keywords (retargeting focus) | $500-1K/mo | $50-100 |

**Primary channels (Year 1):** Reddit, Twitter/X, SEO content -- all organic, $0 cost.
**Scale channels (Year 2+):** Podcast sponsorship, Reddit Ads, Google Ads retargeting.

### 7.3 Content Strategy

- **Weekly during season:** Trade value chart (free, SEO magnet), waiver wire column, start/sit rankings, matchup previews
- **Bi-weekly:** Prospect watch, closer monitor update
- **Monthly:** Strategy deep-dives, methodology explainers
- **Off-season:** Keeper rankings, mock draft guides, projection previews

**The free trade value chart is the single most important marketing asset.** "The only trade value chart powered by 10,000 simulations" is a claim no competitor can make.

### 7.4 Influencer & Partnership Strategy

- **Tier 1 (awareness):** Pitcher List, Razzball, FanGraphs/RotoGraphs writers
- **Tier 2 (best ROI):** Mid-tier Twitter/X analysts, podcast hosts, Discord leaders
- **Tier 3 (micro):** Active r/fantasybaseball contributors, league commissioners
- **Structure:** Free lifetime subscription + 20-25% recurring affiliate commission

### 7.5 Referral Program

- Referrer: 1 free month per signup (stacks to 6 months)
- Referee: 14-day extended trial
- League bonus: 6+ league-mates subscribe -> everyone gets 20% off
- Target viral coefficient: 0.3-0.5

---

## 8. Pricing Strategy

### 8.1 Tier Structure

| Tier | Monthly | Season (7 mo) | Annual | Features |
|------|---------|---------------|--------|----------|
| **Free** | $0 | $0 | $0 | Basic trade value chart, top 100 rankings, 1 league, 2 trades/day, mock draft (1/day) |
| **Pro** | $9.99 | $49.99 | $79.99 | Full lineup optimizer, deterministic trade analyzer, FA rankings, matchup planner, war room, multi-league |
| **Elite** | $19.99 | $99.99 | $119.99 | Everything in Pro + Monte Carlo simulation (10K sims), game theory trade analysis, Bayesian projections, daily optimizer (DCV), streaming advisor, sensitivity analysis |

### 8.2 Pricing Rationale

- **Pro at $9.99/mo** sits between FantasyPros MVP ($8.99) and FanGraphs FG+ ($15/mo)
- **Elite at $99.99/season** = $14.29/mo over 7 months, competitive with RotoWire ($19.99) while more analytically deep
- **Season pricing gives 28-29% discount** over monthly, incentivizing commitment
- **Free tier gates computationally expensive features** (MC simulation, LP optimization) behind paywall
- **Annual pricing reduces churn by 40-60%** -- default to annual

### 8.3 Free Trial

14-day full Elite access, no credit card required. Opt-in trials convert at ~18% (vs 49% CC-required), but build trust and maximize top-of-funnel for a niche product.

### 8.4 Multi-Sport Bundle (Year 2+)

| Plan | Baseball Only | Football Only | All-Sports |
|------|-------------|--------------|-----------|
| Annual | $119.99 | $119.99 | $149.99 (37% savings) |
| Monthly | $14.99 | $14.99 | $19.99 |

---

## 9. Financial Projections

### 9.1 Startup Costs

| Category | Optimistic (Bootstrap) | Base | Pessimistic |
|----------|----------------------|------|-------------|
| Infrastructure (12 mo) | $1,500 | $2,940 | $5,856 |
| Third-party services (12 mo) | $0 | $240 | $240 |
| Data licensing | $0 (personal-use risk) | $15,000 | $50,000 |
| Development | $8,125 | $56,800 | $112,500 |
| **Total Year 1** | **$9,625** | **$74,980** | **$168,596** |

### 9.2 Infrastructure Costs at Scale

| Service | 100 Users | 500 Users | 1,000 Users | 5,000 Users |
|---------|-----------|-----------|-------------|-------------|
| Vercel Pro (frontend) | $20 | $20 | $20 | $40 |
| Railway (Python backend) | $15 | $45 | $80 | $250 |
| Neon PostgreSQL | $5 | $15 | $30 | $100 |
| Upstash Redis | $0 | $0 | $5 | $20 |
| Monitoring (PostHog/Sentry) | $0 | $0 | $26 | $76 |
| **Total/mo** | **$42** | **$82** | **$163** | **$488** |

### 9.3 Revenue Projections (3-Year)

**Assumptions:** $12 blended ARPU, 7-month active season, 35% off-season retention at $4.99/mo, 5% in-season monthly churn, 15% trial conversion.

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| **Revenue** | $4,500 | $21,600 | $90,000 |
| Paying customers (year-end) | 90 | 400 | 1,300 |
| MRR (year-end) | $1,080 | $4,800 | $16,000 |
| ARR run-rate | $12,960 | $57,600 | $192,000 |
| Infrastructure | $1,500 | $3,600 | $8,400 |
| Marketing | $0 | $6,000 | $18,000 |
| Data licensing | $0 | $5,000 | $15,000 |
| Services (Stripe, etc.) | $200 | $1,000 | $4,000 |
| **Net Cash Flow** | **$2,800** | **$6,000** | **$44,600** |
| **Cumulative Net** | **$2,800** | **$8,800** | **$53,400** |

### 9.4 Unit Economics

| Metric | Pessimistic | Base | Optimistic |
|--------|------------|------|-----------|
| Blended CAC | $50 | $35 | $20 |
| Blended LTV | $150 | $255 | $380 |
| **LTV:CAC** | **3.0:1** | **7.3:1** | **19.0:1** |
| Payback period (months) | 5.0 | 2.9 | 1.7 |
| Gross margin/user/mo | 88% | 92% | 95% |

### 9.5 Break-Even Analysis

- **Monthly fixed costs (base):** ~$300/mo
- **Variable cost/user:** ~$0.55/mo (compute + Stripe fees)
- **Contribution margin:** $11.45/user/mo at $12 ARPU
- **Break-even:** ~26 paying users
- **Break-even on $75K startup investment:** Mid-Year 3

---

## 10. Growth Strategy

### 10.1 Product-Led Growth

**Free tier as conversion engine:**
- Free trade value chart (public, weekly) -- SEO magnet
- Basic trade analyzer (2/day, no MC depth) with clear upgrade trigger: "Want to see the Monte Carlo simulation? 10,000 scenarios show you win 73% of outcomes. Upgrade to see full analysis."
- Free mock draft (1/day)
- League sync (Yahoo/ESPN) even on free tier

**Shareable viral loops:**
- Branded trade analysis cards for league chats
- Weekly lineup cards ("My HEATER-optimized lineup for Week 12")
- Season report card graphic
- All include "Analyzed by HEATER" with link

### 10.2 League Adoption Flywheel

One user wins -> league-mates investigate -> compound adoption. Accelerate with:
- League-wide dashboards (unlock at 3+ subscribers in same league)
- Trade negotiation workspace (HEATER-to-HEATER proposals)
- Commissioner league report (free, branded)
- League invite discount (6+ members -> 20% off for all)

### 10.3 Retention (The Seasonal Challenge)

**Target: 70-75% year-over-year retention (Sleeper benchmark)**

**Off-season features (Nov-Mar):**
- Season recap dashboard (how did HEATER's recs perform?)
- Keeper/dynasty analysis
- Mock draft simulator (year-round)
- Prospect watch with projection engine
- Off-season reduced tier at $4.99/mo

**Re-engagement calendar:**
- Nov 1: Season recap + "Your HEATER scorecard"
- Dec 15: Hot stove keeper impact analysis
- Jan 15: "Mock draft simulator is live"
- Feb 15: Early renewal at locked-in price
- Mar 1: "Draft day is coming. Don't go in blind."

### 10.4 Multi-Sport Expansion

| Sport | Launch | Market Multiplier | Reusable Code | New Work |
|-------|--------|-------------------|---------------|----------|
| **Football** | Sep 2027 | 3-4x baseball | 60-70% | 30-40% |
| **Basketball** | Year 3-4 | 1.5x baseball | 70% | 30% |
| **Hockey** | Year 4-5 | 0.5x baseball | 75% | 25% |

**Football timing:** Engine design starts Oct 2026, alpha Jan 2027, beta Aug 2027, GA Sep 2027.

**Seasonal complement:** Baseball (Apr-Oct) + Football (Sep-Jan) = 10 months of active product. Bundle pricing drives year-round retention.

### 10.5 Long-Term Roadmap

| Year | Milestone | Revenue Target |
|------|-----------|---------------|
| **Y1 (2026-27)** | Baseball beta -> lean SaaS, first paying users | $20-50K ARR |
| **Y2 (2027-28)** | Baseball growth + football launch | $250-500K ARR |
| **Y3 (2028-29)** | Multi-sport platform, content arm, API business | $1-2M ARR |
| **Y4-5 (2029-30)** | Scale, potential Series A or acquisition | $2-5M ARR |

### 10.6 Exit Landscape

| Exit Path | Revenue Threshold | Valuation | Timeline |
|-----------|------------------|-----------|----------|
| Lifestyle business | $500K-1M ARR | N/A (cash flow) | Year 2-3 |
| Acqui-hire (FantasyPros/RotoWire) | $500K-2M ARR | 3-5x revenue | Year 3-4 |
| Strategic acquisition (Gambling.com/DraftKings) | $2-5M ARR | 5-8x revenue | Year 4-5 |
| VC-backed growth | $1M+ ARR, 20%+ MoM growth | 10-15x ARR | Year 3+ |

Precedents: RotoWire acquired by Gambling.com (~$30M). PrizePicks acquired by Allwyn ($2.5B). 232 funded fantasy sports startups globally.

---

## 11. Distribution Strategy

### 11.1 Primary: Direct Website

- Next.js frontend on Vercel (global CDN, fast)
- FastAPI Python backend on Railway
- Custom domain (heaterfantasy.com or similar)
- Full SEO control, no revenue share

### 11.2 Secondary: Chrome Extension (Growth Hack)

- Overlay HEATER analysis on Yahoo/ESPN league pages
- Lightweight "trade check" and "start/sit" directly in the user's league interface
- Drives awareness and upgrades

### 11.3 Future: Mobile PWA / Native App

- Responsive PWA via Next.js (included in frontend rebuild)
- Native app (React Native) only if PWA proves insufficient
- iOS/Android app store presence for discoverability (Year 2+)

### 11.4 B2B: White-Label Engine Licensing (Year 2+)

- Expose trade analysis engine as API for content creators
- "Powered by HEATER" trade analyzer widgets embeddable on fantasy blogs
- White-label analytics for league hosting platforms
- Estimated B2B pricing: $5K-25K/year per partner

---

## 12. Risk Analysis

### 12.1 Critical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Yahoo API commercial use violation** | HIGH | CRITICAL | Build manual roster import fallback; add ESPN/Fantrax as alternatives; negotiate Yahoo terms proactively |
| **FanGraphs data access blocked** | MEDIUM | HIGH | Build Marcel-only projection fallback; cache aggressively; negotiate licensing; diversify to SportsDataIO |
| **MLB Stats API commercial block** | MEDIUM | HIGH | Pre-compute and cache data; negotiate MLBAM license proactively; budget $5-15K/yr |
| **AI commoditization** | HIGH | MEDIUM | Lean into AI -- add "HEATER AI" chat using engines as context; league-specific context is the moat chatbots can't replicate |
| **FantasyPros adds MC/LP features** | MEDIUM | MEDIUM | H2H Categories specialization is hard to replicate; speed of iteration as solo dev is advantage; build community switching costs |
| **Seasonality revenue gap** | CERTAIN | MEDIUM | Off-season features; multi-sport expansion; seasonal pricing model; build reserves during peak |
| **Solo developer risk** | HIGH | MEDIUM | Build product that runs with minimal maintenance; automate pipelines; consider co-founder in Year 2 |

### 12.2 Data Licensing Strategy

**Phase A (Beta):** Operate under personal-use API terms. No revenue = no commercial use concern.
**Phase B (Lean SaaS):** Begin with MLB Stats API (free/public domain) + own Marcel projections. Defer FanGraphs/FantasyPros licensing.
**Phase C (Full Platform):** Negotiate formal licenses with revenue to fund them. Budget $15-50K/yr.

**Alternative data strategy:** Build proprietary projection system from freely available box scores and game results (what FanGraphs itself does). Marcel projections (already implemented) require zero external data.

---

## 13. Metrics & KPIs

### North Star Metric

**Weekly Active Leagues (WAL):** Number of unique leagues where at least one HEATER subscriber used a paid feature in the past 7 days. Captures engagement AND viral league dynamic.

### Dashboard Metrics

| Category | Metric | Year 1 Target |
|----------|--------|---------------|
| **Growth** | Free signups | 5,000 |
| | Activation (sync league in 7 days) | 40%+ |
| | Viral coefficient (k) | 0.3+ |
| | Time to first value | < 5 min |
| **Engagement** | WAU/MAU (in-season) | 60%+ |
| | Features per session | 3+ |
| | Sessions per week | 4+ |
| | Trade analyses per user/week | 5+ |
| **Revenue** | MRR (year-end) | $4K+ |
| | ARPU (monthly) | $8-10 |
| | Free-to-paid conversion | 5-8% |
| **Retention** | Monthly churn (in-season) | < 3% |
| | Season-over-season retention | 70%+ |
| | NRR | 85%+ |

---

## 14. Implementation Phases

### Phase A: Validate (Now - Dec 2026)
- Deploy HEATER as-is on small VPS or Streamlit Cloud
- Share with league-mates and 20-50 beta testers
- Collect feedback, measure engagement
- Begin content publishing (free trade value chart)
- **Cost:** $0-50/month. **Goal:** Prove weekly usage retention.

### Phase B: Lean SaaS MVP (Jan - Jul 2027)
- Rebuild frontend on Next.js + React (responsive, mobile-ready)
- FastAPI backend wrapping existing Python engines
- Clerk auth + Stripe billing + Neon PostgreSQL
- Add ESPN integration (doubles addressable market)
- Launch free tier + Pro + Elite tiers
- **Cost:** $5-15K development + $300/mo infrastructure. **Goal:** 90+ paying users by Jul 2027.

### Phase C: Multi-Sport Platform (Aug 2027+)
- Football engine build (Jun-Aug 2027)
- Football beta for NFL 2027 preseason
- Cross-sell baseball -> football subscribers
- Bundle pricing
- Content arm, API business, partnership distribution
- **Goal:** 400+ paying users by Dec 2027, $57K+ ARR.

---

## Appendix A: Competitive Pricing Detail

| Competitor | Cheapest Paid | Most Expensive | Annual Option |
|------------|--------------|----------------|---------------|
| FantasyPros PRO | $3.99/mo | $11.99/mo (HOF) | 6-month packages |
| RotoWire | $9.99/mo (one sport) | $22.99/mo (all + DFS) | 60% annual discount |
| FanGraphs FG+ | $15/mo | $15/mo | $80/yr |
| BaseballHQ | $99/yr | $99/yr | Annual only |
| Baseball Prospectus | $39.95/yr | $54.99/yr (Super Premium) | Annual only |
| Fantasy Alarm | $39.97/mo | $39.97/mo | 40% annual discount |
| Yahoo Fantasy Plus | ~$8/yr | ~$8/yr | Annual only |

## Appendix B: Data Source Risk Matrix

| Source | Free for Commercial? | Risk Level | Fallback |
|--------|---------------------|-----------|----------|
| MLB Stats API | Requires MLBAM authorization | Medium | Cache aggressively; negotiate license |
| FanGraphs | No -- educational/personal use only | High | Marcel local projections; license Steamer directly |
| pybaseball (library) | Yes (MIT license) | Low | Library is fine; data sources are the risk |
| Yahoo Fantasy API | No -- ToS prohibits commercial | High | Manual import; ESPN/Fantrax alternatives |
| FantasyPros | No -- freemium, commercial requires agreement | High | Drop ECR or negotiate license |
| Open-Meteo | Yes (Apache 2.0) | None | N/A |
| Baseball Savant/Statcast | Same as MLB (MLBAM) | Medium | Cache; derive from box scores |

## Appendix C: Sources

- FSGA Industry Research (2025): https://thefsga.org/industry-research/
- US Fantasy Baseball Market: Deep Market Insights
- Fantasy Sports Market: IMARC, SNS Insider, Allied Market Research
- FantasyPros Pricing: https://support.fantasypros.com/hc/en-us/articles/25996886459931
- RotoWire Pricing: https://www.rotowire.com/subscribe/pricing/
- FanGraphs Membership: https://plus.fangraphs.com/product/fangraphs-membership/
- BaseballHQ: https://www.baseballhq.com/pricing/subscribe
- SaaS Churn Benchmarks: Vitally, Growigami, MRRSaver
- CAC Benchmarks: Genesys Growth, First Page Sage, Phoenix Strategy Group
- Free Trial Conversion: First Page Sage, Amra & Elma
- Sleeper Growth: DraftKick, Yahoo Sports, CNBC
- PFF Acquisition: Teamworks Blog (2026)
- PrizePicks Acquisition: PrizePicks Press (2025)
- Fantasy Sports Startups: Tracxn
- Google Ads CPC: WordStream (2024 Benchmarks)
- Podcast Advertising: ADOPTER Media, Awisee
