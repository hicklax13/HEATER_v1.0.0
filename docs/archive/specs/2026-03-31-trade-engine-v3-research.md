# Trade Engine V3 — Deep Research Findings

**Date:** 2026-03-31
**Scope:** 5 research agents, 200+ sources, answering 5 critical questions + missing factors audit

## Research Summary

### Question 1: Multi-Player Trades (2-for-1, 4-for-4)
- Current engine only scans 1-for-1. Need greedy expansion (Tier 2) from top 1-for-1 seeds
- Roster spot value: ~$2-3 per slot, or ~0.15 SGP/slot/week (decaying over season)
- Drop cost = SGP of dropped player minus replacement-level FA SGP
- Cap at 3 players per side (academic consensus from arXiv papers)
- Greedy expansion keeps runtime under 5 seconds (vs 30s for exhaustive search)
- ESPN uses cosine dissimilarity + 0-1 knapsack (HEATER already has cosine)

### Question 2: Opponent Modeling
- Loss aversion lambda = 1.8 (meta-analysis consensus)
- Endowment effect: owners value their players ~2x over equivalent players
- Status quo bias: ~60-80% of objectively beneficial trades get rejected
- Manager archetypes: 5 types (Recency Bias, Stuck-in-Draft, Balancer, Young Optimist, Blockbuster)
- Trade acceptance rate for cold proposals: ~20-30% (much lower than the 88% for self-selected proposals)
- Need-matching is the #1 predictor of acceptance

### Question 3: Schedule-Aware Urgency
- Easy upcoming schedule = lower trade urgency (but not zero)
- Bubble teams (5th-7th place) have highest trade urgency
- Early season trades have highest value (more weeks to benefit)
- Opportunity cost of waiting: recovery difficulty increases weekly as denominator shrinks
- Playoff probability delta > 5% = high urgency trade

### Question 4: Category Correlation
- HR-RBI: r = 0.86 (nearly redundant)
- HR-R: r = 0.74 (strong coupling)
- AVG-OBP: r = 0.70 (tight coupling)
- SB-everything: r < 0.12 (independent — puntable or premium depending on strategy)
- ERA-WHIP: r = 0.81-0.84 (very tight)
- W and SV are functionally independent of quality metrics
- Standard SGP double-counts power cluster by ~15-20%
- SB independence means SB production is genuinely additive (premium, not penalty)

### Question 5: Missing Factors for Maximum Accuracy
**High-impact additions identified:**
1. Statcast regression signals (xwOBA-wOBA delta for buy-low/sell-high)
2. Aging curve discount (0.5 WAR/year after 30)
3. Two-start pitcher schedule value
4. Prospect ETA value
5. Ownership trend momentum
6. Endowment effect adjustment (require 15-20% surplus before recommending)
7. Multi-player trade roster spot valuation
8. Consolidation premium
9. Time-value depreciation
10. Trade timing advisor

**Data availability:**
- Yahoo provides: rosters, standings, matchup scores, FA pool, transactions, draft results, schedule, settings
- Yahoo DOESN'T provide: Statcast, lineup order, minor league stats, injury history depth
- External sources fill gaps: pybaseball (Statcast), MLB Stats API (transactions, schedule, prospects), ESPN API (injuries)
- IMPOSSIBLE to fill: opponent subjective preferences, league politics, undisclosed injuries

## Key Sources (top 20)
- arXiv:2511.17535 — Genetic Algorithm for Fantasy Football Trades
- arXiv:2111.02859 — ESPN Large-Scale Diverse Combinatorial Optimization
- FanGraphs Sample Size Library — stat stabilization rates
- Smart Fantasy Baseball — SGP methodology
- Prospect Theory (Kahneman & Tversky) — loss aversion framework
- Brown et al. 2024 Meta-Analysis — lambda = 1.955
- FanGraphs Community — category correlation analysis
- Columbia/Imperial — DFS portfolio theory
- Oddsmyth — trade evaluation framework
- Baseball Trade Values — surplus value methodology
