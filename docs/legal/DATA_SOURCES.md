# Data Sources — Attribution, Licensing, and Commercialization Strategy

**Heater Analytics LLC**
Last Updated: [EFFECTIVE_DATE]

---

## Overview

HEATER aggregates data from a variety of external sources to power its projections, analytics, and recommendations. This document catalogues each data source, describes its current licensing status relative to commercial SaaS use, and outlines a phased strategy for achieving full commercial compliance as the product scales.

This document is an internal strategy reference and should be reviewed whenever a new data source is added or when moving between commercial phases.

---

## 1. Free / Public Domain Sources

These sources are either explicitly public domain, published under open-source licenses that permit commercial use, or provided under terms that allow commercial API usage without a separate paid license.

| Source | Data Provided | License / Terms | Commercial Status | Notes |
|--------|--------------|-----------------|-------------------|-------|
| **MLB Stats API** (statsapi.mlb.com) | Player rosters, current and historical stats, schedules, game logs, MiLB stats, injury data | Public domain (no explicit license; MLBAM requires API key for commercial volume) | Requires MLBAM commercial agreement at scale | Currently used under public access. Phase B: obtain MLBAM commercial auth. |
| **Open-Meteo** | Weather data for all 30 MLB stadiums, dome detection, game-time forecast | Apache 2.0 | Permitted for commercial use with attribution | Attribution: "Weather data from Open-Meteo.com." Free tier rate limits may apply at scale. |
| **pybaseball library** | Python wrapper for FanGraphs, Baseball Savant, Retrosheet, and Lahman data | MIT License | Library itself is MIT; underlying data sources have their own terms (see Section 2) | The library license covers the wrapper code, not the data it retrieves. |

---

## 2. Sources Requiring Commercial License

These sources are currently accessed under personal, educational, or research use terms. Their terms of service explicitly or implicitly limit commercial use, resale, or high-volume programmatic access. **A paid commercial license or formal partnership agreement is required before using these sources in a revenue-generating SaaS product.**

| Source | Data Provided | Current Access Method | Commercial Use Restriction | Required Action |
|--------|--------------|----------------------|---------------------------|-----------------|
| **FanGraphs** | Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X projections; wRC+, FIP, Statcast derivatives; ADP data | pybaseball JSON API (personal use) | FanGraphs ToS restricts commercial redistribution of projections and stats. Automated bulk access without a data license is not permitted for commercial products. | Negotiate a data licensing agreement with FanGraphs. Budget: see Phase B/C. |
| **Yahoo Fantasy API** | User league rosters, standings, matchup data, free agents, transactions via OAuth | Personal OAuth (yfpy library) | Yahoo's developer terms allow personal apps. Commercial SaaS products serving multiple users require review and approval under Yahoo's developer program. | Contact Yahoo Developer Network to review commercial terms. Explore Yahoo Fantasy Partner Program. |
| **FantasyPros** | ECR (Expert Consensus Rankings) and ADP aggregations | Web data via pybaseball / scraping | FantasyPros ToS prohibits scraping and commercial use of their rankings data without a data partnership. | Obtain a FantasyPros data partnership or API license, or replace with self-computed ECR from public sources. |
| **Baseball Savant / Statcast** | EV, barrel rate, xwOBA, Stuff+, sprint speed, umpire tendencies, catcher framing (via pybaseball) | pybaseball wrapper (personal use) | Baseball Savant is operated by MLBAM. Same commercial considerations as the MLB Stats API apply. High-volume programmatic access for commercial use requires MLBAM agreement. | Covered under Phase B/C MLBAM commercial agreement. |
| **Steamer / ZiPS / THE BAT creators** | Individual projection systems accessed via FanGraphs | Via FanGraphs API | The projection systems are the intellectual property of their respective creators (Steamer: Jared Cross; ZiPS: Dan Szymborski; THE BAT: Derek Carty). FanGraphs licenses their publication; redistribution in a commercial product requires consent of FanGraphs and potentially the creators. | Covered under FanGraphs data license negotiation. Verify scope of license includes commercial SaaS distribution. |
| **Retrosheet** | Historical game logs used for backtesting | pybaseball / direct download | Free for personal and research use. Commercial use requires permission per retrosheet.org/notice.htm. | Contact Retrosheet for commercial permission, or limit backtesting to MLB Stats API data only. |

---

## 3. Proprietary / Self-Computed Data

These outputs are fully owned by Heater Analytics LLC and have no external licensing dependencies. They represent our core intellectual property and competitive advantage.

| Component | Description | Dependency | IP Owner |
|-----------|-------------|------------|----------|
| **Marcel Projections** | Local implementation of the Marcel projection system (regression-to-mean weighted average of prior seasons). Marcel is a public methodology; our implementation is original code. | Historical stats from MLB Stats API only | Heater Analytics LLC |
| **HEATER Blended Projections** | Ridge regression weighted ensemble of multiple projection systems. The blending weights, training methodology, and output model are proprietary. | Requires individual systems as inputs (licensing risk exists if based on FanGraphs-sourced projections) | Heater Analytics LLC |
| **Monte Carlo Simulations** | Paired Monte Carlo simulation framework (10K+ simulations, variance reduction, seed discipline). Simulation logic, parameters, and convergence diagnostics are original. | No external data dependency beyond player stats | Heater Analytics LLC |
| **SGP Calculator** | Standings Gain Points calculation methodology calibrated to league-specific denominators. Original implementation and scoring approach. | League data (Yahoo) for denominator calibration | Heater Analytics LLC |
| **Trade Analyzer Engine** | 6-phase trade evaluation pipeline (deterministic SGP, MC, signal intelligence, contextual adjustments, game theory, production convergence). Fully proprietary. | Player stats as inputs | Heater Analytics LLC |
| **Lineup Optimizer** | PuLP LP solver with 21-module enhancement pipeline, sigmoid urgency scoring, category urgency weighting. Original formulation and calibration. | Player stats as inputs | Heater Analytics LLC |
| **H2H Category Urgency Model** | Sigmoid-based urgency scoring calibrated via grid search. k-values, calibration framework, and scoring logic are proprietary. | No external dependency | Heater Analytics LLC |
| **War Room / Start-Sit Advisor** | Multi-factor decision models for mid-week pivot and start/sit recommendations. Logic and weighting are original. | Player stats as inputs | Heater Analytics LLC |

---

## 4. Phase A: Free Beta Strategy

**Applicable period:** Pre-revenue / free beta launch.

During Phase A, the product operates under personal, educational, and research use interpretations of the above data sources. This is appropriate while:
- No subscription revenue is being collected.
- The user base is limited and invitation-only or waitlisted.
- The product is being validated and iterated upon.

**Phase A operational rules:**
- All data usage is governed by each source's personal/educational use terms.
- No data retrieved from FanGraphs, Baseball Savant, FantasyPros, or Yahoo APIs is resold, sublicensed, or redistributed to third parties.
- Volume of API requests is kept within reasonable personal-use bounds (no bulk scraping or commercial-scale crawling).
- The Data Sources attribution table (this document) is maintained and updated as new sources are added.
- Legal counsel review of this document is recommended before transitioning to Phase B.

**Risk acknowledgment:** Operating a SaaS platform — even at no charge — may be construed by some data providers as commercial use, particularly Yahoo. The risk is low during closed beta but should be resolved before public launch or revenue collection.

---

## 5. Phase B: Initial Commercial Launch Strategy

**Trigger:** First paid subscriber or public launch, whichever comes first.

Phase B establishes a sustainable data foundation using only cleared sources while negotiations with premium data providers proceed.

**Phase B data stack:**

| Action | Priority | Estimated Cost | Timeline |
|--------|----------|---------------|----------|
| Obtain **MLBAM commercial API authorization** for MLB Stats API access | High | $0–$5,000/yr (varies by agreement) | Before revenue launch |
| Replace FanGraphs-sourced projection systems with **Marcel (local) + Retrosheet/MLBAM data only** as interim fallback | High | $0 | Before revenue launch |
| Remove or gate **FantasyPros ECR** data; replace with self-computed Borda Count from freely available sources or ESPN/Yahoo public rankings | Medium | $0 | Before revenue launch |
| Begin **Yahoo Fantasy Partner Program** application; continue personal OAuth in the interim with limited user volume | Medium | TBD per Yahoo agreement | Apply at launch |
| Budget for **FanGraphs data license** in Year 1 operating plan | Medium | $3,000–$10,000/yr (estimated) | Q1 post-launch |
| Obtain **Retrosheet commercial permission** or remove Retrosheet dependency from backtesting | Low | Likely $0 (goodwill license) | Within 90 days of launch |

**Total estimated Phase B data licensing budget: $5,000–$15,000/year.**

**Phase B self-sufficiency principle:** The product must be able to operate with acceptable (though reduced) analytical quality using only Marcel + MLB Stats API data in the event FanGraphs or FantasyPros licenses are delayed.

---

## 6. Phase C: Full Commercial Licensing Strategy

**Trigger:** Meaningful revenue scale (estimated $5,000+ MRR or 500+ paying subscribers).

Phase C establishes formal data partnerships with all major providers and reduces legal risk to near zero.

**Phase C actions:**

### 6.1 MLBAM (Major League Baseball Advanced Media)
- Negotiate a formal **MLBAM data redistribution agreement** covering:
  - MLB Stats API (player stats, schedules, game logs)
  - Baseball Savant / Statcast derivatives
  - Historical game data
- Engage through the **MLB Data Licensing portal** or through an MLB affiliate.
- Expected license structure: annual fee or revenue-share, with rate and volume terms.

### 6.2 FanGraphs
- Negotiate a **commercial data license** covering:
  - Steamer, ZiPS, Depth Charts, ATC, THE BAT, and THE BAT X projections.
  - Aggregate FanGraphs statistics (wRC+, FIP, xFIP, etc.).
- Clarify with FanGraphs whether the license covers sub-licensing to end users within a SaaS context or only internal computation.
- Identify and confirm rights clearance with individual projection system creators (Steamer, ZiPS, THE BAT) if required.

### 6.3 Yahoo Fantasy API (Commercial Partner)
- Complete Yahoo Fantasy Partner Program application and obtain **commercial API credentials**.
- Ensure API usage complies with Yahoo's rate limits and data usage terms for commercial applications.
- Explore whether a revenue-sharing or referral arrangement is available.

### 6.4 FantasyPros
- Negotiate a **FantasyPros data API partnership** for ECR and ADP data.
- Alternatively, evaluate Rotowire, CBS Sports, or other ECR data vendors as substitutes if FantasyPros commercial terms are prohibitive.

### 6.5 Retrosheet
- Obtain written commercial permission from Retrosheet for use of historical game log data in commercial backtesting.

### 6.6 Open-Meteo
- Continue under Apache 2.0 terms. Consider commercial support tier if usage volume warrants it.

---

## 7. Attribution Requirements

The following attributions should appear in the application's data sources or about page once commercially launched:

- **Weather data provided by Open-Meteo** (open-meteo.com) — Apache 2.0
- **Baseball statistics provided by MLB Stats API** — subject to MLBAM terms
- **Projection data from FanGraphs** — subject to FanGraphs license (Phase C)
- **Player news aggregated from MLB.com and Yahoo Sports**

Attribution for pybaseball, Marcel, and other open-source or public methodology components should appear in the application's open-source acknowledgments page.

---

## 8. Review and Maintenance

This document should be reviewed and updated:
- Whenever a new external data source is added to the application.
- When transitioning between commercial phases.
- Annually, or whenever a data provider updates their terms of service.
- Before any fundraising, acquisition, or significant investor due diligence process.

**Owner:** Heater Analytics LLC (Connor Hickey)
**Legal review recommended** before Phase B or C launch.

---

*Last updated: [EFFECTIVE_DATE]*
