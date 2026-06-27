# HEATER data-source inventory & commercial-rights assessment

**Date:** 2026-06-26 · **Phase:** 1 (Legal & data rights), work package 1.1 · **Status:** internal draft for owner decisions
**Owner-gated:** licensing/spend/legal sign-off are Connor's calls; this document is the input.

> **Not legal advice.** This is an engineering+research assessment of publicly stated terms to drive owner decisions. Every "must license / replace / drop" call must be confirmed by qualified counsel before launch, and provider terms change — re-verify at sign-off.

---

## ★ Headline finding (read this first)

**HEATER's core in-season data backbone is, as built, commercial-use-prohibited.** The three sources that power its highest-value features each forbid commercial use without a paid license:

- **MLB Stats API / MLBAM** (live player stats, schedules, news, Statcast/Savant) — free only for *individual, non-commercial, non-bulk* use; commercial use needs **prior written authorization from MLBAM**.
- **FanGraphs** (Steamer/ZiPS/Depth Charts projections, Stuff+, park factors) — ToS prohibits using *any portion* for commercial purposes; FanGraphs licenses this data from its own providers and is actively hostile to scraping.
- **FantasyPros** (ECR consensus) — ToS prohibits commercial use; a separate commercial API agreement would be required.

A paid, public MLB fantasy product **cannot ship on scraped MLBAM/FanGraphs/FantasyPros data.** This is the gating constraint for the entire commercial launch — it is a business-development + budget decision, not a code fix. The good news: there are concrete paths (license, replace with an official distributor, or compute in-house from permissively-licensed data), detailed below.

---

## Per-source inventory

Verdict legend: 🟢 OK commercially (maybe with attribution) · 🟡 needs a paid plan / approval (low-to-moderate) · 🔴 commercial-prohibited as used — must license, replace, or drop.

| Source | What HEATER uses | How acquired | Displayed/redistributed? | Trains/calibrates a model? | Commercial-rights status | Verdict |
|---|---|---|---|---|---|---|
| **MLB Stats API / MLBAM** | Players, live stats, schedules, news, umpires, game logs | `MLB-StatsAPI` (statsapi.mlb.com) | Yes (stats shown to users) | Yes (drives projections, sims) | Free only for individual/non-commercial/non-bulk; commercial needs MLBAM written authorization | 🔴 |
| **Baseball Savant / Statcast** | xwOBA, barrel%, Stuff+, sprint speed, catcher framing | `pybaseball` scrape of Savant (MLBAM) | Yes | Yes | MLBAM property — same restriction as above | 🔴 |
| **FanGraphs** | Steamer/ZiPS/Depth Charts projections, Stuff+, park factors | `pybaseball` + FanGraphs JSON API (scrape) | Yes (projections shown) | Yes (core valuation) | ToS prohibits any commercial use; FanGraphs licenses this data itself + opposes scraping | 🔴 |
| **FantasyPros** | ECR consensus rankings, ADP | Scrape / multi-source aggregation | Yes | Yes (draft/ECR) | ToS prohibits commercial use; commercial API agreement needed | 🔴 |
| **ESPN** | Injury status (`espn_injuries`) | Unofficial/undocumented endpoint | Yes (injury badges) | Indirectly | No license; unofficial endpoint with no commercial grant | 🔴 |
| **Yahoo Fantasy Sports API** | The user's own league (rosters, standings, matchups, transactions) | Per-user OAuth (user authorizes their data) | Yes (to that user) | No | Commercial use requires **prior written Yahoo approval + an approved app + attribution** ("Fantasy data provided by Yahoo Fantasy") | 🟡 |
| **Open-Meteo** | Stadium weather (30 parks, dome detection) | Free hosted API | Indirectly (factors) | Indirectly | Free tier is **non-commercial only**; commercial use needs a paid plan or self-hosting the OSS server | 🟡 |
| **Retrosheet** | Historical game data | Download | Possibly | Possibly (backtests) | **Commercial use ALLOWED** with the required attribution notice | 🟢 (attribution) |
| **Clerk** | Auth / identity | SaaS (paid) | No | No | Commercial use is the product; needs a signed DPA (GDPR/CCPA) | 🟢 (DPA) |
| **Stripe** | Billing / subscriptions | SaaS (paid) | No | No | Commercial use is the product; DPA + PCI handled by Stripe | 🟢 (DPA) |
| **AI providers** (Anthropic, OpenAI, Google, DeepSeek, xAI, OpenRouter) | Bubba chat | API (paid / BYO-key) | Output to user | No (verify no-train defaults) | Commercial use allowed under API terms; verify each provider's data-use/no-train + sign DPAs; user chat = personal data | 🟢 (verify + DPA) |
| **pybaseball** (library) | Wrapper that scrapes FanGraphs/Savant/Statcast | pip dependency | — | — | The *tool* is open-source (fine); the *data* it fetches inherits MLBAM/FanGraphs restrictions | n/a (data risk is upstream) |

---

## Mitigation paths for the 🔴 blockers

1. **Official MLB data (replaces MLBAM scraping + much of Savant/FanGraphs at once).** The standard route for a commercial MLB fantasy product is a license from an **official MLB data distributor** — Sportradar or Stats Perform — or a direct MLBAM/MLB license. This provides live stats, schedules, and play-by-play with a commercial grant. *Cost is the main unknown and is the single biggest launch budget line; needs BD outreach + counsel.*
2. **Projections — compute in-house instead of licensing FanGraphs.** HEATER already ships a **Marcel projection fallback** (`src/marcel_bootstrap.py`) — Marcel is a public-domain method. Computing HEATER's own projections from *licensed* stat data removes the FanGraphs dependency for the core valuation. Stuff+/park-factor signals would either be dropped or sourced from the licensed feed. (Licensing Steamer/ATC/THE BAT directly from their authors is an alternative.)
3. **ECR — drop or license.** Either remove FantasyPros ECR or sign their commercial API agreement. HEATER can compute an internal consensus/value from licensed inputs.
4. **Injuries — replace ESPN.** Source injury status from the licensed feed (official distributor) or a licensed injury provider; retire the unofficial ESPN endpoint.
5. **Yahoo — apply for commercial approval.** Submit HEATER's commercial use case to Yahoo's API program and add the required "Fantasy data provided by Yahoo Fantasy" attribution. (Note: write scope remains separately unavailable — see [[reference_yahoo_integration_state]].) The other connectors (ESPN/CBS/Sleeper, Phase 5) have their own terms to clear.
6. **Weather — buy the Open-Meteo commercial plan** (Standard tier is inexpensive) or self-host the open-source server.

**Net:** the cheapest viable path to a legal product is **one official live-data license (distributor) + in-house projections + Retrosheet for history + a small Open-Meteo plan + Yahoo commercial approval**, dropping FanGraphs/FantasyPros/ESPN scraping. The official-data license cost is the decision that unblocks everything else.

---

## Compliance check (privacy / regulatory) — for the public launch

### Summary
**Proceed with conditions.** Privacy obligations are standard and addressable for a US-first launch; the binding constraint is the data-licensing above, not privacy law.

### Applicable regulations
| Regulation | Relevance | Key requirements |
|---|---|---|
| **CCPA / CPRA** (California) | Public US signups will include CA residents once thresholds are met | Privacy notice at collection; rights to know/delete/correct/opt-out; service-provider contracts (DPAs); 45-day response |
| **Other US state laws** (VA/CO/CT/UT/TX etc.) | Similar consumer-privacy regimes expanding | Mirror CCPA: notice, access, deletion, opt-out |
| **GDPR / UK GDPR** | Applies if you accept EU/UK users | Lawful basis per processing activity; DSAR within 30 days; 72-hr breach notice; SCCs for transfers; Art. 30 records |
| **Card data (PCI)** | Payments | Use Stripe Checkout/Elements so card data never touches HEATER servers (Stripe carries PCI scope) |
| **Fantasy/contest law** | It's advisory, not paid contests/DFS | Keep it advisory-only (no entry fees/prizes) to stay clear of DFS/gambling regimes; add a no-guarantee disclaimer |

### Risk areas
| Risk | Severity | Mitigation |
|---|---|---|
| Commercial use of MLBAM/FanGraphs/FantasyPros without license | **High** | License official feed + drop scraped sources (above) before any public charge |
| ESPN unofficial endpoint | High | Replace with a licensed injury source |
| Per-user Yahoo OAuth credentials (sensitive) at scale | High | Envelope-encrypt + per-user keys (Phase 10); Yahoo commercial approval |
| AI chat + uploaded attachments = personal data | Medium | Per-provider no-train config + DPAs; conversation retention/deletion controls (Phase 1 privacy ops) |
| No public ToS/Privacy/AUP yet | Medium | Draft in WP 1.3; counsel review before launch |
| EU users without GDPR readiness | Medium | Either geo-gate to US at first launch, or stand up GDPR ops (SCCs, DSAR, DPAs) |

### Recommended actions (priority order)
1. **Decide the official-data-license path** (MLBAM direct vs Sportradar/Stats Perform) and start BD outreach — this is the launch's critical path and cost driver.
2. Re-architect data ingestion to the licensed feed; retire FanGraphs/FantasyPros/ESPN scraping; move projections fully in-house (Marcel/own model).
3. Draft public policies (ToS, Privacy, AUP, AI disclosure, no-guarantee disclaimer) — WP 1.3.
4. Add Retrosheet + Yahoo attribution notices; buy the Open-Meteo commercial plan; apply for Yahoo commercial approval.
5. Sign DPAs with Clerk, Stripe, and each AI provider; verify each AI provider's no-train-by-default posture.
6. Decide initial geography (US-only first vs GDPR-ready) — this sizes the privacy work.

### Approvals needed (owner)
| Approver | Why | Status |
|---|---|---|
| Connor (owner) | Data-license budget + path; geography; which sources to drop | Pending |
| Qualified counsel | Confirm all rights determinations + review public policies + licenses | Pending (deferred per "internal-now, certify-later") |
| MLBAM / official distributor | Commercial data license | Not started |
| Yahoo | Commercial API approval | Not started |

### Further review recommended
Outside counsel should confirm the MLBAM/FanGraphs/FantasyPros/ESPN determinations (IP + contract), the DFS/gambling-law boundary for an advisory product, and the privacy posture for the chosen geography before the public flip.

---

## Sources (verify current terms at sign-off)
- MLB Stats API / MLBAM terms: <https://statsapi.mlb.com/> · copyright notice referenced at `gdx.mlb.com/components/copyright.txt`
- FanGraphs Terms of Service: <https://www.fangraphs.com/about/terms-of-service> · <https://blogs.fangraphs.com/the-state-of-fangraphs-2026/>
- FantasyPros Terms of Use: <https://www.fantasypros.com/about/legal/> · API: <https://api.fantasypros.com/v2/docs>
- Retrosheet license: <https://retrosheet.org/> · <https://www.retrosheet.org/game.htm>
- Open-Meteo terms/pricing: <https://open-meteo.com/en/terms> · <https://open-meteo.com/en/pricing> · <https://open-meteo.com/en/licence>
- Yahoo Developer API Terms: <https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html> · <https://developer.yahoo.com/fantasysports/guide/>
