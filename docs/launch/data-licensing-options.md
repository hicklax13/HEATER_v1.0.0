# MLB data licensing options — for the commercial-launch decision

**Date:** 2026-06-26 · **Phase:** 1 (Legal & data rights) · **Status:** research brief for the owner's data-license decision
**Companion to:** [`source_inventory.md`](source_inventory.md)

> **Not legal advice.** Publicly stated terms + landscape research to inform Connor's budget/BD decision. Confirm every determination with counsel and get live quotes before committing.

---

## ★ The key nuance that softens the headline finding

The source inventory's "core data is commercial-prohibited" finding is about the **specific feeds** HEATER scrapes (MLBAM's API, FanGraphs, FantasyPros) — their *terms of service* forbid commercial use of *their* access. It is **not** that the underlying numbers are owned.

**Landmark precedent — CBC Distribution v. MLBAM (8th Cir., 2007):** a court held that a fantasy company's use of MLB players' **names and statistics** is protected and does **not** require an MLB/MLBPA license — baseball stats and player names are publicly available facts. So the *facts* HEATER needs (player X hit Y HRs) are usable; what HEATER can't do is **scrape a competitor's ToS-protected feed** to get them, or use **player photos/team logos** (those are separate likeness/trademark rights — see the caveat below).

**Translation:** the fix isn't "license the data" so much as **"buy the same facts from a feed that grants commercial use,"** then compute everything else in-house. That is available at indie-friendly prices.

---

## Vendor landscape (the commercial-feed options)

Indie-fit legend: 🟢 solo-operator friendly · 🟡 mid-market (sales quote) · 🔴 enterprise.

| Vendor | What it gives | Commercial terms | Rough cost signal | Indie fit |
|---|---|---|---|---|
| **MySportsFeeds** | MLB schedules, scores, boxscores, play-by-play, lineups, **injuries**, DFS, some projections | Explicit commercial licenses; "lots of options based on your budget"; free for hobbyists | Flexible; "scales with update frequency" — near-real-time costs more. Likely the **cheapest licensed path** | 🟢 |
| **API-Sports** (api-sports.io) | MLB scores/stats/standings | Commercial tiers | From **~$10–$50/mo**; free 100 req/day | 🟢 (verify data depth) |
| **SportsDataIO / FantasyData** | Deep fantasy MLB: stats, projections, **injuries**, DFS, odds | "Discovery Lab" dev tier is delayed/next-day ($99–$149/mo) — **not enough for live**; *production* commercial = sales quote; free integration-test key | Production: quote-based, low-to-mid **$100s–$1,000s/mo** range typical for fantasy | 🟡 |
| **Rolling Insights / DataFeeds** | Multi-sport incl. MLB | Commercial | From **~$100/mo**, free trial | 🟢/🟡 |
| **Sportradar** | The official premium feed (powers DraftKings, ESPN Fantasy, Yahoo); advanced metrics | 30-day full free trial → production (customers only) | ~**$499/mo Starter** (100k calls) → custom enterprise | 🟡/🔴 |
| **Stats Perform** | Enterprise official feed | Enterprise sales | Big-operator pricing | 🔴 |
| **Highlightly / TheSportsDB** | Lighter/startup feeds | Varies | Cheap/free | 🟢 (shallow data) |

**Statcast-level metrics caveat:** xwOBA, barrel%, Stuff+, sprint speed (HEATER's `statcast_archive`) are **MLBAM/Savant-specific**. Mid-market vendors carry box-score + standard stats well; deep Statcast/advanced metrics are a premium tier or simply unavailable. Expect to either pay up (Sportradar/SportsDataIO advanced) or **scope those features down** for v1.

---

## Recommended solo-operator path

A legal, affordable stack for the public launch:

1. **One commercial MLB feed** for live stats + schedules + injuries → start with **MySportsFeeds** (cheapest licensed) and **SportsDataIO production** + **Sportradar trial** to compare data depth. Target spend: **low hundreds of $/month**, not an enterprise deal.
2. **Projections in-house** — HEATER's Marcel fallback (`src/marcel_bootstrap.py`) computed on the licensed stats. **$0**, removes FanGraphs.
3. **History** — Retrosheet, free **with the attribution notice**.
4. **Weather** — Open-Meteo commercial plan (small) or self-host.
5. **Injuries** — from the licensed feed (retire ESPN scraping).
6. **Yahoo** — apply for commercial API approval + add the "Fantasy data provided by Yahoo Fantasy" attribution.
7. **Drop** FanGraphs + FantasyPros scraping; compute an internal consensus/value from the licensed inputs.
8. **Statcast features** — decide: pay for an advanced tier, or ship v1 without xwOBA/Stuff+ and add later.

**Net:** the launch is gated by a **low-hundreds-$/month feed subscription + a few weeks of re-wiring ingestion to that feed**, not a $50k enterprise license. That's a very different (and achievable) picture than the raw inventory implied.

---

## ⚠️ Separate caveat: player photos + team logos

The CBC precedent covers **names + stats**, not imagery. HEATER's frontend uses **real player headshots and team identity/logos**. Those are **MLB/MLBPA likeness + trademark** rights and are **not** covered by buying a stats feed. For a commercial product: license them (via the feed vendor's media rights or MLB), or **replace headshots with generic avatars and logos with neutral team marks**. Cheapest path: drop real headshots/logos for v1. Flag for counsel.

---

## Recommended next actions (owner)

1. **Start the free trials** (Sportradar 30-day; SportsDataIO integration key; MySportsFeeds) and have me verify each covers HEATER's required fields (live box score, schedules, injuries, and which advanced metrics).
2. **Decide the Statcast-feature scope** for v1 (pay for advanced tier vs. ship without).
3. **Decide the imagery scope** (license vs. generic avatars/logos for v1).
4. Get **live commercial quotes** from the top 2 once data-depth is confirmed.
5. Counsel confirms the CBC-precedent reliance + reviews the chosen vendor's commercial agreement before launch.

---

## Sources (verify current terms/pricing directly)
- MySportsFeeds pricing/commercial: <https://www.mysportsfeeds.com/feed-pricing/>
- SportsDataIO MLB API / pricing: <https://sportsdata.io/mlb-api> · <https://sportsdata.io/developers> · Discovery Lab tiers via <https://plans.apis.io/plans/sportsdataio/sportsdataio-plans-pricing/>
- Sportradar MLB / trial: <https://sportradar.com/media-tech/data-content/sports-data-api/> · <https://developer.sportradar.com/getting-started/docs/get-started>
- API-Sports / Rolling Insights: <https://rolling-insights.com/datafeeds/price-plans/>
- CBC v. MLBAM background + MLBAM rights: <https://en.wikipedia.org/wiki/MLB_Advanced_Media> · HN discussion <https://news.ycombinator.com/item?id=1791588>
- MLB licensed production/distribution application: <https://www.mlb.com/forms/mlb-online-retailer-policy-application>
