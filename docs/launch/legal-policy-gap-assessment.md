# Legal-policy gap assessment — public commercial launch (Phase 1, WP 1.3)

**Date:** 2026-06-26 · **Status:** internal audit + update punchlist for owner/counsel
**Audited:** `docs/legal/TERMS_OF_SERVICE.md`, `docs/legal/PRIVACY_POLICY.md`, `docs/legal/DATA_SOURCES.md` (all drafted 2026-04-11)

> **Not legal advice.** This audits existing drafts against the public-launch requirements and lists the edits needed. Counsel review before launch is still required (the "internal-now, certify-later" model).

---

## Summary

The April drafts are **strong, comprehensive foundations** — better than expected. The ToS covers service/disclaimers, accounts, billing, acceptable use (incl. an explicit anti-gambling/DFS clause), IP, warranty disclaimer, liability cap, termination, PA governing law, and class-action waiver. The Privacy Policy already anticipates Clerk, Stripe, **PostHog**, Yahoo/ESPN OAuth, a retention table, and access/deletion/portability/correction rights.

**They are not launch-ready as-is for one dominant reason: both predate Bubba (the AI assistant) and the replatform.** The required updates are targeted, not a rewrite.

**This pass delivers:** (1) this audit + punchlist; (2) a new **`docs/legal/AI_DISCLOSURE.md`** draft (the biggest gap); (3) a cross-link from the April `DATA_SOURCES.md` to the newer, more complete launch licensing docs. The in-place edits to the ToS/Privacy prose are listed below for deliberate application (with counsel), not surgically rewritten here.

---

## Per-document audit

### Terms of Service — covered vs. gaps
**Covered well:** service description + "entertainment/informational only" + no-affiliation disclaimer; 18+ eligibility; account security; subscription/auto-renew/price-change/cancellation/refund/tax; acceptable use incl. **anti-gambling/DFS** (matches the compliance finding); IP + limited license; warranty disclaimer; liability cap ($50/3-mo fees); termination; PA governing law + informal-resolution + class-action waiver; change process.

**Gaps / stale:**
| # | Gap | Priority |
|---|---|---|
| T1 | **No AI-assistant (Bubba) terms** — AI outputs are not advice, may be inaccurate/hallucinate; user is responsible for verifying; attachment-upload rules; no using Bubba to build a competing model | **P0** |
| T2 | **Pricing is stale** — lists Pro **$9.99** / Elite **$19.99**; the locked model is tiered **Free/Pro/Premium + usage metering** (leagues + AI). Reconcile to the metered model; final price points set in Phase 15 | P1 |
| T3 | **No free-trial terms** — the build has a 7-day trial; §3 must state trial length, auto-conversion to paid, and cancellation-before-bill | P1 |
| T4 | **Single-provider assumption** — §5.3 names only Yahoo OAuth; public product connects ESPN/CBS/Sleeper + manual import. Generalize | P1 |
| T5 | **Data-source disclaimer** — add that stats/projections derive from licensed third-party feeds + HEATER's own models, provided without warranty (ties to the licensing reality) | P2 |
| T6 | **Domain/contact** — `heaterfantasy.com` + `support@` must be a real registered domain/mailbox before launch (custom-domain is a Phase 15 / Clerk-prod item) | P2 |

### Privacy Policy — covered vs. gaps
**Covered well:** account/payment/league-OAuth/usage/comms collection; use purposes; "data we do NOT collect" (incl. no gambling data, no minors, no precise geo); third-party sub-processors (Stripe/Clerk/**PostHog**/Yahoo/ESPN/MLB); retention table; "we do not sell"; access/deletion/portability/correction/disconnect/opt-out rights; security; children's (18+); change process.

**Gaps / stale:**
| # | Gap | Priority |
|---|---|---|
| P1 | **No AI/Bubba data handling** — must disclose: chat conversations + uploaded attachments are collected; sent to the selected AI provider; the AI providers (Anthropic/OpenAI/Google/DeepSeek/xAI/OpenRouter) as sub-processors; retention + deletion of conversations; the no-train posture (verify per provider) | **P0** |
| P2 | **Sub-processor list incomplete** — add the 6 AI providers (§4) + CBS/Sleeper if added; confirm each has a signed DPA | **P0** |
| P3 | **CCPA/CPRA explicit section** — for public US launch, add the "right to opt-out of sale/sharing" + "we do not sell or share" notice + the CCPA response timelines (the generic §7 should call out CA rights) | P1 |
| P4 | **OAuth credential security** — state that connected-platform tokens are **encrypted at rest** (envelope encryption lands in Phase 10); current §8 is generic | P1 |
| P5 | **GDPR** — only needed if EU/UK users are accepted; if so add lawful-basis + SCCs language. Tie to the Phase-1 geography decision (US-first vs GDPR-ready) | P2 |

### DATA_SOURCES.md — status
**Superseded/extended** by the newer launch docs. The April doc's strategy (FanGraphs/FantasyPros/Yahoo need licenses; Marcel+MLB-API fallback; ~$5–15k/yr) **matches** the new findings, but it predates: ESPN injuries, the AI providers, the **commercial-feed vendor options** (MySportsFeeds/SportsDataIO/Sportradar), the **CBC v. MLBAM public-domain-stats precedent**, and the **player-headshot/logo likeness caveat**. A cross-link to `docs/launch/source_inventory.md` + `docs/launch/data-licensing-options.md` has been added to its top; treat those as the current source of truth.

---

## Privacy operations (WP 1.5) — DESIGNED, implementation sequenced after Postgres

The policies above **promise** export, deletion, disconnection, and OAuth-revocation rights. Those workflows must cover, per data class: account (Clerk/`api_state.db`), subscription (Stripe + store), league data, OAuth credentials, **AI conversations + attachments**, usage analytics (PostHog), and recommendation evidence.

**Implementation is intentionally deferred to after Phase 2 (Postgres) / Phase 3 (tenant model).** Building export/delete now against the current split SQLite (`draft_tool.db` + `api_state.db`) — both of which are being unified and replaced — would be throwaway work. The seam (a `privacy_requests` model + an export/delete service spanning the stores + the user-facing endpoints) is built in Phase 1's data model in the tenant schema, and the per-store logic is filled in as those stores land. This matches the program's "migrate through seams" principle. **Tracked in the evidence registry as `P1-PRIVACY-OPS` (deferred).**

---

## Prioritized punchlist (apply with counsel before launch)
- **P0 (blockers):** T1 (Bubba ToS terms), P1 (Bubba privacy disclosure), P2 (AI providers as sub-processors). → AI disclosure is drafted in `docs/legal/AI_DISCLOSURE.md`; fold its substance into the ToS + Privacy.
- **P1:** T2 pricing reconcile, T3 trial terms, T4 multi-provider, P3 CCPA/CPRA section, P4 OAuth-encryption statement.
- **P2:** T5 data-source disclaimer, T6 domain/contact, P5 GDPR (geography-dependent).
- **Cross-cutting:** replace every `[EFFECTIVE_DATE]` placeholder + confirm the legal entity name (`Heater Analytics LLC`) and registered domain at launch; sign DPAs with Clerk, Stripe, and each AI provider.
