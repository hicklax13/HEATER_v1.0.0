# Closer Monitor — ARCHIVED, revival spec ("Saves Finder")

**Date:** 2026-06-19
**Status:** ARCHIVED (built, verified, NOT merged). Lives on branch `feat/web-closer-monitor`, unmerged. Not in the live app / nav.
**Owner decision:** a static closer depth-chart grid is commoditized/low-value; the high-value version needs backend data that isn't available yet. Defer until it can be built as an actionable "Saves Finder."

## Why archived

The built page is a 30-team grid of closer cards (team accent, headshot, a job-security heat bar, stats). That's a **reference table every fantasy site has** — low differentiation. The *value* of closer tracking is entirely in **actionable, predictive, league-aware** signals, which require backend data the current `/api/closers` doesn't expose. Shipping the shell now is low ROI.

## The high-value version (what makes it worth reviving)

1. **"Next man up" handcuff intelligence** — for each shaky closer, who inherits the saves, so the user stashes him *before* the saves materialize (the single biggest SV-category waiver edge). The card already renders `handcuffs[]` (currently empty from the API).
2. **"Saves on YOUR wire"** — cross-reference closers + handcuffs against the user's *league* free agents (MINE/FREE), surfacing acquirable saves: "CLE's job is shaky and the handcuff is a free agent in your league." Personalized + actionable.
3. **Job-security *trend* / volatility** — not a static %, but a directional signal (recent blown saves, velo drop, gmLI decline, manager-comment news) → the buy/sell call.

## Backend data needed (CEO track) to revive

`/api/closers` is DB-backed but currently MINIMAL. To support the above it must:
- **Populate `handcuffs[]`** — the engine has setup men (`src/closer_monitor.build_depth_data_from_db`, the Streamlit page renders SETUP names); the API returns `[]`. Expose them.
- **Real `confidence` levels** — currently uniformly `"Shaky"`. Map to the engine's real job-security signal + emit a numeric `job_security` (0–100) so the frontend stops deriving it from a label.
- **A `trend` signal** — recent-form / blown-save / gmLI delta per closer.
- **Stats** — proj SV / ERA / WHIP per closer (the card renders them; the API omits them).
- **League-FA cross-reference** (for "saves on your wire") — needs Yahoo roster + FA data per user (M3/M4 territory).

## What's already built (the foundation, on the branch)

- `web/src/app/closers/page.tsx` — `/closers` page, `usePageData` four-state, grid sorted by security desc.
- `web/src/components/closers/CloserCard.tsx` — team-accent card: headshot (`PlayerAvatar`), name (`PlayerLink`), confidence badge, job-security heat bar (`heatColor`), + stats/handcuffs rows that render only when present (graceful degradation).
- `web/src/lib/closers-data.ts` — typed mock + `fetchClosers` live wire to `/api/closers` (flag-gated, mock fallback) + `CONFIDENCE_SECURITY` map.
- `web/src/lib/api/{types,adapters}.ts` — `ApiClosersResponse` alias + `apiClosersToData` (snake→camel, null-safe `toPlayerRef`).
- `web/src/components/chrome/TopBar.tsx` — "Closers" nav item.
- `web/scripts/audit-mock-ids.mjs` — extended to cover `streaming-data.ts` + `closers-data.ts` and the `pr(`/`ref(` mock shapes (this **caught 2 wrong closer headshots** — Mason Miller, Pete Fairbanks — before archive; both fixed). Worth landing on master independently when convenient.

Verified before archive: audit 0 mismatches; tsc + lint + build green; live render (21 real closers from `/api/closers`, headshots, zero console errors) + clean mock fallback.

## How to revive

`git checkout feat/web-closer-monitor` (or cherry-pick). The page + live wire are ready; the work is: (1) CEO populates the backend fields above, (2) redesign the card/page around the 3 actionable features (next-man-up, saves-on-your-wire, trend), (3) re-verify + merge + add to nav.

## Note

The **audit-script extension** (covering streaming + closers + the `pr(`/`ref(` shapes) is genuinely valuable independent of this page — it closed a real gap (streaming-data.ts ids were never audited). Consider landing just that change on master separately.
