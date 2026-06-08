# HEATER UI/UX Revamp — fantasypros.com look & feel (DEFERRED brief)

> **Status: DEFERRED / FUTURE SESSION.** Do NOT start this work until the gate below is met.
> Owner requested 2026-06-07. This is the LAST item, queued behind all current work.

## Gate (all must be true before starting)
1. The owner's items **#1 (verify live)**, **#9 (boot scheduler)**, **#10 (Trade Finder speed)**, and **#4 (per-team QA + league invite)** are **fully complete**.
2. All of the above + all prior work have **passed every GitHub Actions / CI-CD check locally and on GitHub**.
3. All of it is **merged to `master` locally and on GitHub** (origin/master == local master, clean tree).

Only when the owner starts a **new chat session** for the redesign do we begin.

## Goal
Revamp the **entire** HEATER UI/UX and overall application design to look and feel like
**fantasypros.com** (owner shared a screenshot of the FP home page). Apply it to **every page,
tab, feature, table, logo, branding element, color, theme, background, and fill** — the whole app,
not a single page.

## Hard requirements (owner's words, structured)
- **Sidebar: much smaller.** FP uses a thin, dark, icon-first left rail — a small icon with a tiny
  text label under it per nav item. HEATER's current sidebar is much wider/heavier. Shrink it to a
  compact icon rail; keep all current nav destinations reachable.
- **Less stuff on each page.** Far more whitespace, fewer competing elements, a clear focal content
  column, optional right rail for secondary cards. Reduce visual clutter everywhere.
- **Clone fantasypros.com's font.** Inspect the live site's CSS (`font-family` on `body` / headings)
  in this session and replicate it app-wide — **verify, do not guess**. Wire the web font(s) into the
  Streamlit theme/CSS.
- **Restyle everything to match FP:** page layouts, tab strips, tables, cards, buttons, the logo/
  branding treatment, color usage, themes, backgrounds, fills, spacing, border-radius, shadows.
- **Keep the current HEATER colors.** The owner permits **adding or adjusting** specific colors as
  needed to fit the overall vision — but the HEATER palette stays the base identity.

## Screenshot reference (FP home page, 2026-06-07)
- **Left rail (thin, dark navy):** FP roundel logo at top, then small icon+label nav items —
  Home, Draft, Playbook, Rankings, Research, DFS, and a "More" (•••). Compact, ~64–80px wide.
- **Top bar (white):** a rounded "✦ Coach" pill on the left; a wide centered search input
  ("Search … ctrl + K"); on the right a notifications bell (with a count badge), a profile avatar,
  and a yellow rounded **Upgrade** button.
- **Content (white, airy):** a large **featured-article card** — big rounded hero image with a
  "FEATURED" chip, bold large headline, "by … | date" byline, short teaser + "read more".
- **Right rail:** an "Upgrade to Premium" card (icon, heading, copy, yellow CTA), a sponsored video
  slot, and a "Featured Links" list.
- **Below:** a "Latest Articles" tab strip — Featured / NFL / MLB / NBA / NHL / DFS / Premium — with
  a "View All" link.
- **Overall feel:** clean humanist/geometric sans-serif, generous padding, rounded cards, soft
  separators, lots of white space, restrained color (mostly white + dark navy + a yellow accent).

## Where HEATER's UI lives (starting map for the redesign)
- `src/ui_shared.py` — `THEME` palette, `PAGE_ICONS` (inline SVGs), `format_stat`, shared CSS
  injectors (`inject_custom_css`), responsive `@media` rules. The canonical styling hub.
- `app.py` — splash/home, top-level layout, and (under MULTI_USER) `st.navigation(build_pages(...))`.
- `src/nav.py` — `PAGE_REGISTRY`, `build_pages`, the role-aware nav (drives the sidebar under MULTI_USER).
- `pages/*.py` — 13 in-season pages; each calls `inject_custom_css()` + renders its own tables/cards.
- `.streamlit/config.toml` — Streamlit theme (Heater palette). The header-bar hide is `@media (min-width:768px)` only (mobile keeps the menu toggle — see `test_mobile_nav.py`).

## Execution notes for the future session
- **Process:** brainstorm the design system first (skill: brainstorming), then implement
  (skill: frontend-design). Big visual change → screenshot before/after **per page** on the live
  app or a local `streamlit run` preview.
- **Constraints to respect:** no emoji in UI (icons are inline SVGs); Streamlit CSS needs
  `!important`; `T["ink"]` for text-on-accent; keep MULTI_USER flag-off byte-for-byte where the
  structural guards require it. Expect to **update theme/CSS guard tests** (e.g. mobile-nav,
  format-compliance) as styling changes — update them in lockstep, don't weaken them.
- **Scope is app-wide** — plan it as a multi-page sweep (a shared design-system pass in `ui_shared`
  first, then per-page cleanup), not a one-file change.
- **Deploy:** Railway single replica; deploys from `master`; confirm pushes with the owner (CLI novice).
