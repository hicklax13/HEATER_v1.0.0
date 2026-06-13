# Page 04 — Closer Monitor — Test-User Report

---

## 1. Page Purpose & First Impression

**Purpose:** Show a 30-team bullpen depth chart so the manager can quickly spot which teams have a defined closer, who the setup heir is, and how secure each job is — critical for the Saves category in a H2H league.

**First impression (novice lens):** The page loads a grid of instrument cards, each representing one team's closer situation. Five columns, four rows of cards (for the 21 teams shown). The header reads "BULLPEN / FIG.03 — SAVE DEPTH CHART" over a large "Closer Monitor." wordmark in the Combustion design system — clean and on-brand.

The novice's first questions:
- "It says 30-team but I only see 21 — where are the other 9?" — The caption says `Showing 21 teams with closer data.` but gives no further explanation.
- "What does `51% JOB` mean exactly? Is that good or bad?"
- "The SV number on the card — is that what the guy *has* this year, or what he's projected to get?"
- "Why does every card say `SETUP · —`? Are there no setup men on any team?"
- "What is `2026 ACTUAL`? Does that mean the other number isn't actual?"

Within 10 seconds the page is useful in broad strokes but has multiple labels that require inference, not reading.

---

## 2. Methodology

**Source code reviewed:**
- `pages/3_Closer_Monitor.py` — full page logic
- `src/closer_monitor.py` — `build_depth_data_from_db()`, `build_closer_grid()`, `compute_job_security()`, `get_security_color()`, `compute_committee_risk()`, `compute_skill_decay()`, helper functions
- `src/ui_shared.py` — `render_page_header()`, `render_reco_banner()`, `build_heatbar_html()`, `_headshot_img_html()`, `team_color()`, `team_logo_url()`, `text_on()`, `page_timer_footer()`

**Live DB queries (read-only, network-blocked):**
- `players.depth_chart_role` — distribution of all 830 role-assigned players
- `build_depth_data_from_db()` reconstructed and traced step-by-step
- Full grid reconstruction with actual job security scores and projected SVs
- `season_stats` 2026 save leaders (actual)
- Missing-team investigation (9 missing teams + their SV leaders)
- Team code normalization mismatch investigation (AZ → ARI)
- `projections` table for blended SV projections
- `refresh_log` for depth-chart source status

---

## 3. Feature & Control Inventory

| Control | Type | What It Does | Tested? |
|---------|------|-------------|---------|
| Page header (eyebrow + wordmark) | Static HTML | "BULLPEN / FIG.03 — SAVE DEPTH CHART" + "Closer Monitor." title | Yes (source) |
| Reco banner `30-team closer depth chart` | Static markdown banner | Teaser line, no expander (empty `expanded_html`) | Yes (source + code trace) |
| Matchup ticker | Conditional bar | Shows current week matchup — only renders if `yahoo_connected`; Yahoo is offline so does not render | Yes (code trace) |
| `st.info` / `st.warning` banner | Conditional | Shows data source (role data vs estimate) and quality | Yes (DB) |
| `st.caption(...)` | Caption | "Showing 21 teams with closer data." | Yes (DB confirmed: 21) |
| 5-column grid layout | `st.columns(5)` | Renders team cards in rows of 5 | Yes |
| **Team card** (per team × 21) | `st.markdown(unsafe_allow_html)` HTML block | Full card: team badge, logo watermark, headshot, name, heatbar, % JOB, SV/ERA/WHIP stats, "2026 ACTUAL" line, Recent Form label, gmLI row, SETUP row | Yes (real outputs captured) |
| Heatbar | `build_heatbar_html()` | Orange gradient (win=True) or steel (win=False) fill strip | Yes (code trace) |
| "2026 ACTUAL" stats line | Conditional HTML within card | Shows actual saves, ERA, WHIP in green; only appears when player has `sv > 0` in `season_stats` | Yes (DB confirmed) |
| Recent Form (HOT/COLD) label | Conditional HTML | Only appears if `get_matchup_context().get_player_form()` returns trend != "neutral" | Yes (code trace; network blocked) |
| gmLI trust indicator | Conditional HTML | Only appears if `st.session_state["closer_gmli_data"]` is populated; it is not | Yes (DB: no gmli_data table) |
| SETUP row | Static HTML within card | Comma-separated setup arm names; shows "—" if none | Yes (real: 19 of 21 cards show "—") |
| Page timer footer | `page_timer_footer()` | Renders subtle load-time footer | Yes (source) |
| Feedback widget | `render_feedback_widget()` | Feedback popover (MULTI_USER-gated) | Yes (source) |

---

## 4. Feature-by-Feature Test Log With Real Outputs

### 4.1 Page Header

**Observed:** Eyebrow `BULLPEN` + crumb `FIG.03 — SAVE DEPTH CHART`, title `Closer Monitor.` (orange period). Consistent with Combustion design system. FIG number `03` is zero-padded correctly (no inconsistency here unlike the Pitcher Streaming `FIG.4` note in shared context).

### 4.2 Info Banner

When `depth_chart_role` data exists (it does — 23 players with role=`closer`), the page shows:

> "Closer depth charts are populated from the bullpen-role depth chart data (FanGraphs Roster Resource, with an MLB Stats API fallback) loaded at app launch. Run the app bootstrap to refresh."

`refresh_log` for `depth_charts`: `status=no_data, ts=2026-06-07T20:23:35`. The data is **6 days stale** with no indication of age anywhere on the page or in this banner.

### 4.3 Caption

`st.caption("Showing 21 teams with closer data.")`

This is accurate per the DB reconstruction. With 23 closer-role players across 21 distinct team codes (MIL and TOR each have 2 closers; the secondary becomes the setup arm per engine logic), 21 cards render.

### 4.4 21-of-30 Gap Investigation

**Root cause confirmed:** The DB has `depth_chart_role='closer'` for exactly 21 team codes. The `depth_charts` bootstrap phase returned `status=no_data` (Roster Resource scrape returned nothing, as documented as a "Known Design Choice"). The MLB Stats API fallback **only populated 21 of 30 teams** with a `closer` role.

**The 9 missing teams and their actual 2026 SV leaders** (reconstructed from `season_stats`):
- **ATH**: Hogan Harris (5 SV, no closer role), Mark Leiter Jr. (4 SV)
- **BAL**: Ryan Helsley (7 SV — the former Cardinals closer, role=NULL)
- **CHC**: Daniel Palencia (3 SV, role=bullpen)
- **CIN**: Emilio Pagán (6 SV, role=NULL)
- **COL**: Victor Vodnik (4 SV, role=bullpen)
- **LAA**: Jordan Romano (4 SV, role=NULL)
- **MIN**: Yoendrys Gómez (4 SV, role=bullpen)
- **SF**: Caleb Kilian (4 SV, role=bullpen)
- **WSH**: Gus Varland (5 SV, role=bullpen)

Several of these teams *do* have active closers (BAL/Helsley, CIN/Pagán, LAA/Romano, WSH/Varland) but with `depth_chart_role=NULL` or `='bullpen'` — so the engine misses them entirely. The SV-based fallback heuristic in the page would catch these teams (it would show Helsley, Pagán, Romano, Varland etc.) but it only kicks in when `depth_data` is completely empty. Since 21 teams populated successfully, the fallback never runs — leaving 9 teams silently dark.

**The page shows no explanation for why 9 teams are missing.** A novice user legitimately wonders: "Does BAL not have a closer? Or did the data just fail?"

### 4.5 Team Card — Full Real Outputs (Reconstructed)

Complete grid as the page renders:

| Team | Closer | % JOB | Proj SV | ERA | WHIP | Actual SV | Setup |
|------|--------|--------|---------|-----|------|-----------|-------|
| ARI | Paul Sewald | 51% | **0** | **n/a** | **n/a** | 15 | — |
| ATL | Raisel Iglesias | 63% | 10 | 3.25 | 1.19 | 13 | — |
| BOS | Aroldis Chapman | 64% | 11 | 2.53 | 1.20 | 13 | — |
| CLE | Cade Smith | 62% | 7 | 2.86 | 1.06 | 21 | — |
| CWS | Seranthony Domínguez | 55% | 7 | 3.91 | 1.38 | 2 | — |
| DET | Kenley Jansen | 58% | 8 | 4.53 | 1.34 | 7 | — |
| HOU | Bryan King | 48% | 1 | 3.50 | 1.27 | 6 | — |
| KC | Lucas Erceg | 56% | 5 | 4.47 | 1.53 | 12 | — |
| LAD | Tanner Scott | 51% | 3 | 3.37 | 1.13 | 6 | — |
| MIA | Pete Fairbanks | 59% | 9 | 4.56 | 1.33 | 7 | — |
| MIL | Trevor Megill | 48% | 7 | 3.92 | 1.22 | 8 | Abner Uribe |
| NYM | Devin Williams | 60% | 9 | 4.01 | 1.30 | 8 | — |
| NYY | David Bednar | 62% | 9 | 3.63 | 1.26 | 13 | — |
| PHI | Jhoan Duran | 68% | 13 | 2.43 | 1.05 | 16 | — |
| PIT | Gregory Soto | 50% | 2 | 3.58 | 1.22 | 8 | — |
| SD | Mason Miller | 60% | 6 | 1.99 | 0.94 | 18 | — |
| SEA | Andrés Muñoz | 62% | 10 | 3.55 | 1.17 | 10 | — |
| STL | Riley O'Brien | 61% | 7 | 3.69 | 1.30 | 17 | — |
| TB | Bryan Baker | 58% | 5 | 3.55 | 1.22 | 18 | — |
| TEX | Jacob Latz | 53% | 4 | 3.65 | 1.17 | 9 | — |
| TOR | Louis Varland | 44% | 3 | 2.55 | 1.13 | 11 | Jeff Hoffman |

**"2026 ACTUAL" line appears on all 21 cards** (all have `sv > 0` in `season_stats`). Shows actual SV count, ERA, WHIP in green monospace text.

**SV column confusion:** The large orange SV figure (top of the stats trio) shows the **blended projection** (from `projections` table, `system='blended'`), while the green "2026 ACTUAL" line below shows **real YTD saves**. This creates two save numbers with zero label differentiation: a novice cannot tell which is projected vs actual just from the layout.

### 4.6 Critical Bug: ARI / Paul Sewald — Missing Stats

**Root cause confirmed:** `players.team` stores `'AZ'` for Paul Sewald. The page normalizes the depth-data key `'AZ' → 'ARI'`. But `build_closer_grid` then tries to match the player in the pool using `player_pool["team"] == "ARI"` — Sewald's pool row has `team='AZ'`. The match fails. Result: Sewald's ARI card shows:
- SV = `—` (0 projected, renders blank)
- ERA = `—`
- WHIP = `—`

The "2026 ACTUAL" line *does* appear for Sewald (15 SV, 3.47 ERA, 0.73 WHIP) because `_load_actual_sv_stats()` matches by **name only** against `season_stats`. So the ARI card shows `—` / `—` / `—` in the main stat trio but then "2026 ACTUAL · 15 SV · 3.47 ERA · 0.73 WHIP" in the green row below — a confusing contradiction.

This AZ→ARI normalization mismatch affects only Sewald in the current DB state. All other teams use the same abbreviation in `players.team` and the depth key.

### 4.7 % JOB Score — What It Means

The `job_security` formula: `0.6 × closer_confidence + 0.4 × min(1.0, proj_sv / 30.0)`.

`closer_confidence` starts at 0.75, adjusts ±0.20 based on saves, −0.15 if a second closer-role player exists. Because the formula relies on blended **projected** SV (not actual), it produces systematically low scores for closers whose actual saves far exceed preseason projections (e.g., Cade Smith: 21 actual SV but only 7 projected → 62% JOB; Mason Miller: 18 actual SV but only 6 projected → 60% JOB).

Range seen: 44% (TOR/Varland) to 68% (PHI/Duran). All scores are jammed into a narrow 44–68% band — the heatbar is essentially always orange (security ≥ 0.5 → orange gradient), never steel, making the bar visually meaningless as a "warning" signal.

### 4.8 gmLI Feature — Dead

`st.session_state.get("closer_gmli_data", {})` always returns `{}` — nothing populates this key. The `gmli_data` table doesn't exist in the DB. All gmLI HTML renders as the HTML comment `<!-- no gmli data -->`. The feature is architecturally wired but completely inert. A user inspecting the source would see the gmLI logic but never see it in action.

### 4.9 Recent Form Feature — Network-Dependent, Likely Inert

`get_matchup_context().get_player_form(mlb_id)` is called inline per card. With Yahoo offline and the network guard in place, this silently returns `{}` (trend="neutral") for all cards. Even in normal operation, this call is made inside a per-card loop — 21 separate calls to `get_matchup_context()` on every page load — which is a singleton but still potentially heavy.

### 4.10 SETUP Row — Nearly Always Shows "—"

19 of 21 cards show `SETUP · —`. This is because `depth_chart_role='setup'` has **zero rows in the DB** — only `closer`, `bullpen`, and `starter` roles are populated. The MLB Stats API fallback that tagged closers did not tag any setup arms with the `'setup'` role. Only MIL and TOR show a setup arm, and only because they had *two* players tagged as `closer` — the secondary closer gets demoted to setup by the engine logic.

This makes the SETUP row a dead feature for 19/21 teams. A novice sees `SETUP · —` on every card and learns nothing about who gets the next save opportunity when the closer blows it.

### 4.11 Name Truncation

The card uses `white-space:nowrap;overflow:hidden;text-overflow:ellipsis` on the closer name. At a 5-column layout on a typical 1400px viewport, each card is roughly 260px wide. With a 34px headshot and 5px margin taking ~39px, and the team badge taking ~50px in the row above, the name field has roughly 200px. The orchestrator observed `"Raisel Igl"` and `"Paul Sew"` truncations in the live render. The longest names are:
- `Seranthony Domínguez` (20 chars) — very likely truncated
- `Raisel Iglesias` (15 chars) — confirmed truncated
- `Aroldis Chapman` (15 chars) — likely truncated at small viewport widths

### 4.12 Character Encoding

The name `"Seranthony Domínguez"` and `"Andrés Muñoz"` and `"Emilio Pagán"` all have diacritics (accent characters). In the DB query output they rendered as `"Seranthony Dom�guez"` (question mark substitution), indicating encoding issues between the database storage and Python string handling in the audit script — though these names may render correctly in the browser via Streamlit's UTF-8 pipeline.

### 4.13 Projected SV vs Actual SV — Two Numbers, No Labels

The SV stat column on every card shows the **blended preseason projection** (from the `projections` table). In mid-season, these projections are already stale and often lower than the closer's actual pace. For example:
- Cade Smith: projected SV = 7, actual SV = 21
- Mason Miller: projected SV = 6, actual SV = 18
- Bryan Baker: projected SV = 5, actual SV = 18
- Riley O'Brien: projected SV = 7, actual SV = 17

The card labels this column "SV" with no "projected" qualifier. The "2026 ACTUAL" row shows the real numbers in green below. A novice sees two SV-related values (the big orange "7" and then "2026 ACTUAL · 21 SV") with no clear explanation of the difference.

---

## 5. Errors, Issues & Difficulties

### Confirmed Bugs

1. **ARI/AZ team normalization mismatch breaks Sewald's stats:** AZ→ARI normalization happens at the page level but `build_closer_grid` matches against the (already-normalized) team key. The player pool retains `team='AZ'`. Paul Sewald's ARI card shows `—` for SV/ERA/WHIP in the primary stat section while simultaneously showing correct actual stats in the green "2026 ACTUAL" row. Novice interpretation: "Is Paul Sewald injured? Why are his stats blank?"

2. **9 missing teams have no empty-state explanation:** BAL, ATH, CIN, LAA, MIN, CHC, COL, SF, WSH all have active closers who are simply absent from the DB's `depth_chart_role='closer'` set. The "Showing 21 teams with closer data" caption does not tell the user which teams are missing, why, or what to do. Users who own BAL (Ryan Helsley, 7 SV) or CIN (Emilio Pagán, 6 SV) cannot see those teams' situations on this page.

3. **SV metric is projected, not actual, with no label:** The large SV figure on each card is the blended preseason projection from `projections` (system='blended'). Mid-season this is systematically lower than actual saves. No "projected" qualifier is shown on the card, only the label "SV". The "2026 ACTUAL" line below creates two conflicting save values.

4. **SETUP row is empty for 19/21 cards:** Zero `depth_chart_role='setup'` rows exist in the DB. The feature renders `SETUP · —` for 19 of 21 teams, which is noise, not signal.

5. **% JOB scores are clustered in a narrow band (44–68%):** Because `closer_confidence` starts at 0.75 for all named closers, the security scores have low variance and the heatbar is always orange. The visual encoding provides no warning signal.

6. **gmLI feature is dead:** No data, no table, no population path. The code is wired for it but it never fires.

7. **Depth chart data is 6 days stale with no age indicator:** `refresh_log` shows `depth_charts: no_data, 2026-06-07T20:23:35`. The info banner says "loaded at app launch" but gives no timestamp, no freshness badge, and no "last updated N hours ago" line.

### UX / Design Issues

8. **No search, filter, or sort controls:** With 21 cards in a fixed alphabetical grid, a user who wants to find teams with low job security (to target their closer's opponent) must read all 21 cards.

9. **No "teams I care about" highlight:** The user's league roster isn't cross-referenced. No visual indicator shows which closers are on Team Hickey's roster or on opponents' rosters.

10. **Page title says "30-team" but shows 21:** The `render_reco_banner` teaser says "30-team closer depth chart" — this is an outright inaccuracy when only 21 teams appear.

11. **No committee situation detection:** `compute_committee_risk()` exists in `src/closer_monitor.py` and does sophisticated committee analysis, but it is never called from the page. Teams in true committee situations (COL, WSH, etc.) are simply absent.

12. **No "no save opportunities" indicator:** Some "closers" have poor save situations (KC/Erceg with 12 SV on a .444 team, PIT/Soto with 8 SV). Nothing on the card indicates how often the team has save opportunities.

13. **Headshot image load: 21 external MLB CDN calls per page:** Each card calls `_headshot_img_html(mlb_id, size=34)` which builds an `<img>` tag pointing to `mlb-cdn.com`. These are external calls that may fail silently or load slowly, especially if the CDN is rate-limiting.

14. **`render_reco_banner("30-team...", "", "closer")` passes `icon_key="closer"` which does not exist in `PAGE_ICONS`:** `PAGE_ICONS.get("closer", "")` returns `""` — no icon renders in the banner teaser. The `PAGE_ICONS` dict has "zap" (default), but not "closer".

15. **Card uses `font-family:var(--font-display)` for stat figures:** The SV/ERA/WHIP numbers use Archivo (display font) with `font-variant-numeric:tabular-nums`. The design system specifies IBM Plex Mono (`--font-mono`) for figures/numbers. The stat values should use `var(--font-mono)` for correct numerical alignment per the Combustion spec.

16. **`st.markdown(unsafe_allow_html=True)` — no use of `st.components.v1.html`:** Confirmed clean on this cross-cutting check. No deprecated HTML component.

---

## 6. UI/UX & Visual Design Critique

### Layout & Density
The 5-column card grid at a 1400px viewport gives each card approximately 260px of width. This is tight for the content — a 34px headshot + player name on one line, a 4px heatbar, a 3-column stat group (SV / ERA / WHIP), and then up to 3 additional meta rows (ACTUAL stats, Recent Form, SETUP). At smaller browser widths (1280px, laptops) the cards will be tighter and name truncation will be more frequent.

The card ordering is alphabetical by team code (ARI → TOR), which is technically correct but useless for a manager trying to prioritize. Nobody scouts closers alphabetically.

### Color Use
The heatbar is orange for all 21 cards (security ≥ 0.5 → orange gradient). This means the color signal that was designed to warn about shaky closers never triggers. The only color differentiation between "68% JOB (PHI/Duran, very safe)" and "44% JOB (TOR/Varland, shaky)" is the text percentage — the bar looks identical. This is a design-signal failure.

The "2026 ACTUAL" row uses `var(--fp-green)` for the text, which is semantically correct (actual good data in green). But on a white/pale card surface, this creates a color-hierarchy conflict: green typically means "positive" but here it just means "this year's real stats." Low-security closers with bad ERAs (Lucas Erceg: 6.00 ERA) still get the green actual-stats row.

### Typography
Stat values (SV, ERA, WHIP) use `var(--font-display)` (Archivo) with `font-variant-numeric:tabular-nums`. Per the Combustion spec, figures should use `var(--font-mono)` (IBM Plex Mono). This is an inconsistency with the design system's number-formatting rule.

The SETUP, "2026 ACTUAL", and gmLI text use 9–9.5px font sizes. At standard (96dpi) viewing this is on the edge of legibility for many users.

### Information Architecture
The card tries to show:
1. Team identity (badge + watermark)
2. Closer identity (headshot + name)
3. Job security (heatbar + %)
4. Projected saves (SV)
5. Quality metrics (ERA, WHIP)
6. Actual 2026 stats (green row)
7. Recent form trend (conditional)
8. gmLI trust (conditional, dead)
9. Setup heir (SETUP row)

That's 9 information layers in a card roughly 260×172px. It's too dense. The "gmLI" and "Recent Form" rows add visual rows that nearly always render as invisible HTML comments — dead visual space that still takes rendering time.

### Mobile
The 5-column layout will collapse to 1 column on mobile (Streamlit's `st.columns` doesn't automatically reflow to fewer columns; it stacks). On a ~390px mobile viewport, 5 columns each at 78px would be unreadably tiny. The design appears completely unoptimized for mobile.

### Accessibility
The team logo watermark has `aria-hidden="true"` — correct. The headshots have no `alt` text beyond `""`. The color-coded % JOB percentage (green/yellow/danger) has a text equivalent (the number), which is adequate. But the heatbar conveys information through color alone with no text label for color-blind users ("orange = secure") — WCAG 1.4.1 failure.

### Cross-Cutting Findings Confirmed for This Page
- **`st.components.v1.html` deprecated API:** NOT used on this page. Clean.
- **Data freshness:** No staleness indicator anywhere. Depth chart data is 6 days old per `refresh_log`.
- **FIG numbering:** `FIG.03` is zero-padded correctly, consistent with the system.
- **Nav vs title:** Sidebar label is "Closer Monitor"; page H1 is "Closer Monitor." — consistent.
- **No emoji / off-palette hex:** Confirmed clean (this page is generated HTML, not using any off-palette colors in the Python source beyond THEME variables).

---

## 7. Recommendations (≥10, ordered by impact)

### R1 — Fix AZ → ARI team code normalization in `build_closer_grid` (Blocker)
**Problem:** Paul Sewald's ARI card shows blank SV/ERA/WHIP because `build_closer_grid` matches pool rows by `player_pool["team"] == normalized_team_key` (ARI), but the pool has `team='AZ'`. Stats appear missing despite Sewald having 15 actual saves.
**Fix:** Before the pool match in `build_closer_grid`, apply the same `_TEAM_NORMALIZE` mapping in reverse — or, better, normalize the pool's team column before matching, using the same `_TEAM_NORMALIZE` dict. Alternatively, pass the `original_team_key` alongside the normalized key so the pool lookup uses the stored code.

### R2 — Explain (or fill in) the 9 missing teams (High)
**Problem:** BAL, ATH, CIN, LAA, MIN, CHC, COL, SF, WSH all have active closers with saves in `season_stats`, but the DB's `depth_chart_role` column has no `closer` entry for them. The page just omits them silently and says "Showing 21 teams." Users with Helsley, Pagán, Romano on their roster see nothing.
**Fix A (Data):** Make the SV-heuristic fallback apply *per-team* rather than only when `depth_data` is completely empty. If a team has no `closer` role, check `season_stats` for the top SV pitcher and add a ESTIMATE card with a clear warning chip.
**Fix B (UI):** Show empty placeholder cards for the 9 missing teams with a "No data — run bootstrap to populate" message and name the team so users know coverage is incomplete.

### R3 — Label the SV number as "Projected" and make the dual-SV layout unambiguous (High)
**Problem:** The main SV stat (orange, 15px) is the blended *projection*; the green "2026 ACTUAL" row is the real count. Both are visible on every card. A novice reads "SV: 7" and then "2026 ACTUAL · 21 SV" and is confused.
**Fix:** Rename the projected column label from "SV" to "PROJ SV" (or "PROJ"). Alternatively, pivot the card to show actual SV as the primary hero number (since we're mid-season) and project the projected total for the rest of season as secondary.

### R4 — Fix `%JOB` score to incorporate actual (YTD) saves, not just projected (High)
**Problem:** `compute_job_security` uses `projected_sv` from the blended projection table. Mid-season, these projections are systematically stale. Cade Smith has 21 actual saves but his job security shows only 62% because his projected SV was 7.
**Fix:** Blend actual YTD saves (from `season_stats`) with remaining-season projection, or simply replace `projected_sv` in the formula with `ytd_sv` for mid-season calculations. The score would become much more meaningful.

### R5 — Sort cards by actionability, not alphabetical order (High)
**Problem:** Cards are alphabetical by team code (ARI first, TOR last). This is useless for a fantasy manager.
**Fix:** Add a sort control (dropdown or button row) with options: "Saves ↓" (by actual SV, showing most productive closers first), "Job Security ↓", "ERA ↑" (for quality), and "My roster first" (cross-reference with `league_rosters` to bubble up Team Hickey's closers and opponents').

### R6 — Add "My Roster" and "For Trade" badges to cards (High)
**Problem:** Nothing on the page connects closer data to the user's team. A manager scanning for trade targets or waiver adds has to do all the mapping in their head.
**Fix:** Cross-reference each card's `mlb_id` against `league_rosters` for Team Hickey. Cards owned by the user get an orange "MINE" chip; closers on the free agent wire get a "FREE" chip; closers on league opponents get a team-name badge.

### R7 — Add real data to the SETUP row, or hide it when empty (Medium)
**Problem:** 19/21 cards show `SETUP · —` because the DB has zero `depth_chart_role='setup'` rows. The feature is wired but dead data-wise, adding visual noise with no information value.
**Fix A (Data):** Populate `depth_chart_role='setup'` from the MLB Stats API or pybaseball bullpen data during bootstrap — specifically targeting the top 1-2 relievers by hold frequency or highest-leverage appearances per team.
**Fix B (UI):** Until the data exists, hide the SETUP row entirely to avoid the misleading "—" on every card. Show it only when setup data actually exists.

### R8 — Add a data-freshness indicator and "last updated" timestamp (Medium)
**Problem:** The info banner says "loaded at app launch" but `refresh_log` shows the data is 6 days old. There is no timestamp, no staleness badge, no "Run bootstrap to refresh" CTA.
**Fix:** Pull the `last_refresh` timestamp from `refresh_log` for `depth_charts` and render it in the info banner: "Depth chart data last updated 2026-06-07 — 6 days ago. Refresh to update."

### R9 — Explain "% JOB" to a novice (Medium)
**Problem:** `63% JOB` is not self-explanatory. Is this the probability they keep the closer role? The probability they convert a save? The "JOB" abbreviation is confusing.
**Fix:** Add a short tooltip or footnote explaining the metric. A simple legend below the cards: "% JOB = estimated job security (role stability × projected saves). Green ≥70%, Yellow 40-70%, Red <40%."

Also: rename the metric from "% JOB" to "JOB SECURITY" or "ROLE %" — something that describes what it means without ambiguity.

### R10 — Fix the heatbar so it actually shows variation (Medium)
**Problem:** All 21 closers score between 44% and 68% job security. The heatbar threshold for orange vs steel is 50%, so 19/21 bars are identical orange. The bar conveys no information beyond "this guy is probably the closer."
**Fix:** Rescale the heatbar to show relative security within the actual range. Alternatively, lower the threshold for "hot" to 65% so the bar distinguishes high-confidence closers (PHI/Duran, 68%) from borderline ones (ARI/Sewald, 51%).

### R11 — Add missing teams placeholder cards (Medium)
**Problem:** The header says "30-team" but 9 teams are absent. If a user owns Ryan Helsley (BAL, 7 SV) they cannot see BAL's situation.
**Fix:** Render 9 greyed-out placeholder cards at the end of the grid for the missing teams. Show the team name, logo watermark, and a "No closer data" empty state using `render_empty_state()`. This is more honest than pure omission.

### R12 — Fix icon key `"closer"` in `render_reco_banner` (Low/Polish)
**Problem:** `render_reco_banner("30-team closer depth chart", "", "closer")` passes `icon_key="closer"`, which does not exist in `PAGE_ICONS`. The banner renders with no icon (`PAGE_ICONS.get("closer", "")` returns `""`). The `"zap"` default or a bullpen-appropriate icon should be used.
**Fix:** Either add a `"closer"` icon to `PAGE_ICONS` (e.g., a save/fire SVG) or change the call to use an existing key like `"zap"`.

### R13 — Use `var(--font-mono)` for stat figures (Polish)
**Problem:** SV/ERA/WHIP values use `var(--font-display)` (Archivo). The Combustion design spec assigns `var(--font-mono)` (IBM Plex Mono) to figures and numbers. This is an inconsistency.
**Fix:** Change `font-family:var(--font-display)` on the stat value divs to `font-family:var(--font-mono)`.

### R14 — Surface `compute_committee_risk()` for teams without a defined closer (Low)
**Problem:** The engine has a sophisticated committee-detection function (`compute_committee_risk`) that uses save distribution data to identify true closer committees — but it's never called from the page. Missing teams like COL, WSH, MIN are effectively in committee situations.
**Fix:** For teams that appear via the SV fallback (R2 fix), call `compute_committee_risk(team_sv_distribution)` and render a "COMMITTEE" badge on those cards.

---

## 8. Severity-Tagged Issue List

- **[BLOCKER]** ARI/Paul Sewald card shows `—` / `—` / `—` for all stats (SV/ERA/WHIP) due to AZ→ARI team code normalization mismatch in `build_closer_grid` pool lookup. Card simultaneously shows "2026 ACTUAL · 15 SV" below, creating a direct contradiction.

- **[HIGH]** 9 teams missing entirely (BAL, ATH, CIN, LAA, MIN, CHC, COL, SF, WSH) with no explanation. All have active closers. User who owns Ryan Helsley, Jordan Romano, Emilio Pagán sees nothing about those teams.

- **[HIGH]** Page header says "30-team closer depth chart" but only 21 teams appear — outright inaccuracy in the UI label.

- **[HIGH]** The "SV" stat displayed on each card is the **projected** (blended preseason) value, not the actual 2026 saves. In mid-season, actual saves (shown in a separate "2026 ACTUAL" row) are wildly higher than projections (e.g., Cade Smith: projected 7, actual 21). Two SV numbers on one card with no clear labeling of projected vs actual.

- **[HIGH]** `% JOB` formula uses projected SV, not actual YTD saves. Mid-season this produces misleading low scores: CLE/Cade Smith shows 62% despite having 21 saves and clear ownership. The metric does not reflect real-world closer security.

- **[HIGH]** SETUP row is empty (`—`) for 19/21 cards because `depth_chart_role='setup'` has zero DB entries. The feature adds visual complexity with zero information value for nearly every card.

- **[MEDIUM]** All 21 heatbars render orange (security ≥ 50% for 20/21 cards). The orange/steel threshold provides no visual differentiation between secure and shaky closers. Visual encoding is non-functional.

- **[MEDIUM]** No data freshness indicator. Depth chart data is 6 days stale per `refresh_log`. Banner says "loaded at app launch" with no timestamp.

- **[MEDIUM]** Cards sorted alphabetically by team code, not by any actionable dimension (saves, job security, quality metrics). Least useful default sort for a manager.

- **[MEDIUM]** No roster integration — nothing indicates which closers are on the user's team, which are available to add, or which opponents' closers are relevant.

- **[MEDIUM]** `"% JOB"` label is opaque. A novice cannot infer from the label alone what it measures. No legend, tooltip, or explanation anywhere on the page.

- **[MEDIUM]** gmLI feature is completely dead (`closer_gmli_data` never populated, no `gmli_data` table in DB). The code silently renders HTML comments for every card.

- **[MEDIUM]** Name truncation confirmed: "Raisel Igl", "Paul Sew" (orchestrator observation), "Seranthony Dom..." likely. Cards use `text-overflow:ellipsis` but the full name is important for player identification.

- **[LOW]** `render_reco_banner` called with `icon_key="closer"` which does not exist in `PAGE_ICONS`. No icon renders in the teaser banner.

- **[LOW]** Stat values (SV/ERA/WHIP) use `var(--font-display)` (Archivo) instead of `var(--font-mono)` (IBM Plex Mono), violating the Combustion design spec for figure typography.

- **[LOW]** 21 external MLB CDN headshot image requests per page render (`mlbstatic.com` URLs), all loading in parallel. In degraded network conditions, card header rows may flash then load async. No lazy-loading beyond the default HTML.

- **[LOW]** Character encoding issue: names with diacritics (`Andrés Muñoz`, `Seranthony Domínguez`) may render correctly in the browser but showed substitution characters in the DB audit, suggesting inconsistent encoding handling between different code paths.

- **[POLISH]** `compute_committee_risk()` and `compute_skill_decay()` functions exist in `src/closer_monitor.py` but are never called. Page misses detecting committee situations and early skill decline signals.

- **[POLISH]** The 5-column card layout has no responsive fallback. At narrow viewport widths (laptop, mobile), cards will be either tiny or break layout.

- **[POLISH]** Heatbar accessibility: color is the only differentiator between "secure" (orange) and "shaky" (steel — never actually shown). No text-based alternative for color-blind users on the bar itself.
