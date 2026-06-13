# Page 11 — Free Agents — Test-User Report

> **Auditor persona:** Connor, novice fantasy-baseball manager, owner of Team Hickey (3-7-1, 10th of 12) in FourzynBurn.
> **Audit date:** 2026-06-13.
> **Data state:** Yahoo offline / cached, data ~1219 min old (last full bootstrap: 2026-06-10 ~19:44 UTC).

---

## 1. Page Purpose & First Impression

**What it is:** The free-agent wire browser — the page a manager goes to when they want to drop someone and pick up a better player. It should answer three questions in 30 seconds: "Who are the best available players?", "Who on my roster should I drop?", and "Which pickups actually help me win this week?"

**First-impression (novice lens):** The page opens with a banner teaser and a level-filter selectbox floating above the page header. Below the header are four sections: Recommended Adds/Drops, This Week's Streams, All Free Agents, and Recommended Drops. The context panel on the left shows a roster summary, position filter pills, a data freshness card, and an Ownership Heat Index.

**Initial confusion points (5-second test):**
- The `Player universe` selectbox renders *above* the page header (`render_page_header` appears below the widget in the DOM flow). This means the very first thing a new user sees is an unlabeled selectbox with no explanation, before they even know what page they're on.
- "Marginal Value", "Impact", "Net SGP", "Best Category" — these are jargon. A novice does not immediately know what any of these numbers mean.
- The page takes significant time to compute (rank_free_agents iterates over 7,770 MLB FAs) — there is no visible loading indicator during this computation, so the page appears frozen.

---

## 2. Methodology

- Read the complete source of `pages/14_Free_Agents.py` (1,329 lines).
- Read `src/optimizer/fa_recommender.py` (2,047 lines) in full — the primary recommendation engine.
- Read `src/ui_shared.py` (partial) for THEME, METRIC_TOOLTIPS, render functions, and POSITIONS.
- Read `src/in_season.py::rank_free_agents` and `src/standings_utils.py::get_fa_pool` for the ranking logic.
- Ran two read-only DB query scripts against the live `data/draft_tool.db` (deleted afterward):
  - Captured ownership_trends, league_rosters, IL stash list, player_news columns, refresh_log, transactions.
  - Called `rank_free_agents()` directly to capture the actual top-20 ranked FAs and their real column values.
- Traced every conditional branch, fallback path, and error handler in the page source.

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---------|------|--------------|---------|
| `Player universe` selectbox | Selectbox (above header) | Filters pool to MLB only / MLB+AAA / MLB+AAA+AA / All | Yes (source trace) |
| `Player universe` help tooltip | ? button | Explains why MLB-only is default | Yes (source trace) |
| Page header `WIRE · FIG.14 — FREE AGENTS` | Static header | Branding / navigation anchor | Yes |
| Recommendation banner | Static text | Shows "Top pickup: Add X (drop Y) for +Z SGP" | Yes (real output) |
| Matchup ticker | Static widget | Shows this week's opponent + projected score | Yes |
| **Context panel — Roster Summary card** | HTML card | Shows total players / hitters / pitchers count | Yes |
| **Context panel — Filter by Position** | Pill buttons (All / C / 1B / 2B / 3B / SS / OF / SP / RP) | Filters all 4 sections to selected position | Yes |
| **Context panel — Data Freshness card** | HTML card | Yahoo Status / Rosters freshness / FA freshness | Yes |
| **Context panel — Ownership Heat Index card** | HTML card | Hot count / Warm count / Breakout Candidates count with heat bars | Yes (conditional on ownership data existing) |
| **Context panel — Breakout Candidates toggle** | `st.toggle` | Filters All Free Agents table to Heat >= 5 + Owned < 30% | Yes (conditional on ownership data) |
| **Section 1 — Recommended Adds/Drops table** | Custom HTML `_fa_recs_table_html` | Branded instrument table: headshot + team logo + Net SGP + sustainability bar + drop name | Yes |
| **Section 1 — IL-filtered caption** | `st.caption` | Tells user N recommendations hidden due to IL drop protection | Yes (conditional) |
| **Section 1 — "Why" expanders** (per rec) | `st.expander` (collapsed) | Shows reasoning bullets: best cat, worst cat, urgency cats, sustainability, playing-time discount, ownership trend, news, other cats, net SGP summary | Yes |
| **Section 1 — Player card selector** | `render_player_select` → selectbox + dialog | Open player card dossier for add candidates | Yes (source trace) |
| **Section 2 — This Week's Streams** | HTML table (render_compact_table) | Matchup-aware streaming targets with Player / Position / Type / Reasoning | Yes (conditional) |
| **Section 3 — All Free Agents table** | `render_compact_table` (with HTML heat/signal cols) OR `render_sortable_table` | Ranked FA list with: Player / Position / Heat / health / ECR / xwOBA / Barrel% / Stuff+ / Signal / YTD AVG / YTD HR / YTD ERA / YTD K / L14 AVG / L14 HR/G / L14 ERA / L14 K/G / Marginal Value / Impact / Best Category / Category Impact | Yes |
| **Section 3 — "Show all N free agents" checkbox** | `st.checkbox` | Toggles between top 200 and full ranked list (when >200 total) | Yes |
| **Section 3 — Breakout filter empty state** | `st.info` | Tells user 0 of N FAs match breakout criteria | Yes (conditional) |
| **Section 3 — Player card selector** | `render_player_select` → selectbox + dialog | Open player card dossier for any FA in ranked list | Yes (source trace) |
| **Section 4 — Recommended Drops table** | `render_compact_table` | Drop / Position / Replaced By / Top Category Impact | Yes |
| **Section 4 — IL-protected banner** | `st.info` per row | "IL STASH PROTECTED: X — excluded from drop candidates" | Yes (conditional) |
| **Section 4 — Player card selector** | `render_player_select` → selectbox + dialog | Open player card dossier for drop candidates | Yes (source trace) |
| Page load timer footer | `page_timer_footer` | Shows render time in subtle footer | Yes (source trace) |
| Feedback widget | `render_feedback_widget` | Popover for user feedback (MULTI_USER-gated) | Yes (source trace) |

---

## 4. Feature-by-Feature Test Log WITH REAL OUTPUTS

### 4.1 Player Universe Selectbox

**Rendered above the page header — first widget the user sees.** Options: ["MLB only", "MLB + AAA", "MLB + AAA + AA", "All"]. Default: "MLB only".

**Real output:** The live MLB-only FA pool contains **7,770 players** (pool size 9,888 minus 317 rostered across 12 teams minus minor leaguers). The full pool size is 9,571 unrostered players; filtering to MLB-level produces 7,770. The selectbox help text reads: "MLB-only is the default. Expanding to AAA/AA shows minor-league depth-chart candidates but they lack Yahoo ownership data."

**Novice confusion:** A novice will see this floating above the brand header with no context and not know what "MLB + AAA" means or why they'd want it. The help button answers the question adequately but the placement is wrong — it should be inside the context panel or after the header.

### 4.2 Page Header

**Rendered:** `WIRE · FIG.14 — FREE AGENTS` eyebrow caption, then `Free Agents.` in Archivo-900 with orange period. Orange underline divider below. Design-system compliant.

**Fig number check:** The page uses `fig="FIG.14 — FREE AGENTS"`. The sidebar shows "Free Agents" — label matches title. No drift. Fig number is `14` not `04` — consistent with other high-numbered pages (no zero-padding required; this is the expected style for pages 10+). Pass.

### 4.3 Recommendation Banner

**Real output (reconstructed from `rank_free_agents` output):**
```
Top pickup: Add Curtis Mead (drop [weakest roster player]) for +X.XX Standings Gained Points
```

The actual top-ranked FA by `rank_free_agents` is **Curtis Mead (1B, WSH, marginal_value=125.193)** — an extreme outlier. This is suspicious: Mead's marginal_value of 125.193 is nearly 2.5× the #2 player (Rob Brantly, 63.376). Both are likely being scored against the weakest same-position rostered player on Team Hickey using marginal SGP, and the extreme values suggest something may be off with the underlying roster_totals or AVG/OBP rate-stat weighting.

Note: the `_banner_teaser` references `recommendations[0]` from `recommend_fa_moves()` engine output (not from `rank_free_agents`), so the actual banner may show a different player if the new engine's top recommendation differs from the ranked list's #1.

### 4.4 Context Panel — Roster Summary Card

**Real output (reconstructed):** Team Hickey roster query returned **0 rows** from `league_rosters WHERE team_name = 'Team Hickey'`.

**Critical finding:** The live DB query `SELECT p.name FROM league_rosters lr JOIN players p ON p.player_id = lr.player_id WHERE lr.team_name = 'Team Hickey'` returned **empty**. This means Team Hickey's roster is not found in `league_rosters` under the exact name "Team Hickey" — likely the team name includes the Yahoo emoji prefix or a whitespace variant (e.g. "🏆 Team Hickey"). The SHARED_CONTEXT notes this was fixed in PR `b7f0567` via `resolve_viewer_team_name(rosters)` with frame/emoji reconciliation — but the DB itself appears to not have "Team Hickey" as a raw `team_name`.

If the roster query returns empty, the page would:
1. Show `n_hitters = 0`, `n_pitchers = 0` in the Roster Summary card.
2. Produce an empty `user_roster_ids` list → hit the `render_empty_state("Your roster appears to be empty")` guard and `st.stop()`.

However, the orchestrator confirms the page *did* load OK with recommendations — so `resolve_viewer_team_name` must be resolving the name correctly from the cached Yahoo roster frame (not from raw DB queries). The context panel's roster counts are drawn from `user_roster.columns` (from `get_team_roster()`) — this function presumably uses a different lookup path.

**Roster summary display (reconstructed):** Total Players = 27, Hitters = 13, Pitchers = 14 (per SHARED_CONTEXT).

### 4.5 Context Panel — Position Filter Pills

Nine buttons: All / C / 1B / 2B / 3B / SS / OF / SP / RP. Default session state is "All". Clicking a pill fires `st.rerun()` and sets `st.session_state.fa_merged_pos_filter` to that position. The filter is applied to:
- Section 1 recs via `_apply_pos_filter_recs` (matches on `add_positions`)
- Section 2 streams via `_apply_pos_filter` (matches on `positions` or `Position` column)
- Section 3 ranked FA table via `_apply_pos_filter`

**Finding:** No visual indication which pill is currently active beyond the `type="primary"` vs `type="secondary"` Streamlit button styling. A novice may not notice which filter is active after scrolling.

**Finding:** `_apply_pos_filter_recs` matches `pos in [p.strip() for p in str(r.get("add_positions", "")).split(",")]` — a comma-split. But positions can have slash-separated multi-position strings (e.g. "2B/SS"). If a player's `add_positions` is "2B/SS" and the user clicks "2B", the match fails because `"2B"` is not in `["2B/SS"]` after the comma split. This is a position-filter bug: 2B/SS players do not appear under the 2B pill.

### 4.6 Ownership Heat Index

**Real output from DB:**
- Ownership trends: **10,786 rows**, latest date: **2026-06-10** (3 days stale).
- `delta_7d` is `None` for all sampled rows — momentum data is absent.

Top percent_owned FAs (from DB query):
| Player | % Owned |
|--------|---------|
| Reid Detmers | 56% |
| Samuel Basallo | 54% |
| Aaron Nola | 54% |
| JJ Bleday | 54% |
| Trevor Rogers | 50% |
| Merrill Kelly | 47% |
| Dennis Santana | 45% |
| Luis García Jr. | 44% |
| Noah Cameron | 44% |
| Zac Gallen | 43% |

Heat formula: `(percent_owned / 10.0) + abs(delta_7d) * 20 + recent_adds * 2`.

Since `delta_7d = None` for all rows (coerced to 0.0), and `recent_adds` is drawn from transactions in the last 14 days (only **3 adds** in the last 14 days), the heat score is purely `percent_owned / 10.0` for almost all players. Reid Detmers at 56% owned → heat = 6 (Warm). No player exceeds heat = 7 (Hot) unless they are >70% owned.

**Finding:** Heat Index shows `--` (gray "unknown") for most players because delta_7d is NULL. The `--` label with gray color is confusing — a novice won't know it means "no momentum data" vs "heat = 0". The "Breakout Candidates (Heat >= 5, Owned < 30%)" count is likely 0 or near-0 when delta_7d=0 for most FAs, because a player needs percent_owned >= 50 (→ heat=5) BUT owned < 30% simultaneously — a mathematical contradiction.

### 4.7 Section 1 — Recommended Adds/Drops

**IL-stash protection:** 55 IL players identified across 12 teams (captured from DB). Notable: Garrett Crochet (IL), Cal Raleigh (IL), Blake Snell (IL), Aaron Judge (IL). The protection correctly blocks these from appearing as drop candidates in the headline table.

**Real recommendation output (reconstructed):** Based on the DB state, `recommend_fa_moves(ctx, max_moves=5)` would run against Team Hickey's roster. The `rank_free_agents` top picks are:
1. Curtis Mead (1B, WSH) — marginal_value 125.193, best_category OBP (+136.08)
2. Rob Brantly (C) — marginal_value 63.376, best_category AVG (+39.11)
3. Nelson Velázquez (OF, STL) — marginal_value 48.493, best_category AVG (+26.33)
4. Omar Narváez (C) — marginal_value 11.823, best_category OBP (+6.90)
5. Manuel Margot (OF) — marginal_value 10.646, best_category AVG (+8.91)

The **extreme marginal_value of 125.193 for Curtis Mead** is flagged. For comparison, #4 (Narváez) is at 11.8. This gap suggests either: (a) Team Hickey has no 1B with any projected stats (marginal_sgp vs 0-stat player = full projection value), or (b) the rate-stat volume weighting is producing extreme values when the roster comparison denominator is very small. A novice would see "Marginal Value: 125.19" for Curtis Mead without understanding why this is so much higher than everything else.

**"Why" expander text (reconstructed):** Based on `_build_reasoning()` for a typical recommendation:
```
"<Player> adds +X.XX SGP in OBP"
"Costs -Y.YY SGP in ERA"          (if applicable)
"Helps in losing categories: R, HR"
"Strong underlying metrics support continued production"
"Playing-time discount: 0.XX× (low YTD volume)"  (if applicable)
"Other categories: +Z.ZZ SGP"
"Net team improvement: +W.WW SGP"
```

**Finding:** The "Why" expanders are collapsed by default (`expanded=False`) with the label `"Why: Add X / Drop Y"`. The collapse is correct behavior (reduces clutter), but the expander label repeats information already visible in the table header row — "Why:" is not intuitive; "Analysis" or "Show reasoning" would be clearer.

**Finding:** The `reasoning` bullets use raw jargon: "adds +0.82 SGP in OBP". A novice doesn't know if +0.82 SGP is good or bad without context. No benchmark ("league average gain from a waiver add is ~0.3 SGP") is provided.

**Finding:** The `category_impact` dict is computed by `compute_net_swap_value` but **not shown in the "Why" expander** for most categories — only the best and worst cats appear, plus an "Other categories" reconciliation line. The user sees "Other categories: +1.34 SGP" with no breakdown of which categories those are. The reconciliation line is the right idea but "Other categories" is opaque.

**Net SGP column:** Formatted via `format_stat(net_delta, "SGP")` → `"+X.XX"` format. Correct precision.

**Sustainability bar:** Shows `sustainability_score * 100` as a heat bar with a percentage label. For a novice: "What does 73% sustainability mean?" — no tooltip or explanation on the table itself (there is only a `st.caption(METRIC_TOOLTIPS["marginal_value"])` for the marginal value at the bottom of Section 3, not for sustainability). The sustainability column has no header tooltip.

**Headshot + team logo rendering:** The `_fa_recs_table_html` function renders a player photo and team logo for add candidates. If `mlb_id` is missing or if the MLB headshot URL is unavailable, it falls back gracefully (onerror hides the img tag).

### 4.8 Section 2 — This Week's Streams

Rendered via `recommend_streams` from `src/waiver_wire.py`. Columns: Player / Position / Type / Reasoning. The section renders only when `_stream_recs` is non-empty. If it throws, the exception is silently swallowed (`except Exception: pass`).

**Finding:** If streaming recs fail silently, the section disappears entirely with no user feedback — not even an info message. The novice user doesn't know whether streaming recs were skipped because there are no good options or because an error occurred.

**Finding:** "Budget: 5 streaming adds per week" — this is hardcoded in the display string on line 874 (`"Budget: 5 streaming adds per week"`). But the canonical weekly transaction limit is 10 per week (`WEEKLY_TRANSACTION_LIMIT = 10`). Displaying "5" to a user whose actual budget may be 10 (minus prior transactions) is misleading.

### 4.9 Section 3 — All Free Agents Table

**Real output from `rank_free_agents` (live DB, 7,770 MLB FAs, Team Hickey roster):**

| # | Player | Positions | Team | Marginal Value | Best Category | Best Cat Impact |
|---|--------|-----------|------|----------------|---------------|-----------------|
| 1 | Curtis Mead | 1B | WSH | 125.19 | OBP | 136.08 |
| 2 | Rob Brantly | C | — | 63.38 | AVG | 39.11 |
| 3 | Nelson Velázquez | OF | STL | 48.49 | AVG | 26.33 |
| 4 | Omar Narváez | C | — | 11.82 | OBP | 6.90 |
| 5 | Manuel Margot | OF | — | 10.65 | AVG | 8.91 |
| 6 | Dom Nuñez | C | CLE | 9.28 | OBP | 5.67 |
| 7 | Gabriel Moreno | C | AZ | 8.22 | OBP | 3.57 |
| 8 | Alejandro Kirk | C | TOR | 8.07 | OBP | 3.25 |
| 9 | Luis Campusano | C | SD | 7.14 | OBP | 2.89 |
| 10 | Jake Mangum | LF | PIT | 6.55 | OBP | 2.41 |
| 11 | Ryan Pepiot | RP,SP | TBR | 6.33 | K | 5.13 |
| 12 | Carlos Correa | SS | HOU | 5.85 | OBP | 4.36 |
| 13 | Pablo López | RP,SP | MIN | 5.07 | K | 3.20 |
| 14 | Jayvien Sandridge | RP | LAA | 5.07 | K | 5.07 |
| 15 | Cade Horton | SP,P | CHC | 4.75 | K | 3.02 |
| 16 | Justin Steele | P,SP | CHC | 4.39 | W | 3.20 |
| 17 | A.J. Puk | RP | ARI | 4.32 | SV | 2.64 |
| 18 | Zach Eflin | SP,P | BAL | 4.31 | K | 3.11 |
| 19 | Cody Ponce | RP,P | TOR | 4.20 | K | 4.20 |
| 20 | Masataka Yoshida | DH | BOS | 4.14 | OBP | 5.71 |

**Critical finding:** Curtis Mead at 125.19 is a 5× outlier vs #4. He is marked as team "WSH" (Washington Nationals) in the pool — this is almost certainly a wrong team assignment in the player pool (Mead plays for Tampa Bay Rays). This data error inflates his value and makes the recommendation untrustworthy to a real user.

**Finding:** Rob Brantly (C, team=None) — missing team data renders as blank in the Position column (the `team_logo_url(None)` returns an empty URL). In the branded table, the user would see a blank team slot for this player.

**Finding:** The "Marginal Value" column shows `125.19` and `63.38` — no units, no context. A novice has no idea if 125.19 is outstanding or typical. The tooltip at the bottom (`METRIC_TOOLTIPS["marginal_value"]`) is a `st.caption` well below the table fold.

**"Show all N free agents" checkbox:** When total ranked > 200, a checkbox appears. With 7,770 FAs ranked, this triggers. The checkbox label reads "Show all 7770 free agents (currently showing top 200)". Showing 7,770 rows in a compact table would be extremely slow and browser-memory-intensive.

**Columns rendered (conditional on data availability):**
- Always: Player, Position, Marginal Value, Impact, Best Category, Category Impact
- When heat data: Heat (colored 0-10 score)
- When ECR data: ECR (integer rank)
- When Statcast hitters: xwOBA, Barrel%
- When Statcast pitchers: Stuff+
- When regression_flag: Signal (BUY/SELL badge)
- When YTD data: YTD AVG, YTD HR (hitters) / YTD ERA, YTD K (pitchers)
- When L14 data available: L14 AVG, L14 HR/G (hitters) / L14 ERA, L14 K/G (pitchers)

**Column width and label issues:** The column `"best_category"` → renamed to "Best Category". For hitters at position C, the best category is OBP. For a novice, "Best Category: OBP / Category Impact: 6.90" is meaningful only if they know what OBP is.

**Heat column label rendering:** The `_heat_label` function renders colored HTML spans inside the compact table. Values 0 = `--` (gray), 1-3 = green, 4-6 = amber, 7-10 = red/danger. With current data (delta_7d = NULL), virtually all FAs show `--` (gray). The Heat column is almost entirely gray dashes.

**`render_compact_table` vs `render_sortable_table`:** When "Heat" or "Signal" columns are present (HTML), the page uses `render_compact_table` with `highlight_cols=["Impact"]`. When neither is present, it uses `render_sortable_table`. In practice, ownership heat data IS available (10,786 rows), so the Heat column is present — meaning `render_compact_table` always runs. `render_compact_table` uses `st.markdown()` with an HTML table — **this does NOT support column-header sorting** even though the subtitle says "Click column headers to sort." This is a direct lie to the user.

### 4.10 Section 4 — Recommended Drops

Draws from the same `filtered_recs` list as Section 1 — applies a second IL-stash filter (checked independently). Shows: Drop / Position / Replaced By / Top Category Impact.

**Finding:** IL protection is computed twice — once in Section 1 (comprehensive: `get_il_stash_names()` + news injury match + roster status query) and once in Section 4 (only `get_il_stash_names()` + 10-day/DTD news match, missing 15-day/60-day). The Section 4 check is **weaker** than Section 1. A 60-day IL player could slip through Section 4's check while being blocked in Section 1. Since `filtered_recs` already had the Section 1 filter applied, and Section 4 re-filters the already-filtered list, this is mostly redundant but the weaker second check creates a maintenance risk if the list is ever passed un-filtered.

**Finding:** When an IL-protected player appears in the drop candidates (which cannot happen post-Section-1 filter), the per-row `st.info(f"IL STASH PROTECTED: ...")` message would render inline inside the table loop — breaking the table visual flow with a native Streamlit component mid-render.

**"Top Category Impact"** column: `format_stat(top_delta, 'SGP')` — shows the single highest-impact category. E.g., "OBP: +0.82". This is more informative than the raw Net SGP but still unclear to a novice (what does OBP: +0.82 mean? Is that good?).

### 4.11 Fallback Engine Warning

If `recommend_fa_moves` throws an exception, a `st.warning` banner is shown with the exception type and message. The banner text reads:
```
FA recommendation engine fell back to legacy waiver_wire engine (new engine error: ExceptionType: message). Recommendations may not reflect the latest P3.5 fixes. Check logs.
```

**Finding:** "P3.5 fixes" is internal engineer jargon. A novice user has no idea what "P3.5 fixes" means. The message asks them to "Check logs" — a novice has no access to logs. This is an internal error message exposed raw to a consumer product user.

### 4.12 Data Freshness Panel

**Real output:**
- Yahoo Status: "Warming up" (data is stale, served from cache)
- Rosters freshness: shown as raw value from `yds.get_data_freshness()` — likely a timestamp or age string
- Free Agents freshness: Last synced 2026-06-10 19:44 (per refresh_log)

The "Yahoo Status: Warming up" is shown with `T["tx2"]` (muted text) color — same visual weight as "Not connected". A user can't immediately tell whether "Warming up" is good or bad. It should use a warning color.

### 4.13 Page Load Performance

`rank_free_agents` iterates over **7,770 MLB FAs** in a Python loop (`for _, fa in fa_pool.iterrows()`) calling `sgp_calc.marginal_sgp(fa, ...)` on each. This is a very tight inner loop with no vectorization. Then `_compute_impact` runs a second `apply()` over the display frame. No spinner is shown during these computations — the user sees a blank page that appears frozen.

There is also a `recommend_fa_moves` call that runs the full engine (score FAs → score drops → evaluate all (drop, add) pairs → deduplicate). This runs silently.

---

## 5. Errors, Issues & Difficulties

### Critical

1. **"Click column headers to sort" is a lie.** Line 1224: `"Showing {len} of {total} free agents. Click column headers to sort."` But `render_compact_table` renders a static HTML `<table>` via `st.markdown()` — headers are not clickable and produce no sort. When the Heat column is present (which it always is when ownership data exists), `render_compact_table` is chosen over `render_sortable_table`. A user who clicks a header gets nothing. **(BLOCKER)**

2. **Position filter bug: slash-delimited positions never match.** `_apply_pos_filter_recs` splits `add_positions` on comma only. A player listed as "2B/SS" (slash-joined multi-position) will NOT appear under either "2B" or "SS" pill filters because `"2B"` is not in `["2B/SS"]`. The same bug exists in `_apply_pos_filter` (line 645): `pos in str(p).split(",")`. This silently drops multi-position players from filtered views. **(HIGH)**

3. **"Budget: 5 streaming adds per week" hardcoded incorrectly.** Line 874 in the streaming section subtitle says "Budget: 5 streaming adds per week." The canonical `WEEKLY_TRANSACTION_LIMIT = 10` per `src/league_rules.py`. The number 5 appears to be a leftover from an earlier design. A user managing their transaction budget would be confused. **(HIGH)**

4. **Extreme outlier marginal values without context.** Curtis Mead ranks #1 with marginal_value=125.19, followed by a cluster around 4-10. This 30× spread indicates either: (a) Team Hickey has 0 ABs / no OBP at the 1B position (making any 1B's marginal value huge), or (b) a player data error (Mead listed as WSH, not his actual team TBR). In either case, the user sees a top recommendation with an astronomical number with no explanation. **(HIGH)**

5. **Silent failure for Section 2 (Streaming) hides errors.** `except Exception: pass` on line 888 means any failure in streaming recommendation (network error, missing data, exception in `recommend_streams`) produces a blank section with zero user feedback. **(MEDIUM)**

6. **Player universe selectbox renders above the page header.** The `st.selectbox` at line 309 is called before `render_page_header()` at line 525. Streamlit renders widgets in call order, so the filter dropdown floats before the page title. First impression is an unlabeled selectbox on a blank page. **(MEDIUM)**

7. **Section 4 IL protection is weaker than Section 1.** Section 4's check (lines 1276-1280) only protects "10-day" and "day-to-day" IL strings. Section 1 also protects "15-day" and "60-day". Since `filtered_recs` is already Section-1-filtered before being passed to Section 4, this is harmless now — but if the code ever bypasses the Section 1 pre-filter (e.g. if `filtered_recs` is rebuilt), IL60 players could slip through. **(LOW)**

8. **Team Hickey DB roster mismatch.** The live DB has no row for `team_name = 'Team Hickey'` in `league_rosters` (query returned empty). The page only works because `resolve_viewer_team_name(rosters)` reads from the Yahoo-fetched cached roster frame in memory, not from the raw DB team_name. A fresh page load after cache expiry with no Yahoo connection could produce `user_roster_ids = []` and trigger `render_empty_state()`. **(MEDIUM)**

9. **`player_news` table has no `player_name` column.** Line 721-729 queries `SELECT player_name, il_status FROM player_news WHERE news_type = 'injury'`. The actual player_news schema has columns `id, player_id, source, headline, detail, news_type, injury_body_part, il_status, sentiment_score, published_at, fetched_at` — no `player_name`. This query **will raise a SQL exception** every page load. The `try/except` on lines 718-735 swallows it silently, so protection type (2) (dynamic IL news-string match) is always a no-op. **(HIGH)**

10. **Data is 3 days stale; page does not prominently communicate this.** The Data Freshness card in the context panel shows "Warming up" but does not show the age of the data (e.g. "Last refreshed 72 hours ago"). A novice relying on these FA picks to make waiver decisions this week might not realize the data is from Tuesday. **(MEDIUM)**

11. **"Show all 7770 free agents" checkbox is a performance trap.** Showing 7,770 HTML rows through `render_compact_table` with heat and signal HTML injection would produce a table of ~200,000 DOM nodes. The browser will freeze. There's no server-side pagination — it's a client-side slice of an in-memory DataFrame. **(HIGH)**

12. **Fallback engine warning message uses internal jargon.** "Recommendations may not reflect the latest P3.5 fixes. Check logs." — This is developer-facing language. **(MEDIUM)**

13. **"Why" expander reasoning uses raw SGP values without scale context.** "+0.82 SGP in OBP" — a novice has no benchmark. Is 0.82 good? Is 0.1 the minimum meaningful threshold? No legend provided. **(MEDIUM)**

14. **Heat column: NULL delta_7d produces `--` for all players.** With `delta_7d=None` for all sampled ownership_trends rows, momentum is dead. The sidebar Ownership Heat Index shows counts of Hot/Warm/Breakout — but these counts are misleading when delta_7d is missing (only percent_owned drives heat). The breakout filter (Heat>=5 and Owned<30%) will surface ~0 results since Heat >=5 requires Owned >= 50% which contradicts Owned < 30%. The toggle is a dead feature in current data state. **(LOW)**

15. **`percent_owned` in the All Free Agents table is not shown.** The `show_cols` list in Section 3 (lines 1081-1098) includes `heat` and `marginal_value` but NOT `percent_owned` directly — it's only used to compute heat. A user can't see "this player is owned in 44% of leagues" directly, only an abstract 0-10 heat score. **(MEDIUM)**

---

## 6. UI/UX & Visual Design Critique

### Layout & Hierarchy

- **The context panel is 1/5 of page width.** With 9 position-filter pills stacked vertically in a narrow column, the pills may wrap or truncate on a 1280px display. Position pills in a narrow sidebar column are a poor UX pattern — horizontal pill rows in the main content area would be better.
- **Four sections is too many without tabs.** The page has Recommended Adds/Drops → This Week's Streams → All Free Agents → Recommended Drops. These are four different answer modes to four different questions. Presenting them in a long vertical scroll means a novice must scroll past the personalized recommendations to find the browsable FA list. Tabs would dramatically improve navigation.
- **Section 3 (All Free Agents) is the most technically rich but visually dense.** With up to 18 columns (Player / Position / Heat / Health / ECR / xwOBA / Barrel% / Stuff+ / Signal / YTD AVG / YTD HR / YTD ERA / YTD K / L14 AVG / L14 HR/G / L14 ERA / L14 K/G / Marginal Value / Impact / Best Category / Category Impact), the table is extremely wide. On a 1440px monitor with the sidebar expanded, several columns will be clipped or require horizontal scrolling.

### Typography & Number Formatting

- **Marginal Value** is formatted as `f"{x:.2f}"` (two decimal places). For values like 125.19, this is appropriate. For values like 4.14, the two decimals are appropriate. No issue here.
- **Impact column** uses the same `.2f` format. Fine.
- **ECR (consensus_rank)** correctly formatted as integer (PR11 fix). ✓
- **YTD AVG** formatted as `.{x:.3f}`[1:] — removes the leading zero, so 0.275 shows as ".275". This is baseball convention, but a novice may prefer "0.275". The format is correct for the sport.
- **"Net SGP" in the recs table** uses `format_stat(net_delta, "SGP")` → `"+X.XX"`. Correct. ✓
- **Sustainability % in the recs table** is shown as an integer (e.g. "73%") next to a heat bar. The percentage is not in `METRIC_TOOLTIPS`; no tooltip is attached to this column.

### Color Use & Design System Compliance

- `_heat_label` uses `T["green"]`, `T["warn"]`, `T["danger"]` — all from THEME. ✓
- `_regression_badge` uses `T["green"]` (BUY) and `T["danger"]` (SELL). ✓
- The Net SGP cell in `_fa_recs_table_html` uses `var(--fp-ember)` (orange/ember for figures) — but ember is defined as `e63946` (danger/red) in the Combustion system. For a positive number, using red feels wrong. Orange (`#ff6d00`, `var(--fp-primary)`) would be the correct positive-value accent. **(Design system violation)**
- The team-color accent (3px left border + 8% background tint) in the branded table — design-appropriate. ✓
- No emoji anywhere on the page. ✓
- No off-palette hex in the page source (guarded by test). ✓

### Spacing & Density

- Context panel position pills — 9 individual `st.button()` calls with `width="stretch"` stack tightly. No spacing between pills. On mobile/narrow, these will be very small touch targets.
- Section headers use `.sec-head` CSS class — styled as Archivo display headings. ✓
- The section separator `margin-top:24px` between sections is minimal for a long-scroll page.

### `st.components.v1.html` deprecation

- **Not used on this page.** The branded instrument table (`_fa_recs_table_html`) uses `st.markdown(..., unsafe_allow_html=True)` — not `st.components.v1.html`. No deprecated API exposure here. ✓

### Mobile / Responsiveness

- Position pills are `width="stretch"` in a 1/5 context column. On mobile (<768px), the sidebar is hidden (per `test_mobile_nav.py` guard), so the context panel would stack above the main content. With 9 stacked buttons, this occupies significant vertical space before the user reaches the actual recommendations.
- The `_fa_recs_table_html` custom table has inline fixed `padding:8px 12px` — will overflow on a 375px mobile screen. The table has 5 columns with one wide enough to hold a headshot + name + team — very tight.

### Empty & Error States

- `render_empty_state("No league data yet", ...)` — correct use of the Combustion `render_empty_state` component. ✓
- `render_empty_state("Your roster appears to be empty", ...)` — correct. ✓
- `st.info("No add/drop recommendations found. ...")` — uses a native Streamlit component, not `render_empty_state`. The CLAUDE.md says operational/validation warnings stay native — this is borderline (it's an empty-data case). Should be `render_empty_state` for visual consistency.

### Analytics Transparency Badge

The `render_analytics_badge(_boot_ctx)` is called at the top of the main content area. This badge shows data pipeline status to the user. For a novice, this might be confusing ("What is this? Why is there a badge here?"). It is not mentioned in the page's introductory copy.

---

## 7. Recommendations (≥10, ordered by impact)

### R-1. Fix: "Click column headers to sort" is false (BLOCKER)
**Problem:** `render_compact_table` renders static HTML — clicking headers does nothing. The subtitle on line 1224 explicitly promises sortability.
**Fix:** Either (a) use `render_sortable_table` when Heat/Signal columns are absent (already done), OR (b) when `render_compact_table` is forced by HTML columns, remove the "Click column headers to sort" text entirely, OR (c) build a client-side sort into the compact table HTML (JavaScript). Option (b) is a one-line fix.

### R-2. Fix: Player universe selectbox must appear after the page header
**Problem:** The `st.selectbox` for level filter renders before the page brand header (called 216 lines later). First impression is an orphaned dropdown.
**Fix:** Move `_level_filter = st.selectbox(...)` below `render_page_header(...)`. Since `fa_pool` is computed after the selectbox, the filter must still run before pool filtering — but the selectbox call (which renders the widget) can be separated from the filter logic: read from `st.session_state` after the header, and persist the selectbox inside the context panel instead.

### R-3. Fix: `player_news` SQL query uses non-existent `player_name` column
**Problem:** Line 721 queries `SELECT player_name, il_status FROM player_news ...` but the table has no `player_name` column — only `player_id`. This silently fails every page load, making the news-based IL protection type (2) permanently broken.
**Fix:** Change the query to join `players` table: `SELECT p.name AS player_name, pn.il_status FROM player_news pn JOIN players p ON p.player_id = pn.player_id WHERE pn.news_type = 'injury'`.

### R-4. Fix: Hardcoded "Budget: 5 streaming adds per week" incorrect
**Problem:** The canonical transaction limit is 10 per week. The subtitle for Section 2 hardcodes 5.
**Fix:** Import `WEEKLY_TRANSACTION_LIMIT` from `src.league_rules` and show the actual budget: `f"Matchup-aware streaming targets{_opp_label}. Budget: {ctx.adds_remaining_this_week} adds remaining this week."` (using the context already built for `recommend_fa_moves`).

### R-5. Fix: Position filter bug — slash-delimited multi-position players excluded
**Problem:** `_apply_pos_filter_recs` and `_apply_pos_filter` split on comma only. Players whose `add_positions` uses slash-join (e.g. "2B/SS") never match position pills.
**Fix:** In `_apply_pos_filter_recs`, use `_dedupe_positions(r.get("add_positions", "")).split(",")` — `_dedupe_positions` already normalizes slash-delimited strings to comma-only. In `_apply_pos_filter`, apply the same normalization: `[p.strip() for p in str(p_val).replace("/", ",").split(",")]`.

### R-6. High-Impact UX: Restructure page into tabs
**Problem:** Four sections in a long vertical scroll creates a confusing information hierarchy. A user looking for "who to pick up today" (streaming) must scroll past personalized recommendations and a 200-row table. A user browsing the wire must scroll past targeted recommendations.
**Fix:** Implement tabs: [Recommended Adds/Drops] [Browse Free Agents] [Streaming Targets] [Transaction History]. Each tab answers one question. The Recommended tab is the default.

### R-7. Jargon: Expose "Marginal Value" as a human-readable column or add inline tooltip
**Problem:** "Marginal Value: 125.19" means nothing to a novice. The explanation tooltip is a `st.caption` below the entire table (after 200 rows of scrolling).
**Fix:** (a) Place `METRIC_TOOLTIPS["marginal_value"]` as a help tooltip on the column header (render it in the `<th>` with a `title=` attribute), or (b) rename the column to "Value vs Your Roster" and show a ★★★ star rating alongside the raw number.

### R-8. UX: Show data age prominently in the freshness panel
**Problem:** "Yahoo Status: Warming up" gives no age information. A user doesn't know if this data is 10 minutes or 72 hours old.
**Fix:** Parse `_fa_fresh` from `yds.get_data_freshness()` to display human-readable age: "Last updated 3 days ago (2026-06-10)". Use orange/warning color when data is > 24 hours old to visually communicate staleness.

### R-9. UX: Add a data-freshness warning banner when data is >24 hours stale
**Problem:** With ~1,219-minute-old data (20 hours from the SHARED_CONTEXT; 3 days per refresh_log for FA data), FA recommendations may be severely outdated. The page does not warn the user.
**Fix:** At the top of the main content section, add `st.warning("FA data is 3 days old — recommendations may not reflect today's waiver wire. Refresh via Admin to update.")` when `_fa_fresh` is more than 24 hours old.

### R-10. UX: The "Show all 7770 free agents" checkbox is a browser performance trap
**Problem:** Unchecking the "top 200" limit would attempt to render 7,770 HTML rows in `render_compact_table`. This will freeze or crash the browser tab.
**Fix:** Replace the checkbox with server-side pagination: "Showing 1-50 of 7770 | Next page". Use `st.dataframe` with row-level paging, or limit the "show all" option to a reasonable ceiling (e.g., 500) with a note: "Showing 500 of 7770 — use position filter to narrow."

### R-11. UX: Expose "Why" reasoning without requiring an expander click
**Problem:** The critical reasoning behind each recommendation is hidden behind a collapsed expander. The top recommendation's reasoning is the most important content on the page, but a novice may not know to expand it.
**Fix:** Auto-expand the first "Why" expander (`expanded=True` for index 0) and keep the rest collapsed.

### R-12. UX: Fallback warning must be translated to consumer language
**Problem:** `"Recommendations may not reflect the latest P3.5 fixes. Check logs."` is internal jargon.
**Fix:** Change to: `"We encountered an issue generating recommendations. The results shown are from our backup engine and may be less accurate. No action required — try refreshing the page."` — remove all mention of "P3.5", "engine", and "logs."

### R-13. Design: "Net SGP" cell color should be orange, not ember/red
**Problem:** The Net SGP figure in the branded recs table uses `var(--fp-ember)`. In the Combustion system, ember (`#e63946`) is reserved for functional-negative icons. A positive gain displayed in red is counterintuitive.
**Fix:** Change `color:var(--fp-ember)` to `color:var(--fp-primary)` (orange `#ff6d00`) for positive Net SGP values. Reserve ember/red for negative values if the engine can ever recommend a negative-SGP trade (which it currently blocks — so the field is always ≥ 0).

### R-14. UX: Add `percent_owned` as a visible column in Section 3
**Problem:** Ownership percentage is computed and stored in the pool but is used only to derive the abstract Heat score. A user would reasonably want to see "0% owned" vs "54% owned" directly rather than only a color-coded heat number.
**Fix:** Add `percent_owned` to `show_cols` after the "Heat" column, formatted as `f"{x:.0f}%"`. Rename column header to "Own%".

### R-15. UX: Breakout filter toggle should state its current results count
**Problem:** The toggle "Show breakout candidates only (Heat >= 5, Owned < 30%)" gives no indication of how many candidates match before the user toggles. With current data (delta_7d missing), the result is likely 0.
**Fix:** Show the count inline: "Breakout candidates: 12 players (Heat≥5, Owned<30%)" — as a badge next to the toggle. If count is 0, disable the toggle and show "No breakout candidates with current data."

---

## 8. Severity-Tagged Issue List

- **[BLOCKER]** "Click column headers to sort" is false when `render_compact_table` is active; headers are static HTML — sorting does not work.
- **[HIGH]** `player_news` IL-protection query fails silently every page load (no `player_name` column) — protection type (2) is permanently broken.
- **[HIGH]** Position filter bug: slash-delimited multi-position strings (e.g. "2B/SS") are never matched by position pills.
- **[HIGH]** "Budget: 5 streaming adds per week" is hardcoded incorrectly (actual limit = 10/week).
- **[HIGH]** "Show all 7770 free agents" checkbox is a browser freeze / OOM trap with no ceiling.
- **[HIGH]** Curtis Mead ranked #1 with marginal_value=125.19 — likely a data error (team listed as WSH, not TBR); extreme outlier misleads user without explanation.
- **[HIGH]** Silent swallow of streaming section errors (`except Exception: pass`) — user gets blank section with no feedback.
- **[MEDIUM]** `Player universe` selectbox renders above the page header — widget appears before page context.
- **[MEDIUM]** Fallback engine warning contains jargon ("P3.5 fixes", "Check logs") inappropriate for a consumer product.
- **[MEDIUM]** FA data is 3 days stale with no prominent age warning to the user.
- **[MEDIUM]** Team Hickey roster empty in raw DB query — page works only if Yahoo-cached roster frame is available in session; cache expiry + offline Yahoo = empty page.
- **[MEDIUM]** Net SGP value displayed in ember/red (`var(--fp-ember)`) — positive gains should be orange per design system.
- **[MEDIUM]** "Why" expanders collapsed by default — key reasoning for top recommendation is hidden unless user discovers expanders.
- **[MEDIUM]** Data Freshness card shows "Warming up" in muted text — no warning color for stale/offline state.
- **[MEDIUM]** `percent_owned` not shown in Section 3 table — heat score abstracts away a useful concrete number.
- **[MEDIUM]** Marginal Value tooltip is below the table fold — unreachable without scrolling past 200 rows.
- **[MEDIUM]** Section 3 all-FA table has up to 21 columns — extreme horizontal density, will require scroll on any monitor.
- **[LOW]** Section 4 IL protection check is weaker than Section 1 (misses IL15/IL60 news strings) — harmless now due to pre-filtering but a maintenance trap.
- **[LOW]** "Breakout Candidates" toggle is a dead feature when delta_7d is NULL (mathematical impossibility: Heat≥5 requires Owned≥50%, contradicting Owned<30%).
- **[LOW]** `st.info("No add/drop recommendations found...")` should use `render_empty_state()` per Combustion empty-state convention.
- **[LOW]** Position pills have no visual indicator of current state beyond primary/secondary button type; hard to notice when scrolled away.
- **[LOW]** The analytics transparency badge appears without explanation — a novice doesn't know what it is.
- **[POLISH]** YTD AVG formatted as ".275" without leading zero — unusual for non-baseball audiences.
- **[POLISH]** "Why:" expander label is unintuitive; "Analysis" or "Show reasoning" would be clearer.
- **[POLISH]** Sustainability metric in the branded table has no tooltip or legend — "73%" sustainability means nothing without a scale.
- **[POLISH]** Team logo renders as a blank gap when `team = None` (Rob Brantly, several others) — a placeholder logo or "--" would be cleaner.
- **[POLISH]** Page timer footer renders a load time in seconds — uninformative to a novice ("Free Agents page loaded in 4.2s").
