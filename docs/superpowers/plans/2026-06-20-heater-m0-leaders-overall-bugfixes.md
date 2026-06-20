# `/api/leaders/overall` bug fixes (breakout hang + sell 0-rows)

**Goal:** Fix the two `/api/leaders/overall` bugs the CMO hit on the live Research page.
**Scope:** Service layer only (`api/services/leaders_overall_service.py`) — `src/` engines unchanged. API contract unchanged (openapi.json identical).

---

## Bug 1 — `?lens=breakout` hangs >12s (FIXED)

**Root cause:** `compute_breakout_scores_batch(pool)` is O(n²) — every player is
percentile-ranked against the whole input — so scoring the full ~1500-player pool
(actually 9888 with minors) took >12s and tied up the uvicorn worker.

**Fix:** bound the input at the service layer. `_filter_breakout_candidates(pool, top_n=500)`:
1. Keeps players with Statcast data (`barrel_pct`/`xwoba`/`hard_hit_pct`).
2. Of those, takes the `top_n` **most fantasy-relevant** (`consensus_rank` asc, then
   `percent_owned` desc) via `_take_most_relevant` — so the bounded set is the players
   that matter, not an arbitrary slice of the pool's natural order.
3. Total: never raises, never empty-by-surprise (missing-column guards + natural-order fallback).

Breakout scores stay relative to the filtered set. **Live: 2.35s (was >12s), 25 rows.**

**Engine untouched** — `src/leaders.py` is not modified; only the input is bounded.

**Guard tests (deterministic, no live DB, CI-portable):**
- `test_filter_breakout_candidates_caps_input_size` — 1500-row synthetic pool → ≤ 500 (the O(n²) regression guard).
- `test_filter_breakout_candidates_prefers_relevance` — bounded set is the lowest-`consensus_rank` players.
- `test_filter_breakout_candidates_is_total` — empty pool + missing-column pool handled gracefully.

> A wall-clock test was rejected: it called the real service against `load_player_pool()`,
> which needs the 26MB main-checkout DB — absent in the worktree/CI → 0 rows → false failure.
> A count cap is the real regression guard; timing is flaky and DB-dependent.

---

## Bug 2 — `?lens=sell` returns 0 rows (FIXED — it was a real wiring bug, NOT data-dependent)

**Verdict for the CMO:** real wiring bug. NOT "no HOT players" (there are 270+).

**Root cause:** `detect_sell_high_candidates` feeds each `season_stats` row to
`compute_sustainability_score`, which reads `ab` (hitter sample gate + BABIP) and
`xfip`/`stuff_plus` (pitcher regression signal). The service's `_load_season_stats`
query omitted those columns, so **every** score collapsed to the neutral 0.5 fallback
→ never below the `<0.45` sell cap → **always 0 rows, regardless of season**.
(Proven: scoring pool rows, which DO carry the columns, yields candidates; scoring the
service's narrow rows yields a degenerate 0.5 for all 2000 sampled players.)

**Fix:** expand `_load_season_stats` to add `ab, h, sf, fip, xfip, siera, stuff_plus`
(the same columns the production Streamlit Leaders page passes via `load_season_stats()`).
Sustainability now varies (339/2000 non-0.5). **Live: 1 candidate (Andrew Vaughn)** —
matches the Streamlit Leaders sell-high tab exactly.

**Guard test:** `test_sell_lens_maps_candidates` — fabricated engine output → asserts the
sell branch maps candidates to `OverallLeaderRow` stamped `tag='sell'` (deterministic, no DB).

### Known limitation / documented follow-up (owner's call, NOT done here)
The lens is thin (≈1 today) because `xwoba_delta` — the **primary** hitter regression
signal (2× weight in `compute_sustainability_score`) — lives only in the player pool, not
in `season_stats`. This is a pre-existing limitation shared with the Streamlit page, not
introduced here. Merging the pool's `xwoba_delta` into the season-stats frame surfaces
~129 candidates, but it diverges from the proven Streamlit path and the delta=3.0-clamped
sort surfaces fringe small-sample names at the top — an engine-tuning decision, deferred.

---

## Verification
- `tests/api/` 167 passed (worktree empty DB = CI-equivalent).
- ruff check + format clean.
- `api/openapi.json` byte-identical to origin/master (no contract change).
- Live (real DB): all 5 lenses return rows fast; breakout 2.35s, sell 1 row.
