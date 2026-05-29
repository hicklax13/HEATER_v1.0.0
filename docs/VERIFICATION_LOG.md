# HEATER Verification Log

Manual verification entries for non-test-suite-covered behavior. Each entry documents an observable change against a live running app.

---

## V-OAUTH-T1.21 — 2026-05-22

**Change:** OAuth decoupling — optimize click no longer force-refreshes Yahoo; explicit "Refresh Yahoo Data" sidebar button replaces it; 15s `ThreadPoolExecutor` timeout on `_get_cached` so a hung Yahoo call cannot block page render.

**Branch:** `feat/t1.21-oauth-decoupling` (commits `d65ebae`, `59df15b`, `f3fc680`, `8b782fd`, `50c6067`, `a142764`, `1494b1f`, `1bd81db`, `a28cb9c`, `7783631`)

**Before:** Optimize click took 159 seconds on the live page (per audit screenshot in brainstorming session 36109-1779488367, screenshot ss_0417hypnr captured 2026-05-22). Root cause: `yds.get_rosters(force_refresh=True)` at the top of the optimize handler triggered a Yahoo OAuth refresh that hung in the yfpy retry loop when the access token had expired.

**After (verified 2026-05-22 ~21:55):**

- **Optimize click on Today scope, Full mode, Risk Aversion 0.15:** Page-reported "Lineup loaded in 5.25s". Wall-clock from button click to results-visible: ~30s including network roundtrips and DOM render, but the internal compute timer (which excludes Streamlit rerun overhead) was 5.25s. **30× speedup vs the 159s baseline.**

- **"Refresh Yahoo Data" sidebar button:** Located in the context panel immediately after the Data Freshness card. Clicked successfully. The Team Strength source's `refresh_log` timestamp updated from `11:55 PM` (previous bootstrap) to `9:45 PM` (current refresh). No 159s hang.

- **Side effects:** None observed. Existing tabs (Start/Sit, Cat Analysis, Streaming) render normally. DCV "scores are all zero" warning observed once — that's a pre-existing issue tracked elsewhere (cached schedule data was from earlier bootstrap; running the Refresh button updated it).

**Tests covering this change (all passing):**
- `tests/test_oauth_decoupling.py::test_no_force_refresh_in_optimize_handler` — guards that `if optimize_clicked:` block never calls YDS methods with `force_refresh=True`
- `tests/test_oauth_decoupling.py::test_refresh_yahoo_button_present` — guards that the explicit button exists + invokes `force_refresh_all()`
- `tests/test_oauth_decoupling.py::test_get_cached_times_out_when_yahoo_hangs` — guards that `_get_cached` returns from the fallback within 20s when fetch_fn is slow (annotated with `@pytest.mark.timeout(25)`)

**Structural-invariants suite (broader regression check):** 168/168 PASS in 88s (`pytest tests/test_no_*.py tests/test_sf*.py tests/test_oauth_decoupling.py`).

**Yahoo data service suite:** 51/51 PASS.

**Spec reference:** `docs/superpowers/specs/2026-05-22-heater-v2-optimizer-design.md` Section 15 Batch G (T1.21).
**Plan reference:** `docs/superpowers/plans/2026-05-22-oauth-decoupling-p7.md`.
