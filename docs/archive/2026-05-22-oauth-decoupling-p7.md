# P7: OAuth Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 159-second compute hang documented in the v2 design spec (Section 18) by decoupling the optimize button from Yahoo OAuth `force_refresh=True`, adding an explicit "Refresh Yahoo Data" button, and adding timeout protection to all Yahoo API fetches.

**Architecture:** Two changes to the page (`pages/2_Line-up_Optimizer.py`): (1) remove the implicit `force_refresh=True` on optimize click, replacing it with a cached call that falls back to SQLite when Yahoo is unreachable; (2) add an explicit "Refresh Yahoo Data" button to the context sidebar so users can manually force-refresh when they actually want it. One change to `src/yahoo_data_service.py`: wrap `fetch_fn()` in `_get_cached()` with a `ThreadPoolExecutor` timeout so a hanging Yahoo call (e.g. OAuth retry loop) cannot block the page indefinitely. Plus three structural-invariant guards and a CLAUDE.md update.

**Tech Stack:** Python 3.14, Streamlit, pytest, `concurrent.futures.ThreadPoolExecutor`, existing HEATER YahooDataService.

**Source spec:** `docs/superpowers/specs/2026-05-22-heater-v2-optimizer-design.md` (T1.21, Batch G, Phase 1)

---

## File Structure

**Modify (3 files):**
- `pages/2_Line-up_Optimizer.py` — remove `force_refresh=True` at line 846; add "Refresh Yahoo Data" button in context sidebar
- `src/yahoo_data_service.py` — add `ThreadPoolExecutor`-based timeout to `_get_cached`
- `CLAUDE.md` — add gotcha entry + structural-invariant guard row

**Create (1 file):**
- `tests/test_oauth_decoupling.py` — three structural-invariant tests + one timeout behavior test

**No deletions. No new dependencies (concurrent.futures is stdlib).**

---

## Tasks

### Task 1: Create the structural-invariant test file with the first guard

**Files:**
- Create: `tests/test_oauth_decoupling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_oauth_decoupling.py`:

```python
"""T1.21 / Batch G: OAuth decoupling structural-invariant guards.

These tests enforce the architectural constraint that the optimize button
in pages/2_Line-up_Optimizer.py does NOT trigger a forced Yahoo API refresh.
Forcing refresh on every optimize click was the root cause of the 159-second
compute hang documented in docs/superpowers/specs/2026-05-22-heater-v2-optimizer-design.md.

Decoupling makes optimize fast (uses cached data) and gives users explicit
control via the "Refresh Yahoo Data" button.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"
YDS_PATH = REPO_ROOT / "src" / "yahoo_data_service.py"


def _find_optimize_clicked_blocks(tree: ast.Module) -> list[ast.If]:
    """Return all `if optimize_clicked:` blocks in the parsed page."""
    blocks: list[ast.If] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "optimize_clicked":
                blocks.append(node)
    return blocks


def test_no_force_refresh_in_optimize_handler():
    """The `if optimize_clicked:` block must NOT call any YahooDataService
    method with force_refresh=True.

    Per T1.21 (Batch G of HEATER v2). Root cause: yds.get_rosters(
    force_refresh=True) at the top of the optimize handler triggers a
    Yahoo OAuth refresh that can hang ~150s in the OAuth retry loop when
    the access token has expired.
    """
    assert PAGE_PATH.exists(), f"Page not found: {PAGE_PATH}"
    source = PAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    blocks = _find_optimize_clicked_blocks(tree)
    assert len(blocks) >= 1, (
        "Could not find `if optimize_clicked:` block in "
        f"{PAGE_PATH} — page structure may have changed"
    )

    violations: list[str] = []
    for block in blocks:
        for sub in ast.walk(block):
            if isinstance(sub, ast.Call):
                for kw in sub.keywords:
                    if kw.arg == "force_refresh" and isinstance(kw.value, ast.Constant):
                        if kw.value.value is True:
                            violations.append(
                                f"line {sub.lineno}: {ast.unparse(sub)}"
                            )

    assert not violations, (
        "force_refresh=True found inside `if optimize_clicked:` block — "
        "violates T1.21. Move forced refreshes to the explicit "
        "'Refresh Yahoo Data' button. Violations:\n  "
        + "\n  ".join(violations)
    )
```

- [ ] **Step 2: Run test to verify it fails (current code is broken)**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_no_force_refresh_in_optimize_handler -v`

Expected: **FAIL** with a message containing
```
force_refresh=True found inside `if optimize_clicked:` block ...
line 846: yds.get_rosters(force_refresh=True)
```

This proves the bug exists and the test catches it.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_oauth_decoupling.py
git commit -m "test(t1.21): add structural-invariant guard for OAuth decoupling

Failing now — catches yds.get_rosters(force_refresh=True) on line 846
of pages/2_Line-up_Optimizer.py that causes the 159s OAuth hang on
every optimize click. Will pass after the page fix lands."
```

---

### Task 2: Fix `pages/2_Line-up_Optimizer.py` — remove `force_refresh=True`

**Files:**
- Modify: `pages/2_Line-up_Optimizer.py:844-854`

- [ ] **Step 1: Replace the offending line**

In `pages/2_Line-up_Optimizer.py`, find this block (around line 844-854):

```python
            # Force roster refresh to ensure current players only
            try:
                yds.get_rosters(force_refresh=True)
                roster = get_team_roster(user_team_name)
                if "name" in roster.columns and "player_name" not in roster.columns:
                    roster = roster.rename(columns={"name": "player_name"})
                for _sc in ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip", "ip", "pa"]:
                    if _sc in roster.columns:
                        roster[_sc] = pd.to_numeric(roster[_sc], errors="coerce").fillna(0)
            except Exception:
                pass
```

Replace with:

```python
            # T1.21: use cached roster data. Users can force-refresh via the
            # "Refresh Yahoo Data" button in the sidebar. Implicit force_refresh
            # here caused a 159s OAuth hang when the access token expired.
            try:
                yds.get_rosters()
                roster = get_team_roster(user_team_name)
                if "name" in roster.columns and "player_name" not in roster.columns:
                    roster = roster.rename(columns={"name": "player_name"})
                for _sc in ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip", "ip", "pa"]:
                    if _sc in roster.columns:
                        roster[_sc] = pd.to_numeric(roster[_sc], errors="coerce").fillna(0)
            except Exception:
                pass
```

The ONLY change is removing `force_refresh=True` from the `yds.get_rosters()` call and updating the inline comment.

- [ ] **Step 2: Run the test to verify it passes**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_no_force_refresh_in_optimize_handler -v`

Expected: **PASS**.

- [ ] **Step 3: Commit the fix**

```bash
git add pages/2_Line-up_Optimizer.py
git commit -m "fix(optimizer): drop force_refresh=True from optimize click (T1.21)

Closes the 159s OAuth hang documented in the v2 design spec.
Optimize now uses cached Yahoo data — refresh is explicit via the
new 'Refresh Yahoo Data' button added in the next commit."
```

---

### Task 3: Add structural-invariant test for the "Refresh Yahoo Data" button

**Files:**
- Modify: `tests/test_oauth_decoupling.py` (add second test)

- [ ] **Step 1: Append the new test**

Append to `tests/test_oauth_decoupling.py`:

```python
def test_refresh_yahoo_button_present():
    """`pages/2_Line-up_Optimizer.py` must contain an explicit button to
    force-refresh Yahoo data. Replaces the implicit force_refresh=True
    that used to live in the optimize click handler (T1.21).

    The button must invoke YahooDataService.force_refresh_all() so that
    rosters, standings, matchup, free agents, transactions, settings, and
    schedule are all refreshed in one click.
    """
    assert PAGE_PATH.exists(), f"Page not found: {PAGE_PATH}"
    source = PAGE_PATH.read_text(encoding="utf-8")

    # Look for a Streamlit button referencing "Refresh Yahoo"
    assert "Refresh Yahoo Data" in source, (
        "pages/2_Line-up_Optimizer.py must contain an explicit "
        "'Refresh Yahoo Data' button (T1.21). The button replaces the "
        "implicit force_refresh=True that was dropped from the optimize "
        "click handler."
    )

    # Verify the button calls force_refresh_all (the YDS method that
    # refreshes all 7 data types in one shot)
    assert "force_refresh_all()" in source, (
        "The 'Refresh Yahoo Data' button must call yds.force_refresh_all() "
        "so all 7 data caches are invalidated, not just one."
    )
```

- [ ] **Step 2: Run test, verify it FAILS (button doesn't exist yet)**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_refresh_yahoo_button_present -v`

Expected: **FAIL** with `"Refresh Yahoo Data" in source` AssertionError.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_oauth_decoupling.py
git commit -m "test(t1.21): add guard for explicit Refresh Yahoo Data button"
```

---

### Task 4: Add the "Refresh Yahoo Data" button to the context sidebar

**Files:**
- Modify: `pages/2_Line-up_Optimizer.py` (around line 791 — just after the Data Freshness card)

- [ ] **Step 1: Locate the insertion point**

In `pages/2_Line-up_Optimizer.py`, the context panel renders the Data Freshness card via `render_context_card("Data Freshness", _freshness_html)` near line 791. Insert the new button immediately AFTER that line.

Search for the existing line:
```python
    render_context_card("Data Freshness", _freshness_html)
```

- [ ] **Step 2: Add the button**

Add immediately after the `render_context_card("Data Freshness", _freshness_html)` line:

```python

    # T1.21: explicit refresh button replaces the implicit force_refresh=True
    # that used to live in the optimize click path. Users now control when
    # Yahoo is force-refreshed; optimize uses cached data otherwise.
    if st.button(
        "🔄 Refresh Yahoo Data",
        key="lineup_refresh_yahoo",
        help=(
            "Force-refresh rosters, standings, matchup, free agents, and "
            "transactions from Yahoo. Use when you've made changes in the "
            "Yahoo app (added/dropped a player, set lineup, etc.) and want "
            "HEATER to see them immediately. Otherwise leave alone — "
            "optimize uses cached data."
        ),
        width="stretch",
    ):
        with st.spinner("Refreshing Yahoo data (rosters, standings, matchup, FAs)..."):
            try:
                from src.yahoo_data_service import get_yahoo_data_service

                _refresh_results = get_yahoo_data_service().force_refresh_all()
                _ok = sum(1 for v in _refresh_results.values() if v == "Refreshed")
                _total = len(_refresh_results)
                if _ok == _total:
                    st.success(f"Refreshed {_ok}/{_total} Yahoo data types.")
                else:
                    _problems = {
                        k: v for k, v in _refresh_results.items() if v != "Refreshed"
                    }
                    st.warning(
                        f"Refreshed {_ok}/{_total}. Issues: {_problems}"
                    )
            except Exception as _refresh_err:
                st.error(f"Refresh failed: {_refresh_err}")
        st.rerun()
```

The emoji 🔄 is intentional here per Streamlit button convention — it's a sidebar control, not a data file. (CLAUDE.md "no emoji" rule applies to data files and inline content; sidebar action buttons commonly use them.)

Actually — re-read CLAUDE.md "No emoji" gotcha: "No emoji — All icons are inline SVGs from PAGE_ICONS. Injury badges use CSS dots." That applies app-wide. So drop the emoji.

Use this version instead (no emoji):

```python

    # T1.21: explicit refresh button replaces the implicit force_refresh=True
    # that used to live in the optimize click path. Users now control when
    # Yahoo is force-refreshed; optimize uses cached data otherwise.
    if st.button(
        "Refresh Yahoo Data",
        key="lineup_refresh_yahoo",
        help=(
            "Force-refresh rosters, standings, matchup, free agents, and "
            "transactions from Yahoo. Use when you've made changes in the "
            "Yahoo app (added/dropped a player, set lineup, etc.) and want "
            "HEATER to see them immediately. Otherwise leave alone — "
            "optimize uses cached data."
        ),
        width="stretch",
    ):
        with st.spinner("Refreshing Yahoo data (rosters, standings, matchup, FAs)..."):
            try:
                from src.yahoo_data_service import get_yahoo_data_service

                _refresh_results = get_yahoo_data_service().force_refresh_all()
                _ok = sum(1 for v in _refresh_results.values() if v == "Refreshed")
                _total = len(_refresh_results)
                if _ok == _total:
                    st.success(f"Refreshed {_ok}/{_total} Yahoo data types.")
                else:
                    _problems = {
                        k: v for k, v in _refresh_results.items() if v != "Refreshed"
                    }
                    st.warning(
                        f"Refreshed {_ok}/{_total}. Issues: {_problems}"
                    )
            except Exception as _refresh_err:
                st.error(f"Refresh failed: {_refresh_err}")
        st.rerun()
```

- [ ] **Step 3: Run test, verify it PASSES**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_refresh_yahoo_button_present -v`

Expected: **PASS**.

- [ ] **Step 4: Commit**

```bash
git add pages/2_Line-up_Optimizer.py
git commit -m "feat(optimizer): add explicit Refresh Yahoo Data button (T1.21)

Replaces the implicit force_refresh=True that was dropped from the
optimize click handler. Users now control when Yahoo is force-
refreshed; optimize uses cached data otherwise. Button invokes
force_refresh_all() so all 7 data caches refresh in one click."
```

---

### Task 5: Add timeout test for `_get_cached`

**Files:**
- Modify: `tests/test_oauth_decoupling.py` (append third test)

- [ ] **Step 1: Append the timeout test**

Append to `tests/test_oauth_decoupling.py`:

```python
import time
from unittest.mock import MagicMock, patch


def test_get_cached_times_out_when_yahoo_hangs():
    """YahooDataService._get_cached must NOT hang forever when fetch_fn
    is slow (T1.21 root cause: yfpy OAuth retry loop can take 100s+).

    When fetch_fn exceeds the 15-second budget, _get_cached must fall
    back to the db_fallback_fn return value rather than blocking the
    page render.
    """
    import pandas as pd

    from src.yahoo_data_service import YahooDataService

    # Mock client that pretends to be connected
    mock_client = MagicMock()
    mock_client.is_authenticated = True

    service = YahooDataService.__new__(YahooDataService)
    service._client = mock_client
    service._stats = MagicMock()
    service._PREFIX = "_yds_test_"

    # Slow fetch (would take 60s)
    def slow_fetch():
        time.sleep(60)
        return pd.DataFrame({"team_name": ["Hung"]})

    # Fast fallback
    def fast_fallback():
        return pd.DataFrame({"team_name": ["Cached"]})

    start = time.time()
    result = service._get_cached(
        key="test_hang",
        ttl=300,
        fetch_fn=slow_fetch,
        db_fallback_fn=fast_fallback,
        force=True,  # force=True so we definitely call fetch_fn
    )
    elapsed = time.time() - start

    assert elapsed < 20, (
        f"_get_cached blocked for {elapsed:.1f}s when fetch_fn was slow. "
        f"Expected <20s (15s budget + 5s grace). Timeout protection is missing."
    )
    assert not result.empty, "Expected fallback DataFrame, got empty"
    assert result.iloc[0]["team_name"] == "Cached", (
        f"Expected fallback 'Cached', got '{result.iloc[0]['team_name']}'"
    )
```

- [ ] **Step 2: Run test, verify it FAILS (no timeout yet)**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_get_cached_times_out_when_yahoo_hangs -v`

Expected: **FAIL** — the test runs for 60+ seconds (or hits its own assertion `elapsed < 20`).

This proves the timeout is missing.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_oauth_decoupling.py
git commit -m "test(t1.21): add timeout guard for _get_cached when Yahoo hangs"
```

---

### Task 6: Add `ThreadPoolExecutor` timeout to `_get_cached`

**Files:**
- Modify: `src/yahoo_data_service.py:169-220` (the `_get_cached` method)

- [ ] **Step 1: Add the import**

In `src/yahoo_data_service.py`, near the top of the file alongside other stdlib imports, add:

```python
import concurrent.futures
```

(If `concurrent.futures` is already imported, skip this step.)

- [ ] **Step 2: Locate the `_get_cached` method**

It starts at approximately line 169:

```python
    def _get_cached(
        self,
        key: str,
        ttl: int,
        fetch_fn: Callable[[], Any],
        db_fallback_fn: Callable[[], Any],
        sync_fn: Callable[[Any], None] | None = None,
        force: bool = False,
    ) -> Any:
```

- [ ] **Step 3: Wrap the `fetch_fn()` call in a timeout**

Find this section inside `_get_cached`:

```python
        # Tier 2: Yahoo API (live)
        if self.is_connected():
            try:
                data = fetch_fn()
                # Validate we got real data
                if self._is_valid_data(data):
                    # Cache in session_state
                    store[cache_key] = _CacheEntry(value=data, ttl=ttl)
```

Replace with:

```python
        # Tier 2: Yahoo API (live, with 15s timeout — T1.21).
        # Wraps fetch_fn in a ThreadPoolExecutor so a hanging Yahoo call
        # (e.g. yfpy OAuth retry loop when the access token expired)
        # cannot block the page render indefinitely. On timeout we fall
        # through to Tier 3 (SQLite fallback).
        if self.is_connected():
            data = None
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_fn)
                try:
                    data = future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "yahoo_data_service._get_cached: Yahoo fetch for '%s' "
                        "timed out after 15s — falling back to SQLite cache. "
                        "If this persists, the OAuth token may have expired; "
                        "use the 'Refresh Yahoo Data' button after re-auth.",
                        key,
                    )
                    future.cancel()
                    data = None
                except Exception as exc:
                    logger.warning(
                        "yahoo_data_service._get_cached: Yahoo fetch for '%s' "
                        "raised %s — falling back to SQLite cache.",
                        key, type(exc).__name__,
                    )
                    data = None

            if data is not None and self._is_valid_data(data):
                # Cache in session_state
                store[cache_key] = _CacheEntry(value=data, ttl=ttl)
```

The rest of the method (`self._stats.record_yahoo_fetch(...)` etc.) stays the same.

**Important:** keep the existing fall-through to Tier 3 (db_fallback_fn) intact — that's what catches the timeout case.

- [ ] **Step 4: Run the timeout test — verify it PASSES**

Run: `python -m pytest tests/test_oauth_decoupling.py::test_get_cached_times_out_when_yahoo_hangs -v`

Expected: **PASS** in <20 seconds.

- [ ] **Step 5: Run the full existing YDS test suite to verify no regressions**

Run: `python -m pytest tests/test_yahoo_data_service.py -v`

Expected: all existing tests still pass. If any fail, the timeout wrapper is interfering with the cached path — debug before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/yahoo_data_service.py
git commit -m "feat(yahoo): timeout protection on _get_cached fetch_fn (T1.21)

Wraps fetch_fn in ThreadPoolExecutor with 15s timeout. When Yahoo
hangs (e.g. OAuth retry loop on expired token), we fall through to
the SQLite Tier-3 cache instead of blocking the page.

Closes the 'optimize click hangs forever' part of the 159s issue
in addition to the force_refresh removal in pages/2_Line-up_Optimizer.py."
```

---

### Task 7: Run the FULL `test_oauth_decoupling.py` suite together

- [ ] **Step 1: Run all four tests**

Run: `python -m pytest tests/test_oauth_decoupling.py -v`

Expected output:
```
tests/test_oauth_decoupling.py::test_no_force_refresh_in_optimize_handler PASSED
tests/test_oauth_decoupling.py::test_refresh_yahoo_button_present PASSED
tests/test_oauth_decoupling.py::test_get_cached_times_out_when_yahoo_hangs PASSED
================ 3 passed in <20s ================
```

(Note: the file has 3 tests, not 4 — I miscounted in the plan header. That's correct.)

- [ ] **Step 2: Run the broader structural-invariant suite to confirm no regressions**

Run: `python -m pytest tests/test_no_*.py tests/test_sf*.py tests/test_oauth_decoupling.py -v 2>&1 | tail -30`

Expected: all green. No newly-broken tests.

- [ ] **Step 3: No commit needed** (just a verification run)

---

### Task 8: Update CLAUDE.md — new gotcha + structural-invariant guard

**Files:**
- Modify: `CLAUDE.md` (two locations — gotchas section and structural invariants table)

- [ ] **Step 1: Add the new gotcha**

Find the "Gotchas" section, sub-section "Yahoo API". After the existing line that mentions `Auto-reconnect — \`_try_reconnect_yahoo()\``, add this new bullet:

```markdown
- **Optimize click uses cached Yahoo data (T1.21)** — `pages/2_Line-up_Optimizer.py`'s optimize handler does NOT force-refresh Yahoo. Users force-refresh via the "Refresh Yahoo Data" sidebar button. This decoupling closed the 159-second OAuth hang documented in the v2 design spec. `_get_cached` in `src/yahoo_data_service.py` also wraps every Yahoo fetch in a 15-second `ThreadPoolExecutor` timeout so a hung Yahoo call cannot block page rendering. Structurally guarded by `tests/test_oauth_decoupling.py`.
```

- [ ] **Step 2: Add the structural-invariant guard row**

Find the "Structural Invariants (machine-checked)" table. Add this row near the end (before the "FA Engine Overhaul guards" sub-section):

```markdown
| `test_oauth_decoupling.py` | `pages/2_Line-up_Optimizer.py`'s optimize click handler must not call YahooDataService methods with `force_refresh=True`; the "Refresh Yahoo Data" button must be present and invoke `force_refresh_all()`; `_get_cached` must time out at 15s when Yahoo hangs. T1.21 / Batch G of v2. |
```

- [ ] **Step 3: Verify the changes parse**

Run: `python -c "from pathlib import Path; print('CLAUDE.md OK:', len(Path('CLAUDE.md').read_text(encoding='utf-8')))"`

Expected: prints `CLAUDE.md OK: <number>` — confirms the file is readable as UTF-8.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document T1.21 OAuth decoupling + new guard

Adds a Yahoo gotcha entry explaining that optimize click uses cached
data + the new Refresh button + the 15s timeout. Adds the
test_oauth_decoupling.py row to the structural-invariants table."
```

---

### Task 9: Manual verification — run streamlit and time an optimize click

This step does NOT have a Pytest test; it verifies the actual user-facing behavior.

- [ ] **Step 1: Start Streamlit**

Run: `streamlit run app.py --server.headless=true --server.port=8501`

Wait for `You can now view your Streamlit app in your browser. Local URL: http://localhost:8501`.

- [ ] **Step 2: Open the Line-up Optimizer page in a browser**

Navigate to `http://localhost:8501/Line-up_Optimizer`.

- [ ] **Step 3: Click "Optimize Lineup" — time it**

With a stopwatch or by checking the "Lineup loaded in Xs" footer text:

Expected: **<10 seconds on warm cache**. Before this plan, it was 159 seconds.

- [ ] **Step 4: Click the "Refresh Yahoo Data" sidebar button**

Expected: spinner shows for a few seconds, then either:
- `Refreshed 7/7 Yahoo data types.` (success)
- `Refreshed N/7. Issues: {...}` (partial — common if OAuth expired; user re-auths and tries again)
- `Refresh failed: ...` (full failure — actionable error message)

Page should NOT hang for 159 seconds in any case.

- [ ] **Step 5: Document the timing in a verification log**

Append the following to `docs/VERIFICATION_LOG.md` (create the file if missing):

```markdown
## V-OAUTH-T1.21 — 2026-05-22

**Change:** OAuth decoupling — optimize click no longer force-refreshes Yahoo; explicit "Refresh Yahoo Data" sidebar button replaces it; 15s timeout on `_get_cached`.

**Before:** Optimize click took 159s on the live page (per audit screenshot in brainstorming session 36109-1779488367, screenshot ss_0417hypnr).

**After:** Optimize click took <RECORD ACTUAL TIME>s on warm cache.

**Refresh button:** <RECORD result of force_refresh_all() — N/7 refreshed>

**Side effects:** None observed. Existing tabs (Start/Sit, Cat Analysis, Streaming) all render normally.
```

Fill in the `<RECORD ...>` placeholders with actual observed numbers.

- [ ] **Step 6: Commit the verification log**

```bash
git add docs/VERIFICATION_LOG.md
git commit -m "docs(verification): V-OAUTH-T1.21 — confirm <10s optimize click"
```

---

### Task 10: Final integration — run the full test suite

- [ ] **Step 1: Run the full pytest suite**

Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q 2>&1 | tail -50`

Expected: **all tests pass** (modulo the existing 15 skipped that depend on optional libraries). The `-x` flag stops on first failure so we catch regressions immediately.

If any test fails:
1. **Read the failure carefully** — it's likely a test that mocked YDS in a way that conflicts with the new ThreadPoolExecutor wrapping
2. **Check `tests/test_yahoo_data_service.py`** — most likely candidate for impact
3. **Fix the mock setup** — the mock probably needs to be patched at a different level OR the test needs to allow the thread to complete

- [ ] **Step 2: Run ruff lint + format**

Run: `python -m ruff check . && python -m ruff format --check .`

Expected: no errors. If errors, run `python -m ruff format .` to auto-fix.

- [ ] **Step 3: Final celebratory commit if there are any incidental fixes**

If steps 1-2 required tweaks beyond what's already committed, commit them:

```bash
git add -u
git commit -m "chore(t1.21): final lint + test cleanup"
```

If no changes are pending, skip this step.

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a "milestone/t1.21-oauth-decoupling-complete" -m "T1.21 / Batch G: OAuth decoupling complete. Optimize click <10s, explicit Refresh Yahoo Data button, 15s timeout on _get_cached."
```

(No push — user controls remote interactions.)

---

## Self-Review

Spec coverage check (Section 15 Batch G of `docs/superpowers/specs/2026-05-22-heater-v2-optimizer-design.md`):
- ✅ T1.21: "Decouple `yds.get_rosters(force_refresh=True)` from optimize click path" → Task 2 removes it
- ✅ T1.21: "Use cached data unless explicit 'Refresh Yahoo' pressed" → Task 4 adds the explicit button
- ✅ T1.21: "Eliminates the 159s hang documented in the audit" → Tasks 6 + 9 add the timeout protection + verify the fix

Placeholder scan:
- No "TBD" / "TODO" / "implement later" markers
- All code blocks contain runnable code
- All test assertions are concrete
- All commit messages are complete

Type / signature consistency:
- `force_refresh_all()` referenced in Task 4 is the actual method on YDS (verified in `src/yahoo_data_service.py:502`)
- `_get_cached` signature in Task 6 matches the actual method (`key, ttl, fetch_fn, db_fallback_fn, sync_fn=None, force=False`)
- Test file path consistent across all tasks (`tests/test_oauth_decoupling.py`)

Risks:
- The `ThreadPoolExecutor` adds ~1-2ms overhead per Yahoo call. Negligible at HEATER's request rates.
- `future.cancel()` is non-blocking for Python threads — the underlying thread keeps running for its 60s sleep in the test. This is fine; the test only cares that the main path returns fast.
- If `is_connected()` raises (rare — OAuth probe), the existing `try/except` two levels up handles it.

---

## Estimated Effort

**Total: ~4 hours** (single focused session).

| Task | Estimate |
|---|---|
| Task 1: First test (failing) | 20 min |
| Task 2: Fix force_refresh | 10 min |
| Task 3: Second test | 15 min |
| Task 4: Button | 25 min |
| Task 5: Timeout test | 20 min |
| Task 6: Timeout impl | 30 min |
| Task 7: Combined verification | 10 min |
| Task 8: CLAUDE.md | 15 min |
| Task 9: Manual verification + timing | 30 min |
| Task 10: Final suite + tag | 20 min |

Buffer (debugging, edge cases): 45 min.

---

*End of plan — 2026-05-22-oauth-decoupling-p7.md*
