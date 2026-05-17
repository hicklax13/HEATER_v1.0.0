"""BUG-024 fix: Matchup Planner page surfaces a warning when falling back to demo roster.

This test catches the BUG-024 silent-fallback pattern: a `pool.head(...)`
demo-roster fallback MUST be immediately preceded (within 7 lines) by an
st.warning() / st.info() / st.error() call inside the SAME conditional
branch. The earlier ±5-line heuristic was too loose — it matched
unrelated alert calls in nearby exception handlers, so the test passed
pre-fix too.

Window rationale: the user-visible warning is a multi-line `st.warning(
   "..." )` block, so the `st.warning(` token sits a handful of lines
above `pool.head(...)`. 7 lines is wide enough to catch a 5-6 line
multiline call but narrow enough to exclude alerts in nearby
exception handlers (BUG-024's pre-fix code had `st.error(...)` ~12
lines above in a separate `except` branch — a 7-line window would not
have reached it, which is exactly the regression we want to catch).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "5_Matchup_Planner.py"

# Lines immediately preceding the pool.head(...) call that must contain
# an alert marker. Tight enough to reject alerts in unrelated nearby
# handlers; wide enough to catch a multi-line `st.warning("...")` block.
PRECEDING_WINDOW = 7


def test_demo_roster_fallback_preceded_by_user_warning():
    """For each `pool.head(...)` call in pages/5_Matchup_Planner.py, the
    7 lines immediately preceding it must include an `st.warning(` /
    `st.info(` / `st.error(` call. This forces the user-visible
    demo-data signal to live directly with the fallback, not in an
    unrelated nearby handler."""
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    lines = text.splitlines()
    offenders: list[tuple[int, str]] = []
    alert_markers = ("st.warning(", "st.info(", "st.error(")

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "pool.head(" not in line:
            continue
        # Look at the lines immediately preceding pool.head(, not including
        # the pool.head line itself. The window MUST NOT extend past
        # pool.head() — alerts AFTER the fallback don't warn the user
        # before they see the demo data. Comment lines are skipped so
        # commenting out the warning doesn't keep the test green.
        start = max(0, lineno - 1 - PRECEDING_WINDOW)
        end = lineno - 1  # exclusive of the pool.head line itself
        preceding_window = lines[start:end]
        found_alert = any(
            any(marker in pline for marker in alert_markers)
            for pline in preceding_window
            if not pline.strip().startswith("#")
        )
        if not found_alert:
            offenders.append((lineno, stripped))

    assert not offenders, (
        f"BUG-024 regression: pool.head(...) demo-roster fallback NOT "
        f"preceded by an st.warning/st.info/st.error within "
        f"{PRECEDING_WINDOW} lines above. Add a user-visible banner "
        f"directly before the fallback so users know they're seeing "
        f"demo data, not their real team. Offenders: {offenders}"
    )
