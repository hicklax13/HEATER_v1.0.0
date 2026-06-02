"""Deep per-team assertions for Lineup Optimizer (pages/2_Line-up_Optimizer.py).

Renders the page ONCE per team (module-scoped `results` fixture) and checks
value-plausibility on the DEFAULT render (no optimize button is clicked
headlessly). Defensive style: plausibility checks fire only when the data is
present, so a capture-channel mismatch never causes a false failure.

On default render the page shows:
  - Context panel: team name, roster active/IL count, matchup state, IP budget
    (all via st.markdown / render_context_card → HTML in .markdown)
  - Optimizer tab: scope selector + "Optimize Lineup" button (not clicked)
  - Start/Sit tab: "Run the optimizer first…" prompt + manual player multiselect
  - The IP-budget block ("X / 54 IP") is rendered in .markdown when ip_tracker
    is available; it may not be present when the ip_tracker module is absent.

Most data flows through render_compact_table / render_context_card which both
emit HTML via st.markdown.  .dataframes and .metrics may be sparse or empty on
the default render — assertions guard against that.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_2_lineup_optimizer.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/2_Line-up_Optimizer.py"
_BAD = ("nan", "none", "inf", "-inf", "")


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex matching hits actual displayed numbers."""
    return re.sub(r"<[^>]+>", " ", text)


# Patterns that may appear in .markdown on the default render
# IP budget: "39.2 / 54 IP" or "Post-LP Starters 42.0 / 54 IP (78%)"
_RE_IP_BUDGET = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*/\s*54\b", re.IGNORECASE)
# Active roster count from context card: "24 active" or "22 active + 3 IL"
_RE_ACTIVE_COUNT = re.compile(r"\b(\d{1,2})\s+active\b", re.IGNORECASE)
# ERA value (rendered via format_stat as "N.NN")
_RE_ERA = re.compile(r"\bERA\b.*?(\d{1,2}\.\d{2})", re.IGNORECASE)
# WHIP value
_RE_WHIP = re.compile(r"\bWHIP\b.*?(\d{1,2}\.\d{2})", re.IGNORECASE)
# DCV Score values (rendered as integers or decimals in the optimizer table,
# e.g. "12.34" or "0.00")
_RE_DCV = re.compile(r"\bDCV\b.*?(-?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render Lineup Optimizer once per team; reused by every assertion."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


def test_lineup_renders_for_all_teams(results):
    """Per team: ran, no exception, no st.error, and SOMETHING rendered (blank-page guard)."""
    problems = []
    for team, r in results.items():
        if not r.ran:
            problems.append((team, "did-not-run", r.exception))
            continue
        if r.exception:
            problems.append((team, "exception", r.exception))
            continue
        if r.errors:
            problems.append((team, "st.error", r.errors))
            continue
        if not r.dataframes and not r.metrics and len(r.markdown) < 50:
            problems.append((team, "blank", "no dataframes/metrics/markdown"))
    assert not problems, "Lineup Optimizer problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_lineup_no_nan_metrics(results):
    """No displayed st.metric value is NaN/None/inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Lineup Optimizer NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


def test_lineup_roster_dataframe_plausible(results):
    """When a roster/lineup dataframe is captured, row count ∈ [10, 40] and no all-NaN columns.

    The page routes most roster data through render_compact_table (HTML via
    st.markdown) so .dataframes is often empty on the default render.  This
    assertion fires ONLY when a DataFrame is actually captured — a missing
    capture channel is not a failure.  We identify roster-like frames by
    looking for a name-like column ('Player', 'name', 'player_name').
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for i, df in enumerate(r.dataframes):
            if df is None or not hasattr(df, "__len__"):
                continue
            nrows = len(df)
            has_name_col = any(str(c).lower() in ("player", "name", "player_name") for c in df.columns)
            if not has_name_col:
                continue  # not a roster frame — skip
            if not (10 <= nrows <= 40):
                bad.append((team, f"df[{i}]", f"roster row count {nrows} outside [10, 40]"))
                continue
            for col in df.columns:
                if df[col].isna().all():
                    bad.append((team, f"df[{i}].{col}", "all-NaN column"))
    assert not bad, "Lineup Optimizer dataframe problems:\n" + "\n".join(f"  {b}" for b in bad)


def test_lineup_ip_budget_plausible(results):
    """When an IP-budget value (N / 54 IP) appears in .markdown, the numerator ∈ [0, 120].

    The ip_budget block is rendered only when src.ip_tracker is available and
    the roster has pitchers.  When absent the test passes vacuously.  When
    present, values above 120 or below 0 indicate a projection or scaling bug.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        plain = _strip_html(r.markdown)
        for m in _RE_IP_BUDGET.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (0.0 <= val <= 120.0):
                bad.append((team, "IP budget numerator", raw, "outside [0, 120]"))
    assert not bad, "Lineup Optimizer IP budget out of range:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )


def test_lineup_rate_stats_and_roster_count_plausible(results):
    """Scan .markdown for ERA, WHIP, active-roster count, and DCV scores.

    Plausible ranges:
      Active roster count: [10, 40]  (28 FourzynBurn slots; window absorbs IL filtering)
      ERA: [0, 20]
      WHIP: [0, 4]
      DCV scores (if present in HTML): [-50, 200]  (negative = hurts matchup; large
        positive reflects a well-matched SP/hitter; anything outside this range
        is a scaling or sign bug)

    Firing conditions: each assertion fires ONLY when its regex matches at least
    one value.  A page that renders no ERA text passes this test without issue.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        plain = _strip_html(r.markdown)

        # Active roster count
        for m in _RE_ACTIVE_COUNT.finditer(plain):
            raw = m.group(1)
            try:
                n = int(raw)
            except ValueError:
                continue
            if not (10 <= n <= 40):
                bad.append((team, "active_count", raw, "outside [10, 40]"))

        # ERA
        for m in _RE_ERA.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (0.0 <= val <= 20.0):
                bad.append((team, "ERA", raw, "outside [0, 20]"))

        # WHIP
        for m in _RE_WHIP.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (0.0 <= val <= 4.0):
                bad.append((team, "WHIP", raw, "outside [0, 4]"))

        # DCV scores (only fires when optimizer has run; skip on default render
        # where no DCV table is present — regex won't match)
        for m in _RE_DCV.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (-50.0 <= val <= 200.0):
                bad.append((team, "DCV_score", raw, "outside [-50, 200]"))

    assert not bad, "Lineup Optimizer out-of-range values:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )
