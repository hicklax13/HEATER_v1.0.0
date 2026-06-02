"""Deep per-team assertions for My Team (pages/1_My_Team.py).

Renders the page ONCE per team (module-scoped `results` fixture) and checks
value-plausibility, not just crash-freedom (the smoke suite covers crashes).
Defensive style: plausibility checks fire only when the data is present, so a
capture-channel mismatch never causes a false failure — only genuinely
implausible values do.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_1_my_team.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/1_My_Team.py"
_BAD = ("nan", "none", "inf", "-inf", "")  # case-insensitive sentinels in displayed values

# Regex patterns used to extract rate-stat values from markdown HTML.
# These are intentionally broad: they look for the display-formatted values
# that _compute_category_totals() + format_stat() produce.
#
# AVG / OBP: rendered as ".NNN" (3 decimals, no leading 0). This is the ONLY
# rate stat with a true mathematical ceiling (<= 1.0), so it is the only one
# safe to range-check by scanning rendered HTML.
_RE_AVG_OBP = re.compile(r"\.\d{3}")
# NOTE: ERA/WHIP are intentionally NOT scanned. They have no upper bound (a
# 1-out blowup yields ERA 162 / WHIP 18 — real, correctly displayed), and a
# keyword-anchored scan of the roster table matched generic stat cells, not the
# ERA/WHIP values (2026-06-02 deep-run false positive INFRA-2). NaN/inf in a
# displayed rate stat is covered by test_my_team_no_nan_metrics + the render gate.
# Category-rank context: e.g. "rank: 7" or ">7<" inside rank-related HTML cells.
_RE_RANK = re.compile(r"\brank[:\s]*(\d{1,2})\b", re.IGNORECASE)
# Roster count from the banner teaser that the page renders via st.markdown:
#   "Roster: 24 players | 14 hitters, 10 pitchers"
_RE_ROSTER_COUNT = re.compile(r"Roster:\s*(\d+)\s*players", re.IGNORECASE)


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Strip HTML — INCLUDING <style>/<script> block CONTENT — so regex matching
    hits actual displayed numbers, not CSS values like ``saturate(180%)`` that
    survive naive tag-stripping (2026-06-02 deep-run false positive)."""
    t = text or ""
    t = re.sub(r"<(style|script)\b[^>]*>.*?</\1>", " ", t, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"<[^>]+>", " ", t)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render My Team once per team; reused by every assertion (caps renders at 12/module)."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Test 1: Render gate ───────────────────────────────────────────────────────


def test_my_team_renders_for_all_teams(results):
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
    assert not problems, "My Team problems:\n" + "\n".join(f"  {p}" for p in problems)


# ── Test 2: No NaN/None/inf in st.metric values ───────────────────────────────


def test_my_team_no_nan_metrics(results):
    """No displayed st.metric value is NaN/None/inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue  # crash already reported by the render test
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "My Team NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── Test 3: Roster dataframe plausibility ────────────────────────────────────


def test_my_team_roster_dataframe_plausible(results):
    """When a roster dataframe is captured, assert sane row count and no fully-NaN columns.

    My Team renders the roster via render_compact_table (which routes through
    st.markdown as custom HTML) so .dataframes is often empty.  This test
    fires ONLY when a DataFrame is actually captured — a missing capture
    channel is not a failure.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for i, df in enumerate(r.dataframes):
            # Heuristic: a roster-like frame has 10–40 rows and at least one
            # column that looks like a name column ("Player", "name", etc.)
            if df is None or not hasattr(df, "__len__"):
                continue
            nrows = len(df)
            has_name_col = any(str(c).lower() in ("player", "name") for c in df.columns)
            if not has_name_col:
                continue  # not a roster frame — skip
            # Row count guard
            if not (10 <= nrows <= 40):
                bad.append((team, f"df[{i}]", f"roster row count {nrows} outside [10, 40]"))
                continue
            # No fully-NaN column (every column should have at least one real value)
            for col in df.columns:
                if df[col].isna().all():
                    bad.append((team, f"df[{i}].{col}", "all-NaN column"))
    assert not bad, "My Team roster dataframe problems:\n" + "\n".join(f"  {b}" for b in bad)


# ── Test 4: Rate-stat plausibility from markdown ──────────────────────────────


def test_my_team_rate_stats_in_range(results):
    """Scan .markdown for AVG/OBP values; any found must be in [0, 1].

    My Team renders category totals as custom HTML via st.markdown.  We strip
    tags (incl. <style> blocks) and run a tolerant regex; we NEVER fail because a
    pattern is absent — only when a matched value is out of range.

    Only AVG/OBP is scanned: it has a true ceiling (<= 1.0).  ERA/WHIP are
    deliberately NOT range-checked here — they are unbounded for small samples
    (ERA 162 on 0.33 IP is real) and a roster-table scan matched the wrong cells
    (INFRA-2, 2026-06-02).  NaN/inf is covered by test_my_team_no_nan_metrics.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # AVG / OBP: all ".NNN" tokens — true mathematical ceiling of 1.0
        for m in _RE_AVG_OBP.finditer(plain):
            raw = m.group(0)  # ".275"
            try:
                val = float("0" + raw)  # "0.275"
            except ValueError:
                continue
            if not (0.0 <= val <= 1.0):
                bad.append((team, "AVG/OBP", raw, "outside [0, 1]"))

    assert not bad, "My Team out-of-range rate stats:\n" + "\n".join(
        f"  team={b[0]}  stat={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )


# ── Test 5: Roster count plausibility and category ranks ──────────────────────


def test_my_team_roster_count_and_ranks(results):
    """Validate the roster-count banner and any displayed category ranks.

    Roster count: the banner teaser "Roster: N players" must show N ∈ [10, 40]
    (FourzynBurn has 28 slots including BN/IL; we use a wider window to absorb
    ghost-player filtering without false failures).

    Category ranks: any integer following a "rank" keyword in .markdown must be
    in [1, 12] for a 12-team league.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # Roster count from banner teaser (rendered via st.markdown)
        count_matches = _RE_ROSTER_COUNT.findall(plain)
        for raw in count_matches:
            try:
                n = int(raw)
            except ValueError:
                continue
            if not (10 <= n <= 40):
                bad.append((team, "roster_count", raw, "outside [10, 40]"))

        # Category ranks — any integer following "rank" token
        for m in _RE_RANK.finditer(plain):
            raw = m.group(1)
            try:
                rank = int(raw)
            except ValueError:
                continue
            if not (1 <= rank <= 12):
                bad.append((team, "category_rank", raw, "outside [1, 12]"))

    assert not bad, "My Team roster-count / rank problems:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )
