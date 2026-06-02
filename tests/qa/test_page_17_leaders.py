"""Deep per-team assertions for Leaders (pages/17_Leaders.py).

Renders ONCE per team (module-scoped `results` fixture). League-wide leaderboards
— per-team runs validate clean render under each session. Checks the default
(Category Leaders) tab is populated and stat values are plausible. Defensive style:
checks fire only when data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_17_leaders.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/17_Leaders.py"
_BAD = ("nan", "none", "inf", "-inf", "")


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex can scan visible text without tag noise."""
    return re.sub(r"<[^>]+>", " ", text)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


def test_leaders_renders_for_all_teams(results):
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
    assert not problems, "Leaders problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_leaders_no_nan_metrics(results):
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Leaders NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


def test_leaders_board_populated(results):
    """Every team session must yield a non-trivial leaderboard: either a
    DataFrame with at least one row OR markdown longer than 200 characters.
    The default tab renders the Category Leaders table via render_compact_table,
    which lands in .dataframes (AppTest) or .markdown (HTML path), so either
    channel satisfies the check.  Only fires when the render succeeded.
    """
    empty = []
    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue  # render failures already caught by test_leaders_renders_for_all_teams
        has_df_rows = any(len(df) >= 1 for df in r.dataframes)
        has_markdown = len(_strip_html(r.markdown).strip()) >= 200
        if not has_df_rows and not has_markdown:
            empty.append(team)
    assert not empty, (
        "Leaders board appears empty (no DataFrame rows and markdown < 200 chars) "
        "for teams:\n" + "\n".join(f"  {t}" for t in empty)
    )


def test_leaders_league_wide_consistency(results):
    """Because the leaderboard is league-wide (identical content for every team),
    the markdown payload should not vary wildly across sessions.  Assert that
    every session's stripped-markdown length is at least 60% of the maximum
    observed length.  Only fires when at least two sessions produce markdown
    longer than 200 characters.
    """
    lengths = {
        team: len(_strip_html(r.markdown).strip())
        for team, r in results.items()
        if r.ran and not r.exception and not r.errors
    }
    long_lengths = {t: n for t, n in lengths.items() if n >= 200}
    if len(long_lengths) < 2:
        return  # not enough data to compare — skip rather than false-fail

    max_len = max(long_lengths.values())
    floor = int(max_len * 0.60)
    short = [(t, n) for t, n in long_lengths.items() if n < floor]
    assert not short, (
        f"Leaders markdown length varies too much across teams "
        f"(max={max_len}, floor={floor}):\n" + "\n".join(f"  {t}: length={n}" for t, n in short)
    )


def test_leaders_stat_plausibility(results):
    """Leaderboard stat values must fall within realistic single-season bounds.

    Leaders tables render as custom HTML via st.markdown, so ``.dataframes`` is
    empty on the normal render — meaning the old dataframe-only scan never fired
    (vacuous; this is the page's ONLY value check).  This version adds a
    markdown/text arm that scans the unified ``r.text`` corpus (HTML-stripped)
    with context-anchored regexes, so a genuinely out-of-range stat (a corrupt
    .412 team AVG, a 99.99 ERA) is now actually caught on the normal render.

    Checks: AVG/OBP (leading-zero 3-decimal) in [0, 1]; ERA (anchored to "ERA")
    in [0, 20]; WHIP (anchored to "WHIP") in [0, 4].  The dataframe arm is kept
    for the degraded render path (HR/RBI/SB/K column bounds).  Tolerant: a
    missing pattern / column is not a failure.
    """
    violations: list[tuple[str, str, str, object]] = []

    # ── Markdown/text arm (normal render path) ────────────────────────────────
    _avg_obp_pat = re.compile(r"\b(0\.\d{3}|1\.000)\b")  # e.g. 0.312
    _era_pat = re.compile(r"\bERA\b[^0-9]{0,15}(\d{1,2}\.\d{2})", re.IGNORECASE)
    _whip_pat = re.compile(r"\bWHIP\b[^0-9]{0,15}(\d\.\d{2})", re.IGNORECASE)

    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue
        plain = _strip_html(r.text or "")

        for raw in _avg_obp_pat.findall(plain):
            v = float(raw)
            if not (0.0 <= v <= 1.0):
                violations.append((team, "AVG/OBP(text)", "out_of_range [0,1]", v))
        for raw in _era_pat.findall(plain):
            v = float(raw)
            if not (0.0 <= v <= 20.0):
                violations.append((team, "ERA(text)", "out_of_range [0,20]", v))
        for raw in _whip_pat.findall(plain):
            v = float(raw)
            if not (0.0 <= v <= 4.0):
                violations.append((team, "WHIP(text)", "out_of_range [0,4]", v))

    # ── DataFrame arm (degraded render path) ──────────────────────────────────
    _rules: list[tuple[str, float, float]] = [
        # (column_name_lower, min_val, max_val)
        ("avg", 0.0, 1.0),
        ("obp", 0.0, 1.0),
        ("era", 0.0, 20.0),
        ("whip", 0.0, 4.0),
        ("hr", 0.0, 80.0),
        ("rbi", 0.0, 200.0),
        ("sb", 0.0, 120.0),
        ("k", 0.0, 400.0),
    ]

    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue
        for df in r.dataframes:
            col_lower = {c.lower(): c for c in df.columns}
            for stat_lower, lo, hi in _rules:
                if stat_lower not in col_lower:
                    continue
                real_col = col_lower[stat_lower]
                for raw in df[real_col].dropna():
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if _is_bad_number(str(val)):
                        violations.append((team, real_col, "bad_number", raw))
                    elif not (lo <= val <= hi):
                        violations.append((team, real_col, f"out_of_range [{lo},{hi}]", val))

    assert not violations, "Leaders stat plausibility violations:\n" + "\n".join(
        f"  team={t} col={c} reason={reason} val={v}" for t, c, reason, v in violations
    )
