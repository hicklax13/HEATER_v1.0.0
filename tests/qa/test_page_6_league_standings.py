"""Deep per-team assertions for League Standings (pages/6_League_Standings.py).

Renders ONCE per team (module-scoped `results` fixture) and checks standings
plausibility (exactly 12 teams; ranks within 1..12; playoff odds in range).
League-wide page — per-team runs validate clean render under each session.
Defensive style: plausibility checks fire only when data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_6_league_standings.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/6_League_Standings.py"
_BAD = ("nan", "none", "inf", "-inf", "")


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex can scan visible text without tag noise."""
    return re.sub(r"<[^>]+>", " ", text)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ─── Core render guard ────────────────────────────────────────────────────────


def test_standings_renders_for_all_teams(results):
    """Every team session must produce a non-blank render with no exceptions."""
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
    assert not problems, "League Standings problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_standings_no_nan_metrics(results):
    """No metric value should be NaN, None, or inf."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "League Standings NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ─── Team count in standings HTML ─────────────────────────────────────────────


def test_standings_all_teams_appear_in_markdown(results, team_names):
    """When standings HTML is present, every league team name must appear at
    least once across the combined markdown output.  Only fires when the markdown
    is long enough to plausibly contain a standings table (>200 chars).
    """
    missing_by_team: list[tuple[str, list[str]]] = []

    for session_team, r in results.items():
        if not r.ran or r.exception:
            continue
        combined = _strip_html(r.markdown)
        if len(combined) < 200:
            # Not enough content to assert — skip rather than false-fail
            continue

        absent = [t for t in team_names if t not in combined]
        if absent:
            missing_by_team.append((session_team, absent))

    assert not missing_by_team, "League Standings: some team names absent from markdown output:\n" + "\n".join(
        f"  session={s}: missing {m}" for s, m in missing_by_team
    )


def test_standings_dataframe_row_count(results):
    """If a standings-shaped DataFrame is captured (10–14 rows, contains a
    team-like column), assert exactly 12 rows.  Tolerates frames that represent
    other tables (e.g. Finish Distribution with <10 rows) by skipping them.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for df in r.dataframes:
            # Identify standings frames by row count range and column presence
            if len(df) < 10 or len(df) > 14:
                continue  # not a standings table
            has_team_col = any(c.lower() in ("team", "team name", "team_name") for c in df.columns)
            if not has_team_col:
                continue  # not a standings table
            if len(df) != 12:
                bad.append((team, len(df), list(df.columns)))

    assert not bad, "League Standings DataFrames with team column but != 12 rows:\n" + "\n".join(
        f"  session={t}: rows={n} cols={c}" for t, n, c in bad
    )


# ─── Playoff odds range ───────────────────────────────────────────────────────


def test_standings_playoff_odds_in_range(results):
    """Any percentage value appearing near playoff/odds/probability context in
    markdown must fall in [0, 100].  Also checks metric values labelled with
    'playoff', 'odds', or 'probability'.  Only asserts when at least one value
    is found — never fails on absent patterns.
    """
    out_of_range: list[tuple[str, str, float]] = []

    # Regex: capture digits (and optional decimal) immediately before a '%'
    _pct_re = re.compile(r"(\d+(?:\.\d+)?)%")
    # Context words that indicate a playoff-probability value
    _ctx_keywords = re.compile(r"playoff|odds|probability|playoff\s*%|playoff_prob", re.IGNORECASE)

    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # Scan markdown: look for '%' values within a 120-char window that also
        # contains a context keyword (catches "Playoff Probability 72.3%", etc.)
        for m in _pct_re.finditer(plain):
            val = float(m.group(1))
            window_start = max(0, m.start() - 120)
            window = plain[window_start : m.end() + 60]
            if _ctx_keywords.search(window):
                if not (0.0 <= val <= 100.0):
                    out_of_range.append((team, f"markdown '{m.group(0)}'", val))

        # Scan metrics
        for met in r.metrics:
            label = str(met.get("label", "")).lower()
            if not _ctx_keywords.search(label):
                continue
            raw = str(met.get("value", "")).strip().rstrip("%")
            try:
                val = float(raw)
            except ValueError:
                continue
            # Accept fractions [0,1] or percentages [0,100]
            if 0.0 <= val <= 1.0:
                continue  # fractional probability — valid
            if 0.0 <= val <= 100.0:
                continue  # percentage — valid
            out_of_range.append((team, f"metric '{met.get('label')}'", val))

    assert not out_of_range, "League Standings playoff odds out of [0, 100]:\n" + "\n".join(
        f"  session={t}: source={s} value={v}" for t, s, v in out_of_range
    )


# ─── Rank values in [1, 12] ───────────────────────────────────────────────────


def test_standings_ranks_in_valid_range(results):
    """Integer values appearing in explicit rank/# columns in captured DataFrames
    must be within 1..12.  Also scans DataFrames for a 'Rank' column directly.
    Only fires when rank-shaped data is found.
    """
    bad_ranks: list[tuple[str, int]] = []

    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        for df in r.dataframes:
            rank_cols = [c for c in df.columns if str(c).lower() in ("rank", "#", "rank#")]
            for col in rank_cols:
                for raw_val in df[col].dropna():
                    try:
                        rank_int = int(raw_val)
                    except (ValueError, TypeError):
                        continue
                    if not (1 <= rank_int <= 12):
                        bad_ranks.append((team, rank_int))

    assert not bad_ranks, "League Standings rank values outside [1, 12]:\n" + "\n".join(
        f"  session={t}: rank={v}" for t, v in bad_ranks
    )
