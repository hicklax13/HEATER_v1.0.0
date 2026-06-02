"""Deep per-team assertions for Trade Finder (pages/12_Trade_Finder.py).

Renders ONCE per team (module-scoped `results` fixture). Several tabs compute on
interaction, so the headless default render is checked for clean load, non-blank
content, and (where present) trade-value plausibility (0-100 scale).
Defensive style: checks fire only when data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_12_trade_finder.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/12_Trade_Finder.py"
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


def test_trade_finder_renders_for_all_teams(results):
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
    assert not problems, "Trade Finder problems:\n" + "\n".join(f"  {p}" for p in problems)


# ─── No NaN/None/inf in metrics ───────────────────────────────────────────────


def test_trade_finder_no_nan_metrics(results):
    """No st.metric value should be NaN, None, or inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Trade Finder NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ─── Trade value chart: 0-100 range ──────────────────────────────────────────


def test_trade_finder_value_chart_in_range(results):
    """The integer immediately following each "Trade Value" label must be in [0, 100].

    The Value Chart renders tier groups each containing a "Trade Value" column
    with integers on the 0-100 universal scale.  We capture ONLY the first
    integer that immediately follows the "Trade Value" label and range-check that
    single value — the previous version scanned an 80-char window with
    ``\\b(\\d{1,3})\\b`` and range-checked EVERY integer it found, so adjacent
    columns (ADP 235, rank, K totals) were flagged as out-of-range trade values
    (a false-positive risk).

    Scans the unified ``r.text`` corpus (HTML-stripped) so the value is caught
    regardless of which channel rendered the table.  Defensive: if the Value
    Chart didn't render (no "Trade Value" label), the test passes vacuously.
    """
    # Capture the FIRST integer immediately after the label only (skip up to 8
    # non-digit chars of separators/markup), so we never wander into the next
    # column's value.
    _tv_value_re = re.compile(r"Trade\s+Value[^0-9]{0,8}(\d{1,3})", re.IGNORECASE)

    out_of_range: list[tuple[str, int]] = []

    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.text or "")
        for vm in _tv_value_re.finditer(plain):
            val = int(vm.group(1))
            if not (0 <= val <= 100):
                out_of_range.append((team, val))

    assert not out_of_range, "Trade Finder Value Chart: trade values outside [0, 100]:\n" + "\n".join(
        f"  session={t}: value={v}" for t, v in out_of_range
    )


# ─── Non-blank content per team ───────────────────────────────────────────────


def test_trade_finder_non_blank_per_team(results):
    """Every team that ran without error must have produced substantive output:
    either >=1 captured DataFrame, >=1 captured metric, or markdown >=200 chars.
    This catches the case where the page silently returns early (e.g. empty pool
    or no roster) but doesn't raise an exception.
    """
    thin: list[tuple[str, str]] = []
    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue  # already covered by render-guard test
        has_df = bool(r.dataframes)
        has_metric = bool(r.metrics)
        has_markdown = len(r.markdown) >= 200
        if not (has_df or has_metric or has_markdown):
            thin.append(
                (
                    team,
                    f"dfs={len(r.dataframes)} metrics={len(r.metrics)} md_len={len(r.markdown)}",
                )
            )
    assert not thin, "Trade Finder: teams with suspiciously thin renders:\n" + "\n".join(f"  {t}: {d}" for t, d in thin)


# ─── SGP gain values in sane range ───────────────────────────────────────────


def test_trade_finder_sgp_gain_sane(results):
    """If trade-recommendation DataFrames are captured with a 'Your Gain' or
    'Score' numeric column, values must be in a sane band (-30..+30 for SGP,
    -200..+200 for composite score).  Only asserts when such columns are found.
    """
    # The Trade Recs tab calls render_sortable_table which may surface as a
    # captured DataFrame.  Column names from _build_trade_df: 'Your Gain', 'Score'.
    bad: list[tuple[str, str, float]] = []

    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for df in r.dataframes:
            for col, lo, hi in [("Your Gain", -30.0, 30.0), ("Score", -200.0, 200.0)]:
                if col not in df.columns:
                    continue
                for raw in df[col].dropna():
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if _is_bad_number(str(raw)):
                        bad.append((team, col, val))
                        continue
                    if not (lo <= val <= hi):
                        bad.append((team, col, val))

    assert not bad, "Trade Finder: SGP/Score values outside sane range:\n" + "\n".join(
        f"  session={t}: col={c} value={v}" for t, c, v in bad
    )
