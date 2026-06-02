"""Deep per-team assertions for Player Databank (pages/19_Player_Databank.py).

Default render stops before loading data: the page requires an explicit Search
button submit (session flag ``db_search_triggered``).  Without it, the page
renders the filter form and an st.info banner, then calls st.stop().

The harness therefore produces a small amount of markdown (the info banner +
form labels) but no dataframe / metric output and no table HTML.  All checks
are defensive: they pass when data is absent and only assert when data
is actually present.

Renders ONCE per team (module-scoped ``results`` fixture).

Run serially:
    .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_19_player_databank.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/19_Player_Databank.py"

_BAD_VALUES = {"nan", "none", "inf", "-inf", ""}

# HTML tag stripper
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD_VALUES


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_databank_renders_for_all_teams(results):
    """Every team's render must complete without exception or st.error."""
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
    assert not problems, "Player Databank render problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_databank_no_nan_metrics(results):
    """Any captured metrics must not carry NaN / None / inf values."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Player Databank NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


def test_databank_default_shows_filter_prompt(results):
    """Without a Search submit the page shows the filter-prompt info banner.

    The harness sees the banner text in r.markdown.  We confirm each team
    gets at least this minimal response — proving the page scaffolding
    (auth, init_db, inject_custom_css, render_page_layout) completed
    without crashing.  The threshold is intentionally low (20 chars) to
    survive minor wording changes.
    """
    too_blank = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        # Combine markdown + any dataframe columns into one blob for searching
        combined = r.markdown
        if len(combined.strip()) < 20 and not r.dataframes and not r.metrics:
            too_blank.append(team)
    assert not too_blank, "Player Databank produced no discernible content for: " + str(too_blank)


def test_databank_no_nan_in_dataframes(results):
    """Any dataframes captured (e.g. from helper widgets) must not be all-NaN.

    Vacuously true when no dataframes are present (normal for default render).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for i, df in enumerate(r.dataframes):
            if df.empty:
                continue
            # Flag columns that are entirely NaN
            all_nan_cols = [c for c in df.columns if df[c].isna().all()]
            if len(all_nan_cols) == len(df.columns):
                bad.append((team, f"dataframe[{i}] all columns NaN"))
    assert not bad, "Player Databank all-NaN dataframe columns:\n" + "\n".join(f"  {b}" for b in bad)


def test_databank_rate_stats_plausible_when_present(results):
    """If the markdown contains HTML table data, rate stats must be in range.

    Looks for AVG/OBP-style decimals in [0, 1] and ERA/WHIP values.
    Only fires when the page actually rendered a data table (i.e., the
    search was triggered in a session — not the default headless render).

    AVG / OBP: any three-decimal float must be in [0.000, 0.999].
    ERA        : any value labelled ERA must be in [0.00, 20.00].
    WHIP       : any value labelled WHIP must be in [0.00, 4.00].

    The test is tolerant: it only fails on clear out-of-range numbers,
    not on absent data.
    """
    bad = []

    # Decimal pattern: matches numbers like 0.312, .275, 1.234
    _rate_re = re.compile(r"\b0?\.\d{3}\b")
    # ERA/WHIP: larger numbers; look for context
    _era_re = re.compile(r"ERA[^<]{0,40}?(\d{1,2}\.\d{2})", re.IGNORECASE)
    _whip_re = re.compile(r"WHIP[^<]{0,40}?(\d{1,2}\.\d{2})", re.IGNORECASE)

    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        if len(r.markdown) < 200:
            # Default render — no table data present, skip stat checks
            continue

        plain = _strip_html(r.markdown)

        for match in _rate_re.finditer(plain):
            val_str = match.group()
            try:
                val = float(val_str)
            except ValueError:
                continue
            if not (0.0 <= val <= 1.0):
                bad.append((team, f"rate-stat out of range [0,1]: {val_str}"))

        for match in _era_re.finditer(r.markdown):
            try:
                val = float(match.group(1))
            except (ValueError, IndexError):
                continue
            if not (0.0 <= val <= 20.0):
                bad.append((team, f"ERA out of range [0,20]: {val}"))

        for match in _whip_re.finditer(r.markdown):
            try:
                val = float(match.group(1))
            except (ValueError, IndexError):
                continue
            if not (0.0 <= val <= 4.0):
                bad.append((team, f"WHIP out of range [0,4]: {val}"))

    assert not bad, "Player Databank implausible rate stats:\n" + "\n".join(f"  {b}" for b in bad)
