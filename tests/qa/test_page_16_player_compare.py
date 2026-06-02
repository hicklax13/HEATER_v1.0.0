"""Deep per-team assertions for Player Compare (pages/16_Player_Compare.py).

Renders ONCE per team (module-scoped `results` fixture). No players are
pre-selected in the headless harness, so the default render shows the
search inputs, level selectbox, and the "Select Players" context card.
We check for clean load, non-trivial HTML content, no error output,
no NaN/None metrics, and that any rate-stat figures that *do* appear
(e.g. from a future default comparison) fall in plausible ranges.
Defensive style: assertions fire only when data is present.

Run SERIALLY:
    .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_16_player_compare.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/16_Player_Compare.py"

# Values that are never valid for a displayed number
_BAD = ("nan", "none", "inf", "-inf", "")


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex matches plain text."""
    return re.sub(r"<[^>]+>", " ", text)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_player_compare_renders_for_all_teams(results):
    """Every team must get a clean render with non-trivial content.

    The default view (no players selected) always emits at least:
    - the page banner / layout HTML
    - the level-selectbox widget
    - the "Select Players" context card

    That pushes markdown well above 50 chars. A blank page or unhandled
    exception here means a load-time crash, not a missing comparison.
    """
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
        # Accept any of: dataframes, metrics, or meaningful markdown
        has_content = bool(r.dataframes) or bool(r.metrics) or len(r.markdown) >= 50
        if not has_content:
            problems.append((team, "blank", "no dataframes / metrics / markdown ≥ 50 chars"))
    assert not problems, "Player Compare load problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_player_compare_no_nan_metrics(results):
    """No captured st.metric value should be NaN / None / inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Player Compare NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


def test_player_compare_default_shows_select_prompt(results):
    """The default (no-selection) render must contain the 'Select Players'
    context-card text for every team.  This verifies the context panel
    rendered correctly and the page did not short-circuit before that point.
    """
    missing = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        plain = _strip_html(r.markdown)
        # Accept either the exact heading or the instructional sentence that
        # always appears when no comparison is active.
        if "Select Players" not in plain and "Select two" not in plain and "select two" not in plain:
            missing.append(team)
    assert not missing, "Player Compare missing 'Select Players' prompt for teams: " + ", ".join(missing)


def test_player_compare_no_nan_in_dataframes(results):
    """Any captured DataFrames must not contain NaN / None / inf display strings.

    DataFrames are only captured when a comparison table is rendered, so this
    test is a no-op on teams whose render produced no DataFrames (acceptable).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for df in r.dataframes:
            for col in df.columns:
                for val in df[col].astype(str):
                    if val.strip().lower() in _BAD:
                        bad.append((team, col, val))
    assert not bad, "Player Compare NaN/None in DataFrames:\n" + "\n".join(
        f"  team={b[0]} col={b[1]} val={b[2]!r}" for b in bad
    )


def test_player_compare_rate_stats_in_range(results):
    """If any comparison table rendered and surfaced rate-stat strings, they
    must be plausible.  Patterns targeted (scanned over the unified r.text
    corpus, HTML-stripped):
      - AVG / OBP figures: leading-zero 3-decimal number in [0.000, 1.000]
      - ERA figures: a 2-decimal number ANCHORED to a nearby "ERA" label, [0, 20]
      - WHIP figures: a 2-3-decimal number ANCHORED to a nearby "WHIP" label, [0, 4]

    Only fires when such strings are actually present.  All checks are tolerant:
    a missing pattern is not a failure.
    """
    out_of_range: list[tuple[str, str, str, float]] = []

    # AVG/OBP are self-identifying by shape (a leading-zero 3-decimal in [0,1]),
    # so this arm stays unanchored.  ERA/WHIP floats, by contrast, look like any
    # other 2-decimal number — anchoring them to a nearby "ERA"/"WHIP" label (as
    # the My Team test does) stops the old UNANCHORED patterns from range-checking
    # arbitrary floats (e.g. an SGP of 12.34, a Stuff+ of 1.45) as if they were
    # baseball rate stats (a false-positive risk).
    avg_obp_pat = re.compile(r"\b(0\.\d{3}|1\.000)\b")  # e.g. 0.275
    era_pat = re.compile(r"\bERA\b[^0-9]{0,15}(\d{1,2}\.\d{2})", re.IGNORECASE)
    whip_pat = re.compile(r"\bWHIP\b[^0-9]{0,15}([0-3]?\.\d{2,3})", re.IGNORECASE)

    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        plain = _strip_html(r.text or "")

        # AVG / OBP — must be in [0, 1]
        for m in avg_obp_pat.findall(plain):
            v = float(m)
            if not (0.0 <= v <= 1.0):
                out_of_range.append((team, "AVG/OBP", m, v))

        # ERA — anchored to an "ERA" label; must be in [0, 20]
        for m in era_pat.findall(plain):
            v = float(m)
            if not (0.0 <= v <= 20.0):
                out_of_range.append((team, "ERA", m, v))

        # WHIP — anchored to a "WHIP" label; must be in [0, 4]
        for m in whip_pat.findall(plain):
            v = float(m)
            if not (0.0 <= v <= 4.0):
                out_of_range.append((team, "WHIP", m, v))

    assert not out_of_range, "Player Compare out-of-range rate stats:\n" + "\n".join(
        f"  team={b[0]} stat={b[1]} raw={b[2]!r} value={b[3]}" for b in out_of_range
    )
