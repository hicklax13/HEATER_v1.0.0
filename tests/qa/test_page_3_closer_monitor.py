"""Deep per-team assertions for Closer Monitor (pages/3_Closer_Monitor.py).

Renders the page ONCE per team (module-scoped `results` fixture). Closer Monitor
is league-wide (a 30-team MLB closer grid rendered via individual st.markdown card
calls per team-slot); per-team runs validate it renders cleanly under each session.
Defensive style: plausibility checks fire only when data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_3_closer_monitor.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/3_Closer_Monitor.py"
_BAD = ("nan", "none", "inf", "-inf", "")

# Candidate 2-3 letter uppercase token (intersected with the canonical set below).
_MLB_TEAM_CODE_RE = re.compile(r"\b([A-Z]{2,3})\b")

# Canonical 30-team MLB code set (MLB Stats API abbreviations as HEATER emits
# them).  NOTE: HEATER uses "ATH" (not "OAK") for the Athletics — see
# tests/test_no_oak_in_source_modules.py.  Intersecting matches with this set
# stops prose acronyms (SV, ERA, WHIP, HOT, COLD, BN, IL …) from being counted
# as "team codes" — the old bare-regex count inflated so a collapsed grid still
# passed ≥15.
_CANONICAL_MLB_CODES = frozenset(
    {
        "ARI",
        "ATL",
        "ATH",
        "BAL",
        "BOS",
        "CHC",
        "CWS",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "PHI",
        "PIT",
        "SD",
        "SF",
        "SEA",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WSH",
    }
)

# Minimum number of *distinct* canonical MLB team codes we expect to find in the
# combined markdown when depth-chart data is present.
_MIN_TEAM_CODE_COUNT = 15

# The page renders each closer card as a separate st.markdown() call, so
# `results.markdown` is the *concatenation* of all of them.  We call the
# combined string the "grid blob".


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags, leaving plain text for safer pattern matching."""
    return re.sub(r"<[^>]+>", " ", text)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render Closer Monitor once per team; reused by every assertion."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── core render guard ────────────────────────────────────────────────────────


def test_closer_renders_for_all_teams(results):
    """Every team's render must complete without crash, exception, or blank output."""
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
        # Page is not blank: any of dataframes / metrics / markdown counts
        if not r.dataframes and not r.metrics and len(r.markdown) < 50:
            problems.append((team, "blank", "no dataframes/metrics/markdown"))
    assert not problems, "Closer Monitor render problems:\n" + "\n".join(f"  {p}" for p in problems)


# ── metric NaN guard ─────────────────────────────────────────────────────────


def test_closer_no_nan_metrics(results):
    """No st.metric value captured for Closer Monitor should be NaN/None/empty."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Closer Monitor NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── league-wide consistency ───────────────────────────────────────────────────


def test_closer_markdown_consistent_across_teams(results):
    """Closer Monitor is league-wide content; all teams should see roughly the
    same volume of HTML.  The minimum markdown length across successful renders
    must be at least 60 % of the maximum — a large discrepancy means one team
    got a truncated or empty grid while another got the full 30-team layout.

    Skipped (passes trivially) when no team produced markdown longer than 50
    characters (e.g. data not bootstrapped in CI seed environment).
    """
    lengths = {
        team: len(r.markdown) for team, r in results.items() if r.ran and not r.exception and len(r.markdown) >= 50
    }
    if not lengths:
        pytest.skip("No team produced substantial markdown — depth-chart data absent; skipping consistency check.")

    min_len = min(lengths.values())
    max_len = max(lengths.values())
    ratio = min_len / max_len

    assert ratio >= 0.60, (
        f"Closer Monitor markdown volume varies too widely across teams "
        f"(min={min_len}, max={max_len}, ratio={ratio:.2f} < 0.60).  "
        f"Teams with shortest output: " + str([t for t, l in lengths.items() if l == min_len])
    )


# ── MLB team-code coverage ────────────────────────────────────────────────────


def test_closer_grid_covers_expected_mlb_teams(results):
    """When the grid is rendered, the combined markdown (stripped of HTML tags)
    should reference at least 15 distinct CANONICAL MLB team codes.  This guards
    against the grid silently collapsing to a handful of entries.

    Previously this counted every ``\\b[A-Z]{2,3}\\b`` token, which matched prose
    acronyms (SV, ERA, WHIP, HOT, COLD …) as "team codes" — so a collapsed grid
    could still clear the ≥15 bar.  We now intersect the regex matches with the
    canonical 30-team set before counting, so only real team codes contribute.

    Skipped when no team produced meaningful markdown (data not seeded).
    """
    # Collect all markdown from teams that rendered successfully
    all_markdown_blobs = [r.markdown for r in results.values() if r.ran and not r.exception and len(r.markdown) >= 50]
    if not all_markdown_blobs:
        pytest.skip("No substantive markdown found — depth-chart data absent; skipping team-code coverage check.")

    # Use one representative blob (pick the longest — most complete render)
    representative = max(all_markdown_blobs, key=len)
    plain = _strip_html(representative)

    # Collect 2–3 uppercase-letter tokens, then keep ONLY those that are real
    # canonical MLB team codes (prose acronyms are discarded).
    candidates = set(_MLB_TEAM_CODE_RE.findall(plain)) & _CANONICAL_MLB_CODES
    distinct_count = len(candidates)

    assert distinct_count >= _MIN_TEAM_CODE_COUNT, (
        f"Closer Monitor grid references only {distinct_count} distinct canonical "
        f"MLB team codes in the rendered markdown (expected >= {_MIN_TEAM_CODE_COUNT}).  "
        f"Found codes: {sorted(candidates)}"
    )


# ── dataframe NaN guard (defensive — only fires when DFs are present) ─────────


def test_closer_no_nan_in_dataframes(results):
    """No NaN / None / empty-string cell in any DataFrame surfaced by the page.
    Closer Monitor rarely emits DataFrames (it uses HTML cards), so this test
    is a safety-net that passes vacuously when no DFs are captured."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for idx, df in enumerate(r.dataframes):
            for col in df.columns:
                for val in df[col]:
                    if _is_bad_number(val):
                        bad.append((team, f"df[{idx}].{col}", val))
    assert not bad, "Closer Monitor NaN/None dataframe cells:\n" + "\n".join(f"  {b}" for b in bad)
