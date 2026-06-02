"""Deep per-team assertions for Matchup Planner (pages/5_Matchup_Planner.py).

Renders the page ONCE per team (module-scoped `results` fixture) and checks
win-probability plausibility (every probability in [0,1] or [0,100]%).
Defensive style: plausibility checks fire only when the data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_5_matchup_planner.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/5_Matchup_Planner.py"
_BAD = ("nan", "none", "inf", "-inf", "")

# ── Regex patterns anchored to Matchup Planner output ────────────────────────
#
# Category probability bar: the page renders "{pct:.0f}%" inside each bar's
# center div, e.g. "63%" or "100%".  We look for 1-3 digit integers followed
# immediately by "%", excluding CSS "width:" and "opacity:" attributes which
# also contain % but are not win-probabilities.
_RE_PCT_BAR = re.compile(r"(?<!width:)(?<!opacity:)\b(\d{1,3})%")

# Overall win-probability label: e.g. "Win 72%" or "Loss 18%" in the stacked
# bar header, and the banner teaser "63% chance to win".
_RE_OVERALL_PCT = re.compile(r"(?:Win|Loss|Tie|chance\s+to\s+win)\s+(\d{1,3})%", re.IGNORECASE)

# Projected score pattern: "Projected: 7-5" or "projected {pw:.0f}-{pl:.0f}"
# Captures the two integer sides of the matchup score.
_RE_PROJ_SCORE = re.compile(r"[Pp]rojected[:\s]+(\d+)-(\d+)")

# Player matchup rating: rendered as a float in "Rating: N.NN" inside
# per-game detail expanders and the compact table "Rating" column.
_RE_RATING = re.compile(r"[Rr]ating[:\s]+(-?\d{1,3}\.\d{2})")

# Park factor: "Park Factor: N.NN" inside per-game detail.
_RE_PARK_FACTOR = re.compile(r"Park\s+Factor:\s+(\d{1,2}\.\d{2})", re.IGNORECASE)


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex matching hits the actual displayed numbers."""
    return re.sub(r"<[^>]+>", " ", text)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render Matchup Planner once per team; reused by every assertion."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Test 1: Render gate ───────────────────────────────────────────────────────


def test_matchup_renders_for_all_teams(results):
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
    assert not problems, "Matchup Planner problems:\n" + "\n".join(f"  {p}" for p in problems)


# ── Test 2: No NaN/None/inf in st.metric values ───────────────────────────────


def test_matchup_no_nan_metrics(results):
    """No displayed st.metric value is NaN/None/inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Matchup Planner NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── Test 3: Win-probability values are in [0, 100] ────────────────────────────


def test_matchup_win_prob_in_range(results):
    """Any win/loss/tie percentage found in .markdown must be in [0, 100].

    The page embeds percentage values in two places:
      - Per-category probability bars:  e.g. "63%"
      - Overall win/loss/tie header:    e.g. "Win 72%"  "Loss 18%"  "Tie 10%"
      - Banner teaser: "63% chance to win"

    We strip HTML tags before scanning to avoid matching CSS width/opacity
    attributes.  Only fired when values are actually found — absence is not
    a failure (page may fall back to info banners with no probability output).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # Per-category bars  — "63%"
        for m in _RE_PCT_BAR.finditer(plain):
            raw = m.group(1)
            try:
                val = int(raw)
            except ValueError:
                continue
            if not (0 <= val <= 100):
                bad.append((team, "cat_win_pct", raw, "outside [0, 100]"))

        # Overall win/loss/tie labels and banner teaser
        for m in _RE_OVERALL_PCT.finditer(plain):
            raw = m.group(1)
            try:
                val = int(raw)
            except ValueError:
                continue
            if not (0 <= val <= 100):
                bad.append((team, "overall_pct", raw, "outside [0, 100]"))

    assert not bad, "Matchup Planner out-of-range win probabilities:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )


# ── Test 4: Projected score sides are integers in [0, 12] and sum ≤ 12 ───────


def test_matchup_projected_score_plausible(results):
    """When a 'Projected: W-L' score is found, each side ∈ [0, 12] and W+L ≤ 12.

    FourzynBurn has 12 H2H categories, so neither side of the projected score
    can exceed 12 and the sum cannot exceed 12 (ties consume one outcome each).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        for m in _RE_PROJ_SCORE.finditer(plain):
            w_raw, l_raw = m.group(1), m.group(2)
            try:
                w, l = int(w_raw), int(l_raw)
            except ValueError:
                continue
            if not (0 <= w <= 12):
                bad.append((team, "proj_W", w_raw, "outside [0, 12]"))
            if not (0 <= l <= 12):
                bad.append((team, "proj_L", l_raw, "outside [0, 12]"))
            if w + l > 12:
                bad.append((team, "proj_W+L", f"{w}+{l}={w + l}", "sum exceeds 12 categories"))

    assert not bad, "Matchup Planner projected score out of range:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )


# ── Test 5: Player matchup ratings and park factors are plausible ─────────────


def test_matchup_ratings_and_park_factors_plausible(results):
    """Player ratings and park factors from per-game detail must be plausible.

    - Matchup ratings: rendered as "Rating: N.NN" floats.  The page computes
      these via a multiplicative wOBA/xFIP × park × platoon × home/away model.
      Plausible range: (-5, 20).  Values outside this window indicate a
      multiplication blowup or sign error.
    - Park factors: rendered as "Park Factor: N.NN".  Plausible range: [0.5, 2.0]
      (real FG 5-year park factors all live in ~[0.8, 1.2]; we allow a wider
      window for seed/fallback values).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        for m in _RE_RATING.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (-5.0 <= val <= 20.0):
                bad.append((team, "matchup_rating", raw, "outside (-5, 20)"))

        for m in _RE_PARK_FACTOR.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (0.5 <= val <= 2.0):
                bad.append((team, "park_factor", raw, "outside [0.5, 2.0]"))

    assert not bad, "Matchup Planner implausible ratings/park factors:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )
