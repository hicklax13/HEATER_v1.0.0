"""Per-team OWNERSHIP assertions — the launch-critical gap the plausibility
suite misses.

The smoke + deep suites prove each page LOADS and shows PLAUSIBLE values for all
12 teams. But a page could silently render ANOTHER team's roster with every value
still plausible — exactly the 2026-06-01 launch-blocker, where every member saw
the admin's team ("Team Hickey") because pages resolved "my team" via a global
flag instead of the session user's assigned team. A plausibility test passes
green on that bug; only an OWNERSHIP test catches it.

Rosters are disjoint across the 12 teams, so for a page that displays the
viewer's own roster, the viewing team's players must out-number every other
team's on the rendered page. We assert exactly that, plus a calibration guard
(the signal must be live, never vacuous) and a direct wrong-team-name check.

Surfaced by the 2026-06-02 silent-failure audit (Finding 1).

Run SERIALLY:
    .venv\\Scripts\\python.exe -m pytest tests/qa/test_team_ownership.py -q
"""

from __future__ import annotations

import math
import re

import pytest

# Pages that display ONLY the viewing user's own roster as a table, so own-team
# player overlap should dominate. Trade Analyzer is intentionally EXCLUDED: its
# "You Receive" selector legitimately lists OTHER teams' players (trade targets),
# which would contaminate an overlap-based ownership check.
ROSTER_PAGES = [
    ("pages/1_My_Team.py", "My Team"),
    ("pages/2_Line-up_Optimizer.py", "Lineup Optimizer"),
]
MY_TEAM = "pages/1_My_Team.py"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _haystack(r) -> str:
    """All rendered text where a player name could appear: the unified text
    corpus (markdown + headings + metric pairs + dataframe text) plus widget
    option labels (rosters shown in a multiselect live there, not in markdown)."""
    return _strip_html(r.text) + " " + " ".join(str(o) for o in (r.widget_options or []))


def _count_present(names, haystack) -> int:
    """How many full player names (>=5 chars, contains a space) appear verbatim."""
    return sum(1 for n in names if n and len(n) >= 5 and " " in n and n in haystack)


def _displayed_text(r) -> str:
    """Only what the page RENDERS as content (markdown + headings + metric pairs
    + dataframe text) — NOT selectable widget-option labels.

    The bleed detector must read the DISPLAYED roster, not dropdown options.
    Several pages legitimately populate a selector with the whole league pool —
    the Lineup Optimizer's "compare any player" multiselect lists all ~9,900 pool
    players; Trade Analyzer's "You Receive" lists every other team's roster — which
    would make every team's players "present" and false-positive an overlap check
    (the 2026-06-08 regression: HUMAN INTELLIGENCE, the smallest roster at 24, was
    out-counted by 27-player teams whose names only ever appeared in that dropdown).
    A genuine cross-team *display* bug renders another team's roster as on-page
    content, which lands in r.text — so we scan that, and only that.
    """
    return _strip_html(r.text)


def _detect_roster_bleed(displayed, rosters_by_team, team, team_names):
    """Return a problem string if another team's roster DOMINATES the displayed
    page content, else None.

    Fails only on a CLEAR bleed: another team has a meaningful on-page presence
    (>=5 of its players) AND out-numbers the viewer's own roster by a solid margin
    (+3). Operates on DISPLAYED text only (see ``_displayed_text``).
    """
    expected = rosters_by_team.get(team, set())
    if not expected:
        return None  # no known roster for this team; nothing to attribute
    overlaps = {t: _count_present(rosters_by_team[t], displayed) for t in team_names}
    own = overlaps[team]
    best_team = max(team_names, key=lambda t: overlaps[t])
    best = overlaps[best_team]
    if best_team != team and best >= 5 and best >= own + 3:
        top = sorted(overlaps.items(), key=lambda kv: -kv[1])[:3]
        return f"team={team!r} sees '{best_team}' roster dominating (own={own}, {best_team}={best}); top overlaps={top}"
    return None


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rosters_by_team(team_names):
    """{team_name: set(full player names)} from the per-team roster JOIN.

    league_rosters has no name column (structural invariant), so names come from
    get_team_roster(), which JOINs players.name. Rosters are disjoint by player_id
    across the 12 teams.
    """
    from src.league_manager import get_team_roster

    out = {}
    for t in team_names:
        df = get_team_roster(t)
        names = set()
        if not df.empty and "name" in df.columns:
            names = {str(n).strip() for n in df["name"].dropna() if str(n).strip()}
        out[t] = names
    return out


@pytest.fixture(scope="module")
def page_results(run_page_as_team, team_names):
    """Render each roster-display page once per team; reused by every assertion."""
    return {pp: {t: run_page_as_team(pp, t, is_admin=False) for t in team_names} for pp, _ in ROSTER_PAGES}


# ── Test A: cross-team roster bleed detector (the launch-critical check) ──────


@pytest.mark.parametrize("page_path,title", ROSTER_PAGES, ids=[p[1] for p in ROSTER_PAGES])
def test_page_shows_viewers_own_roster(page_results, rosters_by_team, team_names, page_path, title):
    """No OTHER team's roster may dominate the page the viewer is looking at.

    If team A's page silently rendered team B's roster (the 2026-06-01 bug), every
    value would still be plausible — only this overlap check catches it. Fails only
    on a CLEAR bleed: another team has a meaningful on-page presence (>=5 of its
    players) AND out-numbers the viewer's own roster by a solid margin (+3).
    """
    results = page_results[page_path]
    problems = []
    for team in team_names:
        r = results[team]
        if not r.ran or r.exception:
            continue  # crash already reported by the smoke/deep suites
        prob = _detect_roster_bleed(_displayed_text(r), rosters_by_team, team, team_names)
        if prob:
            problems.append(f"{title}: {prob}")
    assert not problems, "CROSS-TEAM ROSTER BLEED:\n" + "\n".join("  " + p for p in problems)


# ── Test A-unit: bleed-detector logic (render-free regression guard) ──────────


def test_detect_roster_bleed_ignores_full_pool_selector_and_catches_display_swap():
    """Lock the 2026-06-08 fix: the bleed detector reads DISPLAYED content only.

    A full-pool selector (Lineup's "compare any player" dropdown lists the whole
    league) leaves the DISPLAYED roster text empty of other teams, so it must NOT
    register as a bleed — while a genuine display swap (the page renders another
    team's roster instead of the viewer's) MUST still be caught.
    """
    a = {"Alpha Aaa", "Alpha Bbb", "Alpha Ccc", "Alpha Ddd", "Alpha Eee", "Alpha Fff"}
    b = {"Bravo Aaa", "Bravo Bbb", "Bravo Ccc", "Bravo Ddd", "Bravo Eee", "Bravo Fff"}
    c = {"Charlie Aaa", "Charlie Bbb", "Charlie Ccc", "Charlie Ddd", "Charlie Eee"}
    rosters = {"A": a, "B": b, "C": c}
    teams = ["A", "B", "C"]

    # Full-pool dropdown only -> DISPLAYED roster text is empty -> no bleed.
    assert _detect_roster_bleed("", rosters, "A", teams) is None
    # Own roster displayed, others absent -> no bleed.
    assert _detect_roster_bleed(" ".join(a), rosters, "A", teams) is None
    # Genuine display swap: A's page renders B's roster, A's own absent -> BLEED.
    assert _detect_roster_bleed(" ".join(b), rosters, "A", teams) is not None


# ── Test B: calibration guard — the ownership signal must be live ─────────────


def test_my_team_ownership_signal_is_live(page_results, rosters_by_team, team_names):
    """Most teams must render >=3 of their OWN players on My Team.

    Proves Test A is meaningful (the harness actually captures roster names and
    they match get_team_roster). If this fails, EITHER the capture/matching is
    mis-calibrated OR there is a widespread bleed — both must be investigated.
    Prevents the ownership test from passing vacuously (the whole point of the
    silent-failure audit).
    """
    results = page_results[MY_TEAM]
    teams_with_signal = 0
    detail = []
    for team in team_names:
        r = results[team]
        expected = rosters_by_team.get(team, set())
        if not r.ran or r.exception or not expected:
            continue
        own = _count_present(expected, _haystack(r))
        detail.append((team, own))
        if own >= 3:
            teams_with_signal += 1
    floor = max(1, math.ceil(0.66 * len(team_names)))
    assert teams_with_signal >= floor, (
        f"My Team ownership signal too weak: only {teams_with_signal}/{len(team_names)} "
        f"teams rendered >=3 of their own players (need >= {floor}). Either the roster "
        f"capture/matching is mis-calibrated or rosters are bleeding. "
        f"per-team own-overlap (sorted): {sorted(detail, key=lambda kv: kv[1])}"
    )


# ── Test C: wrong-team-name-shown (direct 2026-06-01 symptom) ─────────────────


def test_my_team_header_names_viewed_team(page_results, team_names):
    """My Team must not show a DIFFERENT team's name while hiding the viewer's own.

    Directly targets the 2026-06-01 symptom (every member saw "Team Hickey"). If
    any league team name appears on the page, the VIEWED team's name must be among
    them. If no team name is textualised at all, this is skipped (Test A still
    covers ownership via roster overlap).
    """
    results = page_results[MY_TEAM]
    all_teams = {t for t in team_names if t}
    problems = []
    for team in team_names:
        r = results[team]
        if not r.ran or r.exception:
            continue
        haystack = " ".join(r.headings) + " " + _strip_html(r.text)
        present = {t for t in all_teams if t in haystack}
        if not present:
            continue  # page doesn't textualise team names; overlap test covers it
        if team not in present:
            problems.append(f"team={team!r} own name absent but other team name(s) present: {sorted(present - {team})}")
    assert not problems, "WRONG TEAM NAME SHOWN (2026-06-01 class):\n" + "\n".join("  " + p for p in problems)
