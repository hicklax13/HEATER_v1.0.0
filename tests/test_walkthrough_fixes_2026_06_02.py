"""Structural regression guards for the 3 fixes from the 2026-06-02 per-team
browser walkthrough (Phase 3 of the pre-launch QA program).

These three live on Streamlit pages that render via custom HTML; they cannot be
imported or unit-rendered in isolation, and AppTest cannot detect client-side
HTML escaping — so these are source-level guards, matching the repo's existing
``test_no_*`` structural-invariant convention. The behavioral verification is the
post-deploy browser re-walk.

- F-VIS-1: Matchup Planner category-probability bars rendered as escaped raw HTML
  (a multi-line, 4+-space-indented triple-quoted f-string → Streamlit markdown
  treated the 2nd+ rows as a code block).
- F-VIS-2: Free Agents streaming section showed "(vs <viewer's OWN team>)" — it
  called get_opponent_context() with no opponent_name (global single-user path).
- F-PERF-1: Trade Finder's "Re-rank by title odds" toggle (a ~60s MC sim) defaulted
  ON, so every load took ~68s.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _src(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def test_perf1_trade_finder_title_odds_toggle_defaults_off():
    """F-PERF-1: the title-odds re-rank toggle must default OFF (fast load)."""
    src = _src("pages/12_Trade_Finder.py")
    i = src.find("Re-rank by title odds")
    assert i != -1, "title-odds toggle label not found in Trade Finder"
    window = src[i : i + 200]
    assert "value=False" in window, (
        "Trade Finder 'Re-rank by title odds' toggle must default value=False — it is a "
        "~60s Monte-Carlo re-rank and defaulting it ON gave every load a ~68s spinner (F-PERF-1)."
    )


def test_vis1_matchup_bars_not_multiline_fstring():
    """F-VIS-1: the category bars must be single-line HTML, not a multi-line
    triple-quoted f-string (which renders as an escaped markdown code block)."""
    src = _src("pages/5_Matchup_Planner.py")
    m = re.search(r"def _build_category_prob_html\(.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert m, "_build_category_prob_html not found in Matchup Planner"
    fn = m.group(0)
    assert 'f"""' not in fn and "f'''" not in fn, (
        "_build_category_prob_html must not build the bars with a multi-line triple-quoted "
        "f-string — 4+-space-indented HTML renders as a markdown code block and the 2nd+ "
        "bars show as escaped raw HTML (F-VIS-1). Use single-line concatenated f-strings."
    )


def test_vis2_free_agents_resolves_member_opponent():
    """F-VIS-2: FA streaming must resolve the viewer's OWN opponent and pass it
    to get_opponent_context(opponent_name=...), not the no-arg global path."""
    src = _src("pages/14_Free_Agents.py")
    assert "get_opponent_context()" not in src, (
        "Free Agents calls get_opponent_context() with no opponent_name — that uses the "
        "global single-user path and mis-attributes streaming to the member's own team (F-VIS-2)."
    )
    assert re.search(r"get_opponent_context\(\s*opponent_name=", src), (
        "Free Agents must call get_opponent_context(opponent_name=...) with the member's resolved opponent (F-VIS-2)."
    )
    assert "find_user_opponent" in src, (
        "Free Agents must resolve the member's opponent via find_user_opponent (F-VIS-2), mirroring Matchup Planner."
    )


def test_vis3_resolve_viewer_team_matches_emoji_roster_name(monkeypatch):
    """F-VIS-3 (launch, 2026-06-02): an admin assigned 'Team Hickey' (env-seeded,
    no emoji) must resolve to the roster's '🏆 Team Hickey' so their My Team isn't
    empty. resolve_viewer_team_name reconciles the assigned name against the actual
    roster names, tolerating a leading emoji/whitespace mismatch."""
    import pandas as pd

    from src import auth

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setattr(auth, "current_user", lambda: {"team_name": "Team Hickey"})
    rosters = pd.DataFrame({"team_name": ["🏆 Team Hickey", "My Precious", "On a Twosday"]})
    assert auth.resolve_viewer_team_name(rosters) == "🏆 Team Hickey"


def test_vis3_resolve_viewer_team_exact_match_unchanged(monkeypatch):
    """Exact matches (members assigned via the admin dropdown) return as-is; the
    reconciliation only engages on a non-exact match."""
    import pandas as pd

    from src import auth

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setattr(auth, "current_user", lambda: {"team_name": "My Precious"})
    rosters = pd.DataFrame({"team_name": ["🏆 Team Hickey", "My Precious"]})
    assert auth.resolve_viewer_team_name(rosters) == "My Precious"


def test_vis3_resolve_viewer_team_no_false_match(monkeypatch):
    """No normalized match → return the assigned name unchanged (no wrong-team)."""
    import pandas as pd

    from src import auth

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setattr(auth, "current_user", lambda: {"team_name": "Team Hickey"})
    rosters = pd.DataFrame({"team_name": ["My Precious", "On a Twosday"]})
    assert auth.resolve_viewer_team_name(rosters) == "Team Hickey"
