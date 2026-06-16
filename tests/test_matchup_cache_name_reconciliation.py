"""Regression guard: the 2026-06-10 empty-matchup live finding.

``sync_all_team_matchups`` writes ``league_matchup_cache`` rows under YAHOO's
team names — which carry leading emoji/whitespace ("(emoji) Team Hickey") —
while ``_db_fallback_matchup`` looks up the admin-assigned PLAIN name
("Team Hickey") with an exact ``WHERE team_name = ?``. The cache could never
hit, so ``get_matchup()`` returned None for every server session and the
Pitcher Streaming matchup-impact + swap engines saw no matchup at all. Same
DNA as the ``b7f0567`` League Standings incident (frame-reconciled viewer
names). ``load_matchup_cache`` now falls back to the canonical
``_normalize_team_name`` reconciliation when the exact match misses.
"""

from __future__ import annotations

from src.database import init_db, load_matchup_cache, save_matchup_cache

_EMOJI_NAME = "\U0001f3c6 Team Reconcile QA"  # trophy-emoji-prefixed Yahoo form
_PLAIN_NAME = "Team Reconcile QA"


def _matchup(week: int, tag: str) -> dict:
    return {
        "week": week,
        "opp_name": f"Opponent {tag}",
        "categories": [{"cat": "K", "you": "38", "opp": "44", "result": "LOSS"}],
    }


def test_plain_name_finds_emoji_cached_row():
    init_db()
    save_matchup_cache(_EMOJI_NAME, 11, _matchup(11, "emoji"))
    cached = load_matchup_cache(_PLAIN_NAME)
    assert cached is not None, (
        "plain admin-assigned name must reconcile to the emoji-prefixed "
        "Yahoo cache row — exact-match-only lookups left get_matchup() "
        "empty for every server session"
    )
    assert cached["opp_name"] == "Opponent emoji"


def test_emoji_name_finds_plain_cached_row():
    init_db()
    save_matchup_cache(_PLAIN_NAME + " B", 11, _matchup(11, "plain"))
    cached = load_matchup_cache("✨ Team Reconcile QA B")
    assert cached is not None
    assert cached["opp_name"] == "Opponent plain"


def test_exact_match_preferred_over_reconciliation():
    init_db()
    save_matchup_cache("Team Exact", 11, _matchup(11, "exact"))
    save_matchup_cache("\U0001f3c6 Team Exact", 11, _matchup(11, "fuzzy"))
    cached = load_matchup_cache("Team Exact")
    assert cached is not None
    assert cached["opp_name"] == "Opponent exact"


def test_no_match_returns_none():
    init_db()
    assert load_matchup_cache("Team That Does Not Exist Anywhere") is None


def test_week_filter_respected_through_reconciliation():
    init_db()
    save_matchup_cache(_EMOJI_NAME + " W", 10, _matchup(10, "old"))
    save_matchup_cache(_EMOJI_NAME + " W", 12, _matchup(12, "new"))
    cached = load_matchup_cache(_PLAIN_NAME + " W", week=12)
    assert cached is not None
    assert cached["opp_name"] == "Opponent new"


def test_current_week_wins_over_stale_other_name_variant():
    """2026-06-16 stale-record live finding (owner saw 4-2 vs Baty Babies when
    the live record was 4-8).

    A team is cached under TWO name spellings written by different paths:
    the scheduler's ``sync_all_team_matchups`` uses YAHOO's emoji form
    ("(emoji) Team Hickey"); other paths use the admin-assigned PLAIN form
    ("Team Hickey"). With ``week=None`` (the ``_db_fallback_matchup`` path),
    the old lookup returned the freshest row for the EXACT name only and
    reconciled across spellings ONLY on an exact miss — so a stale row under
    one spelling (an early-week snapshot, pitching still 0-0) SHADOWED the
    current-week row under the other spelling. ``load_matchup_cache`` must
    return the CURRENT (highest) week across all name variants."""
    init_db()
    # Current week (12) under the Yahoo emoji name — the REAL live record.
    save_matchup_cache(
        "\U0001f3c6 Team StaleShadow QA",
        12,
        {"week": 12, "opp_name": "Baty Babies", "wins": 4, "losses": 8, "ties": 0, "categories": []},
    )
    # Older week (11) under the plain admin-assigned name — STALE shadow.
    save_matchup_cache(
        "Team StaleShadow QA",
        11,
        {"week": 11, "opp_name": "Old Opp", "wins": 4, "losses": 2, "ties": 0, "categories": []},
    )
    cached = load_matchup_cache("Team StaleShadow QA")
    assert cached is not None
    assert cached["week"] == 12, "must return the current (max) week across name variants"
    assert (cached["wins"], cached["losses"]) == (4, 8), (
        "stale week-11 (4-2) under the plain name must not shadow the current "
        "week-12 (4-8) row cached under the emoji name"
    )
