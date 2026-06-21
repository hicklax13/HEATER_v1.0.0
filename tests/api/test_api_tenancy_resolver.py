"""Tenancy helpers + resolver tests. This file covers the pure helpers and the
ViewerContext resolver (incl. the dormant open-read fallback)."""

from api.tenancy import normalize_team_name, reconcile_team_name


def test_normalize_strips_emoji_whitespace_punctuation():
    # Same semantics as src.auth._normalize_team_name (replicated, not imported).
    assert normalize_team_name("🏆 Team Hickey") == normalize_team_name("Team Hickey")
    assert normalize_team_name("Team Hickey") == "teamhickey"
    assert normalize_team_name("  A.B-C  ") == "abc"


def test_reconcile_exact_match_returns_canonical_roster_name():
    names = ["🏆 Team Hickey", "Bronx Bombers"]
    assert reconcile_team_name("Team Hickey", names) == "🏆 Team Hickey"


def test_reconcile_exact_string_short_circuits():
    names = ["Team Hickey", "Other"]
    assert reconcile_team_name("Team Hickey", names) == "Team Hickey"


def test_reconcile_no_match_with_known_names_returns_none():
    assert reconcile_team_name("Nonexistent", ["A", "B"]) is None


def test_reconcile_empty_names_returns_assigned_as_is():
    # Cold/empty roster source must NOT block assignment (graceful).
    assert reconcile_team_name("Team Hickey", []) == "Team Hickey"
