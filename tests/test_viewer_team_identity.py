"""resolve_viewer_team_name: the SOLE 'which team is the viewer's?' resolver.

Under MULTI_USER it returns the logged-in user's admin-assigned team (per-session
identity), NOT the legacy league_teams.is_user_team flag. Guards the 2026-06-01
launch-blocker where every leaguemate saw the single is_user_team team
(Team Hickey) instead of their own assigned team.
"""

import pandas as pd


def _rosters(flag_team: str = "Team Hickey") -> pd.DataFrame:
    """Two-team roster frame with is_user_team pinned to ``flag_team`` (the v1 flag)."""
    return pd.DataFrame(
        [
            {"team_name": "Team Hickey", "is_user_team": 1 if flag_team == "Team Hickey" else 0},
            {"team_name": "HUMAN INTELLIGENCE", "is_user_team": 1 if flag_team == "HUMAN INTELLIGENCE" else 0},
        ]
    )


def test_multiuser_returns_session_team_over_flag(monkeypatch):
    """THE BUG: a non-admin assigned HUMAN INTELLIGENCE must NOT see Team Hickey."""
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "testuser", "team_name": "HUMAN INTELLIGENCE"})

    # is_user_team flag points at Team Hickey, but the session identity wins.
    assert auth.resolve_viewer_team_name(_rosters(flag_team="Team Hickey")) == "HUMAN INTELLIGENCE"


def test_multiuser_no_session_team_returns_none(monkeypatch):
    """A logged-in member with no assigned team must NOT inherit the admin's team.

    Under MULTI_USER, an empty/blank team_name resolves to None (so pages show a
    'connect / no team' state) — never the legacy is_user_team flag, which points
    at the admin's team. 2026-06-01 audit: empty-team guard.
    """
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "x", "team_name": None})

    # The flag points at Team Hickey, but a teamless member must not see it.
    assert auth.resolve_viewer_team_name(_rosters(flag_team="Team Hickey")) is None


def test_multiuser_blank_team_returns_none(monkeypatch):
    """Whitespace-only team_name is treated as 'no team' (None), not the flag."""
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "x", "team_name": "  "})

    assert auth.resolve_viewer_team_name(_rosters(flag_team="Team Hickey")) is None


def test_single_user_uses_flag_ignoring_session(monkeypatch):
    """v1 byte-for-byte: MULTI_USER off ⇒ the is_user_team flag wins, session ignored."""
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: False)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "x", "team_name": "HUMAN INTELLIGENCE"})

    assert auth.resolve_viewer_team_name(_rosters(flag_team="Team Hickey")) == "Team Hickey"


def test_no_rosters_falls_back_to_db(monkeypatch):
    """With no rosters frame, the v1 path queries league_teams.is_user_team."""
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: False)

    class _Result:
        def fetchone(self):
            return ("Team Hickey",)

    class _Conn:
        def execute(self, sql, *args):
            assert "is_user_team = 1" in sql
            return _Result()

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _Conn())
    assert auth.resolve_viewer_team_name(None) == "Team Hickey"


def test_multiuser_session_team_wins_even_with_no_rosters(monkeypatch):
    """Session identity needs no rosters frame to resolve."""
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "testuser", "team_name": "HUMAN INTELLIGENCE"})

    assert auth.resolve_viewer_team_name(None) == "HUMAN INTELLIGENCE"
