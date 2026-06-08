"""#4 QA finding Q2-2 (2026-06-07): bound ALL statsapi calls with a default timeout.

statsapi.player_stat_data / lookup_player / boxscore_data / schedule call
statsapi.get internally with NO timeout, so a slow/unreachable MLB Stats API makes
requests.get block forever and HANGS the page render — a real production hang for
members (opposing pitchers in Matchup/Lineup via _fetch_single_pitcher; today's
lineups via get_todays_lineups). The GUARDED QA suite cannot catch this (the
network guard fail-fasts the connect), so it was found by inspection.

statsapi.get accepts request_kwargs={"timeout": ...} (live_stats uses it), but the
wrapper functions don't pass it. game_day installs a default-timeout wrapper on
statsapi.get at import so EVERY statsapi call is bounded.
"""

import src.game_day as game_day


def test_inject_default_timeout_adds_when_missing():
    out = game_day._inject_default_timeout({})
    assert out["request_kwargs"]["timeout"] == game_day._API_TIMEOUT


def test_inject_default_timeout_preserves_existing_timeout():
    out = game_day._inject_default_timeout({"request_kwargs": {"timeout": 5}})
    assert out["request_kwargs"]["timeout"] == 5


def test_inject_default_timeout_keeps_other_request_kwargs():
    out = game_day._inject_default_timeout({"request_kwargs": {"verify": False}})
    assert out["request_kwargs"]["verify"] is False
    assert out["request_kwargs"]["timeout"] == game_day._API_TIMEOUT


def test_statsapi_get_is_timeout_wrapped_after_import():
    """Importing game_day must wrap statsapi.get so every statsapi call is bounded."""
    import statsapi

    assert getattr(statsapi.get, "_heater_timeout_wrapped", False), (
        "game_day must install a default-timeout wrapper on statsapi.get at import"
    )


def test_wrapped_statsapi_get_passes_timeout_to_underlying(monkeypatch):
    """The installed wrapper injects the default timeout into the real call."""
    import statsapi

    captured = {}

    def _fake_orig(endpoint, params=None, **kwargs):
        captured["request_kwargs"] = kwargs.get("request_kwargs")
        return {"ok": True}

    # Re-install the wrapper over a capturing fake to verify injection.
    monkeypatch.setattr(statsapi, "get", _fake_orig, raising=True)
    monkeypatch.setattr(game_day, "_statsapi", statsapi, raising=False)
    # Force a fresh install over the fake (the fake lacks the wrapped marker).
    game_day._install_statsapi_default_timeout()
    statsapi.get("teams", {"sportId": 1})
    assert captured["request_kwargs"]["timeout"] == game_day._API_TIMEOUT
