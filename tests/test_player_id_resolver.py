"""Unit tests for the shared Muncy-DNA-safe mlb_id resolver.

statsapi is mocked throughout — no network (the conftest guard blocks it anyway).
These pin the resolver contract that both scripts/backfill_player_mlb_ids.py and
the data_bootstrap mlb_id-enrichment phase depend on.
"""

from __future__ import annotations

from unittest.mock import patch

from src.player_id_resolver import resolve_mlb_id


def _match(pid, full_name, team_id=None):
    m = {"id": pid, "fullName": full_name}
    if team_id is not None:
        m["currentTeam"] = {"id": team_id}
    return m


def test_unique_name_match_returns_id():
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=[_match(661403, "Emmanuel Clase")]):
        mlb_id, reason = resolve_mlb_id("Emmanuel Clase", "CLE")
    assert mlb_id == 661403
    assert "unique name match" in reason


def test_unique_match_failing_surname_check_is_skipped():
    # A loose single hit whose surname doesn't match must NOT be accepted.
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=[_match(1, "Completely Different")]):
        mlb_id, reason = resolve_mlb_id("Emmanuel Clase", "CLE")
    assert mlb_id is None
    assert "surname check" in reason


def test_unique_match_non_numeric_id_is_skipped():
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=[_match("not-an-int", "Emmanuel Clase")]):
        mlb_id, reason = resolve_mlb_id("Emmanuel Clase", "CLE")
    assert mlb_id is None
    assert "non-numeric id" in reason


def test_multi_match_disambiguated_by_team():
    # Two "Luis Garcia"; only one is on the Nationals (team_id 120 = WSH).
    matches = [_match(671277, "Luis Garcia", team_id=120), _match(472610, "Luis Garcia", team_id=108)]
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=matches):
        mlb_id, reason = resolve_mlb_id("Luis Garcia", "WSH")
    assert mlb_id == 671277
    assert "name+team match" in reason


def test_multi_match_disambiguated_via_aliased_team():
    # The fantasy abbreviation SFG must canonicalize to SF (team_id 137).
    matches = [_match(678495, "Randy Rodriguez", team_id=137), _match(999999, "Randy Rodriguez", team_id=108)]
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=matches):
        mlb_id, _reason = resolve_mlb_id("Randy Rodriguez", "SFG")
    assert mlb_id == 678495


def test_multi_match_no_team_survivor_is_skipped():
    # Shared name, but NONE of the candidates is on the requested team -> never guess.
    matches = [_match(671277, "Luis Garcia", team_id=120), _match(472610, "Luis Garcia", team_id=108)]
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=matches):
        mlb_id, reason = resolve_mlb_id("Luis Garcia", "HOU")
    assert mlb_id is None
    assert "ambiguous" in reason


def test_multi_match_unknown_team_is_skipped():
    matches = [_match(1, "Luis Garcia", team_id=120), _match(2, "Luis Garcia", team_id=108)]
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=matches):
        mlb_id, reason = resolve_mlb_id("Luis Garcia", "NONE")
    assert mlb_id is None
    assert "no team id" in reason


def test_lookup_error_is_distinguished_from_absence():
    # An API outage must NOT masquerade as "no such player".
    with patch("src.player_id_resolver.statsapi.lookup_player", side_effect=RuntimeError("boom")):
        mlb_id, reason = resolve_mlb_id("Emmanuel Clase", "CLE")
    assert mlb_id is None
    assert "ERRORED" in reason


def test_no_match_returns_absence_reason():
    with patch("src.player_id_resolver.statsapi.lookup_player", return_value=[]):
        mlb_id, reason = resolve_mlb_id("Nobody Atall", "PIT")
    assert mlb_id is None
    assert "no lookup_player match" in reason


def test_blank_name_returns_none():
    mlb_id, reason = resolve_mlb_id("   ", "CLE")
    assert mlb_id is None
    assert reason == "blank name"


def test_season_fallback_2026_empty_then_2025():
    calls = []

    def fake_lookup(name, sportId=1, season=None):  # noqa: N803 - mirrors statsapi kwarg
        calls.append(season)
        return [_match(661403, "Emmanuel Clase")] if season == 2025 else []

    with patch("src.player_id_resolver.statsapi.lookup_player", side_effect=fake_lookup):
        mlb_id, reason = resolve_mlb_id("Emmanuel Clase", "CLE")
    assert mlb_id == 661403
    assert calls[:2] == [2026, 2025]  # tried 2026 first, fell back to 2025
    assert "season=2025" in reason
