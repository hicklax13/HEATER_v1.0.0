"""Wave 9 / Task 2: fetch_minor_league_players from MLB Stats API."""

from unittest.mock import patch

import pandas as pd


def _make_fake_sports_players_response(level_code: str, num_teams: int = 2, players_per_team: int = 40):
    """Build a fake statsapi.get('sports_players') response for minor leagues."""
    people = []
    for team_idx in range(num_teams):
        team_id = 100 + team_idx
        for p_idx in range(players_per_team):
            people.append(
                {
                    "id": 600000 + team_idx * 1000 + p_idx,
                    "fullName": f"Player {level_code}{team_idx}-{p_idx}",
                    "active": True,
                    "primaryPosition": {"abbreviation": "OF", "type": "Outfielder"},
                    "currentTeam": {"id": team_id},
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "birthDate": "2000-01-01",
                }
            )
    return {"people": people}


def test_fetch_minor_league_players_caps_at_top_n_per_team():
    """fetch_minor_league_players returns at most top_n_per_team rows per affiliate."""
    from src import live_stats

    fake = _make_fake_sports_players_response("AAA", num_teams=3, players_per_team=50)
    with (
        patch.object(live_stats.statsapi, "get", return_value=fake),
        patch.object(live_stats, "_build_team_id_map", return_value={100: "SCR", 101: "OMA", 102: "BUF"}),
    ):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",), top_n_per_team=30)

    assert not df.empty
    # 3 teams × 30 cap = 90 rows
    assert len(df) == 90, f"Expected 90 rows (3×30 cap); got {len(df)}"
    # Each team has exactly 30
    counts_per_team = df.groupby("team").size().to_dict()
    assert all(c == 30 for c in counts_per_team.values()), f"Cap violated: {counts_per_team}"


def test_fetch_minor_league_players_sets_level_column():
    """Each row gets level='AAA' or 'AA' matching the source sportId."""
    from src import live_stats

    aaa_resp = _make_fake_sports_players_response("AAA", num_teams=1, players_per_team=10)
    aa_resp = _make_fake_sports_players_response("AA", num_teams=1, players_per_team=10)

    def _fake_get(endpoint, params, **kwargs):
        if params.get("sportId") == 11:
            return aaa_resp
        if params.get("sportId") == 12:
            return aa_resp
        return {"people": []}

    with (
        patch.object(live_stats.statsapi, "get", side_effect=_fake_get),
        patch.object(live_stats, "_build_team_id_map", return_value={100: "SCR"}),
    ):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA", "AA"), top_n_per_team=30)

    assert "level" in df.columns
    assert set(df["level"].unique()) == {"AAA", "AA"}
    assert (df[df["level"] == "AAA"]["mlb_id"] >= 600000).all()
    assert (df[df["level"] == "AA"]["mlb_id"] >= 600000).all()


def test_fetch_minor_league_players_handles_empty_response():
    """Empty API response → empty DataFrame, no crash."""
    from src import live_stats

    with (
        patch.object(live_stats.statsapi, "get", return_value={"people": []}),
        patch.object(live_stats, "_build_team_id_map", return_value={}),
    ):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",))
    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_fetch_minor_league_players_handles_api_failure():
    """statsapi exception → empty DataFrame + logger.warning, no crash."""
    from src import live_stats

    with (
        patch.object(live_stats.statsapi, "get", side_effect=ConnectionError("simulated")),
        patch.object(live_stats, "_build_team_id_map", return_value={}),
    ):
        df = live_stats.fetch_minor_league_players(season=2026, levels=("AAA",))
    assert df.empty
