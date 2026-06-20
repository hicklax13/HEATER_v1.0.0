"""Matchup roster tables: helper unit tests + fake-service contract test."""

from __future__ import annotations

from api.services.matchup_service import (
    _cat_win,
    _date_tabs,
    _fmt_hitter_stats,
    _fmt_pitcher_stats,
    _game_state,
    _pair_rows,
    _to_match_player,
)


def test_fmt_hitter_stats():
    row = {"h": 120, "ab": 410, "r": 70, "hr": 24, "rbi": 80, "sb": 12, "avg": 0.293, "obp": 0.371}
    assert _fmt_hitter_stats(row) == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_fmt_pitcher_stats():
    row = {"ip": 180.0, "w": 14, "l": 7, "sv": 0, "k": 200, "era": 3.21, "whip": 1.05}
    assert _fmt_pitcher_stats(row) == ["180.0", "14", "7", "0", "200", "3.21", "1.05"]


def test_fmt_stats_nan_safe():
    assert _fmt_hitter_stats({"h": float("nan"), "ab": None})[0] == "0/0"


def test_game_state_maps_status():
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "Final"}]
    state, status = _game_state("NYY", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})
    assert state == "final"
    sched2 = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "In Progress"}]
    assert _game_state("BOS", sched2, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "live"
    # no game today → none
    assert _game_state("LAD", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "none"


def test_cat_win_respects_inverse():
    assert _cat_win(5.0, 3.0, inverse=False) == "you"  # higher wins
    assert _cat_win(5.0, 3.0, inverse=True) == "opp"  # lower wins (ERA/WHIP/L)
    assert _cat_win(3.0, 3.0, inverse=False) == ""  # tie


def test_date_tabs_starts_live_totals():
    tabs = _date_tabs(7)
    assert tabs[0] == "Live"
    assert tabs[1] == "Totals"
    assert len(tabs) >= 3


def test_to_match_player_builds_ref_and_stats():
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "positions": "OF",
                "mlb_id": 592450,
                "team": "NYY",
                "is_hitter": True,
                "h": 120,
                "ab": 410,
                "r": 70,
                "hr": 24,
                "rbi": 80,
                "sb": 12,
                "avg": 0.293,
                "obp": 0.371,
            }
        ]
    )
    mp = _to_match_player(1, "OF", pool, hitter=True, state="final", status="Final")
    assert mp.player.mlb_id == 592450
    assert mp.player.team_id == 147
    assert mp.pos == "OF"
    assert mp.state == "final"
    assert mp.stats == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_pair_rows_zips_sides_and_pads():
    import pandas as pd

    pool = pd.DataFrame(
        [
            {"player_id": i, "name": f"P{i}", "positions": "OF", "mlb_id": i, "team": "NYY", "is_hitter": True}
            for i in (1, 2, 3)
        ]
    )
    you = [_to_match_player(1, "OF", pool, True, "none", ""), _to_match_player(2, "OF", pool, True, "none", "")]
    opp = [_to_match_player(3, "OF", pool, True, "none", "")]
    rows = _pair_rows(you, opp, ["OF", "OF"])
    assert len(rows) == 2
    assert rows[0].you is not None and rows[0].opp is not None
    assert rows[1].you is not None and rows[1].opp is None  # opp padded


def test_matchup_endpoint_includes_roster_tables():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.matchup import MatchPlayer, MatchupResponse, RosterRow
    from api.deps import get_matchup_service
    from api.main import create_app

    class _Fake:
        def get_matchup(self, team_name):
            mp = MatchPlayer(
                player=PlayerRef(id=1, mlb_id=592450, name="Judge", positions="OF", team_abbr="NYY", team_id=147),
                pos="OF",
                state="final",
                stats=["1/4", "1", "0", "0", "0", ".250", ".300"],
            )
            return MatchupResponse(
                team_name=team_name,
                opponent="Rivals",
                week=7,
                hitter_columns=["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"],
                pitcher_columns=["IP", "W", "L", "SV", "K", "ERA", "WHIP"],
                date_tabs=["Live", "Totals"],
                hitters=[RosterRow(slot="OF", you=mp, opp=None)],
            )

    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/matchup?team_name=Team+Hickey").json()
        assert body["hitter_columns"][0] == "H/AB"
        assert body["hitters"][0]["you"]["player"]["mlb_id"] == 592450
        assert body["hitters"][0]["you"]["stats"][0] == "1/4"
    finally:
        app.dependency_overrides.clear()
