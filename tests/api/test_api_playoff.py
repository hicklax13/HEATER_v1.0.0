"""playoff-odds: endpoint contract test + DB-free service unit tests.

The service imports src engines lazily inside methods, so unit tests monkeypatch
them at their SOURCE module (the worktree/CI DB is empty — see the
reference_worktree_empty_db memory)."""

from __future__ import annotations

import pandas as pd
from starlette.testclient import TestClient

from api.contracts.playoff import PlayoffOddsResponse, PlayoffTeam
from api.deps import get_playoff_service
from api.main import create_app
from api.services.playoff_service import PlayoffService


class _FakePlayoffService:
    def get_playoff_odds(self, team_name: str, n_sims: int = 4000) -> PlayoffOddsResponse:
        you = PlayoffTeam(
            team=team_name,
            playoff_odds=62.5,
            projected_wins=14.0,
            projected_record="14-9-3",
            current_wins=7,
            rank=4,
            in_cut=True,
            is_user=True,
        )
        return PlayoffOddsResponse(team_name=team_name, playoff_spots=4, you=you, league=[you], n_sims=4000)


def test_playoff_endpoint_contract():
    app = create_app()
    app.dependency_overrides[get_playoff_service] = lambda: _FakePlayoffService()
    try:
        body = TestClient(app).get("/api/playoff-odds", params={"team_name": "Team Hickey"}).json()
        assert body["team_name"] == "Team Hickey"
        assert body["playoff_spots"] == 4
        assert body["you"]["playoff_odds"] == 62.5 and body["you"]["in_cut"] is True
        assert body["league"][0]["is_user"] is True
    finally:
        app.dependency_overrides.clear()


def _svc():
    return PlayoffService()


def test_to_rows_sorts_ranks_and_marks_user():
    sim = {
        "playoff_probability": {"A": 0.20, "B": 0.95, "🔥 Team Hickey": 0.55, "D": 0.05, "E": 0.80},
        "projected_records": {
            "A": {"W": 10, "L": 14, "T": 2},
            "B": {"W": 20, "L": 5, "T": 1},
            "🔥 Team Hickey": {"W": 14, "L": 10, "T": 2},
            "D": {"W": 6, "L": 18, "T": 2},
            "E": {"W": 17, "L": 8, "T": 1},
        },
    }
    current_wins = {"B": 11, "🔥 Team Hickey": 7}
    # the user passes the plain name; the sim uses the emoji-prefixed Yahoo name → normalize-match
    rows = _svc()._to_rows(sim, current_wins, "Team Hickey")
    assert [r.team for r in rows] == ["B", "E", "🔥 Team Hickey", "A", "D"]  # sorted by odds desc
    assert [r.rank for r in rows] == [1, 2, 3, 4, 5]
    assert [r.in_cut for r in rows] == [True, True, True, True, False]  # top-4
    user = next(r for r in rows if r.is_user)
    assert user.team == "🔥 Team Hickey" and user.playoff_odds == 55.0
    assert user.projected_record == "14-10-2" and user.current_wins == 7


def test_to_rows_record_string_matches_wlt_object_on_half_boundary():
    """M-6: `projected_record` string must agree exactly with the structured
    `projected_record_wlt`. The sim returns FRACTIONAL W/L/T; the old code used
    `f"{w:.0f}"` (round-half-even) for the string but `int(w)` (truncate) for the
    object → off-by-1 on the .5 boundary. Both must derive from the SAME rounded ints."""
    sim = {
        "playoff_probability": {"Rembow": 0.9},
        # 18.5 / 4.5 / 1.5 — every component sits ON the .5 boundary where
        # round-half-even and truncate diverge ("18-4-2" string vs {18,4,1} object).
        "projected_records": {"Rembow": {"W": 18.5, "L": 4.5, "T": 1.5}},
    }
    rows = _svc()._to_rows(sim, current_wins={}, team_name="X")
    r = rows[0]
    wlt = r.projected_record_wlt
    assert r.projected_record == f"{wlt.wins}-{wlt.losses}-{wlt.ties}"


def test_team_weekly_totals_divides_counting_by_26(monkeypatch):
    monkeypatch.setattr(
        "src.database.load_league_rosters",
        lambda: pd.DataFrame({"team_name": ["A", "A"], "player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        "src.database.load_player_pool", lambda: pd.DataFrame({"player_id": [1, 2], "name": ["x", "y"]})
    )
    # HR is a counting cat (→ /26); AVG is a rate cat (→ passthrough)
    monkeypatch.setattr(
        "src.standings_utils.get_all_team_totals",
        lambda **k: {"A": {"HR": 260.0, "AVG": 0.275}},
    )
    weekly = _svc()._team_weekly_totals()
    assert weekly["A"]["HR"] == 10.0  # 260 / 26
    assert weekly["A"]["AVG"] == 0.275  # rate passthrough


def test_current_week_from_games_played():
    svc = _svc()
    assert svc._current_week({"A": {"W": 6, "L": 5, "T": 1}, "B": {"W": 4, "L": 7, "T": 1}}) == 13  # 12 played + 1
    assert svc._current_week({}) == 1


def test_current_standings_skips_bad_rows(monkeypatch):
    # a NaN/blank row must be skipped, not zero the whole panel (per-row guard).
    monkeypatch.setattr(
        "src.database.load_league_records",
        lambda: pd.DataFrame(
            {
                "team_name": ["Good", "", "Bad"],
                "wins": [7, 3, float("nan")],
                "losses": [4, 2, 1],
                "ties": [1, 0, 0],
            }
        ),
    )
    standings, wins = _svc()._current_standings()
    assert standings == {"Good": {"W": 7, "L": 4, "T": 1}}  # blank-name + NaN-wins rows dropped
    assert wins == {"Good": 7}


def test_get_playoff_odds_cold_env_returns_empty(monkeypatch):
    # empty league_rosters → no weekly totals → empty response (never raises)
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    resp = _svc().get_playoff_odds("Team Hickey")
    assert resp.team_name == "Team Hickey" and resp.league == [] and resp.you is None


def test_get_playoff_odds_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "src.database.load_league_rosters",
        lambda: pd.DataFrame({"team_name": ["Team Hickey", "Rival"], "player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        "src.database.load_player_pool", lambda: pd.DataFrame({"player_id": [1, 2], "name": ["x", "y"]})
    )
    monkeypatch.setattr(
        "src.standings_utils.get_all_team_totals",
        lambda **k: {"Team Hickey": {"HR": 260.0}, "Rival": {"HR": 130.0}},
    )
    monkeypatch.setattr(
        "src.database.load_league_records",
        lambda: pd.DataFrame({"team_name": ["Team Hickey", "Rival"], "wins": [7, 5], "losses": [4, 6], "ties": [1, 1]}),
    )
    monkeypatch.setattr("src.database.load_league_schedule_full", lambda: {13: [("Team Hickey", "Rival")]})
    monkeypatch.setattr(
        "src.standings_engine.simulate_season_enhanced",
        lambda **k: {
            "playoff_probability": {"Team Hickey": 0.6, "Rival": 0.3},
            "projected_records": {
                "Team Hickey": {"W": 15, "L": 9, "T": 2},
                "Rival": {"W": 11, "L": 13, "T": 2},
            },
            "n_sims": k.get("n_sims", 0),
        },
    )
    resp = _svc().get_playoff_odds("Team Hickey")
    assert resp.you is not None and resp.you.is_user and resp.you.playoff_odds == 60.0
    assert [r.team for r in resp.league] == ["Team Hickey", "Rival"]  # sorted by odds
    assert resp.you.rank == 1 and resp.you.in_cut is True
    assert resp.n_sims == 2000  # the API default (tuned for read responsiveness)
