"""player-id endpoints: endpoint contract + DB-free service unit tests.

The service imports src loaders lazily inside methods → monkeypatch at the SOURCE
module (worktree/CI DB is empty — see the reference_worktree_empty_db memory)."""

from __future__ import annotations

import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.players import LeagueRostersResponse, LeagueRosterTeam, PlayerSearchResponse
from api.deps import get_roster_query_service
from api.main import create_app
from api.services.roster_query_service import RosterQueryService


class _FakeRosterQuery:
    def search(self, q: str, limit: int = 25) -> PlayerSearchResponse:
        return PlayerSearchResponse(
            query=q,
            results=[PlayerRef(id=1, mlb_id=545361, name="Mike Trout", positions="OF", team_abbr="LAA", team_id=108)],
        )

    def league_rosters(self) -> LeagueRostersResponse:
        return LeagueRostersResponse(
            teams=[
                LeagueRosterTeam(
                    team_name="Team Hickey",
                    manager="Connor",
                    players=[
                        PlayerRef(id=2, mlb_id=592450, name="Judge", positions="OF", team_abbr="NYY", team_id=147)
                    ],
                )
            ]
        )


def test_players_endpoints_contract():
    app = create_app()
    app.dependency_overrides[get_roster_query_service] = lambda: _FakeRosterQuery()
    try:
        client = TestClient(app)
        s = client.get("/api/players/search", params={"q": "trout"}).json()
        assert s["query"] == "trout" and s["results"][0]["mlb_id"] == 545361
        r = client.get("/api/league/rosters").json()
        assert r["teams"][0]["team_name"] == "Team Hickey"
        assert r["teams"][0]["manager"] == "Connor"
        assert r["teams"][0]["players"][0]["id"] == 2
    finally:
        app.dependency_overrides.clear()


def _svc():
    return RosterQueryService()


def test_search_filters_by_name_sorts_by_relevance_and_enriches(monkeypatch):
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Mike Trout",
                "positions": "OF",
                "mlb_id": 545361,
                "team": "LAA",
                "consensus_rank": 8,
            },
            {
                "player_id": 2,
                "name": "Steven Trout",
                "positions": "SP",
                "mlb_id": 700001,
                "team": "CHC",
                "consensus_rank": 410,
            },
            {
                "player_id": 3,
                "name": "Aaron Judge",
                "positions": "OF",
                "mlb_id": 592450,
                "team": "NYY",
                "consensus_rank": 3,
            },
        ]
    )
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    resp = _svc().search("trout", limit=25)
    # only the two Trouts match, most fantasy-relevant (lower consensus_rank) first
    assert [r.name for r in resp.results] == ["Mike Trout", "Steven Trout"]
    assert resp.results[0].mlb_id == 545361 and resp.results[0].team_id == 108  # enriched LAA


def test_search_respects_limit_and_clamps(monkeypatch):
    pool = pd.DataFrame(
        [
            {"player_id": i, "name": f"Smith {i}", "positions": "OF", "mlb_id": 1000 + i, "team": "NYY"}
            for i in range(10)
        ]
    )
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    assert len(_svc().search("smith", limit=3).results) == 3
    assert len(_svc().search("smith", limit=99999).results) == 10  # clamped to ≤200, not the whole pool
    assert len(_svc().search("smith", limit=0).results) >= 1  # clamp floors at 1 (no zero/negative scan)


def test_search_empty_query_and_no_match():
    assert _svc().search("   ").results == []  # blank query → no pool load, empty
    # non-matching query (with a real pool patched) → empty


def test_search_no_match_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "src.database.load_player_pool",
        lambda: pd.DataFrame([{"player_id": 1, "name": "Aaron Judge", "positions": "OF", "mlb_id": 592450}]),
    )
    assert _svc().search("zzzznobody").results == []


def test_league_rosters_groups_and_enriches(monkeypatch):
    lr = pd.DataFrame(
        {
            "team_name": ["Team Hickey", "Team Hickey", "Rival"],
            "player_id": [1, 2, 3],
            "name": ["Judge", "Soto", "Acuna"],
            "positions": ["OF", "OF", "OF"],
        }
    )
    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "Judge", "positions": "OF", "mlb_id": 592450, "team": "NYY"},
            {"player_id": 2, "name": "Soto", "positions": "OF", "mlb_id": 665742, "team": "NYM"},
            {"player_id": 3, "name": "Acuna", "positions": "OF", "mlb_id": 660670, "team": "ATL"},
        ]
    )
    monkeypatch.setattr("src.database.load_league_rosters", lambda: lr)
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    monkeypatch.setattr(RosterQueryService, "_managers", staticmethod(lambda: {"Team Hickey": "Connor"}))
    resp = _svc().league_rosters()
    by_team = {t.team_name: t for t in resp.teams}
    assert set(by_team) == {"Team Hickey", "Rival"}
    assert by_team["Team Hickey"].manager == "Connor"
    assert {p.id for p in by_team["Team Hickey"].players} == {1, 2}
    assert by_team["Rival"].players[0].mlb_id == 660670  # enriched from pool


def test_league_rosters_cold_env_returns_empty(monkeypatch):
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    assert _svc().league_rosters().teams == []
