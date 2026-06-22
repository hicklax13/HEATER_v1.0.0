import pandas as pd
from starlette.testclient import TestClient

from api.contracts.standings import StandingsResponse, TeamStanding
from api.deps import get_standings_service
from api.main import create_app
from api.services.standings_service import StandingsService


def _standings_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a long-format league_standings frame: (team_name, category, total, rank).
    `rank` is the team's CLONED overall rank (every row), mirroring _sync_standings."""
    return pd.DataFrame(rows)


# Three teams. Overall rank is cloned into every category row (as the sync writes it).
# A is overall #1 but WORST in ERA, B is overall #2 but BEST in ERA — so a correct
# per-category rank must DIFFER from the cloned overall rank.
_STANDINGS_ROWS = [
    # Team A — overall rank 1, HR best (100), ERA worst (5.00)
    {"team_name": "Team A", "category": "HR", "total": 100.0, "rank": 1},
    {"team_name": "Team A", "category": "ERA", "total": 5.00, "rank": 1},
    # Team B — overall rank 2, HR mid (80), ERA best (3.00, lowest)
    {"team_name": "Team B", "category": "HR", "total": 80.0, "rank": 2},
    {"team_name": "Team B", "category": "ERA", "total": 3.00, "rank": 2},
    # Team C — overall rank 3, HR worst (60), ERA mid (4.00)
    {"team_name": "Team C", "category": "HR", "total": 60.0, "rank": 3},
    {"team_name": "Team C", "category": "ERA", "total": 4.00, "rank": 3},
]


def _team(teams: list[TeamStanding], name: str) -> TeamStanding:
    return next(t for t in teams if t.team_name == name)


def test_build_teams_per_category_ranks_not_cloned_overall_rank():
    """The core HIGH bug: per-category rank must come from `total`, not the cloned
    overall rank. Team A is overall #1 but worst in ERA → ERA rank must be 3, not 1."""
    teams = StandingsService._build_teams(_standings_frame(_STANDINGS_ROWS), pd.DataFrame())
    a = _team(teams, "Team A")
    b = _team(teams, "Team B")
    assert a.category_ranks["HR"] == 1  # A has the most HR
    assert a.category_ranks["ERA"] == 3  # A has the worst ERA (inverse) — NOT 1 (its overall rank)
    assert b.category_ranks["ERA"] == 1  # B has the best (lowest) ERA — NOT 2 (its overall rank)
    assert b.category_ranks["HR"] == 2


def test_build_teams_reads_wlt_from_records_table():
    """W-L-T come from load_league_records() output (the Railway path)."""
    records = pd.DataFrame(
        [
            {"team_name": "Team A", "wins": 8, "losses": 4, "ties": 1, "rank": 1},
            {"team_name": "Team B", "wins": 6, "losses": 6, "ties": 1, "rank": 2},
        ]
    )
    teams = StandingsService._build_teams(_standings_frame(_STANDINGS_ROWS), records)
    a = _team(teams, "Team A")
    assert (a.wins, a.losses, a.ties) == (8, 4, 1)
    assert a.rank == 1


def test_build_teams_falls_back_to_wlt_categories_when_records_empty():
    """Local/old data path: records table empty, but WINS/LOSSES/TIES live as
    categories in league_standings — derive W-L-T from their `total`."""
    rows = _STANDINGS_ROWS + [
        {"team_name": "Team A", "category": "WINS", "total": 7.0, "rank": 1},
        {"team_name": "Team A", "category": "LOSSES", "total": 5.0, "rank": 1},
        {"team_name": "Team A", "category": "TIES", "total": 0.0, "rank": 1},
    ]
    teams = StandingsService._build_teams(_standings_frame(rows), pd.DataFrame())
    a = _team(teams, "Team A")
    assert (a.wins, a.losses, a.ties) == (7, 5, 0)
    # Meta categories must NOT appear as ranked stat categories.
    assert "WINS" not in a.category_ranks
    assert "LOSSES" not in a.category_ranks
    assert "TIES" not in a.category_ranks


def test_build_teams_records_overall_rank_drives_sort():
    """Overall rank comes from records when present; teams sort by it ascending."""
    records = pd.DataFrame(
        [
            {"team_name": "Team A", "wins": 8, "losses": 4, "ties": 0, "rank": 1},
            {"team_name": "Team B", "wins": 7, "losses": 5, "ties": 0, "rank": 2},
            {"team_name": "Team C", "wins": 3, "losses": 9, "ties": 0, "rank": 3},
        ]
    )
    teams = StandingsService._build_teams(_standings_frame(_STANDINGS_ROWS), records)
    assert [t.team_name for t in teams] == ["Team A", "Team B", "Team C"]
    assert [t.rank for t in teams] == [1, 2, 3]


def test_standings_contract_shape():
    resp = StandingsResponse(
        teams=[
            TeamStanding(
                rank=1,
                team_name="Team Hickey",
                wins=8,
                losses=4,
                ties=0,
                points=62.5,
                category_ranks={"HR": 1, "SB": 3},
            )
        ]
    )
    dumped = resp.model_dump()
    assert dumped["teams"][0]["team_name"] == "Team Hickey"
    assert dumped["teams"][0]["category_ranks"]["HR"] == 1
    # defaults
    assert TeamStanding(rank=2, team_name="Other").wins == 0
    assert TeamStanding(rank=2, team_name="Other").category_ranks == {}


class _FakeStandingsService:
    def get_standings(self) -> StandingsResponse:
        return StandingsResponse(
            teams=[
                TeamStanding(
                    rank=1,
                    team_name="Team Hickey",
                    wins=8,
                    losses=4,
                    ties=0,
                    points=62.5,
                    category_ranks={"HR": 1},
                )
            ]
        )


def test_get_standings_returns_contract():
    app = create_app()
    app.dependency_overrides[get_standings_service] = lambda: _FakeStandingsService()
    client = TestClient(app)
    resp = client.get("/api/standings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["teams"][0]["team_name"] == "Team Hickey"
    assert body["teams"][0]["rank"] == 1
