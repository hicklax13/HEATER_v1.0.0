from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.compare import ComparePlayer, CompareResponse
from api.deps import get_compare_service
from api.main import create_app


def test_compare_contract_shape():
    resp = CompareResponse(
        categories=["HR", "SB", "AVG"],
        players=[
            ComparePlayer(
                player=PlayerRef(id=1, name="Shohei Ohtani", positions="SP/DH"),
                stats={"HR": 40.0, "SB": 10.0, "AVG": 0.290},
            ),
            ComparePlayer(
                player=PlayerRef(id=2, name="Ronald Acuna", positions="OF"),
                stats={"HR": 25.0, "SB": 65.0, "AVG": 0.310},
            ),
        ],
    )
    dumped = resp.model_dump()
    assert dumped["categories"] == ["HR", "SB", "AVG"]
    assert dumped["players"][0]["player"]["name"] == "Shohei Ohtani"
    assert dumped["players"][1]["stats"]["SB"] == 65.0
    # defaults
    assert ComparePlayer(player=PlayerRef(id=1, name="X", positions="")).stats == {}
    assert CompareResponse().categories == []
    assert CompareResponse().players == []


class _FakeCompareService:
    def compare(self, player_ids: list[int]) -> CompareResponse:
        return CompareResponse(
            categories=["HR", "AVG"],
            players=[
                ComparePlayer(
                    player=PlayerRef(id=player_ids[0], name="Alpha", positions="1B"),
                    stats={"HR": 30.0, "AVG": 0.280},
                ),
                ComparePlayer(
                    player=PlayerRef(id=player_ids[1], name="Beta", positions="OF"),
                    stats={"HR": 20.0, "AVG": 0.300},
                ),
            ],
        )


def test_get_compare_returns_contract():
    app = create_app()
    app.dependency_overrides[get_compare_service] = lambda: _FakeCompareService()
    client = TestClient(app)
    resp = client.get("/api/compare?ids=101,202")
    assert resp.status_code == 200
    body = resp.json()
    assert body["players"][0]["player"]["name"] == "Alpha"
    assert body["categories"] == ["HR", "AVG"]
    assert body["players"][1]["player"]["name"] == "Beta"
