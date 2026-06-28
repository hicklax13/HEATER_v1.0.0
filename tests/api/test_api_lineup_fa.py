"""WS1 DI test: fa_suggestions round-trips through POST /api/lineup/optimize."""

from starlette.testclient import TestClient

from api.contracts.common import PlayerRef, StatItem
from api.contracts.lineup import FaSuggestion, LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app


class _FakeLineupService:
    def optimize(self, team_name, date=None, scope="rest_of_season", mode="standard", depth="standard"):
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="Starter", positions="OF"),
                    action="START",
                    status="start",
                    current_slot="BN",
                )
            ],
            summary="1 starter",
            mode="standard" if scope != "today" else "daily",
            fa_suggestions=[
                FaSuggestion(
                    add=PlayerRef(id=10, name="Add Guy", positions="OF"),
                    drop=PlayerRef(id=20, name="Drop Guy", positions="2B"),
                    net_sgp_delta=1.4,
                    category_impact=[StatItem(label="HR", value="+0.30")],
                    reasoning="Upgrades HR.",
                    urgency_categories=["HR"],
                )
            ],
        )


def _client():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    return TestClient(app)


def test_optimize_returns_fa_suggestions():
    body = _client().post("/api/lineup/optimize", json={"team_name": "Team Hickey", "scope": "rest_of_week"}).json()
    assert body["mode"] == "standard"
    fs = body["fa_suggestions"]
    assert len(fs) == 1
    assert fs[0]["add"]["id"] == 10 and fs[0]["drop"]["id"] == 20
    assert fs[0]["category_impact"][0] == {"label": "HR", "value": "+0.30"}
    assert fs[0]["urgency_categories"] == ["HR"]
    # current_slot present on the lineup slot (layout grouping key)
    assert body["slots"][0]["current_slot"] == "BN"


def test_optimize_scope_today_echoes_daily_mode():
    body = _client().post("/api/lineup/optimize", json={"team_name": "Team Hickey", "scope": "today"}).json()
    assert body["mode"] == "daily"
