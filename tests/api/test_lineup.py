import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.lineup import CatImpact, LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app
from api.services.lineup_service import LineupService


def test_lineup_contract_shape():
    resp = LineupOptimizeResponse(
        team_name="Team Hickey",
        date="2027-04-05",
        slots=[
            LineupSlot(
                slot="OF",
                player=PlayerRef(id=1, name="A. Player", positions="OF"),
                action="START",
                projected=4.2,
                forced_start=False,
                reason=None,
            )
        ],
        summary="9 starters set; 0 forced.",
    )
    dumped = resp.model_dump()
    assert dumped["slots"][0]["action"] == "START"
    assert dumped["slots"][0]["player"]["name"] == "A. Player"
    assert dumped["slots"][0]["status"] == "start"  # slice-1 field defaults


class _FakeLineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="A. Player", positions="OF"),
                    action="START",
                    projected=4.2,
                    status="start",
                )
            ],
            summary="1 starter",
            bench=[
                LineupSlot(
                    slot="BN",
                    player=PlayerRef(id=2, name="B. Bench", positions="2B"),
                    action="SIT",
                    projected=0.0,
                    status="bench",
                )
            ],
            optimal=False,
            impact=[CatImpact(key="HR", proj="2", trend="flat")],
        )


def test_post_lineup_optimize_returns_contract():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    client = TestClient(app)
    resp = client.post("/api/lineup/optimize", json={"team_name": "Team Hickey", "date": "2027-04-05"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["slots"][0]["action"] == "START"
    # slice-1 enrichment fields round-trip
    assert body["slots"][0]["status"] == "start"
    assert body["bench"][0]["status"] == "bench" and body["bench"][0]["player"]["id"] == 2
    assert body["optimal"] is False
    assert body["impact"][0] == {"key": "HR", "proj": "2", "trend": "flat"}


# ── slice-1 service mapping (DB-free; synthetic OptimizerResult.lineup shape) ──


def _lineup(assignments, projected_stats=None):
    return {"assignments": assignments, "bench": [], "projected_stats": projected_stats or {}, "status": "Optimal"}


def test_to_slots_starters_from_assignments_and_bench_from_roster():
    lineup = _lineup([{"slot": "OF", "player_name": "Judge", "player_id": 1}])
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Judge", "Benchy", "Stashed"],
            "positions": ["OF", "2B", "SP"],
            "selected_position": ["OF", "BN", "IL"],
        }
    )
    starters, bench = LineupService._to_slots(lineup, pool=None, roster=roster)
    assert [s.player.id for s in starters] == [1] and starters[0].status == "start"
    # bench = roster players NOT in the optimal lineup (2 + 3), with status "bench"
    assert {b.player.id for b in bench} == {2, 3}
    assert all(b.status == "bench" and b.action == "SIT" for b in bench)


def test_impact_formats_counting_and_rate():
    impact = {c.key: c for c in LineupService._impact({"r": 6.4, "avg": 0.275, "era": 3.5, "hr": 2.0})}
    assert impact["R"].proj == "6" and impact["R"].trend == "flat"  # counting → int
    assert impact["AVG"].proj == ".275"  # rate, leading-zero stripped
    assert impact["ERA"].proj == "3.50"  # rate, 2dp
    assert impact["HR"].proj == "2"


def test_impact_empty_when_no_stats():
    assert LineupService._impact(None) == []
    assert LineupService._impact({}) == []


def test_impact_skips_non_finite():
    # a NaN/inf projected stat must never reach the JSON as "nan"/"inf"
    impact = {c.key: c for c in LineupService._impact({"r": float("nan"), "hr": float("inf"), "sb": 3.0})}
    assert set(impact) == {"SB"} and impact["SB"].proj == "3"


def test_optimal_true_when_current_starters_match():
    roster = pd.DataFrame({"player_id": [1, 2, 3], "selected_position": ["OF", "2B", "BN"]})
    # optimizer chose starters {1, 2}; current starters (non-BN) are also {1, 2} → optimal
    assert LineupService._optimal(roster, {1, 2}) is True
    # optimizer chose {1, 3} but current starters are {1, 2} → not optimal
    assert LineupService._optimal(roster, {1, 3}) is False
    # no starters / no selected_position → False
    assert LineupService._optimal(roster, set()) is False
    assert LineupService._optimal(pd.DataFrame({"player_id": [1]}), {1}) is False
