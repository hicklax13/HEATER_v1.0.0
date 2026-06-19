"""FA-pool endpoint: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.fa_pool_service import _key_stats, _tag_from, _to_pool_item, _top_need


def test_top_need_picks_most_negative_gap():
    # gap < 0 means the user is behind in that category (a need)
    assert _top_need({"R": 1.2, "SB": -3.4, "HR": -0.5}) == "SB"
    assert _top_need({}) == ""


def test_tag_from_regression_flag():
    assert _tag_from("BUY_LOW") == "Buy Low"
    assert _tag_from("SELL_HIGH") == "Sell High"
    assert _tag_from(None) is None
    assert _tag_from("") is None


def test_key_stats_hitter_and_pitcher():
    hitter = {"ytd_hr": 24, "ytd_sb": 11, "ytd_avg": 0.285}
    s = _key_stats(hitter, True)
    assert [x.label for x in s] == ["HR", "SB", "AVG"]
    assert s[0].value == "24"
    # format_stat(0.285, "AVG") strips the leading zero for 0 < value < 1 → ".285"
    assert s[2].value == ".285"
    pitcher = {"ytd_k": 180, "ytd_era": 3.21, "ytd_whip": 1.05}
    p = _key_stats(pitcher, False)
    assert [x.label for x in p] == ["K", "ERA", "WHIP"]
    assert p[1].value == "3.21"


def test_to_pool_item_normalizes_value_and_enriches():
    pool = pd.DataFrame([{"player_id": 1, "name": "Judge", "positions": "OF", "mlb_id": 592450, "team": "NYY"}])
    row = {
        "player_id": 1,
        "player_name": "Judge",
        "positions": "OF",
        "is_hitter": True,
        "marginal_value": 5.0,
        "best_category": "HR",
        "regression_flag": "BUY_LOW",
        "percent_owned": 60.0,
        "ytd_hr": 30,
        "ytd_sb": 5,
        "ytd_avg": 0.31,
    }
    item = _to_pool_item(2, row, pool, max_value=10.0)
    assert item.rank == 2
    assert item.value == 50.0  # 5/10*100
    assert item.player.mlb_id == 592450
    assert item.player.team_id == 147
    assert item.player.name == "Judge"
    assert item.own_pct == 60.0
    assert item.own_delta == 0.0
    assert item.hitter is True
    assert item.fit == "HR"
    assert item.tag == "Buy Low"
    assert item.stats[0].value == "30"


def test_to_pool_item_zero_max_value_is_safe():
    pool = pd.DataFrame([{"player_id": 1, "name": "X", "positions": "OF", "mlb_id": 1, "team": "NYY"}])
    item = _to_pool_item(1, {"player_id": 1, "marginal_value": 0.0, "is_hitter": True}, pool, max_value=0.0)
    assert item.value == 0.0  # no divide-by-zero


def test_free_agents_pool_endpoint_returns_contract():
    from starlette.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.free_agents import FreeAgentPoolItem, FreeAgentPoolResponse, StatItem
    from api.deps import get_fa_pool_service
    from api.main import create_app

    class _FakePoolService:
        def get_free_agents_pool(self, team_name, limit=100):
            return FreeAgentPoolResponse(
                top_need="SB",
                free_agents=[
                    FreeAgentPoolItem(
                        player=PlayerRef(
                            id=1,
                            mlb_id=592450,
                            name="Aaron Judge",
                            positions="OF",
                            team_abbr="NYY",
                            team_id=147,
                        ),
                        rank=1,
                        value=100.0,
                        own_pct=45.0,
                        own_delta=0.0,
                        hitter=True,
                        stats=[StatItem(label="HR", value="24")],
                        fit="HR",
                        tag="Buy Low",
                    )
                ],
            )

    app = create_app()
    app.dependency_overrides[get_fa_pool_service] = lambda: _FakePoolService()
    try:
        resp = TestClient(app).get("/api/free-agents/pool?team_name=Team+Hickey")
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_need"] == "SB"
        fa = body["free_agents"][0]
        assert fa["player"]["mlb_id"] == 592450
        assert fa["player"]["team_id"] == 147
        assert fa["value"] == 100.0
        assert fa["fit"] == "HR"
        assert fa["stats"][0] == {"label": "HR", "value": "24"}
    finally:
        app.dependency_overrides.clear()
