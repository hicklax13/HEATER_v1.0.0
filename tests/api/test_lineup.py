import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.lineup import CatImpact, DailyMeta, IpPace, LineupOptimizeResponse, LineupSlot, Swap
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
    def optimize(
        self, team_name: str, date=None, scope: str = "rest_of_season", mode: str = "standard"
    ) -> LineupOptimizeResponse:
        if str(mode).lower() == "daily":
            return LineupOptimizeResponse(
                team_name=team_name,
                date=date or "2027-04-05",
                slots=[
                    LineupSlot(
                        slot="OF",
                        player=PlayerRef(id=1, name="A. Player", positions="OF"),
                        action="START",
                        status="start",
                        value=100.0,
                        matchup="vs SF",
                        current_slot="BN",
                    )
                ],
                summary="1 starter set for 2027-04-05.",
                mode="daily",
                daily=DailyMeta(
                    urgency={"hr": 0.6},
                    rate_modes={"ERA": "compete"},
                    winning=["AVG"],
                    losing=["ERA"],
                    tied=["HR"],
                    ip_pace=IpPace(projected=48.3, target=53.8, pace_pct=90, status="safe", message="ok"),
                    recommendations=["Stream a SP Thursday."],
                    swaps=[Swap(player=PlayerRef(id=1, name="A. Player", positions="OF"), slot="OF", value=100.0)],
                ),
            )
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


# ── slice-2 daily mode (DB-free; synthetic daily_dcv + daily_lineup) ──


def test_post_lineup_optimize_daily_returns_daily_meta():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    client = TestClient(app)
    body = client.post("/api/lineup/optimize", json={"team_name": "Team Hickey", "mode": "daily"}).json()
    assert body["mode"] == "daily"
    s0 = body["slots"][0]
    assert s0["value"] == 100.0 and s0["matchup"] == "vs SF" and s0["current_slot"] == "BN"
    d = body["daily"]
    assert d["losing"] == ["ERA"] and d["rate_modes"]["ERA"] == "compete"
    assert d["ip_pace"]["pace_pct"] == 90
    assert d["swaps"][0]["player"]["id"] == 1  # benched player the optimizer wants to start


def _svc():
    return LineupService()


def test_daily_slots_maps_value_matchup_decision():
    # total_dcv joins daily_lineup → daily_dcv (player_id-bearing); value normalized to best=100
    dcv = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Judge",
                "positions": "OF",
                "team": "NYY",
                "total_dcv": 20.0,
                "matchup_mult": 1.05,
                "reason": "",
            },
            {
                "player_id": 2,
                "name": "Soto",
                "positions": "OF",
                "team": "NYY",
                "total_dcv": 10.0,
                "matchup_mult": 0.60,
                "reason": "",
            },
            {
                "player_id": 3,
                "name": "Cole",
                "positions": "SP",
                "team": "NYY",
                "total_dcv": 0.0,
                "matchup_mult": 1.0,
                "reason": "OFF_DAY",
            },
        ]
    )
    daily_lineup = {
        "starters": [
            {"name": "Judge", "slot": "OF", "total_dcv": 20.0},
            {"name": "Soto", "slot": "OF", "total_dcv": 10.0},
        ],
        "bench": [{"name": "Cole", "total_dcv": 0.0}],
    }
    roster = pd.DataFrame({"player_id": [1, 2, 3], "selected_position": ["OF", "BN", "BN"]})
    # statsapi.schedule shape = FULL team names; dcv team is the abbr "NYY"
    schedule = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox"}]
    starters, bench = _svc()._daily_slots(dcv, daily_lineup, roster, None, schedule)

    by_id = {s.player.id: s for s in starters}
    assert set(by_id) == {1, 2}
    assert by_id[1].value == 100.0 and by_id[2].value == 50.0  # normalized to the best play
    assert by_id[1].matchup == "vs BOS" and by_id[1].current_slot == "OF"
    assert by_id[2].forced_start is True  # started despite matchup_mult 0.60 < 0.70
    assert by_id[1].forced_start is False
    assert all(s.status == "start" and s.action == "START" for s in starters)
    # Cole: not in the LP starters → bench, carries the engine's reason
    assert [b.player.id for b in bench] == [3]
    assert bench[0].status == "bench" and bench[0].reason == "OFF_DAY" and bench[0].value == 0.0


def test_daily_slots_empty_dcv_returns_empty():
    assert _svc()._daily_slots(None, {}, None, None, []) == ([], [])
    assert _svc()._daily_slots(pd.DataFrame(), {}, None, None, []) == ([], [])


def test_matchup_str_home_away_and_unknown():
    # real statsapi.schedule shape (full names); dcv `team` is an abbr → must canonicalize both
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox"}]
    assert LineupService._matchup_str("NYY", sched) == "vs BOS"
    assert LineupService._matchup_str("BOS", sched) == "@ NYY"
    assert LineupService._matchup_str("LAD", sched) == ""  # not playing
    assert LineupService._matchup_str("NYY", None) == ""


def test_days_remaining_bounds_and_invalid():
    assert LineupService._days_remaining("garbage") == 7
    assert LineupService._days_remaining("") == 7
    assert 1 <= LineupService._days_remaining("2026-06-22") <= 7


def test_daily_meta_builds_swaps_and_summary():
    # real engine shape (pipeline.py:514-516): urgency_weights is the FLAT {cat: weight} dict,
    # rate_stat_modes + matchup_summary are separate top-level keys
    result = {
        "urgency_weights": {"hr": 0.6, "era": float("nan")},  # NaN urgency must be dropped
        "rate_stat_modes": {"ERA": "compete"},
        "matchup_summary": {"winning": ["AVG"], "losing": ["ERA"], "tied": ["HR"]},
        "recommendations": ["Stream a SP Thursday."],
    }
    started = LineupSlot(
        slot="OF", player=PlayerRef(id=9, name="X", positions="OF"), action="START", status="start", current_slot="BN"
    )
    benched_start = LineupSlot(
        slot="OF", player=PlayerRef(id=8, name="Y", positions="OF"), action="START", status="start", current_slot="OF"
    )
    meta = _svc()._daily_meta(result, [started, benched_start], roster=None, pool=None, resolved_date="2026-06-20")
    assert meta.urgency == {"hr": 0.6}  # NaN dropped
    assert meta.rate_modes["ERA"] == "compete"
    assert meta.losing == ["ERA"] and meta.winning == ["AVG"] and meta.tied == ["HR"]
    assert meta.recommendations == ["Stream a SP Thursday."]
    # only the currently-benched START is a swap
    assert [sw.player.id for sw in meta.swaps] == [9]
    assert meta.ip_pace is None  # roster=None → no IP pace


def test_ip_pace_filters_pitchers_and_merges_pool_ip(monkeypatch):
    captured = {}

    def _fake_proj(pitchers, days_remaining=7):
        captured["pitchers"] = pitchers
        captured["days"] = days_remaining
        return {"projected_ip": 48.3, "ip_target": 53.8, "ip_pace": 90, "status": "safe", "message": "ok"}

    monkeypatch.setattr("src.ip_tracker.compute_weekly_ip_projection", _fake_proj)
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Judge", "Cole", "Holmes"],
            "positions": ["OF", "SP", "RP"],  # Judge filtered out (no SP/RP/P)
            "status": ["", "", ""],
        }
    )
    pool = pd.DataFrame({"player_id": [2, 3], "ip": [180.0, 65.0]})
    out = _svc()._ip_pace(roster, pool, "2026-06-20")
    assert out is not None and out.pace_pct == 90 and out.projected == 48.3
    names = {p["name"] for p in captured["pitchers"]}
    assert names == {"Cole", "Holmes"}  # only pitchers
    cole = next(p for p in captured["pitchers"] if p["name"] == "Cole")
    assert cole["ip"] == 180.0 and cole["is_starter"] is True  # pool IP merged; SP flagged
