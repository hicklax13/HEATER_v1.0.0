from datetime import UTC, datetime, timedelta

import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.my_team import CategoryLine, MatchupHero, Mover, MyTeamResponse, StatusChip
from api.deps import get_team_service
from api.main import create_app


class _FakeTeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        return MyTeamResponse(
            team_name=team_name,
            record="4-7-1",
            rank=10,
            matchup=MatchupHero(opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35),
            categories=[CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18)],
            eyebrow="Season · Week 13 · Team Hickey",
            subline="4-7-1 · 10th of 12 · 3 GB from 1st",
            freshness_minutes=18.0,
            playoff_cut_rank=4,
            status_chips=[StatusChip(label="IL", value=2, status="warn")],
            movers=[
                Mover(
                    player=PlayerRef(id=1, mlb_id=592450, name="Judge", positions="OF", team_abbr="NYY", team_id=147),
                    stats=["18 HR", ".322 AVG"],
                    trend="up",
                    tag="hot",
                    context="Trending hot vs projection",
                )
            ],
            movers_scope="mine",
        )


def test_get_me_team_returns_contract():
    app = create_app()
    app.dependency_overrides[get_team_service] = lambda: _FakeTeamService()
    client = TestClient(app)
    resp = client.get("/api/me/team", params={"team_name": "Team Hickey"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["rank"] == 10
    assert body["matchup"]["opponent"] == "Baty Babies"
    assert body["categories"][0]["cat"] == "SB"
    # slice-1 dashboard fields round-trip
    assert body["eyebrow"].startswith("Season")
    assert body["freshness_minutes"] == 18.0
    assert body["playoff_cut_rank"] == 4
    assert body["status_chips"][0]["label"] == "IL"
    assert body["movers"][0]["player"]["mlb_id"] == 592450
    assert body["movers"][0]["tag"] == "hot"
    assert body["movers_scope"] == "mine"


def test_my_team_contract_shape():
    resp = MyTeamResponse(
        team_name="Team Hickey",
        record="4-7-1",
        rank=10,
        matchup=MatchupHero(opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35),
        categories=[CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18, inverse=False)],
    )
    # win/tie/loss must sum to ~1
    m = resp.matchup
    assert abs((m.win_prob + m.tie_prob + m.loss_prob) - 1.0) < 1e-6
    # round-trips to the JSON shape the frontend consumes
    dumped = resp.model_dump()
    assert dumped["matchup"]["win_prob"] == 0.46
    assert dumped["categories"][0]["cat"] == "SB"


# ── slice-1 service helpers (DB-free: engines monkeypatched at their source modules,
# because TeamService imports them lazily inside methods; the worktree/CI DB is empty
# — see the reference_worktree_empty_db memory) ──────────────────────────────────────


def _real_service():
    from api.services.team_service import TeamService

    return TeamService()


def test_movers_filters_to_roster_caps_4_and_maps_trend(monkeypatch):
    pool = pd.DataFrame(
        [
            {"player_id": i, "name": f"P{i}", "positions": "OF", "mlb_id": 100 + i, "team": "NYY", "is_hitter": True}
            for i in range(1, 7)
        ]
    )
    # 5 HOT/COLD + 1 NEUTRAL; NEUTRAL must be dropped, then head(4).
    trended = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "P1",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 20,
                "ytd_avg": 0.31,
                "trend_label": "HOT",
                "trend_delta": 0.9,
            },
            {
                "player_id": 2,
                "player_name": "P2",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 5,
                "ytd_avg": 0.20,
                "trend_label": "COLD",
                "trend_delta": -0.8,
            },
            {
                "player_id": 3,
                "player_name": "P3",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 12,
                "ytd_avg": 0.27,
                "trend_label": "HOT",
                "trend_delta": 0.5,
            },
            {
                "player_id": 4,
                "player_name": "P4",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 8,
                "ytd_avg": 0.24,
                "trend_label": "COLD",
                "trend_delta": -0.3,
            },
            {
                "player_id": 5,
                "player_name": "P5",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 1,
                "ytd_avg": 0.15,
                "trend_label": "HOT",
                "trend_delta": 0.2,
            },
            {
                "player_id": 6,
                "player_name": "P6",
                "positions": "OF",
                "is_hitter": True,
                "ytd_hr": 9,
                "ytd_avg": 0.26,
                "trend_label": "NEUTRAL",
                "trend_delta": 0.0,
            },
        ]
    )
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame({"player_id": [1, 2, 3]}))
    monkeypatch.setattr("src.trend_tracker.compute_player_trends", lambda *a, **k: trended)

    movers = _real_service()._movers([1, 2, 3, 4, 5, 6], cfg=None)
    assert len(movers) == 4  # NEUTRAL dropped, capped at 4
    assert movers[0].tag in ("hot", "cold")
    # highest |delta| first → player 1 (0.9, HOT)
    assert movers[0].player.id == 1
    assert movers[0].tag == "hot" and movers[0].trend == "up"
    assert movers[0].player.mlb_id == 101  # enriched from pool
    assert movers[0].stats == ["20 HR", ".310 AVG"]
    # a COLD mover maps to down/cold
    cold = [m for m in movers if m.tag == "cold"][0]
    assert cold.trend == "down"


def test_movers_empty_roster_returns_empty(monkeypatch):
    # no roster → no engine calls, empty list (not a raise)
    assert _real_service()._movers([], cfg=None) == []


def test_il_count_and_status_chips():
    svc = _real_service()
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "status": ["Active", "IL10", "DTD", "active"],
        }
    )
    assert svc._il_count(roster) == 2  # IL10 + DTD
    chips = svc._status_chips(roster, [1, 2, 3, 4])
    il_chip = [c for c in chips if c.label == "IL"][0]
    assert il_chip.value == 2 and il_chip.status == "warn"


def test_freshness_minutes_reports_stalest_and_ignores_offlist(monkeypatch):
    now = datetime.now(UTC)
    snap = [
        {"source": "yahoo_standings", "last_refresh": (now - timedelta(minutes=30)).isoformat()},
        {"source": "season_stats", "last_refresh": (now - timedelta(minutes=300)).isoformat()},
        {"source": "ecr_consensus", "last_refresh": now.isoformat()},  # off the freshness list → ignored
    ]
    monkeypatch.setattr("src.database.get_refresh_log_snapshot", lambda: snap)
    age = _real_service()._freshness_minutes()
    # reports the STALEST core source (300 min), not the freshest 30-min one — and the
    # off-list 0-min source must not pull it toward 0
    assert age is not None and 299.0 <= age <= 301.0


def test_freshness_minutes_clamps_future_timestamp(monkeypatch):
    now = datetime.now(UTC)
    # a future timestamp (clock skew) must not yield a negative age
    snap = [{"source": "yahoo_standings", "last_refresh": (now + timedelta(minutes=10)).isoformat()}]
    monkeypatch.setattr("src.database.get_refresh_log_snapshot", lambda: snap)
    age = _real_service()._freshness_minutes()
    assert age is not None and age >= 0.0


def test_il_count_handles_freeform_yahoo_status():
    # Yahoo emits free-form variants like "IL10 - 3 days remaining"
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "status": ["Active", "IL10 - 3 days remaining", "DTD", "NA"],
        }
    )
    assert _real_service()._il_count(roster) == 3  # IL10-freeform + DTD + NA


def test_mover_stats_pitcher_and_zero_avg():
    from api.services.team_service import _fmt_avg, _mover_stats

    pitcher = {"ytd_k": 142, "ytd_era": 3.27}
    assert _mover_stats(pitcher, hitter=False) == ["142 K", "3.27 ERA"]
    # batting-average display: strip leading zero for a real rate, keep it for 0/NaN
    assert _fmt_avg(0.314) == ".314 AVG"
    assert _fmt_avg(0.0) == "0.000 AVG"


def test_subline_ordinal_and_games_back():
    svc = _real_service()
    standings = pd.DataFrame(
        {
            "team_name": ["A", "B", "C"],
            "category": ["WINS", "WINS", "WINS"],
            "total": ["7-3-0", "4-6-0", "5-5-0"],
        }
    )
    sub = svc._subline("4-6-0", 10, 12, standings)
    assert "10th of 12" in sub
    assert "3 GB from 1st" in sub  # leader 7 wins - your 4 = 3
