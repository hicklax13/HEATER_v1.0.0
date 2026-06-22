from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef, StatItem
from api.contracts.my_team import (
    CategoryLine,
    Lever,
    LeverPickup,
    MatchupHero,
    Mover,
    MyTeamResponse,
    OpsCard,
    StatusChip,
)
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
                    stats=[StatItem(label="HR", value="18"), StatItem(label="AVG", value=".322")],
                    trend="up",
                    tag="hot",
                    context="Trending hot vs projection",
                )
            ],
            movers_scope="mine",
            lever=Lever(
                category_key="SB",
                headline="SB is your weakest category",
                behind_by=13.0,
                pickups=[
                    LeverPickup(
                        player=PlayerRef(id=2, mlb_id=668800, name="Speedy", positions="OF"),
                        proj_stat=StatItem(label="SB", value="24"),
                    )
                ],
            ),
            ops=[
                OpsCard(key="ip_pace", label="IP Pace", value=42.0, total=53.9, verdict="42 IP", status="warn"),
                OpsCard(key="moves_left", label="Moves Left", value=7.0, total=10.0, verdict="7 of 10", status="ok"),
            ],
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
    assert body["movers"][0]["stats"][0] == {"label": "HR", "value": "18"}  # structured StatItem
    assert body["movers_scope"] == "mine"
    # slice-2 lever
    assert body["lever"]["category_key"] == "SB"
    assert body["lever"]["pickups"][0]["player"]["mlb_id"] == 668800
    assert body["lever"]["pickups"][0]["proj_stat"] == {"label": "SB", "value": "24"}
    # slice-3 ops cards
    assert [c["key"] for c in body["ops"]] == ["ip_pace", "moves_left"]
    assert body["ops"][1]["value"] == 7.0 and body["ops"][1]["total"] == 10.0


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
    # stats are structured StatItem{label,value} (CMO contract ask)
    assert [(s.label, s.value) for s in movers[0].stats] == [("HR", "20"), ("AVG", ".310")]
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
    from api.services.team_service import _avg_value, _mover_stats

    pitcher = {"ytd_k": 142, "ytd_era": 3.27}
    # structured StatItem{label,value} (pitcher path)
    assert [(s.label, s.value) for s in _mover_stats(pitcher, hitter=False)] == [("K", "142"), ("ERA", "3.27")]
    # batting-average VALUE: strip leading zero for a real rate, keep it for 0/NaN
    assert _avg_value(0.314) == ".314"
    assert _avg_value(0.0) == "0.000"


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


# ── _rank_and_record (HIGH: surface real W-L records) ────────────────────────
def test_rank_and_record_from_records_table():
    """Records table is authoritative → 'W-L-T' string + its rank (Railway path)."""
    svc = _real_service()
    standings = pd.DataFrame({"team_name": ["Team A"], "category": ["HR"], "total": [50.0], "rank": [3]})
    records = pd.DataFrame([{"team_name": "Team A", "wins": 8, "losses": 4, "ties": 1, "rank": 3}])
    rank, record = svc._rank_and_record(standings, records, "Team A")
    assert record == "8-4-1"
    assert rank == 3


def test_rank_and_record_fallback_wlt_string_category():
    """records empty; WINS category total is already a 'W-L-T' string."""
    svc = _real_service()
    standings = pd.DataFrame({"team_name": ["Team A"], "category": ["WINS"], "total": ["7-3-0"], "rank": [2]})
    rank, record = svc._rank_and_record(standings, pd.DataFrame(), "Team A")
    assert record == "7-3-0"
    assert rank == 2


def test_rank_and_record_fallback_numeric_wlt_categories():
    """records empty; WINS/LOSSES/TIES are separate numeric category rows (local DB)."""
    svc = _real_service()
    standings = pd.DataFrame(
        {
            "team_name": ["Team A", "Team A", "Team A"],
            "category": ["WINS", "LOSSES", "TIES"],
            "total": [6.0, 5.0, 0.0],
            "rank": [5, 5, 5],
        }
    )
    rank, record = svc._rank_and_record(standings, pd.DataFrame(), "Team A")
    assert record == "6-5-0"
    assert rank == 5


def test_rank_and_record_empty_when_no_data():
    svc = _real_service()
    standings = pd.DataFrame({"team_name": ["Team A"], "category": ["HR"], "total": [10.0], "rank": [0]})
    rank, record = svc._rank_and_record(standings, pd.DataFrame(), "Team A")
    assert record == "0-0-0"
    assert rank == 0


# ── _matchup hero (key-mismatch bug: raw matchup uses opp_name, not opponent) ─
def test_matchup_hero_reads_opp_name_and_category_record():
    """MatchupResult keys are opp_name + wins/losses/ties (NOT opponent/win_prob),
    so the hero was always hollow. _matchup must read the real keys and derive the
    current category-lead share (sums to 1)."""
    from src.valuation import LeagueConfig

    raw = {"week": 13, "opp_name": "Baty Babies", "wins": 7, "losses": 4, "ties": 1}
    hero = _real_service()._matchup(raw, LeagueConfig())
    assert hero is not None
    assert hero.opponent == "Baty Babies"
    assert hero.week == 13
    assert hero.win_prob == pytest.approx(7 / 12)
    assert hero.loss_prob == pytest.approx(4 / 12)
    assert hero.tie_prob == pytest.approx(1 / 12)


def test_matchup_hero_none_when_no_opponent():
    """No raw matchup, or a blank opp_name → None (never a hollow 'vs ' hero)."""
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()
    assert _real_service()._matchup(None, cfg) is None
    assert _real_service()._matchup({"week": 5, "opp_name": ""}, cfg) is None


def test_matchup_hero_undecided_week_zero_probs():
    """Opponent known but no categories decided yet → opponent+week show, probs 0."""
    from src.valuation import LeagueConfig

    raw = {"week": 1, "opp_name": "Rivals", "wins": 0, "losses": 0, "ties": 0}
    hero = _real_service()._matchup(raw, LeagueConfig())
    assert hero is not None
    assert hero.opponent == "Rivals"
    assert (hero.win_prob, hero.tie_prob, hero.loss_prob) == (0.0, 0.0, 0.0)


# ── slice-2 lever (DB-free: build_optimizer_context + rank_free_agents monkeypatched
# at their source modules, since _lever imports them lazily inside the method) ──────


class _FakeCtx:
    def __init__(self, gaps, free_agents, pool, roster_ids, roster=None, adds_remaining_this_week=10):
        self.category_gaps = gaps
        self.free_agents = free_agents
        self.player_pool = pool
        self.user_roster_ids = roster_ids
        self.roster = roster if roster is not None else pd.DataFrame()
        self.adds_remaining_this_week = adds_remaining_this_week


def test_lever_picks_weakest_cat_and_filters_pickups(monkeypatch):
    # category_gaps keys are LOWERCASE (real engine shape) but best_category is UPPERCASE;
    # the lever must normalize so the pickup filter matches. SB is the most-negative gap.
    gaps = {"hr": 1.5, "sb": -4.0, "r": -1.0}
    pool = pd.DataFrame(
        [
            {"player_id": 2, "name": "Speedy", "positions": "OF", "mlb_id": 668800, "team": "SD", "ytd_sb": 24},
            {"player_id": 3, "name": "Slugger", "positions": "1B", "mlb_id": 668801, "team": "LAD", "ytd_sb": 1},
        ]
    )
    fas = pd.DataFrame([{"player_id": 2}, {"player_id": 3}])  # non-empty
    ranked = pd.DataFrame(
        [
            {"player_id": 2, "player_name": "Speedy", "positions": "OF", "best_category": "SB", "marginal_value": 1.2},
            {"player_id": 3, "player_name": "Slugger", "positions": "1B", "best_category": "HR", "marginal_value": 1.1},
        ]
    )
    monkeypatch.setattr("src.in_season.rank_free_agents", lambda *a, **k: ranked)

    lever = _real_service()._lever(_FakeCtx(gaps, fas, pool, [1]), cfg=None)
    assert lever is not None
    assert lever.category_key == "SB"
    assert lever.behind_by == 4.0
    # only the SB-best FA is a pickup (Slugger's best is HR → excluded)
    assert len(lever.pickups) == 1
    assert lever.pickups[0].player.mlb_id == 668800
    assert lever.pickups[0].proj_stat.label == "SB" and lever.pickups[0].proj_stat.value == "24"


def test_lever_none_when_winning_all_cats():
    # every gap >= 0 → at-or-ahead everywhere → no weakness → lever None (not a misleading card)
    ctx = _FakeCtx({"hr": 1.5, "sb": 0.2, "r": 3.0}, pd.DataFrame(), pd.DataFrame(), [])
    assert _real_service()._lever(ctx, cfg=None) is None


def test_lever_none_on_no_gaps_and_cold_env():
    # no gaps → None
    assert _real_service()._lever(_FakeCtx({}, pd.DataFrame(), pd.DataFrame(), []), cfg=None) is None
    # cold env (ctx is None, e.g. build_optimizer_context failed) → None
    assert _real_service()._lever(None, cfg=None) is None


def test_build_ctx_none_on_failure(monkeypatch):
    def boom(**k):
        raise RuntimeError("no context")

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: None)
    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", boom)
    assert _real_service()._build_ctx("Team Hickey", cfg=None) is None


def test_cat_stat_formats_rate_vs_counting():
    from api.services.team_service import _cat_stat

    assert _cat_stat({"ytd_sb": 24}, "SB") == StatItem(label="SB", value="24")
    assert _cat_stat({"ytd_era": 3.27}, "ERA") == StatItem(label="ERA", value="3.27")
    assert _cat_stat({"ytd_avg": 0.314}, "AVG") == StatItem(label="AVG", value=".314")
    assert _cat_stat(None, "SB") == StatItem(label="SB", value="0")  # missing pool row → '0'


# ── slice-3 ops cards (DB-free) ──────────────────────────────────────────────


def test_ops_returns_three_cards(monkeypatch):
    # roster has 2 pitchers (1 IL) + 1 hitter; pool gives projected IP.
    roster = pd.DataFrame(
        {
            "player_id": [10, 11, 12],
            "status": ["active", "IL10", "active"],
        }
    )
    pool = pd.DataFrame(
        [
            {"player_id": 10, "name": "Ace", "positions": "SP", "is_hitter": 0, "ip": 180, "gs": 30},
            {"player_id": 11, "name": "Hurt", "positions": "SP", "is_hitter": 0, "ip": 150, "gs": 28},
            {"player_id": 12, "name": "Bat", "positions": "OF", "is_hitter": 1, "ip": 0, "gs": 0},
        ]
    )
    ctx = _FakeCtx({}, pd.DataFrame(), pool, [10, 11, 12], roster=roster, adds_remaining_this_week=7)
    cards = {c.key: c for c in _real_service()._ops(ctx)}
    assert set(cards) == {"ip_pace", "moves_left", "roster_health"}
    # moves-left = ctx.adds_remaining_this_week / 10
    assert cards["moves_left"].value == 7.0 and cards["moves_left"].total == 10.0 and cards["moves_left"].status == "ok"
    # roster-health: 1 of 3 on IL → 2 healthy, warn
    assert cards["roster_health"].value == 2.0 and cards["roster_health"].total == 3.0
    assert cards["roster_health"].status == "warn" and cards["roster_health"].verdict == "1 on IL"
    # ip-pace built from the 2 SPs (IL one excluded by the engine); assert shape, not a
    # fragile numeric bound (the IP target is an engine constant, not the service's to assert).
    ip = cards["ip_pace"]
    assert ip.key == "ip_pace" and ip.status in ("ok", "warn", "danger")
    assert ip.value >= 0.0 and ip.total > 0.0 and ip.value <= ip.total * 3  # sane, finite


def test_ops_empty_ctx_returns_empty():
    assert _real_service()._ops(None) == []


def test_ops_isolates_a_failing_card(monkeypatch):
    # one card builder raising must not drop the other two (per-card try/except).
    from api.services.team_service import TeamService

    svc = _real_service()
    ctx = _FakeCtx(
        {}, pd.DataFrame(), pd.DataFrame(), [], roster=pd.DataFrame({"player_id": [1], "status": ["active"]})
    )
    monkeypatch.setattr(
        TeamService, "_ip_pace_card", staticmethod(lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    keys = {c.key for c in svc._ops(ctx)}
    assert keys == {"moves_left", "roster_health"}  # ip_pace raised → skipped, others survive


def test_moves_left_status_thresholds():
    svc = _real_service()
    base = _FakeCtx({}, pd.DataFrame(), pd.DataFrame(), [])
    base.adds_remaining_this_week = 5
    assert svc._moves_left_card(base).status == "ok"
    base.adds_remaining_this_week = 1
    assert svc._moves_left_card(base).status == "warn"
    base.adds_remaining_this_week = 0
    assert svc._moves_left_card(base).status == "danger"
    base.adds_remaining_this_week = None  # unknown budget → 0 → danger (no crash)
    assert svc._moves_left_card(base).status == "danger"


def test_roster_health_all_active():
    roster = pd.DataFrame({"player_id": [1, 2], "status": ["active", "active"]})
    ctx = _FakeCtx({}, pd.DataFrame(), pd.DataFrame(), [], roster=roster)
    card = _real_service()._roster_health_card(ctx)
    assert card.value == 2.0 and card.status == "ok" and card.verdict == "All active"
