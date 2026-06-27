import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.trade_finder import TradeFinderResponse, TradeSuggestion
from api.deps import get_trade_finder_service
from api.main import create_app
from api.services.trade_finder_service import TradeFinderService


def test_trade_finder_contract_shape():
    resp = TradeFinderResponse(
        team_name="Team Hickey",
        suggestions=[
            TradeSuggestion(
                partner_team="Team Smith",
                giving=[PlayerRef(id=1, name="Player A", positions="OF")],
                receiving=[PlayerRef(id=2, name="Player B", positions="1B")],
                net_sgp=1.5,
                rationale="Improves HR and SB.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert dumped["suggestions"][0]["partner_team"] == "Team Smith"
    assert dumped["suggestions"][0]["giving"][0]["name"] == "Player A"
    assert dumped["suggestions"][0]["net_sgp"] == 1.5
    # defaults
    assert TradeSuggestion(partner_team="X").giving == []
    assert TradeSuggestion(partner_team="X").net_sgp == 0.0
    assert TradeFinderResponse(team_name="T").suggestions == []


class _FakeTradeFinderService:
    def get_suggestions(self, team_name: str, limit: int = 10) -> TradeFinderResponse:
        return TradeFinderResponse(
            team_name=team_name,
            suggestions=[
                TradeSuggestion(
                    partner_team="Rival Team",
                    giving=[PlayerRef(id=10, name="Give Guy", positions="SS")],
                    receiving=[PlayerRef(id=20, name="Recv Guy", positions="2B")],
                    net_sgp=0.8,
                    rationale="Good fit.",
                )
            ],
        )


def test_get_trade_finder_returns_contract():
    app = create_app()
    app.dependency_overrides[get_trade_finder_service] = lambda: _FakeTradeFinderService()
    client = TestClient(app)
    resp = client.get("/api/trade-finder?team_name=Team+Hickey&limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["suggestions"][0]["partner_team"] == "Rival Team"
    assert body["suggestions"][0]["giving"][0]["name"] == "Give Guy"
    assert body["suggestions"][0]["net_sgp"] == 0.8


def test_trade_finder_method_discipline_get_is_supported_post_405():
    """M-8: the live `POST /api/trade-finder` returned 405. The supported method is
    **GET** (the router exposes only @router.get), and the frontend correctly calls it
    with apiGet (`web/src/lib/trades-data.ts`). So the 405 on POST is CORRECT, not a
    server bug — the page's "No trade ideas yet" empty state is the documented
    button-gated / roster-relative empty (suggestions are empty until live Yahoo data).
    This locks the contract: GET → 200, POST → 405."""
    app = create_app()
    app.dependency_overrides[get_trade_finder_service] = lambda: _FakeTradeFinderService()
    client = TestClient(app)

    ok = client.get("/api/trade-finder?team_name=Team+Hickey&limit=5")
    assert ok.status_code == 200
    assert "suggestions" in ok.json()  # the trade-finder contract shape

    # POST is not a defined method on this route → 405 Method Not Allowed (expected).
    assert client.post("/api/trade-finder", json={"team_name": "Team Hickey"}).status_code == 405


# ── Task 2: emoji/whitespace team resolution + all_team_totals (DB-free) ──────


def _fake_pool():
    # minimal pool: ids 1,2,3 (user) + 4,5 (rival), with the cols the refs/diff need
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["You A", "You B", "You C", "Riv A", "Riv B"],
            "positions": ["OF", "SP", "2B", "SS", "RP"],
            "mlb_id": [101, 102, 103, 104, 105],
            "team": ["NYM", "SEA", "KC", "CIN", "CLE"],
            "is_hitter": [1, 0, 1, 1, 0],
        }
    )


class _FakeYds:
    def get_rosters(self):
        # emoji/whitespace team-name keys — the exact mismatch the bug hits
        return pd.DataFrame(
            {
                "team_name": [
                    "\U0001f3c6 Team Hickey",
                    "\U0001f3c6 Team Hickey",
                    "\U0001f3c6 Team Hickey",
                    "Over the Rembow",
                    "Over the Rembow",
                ],
                "player_id": [1, 2, 3, 4, 5],
            }
        )


def test_service_reconciles_emoji_team_name_and_passes_all_team_totals(monkeypatch):
    """The bug: raw .get('Team Hickey') misses the '🏆 Team Hickey' roster key →
    empty user_roster_ids; and all_team_totals=None forces find_trade_opportunities
    to early-return []. Assert the service reconciles the name AND passes non-None
    all_team_totals."""
    captured = {}

    def _fake_find(**kwargs):
        captured.update(kwargs)
        return [
            {
                "giving_ids": [1],
                "receiving_ids": [4],
                "opponent_team": "Over the Rembow",
                "user_sgp_gain": 1.2,
                "grade": "B+",
                "rationale": "ok",
            }
        ]

    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
    monkeypatch.setattr(
        "src.standings_utils.get_all_team_totals",
        lambda *a, **k: {"\U0001f3c6 Team Hickey": {"HR": 100.0}, "Over the Rembow": {"HR": 90.0}},
    )
    monkeypatch.setattr("src.trade_finder.find_trade_opportunities", _fake_find)
    # records lookup degrades to {} (no DB) — Task 3 fields default
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())

    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)

    # reconciled the mismatched name → the user's real roster ids (NOT [])
    assert captured["user_roster_ids"] == [1, 2, 3]
    # all_team_totals passed through non-None (the second compounding bug)
    assert captured["all_team_totals"] is not None
    assert "Over the Rembow" in captured["all_team_totals"]
    # suggestion surfaced
    assert len(resp.suggestions) == 1
    assert resp.suggestions[0].partner_team == "Over the Rembow"


def test_service_exact_name_match_still_works(monkeypatch):
    """An exact-match team_name (no emoji) must still resolve (no regression)."""
    captured = {}
    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())

    class _ExactYds:
        def get_rosters(self):
            return pd.DataFrame({"team_name": ["Team Hickey", "Team Hickey", "Rival"], "player_id": [1, 2, 4]})

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _ExactYds())
    monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"Team Hickey": {}, "Rival": {}})

    def _fake_find(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.trade_finder.find_trade_opportunities", _fake_find)
    TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert captured["user_roster_ids"] == [1, 2]


def test_service_unresolvable_name_returns_empty_not_crash(monkeypatch):
    """A team_name that matches no roster key → empty suggestions, never a 500."""
    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())

    class _Yds:
        def get_rosters(self):
            return pd.DataFrame({"team_name": ["Real Team"], "player_id": [1]})

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _Yds())
    monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"Real Team": {}})
    monkeypatch.setattr("src.trade_finder.find_trade_opportunities", lambda **k: [])
    resp = TradeFinderService().get_suggestions(team_name="Nonexistent Team", limit=10)
    assert resp.suggestions == []


# ── Task 3: grade (engine) + partner_record (load_league_records) ─────────────


def test_build_suggestions_reads_engine_grade_and_partner_record(monkeypatch):
    """grade comes straight off the engine dict (find_trade_opportunities already
    sets trade['grade']); partner_record comes from load_league_records, matched
    emoji-tolerantly to the opponent_team."""
    svc = TradeFinderService()
    pool = _fake_pool()
    raw = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "opponent_team": "Over the Rembow",
            "user_sgp_gain": 1.2,
            "grade": "B+",
            "rationale": "ok",
        }
    ]
    monkeypatch.setattr(
        "src.database.load_league_records",
        lambda: pd.DataFrame(
            {
                "team_name": ["Over the Rembow"],
                "wins": [11],
                "losses": [1],
                "ties": [0],
                "rank": [1],
            }
        ),
    )
    out = svc._build_suggestions(raw, pool)
    assert out[0].grade == "B+"
    assert out[0].partner_record == "11-1-0 · 1st"


def test_build_suggestions_missing_grade_and_records_degrade(monkeypatch):
    """No grade on the dict + no records table → grade '' and partner_record None,
    never a crash."""
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())
    out = TradeFinderService()._build_suggestions(
        [{"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Nobody", "user_sgp_gain": 0.4}],
        _fake_pool(),
    )
    assert out[0].grade == ""
    assert out[0].partner_record is None


# ── Task 4: category_impacts (service-side per-category SGP diff) ──────────────


def test_category_impacts_computed_from_roster_diff():
    """category_impacts = per-category SGP delta of (roster - give + receive) vs
    roster, computed in the service (the finder engine doesn't surface it)."""
    svc = TradeFinderService()
    # pool with real-ish category columns so totals_sgp moves on a swap
    pool = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "player_name": ["You A", "You B", "You C", "Riv A"],
            "positions": ["OF", "OF", "OF", "SS"],
            "mlb_id": [101, 102, 103, 104],
            "team": ["NYM", "SEA", "KC", "CIN"],
            "is_hitter": [1, 1, 1, 1],
            "r": [80, 70, 60, 95],
            "hr": [20, 15, 10, 30],
            "rbi": [70, 60, 50, 90],
            "sb": [5, 3, 2, 30],
            "ab": [550, 540, 500, 560],
            "h": [150, 140, 130, 165],
            "bb": [50, 45, 40, 60],
            "hbp": [3, 2, 2, 4],
            "sf": [4, 3, 3, 5],
            "obp": [0.350, 0.330, 0.320, 0.380],
            "avg": [0.272, 0.259, 0.260, 0.295],
        }
    )
    user_roster_ids = [1, 2, 3]
    impacts = svc._category_impacts(user_roster_ids, [1], [4], pool)
    cats = {ci.cat for ci in impacts}
    # at minimum SB should be a large positive delta (give a 5-SB OF, get a 30-SB SS)
    sb = next((ci for ci in impacts if ci.cat == "SB"), None)
    assert sb is not None and sb.delta > 0
    assert "HR" in cats  # a counting cat is present


def test_category_impacts_empty_when_no_user_roster():
    """No user roster ids → empty impacts, no crash."""
    svc = TradeFinderService()
    assert svc._category_impacts([], [1], [4], _fake_pool()) == []


def test_build_suggestions_threads_category_impacts():
    """When user_roster_ids is passed, each suggestion carries computed
    category_impacts; when omitted (None), impacts default to []."""
    svc = TradeFinderService()
    pool = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "player_name": ["You A", "You B", "You C", "Riv A"],
            "positions": ["OF", "OF", "OF", "SS"],
            "mlb_id": [101, 102, 103, 104],
            "team": ["NYM", "SEA", "KC", "CIN"],
            "is_hitter": [1, 1, 1, 1],
            "r": [80, 70, 60, 95],
            "hr": [20, 15, 10, 30],
            "rbi": [70, 60, 50, 90],
            "sb": [5, 3, 2, 30],
            "ab": [550, 540, 500, 560],
            "h": [150, 140, 130, 165],
            "bb": [50, 45, 40, 60],
            "hbp": [3, 2, 2, 4],
            "sf": [4, 3, 3, 5],
            "obp": [0.350, 0.330, 0.320, 0.380],
            "avg": [0.272, 0.259, 0.260, 0.295],
        }
    )
    raw = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "opponent_team": "Rival",
            "user_sgp_gain": 1.0,
        }
    ]
    with_roster = svc._build_suggestions(raw, pool, [1, 2, 3])
    assert len(with_roster[0].category_impacts) > 0
    without_roster = svc._build_suggestions(raw, pool)
    assert without_roster[0].category_impacts == []


# ── Observability: response.reason makes a recurrence LOUD (not a silent empty) ──


def test_reason_team_not_resolved_when_name_misses_emoji_key(monkeypatch):
    """The original bug was a SILENT empty (raw .get missed the emoji roster key).
    A recurrence must be observable: an unresolved name → reason='team_not_resolved'
    (+ a logged warning), not a bare empty 200."""
    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())

    class _Yds:
        def get_rosters(self):
            # only an emoji-keyed team exists; the request name below won't match
            return pd.DataFrame({"team_name": ["\U0001f3c6 Real Team", "\U0001f3c6 Real Team"], "player_id": [1, 2]})

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _Yds())
    monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"x": {}})
    monkeypatch.setattr("src.trade_finder.find_trade_opportunities", lambda **k: [])
    resp = TradeFinderService().get_suggestions(team_name="Totally Different", limit=10)
    assert resp.suggestions == []
    assert resp.reason == "team_not_resolved"


def test_reason_no_totals_when_get_all_team_totals_empty(monkeypatch):
    """all_team_totals={} forces the engine to yield nothing — surface it as
    reason='no_totals' (+ a logged warning), and return BEFORE the engine call."""
    called = {"engine": False}

    def _fake_find(**kwargs):
        called["engine"] = True
        return []

    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
    monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {})
    monkeypatch.setattr("src.trade_finder.find_trade_opportunities", _fake_find)
    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert resp.reason == "no_totals"
    assert resp.suggestions == []
    assert called["engine"] is False  # short-circuited before the doomed engine call


def test_reason_ok_when_suggestion_produced(monkeypatch):
    """Engine ran (regardless of count) → reason='ok'."""
    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
    monkeypatch.setattr(
        "src.standings_utils.get_all_team_totals",
        lambda *a, **k: {"\U0001f3c6 Team Hickey": {"HR": 100.0}, "Over the Rembow": {"HR": 90.0}},
    )
    monkeypatch.setattr(
        "src.trade_finder.find_trade_opportunities",
        lambda **k: [
            {
                "giving_ids": [1],
                "receiving_ids": [4],
                "opponent_team": "Over the Rembow",
                "user_sgp_gain": 1.2,
                "grade": "B+",
            }
        ],
    )
    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert resp.reason == "ok"
    assert len(resp.suggestions) == 1


def test_reason_no_pool_when_pool_empty(monkeypatch):
    """Empty pool → reason='no_pool' (cold env observable)."""
    monkeypatch.setattr("src.database.load_player_pool", lambda: pd.DataFrame())
    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert resp.reason == "no_pool"
    assert resp.suggestions == []


def test_reason_no_league_data_when_rosters_empty(monkeypatch):
    """No league_rosters → reason='no_league_data'."""
    monkeypatch.setattr("src.database.load_player_pool", _fake_pool)

    class _EmptyYds:
        def get_rosters(self):
            return pd.DataFrame()

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _EmptyYds())
    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert resp.reason == "no_league_data"
    assert resp.suggestions == []
