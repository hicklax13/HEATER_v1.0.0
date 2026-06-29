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


# ── Shared fixtures ───────────────────────────────────────────────────────────


def _hitter_pool():
    """Pool with real category cols so per-player SGP (calc.player_sgp) is NONZERO
    and a give/receive swap moves true_gain. Player 4 (rival) is a clearly stronger
    hitter than user players 1-3, so give=[1] get=[4] is a genuine upgrade
    (true_gain > 0.5 → surfaces); give=[1,2] get=[4] is value-losing (filtered)."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["You A", "You B", "You C", "Riv A", "Riv B"],
            "positions": ["OF", "OF", "2B", "SS", "RP"],
            "mlb_id": [101, 102, 103, 104, 105],
            "team": ["NYM", "SEA", "KC", "CIN", "CLE"],
            "is_hitter": [1, 1, 1, 1, 0],
            "r": [80, 70, 60, 95, 0],
            "hr": [25, 20, 15, 30, 0],
            "rbi": [80, 70, 60, 90, 0],
            "sb": [5, 3, 2, 30, 0],
            "ab": [550, 540, 500, 560, 0],
            "h": [150, 140, 130, 165, 0],
            "bb": [50, 45, 40, 60, 0],
            "hbp": [3, 2, 2, 4, 0],
            "sf": [4, 3, 3, 5, 0],
            "obp": [0.350, 0.330, 0.320, 0.380, 0.0],
            "avg": [0.272, 0.259, 0.260, 0.295, 0.0],
        }
    )


# `_fake_pool` is the DB-free service-path fixture. It carries the SAME stat columns
# as `_hitter_pool` so the surfaced swap (give=[1] get=[4]) has true_gain > 0.5 — a
# pool WITHOUT stat columns would score 0 SGP everywhere and the honest filter would
# (correctly) drop every trade, breaking the service-path assertions.
_fake_pool = _hitter_pool


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


# ── Task 2: emoji/whitespace team resolution + all_team_totals (DB-free) ──────


def test_service_reconciles_emoji_team_name_and_passes_all_team_totals(monkeypatch):
    """The bug: raw .get('Team Hickey') misses the '🏆 Team Hickey' roster key →
    empty user_roster_ids; and all_team_totals=None forces find_trade_opportunities
    to early-return []. Assert the service reconciles the name AND passes non-None
    all_team_totals. The engine emits a GENUINE upgrade (give weak #1, get strong #4)
    so the honest re-valuation keeps it (true_gain > 0.5)."""
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
    # suggestion surfaced (the upgrade has a real positive marginal gain)
    assert len(resp.suggestions) == 1
    assert resp.suggestions[0].partner_team == "Over the Rembow"
    assert resp.suggestions[0].net_sgp > 0.5


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


# ── HONEST RE-VALUATION: grade + net_sgp come from per-player marginal SGP ─────


def test_build_suggestions_regrades_from_marginal_value_and_reads_record(monkeypatch):
    """The honest contract: grade + net_sgp are RE-DERIVED from per-player marginal
    SGP (calc.player_sgp), NOT trusted from the engine dict. Give a weak hitter (#1),
    get a clearly stronger one (#4): the marginal gain is several SGP → an A-tier
    grade. partner_record still reads from load_league_records, emoji-tolerantly."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    raw = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "opponent_team": "Over the Rembow",
            # The engine's OWN gain/grade are deliberately WRONG/absent — the service
            # must ignore them and re-derive from the players' marginal SGP.
            "user_sgp_gain": 99.0,
            "grade": "F",
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
    assert len(out) == 1
    # net_sgp is the HONEST marginal gain, not the engine's 99.0
    assert 3.0 < out[0].net_sgp < 5.0
    # re-graded from true_gain (a multi-SGP gain → top tier), NOT the engine's "F"
    assert out[0].grade.startswith("A")
    # partner_record still surfaces
    assert out[0].partner_record == "11-1-0 · 1st"


def test_build_suggestions_record_degrades_grade_always_computed(monkeypatch):
    """No records table → partner_record None (degrades, never crashes). The grade is
    now ALWAYS computed from marginal value (never blank), so a positive-gain trade
    still surfaces with a real grade."""
    monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())
    out = TradeFinderService()._build_suggestions(
        # give weak #1, get strong #4 → true_gain > 0.5 so it surfaces
        [{"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Nobody"}],
        _hitter_pool(),
    )
    assert len(out) == 1
    assert out[0].partner_record is None
    assert out[0].grade != ""  # grade is always derived from true_gain
    assert out[0].net_sgp > 0.5


# ── Marginal category impacts (roster-id-space independent) ────────────────────


def test_build_suggestions_threads_marginal_category_impacts():
    """Marginal category impacts come from the give/receive players directly
    (cat_net = Σreceiving − Σgiving), so they are present REGARDLESS of whether
    user_roster_ids was passed — roster-id-space independent (more robust than the
    old roster-totals diff)."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    raw = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "opponent_team": "Rival",
        }
    ]
    with_roster = svc._build_suggestions(raw, pool, [1, 2, 3])
    without_roster = svc._build_suggestions(raw, pool)
    # impacts present in BOTH cases (no longer roster-dependent)
    assert len(with_roster[0].category_impacts) > 0
    assert len(without_roster[0].category_impacts) > 0
    # and identical (the marginal diff doesn't depend on the roster)
    wi = {ci.cat: ci.delta for ci in with_roster[0].category_impacts}
    wo = {ci.cat: ci.delta for ci in without_roster[0].category_impacts}
    assert wi == wo


def test_build_suggestions_marginal_impacts_correct_signs():
    """Anti-"AVG +1.03": giving a run-producer (#4, the strong hitter) and receiving a
    weaker one (#1) LOWERS R/HR/RBI (negative deltas), and the AVG delta is realistic
    (|AVG delta| well under 1.0). cat_net = Σreceiving − Σgiving = player_sgp(#1) −
    player_sgp(#4)."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    # give the STRONG hitter (#4), receive a weak one (#1) → counting cats drop
    raw = [{"giving_ids": [4], "receiving_ids": [1], "opponent_team": "Rival"}]
    # This trade is value-LOSING for the user, so it'll be filtered from suggestions.
    # Compute impacts directly to assert the signs/magnitudes.
    impacts = svc._marginal_category_impacts([4], [1], pool)
    by_cat = {ci.cat: ci.delta for ci in impacts}
    assert by_cat["R"] < 0
    assert by_cat["HR"] < 0
    assert by_cat["RBI"] < 0
    # AVG/OBP deltas are realistic — never an absurd "+1.03" rate SGP on one swap
    assert abs(by_cat.get("AVG", 0.0)) < 1.0
    assert abs(by_cat.get("OBP", 0.0)) < 1.0


def test_marginal_category_impacts_empty_for_unknown_players():
    """Unknown player ids → {} per-player SGP → no finite, non-trivial deltas."""
    svc = TradeFinderService()
    assert svc._marginal_category_impacts([991], [992], _hitter_pool()) == []


# ── HONEST FILTER: lopsided value-losing trades are dropped ────────────────────


def test_build_suggestions_filters_lopsided_two_for_one():
    """The LIVE BUG: give two strong hitters, get one lower-SGP hitter. The honest
    re-valuation (true_gain = get − give + slot_credit) is deeply negative, so the
    suggestion is FILTERED OUT — never surfaced as a graded card. This is the exact
    'give Reynolds+Olson, get Moniak → A / You win' class, fixed."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    # give #1 (~6.8 SGP) + #2 (~6.0 SGP) = ~12.8; get #4 (~10.9 SGP); slot_credit 1.0
    # true_gain ≈ 10.9 − 12.8 + 1.0 < 0 → filtered.
    raw = [
        {
            "giving_ids": [1, 2],
            "receiving_ids": [4],
            "opponent_team": "Rival",
            "user_sgp_gain": 5.0,  # the engine's inflated gain — must be ignored
            "grade": "A",  # the engine's bogus grade — must be ignored
        }
    ]
    out = svc._build_suggestions(raw, pool, [1, 2, 3])
    assert out == []  # the value-losing 2-for-1 is dropped


def test_build_suggestions_drops_below_min_true_gain():
    """A trade whose honest marginal gain is at/below the floor (_MIN_TRUE_GAIN) is
    not surfaced. give #1 get #3 — two similar-value hitters → true_gain near 0."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    raw = [{"giving_ids": [1], "receiving_ids": [3], "opponent_team": "Rival", "user_sgp_gain": 0.9}]
    out = svc._build_suggestions(raw, pool, [1, 2, 3])
    assert out == []


def test_build_suggestions_keeps_genuine_upgrade():
    """A genuine upgrade (give weak #1, get strong #4) clears the floor and surfaces
    with an honest net_sgp."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    raw = [{"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Rival", "user_sgp_gain": 0.1}]
    out = svc._build_suggestions(raw, pool, [1, 2, 3])
    assert len(out) == 1
    assert out[0].net_sgp > 0.5


def test_build_suggestions_keeps_positive_distinct_trades():
    """Distinct positive trades (different partner+receiving) are all kept — both are
    genuine upgrades giving a weak hitter for a strong one."""
    svc = TradeFinderService()
    # extend the pool with a second strong rival hitter (#6) for the 2nd trade
    pool = _hitter_pool()
    pool = pd.concat(
        [
            pool,
            pd.DataFrame(
                {
                    "player_id": [6],
                    "player_name": ["Riv C"],
                    "positions": ["2B"],
                    "mlb_id": [106],
                    "team": ["ATL"],
                    "is_hitter": [1],
                    "r": [92],
                    "hr": [28],
                    "rbi": [88],
                    "sb": [25],
                    "ab": [555],
                    "h": [162],
                    "bb": [58],
                    "hbp": [4],
                    "sf": [5],
                    "obp": [0.375],
                    "avg": [0.292],
                }
            ),
        ],
        ignore_index=True,
    )
    raw = [
        {"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Rival", "user_sgp_gain": 0.9},
        {"giving_ids": [2], "receiving_ids": [6], "opponent_team": "Other", "user_sgp_gain": 0.5},
    ]
    out = svc._build_suggestions(raw, pool, [1, 2, 3])
    assert len(out) == 2
    assert all(s.net_sgp > 0.5 for s in out)


def test_build_suggestions_dedupes_same_partner_and_receiving():
    """Two engine trades with the SAME partner + SAME receiving set (give-side differs)
    collapse to ONE card — the near-duplicate the live UI showed. Both are genuine
    upgrades so neither is filtered first; the dedupe keeps the first."""
    svc = TradeFinderService()
    pool = _hitter_pool()
    raw = [
        {"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Rival", "user_sgp_gain": 0.9},
        {"giving_ids": [2], "receiving_ids": [4], "opponent_team": "Rival", "user_sgp_gain": 0.6},
    ]
    out = svc._build_suggestions(raw, pool, [1, 2, 3])
    keys = {(s.partner_team, tuple(sorted(p.id for p in s.receiving))) for s in out}
    assert len(keys) == 1  # collapsed
    assert len(out) == 1


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
    """Engine ran AND a genuine upgrade survived the honest filter → reason='ok'."""
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
                "giving_ids": [1],  # weak hitter
                "receiving_ids": [4],  # strong hitter → genuine upgrade
                "opponent_team": "Over the Rembow",
                "user_sgp_gain": 1.2,
                "grade": "B+",
            }
        ],
    )
    resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
    assert resp.reason == "ok"
    assert len(resp.suggestions) == 1
    assert resp.suggestions[0].net_sgp > 0.5


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


# ── Value basis: ACTUAL 2026 YTD season stats, not the (compressed) projections ──


def test_player_value_uses_ytd_not_projection():
    """Trade value must come from ACTUAL 2026 YTD production, not the projection cols
    (which over-rate scrubs — Olson #23 and Moniak #2021 were within 0.7 SGP on
    projections). A player with elite projections but a bust YTD must score BELOW a
    player with bust projections but an elite YTD."""
    pool = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Proj Stud / YTD Bust", "Proj Bust / YTD Stud"],
            "positions": ["OF", "OF"],
            "is_hitter": [1, 1],
            # projection columns (the OLD basis)
            "r": [100, 20],
            "hr": [40, 3],
            "rbi": [110, 25],
            "sb": [20, 1],
            "avg": [0.310, 0.210],
            "obp": [0.390, 0.270],
            "ab": [600, 300],
            "h": [186, 63],
            "bb": [70, 20],
            "hbp": [3, 2],
            "sf": [5, 3],
            "pa": [680, 330],
            # ACTUAL YTD (the NEW basis) — deliberately inverted from the projections
            "ytd_r": [15, 70],
            "ytd_hr": [2, 25],
            "ytd_rbi": [18, 75],
            "ytd_sb": [0, 12],
            "ytd_avg": [0.200, 0.300],
            "ytd_obp": [0.260, 0.380],
            "ytd_ab": [120, 280],
            "ytd_h": [24, 84],
            "ytd_bb": [10, 45],
            "ytd_hbp": [1, 3],
            "ytd_sf": [2, 4],
            "ytd_pa": [135, 330],
        }
    )
    per_cat = TradeFinderService._player_sgp_lookup(pool)
    v_projstud = sum(per_cat(1).values())  # elite projection, bust YTD
    v_ytdstud = sum(per_cat(2).values())  # bust projection, elite YTD
    assert v_ytdstud > v_projstud, (
        f"value must track YTD, not projection: YTD-stud={v_ytdstud:.2f} should beat YTD-bust={v_projstud:.2f}"
    )


def test_player_value_falls_back_to_projection_when_no_ytd():
    """A player with NO YTD sample (ytd_ab 0 — e.g. a yet-to-debut callup) is valued off
    projections, not dropped to 0."""
    pool = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Callup"],
            "positions": ["OF"],
            "is_hitter": [1],
            "r": [80],
            "hr": [20],
            "rbi": [70],
            "sb": [10],
            "avg": [0.280],
            "obp": [0.350],
            "ab": [500],
            "h": [140],
            "bb": [55],
            "hbp": [3],
            "sf": [4],
            "pa": [562],
            "ytd_r": [0],
            "ytd_hr": [0],
            "ytd_rbi": [0],
            "ytd_sb": [0],
            "ytd_avg": [0.0],
            "ytd_obp": [0.0],
            "ytd_ab": [0],
            "ytd_h": [0],
            "ytd_bb": [0],
            "ytd_hbp": [0],
            "ytd_sf": [0],
            "ytd_pa": [0],
        }
    )
    per_cat = TradeFinderService._player_sgp_lookup(pool)
    assert sum(per_cat(1).values()) > 0, "no-YTD player must fall back to projection value, not 0"


# ── Need-weighting, small-sample guard, best-first sort ───────────────────────


def test_category_need_weights_reflects_standing():
    """Need weight is HIGHER where the user is weak (more to gain), LOWER where strong;
    clamped to [0.6, 1.6]."""
    from api.services.trade_finder_service import _category_need_weights
    from src.valuation import LeagueConfig

    totals = {
        "ME": {"HR": 8, "SB": 60},  # weakest HR, strongest SB
        "A": {"HR": 30, "SB": 12},
        "B": {"HR": 34, "SB": 15},
        "C": {"HR": 40, "SB": 8},
    }
    w = _category_need_weights(totals, "ME", LeagueConfig())
    assert w["HR"] > 1.05, w  # weak in HR → needs it more
    assert w["SB"] < 0.95, w  # strong in SB → needs it less
    assert 0.6 <= w["HR"] <= 1.6 and 0.6 <= w["SB"] <= 1.6  # clamped


def test_sample_reliability_mutes_tiny_samples():
    """A tiny YTD sample → low rate-stat reliability; a full sample → 1.0."""
    from api.services.trade_finder_service import _sample_reliability

    assert _sample_reliability(pd.Series({"is_hitter": 0, "ytd_ip": 60})) == 1.0
    assert _sample_reliability(pd.Series({"is_hitter": 0, "ytd_ip": 9})) < 0.5  # 9-IP WHIP is noise
    assert _sample_reliability(pd.Series({"is_hitter": 1, "ytd_ab": 300})) == 1.0
    assert _sample_reliability(pd.Series({"is_hitter": 1, "ytd_ab": 20})) < 0.5


def test_build_suggestions_sorted_best_first():
    """Surfaced suggestions are ordered by net_sgp (need-weighted value) descending —
    the strongest trade first."""
    pool = _hitter_pool()  # player 4 is the strongest hitter; 1-3 weaker
    raw = [
        {"giving_ids": [2], "receiving_ids": [4], "opponent_team": "Rival A"},
        {"giving_ids": [3], "receiving_ids": [4], "opponent_team": "Rival B"},
    ]
    out = TradeFinderService()._build_suggestions(raw, pool, [1, 2, 3])
    nets = [s.net_sgp for s in out]
    assert nets == sorted(nets, reverse=True), nets
