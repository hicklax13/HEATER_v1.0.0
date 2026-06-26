from api.services.lineup_service import LineupService


def test_team_strength_built_from_schedule(monkeypatch):
    import src.game_day as gd
    import src.valuation as val

    monkeypatch.setattr(
        val, "team_name_to_abbr", lambda x: {"New York Yankees": "NYY", "Boston Red Sox": "BOS"}.get(x, x)
    )
    monkeypatch.setattr(val, "canonicalize_team", lambda x: x)
    monkeypatch.setattr(gd, "get_team_strength", lambda a: {"wrc_plus": 110} if a == "NYY" else {"wrc_plus": 95})
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox"}]
    ts = LineupService()._team_strength(sched)
    assert ts.get("NYY") == {"wrc_plus": 110}
    assert ts.get("BOS") == {"wrc_plus": 95}


def test_recent_form_keyed_by_player_id(monkeypatch):
    import pandas as pd

    import src.game_day as gd

    monkeypatch.setattr(gd, "get_player_recent_form_cached", lambda mlb: {"form": "hot"} if mlb == 545361 else {})
    roster = pd.DataFrame([{"player_id": 7}, {"player_id": 9}])
    pool = pd.DataFrame([{"player_id": 7, "mlb_id": 545361}, {"player_id": 9, "mlb_id": 0}])
    rf = LineupService()._recent_form(roster, pool)
    assert rf == {7: {"form": "hot"}}  # keyed by player_id; pid 9 (no mlb_id) skipped


def test_daily_inputs_has_all_four_keys(monkeypatch):
    import src.database as db
    import src.game_day as gd

    monkeypatch.setattr(db, "load_park_factors", lambda fallback=None: {"NYY": 1.05})
    monkeypatch.setattr(gd, "get_todays_lineups", lambda sched, deadline=None: {"NYY": [1, 2, 3]})
    monkeypatch.setattr(gd, "get_team_strength", lambda a: {})
    monkeypatch.setattr(gd, "get_player_recent_form_cached", lambda mlb: {})
    out = LineupService()._daily_inputs(None, None, [])
    assert set(out) == {"park_factors", "team_strength", "confirmed_lineups", "recent_form"}
    assert out["park_factors"] == {"NYY": 1.05}
    assert out["confirmed_lineups"] == {"NYY": [1, 2, 3]}


def test_daily_inputs_graceful_on_failure(monkeypatch):
    import src.database as db

    def _boom(fallback=None):
        raise RuntimeError("db")

    monkeypatch.setattr(db, "load_park_factors", _boom)
    out = LineupService()._daily_inputs(None, None, [])
    assert out["park_factors"] == {}  # never raises


def test_daily_forwards_rich_inputs(monkeypatch):
    import pandas as pd

    import src.game_day as gd

    captured = {}

    class _FakePipeline:
        def __init__(self, *a, **k):
            pass

        def optimize(self, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr("src.optimizer.pipeline.LineupOptimizerPipeline", _FakePipeline)
    monkeypatch.setattr("src.valuation.LeagueConfig", lambda: object())
    monkeypatch.setattr(gd, "get_target_game_date", lambda: "2026-04-05")

    class _YDS:
        def get_team_roster(self, team_name):
            return pd.DataFrame([{"player_id": 1}])

        def get_matchup(self):
            return None

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _YDS())
    monkeypatch.setattr(LineupService, "_schedule_today", staticmethod(lambda d: []))
    monkeypatch.setattr(LineupService, "_load_pool", staticmethod(lambda: None))
    monkeypatch.setattr(
        LineupService,
        "_daily_inputs",
        lambda self, r, p, s: {
            "park_factors": {"NYY": 1.05},
            "team_strength": {"NYY": {}},
            "confirmed_lineups": {"NYY": []},
            "recent_form": {1: {}},
        },
    )
    LineupService().optimize("Team Hickey", mode="daily")
    assert captured.get("park_factors") == {"NYY": 1.05}
    assert captured.get("confirmed_lineups") == {"NYY": []}
    assert captured.get("recent_form") == {1: {}}
    assert captured.get("team_strength") == {"NYY": {}}
