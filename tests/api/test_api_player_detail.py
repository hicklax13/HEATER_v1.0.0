"""Player-detail endpoint (slice 1): GET /api/players/{mlb_id} → PlayerDetailResponse.

Router test uses a fake-service override (DB-free). Service tests monkeypatch the
engine loaders at their source modules (the service imports them lazily), so they
run against synthetic frames without the live DB."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from api.contracts.player_detail import PlayerDetailResponse
from api.deps import get_player_detail_service
from api.main import create_app


# ── router: contract + 404 ───────────────────────────────────────────────────
class _FakeService:
    def __init__(self, resp=None, raises=None):
        self._resp = resp
        self._raises = raises

    def get(self, mlb_id: int) -> PlayerDetailResponse:
        if self._raises:
            raise self._raises
        return self._resp or PlayerDetailResponse(mlb_id=mlb_id, name="Mike Trout", pos="CF")


def _client(service):
    app = create_app()
    app.dependency_overrides[get_player_detail_service] = lambda: service
    return TestClient(app)


def test_player_detail_router_returns_contract():
    r = _client(_FakeService()).get("/api/players/545361")
    assert r.status_code == 200
    body = r.json()
    assert body["mlb_id"] == 545361
    assert body["name"] == "Mike Trout"


def test_player_detail_router_404_for_unknown():
    svc = _FakeService(raises=HTTPException(status_code=404, detail="Player not found"))
    assert _client(svc).get("/api/players/1").status_code == 404


# ── service: real field mapping (DB-free synthetic frames) ───────────────────
def _svc():
    from api.services.player_detail_service import PlayerDetailService

    return PlayerDetailService()


def _hitter_pool():
    return pd.DataFrame(
        [
            {
                "player_id": 36,
                "mlb_id": 545361.0,
                "name": "Mike Trout",
                "positions": "CF",
                "team": "LAA",
                "bats": "R",
                "throws": "R",
                "is_hitter": 1,
                "percent_owned": 88.0,
                "consensus_rank": 12,
                "ytd_gp": 67,
                "ytd_r": 50,
                "ytd_hr": 14,
                "ytd_rbi": 40,
                "ytd_sb": 5,
                "ytd_avg": 0.223,
                "ytd_obp": 0.34,
                "r": 90,
                "hr": 30,
                "rbi": 85,
                "sb": 9,
                "avg": 0.260,
                "obp": 0.360,
            }
        ]
    )


def test_service_maps_hitter_identity_and_season(monkeypatch):
    monkeypatch.setattr("src.database.load_player_pool", _hitter_pool)
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.database.load_transactions", lambda *a, **k: pd.DataFrame())

    d = _svc().get(545361)
    assert d.mlb_id == 545361
    assert d.name == "Mike Trout"
    assert d.pos == "CF"
    assert d.is_pitcher is False
    assert d.bats == "Bats Right"
    assert d.own_pct == 88.0
    # season line maps the hitter categories (R/HR/RBI/SB/AVG/OBP) from ytd_*
    hr = next(s for s in d.stats if s.cat == "HR")
    assert hr.season == "14"
    avg = next(s for s in d.stats if s.cat == "AVG")
    assert avg.season in (".223", "0.223")
    # ROS projections come from the pool's blended (non-ytd) columns
    hr_proj = next(p for p in d.projections if p.cat == "HR")
    assert hr_proj.ros == "30"


def test_service_rostered_by_from_league_rosters(monkeypatch):
    monkeypatch.setattr("src.database.load_player_pool", _hitter_pool)
    monkeypatch.setattr(
        "src.database.load_league_rosters",
        lambda: pd.DataFrame([{"player_id": 36, "team_name": "BUBBA CROSBY"}]),
    )
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.database.load_transactions", lambda *a, **k: pd.DataFrame())
    assert _svc().get(545361).rostered_by == "BUBBA CROSBY"


def test_service_free_agent_when_not_rostered(monkeypatch):
    monkeypatch.setattr("src.database.load_player_pool", _hitter_pool)
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.database.load_transactions", lambda *a, **k: pd.DataFrame())
    assert _svc().get(545361).rostered_by == "Free Agent"


def test_service_404_for_unknown_mlb_id(monkeypatch):
    monkeypatch.setattr("src.database.load_player_pool", _hitter_pool)
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.database.load_transactions", lambda *a, **k: pd.DataFrame())
    with pytest.raises(HTTPException) as ei:
        _svc().get(999999)
    assert ei.value.status_code == 404


def test_service_nan_safe(monkeypatch):
    pool = _hitter_pool()
    pool.loc[0, "percent_owned"] = float("nan")
    pool.loc[0, "ytd_hr"] = float("nan")
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    monkeypatch.setattr("src.database.load_league_rosters", lambda: pd.DataFrame())
    monkeypatch.setattr("src.database.load_season_stats", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.database.load_transactions", lambda *a, **k: pd.DataFrame())
    d = _svc().get(545361)  # must not raise
    assert d.own_pct == 0.0
