"""leaders/overall: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.leaders_overall_service import (
    _LENS_META,
    _norm_delta,
    _norm_value,
    _overall_stats,
    _to_overall_row,
)


def test_norm_value_maps_and_clamps():
    # category_value is a SUM of up to 6 z-scores (~±10), not a single z.
    assert _norm_value(0.0) == 50.0  # (0+10)/20*100
    assert _norm_value(10.0) == 100.0
    assert _norm_value(-10.0) == 0.0
    assert _norm_value(99.0) == 100.0  # clamp
    assert _norm_value(float("nan")) == 0.0  # NaN-safe
    # regression guard: a realistic top sum-of-z (~6.5) must NOT saturate to 100
    assert _norm_value(6.5) == 82.5
    assert _norm_value(4.0) < 100.0  # was wrongly 100 under the single-z window


def test_norm_delta_maps_and_clamps():
    assert _norm_delta(0.0) == 50.0  # (0+3)/6*100
    assert _norm_delta(3.0) == 100.0
    assert _norm_delta(-3.0) == 0.0
    assert _norm_delta(float("nan")) == 0.0


def test_overall_stats_hitter_and_pitcher():
    hrow = {"ytd_hr": 24, "ytd_r": 58, "ytd_avg": 0.322}
    assert _overall_stats(hrow, True) == ["24 HR", "58 R", ".322 AVG"]
    prow = {"ytd_k": 180, "ytd_era": 3.21, "ytd_whip": 1.05}
    assert _overall_stats(prow, False) == ["180 K", "3.21 ERA", "1.05 WHIP"]


def test_lens_meta_covers_all_five():
    assert set(_LENS_META) == {"overall", "hot", "cold", "breakout", "sell"}
    assert _LENS_META["overall"] == ("", "flat", _LENS_META["overall"][2])  # tag empty for overall
    assert _LENS_META["hot"][0] == "hot" and _LENS_META["hot"][1] == "up"
    assert _LENS_META["cold"][1] == "down"


def test_to_overall_row_enriches_and_stamps_lens():
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "positions": "OF",
                "mlb_id": 592450,
                "team": "NYY",
                "is_hitter": True,
                "ytd_hr": 30,
                "ytd_r": 70,
                "ytd_avg": 0.31,
            }
        ]
    )
    row = {"player_id": 1, "_value": 88.0}
    item = _to_overall_row(2, row, pool, "hot")
    assert item.rank == 2
    assert item.value == 88.0
    assert item.player.mlb_id == 592450
    assert item.player.team_id == 147  # NYY
    assert item.hitter is True
    assert item.tag == "hot"
    assert item.trend == "up"
    assert item.note == _LENS_META["hot"][2]
    assert item.stats == ["30 HR", "70 R", ".310 AVG"]


def test_to_overall_row_missing_pool_row_degrades():
    item = _to_overall_row(1, {"player_id": 999, "_value": 50.0}, pd.DataFrame(), "overall")
    assert item.player.name == "Player 999"
    assert item.stats == []  # no pool row → no stats
    assert item.tag == ""
    assert item.value == 50.0


def test_leaders_overall_endpoint_returns_contract():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.leaders import LeadersOverallResponse, OverallLeaderRow
    from api.deps import get_leaders_overall_service
    from api.main import create_app

    class _Fake:
        def get_leaders_overall(self, lens="overall", limit=25):
            return LeadersOverallResponse(
                lens=lens,
                rows=[
                    OverallLeaderRow(
                        rank=1,
                        player=PlayerRef(
                            id=1, mlb_id=592450, name="Judge", positions="OF", team_abbr="NYY", team_id=147
                        ),
                        value=92.0,
                        stats=["30 HR", "70 R", ".310 AVG"],
                        trend="up",
                        tag="hot",
                        note="Trending hot",
                        hitter=True,
                    )
                ],
            )

    app = create_app()
    app.dependency_overrides[get_leaders_overall_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/leaders/overall?lens=hot").json()
        assert body["lens"] == "hot"
        r = body["rows"][0]
        assert r["player"]["mlb_id"] == 592450
        assert r["value"] == 92.0
        assert r["stats"] == ["30 HR", "70 R", ".310 AVG"]
        assert r["tag"] == "hot"
    finally:
        app.dependency_overrides.clear()


# ── breakout perf guard ──────────────────────────────────────────────────────
# compute_breakout_scores_batch is O(n²) (each player is percentile-ranked vs the
# whole input), so the fix BOUNDS the input size at the service layer. These guard
# the bound deterministically (no live DB / no wall-clock flakiness) — a count cap
# is the real regression guard; a timing assertion would be flaky and DB-dependent.


def _synthetic_pool(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": range(n),
            "name": [f"P{i}" for i in range(n)],
            "barrel_pct": [10.0] * n,
            "xwoba": [0.330] * n,
            "hard_hit_pct": [40.0] * n,
            "consensus_rank": list(range(n, 0, -1)),  # reverse so sort actually reorders
            "percent_owned": [float(i) for i in range(n)],
        }
    )


def test_filter_breakout_candidates_caps_input_size():
    from api.services.leaders_overall_service import _filter_breakout_candidates

    out = _filter_breakout_candidates(_synthetic_pool(1500), top_n=500)
    assert len(out) <= 500  # the O(n²) regression guard: bounded input


def test_filter_breakout_candidates_prefers_relevance():
    """The bounded set must be the most fantasy-relevant players (lowest
    consensus_rank), not an arbitrary slice of the pool's natural order."""
    from api.services.leaders_overall_service import _filter_breakout_candidates

    out = _filter_breakout_candidates(_synthetic_pool(1500), top_n=10)
    assert len(out) == 10
    assert out["consensus_rank"].max() <= 10  # kept the 10 best ranks, dropped the rest


def test_filter_breakout_candidates_is_total():
    """Never raises / never empty-by-surprise: empty pool, and a pool missing the
    Statcast + relevance columns entirely."""
    from api.services.leaders_overall_service import _filter_breakout_candidates

    assert _filter_breakout_candidates(pd.DataFrame(), top_n=500).empty
    bare = pd.DataFrame({"player_id": range(30), "name": [f"P{i}" for i in range(30)]})
    out = _filter_breakout_candidates(bare, top_n=10)
    assert len(out) == 10  # falls back to natural-order head(top_n)


def test_sell_lens_maps_candidates(monkeypatch):
    """Regression guard for the wiring bug: the service's narrow season_stats query
    starved compute_sustainability_score (→ every score 0.5 → 0 sell rows). With the
    engine returning candidates, the sell branch must map them to OverallLeaderRow
    stamped tag='sell'. Fabricated engine output → deterministic, no live DB."""
    import api.services.leaders_overall_service as svc

    pool = pd.DataFrame(
        [{"player_id": 7, "name": "Hot Regression", "positions": "OF", "is_hitter": True, "ytd_hr": 18}]
    )
    fake_candidates = pd.DataFrame(
        [{"player_id": 7, "trend_delta": 2.4, "sustainability_score": 0.2, "is_hitter": True}]
    )
    # The service imports these inside _ranked() (from src... import ...), so patch
    # them at their source module — that's where the late binding resolves them.
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    monkeypatch.setattr(svc.LeadersOverallService, "_load_season_stats", staticmethod(lambda: pool))
    monkeypatch.setattr("src.trend_tracker.detect_sell_high_candidates", lambda *a, **k: fake_candidates)

    result = svc.LeadersOverallService().get_leaders_overall(lens="sell", limit=25)
    assert len(result.rows) == 1
    assert result.rows[0].player.id == 7
    assert result.rows[0].tag == "sell"
    assert result.rows[0].trend == _LENS_META["sell"][1]
