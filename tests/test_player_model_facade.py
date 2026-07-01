"""Tests for the unified player_model facade (Phase 1 slice 4)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.valuation import LeagueConfig


def _pool(n_hitters=8, n_pitchers=6, seed=0):
    """A small synthetic pool: fantasy-relevant hitters + pitchers + a couple scrubs."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_hitters):
        rows.append(
            dict(
                player_id=100 + i,
                name=f"H{i}",
                is_hitter=1,
                age=27,
                positions="OF",
                health_score=0.9,
                r=70 + i * 4,
                hr=18 + i * 2,
                rbi=70 + i * 3,
                sb=8 + i,
                avg=0.250 + i * 0.005,
                obp=0.320 + i * 0.005,
                ab=520,
                pa=580,
                ytd_pa=300,
                ytd_ip=0.0,
                w=0,
                l=0,
                sv=0,
                k=0,
                era=0.0,
                whip=0.0,
                ip=0.0,
            )
        )
    for i in range(n_pitchers):
        rows.append(
            dict(
                player_id=200 + i,
                name=f"P{i}",
                is_hitter=0,
                age=28,
                positions="SP",
                health_score=0.9,
                r=0,
                hr=0,
                rbi=0,
                sb=0,
                avg=0.0,
                obp=0.0,
                ab=0,
                pa=0,
                w=8 + i,
                l=7,
                sv=0,
                k=160 + i * 10,
                era=4.2 - i * 0.2,
                whip=1.30 - i * 0.03,
                ip=150,
                ytd_pa=0,
                ytd_ip=60.0,
            )
        )
    # two zero-volume scrubs that must be filtered out of league baselines
    rows.append(
        dict(
            player_id=900,
            name="ScrubH",
            is_hitter=1,
            age=24,
            positions="OF",
            health_score=0.85,
            r=0,
            hr=0,
            rbi=0,
            sb=0,
            avg=0.0,
            obp=0.0,
            ab=0,
            pa=0,
            ytd_pa=0,
            ytd_ip=0.0,
            w=0,
            l=0,
            sv=0,
            k=0,
            era=0.0,
            whip=0.0,
            ip=0.0,
        )
    )
    rows.append(
        dict(
            player_id=901,
            name="ScrubP",
            is_hitter=0,
            age=24,
            positions="SP",
            health_score=0.85,
            r=0,
            hr=0,
            rbi=0,
            sb=0,
            avg=0.0,
            obp=0.0,
            ab=0,
            pa=0,
            w=0,
            l=0,
            sv=0,
            k=0,
            era=0.0,
            whip=0.0,
            ip=0.0,
            ytd_pa=0,
            ytd_ip=0.0,
        )
    )
    return pd.DataFrame(rows)


def test_league_context_has_all_categories():
    from src.player_model.model import build_league_context

    cfg = LeagueConfig()
    ctx = build_league_context(_pool(), cfg)
    assert set(ctx.means) == set(cfg.all_categories)
    assert set(ctx.sds) == set(cfg.all_categories)
    assert all(math.isfinite(v) for v in ctx.means.values())
    assert all(v >= 0 for v in ctx.sds.values())


def test_league_context_counting_mean_is_per_week():
    from src.player_model.model import build_league_context

    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg, weeks=26)
    # HR league mean should equal mean(season HR of relevant hitters) / 26.
    relevant = pool[(pool["is_hitter"] == 1) & (pool["pa"] >= 1)]
    expected = float(relevant["hr"].mean()) / 26
    assert ctx.means["HR"] == pytest.approx(expected, rel=1e-6)


def test_league_context_excludes_zero_volume_scrubs():
    from src.player_model.model import build_league_context

    cfg = LeagueConfig()
    ctx_full = build_league_context(_pool(), cfg)
    # The two 0-PA/0-IP scrubs must not drag the AVG/ERA baselines toward 0.
    assert ctx_full.means["AVG"] > 0.20
    assert ctx_full.means["ERA"] > 2.0


def test_league_context_empty_pool_no_raise():
    from src.player_model.model import build_league_context

    cfg = LeagueConfig()
    ctx = build_league_context(pd.DataFrame(), cfg)
    assert set(ctx.means) == set(cfg.all_categories)
    assert all(v == 0.0 for v in ctx.sds.values())


def test_build_player_model_assembles_all_three_layers():
    from src.player_model.availability import AvailabilitySurvival
    from src.player_model.model import build_league_context, build_player_model
    from src.player_model.posterior import CategoryPosterior

    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    elite = pool.iloc[7]  # highest-stat hitter (H7)
    pm = build_player_model(elite, ctx, cfg)
    assert pm.player_id == int(elite["player_id"])
    assert pm.is_hitter is True
    assert set(pm.posteriors) == set(cfg.hitting_categories)
    assert all(isinstance(v, CategoryPosterior) for v in pm.posteriors.values())
    assert isinstance(pm.availability, AvailabilitySurvival)
    assert math.isfinite(pm.g_score)
    assert set(pm.g_by_category) == set(cfg.hitting_categories)


def test_build_player_model_gscore_orders_players():
    from src.player_model.model import build_league_context, build_player_model

    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    elite = build_player_model(pool.iloc[7], ctx, cfg)  # best hitter
    weak = build_player_model(pool.iloc[0], ctx, cfg)  # weakest hitter
    assert elite.g_score > weak.g_score


def test_build_player_model_availability_reflects_status():
    from src.player_model.model import build_league_context, build_player_model

    cfg = LeagueConfig()
    pool = _pool()
    ctx = build_league_context(pool, cfg)
    healthy = build_player_model(pool.iloc[3], ctx, cfg, status=None)
    injured = build_player_model(pool.iloc[3], ctx, cfg, status="IL60")
    assert injured.availability.expected_active_fraction < healthy.availability.expected_active_fraction


def test_build_player_models_batch_keyed_by_id():
    from src.player_model.model import build_player_models

    cfg = LeagueConfig()
    pool = _pool()
    models = build_player_models(pool, cfg)
    assert len(models) == len(pool)
    assert all(pid == pm.player_id for pid, pm in models.items())
    # hitters get hitting cats, pitchers get pitching cats
    h = models[107]
    assert set(h.posteriors) == set(cfg.hitting_categories)
    p = models[205]
    assert set(p.posteriors) == set(cfg.pitching_categories)


def test_build_player_models_injects_status_map():
    from src.player_model.model import build_player_models

    cfg = LeagueConfig()
    pool = _pool()
    models = build_player_models(pool, cfg, status_map={103: "IL60"})
    assert models[103].availability.status == "IL60"
    assert models[104].availability.status == "ACTIVE"  # untouched


def test_public_api_reexports():
    import src.player_model as pmpkg

    for name in (
        "PlayerModel",
        "build_player_model",
        "build_player_models",
        "build_league_context",
        "CategoryPosterior",
        "AvailabilitySurvival",
        "LeagueContext",
    ):
        assert hasattr(pmpkg, name)


def test_facade_never_raises_on_sparse_pool():
    from src.player_model.model import build_player_models

    cfg = LeagueConfig()
    sparse = pd.DataFrame(
        [
            {"player_id": 1, "is_hitter": 1},  # missing all stats
            {"player_id": 2, "is_hitter": 0, "k": np.nan, "ip": 150},
            {"player_id": 3},  # missing is_hitter
        ]
    )
    models = build_player_models(sparse, cfg)
    assert len(models) >= 1
    for pm in models.values():
        assert math.isfinite(pm.g_score)
        assert pm.availability is not None


def test_full_player_model_suite_green_together():
    # Sanity: the per-cat posterior + availability + gscore + facade all coexist.
    from src.player_model import PlayerModel, build_player_models

    cfg = LeagueConfig()
    models = build_player_models(_pool(), cfg)
    assert all(isinstance(pm, PlayerModel) for pm in models.values())
