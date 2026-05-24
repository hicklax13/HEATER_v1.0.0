"""Structural-invariant guard: G-score (Rosenof) computation per report B.5/C.2.

The Enhanced Trade Engine report Section B.5 / C.2 references Rosenof
arXiv:2307.02188 which proves that for H2H Cats leagues, the optimal
player-valuation metric is NOT z-score but G-score:

    G_i = (μ_i − μ̄_peers) / sqrt(σ̄²_peers + σ*²)

Where:
  - μ̄_peers, σ̄_peers — cross-sectional mean and std across peer group
  - σ* — team-level weekly variance (per-cat CV × peer mean)

The σ* term in the denominator means VOLATILE players (high CV cats)
get smaller G-scores than equivalent z-scores would predict. This
correctly penalizes week-to-week boom-bust profiles.

Contract locked:
  - compute_player_gscores produces g_{cat} columns + g_composite
  - G-score values ≤ |z-score| for the same player (σ* > 0 → denom > σ̄)
  - Pre-week-1 (μ̄ unknown) edge cases don't crash
  - Inverse cats flipped same way z-score does
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.engine.portfolio.valuation import compute_player_gscores, compute_player_zscores
from src.valuation import LeagueConfig


def _make_pool() -> pd.DataFrame:
    """3 hitters with varied HR/AVG to exercise G-score peer math."""
    rows = []
    # 3 hitters with different power profiles
    for i, (hr, avg) in enumerate([(40, 0.295), (25, 0.260), (10, 0.225)]):
        rows.append(
            {
                "player_id": i + 1,
                "name": f"H{i + 1}",
                "player_name": f"H{i + 1}",
                "is_hitter": 1,
                "positions": "OF",
                "r": 80,
                "hr": hr,
                "rbi": 80,
                "sb": 8,
                "h": int(500 * avg),
                "ab": 500,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "pa": 565,
                "avg": avg,
                "obp": avg + 0.07,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "ytd_ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            }
        )
    # 2 pitchers for peer-group symmetry
    for i, (era, whip) in enumerate([(2.80, 1.05), (4.20, 1.30)]):
        rows.append(
            {
                "player_id": 4 + i,
                "name": f"P{i + 1}",
                "player_name": f"P{i + 1}",
                "is_hitter": 0,
                "positions": "SP",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "avg": 0,
                "obp": 0,
                "w": 12,
                "l": 6,
                "sv": 0,
                "k": 180,
                "ip": 180,
                "ytd_ip": 0,
                "er": int(180 * era / 9),
                "bb_allowed": int(180 * (whip * 0.4)),
                "h_allowed": int(180 * (whip * 0.6)),
                "era": era,
                "whip": whip,
            }
        )
    return pd.DataFrame(rows)


# ── Function existence + structure ───────────────────────────────────


def test_compute_player_gscores_returns_g_columns() -> None:
    """Output must have g_{cat} columns for each league category + g_composite."""
    pool = _make_pool()
    cfg = LeagueConfig()
    result = compute_player_gscores(pool, cfg)
    for cat in cfg.all_categories:
        assert f"g_{cat}" in result.columns, f"Missing g_{cat} column"
    assert "g_composite" in result.columns


def test_g_score_magnitude_lte_z_score_magnitude() -> None:
    """|G_i| ≤ |z_i| because G's denominator is sqrt(σ̄² + σ*²) ≥ σ̄.

    The σ* term in G's denominator only ADDS to the variance, so the
    G-score is always shrunk relative to z-score (in magnitude)."""
    pool = _make_pool()
    cfg = LeagueConfig()
    z = compute_player_zscores(pool, cfg)
    g = compute_player_gscores(pool, cfg)
    for cat in ("HR", "AVG", "ERA"):  # sample a few cats
        z_col = f"z_{cat}"
        g_col = f"g_{cat}"
        for idx in range(len(z)):
            z_val = abs(z.iloc[idx][z_col])
            g_val = abs(g.iloc[idx][g_col])
            if z_val > 1e-9:  # skip zero rows
                assert g_val <= z_val + 1e-9, (
                    f"G-score magnitude {g_val} > z-score magnitude {z_val} at row {idx}, cat {cat}"
                )


def test_inverse_cat_sign_flipped() -> None:
    """ERA/WHIP: lower is better → G-score must be positive for low ERA."""
    pool = _make_pool()
    g = compute_player_gscores(pool, LeagueConfig())
    # Player 4 has ERA 2.80 (better), player 5 has 4.20 (worse)
    g_era_p4 = g.iloc[3]["g_ERA"]
    g_era_p5 = g.iloc[4]["g_ERA"]
    assert g_era_p4 > g_era_p5, f"Better-ERA player must have higher g_ERA; got p4={g_era_p4} vs p5={g_era_p5}"


def test_higher_cat_value_higher_g_score_non_inverse() -> None:
    """For HR (non-inverse), higher HR → higher G-score."""
    pool = _make_pool()
    g = compute_player_gscores(pool, LeagueConfig())
    # Player 1: 40 HR, Player 3: 10 HR
    g_hr_top = g.iloc[0]["g_HR"]
    g_hr_bot = g.iloc[2]["g_HR"]
    assert g_hr_top > g_hr_bot, f"Higher HR must have higher G-score; got p1={g_hr_top} vs p3={g_hr_bot}"


def test_cross_peer_group_zero() -> None:
    """Pitchers should have zero G-score in hitting cats and vice versa."""
    pool = _make_pool()
    g = compute_player_gscores(pool, LeagueConfig())
    # Pitcher (player 4) in HR cat must be 0
    assert g.iloc[3]["g_HR"] == 0.0
    # Hitter (player 1) in ERA cat must be 0
    assert g.iloc[0]["g_ERA"] == 0.0


def test_g_composite_sum_of_g_cols() -> None:
    """g_composite must equal the sum of g_{cat} columns."""
    pool = _make_pool()
    cfg = LeagueConfig()
    g = compute_player_gscores(pool, cfg)
    g_cols = [f"g_{cat}" for cat in cfg.all_categories]
    for idx in range(len(g)):
        expected = sum(g.iloc[idx][c] for c in g_cols)
        actual = g.iloc[idx]["g_composite"]
        assert abs(actual - expected) < 1e-9


def test_custom_sigma_star_cv_override() -> None:
    """sigma_star_cv override changes G-scores (higher CV → smaller G)."""
    pool = _make_pool()
    cfg = LeagueConfig()
    g_default = compute_player_gscores(pool, cfg)
    g_high_cv = compute_player_gscores(pool, cfg, sigma_star_cv={"HR": 0.50})  # 50% CV
    # Higher CV on HR → larger σ* → smaller |g_HR|
    p1_g_default = abs(g_default.iloc[0]["g_HR"])
    p1_g_high = abs(g_high_cv.iloc[0]["g_HR"])
    if p1_g_default > 1e-9:
        assert p1_g_high < p1_g_default, (
            f"Higher CV must shrink G-score; got default={p1_g_default} vs high_cv={p1_g_high}"
        )


def test_empty_pool_returns_zero_g_composite() -> None:
    """Edge case: empty pool returns empty result."""
    pool = pd.DataFrame(columns=["player_id", "is_hitter", "hr", "ip"])
    result = compute_player_gscores(pool, LeagueConfig())
    assert "g_composite" in result.columns
    assert len(result) == 0
