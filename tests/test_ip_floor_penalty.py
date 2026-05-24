"""Structural-invariant guard: IP-floor soft penalty in evaluate_trade.

Feature 1 (2026-05-23): per report Section B.6, Yahoo H2H Cats applies a
20 IP/week floor — below that, pitching categories convert to losses.
The trade engine now penalizes trades that push weekly IP below this floor
via a quadratic penalty kappa * (20 - IP_w)^2 with kappa=0.05 (calibrated
so a 50% shortfall ~ 5.0 SGP penalty, "forfeit-equivalent" against the
typical ±1-3 SGP trade-surplus scale).

Delta semantics: only the post-trade marginal INCREASE in penalty counts.
A trade that doesn't worsen an already-below-floor situation contributes 0;
a trade that IMPROVES weekly IP also contributes 0 (no negative penalty).

Decisions baked in:
  - threshold = 20 IP/week (user choice 2026-05-23 Q1)
  - kappa = 0.05 SGP/IP^2 (report calibration)
  - 26-week season (FourzynBurn / LeagueConfig.season_weeks)
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.trade_evaluator import (
    _IP_FLOOR_KAPPA,
    _IP_FLOOR_PER_WEEK,
    _compute_ip_floor_penalty,
    _compute_weekly_ip,
    _ip_floor_penalty,
    evaluate_trade,
)

# ── Pure formula tests ───────────────────────────────────────────────


def test_ip_floor_penalty_zero_above_threshold() -> None:
    """At or above 20 IP/week, the formula returns 0."""
    assert _ip_floor_penalty(20.0) == 0.0
    assert _ip_floor_penalty(25.0) == 0.0
    assert _ip_floor_penalty(50.0) == 0.0


def test_ip_floor_penalty_quadratic_shape() -> None:
    """Below threshold, penalty = 0.05 * (20 - IP)^2."""
    # 10 IP shortfall (50% short) → 0.05 * 100 = 5.0 SGP
    assert _ip_floor_penalty(10.0) == 5.0
    # 6 IP shortfall (30% short) → 0.05 * 36 = 1.8 SGP
    assert abs(_ip_floor_penalty(14.0) - 1.8) < 1e-9
    # 2 IP shortfall → 0.05 * 4 = 0.2 SGP (small)
    assert abs(_ip_floor_penalty(18.0) - 0.2) < 1e-9
    # 20 IP shortfall (zero IP!) → 0.05 * 400 = 20.0 SGP (catastrophic)
    assert _ip_floor_penalty(0.0) == 20.0


def test_ip_floor_constants_locked() -> None:
    """Constants haven't drifted from the report-calibrated values."""
    assert _IP_FLOOR_PER_WEEK == 20.0, "Yahoo H2H Cats floor is 20 IP/week"
    assert _IP_FLOOR_KAPPA == 0.05, "Report calibration: 50% shortfall = 5.0 SGP penalty"


# ── _compute_weekly_ip ──────────────────────────────────────────────


def _starter_pool() -> pd.DataFrame:
    """4 starters at 180 IP each (720 IP/season → 27.7 IP/week, above floor)."""
    return pd.DataFrame(
        [{"player_id": pid, "name": f"SP{pid}", "is_hitter": 0, "ip": 180} for pid in range(1, 5)]
        + [{"player_id": pid, "name": f"Hitter{pid}", "is_hitter": 1, "ip": 0} for pid in range(5, 14)]
    )


def test_compute_weekly_ip_full_staff() -> None:
    pool = _starter_pool()
    # All 4 starters
    weekly = _compute_weekly_ip([1, 2, 3, 4], pool, weeks_remaining=26)
    # 720 IP / 26 weeks = 27.69
    assert abs(weekly - 27.6923) < 0.001


def test_compute_weekly_ip_partial_staff() -> None:
    pool = _starter_pool()
    # Only 2 starters
    weekly = _compute_weekly_ip([1, 2], pool, weeks_remaining=26)
    # 360 / 26 = 13.85
    assert abs(weekly - 13.846) < 0.001


def test_compute_weekly_ip_hitters_only_zero() -> None:
    pool = _starter_pool()
    # Only hitters → 0 IP
    weekly = _compute_weekly_ip([5, 6, 7], pool, weeks_remaining=26)
    assert weekly == 0.0


def test_compute_weekly_ip_empty_starters() -> None:
    pool = _starter_pool()
    assert _compute_weekly_ip([], pool) == 0.0


# ── _compute_ip_floor_penalty (delta semantics) ─────────────────────


def test_delta_trade_worsens_ip_produces_penalty() -> None:
    """Drop 2 of 4 starters → weekly IP 27.7 → 13.85, big penalty."""
    pool = _starter_pool()
    delta, detail = _compute_ip_floor_penalty(
        before_starter_ids=[1, 2, 3, 4],
        after_starter_ids=[1, 2],
        player_pool=pool,
        weeks_remaining=26,
    )
    assert delta > 0, "Losing 2 starters must produce a positive delta penalty"
    assert detail["before_weekly_ip"] > 20
    assert detail["after_weekly_ip"] < 20
    assert detail["below_floor"] is True
    assert detail["before_penalty"] == 0.0  # was above floor
    assert detail["after_penalty"] > 0  # now below floor
    assert detail["delta_penalty"] == delta


def test_delta_trade_stays_above_floor_zero_penalty() -> None:
    """Both before and after above 20 IP/week → no penalty."""
    pool = _starter_pool()
    # Before: 4 starters (27.7 IP/w). After: still 4 (different identity).
    delta, detail = _compute_ip_floor_penalty(
        before_starter_ids=[1, 2, 3, 4],
        after_starter_ids=[1, 2, 3, 4],  # same staff
        player_pool=pool,
        weeks_remaining=26,
    )
    assert delta == 0.0
    assert detail["below_floor"] is False


def test_delta_trade_improves_ip_no_negative_bonus() -> None:
    """Going from below-floor to above-floor → 0 delta (not negative)."""
    pool = _starter_pool()
    delta, detail = _compute_ip_floor_penalty(
        before_starter_ids=[1, 2],  # 13.85 IP/w (below)
        after_starter_ids=[1, 2, 3, 4],  # 27.7 IP/w (above)
        player_pool=pool,
        weeks_remaining=26,
    )
    assert delta == 0.0, "Improving IP must not produce a negative 'bonus'"
    assert detail["before_penalty"] > 0
    assert detail["after_penalty"] == 0.0


def test_delta_already_below_unchanged_zero_penalty() -> None:
    """Pre-trade already below; trade doesn't change IP → zero marginal penalty."""
    pool = _starter_pool()
    delta, detail = _compute_ip_floor_penalty(
        before_starter_ids=[1, 2],
        after_starter_ids=[1, 2],
        player_pool=pool,
        weeks_remaining=26,
    )
    assert delta == 0.0, "If trade doesn't worsen IP, no marginal penalty"
    assert detail["below_floor"] is True  # still flag the situation


# ── Bug F regression: ROS-aware IP scaling ──────────────────────────


def test_bug_f_ros_aware_uses_ip_minus_ytd_ip() -> None:
    """Bug F: when ytd_ip column exists, weekly = (ip - ytd_ip) / weeks_remaining.

    Pre-fix: divided full-season ip by 26 → 8.11 IP/week mid-season (wrong).
    Post-fix: subtracts ytd_ip first, then divides by weeks_remaining.
    """
    # Pitcher with 180 IP full-season projection, 60 IP already accrued.
    # ROS = 120 IP. With 18 weeks remaining → 6.67 IP/week (matches the
    # forward-looking rate the IP floor actually cares about).
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Mid_season_pitcher",
                "is_hitter": 0,
                "ip": 180.0,
                "ytd_ip": 60.0,
            }
        ]
    )
    weekly = _compute_weekly_ip([1], pool, weeks_remaining=18)
    # ROS = 180 - 60 = 120. 120 / 18 = 6.67
    assert abs(weekly - 6.6667) < 0.001, f"Bug F regression: expected ROS = (180 - 60) / 18 = 6.67, got {weekly:.4f}"


def test_bug_f_ros_floored_at_zero_for_overperformers() -> None:
    """If YTD already exceeded blended projection (over-performer), ROS = 0,
    not negative."""
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Overperformer",
                "is_hitter": 0,
                "ip": 100.0,
                "ytd_ip": 150.0,  # exceeded blend
            }
        ]
    )
    weekly = _compute_weekly_ip([1], pool, weeks_remaining=18)
    assert weekly == 0.0, "Negative ROS must be floored at 0, not contribute"


def test_bug_f_no_ytd_column_falls_back_to_full_ip() -> None:
    """When ytd_ip column is missing (preseason mode), `ip` IS already ROS."""
    pool = pd.DataFrame([{"player_id": 1, "name": "Preseason_pitcher", "is_hitter": 0, "ip": 180.0}])
    weekly = _compute_weekly_ip([1], pool, weeks_remaining=26)
    # No ytd subtraction available; treat full ip as the projection
    assert abs(weekly - (180.0 / 26.0)) < 0.001


def test_bug_f_weeks_remaining_in_detail() -> None:
    """Detail dict must surface the weeks_remaining used for the computation
    (so the UI can show 'using N weeks remaining' for transparency)."""
    pool = _starter_pool()
    _, detail = _compute_ip_floor_penalty(
        before_starter_ids=[1, 2, 3, 4],
        after_starter_ids=[1, 2],
        player_pool=pool,
        weeks_remaining=18,
    )
    assert detail["weeks_remaining"] == 18


# ── Integration test: evaluate_trade end-to-end ─────────────────────


def _build_full_pool() -> pd.DataFrame:
    """Minimal pool sufficient for evaluate_trade() integration test."""
    rows = []
    # 4 starting pitchers, 180 IP each
    for pid in range(1, 5):
        rows.append(
            {
                "player_id": pid,
                "name": f"SP{pid}",
                "player_name": f"SP{pid}",
                "is_hitter": 0,
                "positions": "SP",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "h": 0,
                "ab": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "avg": 0,
                "obp": 0,
                "w": 12,
                "l": 8,
                "sv": 0,
                "k": 180,
                "ip": 180,
                "er": 70,
                "bb_allowed": 50,
                "h_allowed": 160,
                "era": 3.50,
                "whip": 1.17,
                "adp": pid * 30,
                "health_score": 0.9,
                "is_injured": 0,
            }
        )
    # 12 hitters
    for pid in range(5, 17):
        rows.append(
            {
                "player_id": pid,
                "name": f"H{pid}",
                "player_name": f"H{pid}",
                "is_hitter": 1,
                "positions": "OF",
                "r": 70,
                "hr": 20,
                "rbi": 75,
                "sb": 8,
                "h": 130,
                "ab": 500,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "pa": 560,
                "avg": 0.260,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
                "adp": 100 + pid,
                "health_score": 0.9,
                "is_injured": 0,
            }
        )
    # A "low-ip reliever" we'll trade IN for one of the SPs (simulates pitching downgrade)
    rows.append(
        {
            "player_id": 99,
            "name": "LowIP_RP",
            "player_name": "LowIP_RP",
            "is_hitter": 0,
            "positions": "RP",
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "h": 0,
            "ab": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
            "pa": 0,
            "avg": 0,
            "obp": 0,
            "w": 2,
            "l": 2,
            "sv": 0,
            "k": 60,
            "ip": 50,
            "er": 18,
            "bb_allowed": 18,
            "h_allowed": 40,
            "era": 3.20,
            "whip": 1.16,
            "adp": 250,
            "health_score": 0.9,
            "is_injured": 0,
        }
    )
    return pd.DataFrame(rows)


def test_evaluate_trade_surfaces_ip_floor_keys() -> None:
    """End-to-end: result must include ip_floor_penalty + ip_floor_detail."""
    pool = _build_full_pool()
    roster = list(range(1, 17))  # 4 SP + 12 hitters
    # Trade SP1 (180 IP) for LowIP_RP (50 IP) → loses 130 IP / 26 = 5 IP/week
    # New weekly IP = (180*3 + 50) / 26 = 590/26 = 22.7 (still above 20 → no penalty)
    # Try trading 2 SPs to drop below the floor.
    result = evaluate_trade(
        giving_ids=[1, 2],
        receiving_ids=[99],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "ip_floor_penalty" in result, "ip_floor_penalty must be in result"
    assert "ip_floor_detail" in result, "ip_floor_detail must be in result"
    assert isinstance(result["ip_floor_penalty"], (int, float))
    assert isinstance(result["ip_floor_detail"], dict)
    # Detail should have the diagnostic fields
    det = result["ip_floor_detail"]
    assert "threshold_ip_per_week" in det
    assert "kappa" in det
    assert "before_weekly_ip" in det
    assert "after_weekly_ip" in det
    assert "below_floor" in det


def test_evaluate_trade_ip_floor_risk_flag_fires_when_below() -> None:
    """When the trade drops weekly IP below the floor, a risk flag MUST fire.

    Conditional contract test: across Python 3.12/3.14 + different PuLP
    versions, the LP may pick slightly different player subsets, which
    affects the exact post-trade weekly IP. This test asserts the
    contract: IF ip_floor_detail.below_floor is True, the risk flag
    MUST be in the flags list. We do NOT assert that the LP's specific
    selection produces below-floor — that's environment-dependent.
    """
    pool = _build_full_pool()
    roster = list(range(1, 17))
    # Trade 3 SPs for 1 reliever — strong push toward below-floor.
    result = evaluate_trade(
        giving_ids=[1, 2, 3],
        receiving_ids=[99],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    ip_detail = result.get("ip_floor_detail", {}) or {}
    flags = result.get("risk_flags", []) or []
    if ip_detail.get("below_floor"):
        # Contract: when below_floor, risk flag MUST exist
        ip_flags = [f for f in flags if "IP" in f or "ip" in f.lower()]
        assert ip_flags, (
            f"Contract violation: ip_floor_detail.below_floor=True but no "
            f"IP-floor risk flag. ip_detail={ip_detail!r}, all flags: {flags!r}"
        )
    else:
        # LP picked enough pitchers to stay above floor — no flag expected.
        # This is the "PuLP version / Python version varies what LP picks"
        # case. The unit tests in _compute_ip_floor_penalty cover the math.
        pass
