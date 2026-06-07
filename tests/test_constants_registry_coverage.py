"""PR9: ensure key fa_recommender / waiver_wire constants are registered
in constants_registry with proper metadata (citation, bounds, sensitivity).

Bounds adapt the prompt's `bounds` / `sensitivity_tag` keys to the
project's actual `ConstantEntry` shape (lower_bound, upper_bound,
sensitivity) so this guard composes with the existing
`test_constants_registry.py` invariants (strict lower_bound < value <
upper_bound + value-within-bounds).
"""

from __future__ import annotations

import pytest

from src.optimizer.constants_registry import CONSTANTS_REGISTRY

# List the constant keys we want registered after PR9. Keep this list
# narrow — only the truly impactful ones (calibration knobs + thresholds
# that materially change FA recommendations).
EXPECTED_KEYS = [
    "stream_win_prob_min",  # 0.2755 — Razzball / Pitcher List streaming methodology
    "floor_pa_min",  # 50 — PA below which counting-stat floor penalty applies
    "floor_ip_min",  # 20 — IP equivalent for pitchers
    "ownership_boost_delta",  # 5.0 — Δ% ownership trend trigger
    "ownership_boost_mult",  # 1.10 — multiplier when ownership trending up
    "floor_penalty_mult",  # 0.85 — multiplier when below floor
    "pt_gate_grace_days",  # 30 — playing-time gate grace period
    "pt_ratio_full_credit",  # 0.60
    "pt_ratio_mild",  # 0.30
    "pt_hitter_gp_per_day",  # 0.85
    "pt_pitcher_ip_per_day",  # 1.0
    "pt_mult_zero_volume",  # 0.30
    "pt_mult_low",  # 0.60
    "pt_mult_mild",  # 0.85
    "category_worsen_threshold",  # -0.1
    "cross_type_sgp_min",  # 0.3
    "stream_net_sgp_min",  # 0.70
    "stream_net_sgp_relaxed",  # 0.40
    "stream_hurt_threshold",  # -0.10
    "stream_ip_target",  # 54
    "stream_ip_min",  # 20
    "min_active_hitters",  # 10
    "min_active_pitchers",  # 8
    "league_avg_whip",  # 1.30 — already registered pre-PR9; re-asserted here
    # FA-E4 (2026-06-07): last inline FA tunables centralized.
    "ecr_stddev_polarizing_threshold",  # 20 — ECR rank stddev above which a pick is "polarizing"
    "ecr_stddev_consensus_threshold",  # 5 — ECR rank stddev below which a pick is "consensus"
    "ecr_polarizing_mult",  # 0.95 — discount for polarizing picks
    "ecr_consensus_mult",  # 1.02 — premium for consensus picks
    "regression_buy_low_mult",  # 1.05 — BUY_LOW regression-flag nudge
    "regression_sell_high_mult",  # 0.95 — SELL_HIGH regression-flag nudge
]


def test_registry_has_pr9_constants():
    """Each registered constant must have value, citation, bounds, sensitivity."""
    missing = []
    bad_metadata = []
    for key in EXPECTED_KEYS:
        if key not in CONSTANTS_REGISTRY:
            missing.append(key)
            continue
        entry = CONSTANTS_REGISTRY[key]
        # Check metadata completeness (using the project's actual ConstantEntry
        # field names, not the prompt's nominal `bounds` / `sensitivity_tag`).
        for attr in (
            "value",
            "citation",
            "lower_bound",
            "upper_bound",
            "sensitivity",
        ):
            if not hasattr(entry, attr) or getattr(entry, attr) is None:
                bad_metadata.append(f"{key}.{attr}")
        if entry.citation == "":
            bad_metadata.append(f"{key}.citation (empty)")

    assert not missing, f"PR9: missing registry entries for: {missing}"
    assert not bad_metadata, f"PR9: registry entries with bad metadata: {bad_metadata}"


def test_pt_gate_grace_days_value_preserved():
    """Migrating to registry must not change the constant's value."""
    assert CONSTANTS_REGISTRY["pt_gate_grace_days"].value == 30


def test_stream_win_prob_min_value_preserved():
    """0.2755 is from Razzball / Pitcher List streaming methodology."""
    assert CONSTANTS_REGISTRY["stream_win_prob_min"].value == pytest.approx(0.2755)


def test_fa_module_reads_registry_for_pt_gate():
    """fa_recommender must read pt_gate_grace_days from CONSTANTS_REGISTRY
    (so calibration takes effect without code edits)."""
    from src.optimizer import fa_recommender

    assert fa_recommender._PT_GATE_GRACE_DAYS == (CONSTANTS_REGISTRY["pt_gate_grace_days"].value)


def test_fa_module_reads_registry_for_stream_win_prob():
    """fa_recommender must read stream_win_prob_min from CONSTANTS_REGISTRY."""
    from src.optimizer import fa_recommender

    assert fa_recommender._STREAM_WIN_PROB_MIN == (CONSTANTS_REGISTRY["stream_win_prob_min"].value)


def test_waiver_wire_reads_registry_for_league_avg_whip():
    """waiver_wire's _LEAGUE_AVG_WHIP must read from the registry."""
    from src import waiver_wire

    assert waiver_wire._LEAGUE_AVG_WHIP == (CONSTANTS_REGISTRY["league_avg_whip"].value)
