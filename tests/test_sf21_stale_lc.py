"""SF-21: compute_marginal_sgp must use the passed config's denominators,
not a stale module-level singleton."""

from src.engine.portfolio.category_analysis import compute_marginal_sgp
from src.valuation import LeagueConfig


def _build_totals():
    """Multi-team setup so 'better' opponents exist for both directions."""
    return {
        "my_team": {
            "R": 100.0,
            "HR": 30.0,
            "RBI": 90.0,
            "SB": 15.0,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 12.0,
            "L": 8.0,
            "SV": 5.0,
            "K": 180.0,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        "team_b": {
            "R": 110.0,
            "HR": 35.0,
            "RBI": 100.0,
            "SB": 20.0,
            "AVG": 0.280,
            "OBP": 0.350,
            "W": 14.0,
            "L": 6.0,
            "SV": 8.0,
            "K": 200.0,
            "ERA": 3.60,
            "WHIP": 1.10,
        },
        "team_c": {
            "R": 90.0,
            "HR": 25.0,
            "RBI": 80.0,
            "SB": 10.0,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10.0,
            "L": 10.0,
            "SV": 3.0,
            "K": 160.0,
            "ERA": 4.00,
            "WHIP": 1.30,
        },
    }


def test_marginal_sgp_uses_passed_config_denominators():
    """When a config with custom denominators is passed, the SGP must scale accordingly.

    SGP gap = raw_gap / sgp_denom; marginal = 1 / sgp_gap.
    Doubling sgp_denom halves sgp_gap, which doubles the marginal value.
    Expected ratio (doubled / default) ~= 2.0.
    """
    cfg_default = LeagueConfig()
    cfg_doubled = LeagueConfig()
    cfg_doubled.sgp_denominators = {k: v * 2 for k, v in cfg_default.sgp_denominators.items()}

    all_totals = _build_totals()
    your_totals = all_totals["my_team"]

    sgp_default = compute_marginal_sgp(your_totals, all_totals, your_team_name="my_team", config=cfg_default)
    sgp_doubled = compute_marginal_sgp(your_totals, all_totals, your_team_name="my_team", config=cfg_doubled)

    assert isinstance(sgp_default, dict)
    assert isinstance(sgp_doubled, dict)

    # Verify at least one category was actually exercised (not at first-place floor).
    measurable = [c for c in sgp_default if abs(sgp_default[c] - 0.01) > 1e-6 and abs(sgp_default[c]) > 0.01]
    assert measurable, "Test setup failed: no measurable marginal SGP values"

    for cat in measurable:
        ratio = sgp_doubled[cat] / sgp_default[cat]
        assert 1.8 < ratio < 2.2, (
            f"Cat {cat}: default={sgp_default[cat]}, doubled={sgp_doubled[cat]}, expected ~2.0 ratio, got {ratio:.3f}"
        )


def test_marginal_sgp_default_config_matches_module_singleton():
    """Calling without config must keep the existing pre-season default behavior."""
    all_totals = _build_totals()
    your_totals = all_totals["my_team"]

    no_config = compute_marginal_sgp(your_totals, all_totals, your_team_name="my_team")
    default_config = compute_marginal_sgp(your_totals, all_totals, your_team_name="my_team", config=LeagueConfig())

    for cat in no_config:
        assert no_config[cat] == default_config[cat], (
            f"Cat {cat}: no-config={no_config[cat]}, default-config={default_config[cat]} "
            "must match when both use defaults"
        )
