"""MS-E1: unify the divergent per-category weekly-variance tables.

Three independent per-category weekly standard-deviation tables used to drive
H2H / season-projection / playoff win-probabilities and disagreed 2-5x:

    standings_engine.CALIBRATED_WEEKLY_TAU   (R=1.6, K=1.2  — implausibly tight)
    standings_projection.WEEKLY_TAU          (R=6.0, K=8.0)
    playoff_sim._DEFAULT_WEEKLY_SIGMAS       (R=8.5, K=12.0)

The same matchup yielded very different win-probs across the Season Projections,
Playoff Odds, and standings-engine surfaces. MS-E1 promotes the most
empirically-grounded existing set (``h2h_engine.default_category_variances``,
cited to FanGraphs community research + arXiv:2307.02188) to a single canonical
weekly-SD accessor, ``h2h_engine.default_weekly_sigmas()``, and points all three
modules at it.

These tests guard:
  (a) all three modules resolve the SAME per-category weekly SD from the
      canonical source;
  (b) the unified SDs yield win-probs in a sane band for a realistic edge
      (a ~1-sigma edge -> ~0.70-0.80, NOT 0.99 or 0.50).
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from src.optimizer.h2h_engine import default_category_variances, default_weekly_sigmas
from src.valuation import LeagueConfig

_CATS = LeagueConfig().all_categories


# ── (canonical accessor) ─────────────────────────────────────────────


class TestCanonicalAccessor:
    def test_default_weekly_sigmas_has_all_12_categories(self) -> None:
        sds = default_weekly_sigmas()
        for cat in _CATS:
            assert cat in sds, f"canonical weekly SD missing category: {cat}"
        assert len(sds) == 12

    def test_default_weekly_sigmas_all_positive(self) -> None:
        for cat, sd in default_weekly_sigmas().items():
            assert sd > 0, f"{cat} weekly SD non-positive: {sd}"

    def test_sigmas_are_sqrt_of_variances(self) -> None:
        """The SD accessor is the sqrt of the variance accessor (same source)."""
        variances = default_category_variances()
        sds = default_weekly_sigmas()
        stat_map = LeagueConfig().STAT_MAP
        for cat in _CATS:
            key = stat_map.get(cat, cat.lower())
            assert sds[cat] == pytest.approx(math.sqrt(variances[key])), (
                f"{cat}: SD {sds[cat]} != sqrt(var {variances[key]})"
            )

    def test_counting_sds_are_empirically_grounded_not_tight(self) -> None:
        """The implausibly-tight standings_engine values (R=1.6, K=1.2) must be
        gone — a realistic weekly R/K total swings by ~12-15, not ~1-2."""
        sds = default_weekly_sigmas()
        assert sds["R"] > 8.0, f"R weekly SD {sds['R']} is implausibly tight"
        assert sds["K"] > 8.0, f"K weekly SD {sds['K']} is implausibly tight"
        assert sds["RBI"] > 8.0, f"RBI weekly SD {sds['RBI']} is implausibly tight"


# ── (a) all three modules resolve the SAME per-category weekly SD ─────


class TestThreeModulesAgree:
    def test_standings_engine_resolves_canonical(self) -> None:
        from src.standings_engine import CALIBRATED_WEEKLY_TAU

        canonical = default_weekly_sigmas()
        for cat in _CATS:
            assert CALIBRATED_WEEKLY_TAU[cat] == pytest.approx(canonical[cat]), (
                f"standings_engine {cat} SD diverges from canonical"
            )

    def test_standings_projection_resolves_canonical(self) -> None:
        from src.standings_projection import WEEKLY_TAU

        canonical = default_weekly_sigmas()
        for cat in _CATS:
            assert WEEKLY_TAU[cat] == pytest.approx(canonical[cat]), (
                f"standings_projection {cat} SD diverges from canonical"
            )

    def test_playoff_sim_resolves_canonical(self) -> None:
        from src.playoff_sim import _build_weekly_sigmas

        sigmas = _build_weekly_sigmas()
        canonical = default_weekly_sigmas()
        for cat in _CATS:
            assert sigmas[cat] == pytest.approx(canonical[cat]), f"playoff_sim {cat} SD diverges from canonical"

    def test_all_three_modules_pairwise_identical(self) -> None:
        from src.playoff_sim import _build_weekly_sigmas
        from src.standings_engine import CALIBRATED_WEEKLY_TAU
        from src.standings_projection import WEEKLY_TAU

        se = CALIBRATED_WEEKLY_TAU
        sp = WEEKLY_TAU
        ps = _build_weekly_sigmas()
        for cat in _CATS:
            assert se[cat] == pytest.approx(sp[cat]) == pytest.approx(ps[cat]), (
                f"{cat}: standings_engine={se[cat]} standings_projection={sp[cat]} playoff_sim={ps[cat]} disagree"
            )


# ── (b) calibration sanity ───────────────────────────────────────────


class TestCalibrationSanity:
    """A realistic per-category edge must produce a win-prob in a sane band —
    neither compressed toward 0.5 nor saturated near 0.99.

    The consumers all combine two independent team draws, so the difference SD
    is sqrt(2)*single_sd. A 1-single-sigma edge therefore gives
    Phi(1/sqrt(2)) = Phi(0.707) ~ 0.760.
    """

    def test_one_sigma_edge_lands_in_sane_band(self) -> None:
        sds = default_weekly_sigmas()
        for cat in _CATS:
            sd = sds[cat]
            gap = sd  # a one-single-team-sigma edge
            sigma_diff = math.sqrt(2.0) * sd
            p = float(norm.cdf(gap / sigma_diff))
            assert 0.70 <= p <= 0.80, (
                f"{cat}: a 1-sigma edge gives p={p:.3f} (should be ~0.76, "
                f"not compressed toward 0.5 or saturated near 0.99)"
            )

    def test_tiny_edge_not_saturated(self) -> None:
        """A small (0.2-sigma) edge must NOT be near-certain — the old tight
        standings_engine taus saturated even small edges to ~0.99."""
        sds = default_weekly_sigmas()
        for cat in _CATS:
            sd = sds[cat]
            gap = 0.2 * sd
            sigma_diff = math.sqrt(2.0) * sd
            p = float(norm.cdf(gap / sigma_diff))
            assert p < 0.62, f"{cat}: a 0.2-sigma edge gives p={p:.3f} (saturated)"

    def test_no_edge_is_coinflip(self) -> None:
        sds = default_weekly_sigmas()
        for cat in _CATS:
            sd = sds[cat]
            sigma_diff = math.sqrt(2.0) * sd
            p = float(norm.cdf(0.0 / sigma_diff))
            assert p == pytest.approx(0.5), f"{cat}: equal means should be 0.5, got {p}"
