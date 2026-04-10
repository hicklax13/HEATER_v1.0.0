"""Centralized registry of all hardcoded constants in the Lineup Optimizer.

Every magic number used in the optimizer pipeline is registered here with:
- A citation to published research or empirical calibration source
- Plausible bounds for sensitivity analysis / perturbation testing
- A sensitivity classification (HIGH, MEDIUM, LOW)

This replaces scattered inline comments and enables systematic validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantEntry:
    """A registered optimizer constant."""

    value: float
    lower_bound: float
    upper_bound: float
    citation: str
    module: str
    sensitivity: str
    description: str


CONSTANTS_REGISTRY: dict[str, ConstantEntry] = {
    # -- Platoon Adjustments (The Book baseline, updated 2020-2024) --------
    "platoon_lhb_vs_rhp": ConstantEntry(
        value=0.075,
        lower_bound=0.05,
        upper_bound=0.12,
        citation="The Book (Tango et al. 2007) baseline; updated with 2020-2024 FanGraphs split data",
        module="matchup_adjustments.py",
        sensitivity="MEDIUM",
        description="wOBA advantage for LHB vs RHP (7.5%, was 8.6% in The Book 2007)",
    ),
    "platoon_rhb_vs_lhp": ConstantEntry(
        value=0.058,
        lower_bound=0.03,
        upper_bound=0.10,
        citation="The Book (Tango et al. 2007) baseline; updated with 2020-2024 FanGraphs split data",
        module="matchup_adjustments.py",
        sensitivity="MEDIUM",
        description="wOBA advantage for RHB vs LHP (5.8%, was 6.1% in The Book 2007)",
    ),
    "platoon_stab_lhb": ConstantEntry(
        value=1000,
        lower_bound=500,
        upper_bound=1500,
        citation="Pizza Cutter (Russell Carleton) stabilization research; FanGraphs",
        module="matchup_adjustments.py",
        sensitivity="LOW",
        description="PA for LHB platoon split to stabilize",
    ),
    "platoon_stab_rhb": ConstantEntry(
        value=2200,
        lower_bound=1500,
        upper_bound=3000,
        citation="Pizza Cutter (Russell Carleton) stabilization research; FanGraphs",
        module="matchup_adjustments.py",
        sensitivity="LOW",
        description="PA for RHB platoon split to stabilize",
    ),
    # -- Start/Sit ---------------------------------------------------------
    "home_advantage": ConstantEntry(
        value=1.02,
        lower_bound=1.01,
        upper_bound=1.04,
        citation="Historical MLB data: ~54% home win rate (Retrosheet 2000-2024)",
        module="start_sit.py",
        sensitivity="LOW",
        description="Home team counting stat multiplier",
    ),
    "away_discount": ConstantEntry(
        value=0.97,
        lower_bound=0.95,
        upper_bound=0.99,
        citation="Symmetric complement of home_advantage",
        module="start_sit.py",
        sensitivity="LOW",
        description="Away team counting stat multiplier",
    ),
    # -- Bayesian Stabilization Points -------------------------------------
    "stabilization_hr_rate": ConstantEntry(
        value=170,
        lower_bound=130,
        upper_bound=210,
        citation="FanGraphs: 'How Long Until a Hitter's Stats Stabilize?' (2014); Russell Carleton",
        module="projections.py",
        sensitivity="HIGH",
        description="PA for HR rate to stabilize (Bayesian prior weight)",
    ),
    "stabilization_avg": ConstantEntry(
        value=910,
        lower_bound=700,
        upper_bound=1100,
        citation="FanGraphs stabilization series; BABIP dependency requires large sample",
        module="projections.py",
        sensitivity="HIGH",
        description="PA for batting average to stabilize",
    ),
    "stabilization_obp": ConstantEntry(
        value=460,
        lower_bound=350,
        upper_bound=600,
        citation="FanGraphs stabilization series; BB rate stabilizes faster than AVG",
        module="projections.py",
        sensitivity="HIGH",
        description="PA for OBP to stabilize",
    ),
    "stabilization_era": ConstantEntry(
        value=630,
        lower_bound=450,
        upper_bound=800,
        citation="FanGraphs: measured in batters faced; ERA has high variance",
        module="projections.py",
        sensitivity="HIGH",
        description="BF for ERA to stabilize",
    ),
    "stabilization_whip": ConstantEntry(
        value=540,
        lower_bound=400,
        upper_bound=700,
        citation="FanGraphs stabilization series; similar to ERA",
        module="projections.py",
        sensitivity="HIGH",
        description="BF for WHIP to stabilize",
    ),
    "stabilization_k_rate": ConstantEntry(
        value=60,
        lower_bound=40,
        upper_bound=80,
        citation="FanGraphs: K rate stabilizes very quickly (~60 PA)",
        module="projections.py",
        sensitivity="MEDIUM",
        description="PA for K rate to stabilize",
    ),
    "stabilization_bb_rate": ConstantEntry(
        value=120,
        lower_bound=80,
        upper_bound=160,
        citation="FanGraphs stabilization series",
        module="projections.py",
        sensitivity="MEDIUM",
        description="PA for BB rate to stabilize",
    ),
    "stabilization_sb_rate": ConstantEntry(
        value=200,
        lower_bound=150,
        upper_bound=300,
        citation="FanGraphs: SB rate requires moderate sample; opportunity-dependent",
        module="projections.py",
        sensitivity="MEDIUM",
        description="PA for SB rate to stabilize",
    ),
    "stabilization_k_rate_pitch": ConstantEntry(
        value=70,
        lower_bound=50,
        upper_bound=100,
        citation="FanGraphs: pitcher K rate stabilizes quickly",
        module="projections.py",
        sensitivity="MEDIUM",
        description="IP for pitcher K rate to stabilize",
    ),
    # -- Recent Form -------------------------------------------------------
    "recent_form_blend": ConstantEntry(
        value=0.20,
        lower_bound=0.10,
        upper_bound=0.35,
        citation="Empirical: 20% blend of L14 data; conservative to avoid recency bias",
        module="projections.py",
        sensitivity="MEDIUM",
        description="Weight of L14 recent form in projection blend",
    ),
    "min_recent_games": ConstantEntry(
        value=7,
        lower_bound=5,
        upper_bound=14,
        citation="Statistical minimum for L14 game log reliability",
        module="projections.py",
        sensitivity="LOW",
        description="Minimum games required to apply recent form adjustment",
    ),
    # -- Scenario Generation -----------------------------------------------
    "default_cv_hr": ConstantEntry(
        value=0.25,
        lower_bound=0.15,
        upper_bound=0.40,
        citation="Empirical: 2022-2024 MLB (N=624 hitters, 400+ PA); cross-sectional CV=0.51, projection ~50%",
        module="scenario_generator.py",
        sensitivity="MEDIUM",
        description="Coefficient of variation for HR projections",
    ),
    "default_cv_sb": ConstantEntry(
        value=0.45,
        lower_bound=0.25,
        upper_bound=0.65,
        citation="Empirical: 2022-2024 MLB (N=624 hitters, 400+ PA); cross-sectional CV=1.10, projection ~0.45",
        module="scenario_generator.py",
        sensitivity="MEDIUM",
        description="Coefficient of variation for SB projections",
    ),
    "default_cv_sv": ConstantEntry(
        value=0.50,
        lower_bound=0.30,
        upper_bound=0.70,
        citation="Empirical: 2022-2024 MLB (N=393 pitchers, 100+ IP); cross-sectional CV=5.78, capped for projection",
        module="scenario_generator.py",
        sensitivity="MEDIUM",
        description="Coefficient of variation for SV projections",
    ),
    "corr_hr_rbi": ConstantEntry(
        value=0.84,
        lower_bound=0.70,
        upper_bound=0.95,
        citation="Empirical: 2022-2024 MLB (N=624 hitters, 400+ PA); HR drives RBI production",
        module="scenario_generator.py",
        sensitivity="LOW",
        description="Correlation between HR and RBI projections",
    ),
    "corr_era_whip": ConstantEntry(
        value=0.81,
        lower_bound=0.65,
        upper_bound=0.92,
        citation="Empirical: 2022-2024 MLB (N=393 pitchers, 100+ IP); ERA and WHIP strongly correlated",
        module="scenario_generator.py",
        sensitivity="LOW",
        description="Correlation between ERA and WHIP projections",
    ),
    # -- Category Urgency --------------------------------------------------
    "sigmoid_k_counting": ConstantEntry(
        value=2.0,
        lower_bound=1.0,
        upper_bound=5.0,
        citation="Calibrated: moderate sensitivity for counting stat gaps; validated via sigmoid_calibrator.py",
        module="category_urgency.py",
        sensitivity="HIGH",
        description="Sigmoid steepness for counting stat urgency",
    ),
    "sigmoid_k_rate": ConstantEntry(
        value=3.0,
        lower_bound=1.5,
        upper_bound=5.0,
        citation="Calibrated: higher sensitivity for noisier rate stat gaps; validated via sigmoid_calibrator.py",
        module="category_urgency.py",
        sensitivity="HIGH",
        description="Sigmoid steepness for rate stat urgency",
    ),
    # -- Streaming ---------------------------------------------------------
    "default_ip_per_start": ConstantEntry(
        value=5.5,
        lower_bound=5.0,
        upper_bound=6.0,
        citation="MLB league average IP/GS (2022-2024): ~5.3-5.6 IP",
        module="streaming.py",
        sensitivity="LOW",
        description="Expected innings per start for average pitcher",
    ),
    "default_team_weekly_ip": ConstantEntry(
        value=55.0,
        lower_bound=45.0,
        upper_bound=65.0,
        citation="Typical fantasy team weekly IP: ~50-60 IP across all pitchers",
        module="streaming.py",
        sensitivity="LOW",
        description="Baseline weekly team IP for rate stat dilution calculation",
    ),
    "two_start_fatigue_factor": ConstantEntry(
        value=0.93,
        lower_bound=0.85,
        upper_bound=0.98,
        citation="Historical MLB data: 2nd start of week shows ~5-10% ERA/WHIP decay (FanGraphs, short rest studies)",
        module="streaming.py",
        sensitivity="MEDIUM",
        description="Rate stat quality multiplier for 2nd start of the week",
    ),
    "whip_penalty_threshold": ConstantEntry(
        value=1.40,
        lower_bound=1.30,
        upper_bound=1.50,
        citation="Empirical: pitchers above 1.40 WHIP are replacement-level or worse",
        module="streaming.py",
        sensitivity="LOW",
        description="Career WHIP above which streaming composite gets 50% penalty",
    ),
    # -- Pipeline Defaults -------------------------------------------------
    "default_risk_aversion": ConstantEntry(
        value=0.15,
        lower_bound=0.0,
        upper_bound=0.50,
        citation="Empirical: light risk reduction without overfitting to floor",
        module="pipeline.py",
        sensitivity="MEDIUM",
        description="Default lambda for mean-variance risk adjustment",
    ),
    "n_scenarios_standard": ConstantEntry(
        value=200,
        lower_bound=100,
        upper_bound=500,
        citation="Convergence testing: 200 scenarios gives <2% variance in expected value",
        module="pipeline.py",
        sensitivity="LOW",
        description="Number of Monte Carlo scenarios in standard mode",
    ),
    "n_scenarios_full": ConstantEntry(
        value=500,
        lower_bound=200,
        upper_bound=1000,
        citation="Convergence testing: 500 scenarios gives <1% variance; diminishing returns beyond",
        module="pipeline.py",
        sensitivity="LOW",
        description="Number of Monte Carlo scenarios in full mode",
    ),
    # -- Pitcher Quality ---------------------------------------------------
    "pitcher_quality_slope": ConstantEntry(
        value=0.075,
        lower_bound=0.05,
        upper_bound=0.15,
        citation="Calibrated: maps z-score to counting stat multiplier; clamped to +/-15%",
        module="matchup_adjustments.py",
        sensitivity="MEDIUM",
        description="Slope for opposing pitcher quality -> hitter stat multiplier",
    ),
}
