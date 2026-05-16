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
        value=0.25,
        lower_bound=0.10,
        upper_bound=0.35,
        citation="Wave 11B DCV-A1-004: reconciled to 0.25 to match daily_optimizer's dynamic cap and shared_data_layer._RECENT_FORM_WEIGHT_TODAY; was 0.20 (drift)",
        module="projections.py",
        sensitivity="MEDIUM",
        description="Weight of L14 recent form in projection blend (max blend weight reached at 14+ games)",
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
        value=54.0,
        lower_bound=45.0,
        upper_bound=65.0,
        citation="Wave 11B DCV-A2-001/A2-003: reconciled to 54.0 to match streaming._STREAM_IP_TARGET + CLAUDE.md gotcha; was 55.0 (drift)",
        module="streaming.py + daily_optimizer.apply_ip_pace_scaling",
        sensitivity="LOW",
        description="Baseline weekly team IP target; canonical value, consumed by streaming.py and apply_ip_pace_scaling",
    ),
    "streaming_baseline_whip": ConstantEntry(
        value=1.30,
        lower_bound=1.10,
        upper_bound=1.40,
        citation="Wave 11B DCV-A2-002: reconciled to 1.30 to match Wave 8d _LEAGUE_AVG_WHIP in streaming.py + war_room_hotcold.py; was 1.25 (drift)",
        module="streaming.py",
        sensitivity="LOW",
        description="Baseline WHIP for streaming pitcher rate-stat impact calculation",
    ),
    "league_avg_xfip": ConstantEntry(
        value=4.20,
        lower_bound=3.90,
        upper_bound=4.50,
        citation="Wave 11B DCV-A1-009: MLB league-avg xFIP varies yearly (4.05 in 2023, 4.15 in 2024). Annual update procedure: pull from team_strength aggregation or FanGraphs leaderboard.",
        module="daily_optimizer.compute_matchup_multiplier",
        sensitivity="MEDIUM",
        description="League-average xFIP for hitter-vs-pitcher matchup multiplier baseline",
    ),
    "r_stabilization_pa": ConstantEntry(
        value=250.0,
        lower_bound=200.0,
        upper_bound=350.0,
        citation="Wave 11B DCV-A1-007 fix: FanGraphs runs stabilization research; previously 460 (copy-paste of OBP value).",
        module="daily_optimizer.STABILIZATION_POINTS",
        sensitivity="LOW",
        description="PA threshold for runs (R) rate to stabilize in Bayesian blend",
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
    # -- Replacement Levels (rate-stat marginal SGP) -----------------------
    # Used by daily_optimizer's rate-stat SGP computation:
    # annual_sgp = (component - opportunity * replacement) / raw_denom.
    # Per DCV-A1-001 audit: original baselines (0.240/0.305/4.50/1.35) were
    # calibrated by inspection without explicit research citation. Bounds
    # below are conservative; OQ-1 in the audit findings asks for the
    # specific source / FourzynBurn-validated values.
    "repl_avg": ConstantEntry(
        value=0.240,
        lower_bound=0.225,
        upper_bound=0.255,
        citation="12-team H2H mixed league baseline; DCV-A1-001 OQ-1 pending validation",
        module="daily_optimizer.py",
        sensitivity="HIGH",
        description="Replacement-level AVG for hitter rate-stat marginal SGP",
    ),
    "repl_obp": ConstantEntry(
        value=0.305,
        lower_bound=0.290,
        upper_bound=0.320,
        citation="12-team H2H mixed league baseline; DCV-A1-001 OQ-1 pending validation",
        module="daily_optimizer.py",
        sensitivity="HIGH",
        description="Replacement-level OBP for hitter rate-stat marginal SGP",
    ),
    "repl_era": ConstantEntry(
        value=4.50,
        lower_bound=4.20,
        upper_bound=4.80,
        citation="12-team H2H mixed league baseline; DCV-A1-001 OQ-1 pending validation",
        module="daily_optimizer.py",
        sensitivity="HIGH",
        description="Replacement-level ERA for pitcher rate-stat marginal SGP",
    ),
    "repl_whip": ConstantEntry(
        value=1.35,
        lower_bound=1.27,
        upper_bound=1.42,
        citation="12-team H2H mixed league baseline; DCV-A1-001 OQ-1 pending validation",
        module="daily_optimizer.py",
        sensitivity="HIGH",
        description="Replacement-level WHIP for pitcher rate-stat marginal SGP",
    ),
    # -- Raw-Unit SGP Denominators (team-volume × per-stand-point denom) --
    # Used alongside replacement levels in rate-stat marginal SGP:
    # annual_sgp = (component - opportunity * replacement) / raw_denom.
    # Derived from team-volume assumptions (5500 AB, 6100 PA, 1400 IP)
    # multiplied by league-config sgp_denominators. See DCV-A1-005 (MED)
    # for the audit finding that these should be derived from actual league
    # data rather than hardcoded.
    "raw_sgp_denom_avg": ConstantEntry(
        value=22.0,
        lower_bound=18.0,
        upper_bound=26.0,
        citation="0.004 AVG/SP × ~5500 team AB; see DCV-A1-005",
        module="daily_optimizer.py",
        sensitivity="MEDIUM",
        description="Raw hits per standings point for AVG (team-volume × per-SP denom)",
    ),
    "raw_sgp_denom_obp": ConstantEntry(
        value=30.0,
        lower_bound=24.0,
        upper_bound=36.0,
        citation="0.005 OBP/SP × ~6100 team PA; see DCV-A1-005",
        module="daily_optimizer.py",
        sensitivity="MEDIUM",
        description="Raw on-base events per standings point for OBP",
    ),
    "raw_sgp_denom_era": ConstantEntry(
        value=31.0,
        lower_bound=25.0,
        upper_bound=37.0,
        citation="0.20 ERA/SP × ~1400 team IP / 9; see DCV-A1-005",
        module="daily_optimizer.py",
        sensitivity="MEDIUM",
        description="Raw earned runs per standings point for ERA",
    ),
    "raw_sgp_denom_whip": ConstantEntry(
        value=28.0,
        lower_bound=23.0,
        upper_bound=33.0,
        citation="0.020 WHIP/SP × ~1400 team IP; see DCV-A1-005",
        module="daily_optimizer.py",
        sensitivity="MEDIUM",
        description="Raw walks+hits per standings point for WHIP",
    ),
}
