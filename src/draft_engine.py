"""Enhanced draft recommendation engine — 25-feature pipeline.

Orchestrates the full enhanced draft recommendation pipeline by chaining
existing analytics modules (bayesian, injury, statcast, h2h, park factors)
into a unified scoring system with three execution modes.

The engine enhances the base ``pick_score`` from ``value_all_players()`` with
multiplicative adjustments (park factors, injury probability, Statcast delta,
category balance) and additive bonuses (streaming penalty, closer hierarchy,
lineup protection) to produce an ``enhanced_pick_score`` column.

Wires into:
  - src/valuation.py: LeagueConfig, SGPCalculator
  - src/simulation.py: DraftSimulator.evaluate_candidates()
  - src/bayesian.py: BayesianUpdater.batch_update_projections()
  - src/injury_model.py: compute_health_score(), apply_injury_adjustment()
  - src/engine/context/injury_process.py: estimate_injury_probability()
  - src/data_bootstrap.py: PARK_FACTORS
  - src/optimizer/h2h_engine.py: compute_h2h_category_weights()
  - src/engine/signals/statcast.py: aggregate_batter_statcast()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator

if TYPE_CHECKING:
    from src.draft_state import DraftState

logger = logging.getLogger(__name__)

# ── Optional dependency flags ──────────────────────────────────────

try:
    from src.bayesian import BayesianUpdater

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from src.engine.context.injury_process import estimate_injury_probability

    INJURY_PROCESS_AVAILABLE = True
except ImportError:
    INJURY_PROCESS_AVAILABLE = False

try:
    from src.optimizer.h2h_engine import compute_h2h_category_weights

    H2H_ENGINE_AVAILABLE = True
except ImportError:
    H2H_ENGINE_AVAILABLE = False

try:
    from src.engine.signals.statcast import PYBASEBALL_AVAILABLE  # noqa: F401

    STATCAST_MODULE_AVAILABLE = True
except ImportError:
    STATCAST_MODULE_AVAILABLE = False
    PYBASEBALL_AVAILABLE = False

try:
    from src.data_bootstrap import PARK_FACTORS as BOOTSTRAP_PARK_FACTORS

    PARK_FACTORS_AVAILABLE = True
except ImportError:
    BOOTSTRAP_PARK_FACTORS = {}
    PARK_FACTORS_AVAILABLE = False

try:
    from src.injury_model import compute_health_score  # noqa: F401

    INJURY_MODEL_AVAILABLE = True
except ImportError:
    INJURY_MODEL_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────

# Counting stat columns affected by park factors
COUNTING_HITTING_STATS: list[str] = ["r", "hr", "rbi", "sb", "h"]
COUNTING_PITCHING_STATS: list[str] = ["w", "sv", "k"]

# Default health score when column is missing
DEFAULT_HEALTH_SCORE: float = 0.85

# Statcast delta clamp bounds
STATCAST_DELTA_FLOOR: float = -1.0
STATCAST_DELTA_CEILING: float = 1.0

# FIP weight for ERA correction (Tom Tango research)
FIP_WEIGHT: float = 0.6
ERA_WEIGHT: float = 0.4

# Category balance scaling by draft phase
EARLY_ROUND_CEILING: int = 8
LATE_ROUND_FLOOR: int = 17

# Enhanced pick_score multiplicative clamp
MULT_FLOOR: float = 0.5
MULT_CEILING: float = 1.5

# Minimum base SGP floor
BASE_SGP_FLOOR: float = 0.01

# ── All 12 categories in uppercase matching DraftState totals ──────
ALL_CATEGORIES_UPPER: list[str] = [
    "R",
    "HR",
    "RBI",
    "SB",
    "AVG",
    "OBP",
    "W",
    "L",
    "SV",
    "K",
    "ERA",
    "WHIP",
]
INVERSE_CATS_UPPER: set[str] = {"L", "ERA", "WHIP"}


class DraftRecommendationEngine:
    """Orchestrator for enhanced draft recommendations.

    Three modes (mirroring optimizer/pipeline.py MODE_PRESETS pattern):
      - Quick (<1s): Base SGP + category balance + park factors
      - Standard (2-3s): + Bayesian blend + injury prob + Statcast delta
      - Full (5-10s): + ML ensemble + schedule strength + all contextual
    """

    MODE_PRESETS: dict[str, dict[str, bool]] = {
        "quick": {
            "enable_bayesian": False,
            "enable_injury_prob": False,
            "enable_park_factors": True,
            "enable_category_balance": True,
            "enable_statcast": False,
            "enable_streaming": False,
            "enable_contextual": False,
            "enable_ml": False,
            "enable_spring_training": False,
        },
        "standard": {
            "enable_bayesian": True,
            "enable_injury_prob": True,
            "enable_park_factors": True,
            "enable_category_balance": True,
            "enable_statcast": True,
            "enable_streaming": True,
            "enable_contextual": True,
            "enable_ml": False,
            "enable_spring_training": True,
        },
        "full": {
            "enable_bayesian": True,
            "enable_injury_prob": True,
            "enable_park_factors": True,
            "enable_category_balance": True,
            "enable_statcast": True,
            "enable_streaming": True,
            "enable_contextual": True,
            "enable_ml": True,
            "enable_spring_training": True,
        },
    }

    def __init__(self, config: LeagueConfig, mode: str = "standard"):
        """Initialize the draft recommendation engine.

        Args:
            config: League configuration with categories and SGP denominators.
            mode: Execution mode — "quick", "standard", or "full".
        """
        if mode not in self.MODE_PRESETS:
            raise ValueError(f"Unknown mode '{mode}'. Must be one of: {list(self.MODE_PRESETS.keys())}")

        self.config = config
        self.mode = mode
        self.settings = dict(self.MODE_PRESETS[mode])
        self.sgp_calc = SGPCalculator(config)
        self._timing: dict[str, float] = {}

    @property
    def timing(self) -> dict[str, float]:
        """Return timing information from the last enhance/recommend call."""
        return dict(self._timing)

    # ── Public API ────────────────────────────────────────────────────

    def enhance_player_pool(
        self,
        player_pool: pd.DataFrame,
        draft_state: DraftState,
        park_factors: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Apply all enabled enhancement stages to the player pool.

        Chains: park factors -> Bayesian blend -> injury probability ->
        Statcast delta -> FIP correction -> contextual factors ->
        category balance -> ML correction -> compute enhanced_pick_score

        Args:
            player_pool: Full player pool DataFrame (with pick_score column).
            draft_state: Current DraftState instance.
            park_factors: Dict mapping team code to park factor.
                Falls back to PARK_FACTORS from data_bootstrap.py.

        Returns:
            Copy of player_pool with new columns:
            enhanced_pick_score, category_balance_multiplier, park_factor_adj,
            injury_probability, statcast_delta, and more.
        """
        self._timing = {}
        t_start = time.perf_counter()

        pool = player_pool.copy()

        # Ensure pick_score column exists
        if "pick_score" not in pool.columns:
            pool["pick_score"] = 0.0

        # Initialize all enhancement columns with defaults
        pool["park_factor_adj"] = 1.0
        pool["category_balance_multiplier"] = 1.0
        pool["injury_probability"] = 0.0
        pool["statcast_delta"] = 0.0
        pool["fip_era_adj"] = 0.0
        pool["platoon_factor"] = 1.0
        pool["contract_year_factor"] = 1.0
        pool["streaming_penalty"] = 0.0
        pool["lineup_protection_bonus"] = 0.0
        pool["closer_hierarchy_bonus"] = 0.0
        pool["ml_correction"] = 0.0
        pool["flex_bonus"] = 0.0
        pool["buy_fair_avoid"] = "fair"
        pool["st_signal"] = 0.0
        pool["risk_score"] = 50.0

        # Stage 1: Park factor adjustment
        if self.settings["enable_park_factors"]:
            t0 = time.perf_counter()
            pool = self._apply_park_factors(pool, park_factors)
            self._timing["park_factors"] = time.perf_counter() - t0

        # Stage 2: Bayesian blend (standard/full only)
        if self.settings["enable_bayesian"]:
            t0 = time.perf_counter()
            pool = self._apply_bayesian_blend(pool)
            self._timing["bayesian"] = time.perf_counter() - t0

        # Stage 3: Injury probability (standard/full only)
        if self.settings["enable_injury_prob"]:
            t0 = time.perf_counter()
            pool = self._apply_injury_probability(pool)
            self._timing["injury_prob"] = time.perf_counter() - t0

        # Stage 3b: Spring training K-rate signal (standard/full only)
        if self.settings.get("enable_spring_training", False):
            t0 = time.perf_counter()
            pool = self._apply_spring_training_signal(pool)
            self._timing["spring_training"] = time.perf_counter() - t0

        # Stage 4: Statcast delta (standard/full only)
        if self.settings["enable_statcast"]:
            t0 = time.perf_counter()
            pool = self._apply_statcast_delta(pool)
            self._timing["statcast"] = time.perf_counter() - t0

        # Stage 5: FIP-based ERA correction
        pool = self._apply_fip_correction(pool)

        # Stage 6: Contextual factors (standard/full)
        if self.settings["enable_contextual"]:
            t0 = time.perf_counter()
            pool = self._apply_contextual_factors(pool, draft_state)
            self._timing["contextual"] = time.perf_counter() - t0

        # Stage 7: Category balance weighting
        if self.settings["enable_category_balance"]:
            t0 = time.perf_counter()
            pool = self._apply_category_balance(pool, draft_state)
            self._timing["category_balance"] = time.perf_counter() - t0

        # Stage 8: ML correction (full only — placeholder)
        if self.settings["enable_ml"]:
            t0 = time.perf_counter()
            pool = self._apply_ml_correction(pool)
            self._timing["ml_correction"] = time.perf_counter() - t0

        # Final: compute enhanced_pick_score
        pool["enhanced_pick_score"] = pool.apply(self._compute_enhanced_pick_score, axis=1)

        # Buy/fair/avoid classification
        pool["buy_fair_avoid"] = pool.apply(self._classify_buy_fair_avoid, axis=1)

        # Composite risk score (0-100)
        t0 = time.perf_counter()
        pool = self._compute_risk_score(pool)
        self._timing["risk_score"] = time.perf_counter() - t0

        self._timing["total"] = time.perf_counter() - t_start
        return pool

    def recommend(
        self,
        player_pool: pd.DataFrame,
        draft_state: DraftState,
        top_n: int = 8,
        n_simulations: int = 300,
        park_factors: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Full recommendation pipeline: enhance pool -> MC sim -> rank.

        Replaces direct call to DraftSimulator.evaluate_candidates() with
        an enhanced version that uses enhanced_pick_score for MC simulation.

        Args:
            player_pool: Full player pool DataFrame.
            draft_state: Current DraftState instance.
            top_n: Number of top candidates to return.
            n_simulations: Monte Carlo simulations per candidate.
            park_factors: Optional park factor overrides.

        Returns:
            DataFrame with simulation results for each candidate, including
            all enhanced columns plus MC simulation metrics.
        """
        from src.simulation import DraftSimulator

        t_total_start = time.perf_counter()

        # Step 1: Enhance the pool
        enhanced_pool = self.enhance_player_pool(player_pool, draft_state, park_factors)

        # Step 2: Run MC simulation using enhanced_pick_score as the SGP input
        simulator = DraftSimulator(self.config, sigma=10.0)

        # Replace pick_score with enhanced_pick_score for the simulation
        sim_pool = enhanced_pool.copy()
        sim_pool["_original_pick_score"] = sim_pool["pick_score"]
        sim_pool["pick_score"] = sim_pool["enhanced_pick_score"]

        t_mc_start = time.perf_counter()
        results = simulator.evaluate_candidates(
            player_pool=sim_pool,
            draft_state=draft_state,
            top_n=top_n,
            n_simulations=n_simulations,
        )
        self._timing["mc_simulation"] = time.perf_counter() - t_mc_start

        # Merge enhancement columns into results
        if not results.empty:
            enhance_cols = [
                "player_id",
                "enhanced_pick_score",
                "park_factor_adj",
                "category_balance_multiplier",
                "injury_probability",
                "statcast_delta",
                "buy_fair_avoid",
                "streaming_penalty",
                "closer_hierarchy_bonus",
                "lineup_protection_bonus",
                "flex_bonus",
                "fip_era_adj",
                "platoon_factor",
                "contract_year_factor",
                "ml_correction",
                "st_signal",
                "risk_score",
            ]
            available_cols = [c for c in enhance_cols if c in enhanced_pool.columns]
            merge_df = enhanced_pool[available_cols].drop_duplicates(subset=["player_id"])
            results = results.merge(merge_df, on="player_id", how="left")

        # Step 3: Enrich output with ranks, composite value, confidence
        if not results.empty:
            results = self._enrich_output(results)

        self._timing["recommend_total"] = time.perf_counter() - t_total_start
        return results

    # ── Output Enrichment ──────────────────────────────────────────────

    def _enrich_output(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to recommendation output.

        Adds four columns to the already-sorted candidates DataFrame:

        1. ``overall_rank`` (int): Sequential rank 1, 2, 3... based on
           current sort order (combined_score descending).
        2. ``composite_value`` (float): ``combined_score`` normalized to a
           0-100 scale.  ``(score - min) / (max - min) * 100``.  When all
           scores are identical the value is 50.0.
        3. ``position_rank`` (str): Per-position rank for each of the
           player's eligible positions.  Format ``"SS:1/2B:3"``.
        4. ``confidence_level`` (str): ``"HIGH"``, ``"MEDIUM"``, or
           ``"LOW"`` derived from the coefficient of variation of MC
           simulation results.

        Args:
            candidates: DataFrame sorted by combined_score descending,
                produced by ``evaluate_candidates()`` + enhancement merge.

        Returns:
            Same DataFrame with the four new columns appended.
        """
        if candidates.empty:
            candidates["overall_rank"] = pd.Series(dtype="int64")
            candidates["composite_value"] = pd.Series(dtype="float64")
            candidates["position_rank"] = pd.Series(dtype="object")
            candidates["confidence_level"] = pd.Series(dtype="object")
            return candidates

        # 1. Overall rank — sequential starting at 1
        candidates = candidates.reset_index(drop=True)
        candidates["overall_rank"] = candidates.index + 1

        # 2. Composite value — normalize combined_score to [0, 100]
        scores = candidates["combined_score"]
        score_min = scores.min()
        score_max = scores.max()
        if score_max == score_min:
            candidates["composite_value"] = 50.0
        else:
            candidates["composite_value"] = (scores - score_min) / (score_max - score_min) * 100.0

        # 3. Position rank — rank within each eligible position
        candidates["position_rank"] = self._compute_position_ranks(candidates)

        # 4. Confidence level — based on MC coefficient of variation
        candidates["confidence_level"] = candidates.apply(self._compute_confidence_level, axis=1)

        return candidates

    @staticmethod
    def _compute_position_ranks(candidates: pd.DataFrame) -> pd.Series:
        """Rank each player within every position they are eligible for.

        Positions are parsed from the comma-separated ``positions`` column.
        Players are ranked within each position by their row order (which
        is already sorted by combined_score descending, so rank 1 = best).

        Returns:
            Series of strings like ``"SS:1/2B:3"`` or ``"SP:2"``.
        """
        # Build a mapping: position -> ordered list of row indices
        pos_counters: dict[str, int] = {}
        # For each row, store {position: rank}
        row_pos_ranks: list[dict[str, int]] = [{} for _ in range(len(candidates))]

        for idx, row in candidates.iterrows():
            positions_str = row.get("positions", "")
            if not positions_str or not isinstance(positions_str, str):
                continue
            pos_list = [p.strip() for p in positions_str.split(",") if p.strip()]
            for pos in pos_list:
                pos_counters[pos] = pos_counters.get(pos, 0) + 1
                row_pos_ranks[idx][pos] = pos_counters[pos]

        result = []
        for ranks in row_pos_ranks:
            if not ranks:
                result.append("")
            else:
                parts = [f"{pos}:{rank}" for pos, rank in sorted(ranks.items())]
                result.append("/".join(parts))

        return pd.Series(result, index=candidates.index)

    @staticmethod
    def _compute_confidence_level(row: pd.Series) -> str:
        """Derive confidence level from MC simulation variance.

        Uses the coefficient of variation (CV = std / mean):
          - CV < 0.15 → HIGH (stable projection)
          - CV < 0.35 → MEDIUM (moderate uncertainty)
          - CV >= 0.35 or mean ≈ 0 → LOW (high uncertainty)

        When ``mc_mean_sgp`` is 0, NaN, or missing → LOW.
        """
        mean_val = row.get("mc_mean_sgp", None)
        std_val = row.get("mc_std_sgp", None)

        # Guard: missing or zero mean
        if mean_val is None or std_val is None:
            return "LOW"
        try:
            mean_val = float(mean_val)
            std_val = float(std_val)
        except (ValueError, TypeError):
            return "LOW"

        if np.isnan(mean_val) or np.isnan(std_val) or abs(mean_val) < 0.01:
            return "LOW"

        cv = std_val / abs(mean_val)
        if cv < 0.15:
            return "HIGH"
        elif cv < 0.35:
            return "MEDIUM"
        return "LOW"

    # ── Enhancement Stages (private) ──────────────────────────────────

    def _apply_park_factors(
        self,
        pool: pd.DataFrame,
        park_factors: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Multiply counting stats by park factor for each player's team.

        A Coors Field hitter (park_factor=1.38) gets inflated counting stats;
        a Marlins Park pitcher (0.88) gets deflated. The ``park_factor_adj``
        column stores the raw factor for the enhanced_pick_score formula.
        """
        if park_factors is not None:
            factors = park_factors
        else:
            factors = BOOTSTRAP_PARK_FACTORS
        if not factors:
            if "park_factor_adj" not in pool.columns:
                pool["park_factor_adj"] = 1.0
            return pool

        # Normalize dict keys to uppercase for consistent lookup
        norm_factors = {k.upper(): v for k, v in factors.items()}

        def _get_factor(team: str) -> float:
            if not team or not isinstance(team, str):
                return 1.0
            return norm_factors.get(team.strip().upper(), 1.0)

        pool["park_factor_adj"] = pool["team"].apply(_get_factor)
        return pool

    def _apply_bayesian_blend(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Apply Bayesian regression to projections when season stats exist.

        Uses BayesianUpdater.batch_update_projections() to regress observed
        stats toward preseason priors. During pre-season (no observed stats),
        this is a no-op.
        """
        if not BAYESIAN_AVAILABLE:
            logger.debug("Bayesian module unavailable — skipping blend")
            return pool

        # Check if season stats exist (pa > 0 for hitters or ip > 0 for pitchers)
        has_season = False
        if "pa" in pool.columns:
            has_season = (pool["pa"].fillna(0) > 0).any()
        if not has_season and "ip" in pool.columns:
            has_season = (pool["ip"].fillna(0) > 0).any()

        if not has_season:
            logger.debug("No season stats available — Bayesian blend skipped")
            return pool

        try:
            updater = BayesianUpdater()
            # Use pool as both season stats and preseason (self-blend with regression)
            updated = updater.batch_update_projections(
                season_stats=pool,
                preseason=pool,
                config=self.config,
            )
            if updated is not None and not updated.empty:
                # Merge updated projections back — only update stat columns
                stat_cols = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
                available_stat_cols = [c for c in stat_cols if c in updated.columns and c in pool.columns]
                if available_stat_cols and "player_id" in updated.columns:
                    update_df = updated[["player_id"] + available_stat_cols]
                    pool = pool.drop(columns=available_stat_cols)
                    pool = pool.merge(update_df, on="player_id", how="left")
        except Exception as exc:
            logger.warning("Bayesian blend failed — using raw projections: %s", exc)

        return pool

    def _apply_injury_probability(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Add injury_probability column using health scores and age.

        Uses estimate_injury_probability() from the injury process module
        with a full-season horizon (183 days).
        """
        if not INJURY_PROCESS_AVAILABLE:
            logger.debug("Injury process module unavailable — skipping")
            if "injury_probability" not in pool.columns:
                pool["injury_probability"] = 0.0
            return pool

        def _estimate_prob(row: pd.Series) -> float:
            hs = float(row.get("health_score", DEFAULT_HEALTH_SCORE) or DEFAULT_HEALTH_SCORE)
            age = row.get("age", None)
            if age is not None:
                try:
                    age = int(age)
                except (ValueError, TypeError):
                    age = None
            is_pitcher = not bool(row.get("is_hitter", True))
            try:
                return estimate_injury_probability(
                    health_score=hs,
                    age=age,
                    is_pitcher=is_pitcher,
                    horizon_days=183,  # full season
                )
            except Exception:
                return 0.0

        pool["injury_probability"] = pool.apply(_estimate_prob, axis=1)
        return pool

    def _apply_statcast_delta(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Compute xwOBA-wOBA delta for Statcast-based skill correction.

        When Statcast data is available, the delta between expected and actual
        performance indicates underperformers (positive delta = buy candidate)
        and overperformers (negative delta = avoid candidate).

        For draft context, we use pre-computed columns if available, or
        compute a simplified delta from projection columns.
        """
        # Check for pre-computed xwoba column
        if "xwoba" in pool.columns and "avg" in pool.columns:
            # Approximate: xwOBA vs projected wOBA (simplified from AVG/OBP)
            # wOBA ~= OBP * 1.15 (rough approximation for delta calculation)
            woba_approx = pool.get("obp", pool.get("avg", 0.250)) * 1.15
            pool["statcast_delta"] = np.clip(
                pool["xwoba"].fillna(0) - woba_approx.fillna(0.290),
                STATCAST_DELTA_FLOOR,
                STATCAST_DELTA_CEILING,
            )
        else:
            # No Statcast data — delta stays at 0.0
            pool["statcast_delta"] = 0.0

        return pool

    def _apply_fip_correction(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Adjust pitcher ERA using FIP when available.

        Tom Tango's research-backed weighting:
            era_adjusted = 0.6 * FIP + 0.4 * ERA

        Only applies to pitchers (is_hitter == False) with valid FIP.
        Stores the adjustment magnitude in fip_era_adj column.
        """
        if "fip" not in pool.columns:
            return pool

        is_pitcher = pool.get("is_hitter", True) == False  # noqa: E712
        has_fip = pool["fip"].notna() & (pool["fip"] > 0)
        mask = is_pitcher & has_fip

        if not mask.any():
            return pool

        original_era = pool.loc[mask, "era"].fillna(4.50)
        fip_vals = pool.loc[mask, "fip"]
        adjusted_era = FIP_WEIGHT * fip_vals + ERA_WEIGHT * original_era

        pool.loc[mask, "fip_era_adj"] = original_era - adjusted_era
        pool.loc[mask, "era"] = adjusted_era

        return pool

    def _apply_contextual_factors(
        self,
        pool: pd.DataFrame,
        draft_state: DraftState,
    ) -> pd.DataFrame:
        """Apply contextual bonuses: streaming penalty, closer hierarchy,
        lineup protection, and multi-position flexibility.

        - Streaming penalty: Pitchers likely to be streamed (low IP projection)
          get a penalty since their roster spot is volatile.
        - Closer hierarchy: Elite closers (high SV projection) get a bonus
          reflecting the scarcity of saves in H2H.
        - Lineup protection: Hitters batting in strong lineups get a bonus
          for projected R/RBI uplift.
        - Flex bonus: Multi-position eligibility adds roster flexibility.
        """
        # Ensure default values for contextual columns
        if "streaming_penalty" not in pool.columns:
            pool["streaming_penalty"] = 0.0
        pool["streaming_penalty"] = pool["streaming_penalty"].fillna(0.0)

        # Streaming penalty: pitchers with low IP but rostered
        if "ip" in pool.columns:
            is_pitcher = pool.get("is_hitter", True) == False  # noqa: E712
            low_ip = pool["ip"].fillna(0) < 80
            # Pitchers with <80 IP projected are streaming candidates — slight penalty
            pool.loc[is_pitcher & low_ip, "streaming_penalty"] = -0.3

        # Closer hierarchy bonus: SV scarcity in H2H
        if "sv" in pool.columns:
            sv_vals = pool["sv"].fillna(0)
            # Elite closers (20+ saves): bonus scales with saves
            pool["closer_hierarchy_bonus"] = np.where(
                sv_vals >= 20,
                np.clip((sv_vals - 15) * 0.1, 0.0, 2.0),
                0.0,
            )

        # Lineup protection bonus: hitters in strong lineups
        # Proxy: players on teams with multiple drafted hitters
        if "team" in pool.columns:
            is_hitter = pool.get("is_hitter", True) == True  # noqa: E712
            team_hitter_counts = pool[is_hitter].groupby("team").size().to_dict()
            pool["lineup_protection_bonus"] = pool.apply(
                lambda row: min(0.3, team_hitter_counts.get(row.get("team", ""), 0) * 0.05)
                if row.get("is_hitter", True)
                else 0.0,
                axis=1,
            )

        # Multi-position flexibility bonus
        if "positions" in pool.columns:
            pool["flex_bonus"] = pool["positions"].apply(self._compute_flex_bonus)

        return pool

    @staticmethod
    def _compute_flex_bonus(positions_str: str) -> float:
        """Compute flexibility bonus for multi-position eligibility.

        Each extra position beyond the first adds +0.12 SGP (matching
        the existing VORP multi-position premium from valuation.py).
        Scarce positions (C, SS) add an extra +0.08.
        """
        if not positions_str or not isinstance(positions_str, str):
            return 0.0
        pos_list = [p.strip() for p in positions_str.split(",") if p.strip()]
        if len(pos_list) <= 1:
            return 0.0

        bonus = (len(pos_list) - 1) * 0.12
        scarce = {"C", "SS"}
        for pos in pos_list:
            if pos in scarce:
                bonus += 0.08
        return min(bonus, 0.5)  # cap at 0.5

    def _apply_category_balance(
        self,
        pool: pd.DataFrame,
        draft_state: DraftState,
    ) -> pd.DataFrame:
        """Weight players by how much they help balance the user's roster.

        During draft, there is no specific opponent. Use the MEDIAN projected
        team totals as a "virtual opponent":
        1. Get all 12 team totals from draft_state.get_all_team_roster_totals()
        2. Compute median per category
        3. For each category: gap = your_total - median. Use Normal PDF weight.
        4. Below-median categories get 1.2x boost, above-median get 0.9x
        5. Normalize so weights mean = 1.0
        6. Scale by draft progress: rounds 1-8 compress toward 1.0,
           rounds 17-23 amplify 1.5x
        """
        all_totals = draft_state.get_all_team_roster_totals(pool)
        my_totals = draft_state.get_user_roster_totals(pool)

        if not all_totals or not my_totals:
            return pool

        # Compute median totals across all teams
        median_totals = self._compute_median_totals(all_totals)

        # Compute category weights using H2H engine if available
        if H2H_ENGINE_AVAILABLE:
            # Use lowercase keys for h2h_engine
            my_lower = {k.lower(): v for k, v in my_totals.items() if k in ALL_CATEGORIES_UPPER}
            med_lower = {k.lower(): v for k, v in median_totals.items() if k in ALL_CATEGORIES_UPPER}
            try:
                cat_weights = compute_h2h_category_weights(my_lower, med_lower)
            except Exception:
                cat_weights = self._simple_category_weights(my_totals, median_totals)
        else:
            cat_weights = self._simple_category_weights(my_totals, median_totals)

        if not cat_weights:
            return pool

        # Draft progress scaling
        current_round = draft_state.current_round
        progress_scale = self._draft_progress_scale(current_round)

        # Apply category balance multiplier per player
        pool["category_balance_multiplier"] = pool.apply(
            lambda row: self._player_category_multiplier(row, cat_weights, my_totals, median_totals, progress_scale),
            axis=1,
        )

        return pool

    def _apply_ml_correction(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Apply ML-based correction (placeholder for future implementation).

        In full mode, this would run an ensemble model combining historical
        draft outcome data with current features to predict over/under-performance.
        Currently returns 0.0 correction for all players.
        """
        # ML correction is a placeholder — no model trained yet
        pool["ml_correction"] = 0.0
        return pool

    # ── Spring Training & Risk ────────────────────────────────────────

    def _apply_spring_training_signal(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Apply a small spring-training K-rate signal for pitchers.

        Per FanGraphs research, spring training K-rate has a weak but real
        predictive signal for in-season pitcher performance.

        Rules (pitchers only, ``is_hitter == 0``):
          - If spring training K-rate is 20%+ above projected K-rate: +0.02
          - If spring training K-rate is 20%+ below projected K-rate: -0.01
          - Otherwise: 0.0

        Requires at least one ST column to be present in the pool (e.g.
        ``spring_training_k_rate``, ``spring_training_k``, or
        ``spring_training_era`` as a sentinel that ST data was loaded).
        """
        # Ensure st_signal column exists with default
        if "st_signal" not in pool.columns:
            pool["st_signal"] = 0.0

        # Detect spring training data availability
        st_k_rate_col = None
        for candidate in ("spring_training_k_rate", "st_k_rate", "st_k_pct"):
            if candidate in pool.columns:
                st_k_rate_col = candidate
                break

        # If no dedicated K-rate column, try to derive from raw K + IP/BF
        if st_k_rate_col is None:
            has_st_k = "spring_training_k" in pool.columns
            has_st_bf = "spring_training_bf" in pool.columns or "spring_training_ip" in pool.columns
            if has_st_k and has_st_bf:
                bf_col = "spring_training_bf" if "spring_training_bf" in pool.columns else None
                if bf_col:
                    bf = pool[bf_col].fillna(0).replace(0, np.nan)
                    pool["_st_k_rate_derived"] = pool["spring_training_k"].fillna(0) / bf
                    st_k_rate_col = "_st_k_rate_derived"
                elif "spring_training_ip" in pool.columns:
                    # Approximate BF from IP: ~4.3 BF per IP
                    ip = pool["spring_training_ip"].fillna(0).replace(0, np.nan)
                    approx_bf = ip * 4.3
                    pool["_st_k_rate_derived"] = pool["spring_training_k"].fillna(0) / approx_bf
                    st_k_rate_col = "_st_k_rate_derived"

        # Also check for generic sentinel columns indicating ST data exists
        has_st_sentinel = any(
            c in pool.columns for c in ("spring_training_era", "spring_training_k_rate", "st_k_rate", "st_k_pct")
        )

        if st_k_rate_col is None and not has_st_sentinel:
            # No spring training data at all — leave st_signal at 0.0
            return pool

        if st_k_rate_col is None:
            # ST data sentinel exists but no K-rate derivable — no signal
            return pool

        is_pitcher = pool.get("is_hitter", True) == False  # noqa: E712

        # Compute projected K-rate from projected K and IP (approximate BF)
        proj_ip = pool["ip"].fillna(0).replace(0, np.nan)
        proj_bf = proj_ip * 4.3  # approximate batters faced from IP
        proj_k_rate = pool["k"].fillna(0) / proj_bf

        st_k_rate = pool[st_k_rate_col].fillna(0)

        def _signal(row_idx: int) -> float:
            if not is_pitcher.iloc[row_idx]:
                return 0.0
            st_kr = st_k_rate.iloc[row_idx]
            proj_kr = proj_k_rate.iloc[row_idx]
            if np.isnan(st_kr) or np.isnan(proj_kr) or proj_kr <= 0 or st_kr <= 0:
                return 0.0
            ratio = st_kr / proj_kr
            if ratio >= 1.20:
                return 0.02  # positive signal
            elif ratio <= 0.80:
                return -0.01  # weak negative signal
            return 0.0

        pool["st_signal"] = [_signal(i) for i in range(len(pool))]

        # Clean up temporary column
        if "_st_k_rate_derived" in pool.columns:
            pool = pool.drop(columns=["_st_k_rate_derived"])

        return pool

    def _compute_risk_score(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Compute a composite 0-100 risk score per player.

        Higher score = higher risk.  Components (weighted):
          - Health risk (40%): based on ``injury_probability``
          - Projection uncertainty (30%): based on MC std/mean ratio
          - Age risk (15%): peaks for very young (<24) and old (35+)
          - Role instability (15%): platoon/committee roles are riskier

        Formula::

            risk_score = 100 - (
                0.40 * (1 - injury_probability) * 100 +
                0.30 * projection_confidence * 100 +
                0.15 * age_factor * 100 +
                0.15 * role_stability * 100
            )

        Clamped to [0, 100].
        """

        def _age_factor(age: float | None) -> float:
            """Return age stability factor in [0, 1].

            1.0 for prime age (24-30), lower for young/old.
            """
            if age is None or np.isnan(age):
                return 0.7  # unknown age — moderate risk
            age = int(age)
            if 24 <= age <= 30:
                return 1.0
            elif 31 <= age <= 34:
                return 0.8
            elif age >= 35:
                return 0.5
            else:
                # < 24
                return 0.7

        def _role_stability(row: pd.Series) -> float:
            """Return role stability in [0, 1].

            Uses depth_chart_role if available, else infers from position.
            """
            dcr = row.get("depth_chart_role", None)
            if dcr and isinstance(dcr, str):
                dcr_lower = dcr.strip().lower()
                if dcr_lower in ("starter", "closer", "primary"):
                    return 1.0
                elif dcr_lower in ("platoon", "setup", "backup"):
                    return 0.7
                elif dcr_lower in ("committee", "spot", "long relief"):
                    return 0.5
                # Unknown role string — fall through to position-based
            # Fallback: infer from positions
            positions = row.get("positions", "")
            if not positions or not isinstance(positions, str):
                return 0.7  # unknown
            if "SP" in positions or "C" in positions:
                return 1.0  # defined starter role
            if "RP" in positions:
                # Closer (high SV) is stable; setup/committee is not
                sv = float(row.get("sv", 0) or 0)
                if sv >= 15:
                    return 1.0
                elif sv >= 5:
                    return 0.7
                return 0.5
            # Hitter — check for multi-position (may indicate super-utility / platoon risk)
            pos_list = [p.strip() for p in positions.split(",") if p.strip()]
            if len(pos_list) >= 3:
                return 0.8  # super-util — slight role uncertainty
            return 1.0  # standard starter

        def _projection_confidence(row: pd.Series) -> float:
            """Return projection confidence in [0, 1].

            Uses mc_std_sgp / mc_mean_sgp coefficient of variation when
            available; falls back to a moderate default.
            """
            mean_val = row.get("mc_mean_sgp", None)
            std_val = row.get("mc_std_sgp", None)
            if mean_val is not None and std_val is not None:
                try:
                    mean_f = float(mean_val)
                    std_f = float(std_val)
                    if abs(mean_f) > 0.01 and not np.isnan(mean_f) and not np.isnan(std_f):
                        cv = std_f / abs(mean_f)
                        return float(np.clip(1.0 - cv, 0.0, 1.0))
                except (ValueError, TypeError):
                    pass
            return 0.7  # moderate default when no MC data

        def _row_risk(row: pd.Series) -> float:
            inj_prob = float(row.get("injury_probability", 0.0) or 0.0)
            age = row.get("age", None)
            if age is not None:
                try:
                    age = float(age)
                except (ValueError, TypeError):
                    age = None

            health_component = (1.0 - inj_prob) * 100.0
            proj_component = _projection_confidence(row) * 100.0
            age_component = _age_factor(age) * 100.0
            role_component = _role_stability(row) * 100.0

            score = 100.0 - (
                0.40 * health_component + 0.30 * proj_component + 0.15 * age_component + 0.15 * role_component
            )
            return float(np.clip(score, 0.0, 100.0))

        pool["risk_score"] = pool.apply(_row_risk, axis=1)
        return pool

    # ── Score Computation ─────────────────────────────────────────────

    def _compute_enhanced_pick_score(self, row: pd.Series) -> float:
        """Compute the enhanced pick score from base SGP + adjustments.

        Formula:
            enhanced = base_sgp * mult + additive
        where:
            mult = product of multiplicative factors (clamped to [0.5, 1.5])
            additive = sum of additive bonuses
        """
        base_sgp = float(row.get("pick_score", row.get("total_sgp", 0)) or 0)
        if base_sgp <= 0:
            base_sgp = BASE_SGP_FLOOR

        # Multiplicative adjustments
        mult = 1.0
        mult *= float(row.get("category_balance_multiplier", 1.0) or 1.0)
        mult *= float(row.get("park_factor_adj", 1.0) or 1.0)
        mult *= 1 - float(row.get("injury_probability", 0) or 0) * 0.3
        mult *= 1 + float(row.get("statcast_delta", 0) or 0) * 0.15
        mult *= float(row.get("platoon_factor", 1.0) or 1.0)
        mult *= float(row.get("contract_year_factor", 1.0) or 1.0)
        mult *= 1 + float(row.get("st_signal", 0) or 0)
        mult = np.clip(mult, MULT_FLOOR, MULT_CEILING)

        # Additive bonuses
        additive = 0.0
        additive += float(row.get("streaming_penalty", 0.0) or 0.0)
        additive += float(row.get("lineup_protection_bonus", 0.0) or 0.0)
        additive += float(row.get("closer_hierarchy_bonus", 0.0) or 0.0)
        additive += float(row.get("ml_correction", 0.0) or 0.0) * 0.1
        additive += float(row.get("flex_bonus", 0.0) or 0.0)

        return base_sgp * float(mult) + additive

    @staticmethod
    def _classify_buy_fair_avoid(row: pd.Series) -> str:
        """Classify player as buy/fair/avoid based on enhancement signals.

        - "buy": Statcast delta positive AND injury prob low (undervalued skill)
        - "avoid": Statcast delta negative OR high injury prob (overvalued/risky)
        - "fair": Neither signal is strong
        """
        delta = float(row.get("statcast_delta", 0) or 0)
        injury_prob = float(row.get("injury_probability", 0) or 0)

        if delta > 0.02 and injury_prob < 0.15:
            return "buy"
        elif delta < -0.02 or injury_prob > 0.30:
            return "avoid"
        return "fair"

    # ── Helper Methods ────────────────────────────────────────────────

    @staticmethod
    def _compute_median_totals(all_team_totals: list[dict]) -> dict[str, float]:
        """Compute median across all team totals for each category."""
        if not all_team_totals:
            return {}

        median_totals: dict[str, float] = {}
        for cat in ALL_CATEGORIES_UPPER:
            values = [t.get(cat, 0) for t in all_team_totals]
            median_totals[cat] = float(np.median(values))

        return median_totals

    @staticmethod
    def _simple_category_weights(
        my_totals: dict[str, float],
        median_totals: dict[str, float],
    ) -> dict[str, float]:
        """Compute simple category weights when H2H engine is unavailable.

        Categories where user is below median get a 1.2x boost;
        above median get 0.9x. Normalized to mean = 1.0.
        """
        weights: dict[str, float] = {}
        for cat in ALL_CATEGORIES_UPPER:
            my_val = my_totals.get(cat, 0)
            med_val = median_totals.get(cat, 0)

            if cat in INVERSE_CATS_UPPER:
                # Lower is better — flip comparison
                if med_val == 0 and my_val == 0:
                    weights[cat] = 1.0
                elif my_val > med_val:
                    # User is worse (higher ERA/WHIP) — boost
                    weights[cat] = 1.2
                else:
                    weights[cat] = 0.9
            else:
                if my_val < med_val:
                    weights[cat] = 1.2
                else:
                    weights[cat] = 0.9

        # Normalize to mean = 1.0
        if weights:
            mean_w = np.mean(list(weights.values()))
            if mean_w > 0:
                weights = {k: v / mean_w for k, v in weights.items()}

        return weights

    @staticmethod
    def _draft_progress_scale(current_round: int) -> float:
        """Compute draft progress multiplier for category balance.

        Rounds 1-8: compress toward 1.0 (BPA strategy dominates early)
        Rounds 9-16: moderate scaling (1.0-1.25)
        Rounds 17+: amplify to 1.5 (category needs dominate late)
        """
        if current_round <= EARLY_ROUND_CEILING:
            # Early rounds: minimal category balance influence
            return 0.5 + 0.5 * (current_round / EARLY_ROUND_CEILING)
        elif current_round >= LATE_ROUND_FLOOR:
            # Late rounds: maximum category balance influence
            return 1.5
        else:
            # Middle rounds: linear ramp
            frac = (current_round - EARLY_ROUND_CEILING) / (LATE_ROUND_FLOOR - EARLY_ROUND_CEILING)
            return 1.0 + 0.5 * frac

    def _player_category_multiplier(
        self,
        row: pd.Series,
        cat_weights: dict[str, float],
        my_totals: dict[str, float],
        median_totals: dict[str, float],
        progress_scale: float,
    ) -> float:
        """Compute per-player category balance multiplier.

        A player who helps weak categories gets a boost; a player who piles
        onto already-strong categories gets a discount. The magnitude scales
        with draft progress.
        """
        stat_map = self.config.STAT_MAP
        is_hitter = bool(row.get("is_hitter", True))
        relevant_cats = self.config.hitting_categories if is_hitter else self.config.pitching_categories

        weighted_sum = 0.0
        weight_total = 0.0

        for cat in relevant_cats:
            col = stat_map.get(cat, cat.lower())
            player_stat = float(row.get(col, 0) or 0)
            if player_stat == 0:
                continue

            # Use lowercase key if cat_weights are lowercase, else uppercase
            w = cat_weights.get(cat.lower(), cat_weights.get(cat, 1.0))
            weighted_sum += w * abs(player_stat)
            weight_total += abs(player_stat)

        if weight_total == 0:
            raw_mult = 1.0
        else:
            raw_mult = weighted_sum / weight_total

        # Scale by draft progress and clamp
        scaled = 1.0 + (raw_mult - 1.0) * progress_scale
        return float(np.clip(scaled, 0.8, 1.2))
