"""Backtesting harness for draft engine accuracy evaluation.

Enables backtesting the draft engine against historical actual stats to
measure projection accuracy, rank correlation, value capture rate, and
bust rate. Supports running simulated drafts from any position with any
projection system, then comparing the drafted roster's projections against
what those players actually produced.

Key metrics:
  - Per-category RMSE (root mean squared error)
  - Spearman rank correlation between projected and actual ranks
  - Category win rate (fraction of categories where projected >= actual median)
  - Value capture rate (% of available SGP captured vs random baseline)
  - Bust rate (% of picks below replacement level)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator, value_all_players

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

# Categories tracked for accuracy evaluation
HITTING_CATEGORIES: list[str] = ["r", "hr", "rbi", "sb", "avg", "obp"]
PITCHING_CATEGORIES: list[str] = ["w", "l", "sv", "k", "era", "whip"]
ALL_EVAL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Default number of rounds in a draft
DEFAULT_NUM_ROUNDS: int = 23

# Minimum number of players required for meaningful backtesting
MIN_POOL_SIZE: int = 50


# ── Helper: Random Draft Baseline ──────────────────────────────────


def random_draft_baseline(
    pool: pd.DataFrame,
    num_picks: int = DEFAULT_NUM_ROUNDS,
    seed: int | None = None,
) -> pd.DataFrame:
    """Randomly draft players weighted by ADP for comparison.

    Selects ``num_picks`` players from the pool using ADP-weighted
    probabilities: lower ADP (drafted earlier) = higher selection weight.

    Args:
        pool: Player pool DataFrame with at least ``player_id`` and ``adp``.
        num_picks: Number of players to draft.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame subset of the pool representing the random draft.
    """
    if pool.empty or num_picks <= 0:
        return pool.iloc[:0].copy()

    rng = np.random.default_rng(seed)
    n = min(num_picks, len(pool))

    adp = pool["adp"].fillna(999).values.astype(float)
    # Weight = 1 / ADP so top ADP gets highest probability
    weights = 1.0 / np.maximum(adp, 1.0)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        probs = np.ones(len(pool)) / len(pool)
    else:
        probs = weights / weight_sum

    chosen_indices = rng.choice(len(pool), size=n, replace=False, p=probs)
    return pool.iloc[chosen_indices].copy().reset_index(drop=True)


# ── BacktestEngine ─────────────────────────────────────────────────


class BacktestEngine:
    """Engine for backtesting draft recommendations against historical actuals.

    Workflow:
      1. Load historical actual stats for a reference season.
      2. Simulate a draft using projected stats (from blended projections).
      3. Compare the projected roster against what those players actually did.
      4. Compute accuracy metrics (RMSE, rank correlation, win rate, etc.).
    """

    def __init__(self, season: int = 2025, config: LeagueConfig | None = None):
        """Initialize the backtest engine.

        Args:
            season: Historical season to use as ground truth.
            config: League configuration. Defaults to standard 12-team H2H.
        """
        self.season = season
        self.config = config or LeagueConfig()
        self.sgp_calc = SGPCalculator(self.config)
        self.actuals: pd.DataFrame | None = None
        self._results: list[dict] = []

    # ── Data Loading ───────────────────────────────────────────────

    def load_historical_actuals(self, season: int | None = None) -> pd.DataFrame:
        """Load actual stats for a given season from the DB or API.

        Tries the ``season_stats`` table first, then falls back to
        MLB Stats API via ``live_stats.fetch_season_stats()``.

        Args:
            season: Season year. Defaults to ``self.season``.

        Returns:
            DataFrame of actual stats with player_id and stat columns.
        """
        season = self.season if season is None else season

        # Try database first
        try:
            from src.database import load_season_stats

            stats = load_season_stats(season)
            if not stats.empty:
                self.actuals = stats
                return stats
        except Exception:
            logger.debug("Could not load season stats from DB for %d", season)

        # Fall back to MLB Stats API
        try:
            from src.live_stats import fetch_season_stats

            stats = fetch_season_stats(season)
            if not stats.empty:
                self.actuals = stats
                return stats
        except Exception:
            logger.debug("Could not fetch season stats from API for %d", season)

        # Return empty DataFrame with expected columns
        self.actuals = pd.DataFrame(columns=["player_id", "name"] + ALL_EVAL_CATEGORIES)
        return self.actuals

    # ── Draft Simulation ───────────────────────────────────────────

    def simulate_draft(
        self,
        player_pool: pd.DataFrame | None = None,
        projection_system: str = "blended",
        draft_position: int = 1,
        num_teams: int = 12,
        num_rounds: int = DEFAULT_NUM_ROUNDS,
    ) -> pd.DataFrame:
        """Run a full simulated draft using the engine.

        Simulates a complete snake draft where the user drafts from
        ``draft_position`` and opponents pick by ADP. The user's picks
        are selected greedily by ``pick_score``.

        Args:
            player_pool: Pre-loaded player pool. If None, loads from DB.
            projection_system: Projection system to use (for display only;
                pool should already have projections loaded).
            draft_position: 1-indexed draft position for the user.
            num_teams: Number of teams in the draft.
            num_rounds: Number of rounds.

        Returns:
            DataFrame of the user's drafted roster with projected stats.
        """
        from src.draft_state import DraftState

        if player_pool is None:
            try:
                from src.database import load_player_pool

                player_pool = load_player_pool()
            except Exception:
                logger.warning("Could not load player pool from DB")
                return pd.DataFrame()

        if player_pool.empty:
            return pd.DataFrame()

        # Ensure pick_score exists
        if "pick_score" not in player_pool.columns:
            player_pool = value_all_players(player_pool, self.config)

        # User team index is 0-based from 1-based draft_position
        user_team_index = max(0, min(draft_position - 1, num_teams - 1))

        ds = DraftState(
            num_teams=num_teams,
            num_rounds=num_rounds,
            user_team_index=user_team_index,
        )

        pool = player_pool.copy()

        # Run the draft pick by pick
        for _ in range(num_teams * num_rounds):
            if ds.current_pick >= ds.total_picks:
                break

            available = ds.available_players(pool)
            if available.empty:
                break

            team_idx = ds.picking_team_index()

            if team_idx == user_team_index:
                # User picks: greedy by pick_score
                best = available.nlargest(1, "pick_score").iloc[0]
            else:
                # Opponent picks: greedy by ADP (lowest ADP = best)
                adp_col = available["adp"].fillna(999)
                best_idx = adp_col.idxmin()
                best = available.loc[best_idx]

            ds.make_pick(
                player_id=int(best["player_id"]),
                player_name=str(best["name"]),
                positions=str(best["positions"]),
            )

        # Extract user's roster
        user_picks = [entry["player_id"] for entry in ds.pick_log if entry["team_index"] == user_team_index]

        roster = pool[pool["player_id"].isin(user_picks)].copy()
        return roster.reset_index(drop=True)

    # ── Accuracy Evaluation ────────────────────────────────────────

    def evaluate_accuracy(
        self,
        projected_roster: pd.DataFrame,
        actual_stats: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compare projected roster stats against actual stats.

        Computes per-category RMSE, Spearman rank correlation, category
        win rate, value capture rate, and bust rate.

        Args:
            projected_roster: Roster with projected stat columns.
            actual_stats: Actual season stats. If None, uses ``self.actuals``.

        Returns:
            Dict with keys: rmse, rank_correlation, category_win_rate,
            value_capture_rate, bust_rate, per_category.
        """
        if actual_stats is None:
            actual_stats = self.actuals

        if actual_stats is None or actual_stats.empty:
            return self._empty_metrics()

        if projected_roster.empty:
            return self._empty_metrics()

        # Merge projected and actual on player_id
        merged = projected_roster.merge(
            actual_stats,
            on="player_id",
            how="inner",
            suffixes=("_proj", "_actual"),
        )

        if merged.empty:
            return self._empty_metrics()

        metrics: dict[str, Any] = {"per_category": {}}

        # Per-category metrics
        rmse_values = []
        rank_corrs = []
        win_count = 0
        total_cats = 0

        for cat in ALL_EVAL_CATEGORIES:
            proj_col = f"{cat}_proj" if f"{cat}_proj" in merged.columns else cat
            actual_col = f"{cat}_actual" if f"{cat}_actual" in merged.columns else cat

            if actual_col not in merged.columns or proj_col not in merged.columns:
                continue

            proj_vals = pd.to_numeric(merged[proj_col], errors="coerce").fillna(0)
            actual_vals = pd.to_numeric(merged[actual_col], errors="coerce").fillna(0)

            # Skip if all zeros
            if proj_vals.abs().sum() == 0 and actual_vals.abs().sum() == 0:
                continue

            total_cats += 1

            # RMSE
            rmse = float(np.sqrt(np.mean((proj_vals - actual_vals) ** 2)))
            rmse_values.append(rmse)

            # Rank correlation (Spearman)
            rho = _spearman_correlation(proj_vals.values, actual_vals.values)
            rank_corrs.append(rho)

            # Win rate: projected direction correct (above/below median)
            proj_above_median = proj_vals >= proj_vals.median()
            actual_above_median = actual_vals >= actual_vals.median()
            cat_win_rate = float((proj_above_median == actual_above_median).mean())
            if cat_win_rate >= 0.5:
                win_count += 1

            metrics["per_category"][cat] = {
                "rmse": rmse,
                "rank_correlation": rho,
                "win_rate": cat_win_rate,
            }

        # Aggregate metrics
        metrics["rmse"] = float(np.mean(rmse_values)) if rmse_values else 0.0
        metrics["rank_correlation"] = float(np.mean(rank_corrs)) if rank_corrs else 0.0
        metrics["category_win_rate"] = win_count / total_cats if total_cats > 0 else 0.0

        # Value capture rate: SGP captured vs pool average
        metrics["value_capture_rate"] = self._compute_value_capture(projected_roster, pool=actual_stats)

        # Bust rate: picks below replacement level
        metrics["bust_rate"] = self._compute_bust_rate(projected_roster, actual_stats)

        return metrics

    # ── Full Backtest ──────────────────────────────────────────────

    def run_full_backtest(
        self,
        player_pool: pd.DataFrame | None = None,
        n_positions: range | list | None = None,
        systems: list[str] | None = None,
        num_teams: int = 12,
    ) -> dict[str, Any]:
        """Run the draft from multiple positions and average results.

        Args:
            player_pool: Player pool to draft from.
            n_positions: Draft positions to test (1-indexed). Defaults to 1..12.
            systems: Projection systems (for display; currently unused).
            num_teams: Number of teams.

        Returns:
            Aggregated metrics averaged across all positions.
        """
        if n_positions is None:
            n_positions = range(1, num_teams + 1)

        if self.actuals is None:
            self.load_historical_actuals()

        all_metrics: list[dict] = []

        for pos in n_positions:
            roster = self.simulate_draft(
                player_pool=player_pool,
                draft_position=pos,
                num_teams=num_teams,
            )
            metrics = self.evaluate_accuracy(roster)
            metrics["draft_position"] = pos
            all_metrics.append(metrics)

        self._results = all_metrics

        # Average across positions
        return self._average_metrics(all_metrics)

    # ── Summary ────────────────────────────────────────────────────

    def summary_report(self) -> dict[str, Any]:
        """Return accuracy metrics from the last run.

        Returns:
            Dict with keys: projection_rmse, rank_correlation,
            category_win_rate, value_capture_rate, bust_rate,
            n_positions_tested, per_position.
        """
        if not self._results:
            return {
                "projection_rmse": 0.0,
                "rank_correlation": 0.0,
                "category_win_rate": 0.0,
                "value_capture_rate": 0.0,
                "bust_rate": 0.0,
                "n_positions_tested": 0,
                "per_position": [],
            }

        avg = self._average_metrics(self._results)
        avg["n_positions_tested"] = len(self._results)
        avg["per_position"] = self._results
        return avg

    # ── Private Helpers ────────────────────────────────────────────

    def _empty_metrics(self) -> dict[str, Any]:
        """Return zeroed metrics dict when evaluation is not possible."""
        return {
            "rmse": 0.0,
            "rank_correlation": 0.0,
            "category_win_rate": 0.0,
            "value_capture_rate": 0.0,
            "bust_rate": 0.0,
            "per_category": {},
        }

    def _compute_value_capture(self, roster: pd.DataFrame, pool: pd.DataFrame | None = None) -> float:
        """Compute what fraction of available SGP the roster captured.

        Value capture = roster total SGP / (top-N SGP from pool),
        where N = roster size. Falls back to simple ratio when pool unavailable.
        """
        if pool is not None and not pool.empty:
            return self._compute_value_capture_vs_baseline(roster, pool)

        if roster.empty:
            return 0.0

        if "total_sgp" not in roster.columns:
            roster = roster.copy()
            roster["total_sgp"] = roster.apply(self.sgp_calc.total_sgp, axis=1)

        roster_sgp = roster["total_sgp"].sum()

        # Without a pool, return a normalized ratio (roster_sgp / reasonable baseline)
        # Use N * median_sgp as a rough baseline
        n = len(roster)
        median_sgp = float(roster["total_sgp"].median()) if n > 0 else 1.0
        baseline = max(n * max(median_sgp, 0.1), 1.0)
        return float(np.clip(roster_sgp / baseline, 0.0, 1.0))

    def _compute_value_capture_vs_baseline(
        self,
        roster: pd.DataFrame,
        pool: pd.DataFrame,
    ) -> float:
        """Compute value capture relative to a full pool.

        Returns the ratio of roster SGP to top-N SGP from the full pool.
        """
        if roster.empty or pool.empty:
            return 0.0

        n = len(roster)

        if "total_sgp" not in pool.columns:
            pool = pool.copy()
            pool["total_sgp"] = pool.apply(self.sgp_calc.total_sgp, axis=1)

        ideal_sgp = pool.nlargest(n, "total_sgp")["total_sgp"].sum()
        if ideal_sgp <= 0:
            return 0.0

        if "total_sgp" not in roster.columns:
            roster = roster.copy()
            roster["total_sgp"] = roster.apply(self.sgp_calc.total_sgp, axis=1)

        roster_sgp = roster["total_sgp"].sum()
        return float(np.clip(roster_sgp / ideal_sgp, 0.0, 1.0))

    def _compute_bust_rate(
        self,
        roster: pd.DataFrame,
        actual_stats: pd.DataFrame,
    ) -> float:
        """Compute the fraction of drafted players who busted.

        A bust is a player whose actual total SGP fell below replacement
        level (defined as the median SGP of the bottom quartile of the pool).
        """
        if roster.empty or actual_stats.empty:
            return 0.0

        # Merge to get actual stats for roster players
        merged = roster[["player_id"]].merge(actual_stats, on="player_id", how="inner")
        if merged.empty:
            return 0.0

        # Compute replacement level from full pool (not just roster)
        all_sgps = []
        for _, row in actual_stats.iterrows():
            sgp = self.sgp_calc.total_sgp(row)
            all_sgps.append(sgp)
        all_sgps_arr = np.array(all_sgps)
        replacement_level = float(np.percentile(all_sgps_arr, 25))

        # Compute actual SGP for each rostered player
        actual_sgps = []
        for _, row in merged.iterrows():
            sgp = self.sgp_calc.total_sgp(row)
            actual_sgps.append(sgp)

        actual_sgps_arr = np.array(actual_sgps)

        busts = int(np.sum(actual_sgps_arr < replacement_level))
        return float(busts / len(actual_sgps_arr))

    def _average_metrics(self, metrics_list: list[dict]) -> dict[str, Any]:
        """Average numeric metrics across multiple backtest runs."""
        if not metrics_list:
            return self._empty_metrics()

        keys = ["rmse", "rank_correlation", "category_win_rate", "value_capture_rate", "bust_rate"]
        result: dict[str, Any] = {}

        for key in keys:
            values = [m.get(key, 0.0) for m in metrics_list]
            # Use the summary_report key name for RMSE
            out_key = "projection_rmse" if key == "rmse" else key
            result[out_key] = float(np.mean(values)) if values else 0.0

        return result


# ── Utility Functions ──────────────────────────────────────────────


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two arrays.

    Uses scipy if available, otherwise falls back to a manual rank
    correlation implementation.

    Args:
        x: First array of values.
        y: Second array of values.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0

    # Guard against constant arrays (undefined correlation)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y)
        return float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        pass

    # Manual fallback: rank-based Pearson
    n = len(x)
    rank_x = _rankdata(x)
    rank_y = _rankdata(y)
    d = rank_x - rank_y
    d_sq_sum = float(np.sum(d**2))
    denom = n * (n**2 - 1)
    if denom == 0:
        return 0.0
    rho = 1.0 - 6.0 * d_sq_sum / denom
    return float(np.clip(rho, -1.0, 1.0))


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to data (average method for ties)."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    sorter = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)

    # Handle ties: average rank for tied values
    sorted_arr = arr[sorter]
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_arr[j] == sorted_arr[j + 1]:
            j += 1
        if j > i:
            avg_rank = np.mean(np.arange(i + 1, j + 2, dtype=float))
            for k in range(i, j + 1):
                ranks[sorter[k]] = avg_rank
        i = j + 1

    return ranks
