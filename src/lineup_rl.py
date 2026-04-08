"""D7: Category-Aware Lineup Reinforcement Learning.

Contextual bandit that learns from weekly lineup decisions and outcomes.
After 8+ weeks of data, provides category-specific lineup recommendations
that adapt to the user's league and opponent tendencies.

This is experimental and requires accumulated decision/outcome history.
Falls back gracefully to standard optimizer weights when insufficient data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

logger = logging.getLogger(__name__)

# Minimum weeks of data needed for the RL model to be useful
MIN_WEEKS_FOR_RL = 8

# Learning rate for weight updates (Thompson sampling)
ALPHA = 0.1

# Exploration parameter (higher = more exploration)
EXPLORATION_BONUS = 0.5


class LineupContextualBandit:
    """Contextual bandit for category-aware lineup optimization.

    State (context):
        - Current matchup W/L/T state per category
        - Opponent strength profile
        - Days remaining in matchup week
        - Current roster health status

    Actions:
        - Category weight adjustments (12 categories)

    Reward:
        - Categories won at end of week (0 or 1 per category)

    Learning:
        - Thompson sampling with Beta(alpha, beta) per category-context pair
        - Updates posterior after each week's results
    """

    def __init__(self, categories: list[str] | None = None):
        self.categories = categories or [
            "r",
            "hr",
            "rbi",
            "sb",
            "avg",
            "obp",
            "w",
            "l",
            "sv",
            "k",
            "era",
            "whip",
        ]
        # Thompson sampling parameters: Beta(alpha, beta) per (context_bucket, category)
        # alpha = wins when emphasizing this category, beta = losses
        self._alpha: dict[str, dict[str, float]] = {}
        self._beta: dict[str, dict[str, float]] = {}
        self._history: list[dict] = []

    def _context_key(self, matchup_state: dict) -> str:
        """Discretize the matchup state into a context bucket.

        Uses W/L/T counts to create a coarse state representation.
        """
        wins = matchup_state.get("wins", 0)
        losses = matchup_state.get("losses", 0)
        ties = matchup_state.get("ties", 0)

        if wins > losses + 2:
            return "winning_big"
        elif wins > losses:
            return "winning"
        elif losses > wins + 2:
            return "losing_big"
        elif losses > wins:
            return "losing"
        else:
            return "tied"

    def recommend_weights(
        self,
        matchup_state: dict,
        base_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Recommend category weights using Thompson sampling.

        Args:
            matchup_state: Dict with wins, losses, ties, and per-category status.
            base_weights: Existing category weights to blend with RL recommendations.

        Returns:
            Dict mapping category -> weight (0.0 to 2.0).
        """
        if base_weights is None:
            base_weights = {cat: 1.0 for cat in self.categories}

        # If insufficient data, return base weights with slight noise
        if len(self._history) < MIN_WEEKS_FOR_RL:
            return base_weights

        context = self._context_key(matchup_state)
        weights = {}

        for cat in self.categories:
            alpha = self._alpha.get(context, {}).get(cat, 1.0)
            beta = self._beta.get(context, {}).get(cat, 1.0)

            # Thompson sampling: draw from Beta(alpha, beta)
            sample = np.random.beta(alpha, beta)

            # Blend with base weight (50% RL, 50% optimizer)
            base = base_weights.get(cat, 1.0)
            rl_weight = 0.5 + sample * 1.5  # Map [0,1] -> [0.5, 2.0]
            weights[cat] = 0.5 * base + 0.5 * rl_weight

        return weights

    def record_outcome(
        self,
        matchup_state: dict,
        weights_used: dict[str, float],
        category_results: dict[str, str],
    ) -> None:
        """Record weekly outcome to update the bandit's posterior.

        Args:
            matchup_state: The context at the time of the decision.
            weights_used: The category weights that were used.
            category_results: Dict mapping category -> "W", "L", or "T".
        """
        context = self._context_key(matchup_state)

        if context not in self._alpha:
            self._alpha[context] = {cat: 1.0 for cat in self.categories}
            self._beta[context] = {cat: 1.0 for cat in self.categories}

        for cat in self.categories:
            result = category_results.get(cat, "T")
            weight = weights_used.get(cat, 1.0)

            # Only update categories where we made a meaningful decision
            if abs(weight - 1.0) < 0.1:
                continue

            if result == "W":
                # Success: increase alpha (reward emphasizing this category)
                self._alpha[context][cat] += ALPHA * weight
            elif result == "L":
                # Failure: increase beta (penalize emphasizing this category)
                self._beta[context][cat] += ALPHA * weight

        self._history.append(
            {
                "date": datetime.now(UTC).isoformat(),
                "context": context,
                "weights": weights_used,
                "results": category_results,
            }
        )

    @property
    def weeks_of_data(self) -> int:
        """Number of weeks of recorded outcome data."""
        return len(self._history)

    @property
    def is_ready(self) -> bool:
        """Whether the model has enough data to make useful recommendations."""
        return self.weeks_of_data >= MIN_WEEKS_FOR_RL

    def get_category_win_rates(self, context: str | None = None) -> dict[str, float]:
        """Get historical win rate per category for a given context.

        Returns dict mapping category -> win rate (0.0 to 1.0).
        """
        if not self._history:
            return {}

        cat_wins: dict[str, int] = {cat: 0 for cat in self.categories}
        cat_total: dict[str, int] = {cat: 0 for cat in self.categories}

        for entry in self._history:
            if context is not None and entry["context"] != context:
                continue
            for cat, result in entry.get("results", {}).items():
                if cat in cat_total:
                    cat_total[cat] += 1
                    if result == "W":
                        cat_wins[cat] += 1

        return {cat: cat_wins[cat] / max(1, cat_total[cat]) for cat in self.categories}


# Module-level singleton for the bandit (persists within session)
_bandit: LineupContextualBandit | None = None


def get_lineup_bandit(categories: list[str] | None = None) -> LineupContextualBandit:
    """Get or create the session-level lineup bandit singleton."""
    global _bandit  # noqa: PLW0603
    if _bandit is None:
        _bandit = LineupContextualBandit(categories)
    return _bandit
