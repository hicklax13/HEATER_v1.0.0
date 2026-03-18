"""Trade Finder: Scans league for mutually beneficial trade opportunities.

Uses ESPN/IBM-inspired cosine dissimilarity for team pairing (arXiv:2111.02859)
and tiered evaluation: fast deterministic scan for 1-for-1 trades, greedy
expansion to 2-for-1, and on-demand deep analysis via the full trade engine.

Tier 1 — Broad Scan (<3s): cosine pairing + SGP scoring for all 1-for-1 trades
Tier 2 — Focused Expansion (<5s): top seeds expanded to 2-for-1 via greedy add
Tier 3 — Deep Analysis (on-demand): full Phase 1-5 evaluate_trade() per trade
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.in_season import _roster_category_totals
from src.valuation import LeagueConfig

# ── Constants ─────────────────────────────────────────────────────────

MIN_SGP_GAIN = 0.3  # Minimum user SGP gain to surface a trade
MAX_OPP_LOSS = -0.5  # Reject trades where opponent loses more than this (e.g., opp_delta < -0.5)
LOSS_AVERSION = 1.5  # Opponent values outgoing players 1.5x (Kahneman)


# ── Team Vector & Cosine Similarity ───────────────────────────────────


def compute_team_vectors(
    all_team_totals: dict[str, dict[str, float]],
    config: LeagueConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute 12-dimensional category z-score vector per team.

    Used for cosine dissimilarity to find complementary teams.

    Args:
        all_team_totals: {team_name: {category: total}}.
        config: League configuration.

    Returns:
        {team_name: np.ndarray of shape (12,)}
    """
    if config is None:
        config = LeagueConfig()

    cats = config.all_categories
    n_cats = len(cats)

    # Compute mean and std per category across all teams
    cat_values: dict[str, list[float]] = {c: [] for c in cats}
    for totals in all_team_totals.values():
        for c in cats:
            cat_values[c].append(totals.get(c, 0))

    means = {c: np.mean(cat_values[c]) if cat_values[c] else 0 for c in cats}
    stds = {c: max(np.std(cat_values[c]), 1e-6) if cat_values[c] else 1.0 for c in cats}

    vectors = {}
    for team_name, totals in all_team_totals.items():
        vec = np.zeros(n_cats)
        for i, c in enumerate(cats):
            val = totals.get(c, 0)
            z = (val - means[c]) / stds[c]
            # Flip inverse stats so higher z = better
            if c in config.inverse_stats:
                z = -z
            vec[i] = z
        vectors[team_name] = vec

    return vectors


def cosine_dissimilarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine dissimilarity between two team vectors.

    Returns value in [0, 2]. Higher = more complementary needs.
    1.0 = orthogonal (independent needs), 2.0 = opposite strengths.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0  # orthogonal default
    cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return 1.0 - cos_sim


def find_complementary_teams(
    user_team: str,
    all_team_totals: dict[str, dict[str, float]],
    config: LeagueConfig | None = None,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Find the top N most complementary trade partner teams.

    Returns list of (team_name, dissimilarity_score) sorted by dissimilarity desc.
    """
    vectors = compute_team_vectors(all_team_totals, config)

    if user_team not in vectors:
        return []

    user_vec = vectors[user_team]
    scores = []
    for team_name, vec in vectors.items():
        if team_name == user_team:
            continue
        dissim = cosine_dissimilarity(user_vec, vec)
        scores.append((team_name, dissim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ── Acceptance Probability ────────────────────────────────────────────


def estimate_acceptance_probability(
    user_gain_sgp: float,
    opponent_gain_sgp: float,
    need_match_score: float = 0.5,
) -> float:
    """Estimate probability that opponent accepts the trade.

    Uses behavioral model: loss aversion, fairness gap, and need matching.

    Args:
        user_gain_sgp: How much YOU gain (SGP).
        opponent_gain_sgp: How much opponent gains (SGP, negative = they lose).
        need_match_score: How well the trade fills opponent needs (0-1).

    Returns:
        float in [0, 1] — estimated acceptance probability.
    """
    # Fairness gap (adjusted for loss aversion per Kahneman & Tversky):
    # Losses feel LOSS_AVERSION× worse. If opponent loses, multiply loss magnitude.
    if opponent_gain_sgp < 0:
        perceived_opp_gain = opponent_gain_sgp * LOSS_AVERSION  # Losses feel worse
    else:
        perceived_opp_gain = opponent_gain_sgp

    # Sigmoid: P(accept) = 1 / (1 + exp(k * fairness - m * need))
    fairness_gap = abs(user_gain_sgp - perceived_opp_gain)
    exponent = 2.0 * fairness_gap - 1.5 * need_match_score - 0.5 * max(opponent_gain_sgp, 0)
    # Clamp exponent to avoid overflow
    exponent = max(-20.0, min(20.0, exponent))
    prob = 1.0 / (1.0 + math.exp(exponent))

    return max(0.01, min(0.99, prob))


def acceptance_label(prob: float) -> str:
    """Convert acceptance probability to human-readable label."""
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"


# ── 1-for-1 Trade Scanner ────────────────────────────────────────────


def scan_1_for_1(
    user_roster_ids: list[int],
    opponent_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> list[dict]:
    """Fast deterministic scan of all 1-for-1 trades between user and opponent.

    Computes SGP delta for both sides using roster category totals.
    Filters to trades where user gains and opponent doesn't lose too much.

    Returns list of dicts sorted by user_sgp_gain descending.
    """
    if config is None:
        config = LeagueConfig()

    # Pre-compute baseline totals
    user_totals = _roster_category_totals(user_roster_ids, player_pool)
    opp_totals = _roster_category_totals(opponent_roster_ids, player_pool)

    user_baseline = _totals_sgp(user_totals, config)
    opp_baseline = _totals_sgp(opp_totals, config)

    results = []

    for give_id in user_roster_ids:
        give_player = player_pool[player_pool["player_id"] == give_id]
        if give_player.empty:
            continue
        give_name = give_player.iloc[0].get("name", give_player.iloc[0].get("player_name", "?"))

        for recv_id in opponent_roster_ids:
            recv_player = player_pool[player_pool["player_id"] == recv_id]
            if recv_player.empty:
                continue

            # Same type check (hitter for hitter, pitcher for pitcher)
            give_is_hitter = int(give_player.iloc[0].get("is_hitter", 0))
            recv_is_hitter = int(recv_player.iloc[0].get("is_hitter", 0))
            if give_is_hitter != recv_is_hitter:
                continue  # Skip cross-type trades for simplicity

            recv_name = recv_player.iloc[0].get("name", recv_player.iloc[0].get("player_name", "?"))

            # User: lose give_id, gain recv_id
            new_user_ids = [pid for pid in user_roster_ids if pid != give_id] + [recv_id]
            new_user_totals = _roster_category_totals(new_user_ids, player_pool)
            user_new_sgp = _totals_sgp(new_user_totals, config)
            user_delta = user_new_sgp - user_baseline

            if user_delta < MIN_SGP_GAIN:
                continue

            # Opponent: lose recv_id, gain give_id
            new_opp_ids = [pid for pid in opponent_roster_ids if pid != recv_id] + [give_id]
            new_opp_totals = _roster_category_totals(new_opp_ids, player_pool)
            opp_new_sgp = _totals_sgp(new_opp_totals, config)
            opp_delta = opp_new_sgp - opp_baseline

            if opp_delta < MAX_OPP_LOSS:
                continue  # Opponent loses too much

            # Need match: how much does trade fill opponent's weak categories
            need_match = min(1.0, max(0.0, (opp_delta + 1.0) / 2.0))

            p_accept = estimate_acceptance_probability(user_delta, opp_delta, need_match)

            # Composite score: SGP gain weighted by acceptance likelihood
            composite = (
                0.50 * user_delta
                + 0.25 * p_accept * 3.0  # scale probability to SGP-like range
                + 0.15 * max(opp_delta, 0)  # bonus if opponent also benefits
                + 0.10 * need_match * 2.0  # bonus for need-matching trades
            )

            results.append(
                {
                    "giving_ids": [give_id],
                    "receiving_ids": [recv_id],
                    "giving_names": [give_name],
                    "receiving_names": [recv_name],
                    "user_sgp_gain": round(user_delta, 3),
                    "opponent_sgp_gain": round(opp_delta, 3),
                    "acceptance_probability": round(p_accept, 3),
                    "acceptance_label": acceptance_label(p_accept),
                    "composite_score": round(composite, 3),
                    "trade_type": "1-for-1",
                }
            )

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def _totals_sgp(totals: dict, config: LeagueConfig) -> float:
    """Convert roster category totals to total SGP."""
    total = 0.0
    for cat in config.all_categories:
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        if cat in config.inverse_stats:
            total -= val / denom
        else:
            total += val / denom
    return total


# ── Main Trade Finder ─────────────────────────────────────────────────


def find_trade_opportunities(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    user_team_name: str | None = None,
    league_rosters: dict[str, list[int]] | None = None,
    weeks_remaining: int = 16,
    max_results: int = 20,
    top_partners: int = 5,
) -> list[dict]:
    """Scan league for the best trade opportunities.

    Tier 1: Cosine dissimilarity pairing → 1-for-1 scan on top partners.
    Results sorted by composite score.

    Args:
        user_roster_ids: User's player IDs.
        player_pool: Full player pool DataFrame.
        config: League configuration.
        all_team_totals: {team_name: {cat: total}} for all 12 teams.
        user_team_name: User's team name.
        league_rosters: {team_name: [player_ids]} for all teams.
        weeks_remaining: Remaining weeks in season.
        max_results: Maximum trade opportunities to return.
        top_partners: Number of complementary teams to scan.

    Returns:
        List of trade opportunity dicts sorted by composite_score.
    """
    if config is None:
        config = LeagueConfig()

    if player_pool.empty or not user_roster_ids:
        return []

    if not league_rosters or not all_team_totals:
        return []

    # ── Tier 1: Find complementary teams ──────────────────────────────
    if user_team_name and all_team_totals:
        partners = find_complementary_teams(user_team_name, all_team_totals, config, top_n=top_partners)
    else:
        # Scan all opponents
        partners = [(tn, 1.0) for tn in league_rosters if tn != user_team_name]

    # ── Tier 1: Scan 1-for-1 trades with top partners ────────────────
    all_trades: list[dict] = []

    for opp_team, dissim_score in partners:
        opp_roster = league_rosters.get(opp_team, [])
        if not opp_roster:
            continue

        trades = scan_1_for_1(user_roster_ids, opp_roster, player_pool, config)

        for trade in trades:
            trade["opponent_team"] = opp_team
            trade["complementarity"] = round(dissim_score, 3)

        all_trades.extend(trades)

    # ── Grade trades ──────────────────────────────────────────────────
    try:
        from src.engine.output.trade_evaluator import grade_trade

        for trade in all_trades:
            trade["grade"] = grade_trade(trade["user_sgp_gain"])
    except ImportError:
        for trade in all_trades:
            sgp = trade["user_sgp_gain"]
            if sgp > 2.0:
                trade["grade"] = "A+"
            elif sgp > 1.5:
                trade["grade"] = "A"
            elif sgp > 1.0:
                trade["grade"] = "B+"
            elif sgp > 0.5:
                trade["grade"] = "B"
            else:
                trade["grade"] = "C+"

    # ── Sort and limit ────────────────────────────────────────────────
    all_trades.sort(key=lambda x: x["composite_score"], reverse=True)

    # Deduplicate: same giving+receiving set only once
    seen: set[tuple] = set()
    unique_trades: list[dict] = []
    for trade in all_trades:
        key = (
            tuple(sorted(trade["giving_ids"])),
            tuple(sorted(trade["receiving_ids"])),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_trades.append(trade)

    return unique_trades[:max_results]
