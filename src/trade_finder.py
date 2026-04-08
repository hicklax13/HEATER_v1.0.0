"""Trade Finder: Scans league for mutually beneficial trade opportunities.

Uses ESPN/IBM-inspired cosine dissimilarity for team pairing (arXiv:2111.02859)
and tiered evaluation: fast deterministic scan for 1-for-1 trades, greedy
expansion to 2-for-1, and on-demand deep analysis via the full trade engine.

Tier 1 — Broad Scan (<3s): cosine pairing + SGP scoring for all 1-for-1 trades
Tier 2 — Focused Expansion (<5s): top seeds expanded to 2-for-1 via greedy add
Tier 3 — Deep Analysis (on-demand): full Phase 1-5 evaluate_trade() per trade
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.alerts import IL_STASH_NAMES
from src.in_season import _roster_category_totals
from src.valuation import LeagueConfig, SGPCalculator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

MIN_SGP_GAIN = 0.3  # Minimum user SGP gain to surface a trade
MAX_OPP_LOSS = -0.5  # Reject trades where opponent loses more than this (e.g., opp_delta < -0.5)
LOSS_AVERSION = 1.8  # was 1.5 -- meta-analysis consensus (Brown 2024, Walasek 2024)
ACCEPTANCE_FLOOR = 0.15  # Trades below 15% acceptance filtered out entirely
MAX_EFFICIENCY_RATIO = 5.0  # Cap SGP gain ratio to prevent "trade your worst for their best"
ELITE_RETURN_FLOOR = 0.75  # Return must be >= 75% of given player's raw SGP
MAX_WEIGHT_RATIO = 1.5  # No category can be weighted more than 1.5x average

# Trade Finder composite weights (ROADMAP B8 — validate via D6 backtesting)
# Sum must equal 1.0. Acceptance-heavy because rejected trades are worthless.
COMPOSITE_W_SGP = 0.15  # Normalized SGP gain
COMPOSITE_W_ADP = 0.15  # ADP fairness (league draft round or generic ADP)
COMPOSITE_W_ECR = 0.15  # ECR fairness (expert consensus ranking parity)
COMPOSITE_W_ACCEPT = 0.30  # Acceptance probability (sigmoid with loss aversion)
COMPOSITE_W_CAT_FIT = 0.10  # Category fit (give from strength, receive for weakness)
COMPOSITE_W_OPP_NEED = 0.15  # Opponent need match (their weak categories)


def _player_sgp_volume_aware(
    pid: int,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    sgp_calc: SGPCalculator | None = None,
) -> float:
    """Compute SGP for a single player with proper volume weighting for rate stats.

    Use this instead of ``_totals_sgp(_roster_category_totals([pid], pool), config)``
    for individual players. The latter ignores AB/IP volume, inflating the SGP of
    low-PA players (a 200 PA .300 hitter gets the same AVG SGP as a 600 PA .300 hitter).

    ``SGPCalculator.total_sgp()`` properly handles rate stats by computing marginal
    impact relative to a team baseline with volume weighting.
    """
    p = player_pool[player_pool["player_id"] == pid]
    if p.empty:
        return 0.0
    if sgp_calc is None:
        sgp_calc = SGPCalculator(config)
    try:
        return sgp_calc.total_sgp(p.iloc[0])
    except Exception:
        return 0.0


# ── Volume-Aware Individual Player SGP ──────────────────────────────


def _player_sgp_volume_aware(player_id: int, player_pool: pd.DataFrame, config: LeagueConfig) -> float:
    """Compute volume-aware SGP for a single player using SGPCalculator.

    Unlike _totals_sgp() which treats rate stats as raw values (appropriate for
    full-roster aggregates), this function properly accounts for volume: a 600 AB
    .300 hitter gets more AVG SGP than a 200 AB .300 hitter because the former
    moves team AVG 3x more.

    Use this for individual player valuations (elite protection, efficiency cap,
    drop cost). Use _totals_sgp() only for full-roster category totals.
    """
    p = player_pool[player_pool["player_id"] == player_id]
    if p.empty:
        return 0.0
    sgp_calc = SGPCalculator(config)
    return sgp_calc.total_sgp(p.iloc[0])


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
    stds = {c: max(np.std(cat_values[c]), 0.01) if cat_values[c] else 1.0 for c in cats}

    vectors = {}
    for team_name, totals in all_team_totals.items():
        vec = np.zeros(n_cats)
        for i, c in enumerate(cats):
            val = totals.get(c, 0)
            std_c = stds[c]
            if std_c < 0.01:
                # Very low variance across teams: use raw deviation to preserve ordering
                z = val - means[c]
            else:
                z = (val - means[c]) / std_c
            # Flip inverse stats so higher z = better
            if c in config.inverse_stats:
                z = -z
            vec[i] = z
        vectors[team_name] = vec

    return vectors


def cosine_dissimilarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine dissimilarity between two team vectors.

    Returns value in [0, 1]. Higher = more complementary needs.
    0.5 = orthogonal/neutral, 1.0 = opposite strengths (most complementary).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.5  # neutral: insufficient data to differentiate
    cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    # Clamp to [0, 1] for display (raw range is [0, 2])
    return max(0.0, min(1.0, 1.0 - cos_sim))


def find_complementary_teams(
    user_team: str,
    all_team_totals: dict[str, dict[str, float]],
    config: LeagueConfig | None = None,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Find the top N most complementary trade partner teams.

    Uses rank-based needs vectors: a team ranked 12th in a category has
    high need (12), ranked 1st has low need (1). Two teams are complementary
    when one is strong where the other is weak (inverted needs).

    Returns list of (team_name, dissimilarity_score) sorted by dissimilarity desc.
    """
    if config is None:
        config = LeagueConfig()

    if user_team not in all_team_totals:
        return []

    teams = list(all_team_totals.keys())
    cats = list(config.all_categories)

    # Rank all teams in each category (1 = best, N = worst)
    team_ranks: dict[str, dict[str, int]] = {t: {} for t in teams}
    for cat in cats:
        if cat in config.inverse_stats:
            # Lower is better — ascending sort
            ranked = sorted(teams, key=lambda t: all_team_totals[t].get(cat, 999))
        else:
            # Higher is better — descending sort
            ranked = sorted(teams, key=lambda t: all_team_totals[t].get(cat, 0), reverse=True)
        for rank_idx, t in enumerate(ranked):
            team_ranks[t][cat] = rank_idx + 1

    # Build needs vectors: rank value directly (high rank number = high need)
    n_teams = len(teams)
    user_needs = np.array([team_ranks[user_team].get(cat, n_teams // 2) for cat in cats], dtype=float)

    scores = []
    for team_name in teams:
        if team_name == user_team:
            continue
        opp_ranks = np.array([team_ranks[team_name].get(cat, n_teams // 2) for cat in cats], dtype=float)
        # Complementarity: opponent is strong (low rank) where user is weak (high rank)
        # Invert opponent's ranks: strong (rank 1) becomes high value (n_teams)
        opp_strength = (n_teams + 1) - opp_ranks
        # Dot product of user_needs and opp_strength, normalized
        dot = np.dot(user_needs, opp_strength)
        max_dot = n_teams * n_teams * len(cats)  # theoretical max
        min_dot = len(cats)  # theoretical min (all rank 1 vs all rank 1)
        if max_dot > min_dot:
            complementarity = (dot - min_dot) / (max_dot - min_dot)
        else:
            complementarity = 0.5
        complementarity = max(0.0, min(1.0, complementarity))
        scores.append((team_name, round(complementarity, 3)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ── Acceptance Probability ────────────────────────────────────────────


def estimate_acceptance_probability(
    user_gain_sgp: float,
    opponent_gain_sgp: float,
    need_match_score: float = 0.5,
    adp_fairness: float = 0.5,
    opponent_need_match: float = 0.5,
    opponent_standings_rank: int | None = None,
    opponent_trade_willingness: float = 0.5,
    loss_aversion: float = LOSS_AVERSION,
    recv_draft_round: int | None = None,
    recv_ytd_vs_proj: float | None = None,
    recv_recently_acquired: bool = False,
) -> float:
    """Estimate probability that opponent accepts the trade.

    Uses behavioral model: loss aversion, fairness gap, need matching,
    ADP fairness, opponent need alignment, standings position, trade
    willingness archetype, and behavioral biases (F2/F3/F4).

    Args:
        user_gain_sgp: How much YOU gain (SGP).
        opponent_gain_sgp: How much opponent gains (SGP, negative = they lose).
        need_match_score: How well the trade fills opponent needs (0-1).
        adp_fairness: ADP match between traded players (0-1, 1 = perfect).
        opponent_need_match: How well trade fills opponent category gaps (0-1).
        opponent_standings_rank: Opponent rank in standings (1-12, None if unknown).
        opponent_trade_willingness: Archetype willingness to trade (0-1).
        recv_draft_round: F2 — draft round of player opponent gives up (1-23).
        recv_ytd_vs_proj: F3 — ratio of YTD performance to projection (e.g., 1.1 = 10% above).
        recv_recently_acquired: F4 — True if opponent acquired player via trade within 3 weeks.

    Returns:
        float in [0, 1] -- estimated acceptance probability.
    """
    # Fairness gap (adjusted for loss aversion per Kahneman & Tversky):
    # Losses feel loss_aversion x worse. If opponent loses, multiply loss magnitude.
    if opponent_gain_sgp < 0:
        perceived_opp_gain = opponent_gain_sgp * loss_aversion  # Losses feel worse
    else:
        perceived_opp_gain = opponent_gain_sgp

    # Sigmoid: P(accept) = 1 / (1 + exp(exponent))
    fairness_gap = abs(user_gain_sgp - perceived_opp_gain)

    # ADP penalty: strong penalty when opponent gives up a higher-drafted player
    adp_penalty = max(0, (0.5 - adp_fairness) * 3.0)

    # F2: Draft-round anchoring — owners overvalue early-round picks (endowment effect).
    # Rd 1-3 = 1.3x perceived value, Rd 4-8 = 1.15x, Rd 9+ = 1.0x.
    draft_anchor_penalty = 0.0
    if recv_draft_round is not None and recv_draft_round >= 1:
        if recv_draft_round <= 3:
            draft_anchor_penalty = 0.6  # Strong resistance to giving up 1st-3rd rounders
        elif recv_draft_round <= 8:
            draft_anchor_penalty = 0.3  # Moderate for mid-round picks

    # F3: Disposition effect — owners hold winners and sell losers.
    # If the player opponent gives up is outperforming projections, they
    # feel good and are LESS willing to trade. Underperformers → more willing.
    disposition_bonus = 0.0
    if recv_ytd_vs_proj is not None:
        if recv_ytd_vs_proj > 1.10:
            disposition_bonus = -0.3  # Outperforming → harder to pry away
        elif recv_ytd_vs_proj < 0.90:
            disposition_bonus = 0.3  # Underperforming → more willing to deal

    # F4: Recently-acquired penalty — owners who just traded FOR a player
    # won't flip them within 3 weeks (sunk cost + cognitive dissonance).
    recency_penalty = 0.3 if recv_recently_acquired else 0.0

    # Playoff-odds-aware bubble bonus: teams on the bubble trade most actively.
    bubble_bonus = 0.0
    if opponent_standings_rank:
        n_teams = 12
        playoff_odds = max(0.0, 1.0 - (opponent_standings_rank - 1) / (n_teams - 1))

        if playoff_odds < 0.15:
            bubble_bonus = -0.2  # Checked out, less willing
        elif 0.30 <= playoff_odds <= 0.70:
            bubble_bonus = 0.4  # Bubble team, most active
        elif playoff_odds > 0.85:
            bubble_bonus = -0.1  # Conservative contender

    exponent = (
        2.0 * fairness_gap
        - 1.5 * need_match_score
        - 0.5 * max(opponent_gain_sgp, 0)
        - 1.0 * opponent_need_match
        + adp_penalty
        + draft_anchor_penalty
        + recency_penalty
        + disposition_bonus
        - bubble_bonus
        - 0.5 * opponent_trade_willingness
    )
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


# ── ADP Fairness ─────────────────────────────────────────────────────


def compute_adp_fairness(
    give_id: int,
    recv_id: int,
    player_pool: pd.DataFrame,
) -> float:
    """Compute ADP fairness score between two players.

    Uses league draft round (strongest signal) with generic ADP fallback.
    ADP fairness is critical because league mates won't accept trades
    where they give up a higher-drafted player, even if stats favor them.

    Returns:
        float in [0.0, 1.0] where 1.0 = perfect ADP match,
        0.0 = extreme ADP mismatch.
    """
    # Try league-specific draft rounds first
    try:
        from src.database import get_player_draft_round

        give_round = get_player_draft_round(give_id)
        recv_round = get_player_draft_round(recv_id)
        if give_round and recv_round:
            round_gap = abs(give_round - recv_round)
            max_gap = 23  # max rounds in league draft
            return max(0.0, 1.0 - (round_gap / max_gap))
    except Exception:
        pass

    # Fallback to generic ADP
    give_p = player_pool[player_pool["player_id"] == give_id]
    recv_p = player_pool[player_pool["player_id"] == recv_id]
    give_adp = float(give_p.iloc[0].get("adp", 999) or 999) if not give_p.empty else 999
    recv_adp = float(recv_p.iloc[0].get("adp", 999) or 999) if not recv_p.empty else 999

    if give_adp >= 500 or recv_adp >= 500:
        return 0.5  # Unknown ADP = neutral

    # ADP ratio: closer to 1.0 = fairer trade
    ratio = min(give_adp, recv_adp) / max(give_adp, recv_adp, 0.01)
    return ratio**0.5  # sqrt softens extreme gaps


# ── Multi-Player Helpers ─────────────────────────────────────────────

# H2: Dynamic roster spot value — recomputed weekly from FA pool median.
# Fallback to 0.8 if FA pool is unavailable.
ROSTER_SPOT_SGP_DEFAULT = 0.8


def compute_roster_spot_sgp(player_pool: pd.DataFrame, config: LeagueConfig | None = None) -> float:
    """Compute dynamic roster spot value as 80% of median FA SGP.

    When the FA pool has data, uses the median SGP of available free agents
    as the opportunity cost of a roster spot. Multiplied by 0.8 because not
    all FAs are equally accessible (waiver priority, etc.).

    Returns:
        float: ROSTER_SPOT_SGP value (minimum 0.2, maximum 2.0)
    """
    if config is None:
        config = LeagueConfig()
    try:
        # Filter to likely free agents (no roster status or ADP > 300)
        fa_mask = (
            (player_pool.get("status", pd.Series("", index=player_pool.index)).isin(["", "active", "FA"]))
            & (player_pool.get("adp", pd.Series(999, index=player_pool.index)) > 250)
            & (player_pool.get("pa", pd.Series(0, index=player_pool.index)).fillna(0) + player_pool.get("ip", pd.Series(0, index=player_pool.index)).fillna(0) > 0)
        )
        fa_pool = player_pool[fa_mask]
        if len(fa_pool) < 10:
            return ROSTER_SPOT_SGP_DEFAULT

        # Compute SGP for FA players
        from src.valuation import SGPCalculator

        sgp_calc = SGPCalculator(config)
        fa_sgps = []
        for _, row in fa_pool.head(100).iterrows():
            try:
                sgp = sgp_calc.total_sgp(row)
                if sgp > 0:
                    fa_sgps.append(sgp)
            except Exception:
                continue

        if len(fa_sgps) < 5:
            return ROSTER_SPOT_SGP_DEFAULT

        median_sgp = float(np.median(fa_sgps))
        return max(0.2, min(2.0, median_sgp * 0.8))
    except Exception:
        return ROSTER_SPOT_SGP_DEFAULT


def _count_contributing_categories(
    player_row,
    config: LeagueConfig,
    min_pct_of_avg: float = 0.25,
) -> int:
    """Count how many categories a player contributes meaningfully to.

    A player 'contributes' to a category if their per-game production
    is at least min_pct_of_avg of the league average per-game rate.
    This prevents single-category specialists from dominating trade
    recommendations.

    Returns:
        Number of categories with meaningful contribution (0-12).
    """
    # Approximate league-average per-game production benchmarks
    # (full-season averages divided by 162 games)
    avg_per_game = {
        "R": 0.50,
        "HR": 0.15,
        "RBI": 0.50,
        "SB": 0.08,
        "AVG": 0.265,
        "OBP": 0.330,
        "W": 0.06,
        "L": 0.06,
        "SV": 0.03,
        "K": 0.55,
        "ERA": 4.20,
        "WHIP": 1.25,
    }
    count = 0
    for cat in config.all_categories:
        col = cat.lower()
        val = float(player_row.get(col, 0) or 0)
        # Per-game rate (divide season total by 162)
        per_game = val / 162.0 if cat not in config.rate_stats else val
        benchmark = avg_per_game.get(cat, 0)
        if benchmark <= 0:
            continue
        if cat in config.rate_stats:
            # Rate stats: must be within reasonable range of average
            if cat in config.inverse_stats:
                ip = float(player_row.get("ip", 0) or 0)
                if cat in ("ERA", "WHIP") and ip <= 0:
                    continue  # No IP = doesn't contribute to pitching rates
                if val < benchmark * 1.5:  # ERA below 6.30 = contributing
                    count += 1
            else:
                if val > benchmark * min_pct_of_avg:  # AVG above .066 = contributing
                    count += 1
        else:
            if per_game >= benchmark * min_pct_of_avg:
                count += 1
    return count


MIN_CONTRIBUTING_CATEGORIES = 2  # Received player must contribute to at least 2 categories

# Maximum fraction of total weighted SGP from any single category.
# Prevents AVG-only specialists from dominating when AVG is weighted 2x.
MAX_SINGLE_CAT_SGP_FRACTION = 0.40


def _check_category_dominance(
    player_row,
    config: LeagueConfig,
    weights: dict[str, float] | None = None,
) -> bool:
    """Check if a player's value is too concentrated in one category.

    Returns True if the player passes (balanced enough), False if
    any single category contributes > MAX_SINGLE_CAT_SGP_FRACTION
    of their total weighted SGP.
    """
    per_cat = {}
    for cat in config.all_categories:
        col = cat.lower()
        val = float(player_row.get(col, 0) or 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        w = weights.get(cat, 1.0) if weights else 1.0
        if abs(denom) > 1e-9:
            sgp = val / denom * w
            if cat in config.inverse_stats:
                sgp = -sgp
            per_cat[cat] = abs(sgp)

    total = sum(per_cat.values())
    if total <= 0:
        return True  # No production = pass (will be filtered elsewhere)

    max_frac = max(per_cat.values()) / total
    return max_frac <= MAX_SINGLE_CAT_SGP_FRACTION


def _compute_drop_cost(
    roster_ids: list[int],
    incoming_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
) -> tuple[float, int | None]:
    """Find worst bench player and compute SGP cost of dropping them.

    Returns (drop_cost_sgp, drop_player_id) or (0.0, None).
    """
    # Combine current roster + incoming to find the full post-trade roster
    all_ids = set(roster_ids) | set(incoming_ids)

    # Volume-aware SGP per player (accounts for AB/IP in rate stats)
    sgps: dict[int, float] = {}
    for pid in all_ids:
        sgp_val = _player_sgp_volume_aware(pid, player_pool, config)
        if sgp_val != 0.0 or not player_pool[player_pool["player_id"] == pid].empty:
            sgps[pid] = sgp_val

    if not sgps:
        return 0.0, None

    # Multi-factor droppability scoring (lower = more droppable)
    drop_scores: dict[int, float] = {}
    for pid, raw_sgp in sgps.items():
        p = player_pool[player_pool["player_id"] == pid]
        if p.empty:
            drop_scores[pid] = raw_sgp
            continue
        row = p.iloc[0]
        score = raw_sgp
        is_hitter = int(row.get("is_hitter", 0)) == 1

        # DH-only penalty
        positions = str(row.get("positions", "")).upper()
        if is_hitter and positions in ("DH", "UTIL", ""):
            score -= 3.0

        # 0 SB penalty for hitters
        sb = float(row.get("sb", 0) or 0)
        if is_hitter and sb < 1:
            score -= 1.5

        # Below-average AVG drag
        avg = float(row.get("avg", 0) or 0)
        if is_hitter and 0 < avg < 0.245:
            score -= 1.0

        drop_scores[pid] = score

    worst_pid = min(drop_scores, key=drop_scores.get)
    worst_sgp = sgps[worst_pid]

    # Drop cost = positive value lost by dropping worst player.
    # If worst player has negative SGP, dropping them is beneficial (cost = 0).
    return max(0.0, worst_sgp), worst_pid


# ── 2-for-1 Trade Scanner ───────────────────────────────────────────


def scan_2_for_1(
    seeds: list[dict],
    user_roster_ids: list[int],
    opponent_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    category_weights: dict[str, float] | None = None,
    max_expansions: int = 50,
    user_category_profile: dict[str, dict] | None = None,
    loss_aversion: float = LOSS_AVERSION,
    max_opp_loss: float = MAX_OPP_LOSS,
) -> list[dict]:
    """Expand top 1-for-1 seeds by adding a second give player.

    User gives 2 players, receives 1. User gains a roster spot.
    Opponent receives 2, must drop their worst bench player.
    """
    if config is None:
        config = LeagueConfig()

    # H2: Dynamic roster spot value from FA pool
    roster_spot_sgp = compute_roster_spot_sgp(player_pool, config)

    results: list[dict] = []
    evaluated = 0

    for seed in seeds:
        if evaluated >= max_expansions:
            break

        orig_give = seed.get("giving_ids", [])
        orig_recv = seed.get("receiving_ids", [])
        if not orig_give or not orig_recv:
            continue

        # Try adding each user player as a second give
        for add_id in user_roster_ids:
            if add_id in orig_give or add_id in orig_recv:
                continue
            if evaluated >= max_expansions:
                break

            add_player = player_pool[player_pool["player_id"] == add_id]
            if add_player.empty:
                continue
            # Skip IL stash players — AVIS Section 7 protection
            add_name = add_player.iloc[0].get("name", add_player.iloc[0].get("player_name", ""))
            if add_name in IL_STASH_NAMES:
                continue

            new_give = orig_give + [add_id]
            new_recv = list(orig_recv)

            # Compute new SGP delta for user
            new_user_ids = [pid for pid in user_roster_ids if pid not in new_give] + new_recv
            new_user_totals = _roster_category_totals(new_user_ids, player_pool)
            user_baseline = _roster_category_totals(user_roster_ids, player_pool)

            user_new_sgp = _weighted_totals_sgp(new_user_totals, config, category_weights)
            user_base_sgp = _weighted_totals_sgp(user_baseline, config, category_weights)
            user_delta = user_new_sgp - user_base_sgp

            # H2: Add dynamic roster spot bonus (user gains a slot)
            user_delta += roster_spot_sgp

            if user_delta < MIN_SGP_GAIN:
                evaluated += 1
                continue

            # Opponent delta
            new_opp_ids = [pid for pid in opponent_roster_ids if pid not in new_recv] + list(new_give)
            opp_baseline = _roster_category_totals(opponent_roster_ids, player_pool)
            new_opp_totals = _roster_category_totals(new_opp_ids, player_pool)
            opp_delta = _totals_sgp(new_opp_totals, config) - _totals_sgp(opp_baseline, config)

            # Opponent must drop someone (receives 2, gives 1 = +1 roster)
            # Only existing opponent players are drop candidates (not players just received in trade)
            existing_opp = [pid for pid in new_opp_ids if pid not in new_give]
            drop_cost, _drop_pid = _compute_drop_cost(existing_opp, list(new_give), player_pool, config)
            opp_delta -= drop_cost

            if opp_delta < max_opp_loss:
                evaluated += 1
                continue

            # ADP check: the ADDED player must also pass the ADP filter.
            # Trading Raleigh (Rd 2) as a "throw-in" to sweeten a deal is absurd.
            add_adp = float(add_player.iloc[0].get("adp", 999) or 999)
            recv_adp_val = (
                float(player_pool[player_pool["player_id"] == orig_recv[0]].iloc[0].get("adp", 999) or 999)
                if not player_pool[player_pool["player_id"] == orig_recv[0]].empty
                else 999
            )
            if add_adp < 500 and recv_adp_val < 500:
                if add_adp < recv_adp_val * 0.5:
                    # Added player is drafted WAY higher than the return — block
                    evaluated += 1
                    continue

            # Score — use WORST ADP fairness across all give players
            fairness_scores = [compute_adp_fairness(gid, orig_recv[0], player_pool) for gid in new_give]
            adp_fairness = min(fairness_scores) if fairness_scores else 0.5

            need_match = min(1.0, max(0.0, (opp_delta + 1.0) / 2.0))
            p_accept = estimate_acceptance_probability(
                user_delta, opp_delta, need_match, adp_fairness=adp_fairness, loss_aversion=loss_aversion
            )

            # Acceptance floor: skip trades nobody would accept
            if p_accept < ACCEPTANCE_FLOOR:
                evaluated += 1
                continue

            # Category strategic fit for 2-for-1
            category_fit = 0.5
            if user_category_profile:
                try:
                    _give_contribs: dict[str, float] = {}
                    for gid in new_give:
                        gp = player_pool[player_pool["player_id"] == gid]
                        if not gp.empty:
                            for cat, val in _player_category_contribution(gp.iloc[0], config).items():
                                _give_contribs[cat] = _give_contribs.get(cat, 0) + val
                    _recv_contribs: dict[str, float] = {}
                    for rid in new_recv:
                        rp = player_pool[player_pool["player_id"] == rid]
                        if not rp.empty:
                            for cat, val in _player_category_contribution(rp.iloc[0], config).items():
                                _recv_contribs[cat] = _recv_contribs.get(cat, 0) + val
                    category_fit = _compute_category_fit_score(_give_contribs, _recv_contribs, user_category_profile)
                except Exception:
                    category_fit = 0.5

            # Remove roster spot bonus from composite to match 1-for-1 scale
            user_delta_for_score = user_delta - roster_spot_sgp
            norm_sgp_2 = min(user_delta_for_score / 3.0, 1.0) if user_delta_for_score > 0 else 0.0
            composite = (
                COMPOSITE_W_SGP * norm_sgp_2
                + COMPOSITE_W_ADP * adp_fairness
                + COMPOSITE_W_ECR * 0.5  # ECR neutral (not computed for multi-player)
                + COMPOSITE_W_ACCEPT * p_accept
                + COMPOSITE_W_CAT_FIT * category_fit
                + COMPOSITE_W_OPP_NEED * need_match
            ) + 0.05  # Small fixed bonus for roster flexibility

            give_names = seed.get("giving_names", []) + [
                str(add_player.iloc[0].get("name", add_player.iloc[0].get("player_name", "?")))
            ]
            recv_names = seed.get("receiving_names", [])

            results.append(
                {
                    "giving_ids": new_give,
                    "receiving_ids": new_recv,
                    "giving_names": give_names,
                    "receiving_names": recv_names,
                    "user_sgp_gain": round(user_delta, 3),
                    "opponent_sgp_gain": round(opp_delta, 3),
                    "acceptance_probability": round(p_accept, 3),
                    "acceptance_label": acceptance_label(p_accept),
                    "composite_score": round(composite, 3),
                    "trade_type": "2-for-1",
                    "is_closer_trade": False,
                    "adp_fairness": round(adp_fairness, 3),
                }
            )
            evaluated += 1

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results[:20]


# ── LP-Constrained Baseline ─────────────────────────────────────────


def _lp_constrained_totals(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict[str, float] | None:
    """Compute category totals using only LP-optimal starters (no bench).

    Returns a dict with uppercase keys matching ``_roster_category_totals()``
    format (R, HR, AVG, ERA, etc.), or None if the LP solver is unavailable
    or fails.
    """
    try:
        from src.lineup_optimizer import LineupOptimizer

        roster = player_pool[player_pool["player_id"].isin(roster_ids)].copy()
        if roster.empty or len(roster) < 5:
            return None

        # LineupOptimizer needs player_name column
        if "player_name" not in roster.columns and "name" in roster.columns:
            roster = roster.rename(columns={"name": "player_name"})

        optimizer = LineupOptimizer(roster, config=config)
        result = optimizer.optimize_lineup()
        if not result or result.get("status") != "Optimal":
            return None

        projected = result.get("projected_stats")
        if not projected:
            return None

        # Map lowercase LP keys to uppercase _roster_category_totals format
        totals: dict[str, float] = {}
        _key_map = {
            "r": "R",
            "hr": "HR",
            "rbi": "RBI",
            "sb": "SB",
            "w": "W",
            "l": "L",
            "sv": "SV",
            "k": "K",
            "avg": "AVG",
            "obp": "OBP",
            "era": "ERA",
            "whip": "WHIP",
        }
        for lc_key, uc_key in _key_map.items():
            if lc_key in projected:
                totals[uc_key] = float(projected[lc_key])

        # Also carry over component stats for downstream use (ip, ab, etc.)
        for comp_key in ("ab", "h", "bb", "hbp", "sf", "ip", "er", "bb_allowed", "h_allowed"):
            if comp_key in projected:
                totals[comp_key] = float(projected[comp_key])

        # Verify we got the key scoring categories
        required = {"R", "HR", "RBI", "SB", "AVG", "ERA"}
        if not required.issubset(totals.keys()):
            return None

        return totals
    except Exception:
        logger.debug("LP-constrained totals failed, falling back to raw totals")
        return None


# ── 1-for-1 Trade Scanner ────────────────────────────────────────────


def scan_1_for_1(
    user_roster_ids: list[int],
    opponent_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    category_weights: dict[str, float] | None = None,
    fa_comparisons: dict[int, dict] | None = None,
    roster_statuses: dict[int, str] | None = None,
    opponent_team_name: str | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    user_category_profile: dict[str, dict] | None = None,
    elite_return_floor: float = 0.75,
    max_weight_ratio: float = 1.5,
    loss_aversion: float = LOSS_AVERSION,
    max_opp_loss: float = MAX_OPP_LOSS,
) -> list[dict]:
    """Fast deterministic scan of all 1-for-1 trades between user and opponent.

    Computes SGP delta for both sides using roster category totals.
    When ``category_weights`` are provided, uses marginal-weighted SGP
    instead of raw SGP (categories where you can gain standings positions
    are weighted higher, punted categories get zero weight).

    Filters to trades where user gains and opponent doesn't lose too much.
    Excludes NA/minors players and annotates trades with health and FA info.

    Returns list of dicts sorted by user_sgp_gain descending.
    """
    if config is None:
        config = LeagueConfig()

    # Pre-compute baseline totals
    # Try LP-constrained baseline for user (starters only, no bench inflation).
    # The LP solver selects optimal 18 starters; bench production excluded.
    # Only computed ONCE before the scan loop (not per-candidate).
    user_totals = _roster_category_totals(user_roster_ids, player_pool)
    lp_user_totals = _lp_constrained_totals(user_roster_ids, player_pool, config)
    if lp_user_totals is not None:
        user_totals = lp_user_totals
        logger.debug("Using LP-constrained user baseline (starters only)")

    opp_totals = _roster_category_totals(opponent_roster_ids, player_pool)

    opp_baseline = _totals_sgp(opp_totals, config)

    # --- Cap extreme category weights ---
    # Prevent a single weak category from dominating all trade valuations.
    # Without this, being 12th in SB makes speed guys "outvalue" elite power bats.
    MAX_WEIGHT_RATIO = max_weight_ratio  # Scaled by desperation (default 1.5x average)
    capped_weights = category_weights
    if category_weights:
        non_zero = [v for v in category_weights.values() if v > 0]
        if non_zero:
            avg_w = sum(non_zero) / len(non_zero)
            cap = avg_w * MAX_WEIGHT_RATIO
            capped_weights = {k: min(v, cap) for k, v in category_weights.items()}

    # --- Pre-compute raw SGP per user player for elite protection ---
    # Players in the top 20% by raw SGP require the return player to have
    # at least 50% of their raw SGP. Prevents trading away stars for role players.
    ELITE_PERCENTILE = 80  # Top 20%
    ELITE_RETURN_FLOOR = elite_return_floor  # Scaled by desperation (default 0.75)

    user_raw_sgps: dict[int, float] = {}
    for pid in user_roster_ids:
        p = player_pool[player_pool["player_id"] == pid]
        if not p.empty:
            user_raw_sgps[pid] = _player_sgp_volume_aware(pid, player_pool, config)
    if user_raw_sgps:
        elite_threshold = float(np.percentile(list(user_raw_sgps.values()), ELITE_PERCENTILE))
    else:
        elite_threshold = 999.0

    # Baseline must use the same capped weights as post-trade calculations
    user_baseline = _weighted_totals_sgp(user_totals, config, capped_weights)

    # --- Load opponent needs (from opponent_trade_analysis) ---
    opp_needs_analysis: dict = {}
    opp_archetype_willingness: float = 0.5
    if opponent_team_name and all_team_totals:
        try:
            from src.opponent_trade_analysis import compute_opponent_needs, get_opponent_archetype

            opp_needs_analysis = compute_opponent_needs(opponent_team_name, all_team_totals)
            arch = get_opponent_archetype(opponent_team_name)
            opp_archetype_willingness = arch.get("trade_willingness", 0.5)
        except Exception:
            pass

    # --- Load ECR consensus rankings for ranking fairness ---
    ecr_ranks: dict[int, int] = {}
    try:
        from src.database import get_connection

        _conn = get_connection()
        try:
            import pandas as _pd

            _ecr = _pd.read_sql_query("SELECT player_id, consensus_rank FROM ecr_consensus", _conn)
            ecr_ranks = dict(zip(_ecr["player_id"].astype(int), _ecr["consensus_rank"].astype(int)))
        finally:
            _conn.close()
    except Exception:
        pass

    # --- Load YTD 2026 stats for recent performance modifier ---
    ytd_stats: dict[int, dict] = {}
    try:
        from src.database import get_connection as _gc2

        _conn2 = _gc2()
        try:
            _ytd = pd.read_sql_query(
                "SELECT player_id, pa, avg, hr, rbi, sb, era, whip FROM season_stats WHERE season = 2026 AND pa > 0",
                _conn2,
            )
            for _, _row in _ytd.iterrows():
                ytd_stats[int(_row["player_id"])] = {
                    "pa": int(_row.get("pa", 0) or 0),
                    "avg": float(_row.get("avg", 0) or 0),
                    "hr": int(_row.get("hr", 0) or 0),
                }
        finally:
            _conn2.close()
    except Exception:
        pass

    # F4: Load recent transactions to detect recently-acquired players
    _transactions_df = None
    try:
        from src.database import get_connection as _gc3

        _conn3 = _gc3()
        try:
            _transactions_df = pd.read_sql_query(
                "SELECT player_id, type, timestamp FROM transactions WHERE type = 'trade'",
                _conn3,
            )
        finally:
            _conn3.close()
    except Exception:
        pass

    results = []

    for give_id in user_roster_ids:
        give_player = player_pool[player_pool["player_id"] == give_id]
        if give_player.empty:
            continue
        give_name = give_player.iloc[0].get("name", give_player.iloc[0].get("player_name", "?"))
        # Skip IL stash players — AVIS Section 7 protection
        if give_name in IL_STASH_NAMES:
            continue
        give_raw_sgp = user_raw_sgps.get(give_id, 0.0)

        for recv_id in opponent_roster_ids:
            # Filter NA players. H9: Minors players get discounted valuation
            # instead of auto-exclusion — prospects with call-up signals retain value.
            _prospect_discount = 1.0
            if roster_statuses:
                recv_status = str(roster_statuses.get(recv_id, "active")).lower()
                if recv_status in ("na", "not active"):
                    continue
                if recv_status == "minors":
                    # H9: Prospect call-up valuation — P(call_up) * discount
                    try:
                        from src.prospect_engine import compute_call_up_signals

                        _p_row = player_pool[player_pool["player_id"] == recv_id]
                        if not _p_row.empty:
                            _signals = compute_call_up_signals(_p_row.iloc[0].to_dict())
                            _score = _signals.get("call_up_score", 0)
                            if _score >= 50:
                                # Scale: 50 score → 0.3x value, 100 score → 0.8x value
                                _prospect_discount = 0.3 + (_score - 50) * 0.01
                            else:
                                continue  # Low call-up probability — skip
                        else:
                            continue
                    except Exception:
                        continue  # Can't evaluate — skip

            recv_player = player_pool[player_pool["player_id"] == recv_id]
            if recv_player.empty:
                continue

            # Track cross-type trades (hitter ↔ pitcher) — allowed for category strategy
            give_is_hitter = int(give_player.iloc[0].get("is_hitter", 0))
            recv_is_hitter = int(recv_player.iloc[0].get("is_hitter", 0))
            is_cross_type = give_is_hitter != recv_is_hitter

            # --- Elite player protection ---
            # If giving away a top-20% player, the return must be at least 75% as good
            if give_raw_sgp >= elite_threshold:
                recv_raw_sgp = _player_sgp_volume_aware(recv_id, player_pool, config)
                if recv_raw_sgp < give_raw_sgp * ELITE_RETURN_FLOOR:
                    continue  # Don't trade elite players for scrubs

            # --- Draft capital / ADP value floor ---
            # Two-tier check: (1) league-specific draft round, (2) generic ADP.
            # Never trade a player drafted significantly higher than the return.
            # League draft round is the strongest signal — if you drafted Raleigh
            # in Rd 2, you should not trade him for a Rd 16 pick.
            ADP_RATIO_MAX = 2.5  # Generic ADP: max 2.5x ratio
            DRAFT_ROUND_MAX_GAP = 8  # League draft: max 8 rounds apart

            # Initialize draft round vars (used later for ADP fairness + result)
            give_round: int | None = None
            recv_round: int | None = None

            # Try league-specific draft rounds first (strongest signal)
            try:
                from src.database import get_player_draft_round

                give_round = get_player_draft_round(give_id)
                recv_round = get_player_draft_round(recv_id)
                if give_round and recv_round:
                    # A Rd 2 pick should not be traded for a Rd 15 pick (gap > 8)
                    if recv_round - give_round > DRAFT_ROUND_MAX_GAP:
                        continue
            except Exception:
                pass

            # Generic ADP (used for filter + ADP fairness + result)
            give_adp = float(give_player.iloc[0].get("adp", 999) or 999)
            recv_adp = float(recv_player.iloc[0].get("adp", 999) or 999)
            if give_adp < 500 and recv_adp < 500:  # Both have real ADP data
                if recv_adp > give_adp * ADP_RATIO_MAX:
                    continue  # Return player drafted way too late vs given player

            recv_name = recv_player.iloc[0].get("name", recv_player.iloc[0].get("player_name", "?"))

            # --- Category breadth check ---
            # Reject single-category specialists. The received player must
            # contribute meaningfully in at least 2 categories to prevent
            # trades like "give 40 HR hitter, get .303 AVG-only hitter."
            recv_breadth = _count_contributing_categories(recv_player.iloc[0], config)
            if recv_breadth < MIN_CONTRIBUTING_CATEGORIES:
                continue

            # User: lose give_id, gain recv_id
            new_user_ids = [pid for pid in user_roster_ids if pid != give_id] + [recv_id]
            new_user_totals = _roster_category_totals(new_user_ids, player_pool)
            user_new_sgp = _weighted_totals_sgp(new_user_totals, config, capped_weights)
            user_delta = user_new_sgp - user_baseline

            # H9: Apply prospect discount if receiving a minors player
            if _prospect_discount < 1.0:
                user_delta *= _prospect_discount

            # Apply closer scarcity premium if receiving a closer
            recv_sv = float(recv_player.iloc[0].get("sv", 0) or 0)
            sv_bonus = 0.0
            if recv_sv >= 5:
                from src.trade_intelligence import SV_SCARCITY_MULT

                # Add saves-specific scarcity bonus instead of multiplying entire delta
                sv_denom = config.sgp_denominators.get("SV", 1.0)
                if abs(sv_denom) > 1e-9:
                    sv_bonus = (recv_sv / sv_denom) * (SV_SCARCITY_MULT - 1.0)
                    user_delta += sv_bonus

                # Closer stability discount: reduce SV value for shaky closers
                try:
                    from src.closer_monitor import compute_job_security

                    # Default moderate hierarchy confidence
                    hierarchy_conf = 0.7
                    job_sec = compute_job_security(hierarchy_conf, recv_sv)
                    if job_sec < 0.5:
                        # Shaky closer: discount the SV bonus by (1 - job_security)
                        sv_discount = sv_bonus * (1.0 - job_sec)
                        user_delta -= sv_discount
                except Exception:
                    pass  # Keep original SV bonus if closer_monitor unavailable

            if user_delta < MIN_SGP_GAIN:
                continue

            # Opponent: lose recv_id, gain give_id
            new_opp_ids = [pid for pid in opponent_roster_ids if pid != recv_id] + [give_id]
            new_opp_totals = _roster_category_totals(new_opp_ids, player_pool)
            opp_new_sgp = _totals_sgp(new_opp_totals, config)
            opp_delta = opp_new_sgp - opp_baseline

            if opp_delta < max_opp_loss:
                continue  # Opponent loses too much

            # Need match: how much does trade fill opponent's weak categories
            # Use REAL opponent needs analysis when available
            opp_need_match = 0.5
            if opp_needs_analysis:
                opp_weak_cats = [c for c, info in opp_needs_analysis.items() if info.get("rank", 6) >= 8]
                if opp_weak_cats:
                    # Check if the given player helps opponent's weak categories
                    give_helps = sum(1 for c in opp_weak_cats if float(give_player.iloc[0].get(c.lower(), 0) or 0) > 0)
                    opp_need_match = give_helps / max(len(opp_weak_cats), 1)

            need_match = min(1.0, max(0.0, (opp_delta + 1.0) / 2.0))

            # Compute ADP fairness
            adp_fairness = compute_adp_fairness(give_id, recv_id, player_pool)

            # ECR ranking fairness: penalize trades where you receive a much
            # lower-ranked player than you give. ECR rank captures market
            # consensus that raw SGP misses (e.g., Raleigh #25 vs Arraez #124).
            ecr_fairness = 0.5  # default neutral
            give_ecr = ecr_ranks.get(give_id)
            recv_ecr = ecr_ranks.get(recv_id)
            if give_ecr and recv_ecr:
                # Ratio: closer to 1.0 = fairer. Lower rank = better.
                ecr_ratio = min(give_ecr, recv_ecr) / max(give_ecr, recv_ecr, 1)
                ecr_fairness = ecr_ratio**0.5  # sqrt softens extreme gaps

            # H5: YTD performance modifier with trade timing clamp.
            # Weeks 1-4: ±5% (too early, mostly noise).
            # Weeks 5-8: ±10% (standard).
            # Weeks 9+: ±15% (enough data to trust divergence).
            # Bonus 1.1x multiplier during prime trade window (weeks 4-10).
            ytd_modifier = 1.0
            recv_ytd = ytd_stats.get(recv_id, {})
            if recv_ytd.get("pa", 0) >= 10:
                proj_avg = float(recv_player.iloc[0].get("avg", 0.260) or 0.260)
                ytd_avg = recv_ytd.get("avg", proj_avg)
                if proj_avg > 0:
                    perf_ratio = ytd_avg / proj_avg
                    # H5: Dynamic clamp by season phase
                    _season_start = datetime(datetime.now(UTC).year, 3, 25, tzinfo=UTC)
                    _weeks_elapsed = max(1, (datetime.now(UTC) - _season_start).days // 7)
                    if _weeks_elapsed <= 4:
                        _clamp = 0.05  # ±5%
                    elif _weeks_elapsed <= 8:
                        _clamp = 0.10  # ±10%
                    else:
                        _clamp = 0.15  # ±15%
                    ytd_modifier = max(1.0 - _clamp, min(1.0 + _clamp, perf_ratio))

            # Scale YTD modifier by stat reliability — different stats stabilize
            # at different sample sizes (K% ~60 PA, HR ~170 PA, AVG ~910 PA).
            # Early-season performance changes are mostly noise.
            if recv_ytd.get("pa", 0) >= 10:
                is_hitter = int(recv_player.iloc[0].get("is_hitter", 1))
                if is_hitter:
                    # For hitters, AVG stabilizes slowest (~910 PA)
                    reliability = min(1.0, recv_ytd["pa"] / 910.0)
                else:
                    # For pitchers, ERA stabilizes around 540 BF (~180 IP)
                    ip = float(recv_player.iloc[0].get("ip", 0) or 0)
                    reliability = min(1.0, ip / 180.0)
                raw_divergence = ytd_modifier - 1.0
                ytd_modifier = 1.0 + raw_divergence * reliability

            # F2: Draft-round anchoring — look up opponent's player draft round
            recv_round = None
            try:
                from src.database import get_player_draft_round

                recv_round = get_player_draft_round(recv_id)
            except Exception:
                pass

            # F3: Disposition effect — reuse perf_ratio from YTD modifier above
            recv_ytd_ratio = ytd_modifier if ytd_modifier != 1.0 else None

            # F4: Recently-acquired penalty — check if opponent got this player via trade
            recv_recently_acq = False
            if _transactions_df is not None and not _transactions_df.empty:
                try:
                    _three_weeks_ago = (
                        datetime.now(UTC) - timedelta(weeks=3)
                    ).isoformat()
                    recent_trades = _transactions_df[
                        (_transactions_df["player_id"] == recv_id)
                        & (_transactions_df["type"] == "trade")
                        & (_transactions_df["timestamp"] >= _three_weeks_ago)
                    ]
                    recv_recently_acq = not recent_trades.empty
                except Exception:
                    pass

            p_accept = estimate_acceptance_probability(
                user_delta,
                opp_delta,
                need_match,
                adp_fairness=adp_fairness,
                opponent_need_match=opp_need_match,
                opponent_trade_willingness=opp_archetype_willingness,
                loss_aversion=loss_aversion,
                recv_draft_round=recv_round,
                recv_ytd_vs_proj=recv_ytd_ratio,
                recv_recently_acquired=recv_recently_acq,
            )

            # Acceptance floor: skip trades nobody would accept
            if p_accept < ACCEPTANCE_FLOOR:
                continue

            # Efficiency cap: reject "trade your worst for their best" nonsense
            # Use floor of 0.1 SGP so negative-value bench players can't acquire stars
            recv_raw_sgp_check = _player_sgp_volume_aware(recv_id, player_pool, config)
            if recv_raw_sgp_check > 0:
                effective_give = max(give_raw_sgp, 0.1)
                eff_ratio = recv_raw_sgp_check / effective_give
                if eff_ratio > MAX_EFFICIENCY_RATIO:
                    continue

            # Category strategic fit — does this trade address user's category needs?
            category_fit = 0.5  # neutral default
            if user_category_profile:
                try:
                    _give_contrib = _player_category_contribution(give_player.iloc[0], config)
                    _recv_contrib = _player_category_contribution(recv_player.iloc[0], config)
                    category_fit = _compute_category_fit_score(_give_contrib, _recv_contrib, user_category_profile)
                except Exception:
                    category_fit = 0.5

            # Composite score: acceptance-heavy weighting because rejected trades are worthless.
            # Normalize SGP gain to [0,1] range by capping at 5x efficiency.
            norm_sgp = min(user_delta * ytd_modifier / 3.0, 1.0) if user_delta > 0 else 0.0
            composite = (
                COMPOSITE_W_SGP * norm_sgp
                + COMPOSITE_W_ADP * adp_fairness
                + COMPOSITE_W_ECR * ecr_fairness
                + COMPOSITE_W_ACCEPT * p_accept
                + COMPOSITE_W_CAT_FIT * category_fit
                + COMPOSITE_W_OPP_NEED * opp_need_match
            )

            # Regression bonus: reward trades that exploit regression signals
            # G1: xwOBA regression (+0.03 each direction)
            # G2: Stuff+ regression (+0.02 each direction)
            # G3: BABIP regression (+0.02 each direction)
            regression_bonus = 0.0
            recv_row = recv_player.iloc[0]
            give_row = give_player.iloc[0]
            # G1: xwOBA
            if str(recv_row.get("regression_flag", "")) == "BUY_LOW":
                regression_bonus += 0.03
            if str(give_row.get("regression_flag", "")) == "SELL_HIGH":
                regression_bonus += 0.03
            # G3: BABIP (hitters — receiving unlucky, selling lucky)
            if str(recv_row.get("babip_regression_flag", "")) == "BUY_LOW":
                regression_bonus += 0.02
            if str(give_row.get("babip_regression_flag", "")) == "SELL_HIGH":
                regression_bonus += 0.02
            # G2: Stuff+ (pitchers — receiving elite stuff w/ bad luck, selling weak stuff w/ good luck)
            if str(recv_row.get("stuff_regression_flag", "")) == "BUY_LOW":
                regression_bonus += 0.02
            if str(give_row.get("stuff_regression_flag", "")) == "SELL_HIGH":
                regression_bonus += 0.02
            # G5: Velocity trend (pitchers — velo gain = buy, velo decline = sell)
            if str(recv_row.get("velo_regression_flag", "")) == "BUY_LOW":
                regression_bonus += 0.02
            if str(give_row.get("velo_regression_flag", "")) == "SELL_HIGH":
                regression_bonus += 0.02
            composite += regression_bonus

            # H6: Consistency/variance modifier for H2H.
            # In weekly H2H, consistent players are more valuable than volatile ones.
            # Use xwOBA delta and BABIP delta as volatility proxies:
            # large |delta| = outcomes diverge from expected = higher variance.
            # Receiving consistent player = bonus. Receiving volatile = penalty.
            _H6_K = 0.05  # Weight of consistency adjustment
            recv_xwoba_d = abs(float(recv_row.get("xwoba_delta", 0) or 0))
            recv_babip_d = abs(float(recv_row.get("babip_delta", 0) or 0))
            give_xwoba_d = abs(float(give_row.get("xwoba_delta", 0) or 0))
            give_babip_d = abs(float(give_row.get("babip_delta", 0) or 0))
            # Normalize: xwOBA delta ~0.030 = 1 SD, BABIP ~0.030 = 1 SD
            recv_vol = (recv_xwoba_d / 0.030 + recv_babip_d / 0.030) / 2.0
            give_vol = (give_xwoba_d / 0.030 + give_babip_d / 0.030) / 2.0
            # Positive = receiving less volatile player (good for H2H)
            consistency_adj = _H6_K * (give_vol - recv_vol)
            composite += max(-0.05, min(0.05, consistency_adj))

            trade_result: dict = {
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
                "is_closer_trade": recv_sv >= 5,
                "give_adp_round": give_round if give_round else int(give_adp) if give_adp < 500 else "Undrafted",
                "recv_adp_round": recv_round if recv_round else int(recv_adp) if recv_adp < 500 else "Undrafted",
                "adp_fairness": round(adp_fairness, 3),
                "ecr_fairness": round(ecr_fairness, 3),
                "give_ecr_rank": give_ecr or "N/A",
                "recv_ecr_rank": recv_ecr or "N/A",
                "ytd_modifier": round(ytd_modifier, 3),
                "opp_need_match": round(opp_need_match, 3),
                "category_fit": round(category_fit, 3),
                "is_cross_type": is_cross_type,
            }

            # Annotate health risk
            _hs = recv_player.iloc[0].get("health_score", None)
            health = float(_hs) if _hs is not None and not (isinstance(_hs, float) and _hs != _hs) else 0.85
            if health >= 0.85:
                trade_result["health_risk"] = "Low"
            elif health >= 0.65:
                trade_result["health_risk"] = "Moderate"
            elif health >= 0.40:
                trade_result["health_risk"] = "Elevated"
            else:
                trade_result["health_risk"] = "High"

            # Annotate FA alternative
            if fa_comparisons and recv_id in fa_comparisons:
                fa_info = fa_comparisons[recv_id]
                trade_result["fa_alternative"] = fa_info.get("has_alternative", False)
                trade_result["fa_name"] = fa_info.get("fa_name", "")
                trade_result["fa_pct"] = fa_info.get("fa_pct", 0.0)

            # Annotate IL status
            if roster_statuses and recv_id in roster_statuses:
                trade_result["recv_status"] = roster_statuses[recv_id]

            results.append(trade_result)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def _weighted_totals_sgp(
    totals: dict,
    config: LeagueConfig,
    weights: dict[str, float] | None = None,
) -> float:
    """Convert roster category totals to weighted SGP.

    When weights are provided, each category's SGP is multiplied by its
    marginal weight from the category gap analysis. This prioritizes
    categories where the user can gain standings positions.
    """
    if weights is None:
        return _totals_sgp(totals, config)

    total = 0.0
    for cat in config.all_categories:
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        w = weights.get(cat, 1.0)
        if cat in config.inverse_stats:
            total -= (val / denom) * w
        else:
            total += (val / denom) * w
    return total


def _totals_sgp(totals: dict, config: LeagueConfig) -> float:
    """Convert roster category totals to total SGP.

    WARNING: This function should ONLY be used on full-roster aggregated totals
    (from _roster_category_totals() with a full team). For rate stats (AVG, OBP,
    ERA, WHIP), it divides the raw rate value by the SGP denominator, which is
    correct when the rate is already the volume-weighted team average.

    For INDIVIDUAL player SGP, use _player_sgp_volume_aware() instead, which
    properly accounts for volume (AB/IP) via SGPCalculator.total_sgp(). A 600 AB
    .300 hitter moves team AVG 3x more than a 200 AB .300 hitter.
    """
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


# ── Category Strategic Helpers ───────────────────────────────────────


def _compute_user_category_profile(
    user_team_name: str,
    all_team_totals: dict[str, dict[str, float]],
    config,
    weeks_remaining: int = 16,
) -> dict[str, dict]:
    """Classify each scoring category as strong, weak, or punt for the user.

    Strong: rank <= 4 (top third). Weak: rank >= 8 (bottom third, not punted).
    Punt: from gap analysis (rank >= 10, no gainable positions).

    Returns {cat: {"rank": int, "is_strong": bool, "is_weak": bool, "is_punt": bool}}.
    """
    try:
        from src.engine.portfolio.category_analysis import category_gap_analysis

        user_totals = all_team_totals.get(user_team_name, {})
        if not user_totals:
            return {}

        gap_analysis = category_gap_analysis(user_totals, all_team_totals, user_team_name, weeks_remaining)

        profile = {}
        for cat, info in gap_analysis.items():
            rank = info.get("rank", 6)
            is_punt = info.get("is_punt", False)
            profile[cat] = {
                "rank": rank,
                "is_strong": rank <= 4,
                "is_weak": rank >= 8 and not is_punt,
                "is_punt": is_punt,
            }
        return profile
    except Exception:
        return {}


def _player_category_contribution(player_row, config) -> dict[str, float]:
    """Compute per-category SGP contribution for a single player.

    Returns {cat: sgp_value} for categories where the player contributes.
    Inverse stats are sign-flipped so positive = good contribution.
    """
    contributions = {}
    for cat in config.all_categories:
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = float(player_row.get(cat.lower(), player_row.get(cat, 0)) or 0)
        if abs(val) < 1e-9:
            continue
        sgp = val / denom
        if cat in config.inverse_stats:
            sgp = -sgp  # For ERA/WHIP/L, lower raw value = positive contribution
        if sgp > 0.01:
            contributions[cat] = round(sgp, 3)
    return contributions


def _compute_category_fit_score(
    give_contributions: dict[str, float],
    recv_contributions: dict[str, float],
    user_profile: dict[str, dict],
) -> float:
    """Score how well a trade addresses the user's category needs.

    0.0 = terrible fit (giving from weakness, receiving for strength).
    0.5 = neutral.
    1.0 = perfect fit (giving from strength/punt, receiving for weakness).
    """
    if not user_profile:
        return 0.5  # No profile data — neutral

    score = 0.0

    # Giving players: prefer giving from strength/punt, penalize giving from weakness
    for cat, sgp in give_contributions.items():
        info = user_profile.get(cat, user_profile.get(cat.upper(), {}))
        if not info:
            continue
        if info.get("is_punt"):
            score += 0.15 * sgp  # Giving from punt = great (free value)
        elif info.get("is_strong"):
            score += 0.10 * sgp  # Giving from strength = good
        elif info.get("is_weak"):
            score -= 0.10 * sgp  # Giving from weakness = bad

    # Receiving players: prefer receiving for weakness, discount receiving for strength
    for cat, sgp in recv_contributions.items():
        info = user_profile.get(cat, user_profile.get(cat.upper(), {}))
        if not info:
            continue
        if info.get("is_weak"):
            score += 0.15 * sgp  # Receiving for weakness = great
        elif info.get("is_strong"):
            score -= 0.05 * sgp  # Receiving for already-strong = wasteful
        # Punt categories: no bonus for receiving (useless)

    # Normalize to [0, 1] via sigmoid-like clamping
    return max(0.0, min(1.0, 0.5 + score * 0.3))


# ── Main Trade Finder ─────────────────────────────────────────────────


def find_trade_opportunities(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    user_team_name: str | None = None,
    league_rosters: dict[str, list[int]] | None = None,
    weeks_remaining: int | None = None,
    max_results: int = 20,
    top_partners: int = 5,
    standings_rank: int | None = None,
    team_record: tuple | None = None,
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
        weeks_remaining: Remaining weeks in season. If None, computed
            dynamically from today's date.
        max_results: Maximum trade opportunities to return.
        top_partners: Number of complementary teams to scan.
        standings_rank: User's rank in standings (1-12, None if unknown).
        team_record: User's W-L-T record as (wins, losses, ties), None if unknown.

    Returns:
        List of trade opportunity dicts sorted by composite_score.
    """
    if weeks_remaining is None:
        from datetime import datetime, timedelta, timezone

        _ET = timezone(timedelta(hours=-4))
        _season_start = datetime(2026, 3, 25, tzinfo=_ET)
        _now = datetime.now(_ET)
        _weeks_elapsed = max(0, (_now - _season_start).days // 7)
        weeks_remaining = max(1, 24 - _weeks_elapsed)

    if config is None:
        config = LeagueConfig()

    if player_pool.empty or not user_roster_ids:
        return []

    if not league_rosters or not all_team_totals:
        return []

    # ── Compute desperation level based on team performance ───────────
    _desperation = 0.0
    if standings_rank is not None:
        total_teams = 12
        _desperation += (standings_rank / total_teams) * 0.5  # Bottom of standings = more desperate
    if team_record is not None:
        wins, losses, ties = team_record
        total_games = wins + losses + ties
        if total_games > 0:
            _desperation += (losses / total_games) * 0.5  # More losses = more desperate
    _desperation = min(1.0, _desperation)

    # Scale trade aggressiveness with desperation
    _elite_floor = ELITE_RETURN_FLOOR - (_desperation * 0.25)  # 0.75 -> 0.50 at max desperation
    _max_weight = MAX_WEIGHT_RATIO + (_desperation * 1.0)  # 1.5 -> 2.5 at max desperation
    _loss_aversion = LOSS_AVERSION - (_desperation * 0.5)  # 1.8 -> 1.3 at max desperation
    _max_opp_loss_adj = MAX_OPP_LOSS - (_desperation * 1.0)  # -0.5 -> -1.5 at max desperation

    if _desperation > 0.3:
        logger.info(
            "Trade finder desperation mode: level=%.2f, elite_floor=%.2f, weight_ratio=%.1f",
            _desperation,
            _elite_floor,
            _max_weight,
        )

    # ── Compute trade intelligence context ──────────────────────────────
    category_weights: dict[str, float] | None = None
    fa_comparisons: dict[int, dict] = {}
    roster_statuses: dict[int, str] = {}
    user_category_profile: dict[str, dict] = {}

    if user_team_name and all_team_totals:
        try:
            from src.trade_intelligence import (
                apply_scarcity_flags,
                compute_fa_comparisons,
                get_category_weights,
                get_health_adjusted_pool,
            )

            # Health-adjust the pool
            player_pool = get_health_adjusted_pool(player_pool, config)

            # Add scarcity flags
            player_pool = apply_scarcity_flags(player_pool)

            # Category weights — prefer unified service, fall back to direct call
            try:
                from src.matchup_context import get_matchup_context

                _mctx = get_matchup_context()
                category_weights = _mctx.get_category_weights(mode="standings")
            except Exception:
                category_weights = get_category_weights(user_team_name, all_team_totals, config, weeks_remaining)

            # Compute user category profile for strategic fit scoring
            user_category_profile = _compute_user_category_profile(
                user_team_name, all_team_totals, config, weeks_remaining
            )

            # Load roster statuses
            try:
                from src.trade_intelligence import _load_roster_statuses

                roster_statuses = _load_roster_statuses()
            except Exception:
                pass

            # Compute FA comparisons for all opponent players
            try:
                from src.league_manager import get_free_agents as _get_fa

                fa_pool = _get_fa(player_pool)
                all_opp_ids = []
                for tn, pids in league_rosters.items():
                    if tn != user_team_name:
                        all_opp_ids.extend(pids)
                fa_comparisons = compute_fa_comparisons(all_opp_ids, user_roster_ids, fa_pool, player_pool, config)
            except Exception:
                pass

        except ImportError:
            pass  # trade_intelligence not available — run without enhancements

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

        trades = scan_1_for_1(
            user_roster_ids,
            opp_roster,
            player_pool,
            config,
            category_weights=category_weights,
            fa_comparisons=fa_comparisons,
            roster_statuses=roster_statuses,
            opponent_team_name=opp_team,
            all_team_totals=all_team_totals,
            user_category_profile=user_category_profile,
            elite_return_floor=_elite_floor,
            max_weight_ratio=_max_weight,
            loss_aversion=_loss_aversion,
            max_opp_loss=_max_opp_loss_adj,
        )

        for trade in trades:
            trade["opponent_team"] = opp_team
            trade["complementarity"] = round(dissim_score, 3)

        all_trades.extend(trades)

    # ── Tier 2: Multi-player expansion from top seeds ────────────────
    if len(all_trades) >= 5:
        top_seeds = sorted(
            all_trades,
            key=lambda t: t.get("composite_score", 0),
            reverse=True,
        )[:15]

        for opp_team, dissim_score in partners:
            opp_roster = league_rosters.get(opp_team, [])
            if not opp_roster:
                continue

            opp_seeds = [t for t in top_seeds if t.get("opponent_team") == opp_team]
            if not opp_seeds:
                continue

            try:
                multi = scan_2_for_1(
                    opp_seeds[:5],
                    user_roster_ids,
                    opp_roster,
                    player_pool,
                    config,
                    category_weights=category_weights,
                    max_expansions=30,
                    user_category_profile=user_category_profile,
                    loss_aversion=_loss_aversion,
                    max_opp_loss=_max_opp_loss_adj,
                )
                for trade in multi:
                    trade["opponent_team"] = opp_team
                    trade["complementarity"] = round(dissim_score, 3)
                all_trades.extend(multi)
            except Exception:
                logger.debug(
                    "Multi-player expansion failed for %s",
                    opp_team,
                    exc_info=True,
                )

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

    # Ensure both 1-for-1 and multi-player trades are represented in results.
    # Reserve at least half the slots for 1-for-1 trades so they don't get
    # crowded out by 2-for-1 trades with inflated composite scores.
    one_for_one = [t for t in unique_trades if t.get("trade_type") == "1-for-1"]
    multi_player = [t for t in unique_trades if t.get("trade_type") != "1-for-1"]

    half = max_results // 2
    selected_1v1 = one_for_one[:half]
    remaining_slots = max_results - len(selected_1v1)
    selected_multi = multi_player[:remaining_slots]

    combined = selected_1v1 + selected_multi
    combined.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return combined[:max_results]
