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

import numpy as np
import pandas as pd

from src.in_season import _roster_category_totals
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

MIN_SGP_GAIN = 0.3  # Minimum user SGP gain to surface a trade
MAX_OPP_LOSS = -0.5  # Reject trades where opponent loses more than this (e.g., opp_delta < -0.5)
LOSS_AVERSION = 1.8  # was 1.5 -- meta-analysis consensus (Brown 2024, Walasek 2024)


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
    adp_fairness: float = 0.5,
    opponent_need_match: float = 0.5,
    opponent_standings_rank: int | None = None,
    opponent_trade_willingness: float = 0.5,
) -> float:
    """Estimate probability that opponent accepts the trade.

    Uses behavioral model: loss aversion, fairness gap, need matching,
    ADP fairness, opponent need alignment, standings position, and
    trade willingness archetype.

    Args:
        user_gain_sgp: How much YOU gain (SGP).
        opponent_gain_sgp: How much opponent gains (SGP, negative = they lose).
        need_match_score: How well the trade fills opponent needs (0-1).
        adp_fairness: ADP match between traded players (0-1, 1 = perfect).
        opponent_need_match: How well trade fills opponent category gaps (0-1).
        opponent_standings_rank: Opponent rank in standings (1-12, None if unknown).
        opponent_trade_willingness: Archetype willingness to trade (0-1).

    Returns:
        float in [0, 1] -- estimated acceptance probability.
    """
    # Fairness gap (adjusted for loss aversion per Kahneman & Tversky):
    # Losses feel LOSS_AVERSION x worse. If opponent loses, multiply loss magnitude.
    if opponent_gain_sgp < 0:
        perceived_opp_gain = opponent_gain_sgp * LOSS_AVERSION  # Losses feel worse
    else:
        perceived_opp_gain = opponent_gain_sgp

    # Sigmoid: P(accept) = 1 / (1 + exp(exponent))
    fairness_gap = abs(user_gain_sgp - perceived_opp_gain)

    # ADP penalty: strong penalty when opponent gives up a higher-drafted player
    adp_penalty = max(0, (0.5 - adp_fairness) * 3.0)

    # Bubble team bonus: teams ranked 4-8 are fighting and more willing to trade
    bubble_bonus = 0.0
    if opponent_standings_rank and 4 <= opponent_standings_rank <= 8:
        bubble_bonus = 0.4

    exponent = (
        2.0 * fairness_gap
        - 1.5 * need_match_score
        - 0.5 * max(opponent_gain_sgp, 0)
        - 1.0 * opponent_need_match
        + adp_penalty
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

# Roster spot value: SGP bonus for gaining an open slot (2-for-1) or
# penalty for losing one (1-for-2). Decays over the season.
ROSTER_SPOT_SGP = 0.8


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

    # Quick SGP per player
    sgps: dict[int, float] = {}
    for pid in all_ids:
        p = player_pool[player_pool["player_id"] == pid]
        if not p.empty:
            total = 0.0
            for cat in config.all_categories:
                val = float(p.iloc[0].get(cat.lower(), 0) or 0)
                denom = config.sgp_denominators.get(cat, 1.0)
                if abs(denom) > 1e-9:
                    total += val / denom if cat not in config.inverse_stats else -val / denom
            sgps[pid] = total

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
) -> list[dict]:
    """Expand top 1-for-1 seeds by adding a second give player.

    User gives 2 players, receives 1. User gains a roster spot.
    Opponent receives 2, must drop their worst bench player.
    """
    if config is None:
        config = LeagueConfig()

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

            new_give = orig_give + [add_id]
            new_recv = list(orig_recv)

            # Compute new SGP delta for user
            new_user_ids = [pid for pid in user_roster_ids if pid not in new_give] + new_recv
            new_user_totals = _roster_category_totals(new_user_ids, player_pool)
            user_baseline = _roster_category_totals(user_roster_ids, player_pool)

            user_new_sgp = _weighted_totals_sgp(new_user_totals, config, category_weights)
            user_base_sgp = _weighted_totals_sgp(user_baseline, config, category_weights)
            user_delta = user_new_sgp - user_base_sgp

            # Add roster spot bonus (user gains a slot)
            user_delta += ROSTER_SPOT_SGP

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

            if opp_delta < MAX_OPP_LOSS:
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
            p_accept = estimate_acceptance_probability(user_delta, opp_delta, need_match, adp_fairness=adp_fairness)

            # Remove roster spot bonus from composite to match 1-for-1 scale
            user_delta_for_score = user_delta - ROSTER_SPOT_SGP
            composite = (
                0.30 * user_delta_for_score
                + 0.15 * adp_fairness * 2.0
                + 0.15 * 0.5 * 2.0  # ECR neutral (not computed for multi-player)
                + 0.20 * p_accept * 3.0
                + 0.10 * max(opp_delta, 0)
                + 0.10 * need_match * 2.0
            ) + 0.1  # Small fixed bonus for roster flexibility

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
    user_totals = _roster_category_totals(user_roster_ids, player_pool)
    opp_totals = _roster_category_totals(opponent_roster_ids, player_pool)

    opp_baseline = _totals_sgp(opp_totals, config)

    # --- Cap extreme category weights ---
    # Prevent a single weak category from dominating all trade valuations.
    # Without this, being 12th in SB makes speed guys "outvalue" elite power bats.
    MAX_WEIGHT_RATIO = (
        1.5  # No category can be weighted more than 1.5x average (trade decisions are season-long, not weekly)
    )
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
    ELITE_RETURN_FLOOR = 0.75  # Return must be >= 75% of given player's raw SGP

    user_raw_sgps: dict[int, float] = {}
    for pid in user_roster_ids:
        p = player_pool[player_pool["player_id"] == pid]
        if not p.empty:
            user_raw_sgps[pid] = _totals_sgp(_roster_category_totals([pid], player_pool), config)
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

    results = []

    for give_id in user_roster_ids:
        give_player = player_pool[player_pool["player_id"] == give_id]
        if give_player.empty:
            continue
        give_name = give_player.iloc[0].get("name", give_player.iloc[0].get("player_name", "?"))
        give_raw_sgp = user_raw_sgps.get(give_id, 0.0)

        for recv_id in opponent_roster_ids:
            # Filter NA/minors players
            if roster_statuses:
                recv_status = str(roster_statuses.get(recv_id, "active")).lower()
                if recv_status in ("na", "not active", "minors"):
                    continue

            recv_player = player_pool[player_pool["player_id"] == recv_id]
            if recv_player.empty:
                continue

            # Same type check (hitter for hitter, pitcher for pitcher)
            give_is_hitter = int(give_player.iloc[0].get("is_hitter", 0))
            recv_is_hitter = int(recv_player.iloc[0].get("is_hitter", 0))
            if give_is_hitter != recv_is_hitter:
                continue  # Skip cross-type trades for simplicity

            # --- Elite player protection ---
            # If giving away a top-20% player, the return must be at least 75% as good
            if give_raw_sgp >= elite_threshold:
                recv_raw_sgp = _totals_sgp(_roster_category_totals([recv_id], player_pool), config)
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

            # Apply closer scarcity premium if receiving a closer
            recv_sv = float(recv_player.iloc[0].get("sv", 0) or 0)
            if recv_sv >= 5:
                from src.trade_intelligence import SV_SCARCITY_MULT

                # Add saves-specific scarcity bonus instead of multiplying entire delta
                sv_denom = config.sgp_denominators.get("SV", 1.0)
                if abs(sv_denom) > 1e-9:
                    sv_bonus = (recv_sv / sv_denom) * (SV_SCARCITY_MULT - 1.0)
                    user_delta += sv_bonus

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

            # YTD performance modifier: slight bonus for hot players, slight
            # penalty for cold players. Capped at +/- 10% to avoid chasing streaks.
            ytd_modifier = 1.0
            recv_ytd = ytd_stats.get(recv_id, {})
            if recv_ytd.get("pa", 0) >= 10:
                # Compare YTD AVG to projected AVG
                proj_avg = float(recv_player.iloc[0].get("avg", 0.260) or 0.260)
                ytd_avg = recv_ytd.get("avg", proj_avg)
                if proj_avg > 0:
                    perf_ratio = ytd_avg / proj_avg
                    # Clamp to [0.90, 1.10] — max 10% adjustment
                    ytd_modifier = max(0.90, min(1.10, perf_ratio))

            p_accept = estimate_acceptance_probability(
                user_delta,
                opp_delta,
                need_match,
                adp_fairness=adp_fairness,
                opponent_need_match=opp_need_match,
                opponent_trade_willingness=opp_archetype_willingness,
            )

            # Composite score: SGP gain * YTD modifier, weighted by acceptance,
            # ADP fairness, ECR fairness, opponent benefit, and need matching
            composite = (
                0.30 * user_delta * ytd_modifier
                + 0.15 * adp_fairness * 2.0
                + 0.15 * ecr_fairness * 2.0  # ECR ranking fairness
                + 0.20 * p_accept * 3.0
                + 0.10 * max(opp_delta, 0)
                + 0.10 * need_match * 2.0
            )

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

    # ── Compute trade intelligence context ──────────────────────────────
    category_weights: dict[str, float] | None = None
    fa_comparisons: dict[int, dict] = {}
    roster_statuses: dict[int, str] = {}

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

            # Category weights from gap analysis
            category_weights = get_category_weights(user_team_name, all_team_totals, config, weeks_remaining)

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
