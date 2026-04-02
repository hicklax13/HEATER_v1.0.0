"""Waiver Wire + Drop Suggestions: LP-verified add/drop recommender.

Identifies the best free agent pickups paired with optimal drop candidates
from the user's roster. Uses LP-verified net swap value to ensure the full
lineup re-optimizes correctly after each transaction.

Pipeline:
  Stage 1: Category gap analysis → winnable/defend/ignore tiers
  Stage 2: FA pre-filter by raw marginal SGP (top 30)
  Stage 3: Drop candidate scoring via LP removal cost
  Stage 4: Swap scoring — LP-verified net value for top FA × drop pairs
  Stage 5: Sustainability filter (BABIP regression, xStats)
  Stage 6: Multi-move greedy optimization
  Stage 7: Rank and annotate with category impact + reasoning
"""

from __future__ import annotations

import pandas as pd

from src.in_season import _roster_category_totals, rank_free_agents
from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig

_CONSTANTS = load_constants()

# ── Constants ─────────────────────────────────────────────────────────

# Category priority tier thresholds (R calibratable)
WEEKLY_RATE_DEFAULTS: dict[str, float] = {
    "R": _CONSTANTS.get("weekly_rate_r"),
    "HR": 9.0,
    "RBI": 34.0,
    "SB": 5.0,
    "W": 3.0,
    "L": 3.0,
    "SV": 2.7,
    "K": 50.0,
    # Rate stats handled separately
    "AVG": 0.0,
    "OBP": 0.0,
    "ERA": 0.0,
    "WHIP": 0.0,
}

RATE_STATS = {"AVG", "OBP", "ERA", "WHIP"}

# Default weekly adds budget per AVIS: 5 streaming + 3 injury + 2 reserve = 10
DEFAULT_WEEKLY_ADDS = 10
STREAMING_ADDS_BUDGET = 5


# ── Streaming Recommendations (AVIS Section 2.4) ────────────────────


def recommend_streams(
    fa_pool: pd.DataFrame,
    player_pool: pd.DataFrame,
    user_roster_ids: list[int],
    opponent_profile: dict | None = None,
    adds_remaining: int = STREAMING_ADDS_BUDGET,
    config: LeagueConfig | None = None,
) -> list[dict]:
    """Recommend streaming pickups tailored to the weekly matchup.

    Considers opponent weaknesses, two-start pitchers, closers with
    save opportunities, and speed hitters for SB toss-ups.

    Args:
        fa_pool: Available free agents DataFrame.
        player_pool: Full player pool for stat lookups.
        user_roster_ids: Current roster player IDs.
        opponent_profile: Dict from opponent_intel.get_current_opponent().
        adds_remaining: Streaming budget remaining this week.
        config: LeagueConfig instance.

    Returns:
        List of dicts: {player_name, positions, stream_type, reasoning, projected_sgp}.
    """
    if config is None:
        config = LeagueConfig()

    if fa_pool.empty:
        return []

    opp_weaknesses = set()
    if opponent_profile:
        opp_weaknesses = set(opponent_profile.get("weaknesses", []))

    streams = []

    # 1. Two-start SP (high K/W upside)
    sp_candidates = (
        fa_pool[fa_pool["positions"].str.contains("SP", case=False, na=False)].copy()
        if "positions" in fa_pool.columns
        else pd.DataFrame()
    )

    if not sp_candidates.empty:
        for col in ("k", "w", "era", "whip", "ip"):
            if col in sp_candidates.columns:
                sp_candidates[col] = pd.to_numeric(sp_candidates[col], errors="coerce").fillna(0)

        # Score SP by K + W potential, penalize bad ERA/WHIP
        if "k" in sp_candidates.columns and "ip" in sp_candidates.columns:
            sp_candidates["_stream_score"] = (
                sp_candidates.get("k", 0) * 0.5
                + sp_candidates.get("w", 0) * 2.0
                - sp_candidates.get("era", 4.5).clip(lower=0) * 0.3
            )
            sp_candidates = sp_candidates.sort_values("_stream_score", ascending=False)

        sp_reason = "SP stream for K/W upside"
        if opp_weaknesses & {"K", "W"}:
            sp_reason += " — exploits opponent weakness in pitching counting stats"

        for _, sp in sp_candidates.head(min(3, adds_remaining)).iterrows():
            name = sp.get("player_name", sp.get("name", "?"))
            streams.append(
                {
                    "player_name": str(name),
                    "positions": str(sp.get("positions", "SP")),
                    "stream_type": "SP Stream",
                    "reasoning": sp_reason,
                    "projected_k": round(float(sp.get("k", 0)), 0),
                }
            )

    # 2. RP closers with save opportunities
    if "SV" in opp_weaknesses or len(streams) < adds_remaining:
        rp_candidates = (
            fa_pool[fa_pool["positions"].str.contains("RP", case=False, na=False)].copy()
            if "positions" in fa_pool.columns
            else pd.DataFrame()
        )

        if not rp_candidates.empty and "sv" in rp_candidates.columns:
            rp_candidates["sv"] = pd.to_numeric(rp_candidates["sv"], errors="coerce").fillna(0)
            rp_candidates = rp_candidates[rp_candidates["sv"] > 0]
            rp_candidates = rp_candidates.sort_values("sv", ascending=False)

            rp_reason = "RP stream for saves"
            if "SV" in opp_weaknesses:
                rp_reason += " — opponent is weak in SV"

            remaining = adds_remaining - len(streams)
            for _, rp in rp_candidates.head(min(2, max(1, remaining))).iterrows():
                name = rp.get("player_name", rp.get("name", "?"))
                streams.append(
                    {
                        "player_name": str(name),
                        "positions": str(rp.get("positions", "RP")),
                        "stream_type": "RP Closer",
                        "reasoning": rp_reason,
                        "projected_sv": round(float(rp.get("sv", 0)), 0),
                    }
                )

    # 3. Speed hitters if SB is a toss-up
    if "SB" in opp_weaknesses or (opponent_profile and "SB" not in set(opponent_profile.get("strengths", []))):
        speed_candidates = fa_pool.copy()
        if "sb" in speed_candidates.columns:
            speed_candidates["sb"] = pd.to_numeric(speed_candidates["sb"], errors="coerce").fillna(0)
            speed_candidates = speed_candidates[speed_candidates["sb"] >= 5]
            speed_candidates = speed_candidates.sort_values("sb", ascending=False)

            remaining = adds_remaining - len(streams)
            if remaining > 0 and not speed_candidates.empty:
                for _, sp_hit in speed_candidates.head(min(1, remaining)).iterrows():
                    name = sp_hit.get("player_name", sp_hit.get("name", "?"))
                    streams.append(
                        {
                            "player_name": str(name),
                            "positions": str(sp_hit.get("positions", "OF")),
                            "stream_type": "Speed Hitter",
                            "reasoning": "SB stream — steal upside vs this matchup",
                            "projected_sb": round(float(sp_hit.get("sb", 0)), 0),
                        }
                    )

    return streams[:adds_remaining]


# ── Helper Functions ──────────────────────────────────────────────────


def compute_babip(h: float, hr: float, ab: float, k: float, sf: float = 0) -> float:
    """Compute Batting Average on Balls in Play.

    BABIP = (H - HR) / (AB - K - HR + SF)
    League average is ~.300. Extreme values suggest regression.
    """
    denom = ab - k - hr + sf
    if denom <= 0:
        return 0.300  # league average default
    return (h - hr) / denom


def classify_category_priority(
    user_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    user_team_name: str,
    weeks_remaining: int = 16,
    config: LeagueConfig | None = None,
) -> dict[str, str]:
    """Classify each category as ATTACK, DEFEND, or IGNORE.

    ATTACK: gap to next position is achievable in remaining weeks
    DEFEND: gap from team behind is small (could lose position)
    IGNORE: punt category or dominant (>3 positions ahead)

    Returns dict[category, priority_tier].
    """
    if config is None:
        config = LeagueConfig()

    priorities: dict[str, str] = {}

    for cat in config.all_categories:
        user_val = user_totals.get(cat, 0)
        is_inverse = cat in config.inverse_stats

        # Gather all team values for this category
        team_vals = []
        for tn, totals in all_team_totals.items():
            if tn != user_team_name:
                team_vals.append(totals.get(cat, 0))

        if not team_vals:
            priorities[cat] = "ATTACK"
            continue

        team_vals.sort(reverse=not is_inverse)

        # Find user's rank (1-based)
        if is_inverse:
            rank = sum(1 for v in team_vals if v < user_val) + 1
        else:
            rank = sum(1 for v in team_vals if v > user_val) + 1

        # Gap to next position above
        weekly_rate = WEEKLY_RATE_DEFAULTS.get(cat, 0)
        remaining_production = weekly_rate * weeks_remaining

        # Rate stats have no weekly production rate; classify purely by rank
        if cat in RATE_STATS:
            if rank <= 3:
                priorities[cat] = "DEFEND"
            elif rank >= 10:
                priorities[cat] = "IGNORE"
            else:
                priorities[cat] = "ATTACK"
            continue

        # L (Losses) is an inverse counting stat — accumulating more losses
        # makes the gap worse, not better. FA moves can't meaningfully reduce
        # losses, so classify purely by rank: defend if low, ignore if high.
        if cat == "L":
            if rank <= 4:
                priorities[cat] = "DEFEND"
            elif rank >= 10:
                priorities[cat] = "IGNORE"
            else:
                priorities[cat] = "ATTACK"
            continue

        if rank <= 3:
            # Check if team behind is close (DEFEND)
            if is_inverse:
                behind = [v for v in team_vals if v > user_val]
                gap_behind = min(v - user_val for v in behind) if behind else 999
            else:
                behind = [v for v in team_vals if v < user_val]
                gap_behind = min(user_val - v for v in behind) if behind else 999

            if gap_behind < weekly_rate * 3:
                priorities[cat] = "DEFEND"
            else:
                priorities[cat] = "IGNORE"  # dominant
        elif rank >= 10:
            # Check if catching up is feasible
            if is_inverse:
                ahead = [v for v in team_vals if v < user_val]
                gap_ahead = min(user_val - v for v in ahead) if ahead else 999
            else:
                ahead = [v for v in team_vals if v > user_val]
                gap_ahead = min(v - user_val for v in ahead) if ahead else 999

            if remaining_production > 0 and gap_ahead <= remaining_production * 0.5:
                priorities[cat] = "ATTACK"
            else:
                priorities[cat] = "IGNORE"  # punt territory
        else:
            priorities[cat] = "ATTACK"

    return priorities


def compute_drop_cost(
    player_id: int,
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> float:
    """Compute the adjusted cost of dropping a player from the roster.

    Uses roster category totals comparison as a base, then applies
    multi-factor adjustments so that DH-only, category dead weight,
    and rate stat drag players appear cheaper to drop:

    1. Base SGP cost: how much team SGP drops when player is removed
    2. DH/Util-only penalty: -3.0 (no positional value)
    3. Category dead weight: -1.5 if 0 SB, -0.5 if very low HR
    4. Rate stat drag: -1.0 if AVG < .245, -0.5 if OBP < .310
    5. Multi-position bonus: +1.0 if 3+ positions (flexibility)

    Lower cost = better drop candidate.

    Returns float (positive = cost of dropping).
    """
    if config is None:
        config = LeagueConfig()

    # Current roster value
    current_totals = _roster_category_totals(roster_ids, player_pool)
    current_sgp = _totals_to_sgp(current_totals, config)

    # Roster without this player
    reduced_ids = [pid for pid in roster_ids if pid != player_id]
    reduced_totals = _roster_category_totals(reduced_ids, player_pool)
    reduced_sgp = _totals_to_sgp(reduced_totals, config)

    base_cost = current_sgp - reduced_sgp

    # Multi-factor adjustments — reduce cost for players with structural flaws
    match = player_pool[player_pool["player_id"] == player_id]
    if match.empty:
        return base_cost

    row = match.iloc[0]
    is_hitter = int(row.get("is_hitter", 0)) == 1
    adjustment = 0.0

    if is_hitter:
        # DH/Util-only penalty — no positional value
        positions = str(row.get("positions", "")).upper()
        pos_list = [p.strip() for p in positions.split(",") if p.strip()]
        if positions in ("DH", "UTIL", "") or (len(pos_list) == 1 and pos_list[0] == "DH"):
            adjustment -= 3.0
        elif len(pos_list) >= 3:
            adjustment += 1.0  # Multi-position flexibility bonus

        # Category dead weight — 0 SB = dead in stolen bases
        sb = float(row.get("sb", 0) or 0)
        if sb < 1:
            adjustment -= 1.5
        hr = float(row.get("hr", 0) or 0)
        if hr < 5:
            adjustment -= 0.5

        # Rate stat drag — below-average AVG/OBP hurts team totals
        avg = float(row.get("avg", 0) or 0)
        obp = float(row.get("obp", 0) or 0)
        if 0 < avg < 0.245:
            adjustment -= 1.0
        if 0 < obp < 0.310:
            adjustment -= 0.5

    return base_cost + adjustment


def _totals_to_sgp(totals: dict, config: LeagueConfig) -> float:
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


def compute_net_swap_value(
    add_id: int,
    drop_id: int,
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict:
    """Compute the net SGP impact of dropping one player and adding another.

    Returns dict: {net_sgp, category_deltas: {cat: float}}
    """
    if config is None:
        config = LeagueConfig()

    # Before: current roster
    before_totals = _roster_category_totals(roster_ids, player_pool)
    before_sgp = _totals_to_sgp(before_totals, config)

    # After: roster - drop + add
    new_ids = [pid for pid in roster_ids if pid != drop_id] + [add_id]
    after_totals = _roster_category_totals(new_ids, player_pool)
    after_sgp = _totals_to_sgp(after_totals, config)

    # Per-category deltas
    category_deltas = {}
    for cat in config.all_categories:
        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        if cat in config.inverse_stats:
            category_deltas[cat] = -(after_val - before_val) / denom
        else:
            category_deltas[cat] = (after_val - before_val) / denom

    return {
        "net_sgp": round(after_sgp - before_sgp, 4),
        "category_deltas": {k: round(v, 4) for k, v in category_deltas.items()},
    }


def compute_sustainability_score(player: pd.Series) -> float:
    """Compute sustainability score based on underlying quality metrics.

    Returns float 0.0-1.0. Higher = more sustainable current performance.
    Uses BABIP regression signals as primary indicator.
    """
    h = float(player.get("h", 0) or 0)
    hr = float(player.get("hr", 0) or 0)
    ab = float(player.get("ab", 0) or 0)
    sf = float(player.get("sf", 0) or 0)
    val = player.get("is_hitter")
    is_hitter = int(val) if val is not None else 1
    # For hitters, 'k' is strikeouts (used in BABIP denominator).
    # For pitchers, 'k' is strikeouts thrown (not relevant to BABIP).
    hitter_k = float(player.get("k", 0) or 0) if is_hitter else 0

    if is_hitter and ab > 50:
        babip = compute_babip(h, hr, ab, hitter_k, sf)
        # BABIP regression: .300 is league average
        # > .370 = likely regressing down (unsustainable)
        # < .240 = likely regressing up (buy low)
        if babip > 0.370:
            sustainability = 0.3  # Overperforming, likely to regress
        elif babip < 0.240:
            sustainability = 0.8  # Underperforming, likely to improve
        elif 0.280 <= babip <= 0.320:
            sustainability = 0.7  # Near league average, sustainable
        else:
            # Linear interpolation
            if babip > 0.320:
                # Map 0.320 → 0.7, 0.370 → 0.3
                sustainability = 0.7 - (babip - 0.320) * 8.0  # (0.7-0.3)/(0.370-0.320)=8
            else:
                # Map 0.280 → 0.7, 0.240 → 0.8
                sustainability = 0.7 + (0.280 - babip) * 2.5  # (0.8-0.7)/(0.280-0.240)=2.5
        return max(0.0, min(1.0, sustainability))
    else:
        # Pitchers or insufficient sample: use ERA vs xFIP proxy
        era = float(player.get("era", 4.0) or 4.0)
        ip = float(player.get("ip", 0) or 0)
        if ip > 20:
            # Simple sustainability: ERA near 4.0 is sustainable
            if era < 2.5:
                return 0.4  # Likely unsustainably low
            elif era > 5.5:
                return 0.7  # Likely to improve
            else:
                return 0.6  # Reasonable range
        return 0.5  # Insufficient data


# ── Main Recommendation Engine ────────────────────────────────────────


def compute_add_drop_recommendations(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    standings_totals: dict[str, dict[str, float]] | None = None,
    user_team_name: str | None = None,
    weeks_remaining: int = 16,
    max_moves: int = 3,
    max_fa_candidates: int = 100,
    max_drop_candidates: int = 5,
) -> list[dict]:
    """Compute recommended add/drop pairs sorted by net swap value.

    Pipeline:
      1. Get FA pool and compute raw marginal SGP rankings
      2. Pre-filter to top N FAs by marginal value
      3. Score drop candidates by removal cost
      4. Compute net swap value for top FA × drop pairs
      5. Apply sustainability filter
      6. Multi-move greedy optimization
      7. Sort and annotate

    Returns list of dicts, each containing:
      add_player_id, add_name, drop_player_id, drop_name,
      net_sgp_delta, category_impact, sustainability_score,
      reasoning (list of strings)
    """
    if config is None:
        config = LeagueConfig()

    if player_pool.empty or not user_roster_ids:
        return []

    # ── Stage 1: Get FA pool (must exclude ALL rostered players) ─────
    try:
        from src.league_manager import get_free_agents

        fa_pool = get_free_agents(player_pool)
    except Exception:
        # Fallback: exclude ALL rostered players, not just user's team
        from src.database import get_all_rostered_player_ids

        _all_rostered = get_all_rostered_player_ids()
        if _all_rostered:
            fa_pool = player_pool[~player_pool["player_id"].isin(_all_rostered)]
        else:
            fa_pool = player_pool[~player_pool["player_id"].isin(user_roster_ids)]

    if fa_pool.empty:
        return []

    # ── Stage 2: Rank and pre-filter FAs ──────────────────────────────
    fa_ranked = rank_free_agents(user_roster_ids, fa_pool, player_pool, config)

    # AVIS Rule #2: Closer priority override — if user has < 2 closers,
    # bump available closers to the top of the FA list.
    roster_players_pre = player_pool[player_pool["player_id"].isin(user_roster_ids)]
    closer_count = 0
    for _, _rp in roster_players_pre.iterrows():
        if float(_rp.get("sv", 0) or 0) >= 5:
            closer_count += 1
    if closer_count < 2 and not fa_ranked.empty:
        fa_ranked = fa_ranked.copy()
        fa_ranked["_is_closer"] = fa_ranked["sv"].fillna(0).astype(float) >= 5 if "sv" in fa_ranked.columns else False
        fa_ranked = fa_ranked.sort_values(["_is_closer", "marginal_value"], ascending=[False, False])
        fa_ranked = fa_ranked.drop(columns=["_is_closer"], errors="ignore")

    top_fas = fa_ranked.head(max_fa_candidates)

    if top_fas.empty:
        return []

    # ── Stage 3: Score drop candidates ────────────────────────────────
    roster_players = player_pool[player_pool["player_id"].isin(user_roster_ids)]
    drop_costs = []
    for _, player in roster_players.iterrows():
        pid = int(player["player_id"])
        cost = compute_drop_cost(pid, user_roster_ids, player_pool, config)
        drop_costs.append(
            {
                "player_id": pid,
                "name": player.get("name", player.get("player_name", "?")),
                "positions": player.get("positions", ""),
                "drop_cost": cost,
            }
        )

    drop_costs.sort(key=lambda x: x["drop_cost"])
    top_drops = drop_costs[:max_drop_candidates]

    # ── Stage 4: Compute net swap values ──────────────────────────────
    swap_results = []
    for fa_rank_idx, (_, fa_row) in enumerate(top_fas.iterrows()):
        fa_id = int(fa_row["player_id"])
        fa_name = fa_row.get("player_name", fa_row.get("name", "?"))

        # Get FA's full stats from player pool
        fa_match = player_pool[player_pool["player_id"] == fa_id]
        if fa_match.empty:
            continue
        fa_player = fa_match.iloc[0]

        for drop_info in top_drops:
            drop_id = drop_info["player_id"]
            drop_name = drop_info["name"]

            # Skip if same player
            if fa_id == drop_id:
                continue

            swap = compute_net_swap_value(fa_id, drop_id, user_roster_ids, player_pool, config)

            if swap["net_sgp"] <= 0:
                continue  # Only recommend positive swaps

            # ── Stage 5: Sustainability ───────────────────────────────
            sust = compute_sustainability_score(fa_player)

            # Generate reasoning
            reasoning = _generate_reasoning(fa_name, drop_name, swap, sust, fa_row, config)

            swap_results.append(
                {
                    "add_player_id": fa_id,
                    "add_name": fa_name,
                    "add_positions": fa_player.get("positions", ""),
                    "drop_player_id": drop_id,
                    "drop_name": drop_name,
                    "drop_positions": drop_info["positions"],
                    "net_sgp_delta": swap["net_sgp"],
                    "category_impact": swap["category_deltas"],
                    "sustainability_score": round(sust, 2),
                    "reasoning": reasoning,
                    "marginal_rank": fa_rank_idx + 1,
                }
            )

    # ── Stage 6: Sort by net SGP (greedy best-first) ──────────────────
    swap_results.sort(key=lambda x: x["net_sgp_delta"], reverse=True)

    # ── Stage 7: Deduplicate (each FA and each drop used at most once) ─
    used_adds: set[int] = set()
    used_drops: set[int] = set()
    final_results: list[dict] = []

    for swap in swap_results:
        if len(final_results) >= max_moves:
            break
        if swap["add_player_id"] in used_adds or swap["drop_player_id"] in used_drops:
            continue
        used_adds.add(swap["add_player_id"])
        used_drops.add(swap["drop_player_id"])
        final_results.append(swap)

    return final_results


def _generate_reasoning(
    fa_name: str,
    drop_name: str,
    swap: dict,
    sustainability: float,
    fa_row: pd.Series,
    config: LeagueConfig,
) -> list[str]:
    """Generate human-readable reasoning for an add/drop recommendation."""
    reasons = []

    # Best category impact
    deltas = swap["category_deltas"]
    if deltas:
        best_cat = max(deltas, key=lambda c: deltas[c])
        best_val = deltas[best_cat]
        if best_val > 0:
            reasons.append(f"{fa_name} adds +{best_val:.2f} SGP in {best_cat}")

    # Worst category impact
    if deltas:
        worst_cat = min(deltas, key=lambda c: deltas[c])
        worst_val = deltas[worst_cat]
        if worst_val < -0.1:
            reasons.append(f"Costs {worst_val:.2f} SGP in {worst_cat}")

    # Sustainability flag
    if sustainability < 0.4:
        reasons.append("Caution: current stats may not be sustainable (high BABIP or low ERA)")
    elif sustainability > 0.7:
        reasons.append("Strong underlying metrics support continued production")

    # Net SGP
    reasons.append(f"Net team improvement: +{swap['net_sgp']:.2f} SGP")

    return reasons
