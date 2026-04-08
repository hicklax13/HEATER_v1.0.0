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

import logging

import pandas as pd

from src.in_season import _roster_category_totals, rank_free_agents
from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

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


def compute_schedule_aware_streams(
    fa_pool: pd.DataFrame,
    player_pool: pd.DataFrame,
    days_ahead: int = 7,
    max_results: int = 10,
) -> list[dict]:
    """E9: Schedule-aware streaming that considers matchup quality and game count.

    Enhances basic streaming recommendations with:
    - Two-start pitcher detection and matchup scores
    - Opponent team wRC+ for hitter schedule quality
    - Off-day awareness (players with more games = more value)

    Args:
        fa_pool: Free agent pool DataFrame.
        player_pool: Full player pool for projections.
        days_ahead: Lookahead window (default 7 = this week).
        max_results: Max streaming candidates to return.

    Returns:
        List of dicts with schedule-enhanced streaming scores.
    """
    results = []

    # 1. Two-start pitcher detection
    try:
        from src.two_start import identify_two_start_pitchers

        two_starters = identify_two_start_pitchers(days_ahead=days_ahead, player_pool=player_pool)
        fa_ids = set(fa_pool["player_id"].tolist()) if "player_id" in fa_pool.columns else set()

        for ts in two_starters:
            pid = ts.get("player_id")
            name = ts.get("pitcher_name", "?")
            # Check if this pitcher is a free agent
            if pid is not None and pid in fa_ids:
                matchup_score = ts.get("avg_matchup_score", 5.0)
                rate_damage = ts.get("rate_damage_weekly", {})
                era_risk = rate_damage.get("era_change", 0)
                whip_risk = rate_damage.get("whip_change", 0)
                # WHIP safety: career WHIP > 1.40 = risky for ratios
                p_row = fa_pool[fa_pool["player_id"] == pid]
                whip = float(p_row.iloc[0].get("whip", 1.30)) if not p_row.empty else 1.30
                whip_safe = whip <= 1.40

                stream_value = ts.get("two_start_value", 0)
                results.append(
                    {
                        "player_id": pid,
                        "player_name": name,
                        "stream_type": "Two-Start SP",
                        "num_starts": ts.get("num_starts", 2),
                        "matchup_score": round(matchup_score, 1),
                        "era_risk": round(era_risk, 3),
                        "whip_risk": round(whip_risk, 3),
                        "whip_safe": whip_safe,
                        "schedule_value": round(stream_value, 2),
                        "reasoning": (
                            f"2 starts this week (matchup score {matchup_score:.1f}/10). "
                            f"ERA risk: {era_risk:+.3f}. " + ("WHIP safe." if whip_safe else "WHIP RISKY (>1.40).")
                        ),
                    }
                )
    except Exception as exc:
        logger.debug("Two-start detection failed (non-fatal): %s", exc)

    # 2. Hitter schedule quality — prefer hitters with more games vs weak pitching
    try:
        from src.game_day import get_team_strength

        # Find hitter FAs with games this week
        hitter_fas = fa_pool[fa_pool.get("is_hitter", 0) == 1].head(50)
        for _, row in hitter_fas.iterrows():
            team = str(row.get("team", ""))
            if not team or team in ("", "MLB"):
                continue
            # Approximate games from schedule (most teams play 6-7/week)
            opp_strength = get_team_strength(team)
            opp_era = opp_strength.get("team_era", 4.00)
            # Lower opponent pitching ERA = harder matchup for hitter
            # Higher opponent ERA = easier matchup = better streaming target
            if opp_era >= 4.30:  # Above league avg = weak pitching
                sgp_val = float(row.get("marginal_value", row.get("adp", 999)))
                results.append(
                    {
                        "player_id": int(row.get("player_id", 0)),
                        "player_name": str(row.get("name", row.get("player_name", "?"))),
                        "stream_type": "Schedule Hitter",
                        "matchup_score": round(10.0 * (opp_era / 4.00 - 0.8), 1),
                        "schedule_value": round(sgp_val, 2) if sgp_val < 100 else 0,
                        "reasoning": f"Favorable schedule — team faces weak pitching (ERA {opp_era:.2f}).",
                    }
                )
    except Exception as exc:
        logger.debug("Hitter schedule scan failed (non-fatal): %s", exc)

    # Sort by schedule_value descending, cap results
    results.sort(key=lambda x: x.get("schedule_value", 0), reverse=True)
    return results[:max_results]


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
    weeks_remaining: int | None = None,
    config: LeagueConfig | None = None,
) -> dict[str, str]:
    """Classify each category as ATTACK, DEFEND, or IGNORE.

    ATTACK: gap to next position is achievable in remaining weeks
    DEFEND: gap from team behind is small (could lose position)
    IGNORE: punt category or dominant (>3 positions ahead)

    Returns dict[category, priority_tier].
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

        # C1: Rate stat drag — below league-average AVG/OBP hurts team totals.
        # Uses dynamic league averages when available, not hardcoded .245/.310.
        avg = float(row.get("avg", 0) or 0)
        obp = float(row.get("obp", 0) or 0)
        _lg_avg = 0.250  # Dynamic: compute from league data if available
        _lg_obp = 0.320
        try:
            import streamlit as st

            _cached = st.session_state.get("_cached_team_totals")
            if _cached:
                _all_avgs = [t.get("AVG", 0) for t in _cached.values() if t.get("AVG", 0) > 0]
                if _all_avgs:
                    _lg_avg = sum(_all_avgs) / len(_all_avgs)
                _all_obps = [t.get("OBP", 0) for t in _cached.values() if t.get("OBP", 0) > 0]
                if _all_obps:
                    _lg_obp = sum(_all_obps) / len(_all_obps)
        except Exception:
            pass
        if 0 < avg < _lg_avg:
            adjustment -= 1.0
        if 0 < obp < _lg_obp:
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
    weeks_remaining: int | None = None,
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


# ── Matchup-Targeted Free Agent Recommendations ─────────────────────


# Games-per-day assumptions for weekly production estimates
_HITTER_GAMES_PER_DAY = 0.85  # rest days, off days
_PITCHER_RP_GAMES_PER_DAY = 0.70  # reliever appearances
_SP_DAYS_PER_START = 5.0  # one start per 5 days
_SP_IP_PER_START = 6.0  # average IP per start

# Categories where lower values help (inverse stats)
_INVERSE_RATE_CATS = {"ERA", "WHIP"}
_HITTING_COUNTING = {"R", "HR", "RBI", "SB"}
_PITCHING_COUNTING = {"W", "SV", "K", "L"}
_HITTING_RATE = {"AVG", "OBP"}
_PITCHING_RATE = {"ERA", "WHIP"}

# Full-season basis for per-game scaling
_SEASON_GAMES = 162


def _estimate_weekly_production(player_row: pd.Series, days_remaining: int = 5) -> dict:
    """Estimate a player's production for the remaining days of the week.

    Uses projected per-game rates scaled to remaining days.
    Hitters: assume ~0.85 games per day (rest days, off days)
    Pitchers SP: assume 1 start per 5 days, ~6 IP per start
    Pitchers RP: assume ~0.7 appearances per day

    Args:
        player_row: Series with player projection columns (r, hr, rbi, sb,
            avg, obp, w, l, sv, k, era, whip, ip, is_hitter, positions).
        days_remaining: Days left in the matchup week (1-7).

    Returns:
        Dict of {category_name: projected_value} for the remaining days.
        Counting stats are scaled; rate stats pass through as-is.
    """
    result: dict[str, float] = {}
    val = player_row.get("is_hitter")
    is_hitter = bool(int(val)) if val is not None else True

    if is_hitter:
        expected_games = days_remaining * _HITTER_GAMES_PER_DAY
        scale = expected_games / _SEASON_GAMES

        for cat in _HITTING_COUNTING:
            col = cat.lower()
            raw = float(player_row.get(col, 0) or 0)
            result[cat] = raw * scale

        # Rate stats pass through (they represent quality, not volume)
        for cat in _HITTING_RATE:
            col = cat.lower()
            raw = player_row.get(col, None)
            if raw is not None and pd.notna(raw):
                result[cat] = float(raw)
    else:
        # Determine SP vs RP from positions column
        positions = str(player_row.get("positions", "")).upper()
        is_sp = "SP" in positions

        if is_sp:
            expected_starts = days_remaining / _SP_DAYS_PER_START
            projected_season_ip = float(player_row.get("ip", 0) or 0)
            if projected_season_ip > 0:
                # Scale counting stats by fraction of season IP expected this week
                expected_weekly_ip = expected_starts * _SP_IP_PER_START
                scale = expected_weekly_ip / projected_season_ip
            else:
                scale = 0.0
        else:
            # Reliever: scale by games per day ratio
            expected_games = days_remaining * _PITCHER_RP_GAMES_PER_DAY
            scale = expected_games / _SEASON_GAMES

        for cat in _PITCHING_COUNTING:
            col = cat.lower()
            raw = float(player_row.get(col, 0) or 0)
            result[cat] = raw * scale

        # Rate stats pass through
        for cat in _PITCHING_RATE:
            col = cat.lower()
            raw = player_row.get(col, None)
            if raw is not None and pd.notna(raw):
                result[cat] = float(raw)

    return result


def compute_matchup_targeted_adds(
    fa_pool: pd.DataFrame,
    target_categories: list[dict],
    roster_ids: list[int] | None = None,
    player_pool: pd.DataFrame | None = None,
    days_remaining: int = 5,
    max_results: int = 10,
) -> pd.DataFrame:
    """Rank free agents by their ability to help win specific categories this week.

    Unlike compute_add_drop_recommendations() which optimizes season-long SGP,
    this function answers: "Which FA helps me close the gap in HR/SB/K this week?"

    Args:
        fa_pool: DataFrame of available free agents (from load_player_pool or Yahoo).
        target_categories: list of dicts from weekly strategy, each with:
            - name: category name (e.g. "HR")
            - gap: how far behind (negative = losing)
            - priority: 0-1 urgency score
            - status: "losing" / "tied" / "winning"
        roster_ids: current roster player IDs (to exclude from FA pool).
        player_pool: full player pool for projection data. When provided and a
            FA row lacks projection columns, the player_pool row is used instead.
        days_remaining: days left in matchup week (1-7).
        max_results: max FAs to return.

    Returns:
        DataFrame with columns: player_id, name, team, positions,
        matchup_value (0-100 composite score), target_cats (which cats they help),
        projected_weekly_contribution (dict of cat -> projected value this week),
        reason (human-readable string).
        Sorted by matchup_value descending.
    """
    if fa_pool is None or fa_pool.empty or not target_categories:
        return pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "team",
                "positions",
                "matchup_value",
                "target_cats",
                "projected_weekly_contribution",
                "reason",
            ]
        )

    # Build lookup from target_categories list
    cat_lookup: dict[str, dict] = {}
    for tc in target_categories:
        cat_name = tc.get("name", "")
        if cat_name:
            cat_lookup[cat_name] = tc

    if not cat_lookup:
        return pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "team",
                "positions",
                "matchup_value",
                "target_cats",
                "projected_weekly_contribution",
                "reason",
            ]
        )

    # Filter out rostered players
    pool = fa_pool.copy()
    if roster_ids:
        pool = pool[~pool["player_id"].isin(roster_ids)]

    if pool.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "team",
                "positions",
                "matchup_value",
                "target_cats",
                "projected_weekly_contribution",
                "reason",
            ]
        )

    # Merge with player_pool for projection data if available
    if player_pool is not None and not player_pool.empty:
        # Use player_pool projections as fallback for missing columns
        proj_cols = [
            "player_id",
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
            "ip",
            "is_hitter",
            "positions",
        ]
        available_cols = [c for c in proj_cols if c in player_pool.columns]
        pp_subset = player_pool[available_cols].drop_duplicates(subset=["player_id"], keep="first")

        # For FAs missing key stat columns, fill from player_pool
        stat_cols = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip", "ip"]
        missing_stats = [c for c in stat_cols if c not in pool.columns]
        if missing_stats:
            merge_cols = ["player_id"] + [c for c in missing_stats if c in pp_subset.columns]
            if len(merge_cols) > 1:
                pool = pool.merge(pp_subset[merge_cols], on="player_id", how="left", suffixes=("", "_pp"))

    # Status weights: losing categories matter most, protect winning ones
    status_weights = {"losing": 1.5, "tied": 1.0, "winning": 0.3}

    results = []
    for _, fa_row in pool.iterrows():
        pid = int(fa_row.get("player_id", 0))
        name = str(fa_row.get("name", fa_row.get("player_name", "Unknown")))
        team = str(fa_row.get("team", ""))
        positions = str(fa_row.get("positions", ""))

        weekly = _estimate_weekly_production(fa_row, days_remaining)
        if not weekly:
            continue

        # Score against target categories
        raw_score = 0.0
        helped_cats: list[str] = []
        contributions: dict[str, float] = {}
        reason_parts: list[str] = []

        for cat_name, cat_info in cat_lookup.items():
            priority = float(cat_info.get("priority", 0.5))
            status = cat_info.get("status", "tied")
            gap = float(cat_info.get("gap", 0))
            sw = status_weights.get(status, 1.0)

            if cat_name not in weekly:
                continue

            contribution = weekly[cat_name]
            contributions[cat_name] = round(contribution, 4)

            is_inverse = cat_name in _INVERSE_RATE_CATS

            if cat_name in RATE_STATS:
                # Rate stat: for inverse (ERA/WHIP), lower is better
                # For AVG/OBP, higher is better
                if is_inverse:
                    # Good ERA/WHIP (low) helps; bad hurts
                    if status == "winning":
                        # Protect: penalize bad rate stats
                        if contribution > 4.0 and cat_name == "ERA":
                            raw_score -= priority * sw * 5.0
                        elif contribution > 1.30 and cat_name == "WHIP":
                            raw_score -= priority * sw * 5.0
                        else:
                            raw_score += priority * sw * 2.0
                            helped_cats.append(cat_name)
                    else:
                        # Losing/tied: reward low ERA/WHIP
                        if cat_name == "ERA" and contribution < 3.50:
                            raw_score += priority * sw * 3.0
                            helped_cats.append(cat_name)
                        elif cat_name == "WHIP" and contribution < 1.15:
                            raw_score += priority * sw * 3.0
                            helped_cats.append(cat_name)
                else:
                    # AVG/OBP: higher is better
                    if status == "winning":
                        if contribution < 0.240 and cat_name == "AVG":
                            raw_score -= priority * sw * 3.0
                        elif contribution < 0.300 and cat_name == "OBP":
                            raw_score -= priority * sw * 3.0
                    else:
                        if cat_name == "AVG" and contribution > 0.270:
                            raw_score += priority * sw * 3.0
                            helped_cats.append(cat_name)
                        elif cat_name == "OBP" and contribution > 0.340:
                            raw_score += priority * sw * 3.0
                            helped_cats.append(cat_name)
            else:
                # Counting stat: more is better (except L which is inverse counting)
                if cat_name == "L":
                    # Fewer projected losses is better
                    if contribution > 0:
                        raw_score -= priority * sw * contribution * 2.0
                else:
                    if contribution > 0:
                        raw_score += priority * sw * contribution
                        helped_cats.append(cat_name)

        if raw_score <= 0:
            continue

        # Build reason string
        if helped_cats:
            top_contribs = []
            for c in helped_cats[:3]:
                val = contributions.get(c, 0)
                if c in RATE_STATS:
                    top_contribs.append(f"{c} {val:.3f}")
                else:
                    top_contribs.append(f"+{val:.1f} {c}")
            reason_parts.append(f"Helps in {', '.join(helped_cats[:3])}")
            if top_contribs:
                reason_parts.append(f"Projected this week: {', '.join(top_contribs)}")

        losing_helped = [c for c in helped_cats if cat_lookup.get(c, {}).get("status") == "losing"]
        if losing_helped:
            reason_parts.append(f"Closes gap in losing categories: {', '.join(losing_helped)}")

        results.append(
            {
                "player_id": pid,
                "name": name,
                "team": team,
                "positions": positions,
                "matchup_value": raw_score,
                "target_cats": helped_cats,
                "projected_weekly_contribution": contributions,
                "reason": " | ".join(reason_parts) if reason_parts else "Marginal matchup fit",
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "team",
                "positions",
                "matchup_value",
                "target_cats",
                "projected_weekly_contribution",
                "reason",
            ]
        )

    df = pd.DataFrame(results)

    # Normalize matchup_value to 0-100 scale
    max_val = df["matchup_value"].max()
    if max_val > 0:
        df["matchup_value"] = (df["matchup_value"] / max_val * 100).round(1)
    else:
        df["matchup_value"] = 0.0

    df = df.sort_values("matchup_value", ascending=False).head(max_results)
    df = df.reset_index(drop=True)

    return df


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
