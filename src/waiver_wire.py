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
from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig, SGPCalculator

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

# 2026-05-19 D6: snapshot from LeagueConfig (was {"AVG", "OBP", "ERA", "WHIP"} literal).
from src.valuation import LeagueConfig as _LC_FOR_RATES  # noqa: E402

RATE_STATS = set(_LC_FOR_RATES().rate_stats)

# Default weekly adds budget: 5 streaming + 3 injury + 2 reserve = 10
DEFAULT_WEEKLY_ADDS = 10
STREAMING_ADDS_BUDGET = 5

# League-average pitcher WHIP — fallback when a roster row lacks a value.
# FA-engine overhaul P3 PR9 (2026-05-21): sourced from CONSTANTS_REGISTRY.
_LEAGUE_AVG_WHIP: float = CONSTANTS_REGISTRY["league_avg_whip"].value
# WHIP > this value = "ratio-destruction risk" — gates pitcher streams.
_WHIP_SAFETY_CEILING: float = 1.40


# ── Streaming Recommendations ────────────────────────────────────────


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
    replacement_levels: dict | None = None,
) -> float:
    """Compute the adjusted cost of dropping a player from the roster.

    Uses roster category totals comparison as a base, then applies
    multi-factor adjustments so that DH-only, category dead weight,
    and rate stat drag players appear cheaper to drop:

    1. Base SGP cost: how much team SGP drops when player is removed
    2. Positional scarcity multiplier (FA P3.5 PR15) — when
       ``replacement_levels`` is provided, multiplies the base cost so
       dropping a scarce-position player (catcher, SS) costs MORE than
       dropping a deep-position player (OF, 1B) with equivalent raw SGP.
       This is the symmetric counterpart to the add-side scarcity boost
       applied by ``_compute_base_value`` (FA P1 PR3). When None or
       empty, behavior is unchanged from pre-PR15 (backward-compat).
    3. DH/Util-only penalty: -3.0 (no positional value)
    4. Category dead weight: -1.5 if 0 SB, -0.5 if very low HR
    5. Rate stat drag: -1.0 if AVG < league avg, -0.5 if OBP < league avg
    6. Multi-position bonus: +1.0 if 3+ positions (flexibility)

    Lower cost = better drop candidate.

    Returns float (positive = cost of dropping).
    """
    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)

    # Current roster value
    current_totals = _roster_category_totals(roster_ids, player_pool)
    current_sgp = sgp_calc.totals_sgp(current_totals)

    # Roster without this player
    reduced_ids = [pid for pid in roster_ids if pid != player_id]
    reduced_totals = _roster_category_totals(reduced_ids, player_pool)
    reduced_sgp = sgp_calc.totals_sgp(reduced_totals)

    base_cost = current_sgp - reduced_sgp

    # Multi-factor adjustments — reduce cost for players with structural flaws
    match = player_pool[player_pool["player_id"] == player_id]
    if match.empty:
        return base_cost

    row = match.iloc[0]

    # FA-engine overhaul P3.5 PR15 (2026-05-20): symmetric positional
    # scarcity. Apply BEFORE the structural-flaw adjustments below so the
    # scarcity multiplier scales the raw replacement-cost signal — the
    # adjustments are additive flat amounts that compose normally on top.
    # Without this, dropping a top SS to add a backup catcher looked free
    # because the SS didn't pay scarcity on the cost side while the catcher
    # got a 1.20× boost on the add side (see docs/2026-05-20-fa-engine-p3.5-plan.md).
    if replacement_levels:
        from src.valuation import compute_positional_scarcity_factor

        scarcity_mult = compute_positional_scarcity_factor(str(row.get("positions", "")), replacement_levels)
        base_cost *= scarcity_mult

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
        except Exception as exc:
            logger.warning(
                "waiver_wire: failed to read cached league AVG/OBP from session_state; "
                "drop-cost adjustment will use prior season defaults (.250/.320): %s",
                exc,
                exc_info=True,
            )
        if 0 < avg < _lg_avg:
            adjustment -= 1.0
        if 0 < obp < _lg_obp:
            adjustment -= 0.5

    return base_cost + adjustment


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

    sgp_calc = SGPCalculator(config)

    # Before: current roster
    before_totals = _roster_category_totals(roster_ids, player_pool)
    before_sgp = sgp_calc.totals_sgp(before_totals)

    # After: roster - drop + add
    new_ids = [pid for pid in roster_ids if pid != drop_id] + [add_id]
    after_totals = _roster_category_totals(new_ids, player_pool)
    after_sgp = sgp_calc.totals_sgp(after_totals)

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
    """Compute sustainability of current performance — 0.0 to 1.0 calibrated.

    FA-engine overhaul P2 PR5 (2026-05-20): rewritten to use the canonical
    industry regression signals (xwOBA-wOBA gap for hitters, xFIP/SIERA-ERA
    gap for pitchers) via a sigmoid combination, replacing the previous
    5-bucket step function.

    FA-engine overhaul P5e (2026-05-21): counting-stat regression signals
    added on top of the rate-stat anchor. xwOBA-wOBA predicts AVG/OBP
    regression but doesn't predict HR sustainability — HR/FB% drives
    that. LOB% (strand rate) drives ERA regression over a longer
    horizon than SIERA-ERA. Both terms are additive on the existing
    logit and skipped entirely when the underlying column is absent
    (backward compat with the P2 PR5 rate-only behavior).

    Hitters:
      Primary:   xwOBA - wOBA delta (POSITIVE = overperforming → LOW sus)
      Secondary: BABIP vs career baseline / .300 league avg
      NEW (P5e): HR/FB% vs league avg ~0.13 (HIGH = HR regression risk)

    Pitchers:
      Primary:   ERA - xFIP delta (POSITIVE = unlucky → HIGH sus, buy-low)
      Secondary: Stuff+ vs league avg 100
      NEW (P5e): LOB% vs league avg 0.74 (HIGH = strand-rate regression
                 risk → LOW sustainability)

    Returns 0.5 (neutral) when sample size insufficient — small samples
    are too noisy for either regression signal. Threshold AB > 80 / IP > 30
    (closer to BABIP/xwOBA stabilization point per Pizza Cutter / Russell
    Carleton research).
    """
    val = player.get("is_hitter")
    is_hitter = int(val) if val is not None else 1

    def _sigmoid(x: float) -> float:
        """Standard logistic. Output in (0, 1)."""
        import math

        # Clamp to avoid overflow on extreme inputs.
        if x > 12:
            return 0.999
        if x < -12:
            return 0.001
        return 1.0 / (1.0 + math.exp(-x))

    if is_hitter:
        ab = float(player.get("ab", 0) or 0)
        if ab < 80:
            return 0.5  # Insufficient sample (was AB > 50 — too lax)

        # Primary signal: xwOBA - wOBA gap (canonical regression signal per
        # FanGraphs, Pitcher List, The Athletic). Pool's xwoba_delta column
        # is already woba - xwoba (per src/database.py convention) — so a
        # POSITIVE delta means wOBA > xwOBA = OVERPERFORMING = regression
        # DOWN = unsustainable. The sustainability score should DECREASE.
        xwoba_delta = float(player.get("xwoba_delta", 0) or 0)
        # Industry threshold: |gap| > 0.030 wOBA points is meaningful signal.
        # Sigmoid scaled so delta=+0.030 → ~0.30 sustainability (sell-high)
        # and delta=-0.030 → ~0.80 (buy-low). Coefficient: 0.7 / 0.030 ≈ 23.
        primary_logit = -23.0 * xwoba_delta

        # Secondary signal: BABIP vs career-typical .300 (or career_babip if
        # supplied). Lighter weight than primary.
        if "babip" in player.index and player.get("babip") not in (None, 0):
            # Caller supplied BABIP directly — honor it.
            babip = float(player.get("babip", 0) or 0)
        else:
            h = float(player.get("h", 0) or 0)
            hr = float(player.get("hr", 0) or 0)
            sf = float(player.get("sf", 0) or 0)
            hitter_k = float(player.get("k", 0) or 0)
            babip = compute_babip(h, hr, ab, hitter_k, sf)
        # Prefer career_babip baseline when supplied; else .300 league avg.
        career_babip = float(player.get("career_babip", 0) or 0)
        babip_baseline = career_babip if career_babip > 0 else 0.300
        # BABIP delta from baseline — positive = overperforming.
        babip_logit = -8.0 * (babip - babip_baseline)

        # P5e counting-stat signal: HR/FB%. League avg ~0.13, SD ~0.05.
        # Positive z-score (e.g. 30% HR/FB) → unsustainable HR rate →
        # LOWER sustainability. Skip the term entirely when the column
        # isn't supplied (backward compat with PR5 rate-only behavior).
        hr_fb_logit = 0.0
        hr_per_fb_raw = player.get("hr_per_fb")
        if hr_per_fb_raw not in (None,):
            try:
                hr_per_fb = float(hr_per_fb_raw or 0)
            except (TypeError, ValueError):
                hr_per_fb = 0.0
            if hr_per_fb > 0:
                # z = (hr_per_fb - 0.13) / 0.05. Coefficient -2.0 so that
                # a +2 SD elevation (HR/FB ~0.23) shifts logit by -4 (~-0.7
                # in probability space — meaningful but not overwhelming).
                hr_fb_z = (hr_per_fb - 0.13) / 0.05
                hr_fb_logit = -2.0 * hr_fb_z

        # Combine — primary signal weighted 2x relative to secondaries.
        logit = 2.0 * primary_logit + babip_logit + hr_fb_logit
        # Center at sigmoid(0) = 0.5 (neutral when no signal).
        return _sigmoid(logit)

    # Pitchers
    ip = float(player.get("ip", 0) or 0)
    if ip < 30:
        return 0.5  # Insufficient sample (was IP > 20)

    era = float(player.get("era", 4.0) or 4.0)
    # Primary signal: xFIP-ERA delta. Pool has xfip column. Sign:
    # xFIP > ERA → ERA was LUCKY (lower than skill warrants) → regression
    # UP coming → unsustainable. Sustainability LOW.
    # ERA > xFIP → ERA was UNLUCKY → regression DOWN coming → BUY LOW.
    # Sustainability HIGH.
    xfip = float(player.get("xfip", era) or era)
    era_minus_xfip = era - xfip
    # |gap| > 0.50 ERA points is meaningful. Coefficient: 0.7 / 0.50 = 1.4.
    primary_logit = 1.4 * era_minus_xfip

    # Secondary: Stuff+ vs league average 100. Higher Stuff+ → underlying
    # skill is real → higher sustainability.
    stuff_plus = float(player.get("stuff_plus", 100.0) or 100.0)
    stuff_logit = (stuff_plus - 100.0) / 20.0  # ±10 Stuff+ ≈ ±0.5 logit shift

    # P5e counting-stat signal: LOB% (strand rate). League avg ~0.74,
    # SD ~0.04. Positive z (e.g. 0.85 LOB%) → unsustainable strand rate
    # → regression UP coming → LOWER sustainability. Skip when missing
    # (backward compat).
    lob_logit = 0.0
    lob_pct_raw = player.get("lob_pct")
    if lob_pct_raw not in (None,):
        try:
            lob_pct = float(lob_pct_raw or 0)
        except (TypeError, ValueError):
            lob_pct = 0.0
        if lob_pct > 0:
            # z = (lob_pct - 0.74) / 0.04. Coefficient -1.5 so that
            # +2.75 SD elevation (LOB% ~0.85) shifts logit by ~-4.1.
            lob_z = (lob_pct - 0.74) / 0.04
            lob_logit = -1.5 * lob_z

    logit = 2.0 * primary_logit + stuff_logit + lob_logit
    return _sigmoid(logit)


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

    .. deprecated:: 2026-05-20
        This function has been superseded by
        :func:`src.optimizer.fa_recommender.recommend_fa_moves`. The newer
        engine has opponent context (urgency weights from current matchup),
        IL stash protection wired into the scoring (not just UI), news-warning
        surfacing, and slot-aware drop selection — none of which exist here.

        Today's Crochet/Kirk bad-recommendation bug traced to this engine
        zeroing out IL players in ``_roster_category_totals`` (fixed in PR #90
        at the engine layer) and not factoring opponent category needs. The
        Free Agents page (``pages/14_Free_Agents.py``) now calls
        ``recommend_fa_moves`` first and falls back to this function only on
        exception.

        This function will be removed in the next audit sweep cycle. New
        callers should use ``recommend_fa_moves(ctx)`` with an
        ``OptimizerDataContext`` built via ``build_optimizer_context``.

    Pipeline (legacy):
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
    import warnings

    warnings.warn(
        "compute_add_drop_recommendations is deprecated and will be removed in the next sweep. "
        "Use src.optimizer.fa_recommender.recommend_fa_moves(ctx) instead — see "
        "docs/2026-05-20-fa-engine-overhaul-plan.md.",
        DeprecationWarning,
        stacklevel=2,
    )

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

    # Closer priority override — if user has < 2 closers,
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
    # Pre-compute roster composition for position-type guard
    _MIN_PITCHERS = 8  # 2SP + 2RP + 4P slots
    _MIN_HITTERS = 10  # C + 1B + 2B + 3B + SS + 3OF + 2Util

    def _pool_is_pitcher(pid: int) -> bool:
        """Check if a player_id is a pitcher in the pool."""
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            return False
        return int(match.iloc[0].get("is_hitter", 1)) == 0

    roster_pitcher_count = sum(1 for pid in user_roster_ids if _pool_is_pitcher(pid))
    roster_hitter_count = len(user_roster_ids) - roster_pitcher_count

    swap_results = []
    for fa_rank_idx, (_, fa_row) in enumerate(top_fas.iterrows()):
        fa_id = int(fa_row["player_id"])
        fa_name = fa_row.get("player_name", fa_row.get("name", "?"))

        # Get FA's full stats from player pool
        fa_match = player_pool[player_pool["player_id"] == fa_id]
        if fa_match.empty:
            continue
        fa_player = fa_match.iloc[0]
        fa_is_pitcher = int(fa_player.get("is_hitter", 1)) == 0

        for drop_info in top_drops:
            drop_id = drop_info["player_id"]
            drop_name = drop_info["name"]

            # Skip if same player
            if fa_id == drop_id:
                continue

            # Roster composition guard: enforce minimum pitchers / hitters
            drop_is_pitcher = _pool_is_pitcher(drop_id)
            if drop_is_pitcher and not fa_is_pitcher:
                # Dropping pitcher, adding hitter — check pitcher floor
                if roster_pitcher_count <= _MIN_PITCHERS:
                    continue
            if not drop_is_pitcher and fa_is_pitcher:
                # Dropping hitter, adding pitcher — check hitter floor
                if roster_hitter_count <= _MIN_HITTERS:
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
            float(cat_info.get("gap", 0))
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
                        if contribution > 4.0 and cat_name == "ERA" or contribution > 1.30 and cat_name == "WHIP":
                            raw_score -= priority * sw * 5.0
                        else:
                            raw_score += priority * sw * 2.0
                            helped_cats.append(cat_name)
                    else:
                        # Losing/tied: reward low ERA/WHIP
                        if cat_name == "ERA" and contribution < 3.50 or cat_name == "WHIP" and contribution < 1.15:
                            raw_score += priority * sw * 3.0
                            helped_cats.append(cat_name)
                else:
                    # AVG/OBP: higher is better
                    if status == "winning":
                        if contribution < 0.240 and cat_name == "AVG" or contribution < 0.300 and cat_name == "OBP":
                            raw_score -= priority * sw * 3.0
                    else:
                        if cat_name == "AVG" and contribution > 0.270 or cat_name == "OBP" and contribution > 0.340:
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

    # Best category impact — attribute to the correct side of the swap.
    # 2026-05-19 D4: LeagueConfig is always constructible; fallback simplified.
    from src.valuation import LeagueConfig

    inverse_cats = (config or LeagueConfig()).inverse_stats
    deltas = swap["category_deltas"]
    if deltas:
        best_cat = max(deltas, key=lambda c: deltas[c])
        best_val = deltas[best_cat]
        if best_val > 0:
            if best_cat in inverse_cats:
                # Inverse stat gain comes from dropping the player (reducing L/ERA/WHIP)
                reasons.append(f"Dropping {drop_name} saves +{best_val:.2f} SGP in {best_cat}")
            else:
                reasons.append(f"{fa_name} adds +{best_val:.2f} SGP in {best_cat}")

    # Worst category impact — attribute to the correct side
    if deltas:
        worst_cat = min(deltas, key=lambda c: deltas[c])
        worst_val = deltas[worst_cat]
        if worst_val < -0.1:
            if worst_cat in inverse_cats:
                # Inverse stat loss means the add player worsens L/ERA/WHIP
                reasons.append(f"{fa_name} costs {worst_val:.2f} SGP in {worst_cat}")
            else:
                # Normal stat loss means the drop player had value here
                reasons.append(f"Dropping {drop_name} costs {worst_val:.2f} SGP in {worst_cat}")

    # Sustainability flag
    if sustainability < 0.4:
        reasons.append("Caution: current stats may not be sustainable (high BABIP or low ERA)")
    elif sustainability > 0.7:
        reasons.append("Strong underlying metrics support continued production")

    # Net SGP
    reasons.append(f"Net team improvement: +{swap['net_sgp']:.2f} SGP")

    return reasons
