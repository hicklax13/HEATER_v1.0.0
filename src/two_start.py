"""Two-Start Pitcher Planner.

Identifies pitchers with 2+ starts in an upcoming week and computes
their streaming value with rate stat damage analysis. Reuses existing
schedule, streaming, and park factor infrastructure.

Wires into:
  - src/optimizer/matchup_adjustments.py: get_weekly_schedule(), park_factor_adjustment()
  - src/optimizer/streaming.py: quantify_two_start_value(), compute_streaming_value()
  - src/data_bootstrap.py: PARK_FACTORS (30-team dict)
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Module-level imports with graceful degradation ───────────────────
# These are imported at module level so they can be patched in tests.

try:
    from src.data_bootstrap import PARK_FACTORS as _PARK_FACTORS
except ImportError:
    _PARK_FACTORS = {}

try:
    from src.optimizer.matchup_adjustments import (
        _MLB_TEAM_ABBREVS,
        get_weekly_schedule,
    )
except ImportError:
    _MLB_TEAM_ABBREVS = {}

    def get_weekly_schedule(days_ahead: int = 7) -> list:
        """Stub when matchup_adjustments is unavailable."""
        return []


try:
    from src.optimizer.streaming import (
        compute_streaming_value,
        quantify_two_start_value,
    )
except ImportError:
    _STREAMING_AVAILABLE = False
else:
    _STREAMING_AVAILABLE = True


# ── Constants ────────────────────────────────────────────────────────

# League-average defaults for fallback when API data is unavailable.
_LEAGUE_AVG_K_PCT: float = 0.225
_LEAGUE_AVG_WRC_PLUS: float = 100.0
_LEAGUE_AVG_ERA: float = 4.00
_LEAGUE_AVG_WHIP: float = 1.25
_LEAGUE_AVG_CSW_PCT: float = 0.295
_LEAGUE_AVG_XFIP: float = 4.10

# Home/away adjustment: home pitchers historically perform ~3% better.
_HOME_ADVANTAGE: float = 1.03
_AWAY_DISADVANTAGE: float = 0.97

# Confidence tier thresholds (days until start).
_HIGH_CONFIDENCE_DAYS: int = 2
_MEDIUM_CONFIDENCE_DAYS: int = 5

# Default IP per start for a typical starting pitcher.
_DEFAULT_IP_PER_START: float = 5.5

# Cached team batting stats (module-level, reset per session).
_team_batting_cache: dict[str, dict[str, float]] | None = None
_team_batting_lock = threading.Lock()


# ── Team Batting Stats ───────────────────────────────────────────────


def _league_avg_batting() -> dict[str, float]:
    """Return league-average batting defaults.

    Used as fallback when MLB Stats API is unavailable.

    Returns:
        Dict with wrc_plus and k_pct at league-average values.
    """
    return {
        "wrc_plus": _LEAGUE_AVG_WRC_PLUS,
        "k_pct": _LEAGUE_AVG_K_PCT,
    }


def fetch_team_batting_stats(season: int | None = None) -> dict[str, dict[str, float]]:
    """Fetch team-level offensive stats via MLB Stats API.

    Results are cached at the module level so repeated calls within
    the same session avoid redundant API requests.

    Args:
        season: MLB season year. Defaults to current year.

    Returns:
        Dict mapping team abbreviation (e.g. "NYY") to a dict with
        keys ``wrc_plus`` and ``k_pct``. Falls back to league-average
        defaults for all 30 teams when the API is unavailable.
    """
    global _team_batting_cache
    with _team_batting_lock:
        if _team_batting_cache is not None:
            return _team_batting_cache

    try:
        import statsapi
    except ImportError:
        logger.debug("statsapi not installed; using league-average batting defaults")
        # Import failure is permanent — safe to cache empty.
        with _team_batting_lock:
            _team_batting_cache = {}
            return _team_batting_cache

    # Use module-level _MLB_TEAM_ABBREVS (already imported at top of file)
    if not _MLB_TEAM_ABBREVS:
        with _team_batting_lock:
            _team_batting_cache = {}
            return _team_batting_cache

    if season is None:
        season = datetime.now(UTC).year

    try:
        teams_data = statsapi.get("teams", {"sportId": 1, "season": season})
        team_list = teams_data.get("teams", [])

        result: dict[str, dict[str, float]] = {}
        full_to_abbrev = _MLB_TEAM_ABBREVS

        for team_info in team_list:
            full_name = team_info.get("name", "")
            abbrev = full_to_abbrev.get(full_name, "")
            if not abbrev:
                continue

            team_id = team_info.get("id")
            if team_id is None:
                continue

            try:
                stats_resp = statsapi.get(
                    "team_stats",
                    {
                        "teamId": team_id,
                        "group": "hitting",
                        "stats": "season",
                        "season": season,
                    },
                )
                splits = stats_resp.get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
                so = float(splits.get("strikeOuts", 0))
                pa = float(splits.get("plateAppearances", 1))
                k_pct = so / pa if pa > 0 else _LEAGUE_AVG_K_PCT

                # MLB Stats API doesn't expose wRC+ directly; use OPS+ as proxy
                ops_plus = float(splits.get("opsPlus", _LEAGUE_AVG_WRC_PLUS))
                result[abbrev] = {
                    "wrc_plus": ops_plus,
                    "k_pct": k_pct,
                }
            except Exception:
                result[abbrev] = _league_avg_batting()

        with _team_batting_lock:
            _team_batting_cache = result
            return _team_batting_cache

    except Exception:
        logger.warning("Failed to fetch team batting stats", exc_info=True)
        # Transient failure — do NOT cache empty dict so retries are possible.
        return {}


def clear_team_batting_cache() -> None:
    """Clear the cached team batting stats, forcing a fresh fetch."""
    global _team_batting_cache
    with _team_batting_lock:
        _team_batting_cache = None


# ── Rate Stat Damage ─────────────────────────────────────────────────


def rate_stat_damage(
    pitcher_era: float,
    pitcher_whip: float,
    pitcher_ip: float,
    team_era: float,
    team_whip: float,
    team_ip: float,
) -> dict[str, float]:
    """Compute ERA/WHIP change from adding a pitcher's start to team totals.

    Positive values mean the team's rate stat goes UP (bad for ERA/WHIP).
    Negative values mean the team's rate stat goes DOWN (good).

    Formula::

        era_change = (pitcher_ERA - team_ERA) * IP / (team_IP + IP)
        whip_change = (pitcher_WHIP - team_WHIP) * IP / (team_IP + IP)

    Args:
        pitcher_era: Pitcher's ERA.
        pitcher_whip: Pitcher's WHIP.
        pitcher_ip: Innings pitched in this start.
        team_era: Team's current ERA.
        team_whip: Team's current WHIP.
        team_ip: Team's current total IP.

    Returns:
        Dict with keys ``era_change`` and ``whip_change``, each a float
        representing the delta to the team's rate stat.
    """
    if team_ip + pitcher_ip <= 0:
        return {"era_change": 0.0, "whip_change": 0.0}

    denominator = team_ip + pitcher_ip
    era_change = (pitcher_era - team_era) * pitcher_ip / denominator
    whip_change = (pitcher_whip - team_whip) * pitcher_ip / denominator

    return {
        "era_change": round(era_change, 6),
        "whip_change": round(whip_change, 6),
    }


# ── Pitcher Matchup Score ────────────────────────────────────────────


def _normalize(value: float, low: float, high: float) -> float:
    """Normalize a value to the 0-1 range given bounds.

    Args:
        value: Raw value.
        low: Lower bound (maps to 0).
        high: Upper bound (maps to 1).

    Returns:
        Normalized value clipped to [0, 1].
    """
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_pitcher_matchup_score(
    pitcher_stats: dict[str, float],
    opponent_team_stats: dict[str, float] | None = None,
    park_factor: float = 1.0,
    is_home: bool = True,
) -> float:
    """Combine pitcher skill with opponent quality for a 0-10 matchup rating.

    Pitcher quality is a weighted blend of K-BB%, inverted xFIP, and
    CSW% (called + swinging strike percentage). This is then adjusted
    by opponent quality (inverse wRC+, strikeout tendency), park factor,
    and home/away.

    Formula::

        pitcher_quality = 0.40 * norm(K_BB%) + 0.30 * norm(10 - xFIP) + 0.30 * norm(CSW%)
        opponent_factor = (100 / opp_wRC+) * (opp_K% / league_K%)
        matchup_score   = pitcher_quality * opponent_factor * park_adj * home_away * 10

    Args:
        pitcher_stats: Dict with optional keys ``k_bb_pct`` (0-0.30 typical),
            ``xfip`` (2.50-6.00 typical), ``csw_pct`` (0.25-0.35 typical),
            ``era``, ``whip``. Missing keys fall back to league averages.
        opponent_team_stats: Dict with optional ``wrc_plus`` and ``k_pct``.
            None defaults to league averages.
        park_factor: Park factor for the game venue (pitcher perspective:
            lower is better, so we invert).
        is_home: True if pitcher is at home.

    Returns:
        Matchup score from 0.0 to 10.0, where 10 is the best possible
        matchup for the pitcher.
    """
    # Extract pitcher skill metrics with fallbacks.
    k_bb_pct = pitcher_stats.get("k_bb_pct", 0.10)
    xfip = pitcher_stats.get("xfip", _LEAGUE_AVG_XFIP)
    csw_pct = pitcher_stats.get("csw_pct", _LEAGUE_AVG_CSW_PCT)

    # Normalize each component to 0-1. Ranges reflect MLB extremes.
    norm_k_bb = _normalize(k_bb_pct, -0.05, 0.30)
    norm_xfip = _normalize(10.0 - xfip, 4.0, 7.5)  # invert: lower xFIP = better
    norm_csw = _normalize(csw_pct, 0.22, 0.35)

    pitcher_quality = 0.40 * norm_k_bb + 0.30 * norm_xfip + 0.30 * norm_csw

    # Opponent quality: weak opponents inflate the score.
    if opponent_team_stats is None:
        opponent_team_stats = _league_avg_batting()

    opp_wrc_plus = max(opponent_team_stats.get("wrc_plus", _LEAGUE_AVG_WRC_PLUS), 1.0)
    opp_k_pct = opponent_team_stats.get("k_pct", _LEAGUE_AVG_K_PCT)
    if opp_k_pct <= 0:
        opp_k_pct = _LEAGUE_AVG_K_PCT

    opponent_factor = (100.0 / opp_wrc_plus) * (opp_k_pct / _LEAGUE_AVG_K_PCT)

    # Park factor adjustment (for pitchers, lower park factor = better).
    # Invert around 1.0 so Coors (1.38) hurts and MIA (0.88) helps.
    park_adj = 2.0 - park_factor if park_factor > 0 else 1.0
    park_adj = max(park_adj, 0.5)  # floor to avoid extreme values

    # Home/away adjustment.
    home_away_adj = _HOME_ADVANTAGE if is_home else _AWAY_DISADVANTAGE

    raw_score = pitcher_quality * opponent_factor * park_adj * home_away_adj * 10.0

    # Clip to [0, 10] range.
    return round(max(0.0, min(10.0, raw_score)), 2)


# ── Confidence Tier ──────────────────────────────────────────────────


def _confidence_tier(game_date_str: str) -> str:
    """Determine confidence tier based on days until the game.

    Args:
        game_date_str: Date string in YYYY-MM-DD format.

    Returns:
        "HIGH" (1-2 days), "MEDIUM" (3-5 days), or "LOW" (6+ days).
    """
    try:
        game_date = datetime.strptime(game_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        days_ahead = (game_date - today).days
    except (ValueError, TypeError):
        return "LOW"

    if days_ahead < 0:
        return "LOW"
    if days_ahead <= _HIGH_CONFIDENCE_DAYS:
        return "HIGH"
    elif days_ahead <= _MEDIUM_CONFIDENCE_DAYS:
        return "MEDIUM"
    else:
        return "LOW"


# ── Main Identification Function ─────────────────────────────────────


def identify_two_start_pitchers(
    days_ahead: int = 7,
    team_era: float = _LEAGUE_AVG_ERA,
    team_whip: float = _LEAGUE_AVG_WHIP,
    team_ip: float = 55.0,
    player_pool: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Identify pitchers with 2+ starts in an upcoming week.

    Fetches MLB schedule via ``get_weekly_schedule()``, counts pitcher
    appearances as probable starters, and computes streaming value with
    rate stat damage analysis for each two-start pitcher.

    Args:
        days_ahead: Number of days to look ahead (default 7).
        team_era: User's team ERA for rate-stat damage calculation.
        team_whip: User's team WHIP for rate-stat damage calculation.
        team_ip: User's team total IP for rate-stat damage calculation.

    Returns:
        List of dicts sorted by matchup_score descending, each containing:
          - pitcher_name: str
          - team: str
          - num_starts: int (always >= 2)
          - starts: list of dicts with game_date, opponent, is_home,
              park_factor, matchup_score, confidence
          - avg_matchup_score: float (0-10)
          - rate_damage: dict with era_change and whip_change (per start)
          - two_start_value: dict from quantify_two_start_value()
          - streaming_value: dict from compute_streaming_value()

        Returns empty list if schedule is unavailable or no pitchers
        have 2+ starts.
    """
    if not _MLB_TEAM_ABBREVS or not _STREAMING_AVAILABLE:
        logger.warning("Required modules not available for two-start planner")
        return []

    schedule = get_weekly_schedule(days_ahead=days_ahead)
    if not schedule:
        return []

    # Build reverse lookup: full team name -> abbreviation
    full_to_abbrev = _MLB_TEAM_ABBREVS

    # Count pitcher appearances across the schedule.
    # Key: pitcher name, Value: list of start details.
    pitcher_starts: dict[str, list[dict[str, Any]]] = {}
    pitcher_teams: dict[str, str] = {}

    for game in schedule:
        game_date = game.get("game_date", "")
        home_name = game.get("home_name", "")
        away_name = game.get("away_name", "")
        home_abbrev = full_to_abbrev.get(home_name, "")
        away_abbrev = full_to_abbrev.get(away_name, "")

        # Home probable pitcher
        home_pitcher = game.get("home_probable_pitcher", "")
        if home_pitcher:
            pitcher_starts.setdefault(home_pitcher, []).append(
                {
                    "game_date": game_date,
                    "opponent": away_abbrev,
                    "opponent_full": away_name,
                    "is_home": True,
                    "park_team": home_abbrev,
                }
            )
            pitcher_teams[home_pitcher] = home_abbrev

        # Away probable pitcher
        away_pitcher = game.get("away_probable_pitcher", "")
        if away_pitcher:
            pitcher_starts.setdefault(away_pitcher, []).append(
                {
                    "game_date": game_date,
                    "opponent": home_abbrev,
                    "opponent_full": home_name,
                    "is_home": False,
                    "park_team": home_abbrev,
                }
            )
            pitcher_teams[away_pitcher] = away_abbrev

    # Filter to pitchers with 2+ starts.
    team_batting = fetch_team_batting_stats()
    results: list[dict[str, Any]] = []

    # Build pitcher name lookup from player_pool for actual stats
    pitcher_stats_lookup: dict[str, dict[str, float]] = {}
    if player_pool is not None and not player_pool.empty:
        pitchers = player_pool[player_pool.get("is_hitter", pd.Series(dtype=int)) == 0]
        if not pitchers.empty:
            name_col = "name" if "name" in pitchers.columns else "player_name"
            for _, p in pitchers.iterrows():
                pname = str(p.get(name_col, ""))
                if pname:
                    pitcher_stats_lookup[pname] = {
                        "era": float(p.get("era", 0) or 0),
                        "whip": float(p.get("whip", 0) or 0),
                        "k": float(p.get("k", 0) or 0),
                        "w": float(p.get("w", 0) or 0),
                        "l": float(p.get("l", 0) or 0),
                        "ip": float(p.get("ip", 0) or 0),
                        "k_bb_pct": float(p.get("k_bb_pct", 0) or 0),
                        "xfip": float(p.get("xfip", 0) or 0),
                        "csw_pct": float(p.get("csw_pct", 0) or 0),
                    }

    for pitcher_name, starts in pitcher_starts.items():
        if len(starts) < 2:
            continue

        team_abbrev = pitcher_teams.get(pitcher_name, "")

        # Look up actual pitcher stats; fall back to league-average defaults
        p_stats = pitcher_stats_lookup.get(pitcher_name, {})
        p_era = p_stats.get("era") or _LEAGUE_AVG_ERA
        p_whip = p_stats.get("whip") or _LEAGUE_AVG_WHIP
        p_ip = p_stats.get("ip") or _DEFAULT_IP_PER_START
        p_k = p_stats.get("k") or 6.0
        p_w = p_stats.get("w") or 0.5
        p_l = p_stats.get("l") or 0.3

        # Compute per-start IP (if season totals provided, estimate per-start)
        if p_ip > 30:
            # Season totals: estimate per-start IP from total IP
            est_starts = max(1, round(p_ip / _DEFAULT_IP_PER_START))
            ip_per_start = p_ip / est_starts
            # Normalize K/W/L to per-start values (they are season totals too)
            k_per_start = p_k / est_starts if est_starts > 0 else p_k
            w_per_start = p_w / est_starts if est_starts > 0 else p_w
            l_per_start = p_l / est_starts if est_starts > 0 else p_l
        else:
            ip_per_start = p_ip if p_ip > 0 else _DEFAULT_IP_PER_START
            k_per_start = p_k
            w_per_start = p_w
            l_per_start = p_l

        # Build pitcher skill dict for matchup scoring
        pitcher_skill_stats = {
            "k_bb_pct": p_stats.get("k_bb_pct", 0.10),
            "xfip": p_stats.get("xfip") or _LEAGUE_AVG_XFIP,
            "csw_pct": p_stats.get("csw_pct") or _LEAGUE_AVG_CSW_PCT,
            "era": p_era,
            "whip": p_whip,
        }

        # Enrich each start with park factor, matchup score, confidence.
        enriched_starts: list[dict[str, Any]] = []
        total_matchup_score = 0.0

        for start in starts:
            park_team = start.get("park_team", "")
            pf = _PARK_FACTORS.get(park_team, 1.0)
            opp_abbrev = start.get("opponent", "")
            opp_stats = team_batting.get(opp_abbrev) if team_batting else None

            matchup_score = compute_pitcher_matchup_score(
                pitcher_stats=pitcher_skill_stats,
                opponent_team_stats=opp_stats,
                park_factor=pf,
                is_home=start.get("is_home", False),
            )

            confidence = _confidence_tier(start.get("game_date", ""))

            enriched_starts.append(
                {
                    "game_date": start["game_date"],
                    "opponent": opp_abbrev,
                    "is_home": start["is_home"],
                    "park_factor": pf,
                    "matchup_score": matchup_score,
                    "confidence": confidence,
                }
            )
            total_matchup_score += matchup_score

        avg_matchup = total_matchup_score / len(enriched_starts) if enriched_starts else 0.0
        num_starts = len(enriched_starts)

        # Rate stat damage per start using actual pitcher stats.
        damage_per_start = rate_stat_damage(
            pitcher_era=p_era,
            pitcher_whip=p_whip,
            pitcher_ip=ip_per_start,
            team_era=team_era,
            team_whip=team_whip,
            team_ip=team_ip,
        )

        # Cumulative weekly rate damage across all starts
        # Compute directly using total IP to avoid linear approximation error
        total_pitcher_ip = ip_per_start * num_starts
        if team_ip + total_pitcher_ip > 0:
            cumulative_era = (p_era - team_era) * total_pitcher_ip / (team_ip + total_pitcher_ip)
            cumulative_whip = (p_whip - team_whip) * total_pitcher_ip / (team_ip + total_pitcher_ip)
        else:
            cumulative_era = 0.0
            cumulative_whip = 0.0
        cumulative_damage = {
            "era_change": round(cumulative_era, 6),
            "whip_change": round(cumulative_whip, 6),
        }

        # Average park factor across starts for streaming value
        avg_pf = sum(s["park_factor"] for s in enriched_starts) / num_starts if enriched_starts else 1.0

        # Two-start value from existing streaming module.
        two_start_val = quantify_two_start_value(
            pitcher_stats={
                "k": k_per_start,
                "w": w_per_start,
                "l": l_per_start,
                "era": p_era,
                "whip": p_whip,
                "ip": ip_per_start,
            },
            team_era=team_era,
            team_whip=team_whip,
        )

        # Streaming value for the full week, with park factor.
        streaming_val = compute_streaming_value(
            pitcher={
                "k": k_per_start,
                "w": w_per_start,
                "l": l_per_start,
                "era": p_era,
                "whip": p_whip,
                "ip": ip_per_start,
            },
            weekly_games=num_starts,
            team_park_factor=avg_pf,
        )

        results.append(
            {
                "pitcher_name": pitcher_name,
                "team": team_abbrev,
                "num_starts": num_starts,
                "starts": enriched_starts,
                "avg_matchup_score": round(avg_matchup, 2),
                "rate_damage_per_start": damage_per_start,
                "rate_damage_weekly": cumulative_damage,
                "two_start_value": two_start_val,
                "streaming_value": streaming_val,
            }
        )

    # Sort by avg matchup score descending.
    results.sort(key=lambda x: x["avg_matchup_score"], reverse=True)

    return results
