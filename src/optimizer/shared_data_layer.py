"""Shared Data Layer for the Line-up Optimizer.

Builds a single OptimizerDataContext that all optimizer tabs consume.
This eliminates redundant DB queries, inconsistent category weights,
and diverging projection sources across tabs.

Usage:
    ctx = build_optimizer_context("rest_of_week", yds, config)
    # All tabs read from ctx instead of computing independently.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.optimizer.data_freshness import DataFreshnessTracker
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Category correlation adjustments (from standings_engine.CATEGORY_CORRELATIONS).
# When a correlated category is already strong, reduce marginal weight of
# its partner to avoid double-counting the same production cluster.
_CORRELATION_PAIRS: dict[tuple[str, str], float] = {
    ("HR", "R"): 0.72,
    ("HR", "RBI"): 0.68,
    ("R", "RBI"): 0.65,
    ("AVG", "OBP"): 0.85,
    ("ERA", "WHIP"): 0.78,
    ("W", "K"): 0.45,
    ("SB", "AVG"): -0.15,
    ("W", "L"): -0.30,
    ("SV", "W"): -0.10,
}

# Correlation dampening range: if a correlated category is already well-served
# (urgency < 0.3), reduce the partner's weight by this factor.
_CORR_DAMPEN_STRONG = 0.85
_CORR_BOOST_WEAK = 1.10

# Playoff premium for weeks 21-24 schedule
_PLAYOFF_WEEKS = {21, 22, 23, 24}
_PLAYOFF_SCHEDULE_PREMIUM = 1.15

# Approximate games per MLB season (162 game schedule)
_SEASON_GAMES = 162

# Recent form blend weights per scope
_RECENT_FORM_WEIGHT_TODAY = 0.25
_RECENT_FORM_WEIGHT_WEEK = 0.30
_RECENT_FORM_WEIGHT_SEASON = 0.20

# Two-start pitcher counting stat multiplier
_TWO_START_COUNTING_MULT = 2.0

# Opposing pitcher strength adjustment range
_OPP_PITCHER_STRONG_MULT = 0.92  # vs top-tier pitching
_OPP_PITCHER_WEAK_MULT = 1.08  # vs bottom-tier pitching
_OPP_PITCHER_FIP_STRONG = 3.20  # FIP threshold for "strong"
_OPP_PITCHER_FIP_WEAK = 4.50  # FIP threshold for "weak"

# League rules
_MAX_WEEKLY_ADDS = 10
_MIN_CLOSERS = 2
_CLOSER_SV_THRESHOLD = 5  # Projected SV >= 5 counts as closer


# ---------------------------------------------------------------------------
# Data Context
# ---------------------------------------------------------------------------


@dataclass
class OptimizerDataContext:
    """Immutable data context built once per optimization run.

    All tabs read from this instead of computing independently.
    """

    # Core data
    roster: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    free_agents: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_roster_ids: list[int] = field(default_factory=list)

    # Matchup state
    live_matchup: dict | None = None
    my_totals: dict[str, float] = field(default_factory=dict)
    opp_totals: dict[str, float] = field(default_factory=dict)
    opponent_name: str = ""
    win_loss_tie: tuple[int, int, int] = (0, 0, 0)

    # Category intelligence
    urgency_weights: dict = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)
    category_gaps: dict[str, float] = field(default_factory=dict)
    h2h_strategy: dict = field(default_factory=dict)  # weekly_h2h_strategy output

    # Schedule & matchup context
    todays_schedule: list[dict] = field(default_factory=list)
    confirmed_lineups: dict[str, list] = field(default_factory=dict)
    remaining_games_this_week: dict[str, int] = field(default_factory=dict)
    two_start_pitchers: list[int] = field(default_factory=list)
    opposing_pitchers: dict = field(default_factory=dict)
    team_strength: dict[str, dict] = field(default_factory=dict)
    park_factors: dict = field(default_factory=dict)
    weather: dict[str, dict] = field(default_factory=dict)

    # Player intelligence
    recent_form: dict[int, dict] = field(default_factory=dict)
    health_scores: dict[int, float] = field(default_factory=dict)
    news_flags: dict[int, str] = field(default_factory=dict)
    ownership_trends: dict[int, dict] = field(default_factory=dict)

    # Config
    scope: str = "rest_of_week"
    weeks_remaining: int = 16
    config: LeagueConfig = field(default_factory=LeagueConfig)
    data_timestamps: dict[str, str] = field(default_factory=dict)

    # League constraints
    adds_remaining_this_week: int = _MAX_WEEKLY_ADDS
    closer_count: int = 0
    il_stash_ids: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_optimizer_context(
    scope: str,
    yds,
    config: LeagueConfig | None = None,
    weeks_remaining: int = 16,
    park_factors: dict | None = None,
    user_team_name: str | None = None,
    roster: pd.DataFrame | None = None,
) -> OptimizerDataContext:
    """Build the complete data context for any optimizer scope.

    This is the ONLY place data is loaded and computed.
    All tabs consume this context — no independent DB queries.

    Parameters
    ----------
    scope : str
        One of "today", "rest_of_week", "rest_of_season".
    yds : YahooDataService
        The Yahoo data service singleton.
    config : LeagueConfig, optional
        League configuration. Defaults to LeagueConfig().
    weeks_remaining : int
        Weeks remaining in the fantasy season.
    park_factors : dict, optional
        Team → park factor mapping from bootstrap.
    user_team_name : str, optional
        User's team name. Auto-detected from roster if not provided.
    roster : pd.DataFrame, optional
        Pre-loaded roster. If None, fetched from yds.
    """
    if config is None:
        config = LeagueConfig()

    tracker = DataFreshnessTracker()
    ctx = OptimizerDataContext(scope=scope, weeks_remaining=weeks_remaining, config=config)
    ctx.park_factors = park_factors or {}

    # ── Step 1: Load enriched player pool ─────────────────────────────
    try:
        from src.database import load_player_pool

        ctx.player_pool = load_player_pool()
    except Exception:
        logger.warning("Failed to load player pool")
        ctx.player_pool = pd.DataFrame()

    # ── Step 2: Load roster ───────────────────────────────────────────
    if roster is not None and not roster.empty:
        ctx.roster = roster
    else:
        try:
            rosters = yds.get_rosters()
            if not rosters.empty and user_team_name:
                ctx.roster = rosters[rosters["team_name"] == user_team_name].copy()
            elif not rosters.empty:
                ctx.roster = rosters
        except Exception:
            logger.warning("Failed to load rosters from Yahoo")

    if "player_id" in ctx.roster.columns:
        ctx.user_roster_ids = ctx.roster["player_id"].dropna().astype(int).tolist()

    if not ctx.roster.empty:
        tracker.record(
            "yahoo_roster",
            ttl_hours=0.5,
            source_label="Yahoo Fantasy API",
            data_as_of="Current roster state",
        )

    # Track projections from player pool
    if not ctx.player_pool.empty:
        tracker.record(
            "projections",
            ttl_hours=24.0,
            source_label="Blended (Steamer/ZiPS/DC)",
            data_as_of="ROS projections",
        )

    # ── Step 3: Load free agents (enriched with player pool data) ─────
    try:
        _raw_fa = yds.get_free_agents()
        if not _raw_fa.empty and not ctx.player_pool.empty:
            # Enrich Yahoo FA data with player_id, is_hitter, stat columns
            # from the player pool (matched by name)
            _fa_name_col = "player_name" if "player_name" in _raw_fa.columns else "name"
            _pool_name_col = "name" if "name" in ctx.player_pool.columns else "player_name"
            # Columns to bring in from pool (skip those already in FA data)
            _pool_cols = [c for c in ctx.player_pool.columns if c not in _raw_fa.columns or c == _pool_name_col]
            _pool_for_merge = ctx.player_pool[_pool_cols].drop_duplicates(subset=[_pool_name_col], keep="first")
            ctx.free_agents = _raw_fa.merge(
                _pool_for_merge,
                left_on=_fa_name_col,
                right_on=_pool_name_col,
                how="left",
                suffixes=("", "_pool"),
            )
        else:
            ctx.free_agents = _raw_fa
        tracker.record("free_agents", ttl_hours=1.0, source_label="Yahoo Fantasy API", data_as_of="Current FA pool")
    except Exception:
        logger.warning("Failed to load free agents")

    # ── Step 4: Load live matchup ─────────────────────────────────────
    try:
        ctx.live_matchup = yds.get_matchup()
        tracker.record("live_matchup", ttl_hours=0.083, source_label="Yahoo Fantasy API", data_as_of="Live H2H scores")
    except Exception:
        logger.warning("Failed to load live matchup")

    # ── Step 5: Build my_totals, opp_totals, opponent_name ────────────
    _build_matchup_totals(ctx, yds, user_team_name)

    # ── Step 6: Category urgency from live W-L-T ──────────────────────
    try:
        from src.optimizer.category_urgency import compute_urgency_weights

        uw = compute_urgency_weights(ctx.live_matchup, config)
        ctx.urgency_weights = uw
    except Exception:
        logger.warning("Failed to compute urgency weights")

    # ── Step 7: Category gaps from live matchup ───────────────────────
    _build_category_gaps(ctx)

    # ── Step 7b: Weekly H2H strategy (winnable/protect/punt) ──────────
    _load_h2h_strategy(ctx, yds)

    # ── Step 8: Unified category weights ──────────────────────────────
    _build_unified_category_weights(ctx)

    # ── Step 9: Today's schedule ──────────────────────────────────────
    _load_schedule_data(ctx)

    # ── Step 10: Confirmed lineups ────────────────────────────────────
    _load_confirmed_lineups(ctx)
    if ctx.confirmed_lineups:
        tracker.record(
            "confirmed_lineups",
            ttl_hours=2.0,
            source_label="MLB Stats API",
            data_as_of="Today's confirmed lineups",
        )

    # ── Step 11: Remaining games this week per team ───────────────────
    _compute_remaining_games(ctx)
    if ctx.remaining_games_this_week:
        tracker.record("schedule", ttl_hours=2.0, source_label="MLB Stats API", data_as_of="Today's game schedule")

    # ── Step 12: Two-start pitchers ───────────────────────────────────
    _detect_two_start_pitchers(ctx)

    # ── Step 13: Opposing pitcher data ────────────────────────────────
    _load_opposing_pitchers(ctx)
    if ctx.opposing_pitchers:
        tracker.record(
            "opposing_pitchers",
            ttl_hours=2.0,
            source_label="MLB Stats API",
            data_as_of="Today's probable pitchers",
        )

    # ── Step 14: Team strength ────────────────────────────────────────
    _load_team_strength(ctx)

    # ── Step 15: Weather ──────────────────────────────────────────────
    _load_weather(ctx)
    if ctx.weather:
        tracker.record("weather", ttl_hours=2.0, source_label="Open-Meteo", data_as_of="Today's game-time forecast")

    # ── Step 16: Recent form ──────────────────────────────────────────
    _load_recent_form(ctx)
    if ctx.recent_form:
        tracker.record(
            "recent_form",
            ttl_hours=2.0,
            source_label="Season Stats (DB)",
            data_as_of="Last 14 games performance",
        )

    # ── Step 17: Health scores (single source) ────────────────────────
    _build_health_scores(ctx)
    if ctx.health_scores:
        tracker.record(
            "health_status",
            ttl_hours=1.0,
            source_label="Yahoo + ESPN",
            data_as_of="Current IL/DTD status",
        )

    # ── Step 18: News flags ───────────────────────────────────────────
    _load_news_flags(ctx)
    if ctx.news_flags:
        tracker.record(
            "news_flags", ttl_hours=1.0, source_label="MLB + ESPN + RotoWire", data_as_of="Recent player news"
        )

    # ── Step 19: Ownership trends ─────────────────────────────────────
    _load_ownership_trends(ctx)

    # ── Step 20: League constraints ────────────────────────────────────
    _compute_league_constraints(ctx, yds)

    ctx.data_timestamps = tracker.get_all()

    return ctx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_matchup_totals(ctx: OptimizerDataContext, yds, user_team_name: str | None) -> None:
    """Build my_totals and opp_totals from standings or live matchup."""
    # Try standings first
    try:
        standings = yds.get_standings()
        if not standings.empty and "category" in standings.columns:
            _cats = {
                c.lower()
                for c in standings["category"].unique()
                if c.lower() not in ("w-l-t", "record", "wins", "losses", "ties")
            }
            for _, row in standings.iterrows():
                team = str(row.get("team_name", ""))
                cat = str(row.get("category", "")).lower()
                val = row.get("total", 0)
                if cat in _cats:
                    if team == user_team_name:
                        try:
                            ctx.my_totals[cat] = float(val)
                        except (ValueError, TypeError):
                            pass
                    elif team == ctx.opponent_name:
                        try:
                            ctx.opp_totals[cat] = float(val)
                        except (ValueError, TypeError):
                            pass
    except Exception:
        logger.warning("Failed to load standings")

    # Override with live matchup if available (more current)
    if ctx.live_matchup and "categories" in ctx.live_matchup:
        w, l, t = 0, 0, 0
        my_live: dict[str, float] = {}
        opp_live: dict[str, float] = {}
        for mc in ctx.live_matchup["categories"]:
            cat = str(mc.get("cat", "")).lower()
            result = str(mc.get("result", "")).upper()
            if result == "WIN":
                w += 1
            elif result == "LOSS":
                l += 1
            elif result == "TIE":
                t += 1
            try:
                you_val = mc.get("you", "-")
                opp_val = mc.get("opp", "-")
                if you_val != "-":
                    my_live[cat] = float(you_val)
                if opp_val != "-":
                    opp_live[cat] = float(opp_val)
            except (ValueError, TypeError):
                pass
        if my_live:
            ctx.my_totals = my_live
        if opp_live:
            ctx.opp_totals = opp_live
        ctx.win_loss_tie = (w, l, t)
        ctx.opponent_name = str(ctx.live_matchup.get("opp_name", ""))


def _build_category_gaps(ctx: OptimizerDataContext) -> None:
    """Compute per-category gap values from live matchup."""
    if not ctx.my_totals or not ctx.opp_totals:
        return
    inverse_cats = {"l", "era", "whip"}
    for cat in ctx.my_totals:
        my_val = ctx.my_totals.get(cat, 0)
        opp_val = ctx.opp_totals.get(cat, 0)
        if cat in inverse_cats:
            # For inverse stats, positive gap = we're ahead (lower is better)
            ctx.category_gaps[cat] = opp_val - my_val
        else:
            ctx.category_gaps[cat] = my_val - opp_val


def _load_h2h_strategy(ctx: OptimizerDataContext, yds) -> None:
    """Load weekly H2H strategy: winnable, protect, and punt classifications.

    This bridges the gap between raw urgency (0-1 per category) and
    strategic decision-making (which categories to chase vs concede).
    """
    try:
        from src.weekly_h2h_strategy import compute_weekly_matchup_state

        state = compute_weekly_matchup_state(yds)
        ctx.h2h_strategy = state
        logger.info(
            "H2H strategy: winnable=%s, protect=%s, punt=%s",
            state.get("winnable_cats", []),
            state.get("protect_cats", []),
            state.get("punt_cats", []),
        )
    except Exception:
        logger.warning("Failed to load H2H strategy", exc_info=True)


def _build_unified_category_weights(ctx: OptimizerDataContext) -> None:
    """Build unified category weights: urgency * correlation adjustment.

    These weights are used by BOTH the LP solver and DCV engine,
    ensuring consistent category prioritization across all tabs.
    """
    urgency = ctx.urgency_weights.get("urgency", {})
    if not urgency:
        # Fallback: equal weights
        all_cats = list(ctx.config.hitting_categories) + list(ctx.config.pitching_categories)
        ctx.category_weights = {c: 1.0 for c in all_cats}
        return

    # Get H2H strategy classifications if available
    punt_cats = {c.lower() for c in ctx.h2h_strategy.get("punt_cats", [])}
    winnable_cats = {c.lower() for c in ctx.h2h_strategy.get("winnable_cats", [])}
    protect_cats = {c.lower() for c in ctx.h2h_strategy.get("protect_cats", [])}

    weights: dict[str, float] = {}
    for cat, urg in urgency.items():
        cat_l = cat.lower()

        if cat_l in punt_cats:
            # Punt: near-zero weight — don't waste lineup slots chasing uncloseable gaps
            w = 0.1
        elif cat_l in winnable_cats:
            # Winnable: boost weight — these are the categories that can flip the matchup
            w = 0.5 + urg * 1.3  # Range: 0.5 to ~1.8
        elif cat_l in protect_cats:
            # Protect: moderate weight — don't sacrifice leads
            w = 0.6 + urg * 0.4  # Range: 0.6 to ~1.0
        else:
            # Default: standard urgency transform
            w = 0.5 + urg  # Range: 0.5 to 1.5
        weights[cat] = w

    # Apply correlation dampening
    for (cat_a, cat_b), corr in _CORRELATION_PAIRS.items():
        if corr > 0.4:  # Only dampen positively correlated pairs
            cat_a_l = cat_a.lower()
            cat_b_l = cat_b.lower()
            urg_a = urgency.get(cat_a_l, urgency.get(cat_a, 0.5))
            urg_b = urgency.get(cat_b_l, urgency.get(cat_b, 0.5))
            # If one category in a correlated pair is already well-served
            # (urgency < 0.3 means we're winning), dampen its partner
            if urg_a < 0.3 and cat_b_l in weights:
                weights[cat_b_l] *= _CORR_DAMPEN_STRONG
            if urg_b < 0.3 and cat_a_l in weights:
                weights[cat_a_l] *= _CORR_DAMPEN_STRONG
            # If both are losing, slight boost (double opportunity)
            if urg_a > 0.7 and urg_b > 0.7:
                if cat_a_l in weights:
                    weights[cat_a_l] *= _CORR_BOOST_WEAK
                if cat_b_l in weights:
                    weights[cat_b_l] *= _CORR_BOOST_WEAK

    ctx.category_weights = weights


def _load_schedule_data(ctx: OptimizerDataContext) -> None:
    """Load target date MLB schedule (auto-advances to tomorrow if all games final)."""
    try:
        import statsapi

        from src.game_day import get_target_game_date

        target_date = get_target_game_date()
        games = statsapi.schedule(date=target_date)
        ctx.todays_schedule = games if games else []
    except Exception:
        logger.warning("Failed to load schedule")


def _load_confirmed_lineups(ctx: OptimizerDataContext) -> None:
    """Load confirmed batting orders for today's games."""
    if not ctx.todays_schedule:
        return
    try:
        from src.game_day import get_todays_lineups

        ctx.confirmed_lineups = get_todays_lineups(ctx.todays_schedule)
    except Exception:
        logger.warning("Failed to load confirmed lineups")


def _compute_remaining_games(ctx: OptimizerDataContext) -> None:
    """Compute remaining games this week per team."""
    if not ctx.todays_schedule:
        return
    try:
        import statsapi

        today = datetime.now(UTC)
        # Find end of fantasy week (Sunday)
        days_until_sunday = 6 - today.weekday()
        if days_until_sunday < 0:
            days_until_sunday = 0

        remaining: dict[str, int] = {}
        for day_offset in range(days_until_sunday + 1):
            check_date = today + timedelta(days=day_offset)
            date_str = check_date.strftime("%Y-%m-%d")
            try:
                games = statsapi.schedule(date=date_str)
                for g in games or []:
                    for team_key in ("home_name", "away_name"):
                        team = str(g.get(team_key, ""))
                        if team:
                            remaining[team] = remaining.get(team, 0) + 1
            except Exception:
                pass
        ctx.remaining_games_this_week = remaining
    except Exception:
        logger.warning("Failed to compute remaining games")


def _detect_two_start_pitchers(ctx: OptimizerDataContext) -> None:
    """Identify two-start pitchers for the current week."""
    if ctx.scope == "today":
        return  # Not relevant for single-day scope
    try:
        from src.two_start import identify_two_start_pitchers

        two_starters = identify_two_start_pitchers(
            days_ahead=7,
            player_pool=ctx.player_pool if not ctx.player_pool.empty else None,
        )
        # Map pitcher names to player IDs
        if not ctx.player_pool.empty and two_starters:
            name_to_id = dict(
                zip(
                    ctx.player_pool["name"].str.lower(),
                    ctx.player_pool["player_id"],
                    strict=False,
                )
            )
            for ts in two_starters:
                pname = str(ts.get("pitcher_name", "")).lower()
                pid = name_to_id.get(pname)
                if pid is not None:
                    ctx.two_start_pitchers.append(int(pid))
    except Exception:
        logger.warning("Failed to detect two-start pitchers")


def _load_opposing_pitchers(ctx: OptimizerDataContext) -> None:
    """Load opposing pitcher stats from DB."""
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            df = pd.read_sql("SELECT * FROM opp_pitcher_stats", conn)
            if not df.empty:
                for _, row in df.iterrows():
                    team = str(row.get("team", ""))
                    if team:
                        ctx.opposing_pitchers[team] = row.to_dict()
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to load opposing pitcher data")


def _load_team_strength(ctx: OptimizerDataContext) -> None:
    """Load team strength data via MatchupContextService."""
    try:
        from src.matchup_context import get_matchup_context

        mcs = get_matchup_context()
        # Load for all teams in roster
        teams_seen = set()
        if not ctx.roster.empty and "team" in ctx.roster.columns:
            teams_seen.update(ctx.roster["team"].dropna().unique())
        for team in teams_seen:
            try:
                ctx.team_strength[str(team)] = mcs.get_team_strength(str(team))
            except Exception:
                pass
    except Exception:
        logger.warning("Failed to load team strength")


def _load_weather(ctx: OptimizerDataContext) -> None:
    """Load weather data for today's games."""
    if ctx.scope != "today" and ctx.scope != "rest_of_week":
        return  # Weather only relevant for short-term scopes
    try:
        from src.matchup_context import get_matchup_context

        mcs = get_matchup_context()
        teams_seen = set()
        if not ctx.roster.empty and "team" in ctx.roster.columns:
            teams_seen.update(ctx.roster["team"].dropna().unique())
        for team in teams_seen:
            try:
                ctx.weather[str(team)] = mcs.get_weather(str(team))
            except Exception:
                pass
    except Exception:
        logger.warning("Failed to load weather")


def _load_recent_form(ctx: OptimizerDataContext) -> None:
    """Load L7/L14/L30 recent form for rostered players."""
    try:
        from src.game_day import get_player_recent_form_cached

        if ctx.roster.empty or "mlb_id" not in ctx.roster.columns:
            return
        for _, row in ctx.roster.iterrows():
            mlb_id = row.get("mlb_id")
            pid = row.get("player_id")
            if mlb_id is None or pid is None:
                continue
            try:
                mlb_id_int = int(mlb_id)
                if math.isnan(mlb_id_int):
                    continue
            except (ValueError, TypeError):
                continue
            try:
                form = get_player_recent_form_cached(mlb_id_int)
                if form:
                    ctx.recent_form[int(pid)] = form
            except Exception:
                pass
    except Exception:
        logger.warning("Failed to load recent form data")


def _build_health_scores(ctx: OptimizerDataContext) -> None:
    """Build health scores from enriched player pool (single source)."""
    if ctx.player_pool.empty or "health_score" not in ctx.player_pool.columns:
        return
    for _, row in ctx.player_pool.iterrows():
        pid = row.get("player_id")
        hs = row.get("health_score")
        if pid is not None and hs is not None:
            try:
                ctx.health_scores[int(pid)] = float(hs)
            except (ValueError, TypeError):
                pass


def _load_news_flags(ctx: OptimizerDataContext) -> None:
    """Load latest news headline per rostered player."""
    try:
        from src.database import get_connection

        if not ctx.user_roster_ids:
            return
        conn = get_connection()
        try:
            placeholders = ",".join("?" * len(ctx.user_roster_ids))
            df = pd.read_sql(
                f"SELECT player_name, headline FROM player_news "
                f"WHERE player_name IN (SELECT name FROM players WHERE player_id IN ({placeholders})) "
                f"ORDER BY published_date DESC",
                conn,
                params=ctx.user_roster_ids,
            )
            if not df.empty:
                # Deduplicate: keep latest per player
                seen = set()
                for _, row in df.iterrows():
                    pname = str(row.get("player_name", ""))
                    if pname and pname not in seen:
                        seen.add(pname)
                        # Map name back to player_id
                        match = ctx.player_pool[ctx.player_pool["name"] == pname]
                        if not match.empty:
                            pid = int(match.iloc[0]["player_id"])
                            ctx.news_flags[pid] = str(row.get("headline", ""))
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to load news flags")


def _load_ownership_trends(ctx: OptimizerDataContext) -> None:
    """Load ownership trend data for free agents."""
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            df = pd.read_sql(
                "SELECT player_id, percent_owned, date FROM ownership_trends ORDER BY date DESC",
                conn,
            )
            if not df.empty:
                # Compute 7-day delta per player
                for pid in df["player_id"].unique():
                    player_rows = df[df["player_id"] == pid].head(2)
                    if len(player_rows) >= 1:
                        current = float(player_rows.iloc[0].get("percent_owned", 0))
                        prev = (
                            float(player_rows.iloc[1].get("percent_owned", current))
                            if len(player_rows) > 1
                            else current
                        )
                        ctx.ownership_trends[int(pid)] = {
                            "pct_owned": current,
                            "delta_7d": current - prev,
                        }
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to load ownership trends")


def _compute_league_constraints(ctx: OptimizerDataContext, yds) -> None:
    """Compute league rule constraints (add budget, closer count)."""
    # Count adds this week
    try:
        txns = yds.get_transactions()
        if not txns.empty:
            # Filter to this week's adds by user
            now = datetime.now(UTC)
            week_start = now - timedelta(days=now.weekday())
            adds_this_week = 0
            for _, row in txns.iterrows():
                if str(row.get("type", "")).lower() == "add":
                    try:
                        ts = pd.to_datetime(row.get("timestamp", ""))
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        if ts >= week_start:
                            adds_this_week += 1
                    except Exception:
                        pass
            ctx.adds_remaining_this_week = max(0, _MAX_WEEKLY_ADDS - adds_this_week)
    except Exception:
        ctx.adds_remaining_this_week = _MAX_WEEKLY_ADDS  # Assume full budget on error

    # Count closers on roster
    if not ctx.roster.empty:
        closer_count = 0
        for _, row in ctx.roster.iterrows():
            pid = row.get("player_id")
            # Check is_closer from enriched pool
            pool_match = (
                ctx.player_pool[ctx.player_pool["player_id"] == pid]
                if not ctx.player_pool.empty and pid is not None
                else pd.DataFrame()
            )
            if not pool_match.empty:
                is_closer = pool_match.iloc[0].get("is_closer", False)
                sv_proj = pool_match.iloc[0].get("sv", 0)
                if is_closer or (sv_proj is not None and float(sv_proj or 0) >= _CLOSER_SV_THRESHOLD):
                    closer_count += 1
        ctx.closer_count = closer_count

    # Build IL stash set
    try:
        from src.alerts import IL_STASH_NAMES

        il_stash_names = set(IL_STASH_NAMES)
    except Exception:
        il_stash_names = {"Shane Bieber", "Spencer Strider"}

    # Add players with ESPN return dates within 2 weeks
    try:
        from src.alerts import _get_il_return_dates

        return_dates = _get_il_return_dates(ctx.roster)
        two_weeks = datetime.now(UTC) + timedelta(weeks=2)
        for key, ret_date in return_dates.items():
            if ret_date <= two_weeks:
                if isinstance(key, int):
                    ctx.il_stash_ids.add(key)
                elif isinstance(key, str):
                    il_stash_names.add(key)
    except Exception:
        pass

    # Map IL stash names to player IDs
    if not ctx.roster.empty and "name" in ctx.roster.columns:
        for _, row in ctx.roster.iterrows():
            pname = str(row.get("name", row.get("player_name", "")))
            if pname in il_stash_names:
                pid = row.get("player_id")
                if pid is not None:
                    ctx.il_stash_ids.add(int(pid))


# ---------------------------------------------------------------------------
# Scope-specific projection scaling
# ---------------------------------------------------------------------------


def scale_projections_for_scope(
    ctx: OptimizerDataContext,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """Scale projections based on scope selection.

    Parameters
    ----------
    ctx : OptimizerDataContext
        The fully-built context.
    roster : pd.DataFrame
        Roster with projection columns to scale.

    Returns
    -------
    pd.DataFrame
        Copy of roster with projections scaled for the selected scope.
    """
    df = roster.copy()
    counting_cats = ["r", "hr", "rbi", "sb", "w", "l", "sv", "k"]
    rate_cats = ["avg", "obp", "era", "whip"]

    if ctx.scope == "today":
        # Per-game scaling: counting stats / season_games, adjusted by volume
        for col in counting_cats:
            if col in df.columns:
                df[col] = df[col].fillna(0) / _SEASON_GAMES

    elif ctx.scope == "rest_of_week":
        # Scale by remaining games this week
        for idx, row in df.iterrows():
            team = str(row.get("team", ""))
            remaining = ctx.remaining_games_this_week.get(team, 0)
            # Use full team names as fallback
            if remaining == 0:
                for full_name, count in ctx.remaining_games_this_week.items():
                    if team.upper() in full_name.upper():
                        remaining = count
                        break
            season_rate = remaining / max(_SEASON_GAMES, 1)
            pid = row.get("player_id")

            # Two-start pitcher bonus (guard against NaN player_id from pandas)
            try:
                is_two_start = pid is not None and int(pid) in ctx.two_start_pitchers
            except (ValueError, TypeError):
                is_two_start = False
            pitcher_mult = _TWO_START_COUNTING_MULT if is_two_start else 1.0

            for col in counting_cats:
                if col in df.columns:
                    base = float(row.get(col, 0) or 0)
                    if row.get("is_hitter", True) or col in ("r", "hr", "rbi", "sb"):
                        df.at[idx, col] = base * season_rate
                    else:
                        # Pitching counting stats
                        df.at[idx, col] = base * season_rate * pitcher_mult

    elif ctx.scope == "rest_of_season":
        # Full season projections — apply schedule strength adjustment
        for idx, row in df.iterrows():
            team = str(row.get("team", ""))
            ts = ctx.team_strength.get(team, {})
            if ts:
                # Hitters: teams facing weak pitching get boost
                opp_fip = ts.get("fip", 4.00)
                if row.get("is_hitter", True):
                    if opp_fip > _OPP_PITCHER_FIP_WEAK:
                        mult = _OPP_PITCHER_WEAK_MULT
                    elif opp_fip < _OPP_PITCHER_FIP_STRONG:
                        mult = _OPP_PITCHER_STRONG_MULT
                    else:
                        mult = 1.0
                    for col in counting_cats:
                        if col in df.columns and col in ("r", "hr", "rbi", "sb"):
                            df.at[idx, col] = float(row.get(col, 0) or 0) * mult

    # Apply opposing pitcher adjustment for today/week scopes
    if ctx.scope in ("today", "rest_of_week") and ctx.opposing_pitchers:
        # Build team→opponent mapping from today's schedule
        _team_opponent: dict[str, str] = {}
        for _game in ctx.todays_schedule or []:
            _h = str(_game.get("home_name", ""))
            _a = str(_game.get("away_name", ""))
            if _h and _a:
                _team_opponent[_h] = _a
                _team_opponent[_a] = _h
        for idx, row in df.iterrows():
            if not row.get("is_hitter", True):
                continue
            team = str(row.get("team", ""))
            opp_team = _team_opponent.get(team, "")
            if not opp_team or opp_team not in ctx.opposing_pitchers:
                continue
            opp_data = ctx.opposing_pitchers[opp_team]
            opp_fip = float(opp_data.get("fip", 4.00) or 4.00)
            if opp_fip < _OPP_PITCHER_FIP_STRONG:
                mult = _OPP_PITCHER_STRONG_MULT
            elif opp_fip > _OPP_PITCHER_FIP_WEAK:
                mult = _OPP_PITCHER_WEAK_MULT
            else:
                mult = 1.0
            if mult != 1.0:
                for col in counting_cats:
                    if col in df.columns and col in ("r", "hr", "rbi", "sb"):
                        df.at[idx, col] = float(df.at[idx, col] or 0) * mult

    return df


def get_recent_form_weight(scope: str, n_games: int | None = None) -> float:
    """Return scope-specific recent form weight, optionally scaled by sample size.

    With n_games: scales between min_weight and max_weight based on game count.
    Without n_games: returns the fixed scope-based weight (backward compatible).

    Scaling: linear interpolation from min_weight at 7 games to max_weight at 14 games.
    Below 7 games: returns 0.0 (insufficient sample).
    Above 14 games: capped at max_weight.
    """
    # Scope max weights
    if scope == "today":
        max_weight = _RECENT_FORM_WEIGHT_TODAY
    elif scope == "rest_of_week":
        max_weight = _RECENT_FORM_WEIGHT_WEEK
    else:
        max_weight = _RECENT_FORM_WEIGHT_SEASON

    # Backward compatible: no n_games returns fixed weight
    if n_games is None:
        return max_weight

    # Dynamic scaling by sample size
    _MIN_GAMES = 7
    _MAX_GAMES = 14
    min_weight = max_weight * 0.5

    if n_games < _MIN_GAMES:
        return 0.0
    if n_games >= _MAX_GAMES:
        return max_weight

    # Linear interpolation between min_weight (at 7 games) and max_weight (at 14 games)
    t = (n_games - _MIN_GAMES) / (_MAX_GAMES - _MIN_GAMES)
    return min_weight + t * (max_weight - min_weight)
