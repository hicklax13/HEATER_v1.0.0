"""Player Card — data assembly for the full player card dialog.

Pure functions only — no Streamlit dependency. All data is fetched from
the SQLite database and returned as plain dicts for rendering.
"""

from datetime import UTC, datetime

import pandas as pd

from src.database import get_connection
from src.injury_model import compute_health_score, get_injury_badge
from src.valuation import LeagueConfig

_LC = LeagueConfig()
_HITTING_CATS = _LC.hitting_categories  # ["R", "HR", "RBI", "SB", "AVG", "OBP"]
_PITCHING_CATS = _LC.pitching_categories  # ["W", "L", "SV", "K", "ERA", "WHIP"]
_RATE_STATS = {"AVG", "OBP", "ERA", "WHIP"}

_MLB_HEADSHOT_TEMPLATE = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "d_people:generic:headshot:67:current.png/"
    "w_213,q_auto:best/v1/people/{mlb_id}/headshot/67/current"
)


def _get_headshot_url(mlb_id) -> str:
    """Return MLB static headshot URL, or empty string if no mlb_id."""
    if mlb_id is None or mlb_id == 0:
        return ""
    try:
        mid = int(mlb_id)
    except (ValueError, TypeError):
        return ""
    if mid == 0:
        return ""
    return _MLB_HEADSHOT_TEMPLATE.format(mlb_id=mid)


def _compute_age(birth_date) -> int | None:
    """Compute age from birth_date string (YYYY-MM-DD)."""
    if not birth_date:
        return None
    try:
        bd = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d")
        today = datetime.now(UTC)
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except (ValueError, TypeError):
        return None


def _dedup_news(news_items: list[dict], max_items: int = 5) -> list[dict]:
    """Deduplicate news by headline (case-insensitive), return newest first."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in news_items:
        key = (item.get("headline") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    # Sort by published_at descending
    unique.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return unique[:max_items]


def _format_news_datetime(published_at) -> str:
    """Format a published_at value into human-readable datetime string."""
    if not published_at:
        return ""
    try:
        dt_str = str(published_at)
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return str(published_at)[:16] if published_at else ""


def _compute_radar_percentiles(
    player_stats: dict,
    position: str,
    is_hitter: bool,
) -> dict:
    """Compute 0-100 percentile ranks for a player vs league and MLB pool.

    Returns dict with keys: "player", "league_avg", "mlb_avg".
    Each value is a dict of {category: percentile_value}.
    """
    cats = _HITTING_CATS if is_hitter else _PITCHING_CATS

    conn = get_connection()
    try:
        # Load blended projections for all players of same type
        pool = pd.read_sql_query(
            """SELECT p.player_id, p.positions,
                      proj.r, proj.hr, proj.rbi, proj.sb, proj.avg, proj.obp,
                      proj.w, proj.l, proj.sv, proj.k, proj.era, proj.whip
               FROM players p
               JOIN projections proj ON p.player_id = proj.player_id
               WHERE proj.system = 'blended' AND p.is_hitter = ?""",
            conn,
            params=[1 if is_hitter else 0],
        )

        # Load league-rostered players
        rostered_ids = pd.read_sql_query("SELECT DISTINCT player_id FROM league_rosters", conn)["player_id"].tolist()
    finally:
        conn.close()

    if pool.empty:
        empty = {c: 50 for c in cats}
        return {"player": empty, "league_avg": empty, "mlb_avg": empty}

    # Compute percentiles for each category
    result = {"player": {}, "league_avg": {}, "mlb_avg": {}}

    league_pool = pool[pool["player_id"].isin(rostered_ids)]

    for cat in cats:
        col = cat.lower()
        if col not in pool.columns:
            result["player"][cat] = 50
            result["league_avg"][cat] = 50
            result["mlb_avg"][cat] = 50
            continue

        mlb_vals = pd.to_numeric(pool[col], errors="coerce").dropna()
        league_vals = (
            pd.to_numeric(league_pool[col], errors="coerce").dropna()
            if not league_pool.empty
            else pd.Series(dtype=float)
        )

        player_val = player_stats.get(cat) or player_stats.get(col, 0)
        try:
            player_val = float(player_val)
        except (ValueError, TypeError):
            player_val = 0.0

        # For inverse stats (ERA, WHIP, L), lower is better
        inverse = cat in {"ERA", "WHIP", "L"}

        # Player percentile vs MLB
        if len(mlb_vals) > 0:
            if inverse:
                pct = (mlb_vals >= player_val).sum() / len(mlb_vals) * 100
            else:
                pct = (mlb_vals <= player_val).sum() / len(mlb_vals) * 100
            result["player"][cat] = min(max(round(pct), 0), 100)
        else:
            result["player"][cat] = 50

        # League average
        if len(league_vals) > 0:
            result["league_avg"][cat] = (
                round(float(league_vals.mean()), 3) if cat in _RATE_STATS else int(league_vals.mean())
            )
        else:
            result["league_avg"][cat] = 0

        # MLB average
        if len(mlb_vals) > 0:
            result["mlb_avg"][cat] = round(float(mlb_vals.mean()), 3) if cat in _RATE_STATS else int(mlb_vals.mean())
        else:
            result["mlb_avg"][cat] = 0

    return result


def _build_sparkline_data(historical: list[dict]) -> dict:
    """Extract 3-year trend arrays per category from historical stats.

    Returns dict like {"R": [122, 101, 62], "HR": [29, 23, 15], ...}.
    Values are in chronological order (oldest first).
    """
    if not historical:
        return {}

    # Historical comes newest-first; reverse for chronological
    chron = list(reversed(historical))
    all_cats = _HITTING_CATS + _PITCHING_CATS
    trends: dict[str, list] = {}

    for cat in all_cats:
        vals = []
        for season_data in chron:
            v = season_data.get(cat)
            if v is None:
                v = season_data.get(cat.lower())
            vals.append(v)
        # Only include if at least one non-None value
        if any(v is not None for v in vals):
            trends[cat] = vals

    return trends


def build_player_card_data(player_id: int) -> dict:
    """Assemble all data for a player card dialog.

    Pure function — no Streamlit dependency. Returns a dict with sections:
    profile, projections, historical, advanced, injury_history, rankings,
    radar, trends, news, prospect.

    Returns a safe empty-ish dict if the player is not found.
    """
    conn = get_connection()
    try:
        # ── Profile ─────────────────────────────────────────
        player = pd.read_sql_query(
            "SELECT * FROM players WHERE player_id = ?",
            conn,
            params=[player_id],
        )
        if player.empty:
            return {
                "profile": {
                    "name": f"Player {player_id}",
                    "team": "",
                    "positions": "",
                    "bats": "",
                    "throws": "",
                    "age": None,
                    "headshot_url": "",
                    "health_score": 0,
                    "health_label": "Unknown",
                    "tags": [],
                },
                "projections": {"blended": {}, "systems": {}},
                "historical": [],
                "advanced": {},
                "injury_history": [],
                "rankings": {},
                "radar": {"player": {}, "league_avg": {}, "mlb_avg": {}},
                "trends": {},
                "news": [],
                "prospect": None,
            }

        p = player.iloc[0]
        name = p["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        mlb_id = p.get("mlb_id")
        is_hitter = bool(p.get("is_hitter", 1))
        positions = p.get("positions", "") or ""

        # Health score from injury history
        injury_df = pd.read_sql_query(
            "SELECT * FROM injury_history WHERE player_id = ?",
            conn,
            params=[player_id],
        )
        if not injury_df.empty:
            gp = injury_df["games_played"].tolist()
            ga = injury_df["games_available"].tolist()
            health_score = compute_health_score(gp, ga)
            _badge_html, health_label = get_injury_badge(health_score)
        else:
            health_score = 0.85
            health_label = "No Data"

        # Tags
        tags_df = pd.read_sql_query(
            "SELECT tag FROM player_tags WHERE player_id = ?",
            conn,
            params=[player_id],
        )
        tags = tags_df["tag"].tolist() if not tags_df.empty else []

        profile = {
            "name": name,
            "team": p.get("team", "") or "",
            "positions": positions,
            "bats": p.get("bats", "") or "",
            "throws": p.get("throws", "") or "",
            "age": _compute_age(p.get("birth_date")),
            "headshot_url": _get_headshot_url(mlb_id),
            "health_score": round(health_score, 2),
            "health_label": health_label,
            "tags": tags,
        }

        # ── Projections ────────────────────────────────────
        proj_df = pd.read_sql_query(
            "SELECT * FROM projections WHERE player_id = ?",
            conn,
            params=[player_id],
        )
        stat_cols = [
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
            "pa",
            "ab",
            "ip",
            "fip",
            "xfip",
            "siera",
            "stuff_plus",
            "location_plus",
            "pitching_plus",
        ]

        projections = {"blended": {}, "systems": {}}
        for _, row in proj_df.iterrows():
            system = row.get("system", "unknown")
            stats = {}
            for col in stat_cols:
                if col in row.index:
                    val = row[col]
                    try:
                        stats[col.upper()] = float(val) if val is not None and pd.notna(val) else None
                    except (ValueError, TypeError):
                        stats[col.upper()] = None
            if system == "blended":
                projections["blended"] = stats
            else:
                projections["systems"][system] = stats

        # ── Historical Stats ───────────────────────────────
        hist_df = pd.read_sql_query(
            "SELECT * FROM season_stats WHERE player_id = ? ORDER BY season DESC",
            conn,
            params=[player_id],
        )
        historical = []
        for _, row in hist_df.iterrows():
            season_data = {"season": int(row.get("season", 0))}
            for col in stat_cols + ["games_played"]:
                if col in row.index:
                    val = row[col]
                    try:
                        season_data[col.upper()] = float(val) if val is not None and pd.notna(val) else None
                    except (ValueError, TypeError):
                        season_data[col.upper()] = None
            historical.append(season_data)

        # ── Advanced Metrics ───────────────────────────────
        advanced = {}
        if not is_hitter:
            # Use latest season or blended projection
            source = historical[0] if historical else projections.get("blended", {})
            for metric in ["FIP", "XFIP", "SIERA", "STUFF_PLUS", "LOCATION_PLUS", "PITCHING_PLUS"]:
                advanced[metric] = source.get(metric)

        # ── Injury History ─────────────────────────────────
        injury_history = []
        for _, row in injury_df.iterrows():
            injury_history.append(
                {
                    "season": int(row.get("season", 0)),
                    "GP": int(row.get("games_played", 0)),
                    "GA": int(row.get("games_available", 0)),
                    "IL_stints": int(row.get("il_stints", 0)),
                    "IL_days": int(row.get("il_days", 0)),
                }
            )

        # ── Rankings & ADP ─────────────────────────────────
        ecr = pd.read_sql_query(
            "SELECT * FROM ecr_consensus WHERE player_id = ?",
            conn,
            params=[player_id],
        )
        adp = pd.read_sql_query(
            "SELECT * FROM adp WHERE player_id = ?",
            conn,
            params=[player_id],
        )

        rankings = {}
        if not ecr.empty:
            e = ecr.iloc[0]
            rankings["consensus_rank"] = e.get("consensus_rank")
            rankings["rank_range"] = [e.get("rank_min"), e.get("rank_max")]
            rankings["rank_stddev"] = e.get("rank_stddev")
            rankings["n_sources"] = e.get("n_sources")
        if not adp.empty:
            a = adp.iloc[0]
            rankings["yahoo_adp"] = a.get("yahoo_adp")
            rankings["fantasypros_adp"] = a.get("fantasypros_adp")
            rankings["nfbc_adp"] = a.get("nfbc_adp")
            rankings["composite_adp"] = a.get("adp")

        # ── Radar Percentiles ──────────────────────────────
        blended_stats = projections.get("blended", {})
        radar = _compute_radar_percentiles(blended_stats, positions, is_hitter)

        # ── Sparkline Trends ───────────────────────────────
        trends = _build_sparkline_data(historical)

        # ── News ───────────────────────────────────────────
        news_df = pd.read_sql_query(
            "SELECT * FROM player_news WHERE player_id = ? ORDER BY published_at DESC",
            conn,
            params=[player_id],
        )
        raw_news = []
        for _, row in news_df.iterrows():
            raw_news.append(
                {
                    "headline": row.get("headline", ""),
                    "source": row.get("source", ""),
                    "published_at": row.get("published_at"),
                    "date_display": _format_news_datetime(row.get("published_at")),
                    "sentiment": row.get("sentiment_score", 0.0) or 0.0,
                    "news_type": row.get("news_type", "general"),
                    "il_status": row.get("il_status", ""),
                }
            )
        news = _dedup_news(raw_news)

        # ── Prospect Data ──────────────────────────────────
        prospect = None
        if mlb_id is not None and mlb_id != 0:
            try:
                _mlb_int = int(mlb_id)
            except (ValueError, TypeError):
                _mlb_int = 0
            prospect_df = pd.read_sql_query(
                "SELECT * FROM prospect_rankings WHERE mlb_id = ?",
                conn,
                params=[_mlb_int],
            )
            if not prospect_df.empty:
                pr = prospect_df.iloc[0]
                prospect = {
                    "fg_rank": pr.get("fg_rank"),
                    "fg_fv": pr.get("fg_fv"),
                    "fg_eta": pr.get("fg_eta"),
                    "fg_risk": pr.get("fg_risk"),
                    "hit": {"present": pr.get("hit_present"), "future": pr.get("hit_future")},
                    "game_power": {"present": pr.get("game_present"), "future": pr.get("game_future")},
                    "raw_power": {"present": pr.get("raw_present"), "future": pr.get("raw_future")},
                    "speed": pr.get("speed"),
                    "field": pr.get("field"),
                    "control": {"present": pr.get("ctrl_present"), "future": pr.get("ctrl_future")},
                    "scouting_report": pr.get("scouting_report", ""),
                    "tldr": pr.get("tldr", ""),
                    "milb_stats": {
                        "AVG": pr.get("milb_avg"),
                        "OBP": pr.get("milb_obp"),
                        "SLG": pr.get("milb_slg"),
                        "HR": pr.get("milb_hr"),
                        "SB": pr.get("milb_sb"),
                        "K%": pr.get("milb_k_pct"),
                        "BB%": pr.get("milb_bb_pct"),
                    },
                    "milb_level": pr.get("milb_level"),
                    "readiness_score": pr.get("readiness_score"),
                }

    finally:
        conn.close()

    return {
        "profile": profile,
        "projections": projections,
        "historical": historical,
        "advanced": advanced,
        "injury_history": injury_history,
        "rankings": rankings,
        "radar": radar,
        "trends": trends,
        "news": news,
        "prospect": prospect,
    }
