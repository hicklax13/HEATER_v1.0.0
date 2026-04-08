# src/prospect_engine.py
"""Prospect Rankings Engine — FanGraphs Board API + MLB Stats API MiLB stats.

Computes MLB Readiness Score (0-100) combining FV, age-level performance,
ETA proximity, and risk factor.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import pandas as pd

logger = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

_CURRENT_SEASON = datetime.now(UTC).year

_LEVEL_AVG_WOBA = {"AAA": 0.330, "AA": 0.310, "High-A": 0.300, "A": 0.290, "A+": 0.300}
_LEVEL_AVG_AGE = {"AAA": 25, "AA": 23, "High-A": 22, "A": 21, "A+": 22}

_FG_BOARD_URL = "https://www.fangraphs.com/api/prospects/board/data?draft=false&season={season}"
_MLB_MILB_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{mlb_id}/stats?stats=yearByYear&leagueListId=milb_all"

_RISK_MAP = {"Low": 1.0, "Medium": 0.8, "High": 0.6, "Extreme": 0.4}
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/prospects",
}

# -- Fallback static list (last resort) --------------------------------------

_STATIC_PROSPECTS = [
    {"rank": 1, "name": "Roki Sasaki", "team": "LAD", "position": "SP", "fg_eta": "2025", "fg_fv": 80},
    {"rank": 2, "name": "Roman Anthony", "team": "BOS", "position": "OF", "fg_eta": "2025", "fg_fv": 70},
    {"rank": 3, "name": "Travis Bazzana", "team": "CLE", "position": "2B", "fg_eta": "2026", "fg_fv": 65},
    {"rank": 4, "name": "Charlie Condon", "team": "COL", "position": "3B", "fg_eta": "2027", "fg_fv": 65},
    {"rank": 5, "name": "Jac Caglianone", "team": "KC", "position": "1B/SP", "fg_eta": "2027", "fg_fv": 65},
    {"rank": 6, "name": "Sebastian Walcott", "team": "TEX", "position": "SS", "fg_eta": "2027", "fg_fv": 65},
    {"rank": 7, "name": "Kristian Campbell", "team": "BOS", "position": "SS", "fg_eta": "2026", "fg_fv": 60},
    {"rank": 8, "name": "Marcelo Mayer", "team": "BOS", "position": "SS", "fg_eta": "2026", "fg_fv": 60},
    {"rank": 9, "name": "JJ Wetherholt", "team": "PIT", "position": "2B", "fg_eta": "2026", "fg_fv": 60},
    {"rank": 10, "name": "Coby Mayo", "team": "BAL", "position": "3B", "fg_eta": "2025", "fg_fv": 55},
    {"rank": 11, "name": "Nick Kurtz", "team": "OAK", "position": "1B", "fg_eta": "2027", "fg_fv": 60},
    {"rank": 12, "name": "James Wood", "team": "WSH", "position": "OF", "fg_eta": "2025", "fg_fv": 55},
    {"rank": 13, "name": "Bubba Chandler", "team": "PIT", "position": "SS/SP", "fg_eta": "2026", "fg_fv": 60},
    {"rank": 14, "name": "Chase Burns", "team": "CIN", "position": "SP", "fg_eta": "2026", "fg_fv": 60},
    {"rank": 15, "name": "Tink Hence", "team": "STL", "position": "SP", "fg_eta": "2026", "fg_fv": 55},
    {"rank": 16, "name": "Samuel Basallo", "team": "BAL", "position": "C", "fg_eta": "2026", "fg_fv": 55},
    {"rank": 17, "name": "Braden Montgomery", "team": "BOS", "position": "OF", "fg_eta": "2027", "fg_fv": 60},
    {"rank": 18, "name": "Leodalis De Vries", "team": "TEX", "position": "SS", "fg_eta": "2028", "fg_fv": 60},
    {"rank": 19, "name": "Colt Emerson", "team": "CLE", "position": "SS", "fg_eta": "2028", "fg_fv": 60},
    {"rank": 20, "name": "Ethan Salas", "team": "SD", "position": "C", "fg_eta": "2026", "fg_fv": 55},
]


# -- Readiness score components -----------------------------------------------


def _fv_normalized(fv: int | None) -> float:
    """Normalize FV (20-80 scale) to 0-100."""
    if fv is None:
        return 50.0
    return max(0.0, min(100.0, (fv - 20) / 60 * 100))


def _eta_proximity(eta: str | None) -> float:
    """Score ETA proximity (0-100). Current season or past = 100."""
    if not eta:
        return 50.0
    try:
        eta_year = int(eta)
    except (ValueError, TypeError):
        return 50.0
    diff = eta_year - _CURRENT_SEASON
    if diff <= 0:
        return 100.0
    mapping = {1: 75.0, 2: 50.0, 3: 25.0}
    return mapping.get(diff, 0.0)


def _risk_factor(risk: str | None) -> float:
    """Map risk level to multiplier."""
    if not risk:
        return 0.7
    return _RISK_MAP.get(risk, 0.7)


def _age_level_performance(row: dict) -> float:
    """Compute age-level performance score (0-100).
    wOBA proxy = (OBP * 1.2 + SLG * 0.8) / 2
    Bonus/penalty for age vs level average.
    """
    obp = row.get("milb_obp")
    slg = row.get("milb_slg")
    level = row.get("milb_level") or ""
    age = row.get("age")

    if obp is None or slg is None:
        return 50.0  # neutral when no MiLB data

    woba_proxy = obp * 0.92 + slg * 0.47 - 0.08
    level_avg = _LEVEL_AVG_WOBA.get(level, 0.310)

    # Scale: each 0.020 above avg = +10 points from 50 baseline
    perf_score = 50.0 + (woba_proxy - level_avg) / 0.020 * 10.0

    # Age adjustment: younger than level avg = bonus
    if age is not None:
        level_age = _LEVEL_AVG_AGE.get(level, 23)
        age_bonus = (level_age - age) * 5.0  # +5 per year younger
        perf_score += age_bonus

    return max(0.0, min(100.0, perf_score))


def compute_mlb_readiness_score(row: dict) -> float:
    """Compute MLB Readiness Score (0-100).

    Formula (additive, per spec):
        0.40 * fv_normalized          (FV 80->100, FV 20->0)
      + 0.25 * age_level_performance  (wOBA vs level avg, age-adjusted, 0-100)
      + 0.20 * eta_proximity          (2026->100, 2027->75, 2028->50, 2029->25)
      + 0.15 * risk_factor_scaled     (Low->100, Med->80, High->60, Extreme->40)
    """
    fv = _fv_normalized(row.get("fg_fv"))
    perf = _age_level_performance(row)
    eta = _eta_proximity(row.get("fg_eta"))
    risk = _risk_factor(row.get("fg_risk"))

    # Additive: risk_factor is 0-1, scale to 0-100 for the 15% weight
    score = 0.40 * fv + 0.25 * perf + 0.20 * eta + 0.15 * (risk * 100)
    return round(max(0.0, min(100.0, score)), 1)


# -- FanGraphs Board API ------------------------------------------------------


def _parse_fg_prospects(data: list[dict]) -> pd.DataFrame:
    """Parse FanGraphs Board API response into a DataFrame."""
    rows = []
    for i, item in enumerate(data):
        # Name: API returns FirstName + LastName separately
        first = item.get("FirstName", item.get("PlayerName", ""))
        last = item.get("LastName", "")
        name = f"{first} {last}".strip() if last else first

        # FV: API uses FV_Current
        try:
            fv_raw = item.get("FV_Current", item.get("FV", "0"))
            fv = int(fv_raw) if fv_raw else 0
        except (ValueError, TypeError):
            fv = 0

        # ETA: API uses ETA_Current
        eta_raw = item.get("ETA_Current", item.get("ETA", ""))
        eta = str(int(eta_raw)) if eta_raw and eta_raw != "" else ""

        # Risk: API uses Variance
        risk = item.get("Variance", item.get("Risk", ""))

        # Age: API returns as string like "22.0194444"
        age_raw = item.get("Age")
        age = None
        if age_raw is not None:
            try:
                age = int(float(age_raw))
            except (ValueError, TypeError):
                pass

        # Rank: API uses Ovr_Rank
        rank = item.get("Ovr_Rank", i + 1)

        # Scouting tools: format is "present / future" string
        def _parse_grade(val):
            """Parse '45 / 60' format, return int or None."""
            if val is None or val == "":
                return None
            if isinstance(val, (int, float)):
                return int(val)
            parts = str(val).split("/")
            try:
                return int(parts[0].strip())
            except (ValueError, IndexError):
                return None

        def _parse_future_grade(val):
            """Parse '45 / 60' format, return future grade or None."""
            if val is None or val == "":
                return None
            if isinstance(val, (int, float)):
                return int(val)
            parts = str(val).split("/")
            try:
                return int(parts[-1].strip())
            except (ValueError, IndexError):
                return None

        row = {
            "name": name,
            "team": item.get("Team", ""),
            "position": item.get("Position", ""),
            "fg_rank": rank,
            "fg_fv": fv,
            "fg_eta": eta,
            "fg_risk": risk,
            "mlb_id": item.get("PlayerId", item.get("minorMasterId")),
            "age": age,
            "hit_present": _parse_grade(item.get("Hit")) if item.get("Hit") else _parse_grade(item.get("pHit")),
            "hit_future": _parse_future_grade(item.get("Hit")) if item.get("Hit") else _parse_grade(item.get("fHit")),
            "game_present": _parse_grade(item.get("Game")) if item.get("Game") else _parse_grade(item.get("pGame")),
            "game_future": _parse_future_grade(item.get("Game"))
            if item.get("Game")
            else _parse_grade(item.get("fGame")),
            "raw_present": _parse_grade(item.get("Raw")) if item.get("Raw") else _parse_grade(item.get("pRaw")),
            "raw_future": _parse_future_grade(item.get("Raw")) if item.get("Raw") else _parse_grade(item.get("fRaw")),
            "speed": _parse_grade(item.get("Spd")) if item.get("Spd") else _parse_grade(item.get("pSpd")),
            "field": _parse_grade(item.get("Fld")),
            "ctrl_present": _parse_grade(item.get("CMD")) if item.get("CMD") else _parse_grade(item.get("pCtl")),
            "ctrl_future": _parse_future_grade(item.get("CMD")) if item.get("CMD") else _parse_grade(item.get("fCtl")),
            "scouting_report": item.get("Summary", item.get("Report", "")),
            "tldr": item.get("TLDR", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_fg_board(season: int = _CURRENT_SEASON) -> pd.DataFrame:
    """Fetch prospect data from FanGraphs Board API.
    Returns DataFrame or empty DataFrame on failure.
    Tries current season first, falls back to previous season.
    """
    try:
        import requests

        url = _FG_BOARD_URL.format(season=season)
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        # If current season returns error, try previous season
        if resp.status_code == 404 and season == _CURRENT_SEASON:
            logger.info("FG Board API: season %d not available, trying %d", season, season - 1)
            url = _FG_BOARD_URL.format(season=season - 1)
            resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("FanGraphs Board API returned empty/non-list response")
            return pd.DataFrame()
        return _parse_fg_prospects(data)
    except Exception:
        logger.warning("FanGraphs Board API fetch failed", exc_info=True)
        return pd.DataFrame()


# Backward-compat alias
fetch_fangraphs_prospects = fetch_fg_board


# -- MLB Stats API MiLB stats -------------------------------------------------


def fetch_milb_stats(mlb_ids: list[int]) -> pd.DataFrame:
    """Fetch MiLB stats for a list of MLB IDs.
    Returns DataFrame with one row per player (most recent MiLB season).
    """
    rows = []
    for mlb_id in mlb_ids:
        if mlb_id is None:
            continue
        try:
            import requests

            url = _MLB_MILB_STATS_URL.format(mlb_id=mlb_id)
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            stats_list = data.get("stats", [])
            if not stats_list:
                continue
            # Find the most recent MiLB season
            splits = stats_list[0].get("splits", [])
            if not splits:
                continue
            latest = splits[-1]  # most recent
            stat = latest.get("stat", {})
            league = latest.get("league", {})
            level_name = league.get("name", "")
            # Determine if hitter or pitcher stats
            row = {
                "mlb_id": mlb_id,
                "milb_level": _normalize_level(level_name),
                "milb_avg": _safe_float(stat.get("avg")),
                "milb_obp": _safe_float(stat.get("obp")),
                "milb_slg": _safe_float(stat.get("slg")),
                "milb_hr": _safe_int(stat.get("homeRuns")),
                "milb_sb": _safe_int(stat.get("stolenBases")),
                "milb_k_pct": _compute_k_pct(stat),
                "milb_bb_pct": _compute_bb_pct(stat),
                "milb_ip": _safe_float(stat.get("inningsPitched")),
                "milb_era": _safe_float(stat.get("era")),
                "milb_whip": _safe_float(stat.get("whip")),
                "milb_k9": _safe_float(stat.get("strikeoutsPer9Inn")),
                "milb_bb9": _safe_float(stat.get("walksPer9Inn")),
            }
            rows.append(row)
            time.sleep(0.3)  # rate limit
        except Exception:
            logger.debug("MiLB stats fetch failed for mlb_id=%s", mlb_id, exc_info=True)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_level(level_name: str) -> str:
    name = level_name.lower()
    if "triple" in name or "aaa" in name or "international" in name or "pacific" in name:
        return "AAA"
    if "double" in name or "eastern" in name or "southern" in name or "texas" in name:
        return "AA"
    if "high" in name or "a+" in name:
        return "High-A"
    if "single" in name or "south atlantic" in name or "midwest" in name:
        return "A"
    return level_name


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _compute_k_pct(stat: dict) -> float | None:
    so = _safe_int(stat.get("strikeOuts"))
    pa = _safe_int(stat.get("plateAppearances")) or _safe_int(stat.get("atBats"))
    if so is not None and pa and pa > 0:
        return round(so / pa * 100, 1)
    return None


def _compute_bb_pct(stat: dict) -> float | None:
    bb = _safe_int(stat.get("baseOnBalls"))
    pa = _safe_int(stat.get("plateAppearances")) or _safe_int(stat.get("atBats"))
    if bb is not None and pa and pa > 0:
        return round(bb / pa * 100, 1)
    return None


# -- DB persistence -----------------------------------------------------------


def _store_prospects(df: pd.DataFrame) -> int:
    """Store prospect rankings to DB. Returns count stored."""
    if df.empty:
        return 0
    # Validate data quality — don't store garbage
    if "name" in df.columns:
        non_empty = df["name"].astype(str).str.strip().ne("").sum()
        if non_empty < len(df) * 0.5:
            logger.warning(
                "Prospect data quality check failed: %d/%d names empty, skipping store", len(df) - non_empty, len(df)
            )
            return 0
    from src.database import get_connection

    conn = get_connection()
    try:
        # Clear and re-insert
        conn.execute("DELETE FROM prospect_rankings")
        now = datetime.now(UTC).isoformat()
        count = 0
        for _, row in df.iterrows():
            conn.execute(
                """INSERT INTO prospect_rankings
                   (mlb_id, name, team, position, fg_rank, fg_fv, fg_eta, fg_risk,
                    age, hit_present, hit_future, game_present, game_future,
                    raw_present, raw_future, speed, field, ctrl_present, ctrl_future,
                    scouting_report, tldr, milb_level, milb_avg, milb_obp, milb_slg,
                    milb_k_pct, milb_bb_pct, milb_hr, milb_sb,
                    milb_ip, milb_era, milb_whip, milb_k9, milb_bb9,
                    readiness_score, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.get("mlb_id"),
                    row.get("name"),
                    row.get("team"),
                    row.get("position"),
                    row.get("fg_rank"),
                    row.get("fg_fv"),
                    row.get("fg_eta"),
                    row.get("fg_risk"),
                    row.get("age"),
                    row.get("hit_present"),
                    row.get("hit_future"),
                    row.get("game_present"),
                    row.get("game_future"),
                    row.get("raw_present"),
                    row.get("raw_future"),
                    row.get("speed"),
                    row.get("field"),
                    row.get("ctrl_present"),
                    row.get("ctrl_future"),
                    row.get("scouting_report"),
                    row.get("tldr"),
                    row.get("milb_level"),
                    row.get("milb_avg"),
                    row.get("milb_obp"),
                    row.get("milb_slg"),
                    row.get("milb_k_pct"),
                    row.get("milb_bb_pct"),
                    row.get("milb_hr"),
                    row.get("milb_sb"),
                    row.get("milb_ip"),
                    row.get("milb_era"),
                    row.get("milb_whip"),
                    row.get("milb_k9"),
                    row.get("milb_bb9"),
                    row.get("readiness_score"),
                    now,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def _fetch_from_db(top_n: int = 100) -> pd.DataFrame:
    """Load prospect rankings from DB."""
    from src.database import get_connection

    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM prospect_rankings ORDER BY fg_rank LIMIT ?",
            conn,
            params=(top_n,),
        )
    finally:
        conn.close()


# -- Public API ---------------------------------------------------------------


def _scrape_mlb_pipeline() -> pd.DataFrame:
    """Fallback Level 2: Scrape MLB Pipeline prospect list.
    Returns DataFrame with basic ranking + stats (no scouting tools).
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        # MLB Pipeline prospect list page
        url = "https://www.mlb.com/prospects"
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        # Try JSON embedded in page first (MLB often embeds data)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for prospect data in script tags
        import json as _json

        for script in soup.find_all("script"):
            text = script.string or ""
            if "prospects" in text.lower() and "rank" in text.lower():
                # Extract JSON arrays by finding balanced bracket pairs (O(N))
                idx = 0
                while idx < len(text):
                    start = text.find("[", idx)
                    if start == -1:
                        break
                    # Find matching close bracket via depth counting
                    depth = 0
                    end = start
                    for end in range(start, len(text)):
                        if text[end] == "[":
                            depth += 1
                        elif text[end] == "]":
                            depth -= 1
                            if depth == 0:
                                break
                    if depth != 0:
                        idx = start + 1
                        continue
                    try:
                        data = _json.loads(text[start : end + 1])
                        if isinstance(data, list) and len(data) > 5:
                            rows = []
                            for i, item in enumerate(data[:100]):
                                rows.append(
                                    {
                                        "name": item.get("name", item.get("fullName", "")),
                                        "team": item.get("team", {}).get("abbreviation", "")
                                        if isinstance(item.get("team"), dict)
                                        else str(item.get("team", "")),
                                        "position": item.get("position", item.get("primaryPosition", "")),
                                        "fg_rank": i + 1,
                                        "mlb_id": item.get("playerId", item.get("id")),
                                    }
                                )
                            if rows:
                                logger.info("MLB Pipeline scrape: found %d prospects", len(rows))
                                return pd.DataFrame(rows)
                    except (ValueError, TypeError):
                        pass
                    idx = end + 1
        logger.warning("MLB Pipeline scrape: no parseable data found")
        return pd.DataFrame()
    except Exception:
        logger.warning("MLB Pipeline scrape failed", exc_info=True)
        return pd.DataFrame()


def refresh_prospect_rankings(force: bool = False) -> pd.DataFrame:
    """Refresh prospect rankings from external sources.

    Fallback chain:
    1. FanGraphs Board API (richest -- scouting tools + reports)
    2. MLB Pipeline scrape (ranking + basic stats, no scouting tools)
    3. Static list (last resort)
    """
    from src.database import check_staleness

    if not force and not check_staleness("prospect_rankings", 168):
        return _fetch_from_db()

    def _enrich_and_store(df: pd.DataFrame) -> pd.DataFrame:
        """Fetch MiLB stats, compute readiness, store to DB."""
        mlb_ids = []
        if "mlb_id" in df.columns:
            for raw in df["mlb_id"].dropna().tolist():
                try:
                    mid = int(raw)
                    if mid > 0:
                        mlb_ids.append(mid)
                except (ValueError, TypeError):
                    pass  # skip non-numeric IDs (e.g., FG "sa3017662" format)
        if mlb_ids:
            milb_df = fetch_milb_stats(mlb_ids)
            if not milb_df.empty:
                df = df.merge(milb_df, on="mlb_id", how="left", suffixes=("", "_milb"))
                for col in milb_df.columns:
                    if col != "mlb_id" and col in df.columns and f"{col}_milb" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_milb"])
                        df.drop(columns=[f"{col}_milb"], inplace=True)
        df["readiness_score"] = df.apply(lambda r: compute_mlb_readiness_score(r.to_dict()), axis=1)
        _store_prospects(df)
        return df

    # Level 1: FanGraphs Board API
    df = fetch_fg_board()
    if not df.empty:
        return _enrich_and_store(df)

    # Level 2: MLB Pipeline scrape
    df = _scrape_mlb_pipeline()
    if not df.empty:
        logger.info("Using MLB Pipeline scrape (no scouting tools)")
        return _enrich_and_store(df)

    # Level 3: Static fallback (compute readiness scores and store)
    logger.warning("All prospect sources failed -- using static list")
    static_df = pd.DataFrame(_STATIC_PROSPECTS)
    static_df["readiness_score"] = static_df.apply(lambda r: compute_mlb_readiness_score(r.to_dict()), axis=1)
    _store_prospects(static_df)
    return static_df


def get_prospect_rankings(
    top_n: int = 100,
    position: str | None = None,
    org: str | None = None,
) -> pd.DataFrame:
    """Get prospect rankings, optionally filtered.

    Tries DB first, refreshes if empty.
    """
    df = _fetch_from_db(top_n=500)  # fetch more than needed for filtering
    # Validate DB data — if names are mostly empty, refresh
    if not df.empty and "name" in df.columns:
        non_empty = df["name"].astype(str).str.strip().ne("").sum()
        if non_empty < len(df) * 0.5:
            logger.warning("DB prospect data has empty names, refreshing")
            df = pd.DataFrame()  # force refresh
    if df.empty:
        df = refresh_prospect_rankings()
    if df.empty:
        # Ultimate fallback
        df = pd.DataFrame(_STATIC_PROSPECTS)

    if position:
        df = df[df["position"].str.contains(position, case=False, na=False)]
    if org:
        df = df[df["team"].str.upper() == org.upper()]

    return df.head(top_n).reset_index(drop=True)


def get_prospect_detail(prospect_id: int) -> dict | None:
    """Get full detail for a single prospect by prospect_id."""
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM prospect_rankings WHERE prospect_id = ?",
            (prospect_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def compute_call_up_signals(
    prospect_row: dict,
    mlb_team_il_players: list[dict] | None = None,
) -> dict:
    """Compute call-up probability signals for a prospect.

    Args:
        prospect_row: Dict-like prospect data with keys like on_40_man,
            fg_eta/eta, position, team.
        mlb_team_il_players: Optional list of IL player dicts with team
            and positions keys, used to detect position openings.

    Returns:
        dict with on_40_man (bool), call_up_score (float 0-100),
        signal (str: "IMMINENT", "WATCH", or "").
    """
    on_40_man = bool(prospect_row.get("on_40_man", False))
    eta = str(prospect_row.get("fg_eta", prospect_row.get("eta", "")))

    score = 0
    if on_40_man:
        score += 40
    if "2026" in eta:
        score += 30
    elif "2027" in eta:
        score += 10

    if mlb_team_il_players:
        pos = str(prospect_row.get("position", ""))
        team = str(prospect_row.get("team", ""))
        for il_player in mlb_team_il_players:
            if il_player.get("team") == team and pos in str(il_player.get("positions", "")):
                score += 20
                break

    signal = "IMMINENT" if score >= 70 else "WATCH" if score >= 40 else ""
    return {"on_40_man": on_40_man, "call_up_score": min(score, 100), "signal": signal}


def compute_fantasy_relevance_score(
    prospect_row: dict,
    user_roster_positions: list[str] | None = None,
) -> dict:
    """O2: Prospect fantasy relevance score.

    Adjusts raw FV (Future Value) by:
    - ETA proximity (2026 = 1.0x, 2027 = 0.6x, 2028+ = 0.3x)
    - Position scarcity in user's league (C/SS/2B get 1.2x)
    - Path to playing time (40-man = 1.3x, not 40-man = 0.7x)
    - Historical FV hit rates (55 FV = 67% become regulars)

    Returns:
        dict with fantasy_relevance (0-100), fv, eta_mult, scarcity_mult,
        playing_time_mult, historical_hit_rate.
    """
    fv = float(prospect_row.get("fv", prospect_row.get("future_value", 40)))
    eta = str(prospect_row.get("fg_eta", prospect_row.get("eta", "")))
    position = str(prospect_row.get("position", ""))
    on_40_man = bool(prospect_row.get("on_40_man", False))

    # ETA proximity multiplier
    if "2026" in eta:
        eta_mult = 1.0
    elif "2027" in eta:
        eta_mult = 0.6
    else:
        eta_mult = 0.3

    # Position scarcity (C, SS, 2B are scarce in 12-team leagues)
    scarce = {"C", "SS", "2B"}
    pos_set = set(p.strip() for p in position.split(",") if p.strip())
    scarcity_mult = 1.2 if pos_set & scarce else 1.0

    # Path to playing time
    playing_time_mult = 1.3 if on_40_man else 0.7

    # Historical FV hit rates (FanGraphs data)
    # FV 80 = 95% become stars, FV 55 = 67% become regulars, FV 40 = 30% reach majors
    fv_hit_rates = {80: 0.95, 70: 0.85, 65: 0.80, 60: 0.75, 55: 0.67, 50: 0.50, 45: 0.40, 40: 0.30}
    hit_rate = 0.30
    for fv_threshold in sorted(fv_hit_rates.keys()):
        if fv >= fv_threshold:
            hit_rate = fv_hit_rates[fv_threshold]

    # Composite: FV normalized to 0-100 * adjustments * hit rate
    base = min(100.0, max(0.0, (fv - 30) * 2.0))  # FV 30-80 → 0-100
    relevance = base * eta_mult * scarcity_mult * playing_time_mult * hit_rate
    relevance = min(100.0, max(0.0, relevance))

    return {
        "fantasy_relevance": round(relevance, 1),
        "fv": fv,
        "eta_mult": eta_mult,
        "scarcity_mult": scarcity_mult,
        "playing_time_mult": playing_time_mult,
        "historical_hit_rate": hit_rate,
    }
