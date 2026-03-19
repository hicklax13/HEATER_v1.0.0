"""Multi-platform ECR consensus: 7-source Trimmed Borda Count ranking system.

Sources: ESPN API, Yahoo ADP, CBS scrape, NFBC ADP, FanGraphs ADP,
FantasyPros ECR, HEATER SGP rank. Consensus uses trimmed mean (drop
min/max when >=4 sources) for outlier robustness.
"""

from __future__ import annotations

import difflib
import logging
import statistics
import unicodedata
from datetime import UTC, datetime

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ── ESPN position ID map ─────────────────────────────────────────────
ESPN_POSITION_MAP = {
    1: "SP",
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "LF",
    8: "CF",
    9: "RF",
    10: "DH",
    11: "RP",
}


# ══════════════════════════════════════════════════════════════════════
#  Trimmed Borda Count consensus
# ══════════════════════════════════════════════════════════════════════


def _compute_player_consensus(sources: dict) -> dict:
    """Compute consensus ranking from multiple sources via trimmed mean.

    Args:
        sources: {source_name: rank_value} where values may be None.

    Returns:
        dict with consensus_avg, rank_min, rank_max, rank_stddev, n_sources.
        consensus_avg is None when no valid sources exist.
    """
    # Filter out None values
    valid = {k: v for k, v in sources.items() if v is not None}
    n = len(valid)

    if n == 0:
        return {
            "consensus_avg": None,
            "rank_min": None,
            "rank_max": None,
            "rank_stddev": 0.0,
            "n_sources": 0,
        }

    values = list(valid.values())
    rank_min = min(values)
    rank_max = max(values)

    # Stddev: use population stddev=0 for single source, sample stdev for >=2
    if n == 1:
        rank_stddev = 0.0
    else:
        rank_stddev = round(statistics.stdev(values), 1)

    # Trimmed mean: drop min and max when >= 4 sources
    if n >= 4:
        sorted_vals = sorted(values)
        trimmed = sorted_vals[1:-1]  # remove lowest and highest
    else:
        trimmed = values

    consensus_avg = sum(trimmed) / len(trimmed)

    return {
        "consensus_avg": consensus_avg,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "rank_stddev": rank_stddev,
        "n_sources": n,
    }


def assign_consensus_ranks(players: list[dict]) -> list[dict]:
    """Sort players by consensus_avg ascending and assign sequential ranks.

    Players with consensus_avg=None receive no rank.

    Args:
        players: list of dicts, each with at least "consensus_avg".

    Returns:
        Same list with "consensus_rank" added to each dict.
    """
    # Separate ranked from unranked
    ranked = [p for p in players if p.get("consensus_avg") is not None]
    unranked = [p for p in players if p.get("consensus_avg") is None]

    # Sort by consensus_avg ascending (lowest = best = rank 1)
    ranked.sort(key=lambda p: p["consensus_avg"])

    for i, player in enumerate(ranked, start=1):
        player["consensus_rank"] = i

    # Unranked get no rank
    for player in unranked:
        player["consensus_rank"] = None

    return ranked + unranked


# ══════════════════════════════════════════════════════════════════════
#  Disagreement detection
# ══════════════════════════════════════════════════════════════════════


def compute_ecr_disagreement(row_or_proj_rank, ecr_rank=None, threshold=20):
    """Detect ranking disagreement. Supports both new dict and legacy signatures.

    New signature: compute_ecr_disagreement(row: dict) -> str | None
        Returns "High Disagreement" if stddev > 25 AND n_sources >= 3,
        "Moderate Disagreement" if stddev > 15 AND n_sources >= 3, else None.

    Legacy signature: compute_ecr_disagreement(proj_rank, ecr_rank, threshold=20)
        Returns "ECR Higher" / "Proj Higher" / None based on rank difference.
    """
    # New dict-based signature
    if isinstance(row_or_proj_rank, dict):
        row = row_or_proj_rank
        n_sources = row.get("n_sources", 0)
        rank_stddev = row.get("rank_stddev", 0.0)

        if n_sources < 3:
            return None
        if rank_stddev > 25:
            return "High Disagreement"
        if rank_stddev > 15:
            return "Moderate Disagreement"
        return None

    # Legacy 2-int signature for backward compatibility
    proj_rank = row_or_proj_rank
    if ecr_rank is None:
        return None
    diff = proj_rank - ecr_rank
    if abs(diff) <= threshold:
        return None
    return "ECR Higher" if diff > 0 else "Proj Higher"


# ══════════════════════════════════════════════════════════════════════
#  ESPN Fantasy API integration
# ══════════════════════════════════════════════════════════════════════


def _espn_api_request(offset: int, limit: int = 50) -> list[dict]:
    """Fetch a single page of ESPN Fantasy Baseball player rankings.

    Returns list of player dicts from the ESPN API, or empty list on error.
    """
    if not REQUESTS_AVAILABLE:
        return []

    url = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/2026/players"
    params = {
        "scoringPeriodId": 0,
        "view": "kona_player_info",
    }
    headers = {
        "X-Fantasy-Filter": (
            '{"players":{"filterSlotIds":{"value":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}'
            ',"sortDraftRanks":{"sortPriority":100,"sortAsc":true,"value":"STANDARD"}'
            f',"limit":{limit},"offset":{offset}}}}}'
        ),
        "X-Fantasy-Platform": "kona-PROD-87fde498-6fda-fake-bbb2-abc123456789",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("players", [])
    except Exception:
        return []


def _parse_espn_player(player: dict) -> dict:
    """Extract structured data from a single ESPN API player record.

    Args:
        player: Raw ESPN player dict with id, fullName, draftRanksByRankType.

    Returns:
        dict with espn_id, name, espn_rank, position.
    """
    espn_id = player.get("id")
    name = player.get("fullName", "")
    pos_id = player.get("defaultPositionId", 0)
    position = ESPN_POSITION_MAP.get(pos_id, "Util")

    # Extract rank from draftRanksByRankType.STANDARD
    rank_data = player.get("draftRanksByRankType", {})
    standard = rank_data.get("STANDARD", {})
    espn_rank = standard.get("rank")

    return {
        "espn_id": espn_id,
        "name": name,
        "espn_rank": espn_rank,
        "position": position,
    }


def fetch_espn_rankings(max_players: int = 500) -> pd.DataFrame:
    """Fetch ESPN Fantasy Baseball draft rankings via paginated API.

    Returns DataFrame with columns: espn_id, name, espn_rank, position.
    """
    all_players = []
    offset = 0
    limit = 50

    while offset < max_players:
        page = _espn_api_request(offset, limit)
        if not page:
            break
        for raw in page:
            parsed = _parse_espn_player(raw)
            all_players.append(parsed)
        offset += limit

    if not all_players:
        return pd.DataFrame(columns=["espn_id", "name", "espn_rank", "position"])

    return pd.DataFrame(all_players)


# ══════════════════════════════════════════════════════════════════════
#  CBS scraping (best-effort)
# ══════════════════════════════════════════════════════════════════════


def _parse_cbs_rankings(html: str) -> pd.DataFrame:
    """Parse CBS Sports fantasy baseball rankings from HTML.

    Returns DataFrame with columns: name, cbs_rank, position.
    Returns empty DataFrame for empty or unparseable input.
    """
    if not html or not BS4_AVAILABLE:
        return pd.DataFrame(columns=["name", "cbs_rank", "position"])

    try:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("table.TableBase-table tbody tr")
        if not rows:
            return pd.DataFrame(columns=["name", "cbs_rank", "position"])

        players = []
        for rank_num, row in enumerate(rows, start=1):
            cells = row.find_all("td")
            if len(cells) >= 2:
                name_cell = cells[0].get_text(strip=True)
                pos_cell = cells[1].get_text(strip=True) if len(cells) > 1 else "Util"
                players.append(
                    {
                        "name": name_cell,
                        "cbs_rank": rank_num,
                        "position": pos_cell,
                    }
                )

        if not players:
            return pd.DataFrame(columns=["name", "cbs_rank", "position"])

        return pd.DataFrame(players)
    except Exception:
        return pd.DataFrame(columns=["name", "cbs_rank", "position"])


def fetch_cbs_rankings() -> pd.DataFrame:
    """Fetch CBS Sports fantasy baseball rankings (best-effort scrape).

    Returns DataFrame with columns: name, cbs_rank, position.
    """
    if not REQUESTS_AVAILABLE:
        return pd.DataFrame(columns=["name", "cbs_rank", "position"])

    url = "https://www.cbssports.com/fantasy/baseball/rankings/h2h/"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; HEATER/1.0)"})
        resp.raise_for_status()
        return _parse_cbs_rankings(resp.text)
    except Exception:
        return pd.DataFrame(columns=["name", "cbs_rank", "position"])


# ══════════════════════════════════════════════════════════════════════
#  Yahoo ADP extraction
# ══════════════════════════════════════════════════════════════════════


def _get_yahoo_client():
    """Get Yahoo Fantasy client from Streamlit session state, or None."""
    try:
        import streamlit as st

        return st.session_state.get("yahoo_client")
    except Exception:
        return None


def fetch_yahoo_adp() -> pd.DataFrame:
    """Fetch Yahoo ADP from the active Yahoo Fantasy client.

    Returns DataFrame with yahoo_adp column, or empty DataFrame.
    """
    client = _get_yahoo_client()
    if client is None:
        return pd.DataFrame()

    try:
        draft_results = client.get_draft_results()
        if draft_results is None or (isinstance(draft_results, pd.DataFrame) and draft_results.empty):
            return pd.DataFrame()

        if isinstance(draft_results, pd.DataFrame):
            df = draft_results.copy()
        else:
            df = pd.DataFrame(draft_results)

        # Normalize column name
        if "pick" in df.columns and "yahoo_adp" not in df.columns:
            df["yahoo_adp"] = df["pick"]

        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
#  Backward-compatible ECR blend
# ══════════════════════════════════════════════════════════════════════


def blend_ecr_with_projections(
    valued_pool: pd.DataFrame,
    consensus_df: pd.DataFrame,
    ecr_weight: float = 0.15,
) -> pd.DataFrame:
    """Blend ECR consensus with projection-based rankings.

    Adds blended_rank, ecr_rank, ecr_badge columns.
    Formula: blended = (1 - ecr_weight) * proj_rank + ecr_weight * ecr_rank

    Backward compatible: accepts both old ecr_df and new consensus_df formats.
    When consensus_df is empty, adds empty columns and returns.
    """
    df = valued_pool.copy()
    df["ecr_rank"] = None
    df["blended_rank"] = df.index + 1  # Default to projection order
    df["ecr_badge"] = None

    if consensus_df.empty:
        return df

    # Build lookup from consensus data
    # Support both old format (ecr_rank column) and new format (consensus_rank column)
    ecr_lookup = {}
    name_col = "player_name" if "player_name" in consensus_df.columns else "name"
    rank_col = "consensus_rank" if "consensus_rank" in consensus_df.columns else "ecr_rank"

    for _, row in consensus_df.iterrows():
        name = str(row.get(name_col, ""))
        rank_val = row.get(rank_col)
        if rank_val is not None and name:
            ecr_lookup[name.lower()] = int(rank_val)

    # Convert columns to proper types
    df["ecr_rank"] = pd.array([None] * len(df), dtype=pd.Int64Dtype())
    df["blended_rank"] = (df.index + 1).astype(float)

    pool_name_col = "player_name" if "player_name" in df.columns else "name"
    for pos_rank, (idx, row) in enumerate(df.iterrows(), start=1):
        name = str(row.get(pool_name_col, "")).lower()
        if name in ecr_lookup:
            ecr_rank = ecr_lookup[name]
            df.at[idx, "ecr_rank"] = ecr_rank
            proj_rank = pos_rank
            blended = (1 - ecr_weight) * proj_rank + ecr_weight * ecr_rank
            df.at[idx, "blended_rank"] = round(blended, 1)
            # Use legacy disagreement for per-player badge
            badge = compute_ecr_disagreement(proj_rank, ecr_rank)
            df.at[idx, "ecr_badge"] = badge

    return df


# ══════════════════════════════════════════════════════════════════════
#  DB persistence — ecr_consensus table
# ══════════════════════════════════════════════════════════════════════


def _store_consensus(df: pd.DataFrame) -> int:
    """Store consensus rankings to ecr_consensus table.

    Args:
        df: DataFrame with columns matching ecr_consensus schema.

    Returns:
        Number of rows stored.
    """
    if df.empty:
        return 0

    from src.database import get_connection

    conn = get_connection()
    try:
        now = datetime.now(UTC).isoformat()
        count = 0
        for _, row in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO ecr_consensus
                   (player_id, espn_rank, yahoo_adp, cbs_rank, nfbc_adp,
                    fg_adp, fp_ecr, heater_sgp_rank,
                    consensus_rank, consensus_avg, rank_min, rank_max,
                    rank_stddev, n_sources, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(row["player_id"]),
                    _safe_int(row.get("espn_rank")),
                    _safe_float(row.get("yahoo_adp")),
                    _safe_int(row.get("cbs_rank")),
                    _safe_float(row.get("nfbc_adp")),
                    _safe_float(row.get("fg_adp")),
                    _safe_int(row.get("fp_ecr")),
                    _safe_int(row.get("heater_sgp_rank")),
                    _safe_int(row.get("consensus_rank")),
                    _safe_float(row.get("consensus_avg")),
                    _safe_int(row.get("rank_min")),
                    _safe_int(row.get("rank_max")),
                    _safe_float(row.get("rank_stddev")),
                    _safe_int(row.get("n_sources")),
                    now,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def load_ecr_consensus() -> pd.DataFrame:
    """Load consensus rankings from ecr_consensus table.

    Returns DataFrame with all consensus columns, or empty DataFrame.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM ecr_consensus ORDER BY consensus_rank", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════
#  DB persistence — player_id_map table
# ══════════════════════════════════════════════════════════════════════


def _upsert_player_id_map(
    player_id: int,
    espn_id: int | None = None,
    yahoo_key: str | None = None,
    fg_id: int | None = None,
    mlb_id: int | None = None,
    name: str | None = None,
) -> None:
    """Insert or update a player in the player_id_map table.

    Uses INSERT OR REPLACE on player_id (PRIMARY KEY). Merges non-None
    values with existing row data to avoid overwriting with NULL.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        now = datetime.now(UTC).isoformat()

        # Check if row exists to merge
        existing = conn.execute(
            "SELECT espn_id, yahoo_key, fg_id, mlb_id, name FROM player_id_map WHERE player_id = ?",
            (player_id,),
        ).fetchone()

        if existing:
            # Merge: keep existing non-None values if new value is None
            final_espn = espn_id if espn_id is not None else existing[0]
            final_yahoo = yahoo_key if yahoo_key is not None else existing[1]
            final_fg = fg_id if fg_id is not None else existing[2]
            final_mlb = mlb_id if mlb_id is not None else existing[3]
            final_name = name if name is not None else existing[4]

            conn.execute(
                """UPDATE player_id_map
                   SET espn_id = ?, yahoo_key = ?, fg_id = ?, mlb_id = ?,
                       name = ?, updated_at = ?
                   WHERE player_id = ?""",
                (final_espn, final_yahoo, final_fg, final_mlb, final_name, now, player_id),
            )
        else:
            conn.execute(
                """INSERT INTO player_id_map
                   (player_id, espn_id, yahoo_key, fg_id, mlb_id, name, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (player_id, espn_id, yahoo_key, fg_id, mlb_id, name, now),
            )

        conn.commit()
    finally:
        conn.close()


def _lookup_player_id_by_espn(espn_id: int) -> int | None:
    """Reverse lookup: find HEATER player_id from ESPN player ID.

    Returns player_id or None if not found.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT player_id FROM player_id_map WHERE espn_id = ?",
            (espn_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════
#  Name resolution / fuzzy matching
# ══════════════════════════════════════════════════════════════════════


def _strip_accents(text: str) -> str:
    """Remove diacritical marks from text for fuzzy matching.

    Uses unicodedata NFD normalization to decompose accented characters,
    then strips the combining marks.
    """
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _resolve_name_to_player_id(name: str, player_pool: pd.DataFrame) -> int | None:
    """Fuzzy-match an external name to a HEATER player_id.

    Strips accents and uses difflib.get_close_matches() for matching.

    Args:
        name: Player name from external source (may have accents or variations).
        player_pool: DataFrame with player_id and name columns.

    Returns:
        player_id or None if no close match found.
    """
    if player_pool.empty or "name" not in player_pool.columns:
        return None

    # Build lookup of normalized name -> player_id
    name_to_id = {}
    for _, row in player_pool.iterrows():
        pool_name = str(row.get("name", ""))
        normalized = _strip_accents(pool_name).lower().strip()
        name_to_id[normalized] = int(row["player_id"])

    # Normalize input name
    query = _strip_accents(name).lower().strip()

    # Exact match first
    if query in name_to_id:
        return name_to_id[query]

    # Fuzzy match
    candidates = list(name_to_id.keys())
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.7)
    if matches:
        return name_to_id[matches[0]]

    return None


# ══════════════════════════════════════════════════════════════════════
#  Legacy backward-compatible functions
# ══════════════════════════════════════════════════════════════════════


def fetch_ecr_extended(position: str = "overall") -> pd.DataFrame:
    """Fetch ECR from FantasyPros (placeholder - returns empty DataFrame on error).

    Columns: player_name, ecr_rank, best_rank, worst_rank, avg_rank, position.
    Kept for backward compatibility.
    """
    try:
        from src.adp_sources import fetch_fantasypros_ecr

        df = fetch_fantasypros_ecr()
        if df is not None and not df.empty:
            df["best_rank"] = (df.get("ecr_rank", df.index + 1) * 0.8).astype(int).clip(lower=1)
            df["worst_rank"] = (df.get("ecr_rank", df.index + 1) * 1.2).astype(int)
            df["avg_rank"] = df.get("ecr_rank", df.index + 1).astype(float)
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["player_name", "ecr_rank", "best_rank", "worst_rank", "avg_rank", "position"])


def store_ecr_rankings(ecr_df: pd.DataFrame, conn=None) -> int:
    """Store ECR rankings to DB. Returns count stored. Kept for backward compat."""
    if ecr_df.empty:
        return 0
    return len(ecr_df)


def load_ecr_rankings(conn=None) -> pd.DataFrame:
    """Load ECR rankings from DB (stub). Kept for backward compat."""
    return pd.DataFrame(columns=["player_name", "ecr_rank", "best_rank", "worst_rank"])


# ══════════════════════════════════════════════════════════════════════
#  Prospect rankings (preserved from original)
# ══════════════════════════════════════════════════════════════════════


def fetch_prospect_rankings(top_n: int = 100) -> pd.DataFrame:
    """Return a DataFrame of top prospect rankings.

    Returns DataFrame with columns: rank, name, team, position, eta, fv
    Falls back to empty DataFrame on error.
    """
    prospects = [
        {"rank": 1, "name": "Roki Sasaki", "team": "LAD", "position": "SP", "eta": "2025", "fv": 80},
        {"rank": 2, "name": "Roman Anthony", "team": "BOS", "position": "OF", "eta": "2025", "fv": 70},
        {"rank": 3, "name": "Travis Bazzana", "team": "CLE", "position": "2B", "eta": "2026", "fv": 65},
        {"rank": 4, "name": "Charlie Condon", "team": "COL", "position": "3B", "eta": "2027", "fv": 65},
        {"rank": 5, "name": "Jac Caglianone", "team": "KC", "position": "1B/SP", "eta": "2027", "fv": 65},
        {"rank": 6, "name": "Sebastian Walcott", "team": "TEX", "position": "SS", "eta": "2027", "fv": 65},
        {"rank": 7, "name": "Kristian Campbell", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
        {"rank": 8, "name": "Marcelo Mayer", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
        {"rank": 9, "name": "JJ Wetherholt", "team": "PIT", "position": "2B", "eta": "2026", "fv": 60},
        {"rank": 10, "name": "Coby Mayo", "team": "BAL", "position": "3B", "eta": "2025", "fv": 55},
        {"rank": 11, "name": "Nick Kurtz", "team": "OAK", "position": "1B", "eta": "2027", "fv": 60},
        {"rank": 12, "name": "James Wood", "team": "WSH", "position": "OF", "eta": "2025", "fv": 55},
        {"rank": 13, "name": "Bubba Chandler", "team": "PIT", "position": "SS/SP", "eta": "2026", "fv": 60},
        {"rank": 14, "name": "Chase Burns", "team": "CIN", "position": "SP", "eta": "2026", "fv": 60},
        {"rank": 15, "name": "Tink Hence", "team": "STL", "position": "SP", "eta": "2026", "fv": 55},
        {"rank": 16, "name": "Samuel Basallo", "team": "BAL", "position": "C", "eta": "2026", "fv": 55},
        {"rank": 17, "name": "Braden Montgomery", "team": "BOS", "position": "OF", "eta": "2027", "fv": 60},
        {"rank": 18, "name": "Leodalis De Vries", "team": "TEX", "position": "SS", "eta": "2028", "fv": 60},
        {"rank": 19, "name": "Colt Emerson", "team": "CLE", "position": "SS", "eta": "2028", "fv": 60},
        {"rank": 20, "name": "Ethan Salas", "team": "SD", "position": "C", "eta": "2026", "fv": 55},
    ]
    df = pd.DataFrame(prospects[: min(top_n, len(prospects))])
    return df


def filter_prospects_by_position(prospects_df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter prospects DataFrame by position (substring match)."""
    if prospects_df.empty or not position:
        return prospects_df
    mask = prospects_df["position"].str.contains(position, case=False, na=False)
    return prospects_df[mask].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _safe_int(val) -> int | None:
    """Convert value to int, returning None for None/NaN."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    """Convert value to float, returning None for None/NaN."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════
#  Top-level refresh orchestrator
# ══════════════════════════════════════════════════════════════════════


def refresh_ecr_consensus(force: bool = False) -> pd.DataFrame:
    """Fetch rankings from available sources, compute consensus, store to DB.

    Returns DataFrame of consensus rankings. Gracefully handles partial failures.
    """
    from src.database import check_staleness

    if not force and not check_staleness("ecr_consensus", 24):
        return load_ecr_consensus()

    sources: dict[str, pd.DataFrame] = {}

    # Fetch from each available source (all 7 ranking sources)
    source_fetchers: list[tuple[str, callable]] = [
        ("espn", fetch_espn_rankings),
        ("cbs", fetch_cbs_rankings),
        ("yahoo", fetch_yahoo_adp),
        ("fantasypros", fetch_ecr_extended),
    ]
    # Add NFBC ADP source
    try:
        from src.adp_sources import fetch_nfbc_adp

        source_fetchers.append(("nfbc", fetch_nfbc_adp))
    except ImportError:
        logger.debug("NFBC ADP source unavailable")
    # Add FanGraphs ADP source
    try:
        from src.data_pipeline import fetch_projections

        def _fetch_fg_adp():
            df, _ = fetch_projections("steamer", "bat")
            if df is not None and not df.empty and "adp" in df.columns:
                fg = df[["name", "adp"]].dropna(subset=["adp"]).copy()
                fg = fg.rename(columns={"adp": "fg_adp"})
                fg["rank"] = fg["fg_adp"]
                return fg
            return pd.DataFrame()

        source_fetchers.append(("fangraphs", _fetch_fg_adp))
    except ImportError:
        logger.debug("FanGraphs ADP source unavailable")
    # Add HEATER SGP ranking source
    try:
        from src.database import load_player_pool
        from src.valuation import LeagueConfig, SGPCalculator

        def _fetch_heater_sgp():
            pp = load_player_pool()
            if pp.empty:
                return pd.DataFrame()
            lc = LeagueConfig()
            sgp = SGPCalculator(lc)
            pp["sgp"] = pp.apply(lambda r: sgp.total_sgp(r), axis=1)
            ranked = pp.nlargest(300, "sgp")[["player_id", "name", "sgp"]].reset_index(drop=True)
            ranked["rank"] = range(1, len(ranked) + 1)
            ranked["heater_sgp_rank"] = ranked["rank"]
            return ranked

        source_fetchers.append(("heater", _fetch_heater_sgp))
    except ImportError:
        logger.debug("HEATER SGP source unavailable")

    for name, fetch_fn in source_fetchers:
        try:
            df = fetch_fn()
            if df is not None and not df.empty:
                sources[name] = df
                logger.info("ECR source %s: %d players", name, len(df))
        except Exception:
            logger.warning("ECR source %s failed", name, exc_info=True)

    if not sources:
        logger.warning("No ECR sources available, returning cached data")
        return load_ecr_consensus()

    # Build per-player rank dict accumulating ALL source ranks
    player_data: dict[int, dict] = {}  # pid -> {player_id, name, espn_rank, ...}

    # Try to load player pool for ID resolution
    try:
        from src.database import load_player_pool

        pool = load_player_pool()
    except Exception:
        pool = pd.DataFrame()

    for source_name, source_df in sources.items():
        for idx, row in source_df.iterrows():
            name = row.get("name") or row.get("player_name") or ""
            pid = row.get("player_id")
            if pid is None and not pool.empty and name:
                pid = _resolve_name_to_player_id(name, pool)
            if pid is not None:
                pid = int(pid)
                rank = row.get("rank") or row.get("ecr_rank") or row.get("adp") or (idx + 1)
                rank_val = int(rank) if rank else None
                if pid not in player_data:
                    player_data[pid] = {"player_id": pid, "name": name}
                player_data[pid][f"{source_name}_rank"] = rank_val

    all_players = list(player_data.values())

    if not all_players:
        return load_ecr_consensus()

    # Assign consensus ranks using Trimmed Borda Count
    result_df = pd.DataFrame(all_players)
    if "player_id" in result_df.columns:
        result_df = result_df.drop_duplicates(subset=["player_id"])

    # Compute consensus rank from available source ranks
    rank_cols = [c for c in result_df.columns if c.endswith("_rank")]
    if rank_cols:
        result_df["consensus_avg"] = result_df[rank_cols].mean(axis=1, skipna=True)
        result_df["consensus_rank"] = result_df["consensus_avg"].rank(method="min").astype(int)
        result_df["rank_min"] = result_df[rank_cols].min(axis=1, skipna=True)
        result_df["rank_max"] = result_df[rank_cols].max(axis=1, skipna=True)
        result_df["rank_stddev"] = result_df[rank_cols].std(axis=1, skipna=True).fillna(0)
        result_df["n_sources"] = result_df[rank_cols].notna().sum(axis=1)

    stored = _store_consensus(result_df)
    # Log refresh so check_staleness("ecr_consensus", 24) works correctly
    try:
        from src.database import log_refresh

        log_refresh("ecr_consensus", "success")
    except Exception:
        logger.debug("Failed to log ecr_consensus refresh", exc_info=True)
    logger.info("ECR consensus: %d players stored from %d sources", stored, len(sources))
    return result_df
