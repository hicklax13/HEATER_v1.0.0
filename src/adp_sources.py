"""Multi-source ADP module: FantasyPros ECR, NFBC ADP, and composite ADP.

Fetches average draft position data from multiple external sources and
provides a composite coalescing function for the best available ADP.
All HTTP functions return empty DataFrames on any exception (HTTP error,
parse error, timeout, etc.) for graceful degradation.
"""

import logging
import math

import pandas as pd

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

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_HEADERS = {"User-Agent": "Fantasy Baseball Draft Tool"}
_TIMEOUT = 15  # seconds per request

_FANTASYPROS_URL = "https://www.fantasypros.com/mlb/rankings/overall.php"
_NFBC_URL = "https://nfc.shgn.com/adp/baseball"


# ── FantasyPros ECR ────────────────────────────────────────────────


def fetch_fantasypros_ecr() -> pd.DataFrame:
    """Fetch FantasyPros Expert Consensus Rankings.

    Scrapes the FantasyPros MLB overall rankings page and extracts
    player name, ECR rank, team, and position.

    .. note::
        As of 2025, FantasyPros renders its ranking tables via JavaScript
        (React/Next.js).  A plain ``requests.get()`` receives an empty
        shell page with no ``<table>`` elements.  To fix this properly
        you would need either:
        - Playwright / Selenium to render the JS first, or
        - A FantasyPros API subscription (paid).
        Until then this function will log a warning and return an empty
        DataFrame, which the composite ADP logic handles gracefully.

    Returns:
        DataFrame with columns: player_name, ecr_rank, team, position.
        Empty DataFrame on any error.
    """
    if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
        logger.warning("requests or beautifulsoup4 not available; skipping FantasyPros ECR fetch")
        return pd.DataFrame(columns=["player_name", "ecr_rank", "team", "position"])

    try:
        resp = requests.get(_FANTASYPROS_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "ranking-table"})
        if table is None:
            # Fallback: try finding any table with player data
            table = soup.find("table")
        if table is None:
            # FantasyPros rankings table not found — site requires
            # JavaScript rendering (React/Next.js).  A plain HTTP GET
            # returns an empty shell.  Consider using Playwright or
            # their paid API to obtain rankings.
            logger.warning(
                "FantasyPros rankings table not found — site requires "
                "JavaScript rendering. Consider using Playwright or their API."
            )
            return pd.DataFrame(columns=["player_name", "ecr_rank", "team", "position"])

        rows = table.find_all("tr")
        records = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            # Extract rank from first cell
            rank_text = cells[0].get_text(strip=True)
            try:
                ecr_rank = int(rank_text)
            except (ValueError, TypeError):
                continue

            # Extract player name and metadata from second cell
            player_cell = cells[1]
            # FantasyPros uses <a class="player-name"> for player links
            name_tag = player_cell.find("a", class_="player-name")
            if name_tag is None:
                name_tag = player_cell.find("a")
            if name_tag is None:
                # Try plain text
                player_name = player_cell.get_text(strip=True)
            else:
                player_name = name_tag.get_text(strip=True)

            if not player_name:
                continue

            # Extract team/position from small text or separate cells
            small_tag = player_cell.find("small")
            team = ""
            position = ""
            if small_tag:
                meta = small_tag.get_text(strip=True)
                # Format is typically "Team - POS" or "POS - Team"
                parts = [p.strip() for p in meta.split("-")]
                if len(parts) >= 2:
                    team = parts[0].strip()
                    position = parts[1].strip()
                elif len(parts) == 1:
                    position = parts[0].strip()

            records.append(
                {
                    "player_name": player_name,
                    "ecr_rank": ecr_rank,
                    "team": team,
                    "position": position,
                }
            )

        if not records:
            logger.warning("FantasyPros: parsed 0 players from ranking table")
            return pd.DataFrame(columns=["player_name", "ecr_rank", "team", "position"])

        logger.info("FantasyPros: fetched %d player ECR rankings", len(records))
        return pd.DataFrame(records)

    except Exception:
        logger.warning("FantasyPros ECR fetch failed", exc_info=True)
        return pd.DataFrame(columns=["player_name", "ecr_rank", "team", "position"])


# ── NFBC ADP ───────────────────────────────────────────────────────


def fetch_nfbc_adp() -> pd.DataFrame:
    """Fetch NFBC (National Fantasy Baseball Championship) ADP data.

    Scrapes the NFBC ADP page and extracts player name and ADP value.

    .. note::
        As of 2025, the NFBC ADP page (nfc.shgn.com) renders its data
        table via JavaScript.  A plain ``requests.get()`` receives a
        page with no ``<table>`` elements.  To fix this properly you
        would need Playwright / Selenium to render the JS first.
        Until then this function will log a warning and return an empty
        DataFrame, which the composite ADP logic handles gracefully.

    Returns:
        DataFrame with columns: player_name, nfbc_adp.
        Empty DataFrame on any error.
    """
    if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
        logger.warning("requests or beautifulsoup4 not available; skipping NFBC ADP fetch")
        return pd.DataFrame(columns=["player_name", "nfbc_adp"])

    try:
        resp = requests.get(_NFBC_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            # NFBC ADP table not found — nfc.shgn.com requires
            # JavaScript rendering.  A plain HTTP GET returns a page
            # with no data tables.  Consider using Playwright or
            # Selenium to obtain ADP data.
            logger.warning(
                "NFBC ADP table not found — site requires JavaScript rendering. Consider using Playwright or Selenium."
            )
            return pd.DataFrame(columns=["player_name", "nfbc_adp"])

        rows = table.find_all("tr")

        # Detect column indices from header row
        name_col_idx = None
        adp_col_idx = None
        header_row = table.find("tr")
        if header_row:
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
            for idx, h in enumerate(headers):
                if h in ("player", "name") and name_col_idx is None:
                    name_col_idx = idx
                elif h == "adp" and adp_col_idx is None:
                    adp_col_idx = idx

        records = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            player_name = ""
            nfbc_adp = None

            if name_col_idx is not None and adp_col_idx is not None:
                # Use detected column indices
                if name_col_idx < len(cells):
                    player_name = cells[name_col_idx].get_text(strip=True)
                if adp_col_idx < len(cells):
                    try:
                        nfbc_adp = float(cells[adp_col_idx].get_text(strip=True))
                    except (ValueError, TypeError):
                        pass
            else:
                # Fallback: heuristic scan — name is first alpha cell,
                # ADP is first float AFTER the name cell (skip rank column).
                name_found = False
                for cell in cells:
                    text = cell.get_text(strip=True)
                    if not name_found and text and any(c.isalpha() for c in text):
                        player_name = text
                        name_found = True
                    elif name_found and player_name:
                        try:
                            val = float(text)
                            if nfbc_adp is None and 0 < val < 1000:
                                nfbc_adp = val
                        except (ValueError, TypeError):
                            continue

            if player_name and nfbc_adp is not None and nfbc_adp > 0:
                records.append(
                    {
                        "player_name": player_name,
                        "nfbc_adp": nfbc_adp,
                    }
                )

        if not records:
            logger.warning("NFBC: parsed 0 players from ADP table")
            return pd.DataFrame(columns=["player_name", "nfbc_adp"])

        logger.info("NFBC: fetched %d player ADP values", len(records))
        return pd.DataFrame(records)

    except Exception:
        logger.warning("NFBC ADP fetch failed", exc_info=True)
        return pd.DataFrame(columns=["player_name", "nfbc_adp"])


# ── Composite ADP ──────────────────────────────────────────────────


def compute_composite_adp(row: dict) -> float:
    """Compute composite ADP from multiple sources using coalesce priority.

    Priority order: yahoo_adp > fantasypros_adp > nfbc_adp > adp.
    Returns the first valid (non-None, non-NaN, > 0) value found.
    If all sources are missing or invalid, returns 999.0.

    Args:
        row: Dict with optional keys: yahoo_adp, fantasypros_adp, nfbc_adp, adp.

    Returns:
        Float ADP value (999.0 if all sources missing).
    """
    for key in ("yahoo_adp", "fantasypros_adp", "nfbc_adp", "adp"):
        val = row.get(key)
        if val is not None:
            try:
                fval = float(val)
                if not math.isnan(fval) and not math.isinf(fval) and fval > 0:
                    return fval
            except (ValueError, TypeError):
                continue
    return 999.0
