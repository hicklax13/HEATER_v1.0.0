"""Contract year data: scrape Baseball Reference free agent lists.

Functions
---------
fetch_contract_year_players
    Scrape BB-Ref upcoming free agent list to identify contract-year players.
is_contract_year
    Case-insensitive check if a player is in their contract year.
mark_contract_years
    Add/update ``contract_year`` column in a player pool DataFrame.
"""

import logging
from datetime import UTC, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BBREF_FA_URL = "https://www.baseball-reference.com/leagues/majors/{fa_year}-free-agents.shtml"
_HEADERS = {"User-Agent": "Fantasy Baseball Draft Tool"}
_TIMEOUT = 15


def fetch_contract_year_players(fa_year: int | None = None) -> set[str]:
    """Scrape Baseball Reference free agent list for a given year.

    Players who become free agents after the *previous* season are in
    their contract year during that season.  For example, players on the
    2027 free-agent list are in their contract year in 2026.

    When *fa_year* is ``None`` (default), the function tries three
    candidate years in order — ``current_year + 1``, ``current_year``,
    ``current_year - 1`` — and returns the first result that produces
    at least one player name.  BB-Ref may return 404 for future years
    that haven't been published yet, so this fallback avoids a hard
    failure.

    Parameters
    ----------
    fa_year : int or None
        The free-agent class year.  When ``None``, auto-detects via
        year fallback (see above).

    Returns
    -------
    set[str]
        Lowercased player names.  Returns an empty set on any failure
        (network error, parse error, etc.).
    """
    if fa_year is not None:
        return _fetch_for_year(fa_year)

    current_year = datetime.now(UTC).year
    for year in [current_year + 1, current_year, current_year - 1]:
        result = _fetch_for_year(year)
        if result:
            return result
    logger.warning(
        "BB-Ref free agent fetch failed for all candidate years (%d-%d)",
        current_year - 1,
        current_year + 1,
    )
    return set()


def _fetch_for_year(fa_year: int) -> set[str]:
    """Fetch the BB-Ref free agent page for a single year.

    Returns
    -------
    set[str]
        Lowercased player names, or an empty set on any failure.
    """
    url = _BBREF_FA_URL.format(fa_year=fa_year)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        logger.warning("Failed to fetch BB-Ref free agent page: %s", url)
        return set()

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        names: set[str] = set()

        # BB-Ref free agent tables use <table id="fa_..."> or similar.
        # We look for all player links inside any table on the page.
        for table in soup.find_all("table"):
            for link in table.find_all("a"):
                href = link.get("href", "")
                # Player links look like /players/t/troutmi01.shtml
                if "/players/" in href and href.endswith(".shtml"):
                    name = link.get_text(strip=True)
                    if name:
                        names.add(name.lower())

        if not names:
            logger.warning("No player names parsed from BB-Ref free agent page: %s", url)
        else:
            logger.info(
                "Fetched %d contract-year players from BB-Ref (%d FA class)",
                len(names),
                fa_year,
            )
        return names
    except Exception:
        logger.warning("Failed to parse BB-Ref free agent page: %s", url, exc_info=True)
        return set()


def is_contract_year(player_name: str, contract_year_set: set[str]) -> bool:
    """Case-insensitive check if a player is in their contract year.

    Parameters
    ----------
    player_name : str
        The player's name (any casing).
    contract_year_set : set[str]
        Set of lowercased player names from :func:`fetch_contract_year_players`.

    Returns
    -------
    bool
        ``True`` if the player is in the contract-year set.
    """
    if not player_name or not isinstance(player_name, str) or not contract_year_set:
        return False
    return player_name.strip().lower() in contract_year_set


def mark_contract_years(player_pool: pd.DataFrame, contract_year_set: set[str]) -> pd.DataFrame:
    """Add or update a ``contract_year`` boolean column in the player pool.

    Parameters
    ----------
    player_pool : pd.DataFrame
        Must contain a ``name`` column with player names.
    contract_year_set : set[str]
        Set of lowercased player names from :func:`fetch_contract_year_players`.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with a ``contract_year`` column added/updated.
        Values are ``True`` / ``False``.
    """
    player_pool = player_pool.copy()
    if contract_year_set and "name" in player_pool.columns:
        player_pool["contract_year"] = player_pool["name"].apply(lambda n: is_contract_year(n, contract_year_set))
    else:
        player_pool["contract_year"] = False
    return player_pool
