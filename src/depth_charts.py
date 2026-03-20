"""Roster Resource depth chart scraper and role classification.

Scrapes `https://www.rosterresource.com/mlb-depth-charts/` for lineup order,
rotation order, and bullpen hierarchy.  Results feed into
``compute_lineup_protection()`` in ``contextual_factors.py`` and closer
hierarchy detection.

All HTTP calls use a custom User-Agent header and 15-second timeout.
On **any** failure (network, parse, missing data), every function returns
an empty/neutral result so the rest of the pipeline is unaffected.

Functions
---------
fetch_depth_charts
    Scrape Roster Resource for all 30 MLB teams.
classify_role
    Map depth chart slot data to a human-readable role string.
get_player_lineup_slot
    Look up a player's batting-order slot (1-9) across all teams.
get_player_role
    Derive a player's role string from depth chart data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

# ── Constants ─────────────────────────────────────────────────────────

ROSTER_RESOURCE_URL = "https://www.rosterresource.com/mlb-depth-charts/"

_REQUEST_HEADERS = {"User-Agent": "Fantasy Baseball Draft Tool"}
_REQUEST_TIMEOUT = 15  # seconds

# Team abbreviation mapping from Roster Resource conventions to our canonical codes.
# Roster Resource uses full city/team names in URLs; we normalise to 2-3 letter codes.
_TEAM_SLUG_TO_CODE: dict[str, str] = {
    "arizona-diamondbacks": "ARI",
    "atlanta-braves": "ATL",
    "baltimore-orioles": "BAL",
    "boston-red-sox": "BOS",
    "chicago-cubs": "CHC",
    "chicago-white-sox": "CWS",
    "cincinnati-reds": "CIN",
    "cleveland-guardians": "CLE",
    "colorado-rockies": "COL",
    "detroit-tigers": "DET",
    "houston-astros": "HOU",
    "kansas-city-royals": "KC",
    "los-angeles-angels": "LAA",
    "los-angeles-dodgers": "LAD",
    "miami-marlins": "MIA",
    "milwaukee-brewers": "MIL",
    "minnesota-twins": "MIN",
    "new-york-mets": "NYM",
    "new-york-yankees": "NYY",
    "oakland-athletics": "OAK",
    "philadelphia-phillies": "PHI",
    "pittsburgh-pirates": "PIT",
    "san-diego-padres": "SD",
    "san-francisco-giants": "SF",
    "seattle-mariners": "SEA",
    "st-louis-cardinals": "STL",
    "tampa-bay-rays": "TB",
    "texas-rangers": "TEX",
    "toronto-blue-jays": "TOR",
    "washington-nationals": "WSH",
}

# Reverse lookup: canonical code -> slug (for team page URLs)
_CODE_TO_SLUG: dict[str, str] = {v: k for k, v in _TEAM_SLUG_TO_CODE.items()}

# Valid bullpen role tags we recognise
_BULLPEN_ROLES = {"CL", "SU", "MR", "LR", "LHP", "RHP"}


# ── Public API ────────────────────────────────────────────────────────


def fetch_depth_charts() -> dict[str, dict[str, Any]]:
    """Scrape Roster Resource for all 30 MLB team depth charts.

    Returns
    -------
    dict[str, dict]
        Mapping of team code to depth chart data::

            {
                "NYY": {
                    "lineup": ["Aaron Judge", "Juan Soto", ...],   # 1-9 order
                    "rotation": ["Gerrit Cole", "Carlos Rodon", ...],  # 1-5
                    "bullpen": {
                        "CL": "Clay Holmes",
                        "SU": ["Jonathan Loaisiga", ...]
                    }
                },
                ...
            }

        Returns an empty ``dict`` on any failure.
    """
    if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
        logger.warning("requests or beautifulsoup4 not available; skipping depth chart fetch")
        return {}

    # Quick connection test before attempting 30 individual team pages.
    # Roster Resource is sometimes unreachable; fail fast to avoid a
    # long cascade of per-team timeouts.
    try:
        resp = requests.get(
            ROSTER_RESOURCE_URL,
            headers=_REQUEST_HEADERS,
            timeout=5,
        )
        if resp.status_code != 200:
            logger.warning(
                "Roster Resource unavailable (HTTP %s). Depth charts will not be populated.",
                resp.status_code,
            )
            return {}
    except Exception as exc:
        logger.warning(
            "Roster Resource unreachable: %s. Depth charts will not be populated.",
            exc,
        )
        return {}

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        return _parse_index_page(soup)
    except Exception:
        logger.warning("Failed to parse Roster Resource depth charts")
        return {}


def classify_role(
    lineup_slot: int | None,
    rotation_slot: int | None,
    bullpen_role: str | None,
    *,
    is_committee: bool = False,
) -> str:
    """Classify a player's role from depth chart position data.

    Priority order: lineup starter > platoon > closer/committee > setup >
    rotation > generic bullpen > bench.

    Parameters
    ----------
    lineup_slot : int or None
        Batting order position (1-9) if the player is in the starting
        lineup.  Slots 10-18 indicate a backup/platoon player.
    rotation_slot : int or None
        Rotation order (1-5) if the player is a starting pitcher.
    bullpen_role : str or None
        Bullpen role tag (``"CL"``, ``"SU"``, ``"MR"``, etc.).
    is_committee : bool
        If ``True`` and `bullpen_role` is ``"CL"``, return ``"committee"``
        instead of ``"closer"``.

    Returns
    -------
    str
        One of ``"starter"``, ``"platoon"``, ``"closer"``, ``"committee"``,
        ``"setup"``, ``"rotation"``, ``"bullpen"``, ``"bench"``.
    """
    # Position players in the batting order
    if lineup_slot is not None and isinstance(lineup_slot, (int, float)):
        slot = int(lineup_slot)
        if 1 <= slot <= 9:
            return "starter"
        if 10 <= slot <= 18:
            return "platoon"

    # Pitching roles — check bullpen first (closer/setup are more specific)
    if bullpen_role is not None and isinstance(bullpen_role, str):
        role_upper = bullpen_role.strip().upper()
        if role_upper == "CL":
            return "committee" if is_committee else "closer"
        if role_upper == "SU":
            return "setup"
        # Any other bullpen tag (MR, LR, etc.)
        if role_upper:
            return "bullpen"

    # Starting rotation
    if rotation_slot is not None and isinstance(rotation_slot, (int, float)):
        slot = int(rotation_slot)
        if 1 <= slot <= 5:
            return "starter"

    return "bench"


def get_player_lineup_slot(
    player_name: str,
    depth_data: dict[str, dict[str, Any]],
) -> int | None:
    """Find a player's batting-order slot (1-9) across all teams.

    Uses case-insensitive matching.

    Parameters
    ----------
    player_name : str
        Full player name (e.g. ``"Aaron Judge"``).
    depth_data : dict
        Output from :func:`fetch_depth_charts`.

    Returns
    -------
    int or None
        Lineup slot (1-based) or ``None`` if the player is not found
        in any team's starting lineup.
    """
    if not player_name or not depth_data:
        return None

    target = player_name.strip().lower()

    for _team_code, team_data in depth_data.items():
        lineup = team_data.get("lineup", [])
        for idx, name in enumerate(lineup):
            if isinstance(name, str) and name.strip().lower() == target:
                return idx + 1  # 1-based slot

    return None


def get_player_role(
    player_name: str,
    depth_data: dict[str, dict[str, Any]],
) -> str:
    """Get the role string for a player from depth chart data.

    Checks lineup, rotation, and bullpen in order.  Uses
    case-insensitive matching.

    Parameters
    ----------
    player_name : str
        Full player name.
    depth_data : dict
        Output from :func:`fetch_depth_charts`.

    Returns
    -------
    str
        One of ``"starter"``, ``"closer"``, ``"setup"``, ``"rotation"``,
        ``"bullpen"``, ``"bench"``.
    """
    if not player_name or not depth_data:
        return "bench"

    target = player_name.strip().lower()

    for _team_code, team_data in depth_data.items():
        # Check lineup (batting order 1-9)
        lineup = team_data.get("lineup", [])
        for idx, name in enumerate(lineup):
            if isinstance(name, str) and name.strip().lower() == target:
                return classify_role(
                    lineup_slot=idx + 1,
                    rotation_slot=None,
                    bullpen_role=None,
                )

        # Check rotation
        rotation = team_data.get("rotation", [])
        for idx, name in enumerate(rotation):
            if isinstance(name, str) and name.strip().lower() == target:
                return classify_role(
                    lineup_slot=None,
                    rotation_slot=idx + 1,
                    bullpen_role=None,
                )

        # Check bullpen
        bullpen = team_data.get("bullpen", {})
        if isinstance(bullpen, dict):
            for role_tag, names in bullpen.items():
                # Detect committee closers: CL entry with comma-separated names
                _is_committee = role_tag.upper() == "CL" and isinstance(names, str) and "," in names
                if isinstance(names, str):
                    # Check each name in a potentially comma-separated string
                    for individual_name in names.split(","):
                        if individual_name.strip().lower() == target:
                            return classify_role(
                                lineup_slot=None,
                                rotation_slot=None,
                                bullpen_role=role_tag,
                                is_committee=_is_committee,
                            )
                elif isinstance(names, list):
                    for name in names:
                        if isinstance(name, str) and name.strip().lower() == target:
                            return classify_role(
                                lineup_slot=None,
                                rotation_slot=None,
                                bullpen_role=role_tag,
                            )

    return "bench"


# ── Internal parsing helpers ──────────────────────────────────────────


def _parse_index_page(soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
    """Parse the Roster Resource index page for team links and data.

    The site structure may change at any time — this parser is
    intentionally resilient and returns partial results rather than
    raising.
    """
    result: dict[str, dict[str, Any]] = {}

    # Strategy: find all team links on the index page and fetch each
    # team page individually.  If the index page already contains
    # inline depth chart data (tables/divs), parse those instead.

    # Look for team section links
    team_links: dict[str, str] = {}
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        for slug, code in _TEAM_SLUG_TO_CODE.items():
            if slug in href and code not in team_links:
                team_links[code] = href
                break

    if not team_links:
        logger.warning("No team links found on Roster Resource index page")
        return {}

    for team_code, href in team_links.items():
        try:
            team_data = _fetch_team_depth_chart(href)
            if team_data:
                result[team_code] = team_data
        except Exception:
            logger.warning("Failed to parse depth chart for %s", team_code)
            continue
        time.sleep(0.5)

    return result


def _fetch_team_depth_chart(url: str) -> dict[str, Any] | None:
    """Fetch and parse a single team's depth chart page."""
    # Ensure absolute URL
    if url.startswith("/"):
        url = "https://www.rosterresource.com" + url

    try:
        resp = requests.get(
            url,
            headers=_REQUEST_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception:
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        return _parse_team_page(soup)
    except Exception:
        return None


def _parse_team_page(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract lineup, rotation, and bullpen from a team page.

    The Roster Resource layout typically organises depth charts into
    positional sections.  This parser looks for common patterns:
    - Tables with position headers
    - Ordered lists or divs containing player names
    - Specific CSS classes used by the site

    Since the site structure is external and may change, every extraction
    step is wrapped in try/except.  Partial results are fine — an empty
    lineup list simply means ``compute_lineup_protection()`` will use
    its built-in heuristic fallback.
    """
    lineup: list[str] = []
    rotation: list[str] = []
    bullpen: dict[str, Any] = {}

    # Try to find lineup data — look for tables or sections mentioning
    # batting order, lineup, or position player names.
    lineup = _extract_lineup(soup)
    rotation = _extract_rotation(soup)
    bullpen = _extract_bullpen(soup)

    return {
        "lineup": lineup,
        "rotation": rotation,
        "bullpen": bullpen,
    }


def _extract_lineup(soup: BeautifulSoup) -> list[str]:
    """Extract the batting order (up to 9 names) from a team page."""
    names: list[str] = []
    try:
        # Look for elements that contain lineup / batting order info.
        # Common patterns: tables with "Lineup" header, divs with class
        # containing "lineup", ordered lists under a lineup heading.
        for table in soup.find_all("table"):
            header_text = table.get_text(" ", strip=True).lower()
            if "lineup" in header_text or "batting" in header_text:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    for cell in cells:
                        # Look for links (player names are often links)
                        link = cell.find("a")
                        if link:
                            name = link.get_text(strip=True)
                            if name and len(name) > 2 and len(names) < 9:
                                names.append(name)
                if names:
                    return names[:9]

        # Fallback: look for divs/sections with "lineup" class or id
        for div in soup.find_all(["div", "section"]):
            div_id = (div.get("id") or "").lower()
            div_class = " ".join(div.get("class") or []).lower()
            if "lineup" in div_id or "lineup" in div_class:
                for link in div.find_all("a"):
                    name = link.get_text(strip=True)
                    if name and len(name) > 2 and len(names) < 9:
                        names.append(name)
                if names:
                    return names[:9]
    except Exception:
        pass

    return names[:9]


def _extract_rotation(soup: BeautifulSoup) -> list[str]:
    """Extract the starting rotation (up to 5 names) from a team page."""
    names: list[str] = []
    try:
        for table in soup.find_all("table"):
            header_text = table.get_text(" ", strip=True).lower()
            if "rotation" in header_text or "starting pitcher" in header_text:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    for cell in cells:
                        link = cell.find("a")
                        if link:
                            name = link.get_text(strip=True)
                            if name and len(name) > 2 and len(names) < 5:
                                names.append(name)
                if names:
                    return names[:5]

        for div in soup.find_all(["div", "section"]):
            div_id = (div.get("id") or "").lower()
            div_class = " ".join(div.get("class") or []).lower()
            if "rotation" in div_id or "rotation" in div_class:
                for link in div.find_all("a"):
                    name = link.get_text(strip=True)
                    if name and len(name) > 2 and len(names) < 5:
                        names.append(name)
                if names:
                    return names[:5]
    except Exception:
        pass

    return names[:5]


def _extract_bullpen(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract bullpen roles (closer, setup, etc.) from a team page."""
    result: dict[str, Any] = {}
    try:
        for table in soup.find_all("table"):
            header_text = table.get_text(" ", strip=True).lower()
            if "bullpen" in header_text or "closer" in header_text or "relief" in header_text:
                for row in table.find_all("tr"):
                    row_text = row.get_text(" ", strip=True)
                    row_lower = row_text.lower()
                    cells = row.find_all(["td", "th"])
                    # Try to identify role from row context
                    role = _detect_bullpen_role(row_lower)
                    if role and cells:
                        names_in_row = []
                        for cell in cells:
                            link = cell.find("a")
                            if link:
                                name = link.get_text(strip=True)
                                if name and len(name) > 2:
                                    names_in_row.append(name)
                        if names_in_row:
                            if role == "CL":
                                result["CL"] = ", ".join(names_in_row) if len(names_in_row) > 1 else names_in_row[0]
                            elif role == "SU":
                                result.setdefault("SU", []).extend(names_in_row)
                            else:
                                result.setdefault(role, []).extend(names_in_row)
                if result:
                    return result

        # Fallback: look for divs/sections
        for div in soup.find_all(["div", "section"]):
            div_id = (div.get("id") or "").lower()
            div_class = " ".join(div.get("class") or []).lower()
            if "bullpen" in div_id or "bullpen" in div_class:
                for link in div.find_all("a"):
                    name = link.get_text(strip=True)
                    if name and len(name) > 2:
                        # Without clear role tags, add to generic list
                        result.setdefault("MR", []).append(name)
                if result:
                    return result
    except Exception:
        pass

    return result


def _detect_bullpen_role(text: str) -> str | None:
    """Detect bullpen role from row/cell text content."""
    text_lower = text.lower()
    if (
        "closer" in text_lower
        or "cl " in text_lower
        or text_lower == "cl"
        or text_lower.startswith("cl\t")
        or text_lower.startswith("cl:")
    ):
        return "CL"
    if "setup" in text_lower or "set up" in text_lower or "su " in text_lower:
        return "SU"
    if "middle" in text_lower or "mr " in text_lower:
        return "MR"
    if "long" in text_lower or "lr " in text_lower:
        return "LR"
    return None
