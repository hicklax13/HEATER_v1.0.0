"""Yahoo Fantasy Sports API client using yfpy.

Provides authenticated access to Yahoo Fantasy Sports data including league
settings, rosters, standings, free agents, transactions, and draft results.
All data can be synced to the local SQLite database for offline use.

Setup (Streamlit web app):
    1. Go to https://developer.yahoo.com/apps/ and click "Create an App"
    2. Select "Web Application"
    3. Under API Permissions, check "Fantasy Sports (Read)"
    4. Set Redirect URI to your Streamlit app URL (e.g. http://localhost:8501/)
    5. Copy your Consumer Key and Consumer Secret
    6. Set environment variables:
       - YAHOO_CLIENT_ID=<your consumer key>
       - YAHOO_CLIENT_SECRET=<your consumer secret>
       - YAHOO_LEAGUE_ID=<your numeric league ID from the Yahoo URL>
    7. The app will handle OAuth via streamlit-oauth browser redirect.

Token persistence:
    yfpy stores OAuth tokens in the auth_dir (data/ by default). Subsequent
    calls reuse the cached token automatically. Call refresh_token() if the
    token expires mid-session.

Rate limiting:
    Yahoo's API allows ~2000 requests/hour. This module inserts a 0.5-second
    delay between consecutive API calls to stay well under the limit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

try:
    from yfpy.query import YahooFantasySportsQuery

    YFPY_AVAILABLE = True
except ImportError:
    YFPY_AVAILABLE = False

logger = logging.getLogger(__name__)

_AUTH_DIR = Path(__file__).parent.parent / "data"
_RATE_LIMIT_SECONDS = 0.5
_last_request_time: float = 0.0


def _rate_limit():
    """Sleep if needed to respect Yahoo API rate limits."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def build_oauth_url(consumer_key: str, redirect_uri: str = "oob") -> str:
    """Build Yahoo OAuth authorization URL.

    Args:
        consumer_key: Yahoo app consumer key.
        redirect_uri: Redirect URI. Use "oob" for installed applications,
            or your app URL for web-based flow.

    Returns:
        The authorization URL string the user should visit in a browser.
    """
    base = "https://api.login.yahoo.com/oauth2/request_auth"
    return f"{base}?client_id={consumer_key}&redirect_uri={redirect_uri}&response_type=code&language=en-us"


def get_league_id_from_env() -> str | None:
    """Read YAHOO_LEAGUE_ID from environment.

    Returns:
        The numeric league ID string, or None if not set.
    """
    import os

    lid = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
    return lid if lid else None


def create_streamlit_oauth_component(consumer_key: str, consumer_secret: str):
    """Create a streamlit-oauth OAuth2Component for Yahoo Fantasy.

    Returns the component if streamlit-oauth is available, else None.
    """
    try:
        from streamlit_oauth import OAuth2Component

        return OAuth2Component(
            client_id=consumer_key,
            client_secret=consumer_secret,
            authorize_endpoint="https://api.login.yahoo.com/oauth2/request_auth",
            token_endpoint="https://api.login.yahoo.com/oauth2/get_token",
        )
    except ImportError:
        logger.warning("streamlit-oauth not installed. Yahoo browser OAuth unavailable.")
        return None


def validate_credentials(consumer_key: str, consumer_secret: str) -> bool:
    """Quick check if credentials have a valid format.

    Yahoo consumer keys are typically 72 characters and secrets are 32.
    This does NOT verify them against Yahoo's servers --- it only catches
    obviously empty or malformed values.

    Args:
        consumer_key: Yahoo app consumer key.
        consumer_secret: Yahoo app consumer secret.

    Returns:
        True if both values look structurally plausible.
    """
    if not consumer_key or not consumer_secret:
        return False
    if not isinstance(consumer_key, str) or not isinstance(consumer_secret, str):
        return False
    key = consumer_key.strip()
    secret = consumer_secret.strip()
    if len(key) < 10 or len(secret) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class YahooFantasyClient:
    """Yahoo Fantasy Sports API client using yfpy.

    Usage::

        client = YahooFantasyClient(league_id="12345")
        if client.authenticate(consumer_key="...", consumer_secret="..."):
            settings = client.get_league_settings()
            standings = client.get_league_standings()
            client.sync_to_db()
    """

    def __init__(
        self,
        league_id: str,
        game_code: str = "mlb",
        season: int = 2026,
    ):
        self.league_id = league_id
        self.game_code = game_code
        self.season = season
        self._query: YahooFantasySportsQuery | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(
        self,
        consumer_key: str,
        consumer_secret: str,
        token_data: dict | None = None,
    ) -> bool:
        """Initialize yfpy Game object with OAuth credentials.

        On the very first call yfpy will open a browser window for the user
        to authorize the app.  The resulting token is cached in *data/* so
        future calls skip the browser flow.

        Args:
            consumer_key: Yahoo app consumer key.
            consumer_secret: Yahoo app consumer secret.
            token_data: Optional pre-existing token dict from streamlit-oauth
                browser flow.  When provided, the token is written to disk so
                yfpy can pick it up without triggering its terminal-based
                OAuth consent prompt.

        Returns:
            True on successful authentication, False otherwise.
        """
        if not YFPY_AVAILABLE:
            logger.warning("yfpy is not installed. Run: pip install 'yfpy>=17.0'")
            return False

        if not validate_credentials(consumer_key, consumer_secret):
            logger.error("Invalid credential format --- key or secret too short / empty.")
            return False

        try:
            _AUTH_DIR.mkdir(parents=True, exist_ok=True)

            # If the caller supplies a pre-existing OAuth token (e.g. from
            # streamlit-oauth), persist it so yfpy skips the browser flow.
            if token_data is not None:
                import json

                token_path = _AUTH_DIR / "token.json"
                token_path.write_text(json.dumps(token_data))

            self._query = YahooFantasySportsQuery(
                auth_dir=str(_AUTH_DIR),
                league_id=self.league_id,
                game_code=self.game_code,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
            )
            # Force a lightweight call to confirm the token is valid.
            _rate_limit()
            self._query.get_league_metadata()
            logger.info("Yahoo Fantasy API authenticated successfully.")
            return True
        except Exception:
            logger.exception("Yahoo API authentication failed.")
            self._query = None
            return False

    @property
    def is_authenticated(self) -> bool:
        """Check if client has valid authentication."""
        return self._query is not None

    def refresh_token(self) -> bool:
        """Refresh an expired OAuth token.

        Returns:
            True on success, False if refresh fails or client not authenticated.
        """
        if not YFPY_AVAILABLE or self._query is None:
            logger.warning("Cannot refresh token --- client not authenticated.")
            return False
        try:
            # yfpy automatically refreshes on the next API call, but we
            # trigger it explicitly with a lightweight metadata request.
            _rate_limit()
            self._query.get_league_metadata()
            logger.info("OAuth token refreshed successfully.")
            return True
        except Exception:
            logger.exception("Token refresh failed.")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_auth(self) -> bool:
        """Log a warning and return False if not authenticated."""
        if not self.is_authenticated:
            logger.warning("Yahoo API call attempted without authentication.")
            return False
        return True

    @staticmethod
    def _safe_attr(obj, attr: str, default=None):
        """Safely read an attribute from a yfpy model object."""
        return getattr(obj, attr, default)

    # ------------------------------------------------------------------
    # League data
    # ------------------------------------------------------------------

    def get_league_settings(self) -> dict:
        """Get scoring categories, roster slots, draft type.

        Returns:
            Dict with keys: ``scoring_categories``, ``roster_positions``,
            ``draft_type``, ``num_teams``, ``name``.  Returns empty dict on
            error or if not authenticated.
        """
        if not self._ensure_auth():
            return {}
        try:
            _rate_limit()
            settings = self._query.get_league_settings()

            # Scoring categories
            scoring_categories: list[str] = []
            stat_cats = self._safe_attr(settings, "stat_categories")
            if stat_cats:
                for sc in getattr(stat_cats, "stats", []):
                    stat = getattr(sc, "stat", sc)
                    display = self._safe_attr(stat, "display_name") or self._safe_attr(stat, "name")
                    if display:
                        scoring_categories.append(str(display))

            # Roster positions
            roster_positions: dict[str, int] = {}
            for rp in self._safe_attr(settings, "roster_positions", []):
                pos = self._safe_attr(rp, "position") or self._safe_attr(rp, "position_type")
                count = self._safe_attr(rp, "count", 1)
                if pos:
                    roster_positions[str(pos)] = int(count)

            return {
                "name": str(self._safe_attr(settings, "name", "Unknown")),
                "num_teams": int(self._safe_attr(settings, "num_teams", 12)),
                "draft_type": str(self._safe_attr(settings, "draft_type", "live")),
                "scoring_categories": scoring_categories,
                "roster_positions": roster_positions,
            }
        except Exception:
            logger.exception("Failed to fetch league settings.")
            return {}

    def get_league_standings(self) -> pd.DataFrame:
        """Get current roto standings for all teams.

        Returns:
            DataFrame with columns: ``team_name``, ``team_key``, ``rank``,
            plus one column per scoring category.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            standings_data = self._query.get_league_standings()

            teams = self._safe_attr(standings_data, "teams", [])
            rows: list[dict] = []
            for entry in teams:
                team = getattr(entry, "team", entry)
                row: dict = {
                    "team_name": str(self._safe_attr(team, "name", "")),
                    "team_key": str(self._safe_attr(team, "team_key", "")),
                    "rank": int(self._safe_attr(team, "rank", 0)),
                }
                # Extract per-category stat values
                team_stats = self._safe_attr(team, "team_stats")
                if team_stats:
                    for s in getattr(team_stats, "stats", []):
                        stat_obj = getattr(s, "stat", s)
                        name = self._safe_attr(stat_obj, "display_name") or self._safe_attr(stat_obj, "name")
                        value = self._safe_attr(stat_obj, "value", 0)
                        if name:
                            try:
                                row[str(name).lower()] = float(value)
                            except (ValueError, TypeError):
                                row[str(name).lower()] = value
                rows.append(row)

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch league standings.")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Rosters
    # ------------------------------------------------------------------

    def get_all_rosters(self) -> pd.DataFrame:
        """Get all teams' rosters.

        Returns:
            DataFrame with columns: ``team_name``, ``team_key``,
            ``player_name``, ``player_id``, ``position``, ``status``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            teams = self._query.get_league_teams()
            all_rows: list[dict] = []

            for entry in teams:
                team = getattr(entry, "team", entry)
                team_name = str(self._safe_attr(team, "name", ""))
                team_key = str(self._safe_attr(team, "team_key", ""))

                _rate_limit()
                roster = self._query.get_team_roster_by_week(team_key)
                for player_entry in roster or []:
                    player = getattr(player_entry, "player", player_entry)
                    name_obj = self._safe_attr(player, "name")
                    full_name = ""
                    if name_obj:
                        full_name = str(self._safe_attr(name_obj, "full", name_obj))

                    positions = self._safe_attr(player, "eligible_positions", [])
                    pos_str = "/".join(str(p) for p in positions) if positions else ""

                    all_rows.append(
                        {
                            "team_name": team_name,
                            "team_key": team_key,
                            "player_name": full_name,
                            "player_id": str(self._safe_attr(player, "player_id", "")),
                            "position": pos_str,
                            "status": str(self._safe_attr(player, "status", "active")),
                        }
                    )

            return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch all rosters.")
            return pd.DataFrame()

    def get_team_roster(self, team_key: str) -> pd.DataFrame:
        """Get a single team's roster.

        Args:
            team_key: The Yahoo team key (e.g. ``"422.l.12345.t.1"``).

        Returns:
            DataFrame with columns: ``player_name``, ``player_id``,
            ``position``, ``status``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            roster = self._query.get_team_roster_by_week(team_key)
            rows: list[dict] = []
            for player_entry in roster or []:
                player = getattr(player_entry, "player", player_entry)
                name_obj = self._safe_attr(player, "name")
                full_name = ""
                if name_obj:
                    full_name = str(self._safe_attr(name_obj, "full", name_obj))

                positions = self._safe_attr(player, "eligible_positions", [])
                pos_str = "/".join(str(p) for p in positions) if positions else ""

                rows.append(
                    {
                        "player_name": full_name,
                        "player_id": str(self._safe_attr(player, "player_id", "")),
                        "position": pos_str,
                        "status": str(self._safe_attr(player, "status", "active")),
                    }
                )

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch team roster for %s.", team_key)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Free agents & transactions
    # ------------------------------------------------------------------

    def get_free_agents(
        self,
        position: str | None = None,
        count: int = 50,
    ) -> pd.DataFrame:
        """Get available free agents.

        Args:
            position: Filter by eligible position (e.g. ``"SS"``). ``None``
                returns all positions.
            count: Maximum number of players to return.

        Returns:
            DataFrame with columns: ``player_name``, ``player_id``,
            ``positions``, ``percent_owned``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            fa_list = self._query.get_league_players(
                player_count=count,
                player_count_start=0,
                status="FA",
            )

            rows: list[dict] = []
            for entry in fa_list or []:
                player = getattr(entry, "player", entry)
                name_obj = self._safe_attr(player, "name")
                full_name = ""
                if name_obj:
                    full_name = str(self._safe_attr(name_obj, "full", name_obj))

                positions = self._safe_attr(player, "eligible_positions", [])
                pos_list = [str(p) for p in positions] if positions else []

                # Apply position filter if requested
                if position and position not in pos_list:
                    continue

                pct = self._safe_attr(player, "percent_owned", 0)
                try:
                    pct = float(pct)
                except (ValueError, TypeError):
                    pct = 0.0

                rows.append(
                    {
                        "player_name": full_name,
                        "player_id": str(self._safe_attr(player, "player_id", "")),
                        "positions": "/".join(pos_list),
                        "percent_owned": pct,
                    }
                )

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch free agents.")
            return pd.DataFrame()

    def get_league_transactions(self, days: int = 7) -> pd.DataFrame:
        """Get recent adds/drops/trades.

        Args:
            days: Number of past days to include.

        Returns:
            DataFrame with columns: ``transaction_id``, ``type``,
            ``player_name``, ``team_from``, ``team_to``, ``timestamp``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            transactions = self._query.get_league_transactions()

            rows: list[dict] = []
            for entry in transactions or []:
                tx = getattr(entry, "transaction", entry)
                tx_id = str(self._safe_attr(tx, "transaction_id", ""))
                tx_type = str(self._safe_attr(tx, "type", ""))
                ts = str(self._safe_attr(tx, "timestamp", ""))

                # Each transaction can involve multiple players
                players = self._safe_attr(tx, "players", [])
                for p_entry in players or []:
                    player = getattr(p_entry, "player", p_entry)
                    name_obj = self._safe_attr(player, "name")
                    full_name = ""
                    if name_obj:
                        full_name = str(self._safe_attr(name_obj, "full", name_obj))

                    tx_data = self._safe_attr(player, "transaction_data")
                    team_from = ""
                    team_to = ""
                    if tx_data:
                        team_from = str(self._safe_attr(tx_data, "source_team_name", ""))
                        team_to = str(self._safe_attr(tx_data, "destination_team_name", ""))

                    rows.append(
                        {
                            "transaction_id": tx_id,
                            "type": tx_type,
                            "player_name": full_name,
                            "team_from": team_from,
                            "team_to": team_to,
                            "timestamp": ts,
                        }
                    )

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch league transactions.")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Draft
    # ------------------------------------------------------------------

    def get_draft_results(self) -> pd.DataFrame:
        """Get draft results for opponent modeling.

        Returns:
            DataFrame with columns: ``pick_number``, ``round``,
            ``team_name``, ``team_key``, ``player_name``, ``player_id``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()
        try:
            _rate_limit()
            draft_results = self._query.get_league_draft_results()
            if not draft_results:
                return pd.DataFrame()

            rows: list[dict] = []
            for entry in draft_results:
                pick_data = getattr(entry, "draft_result", entry)

                player_name = ""
                player = self._safe_attr(pick_data, "player")
                if player:
                    name_obj = self._safe_attr(player, "name")
                    if name_obj:
                        player_name = str(self._safe_attr(name_obj, "full", name_obj))
                    else:
                        player_name = str(player)

                player_key = str(self._safe_attr(pick_data, "player_key", ""))
                if not player_name:
                    player_name = f"Player {player_key}"

                rows.append(
                    {
                        "pick_number": int(self._safe_attr(pick_data, "pick", 0)),
                        "round": int(self._safe_attr(pick_data, "round", 0)),
                        "team_name": str(self._safe_attr(pick_data, "team_name", "")),
                        "team_key": str(self._safe_attr(pick_data, "team_key", "")),
                        "player_name": player_name,
                        "player_id": player_key,
                    }
                )

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("pick_number").reset_index(drop=True)
            return df
        except Exception:
            logger.exception("Failed to fetch draft results.")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Database sync
    # ------------------------------------------------------------------

    def sync_to_db(self) -> dict:
        """Push all Yahoo data to SQLite tables.

        Syncs league standings and rosters to the database using the
        existing upsert functions in ``src.database``.

        Returns:
            Dict of ``{"standings": N, "rosters": N}`` with row counts
            synced, or empty dict on failure.
        """
        if not self._ensure_auth():
            return {}

        # Late import to avoid circular dependency and keep this module
        # free of Streamlit imports.
        from src.database import (
            clear_league_rosters,
            update_refresh_log,
            upsert_league_roster_entry,
            upsert_league_standing,
        )

        counts: dict[str, int] = {"standings": 0, "rosters": 0}

        # --- Standings ---
        try:
            standings_df = self.get_league_standings()
            if not standings_df.empty:
                # Determine which columns are category values (not metadata)
                meta_cols = {"team_name", "team_key", "rank"}
                cat_cols = [c for c in standings_df.columns if c not in meta_cols]

                for _, row in standings_df.iterrows():
                    team_name = row.get("team_name", "")
                    rank = int(row.get("rank", 0))
                    for cat in cat_cols:
                        upsert_league_standing(
                            team_name=team_name,
                            category=cat.upper(),
                            total=float(row.get(cat, 0)),
                            rank=rank,
                        )
                        counts["standings"] += 1

                update_refresh_log("yahoo_standings", "success")
                logger.info("Synced %d standing entries to DB.", counts["standings"])
        except Exception:
            logger.exception("Failed to sync standings to DB.")
            update_refresh_log("yahoo_standings", "error")

        # --- Rosters ---
        try:
            rosters_df = self.get_all_rosters()
            if not rosters_df.empty:
                clear_league_rosters()
                team_indices: dict[str, int] = {}
                idx = 0
                for _, row in rosters_df.iterrows():
                    team_name = row.get("team_name", "")
                    if team_name not in team_indices:
                        team_indices[team_name] = idx
                        idx += 1

                    player_id_str = row.get("player_id", "0")
                    try:
                        player_id = int(player_id_str)
                    except (ValueError, TypeError):
                        player_id = hash(player_id_str) % (10**9)

                    upsert_league_roster_entry(
                        team_name=team_name,
                        team_index=team_indices[team_name],
                        player_id=player_id,
                        roster_slot=row.get("position", ""),
                    )
                    counts["rosters"] += 1

                update_refresh_log("yahoo_rosters", "success")
                logger.info("Synced %d roster entries to DB.", counts["rosters"])
        except Exception:
            logger.exception("Failed to sync rosters to DB.")
            update_refresh_log("yahoo_rosters", "error")

        return counts
