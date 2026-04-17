"""Yahoo Fantasy Sports API client using yfpy.

Provides authenticated access to Yahoo Fantasy Sports data including league
settings, rosters, standings, free agents, transactions, and draft results.
All data can be synced to the local SQLite database for offline use.

Setup (Streamlit web app):
    1. Go to https://developer.yahoo.com/apps/ and click "Create an App"
    2. Select "Installed Application" (recommended) or "Web Application"
    3. Under API Permissions, check "Fantasy Sports (Read)"
    4. Copy your Consumer Key and Consumer Secret
    5. Set environment variables:
       - YAHOO_CLIENT_ID=<your consumer key>
       - YAHOO_CLIENT_SECRET=<your consumer secret>
       - YAHOO_LEAGUE_ID=<your numeric league ID from the Yahoo URL>
    6. The app uses Yahoo's out-of-band (oob) OAuth flow: click a link,
       authorize on Yahoo, paste the verification code back in the app.

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
import requests as _requests

try:
    from yfpy.query import YahooFantasySportsQuery

    YFPY_AVAILABLE = True
except ImportError:
    YFPY_AVAILABLE = False

logger = logging.getLogger(__name__)

_AUTH_DIR = Path(__file__).parent.parent / "data"
_RATE_LIMIT_SECONDS = 0.5
_last_request_time: float = 0.0
_MAX_RETRIES = 3


def _rate_limit():
    """Sleep if needed to respect Yahoo API rate limits."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _request_with_backoff(url: str, headers: dict, timeout: int = 15) -> _requests.Response:
    """GET request with exponential backoff on 429 Too Many Requests.

    Retries up to ``_MAX_RETRIES`` times with delays of 1s, 2s, 4s, ...
    Raises the final response's HTTP error if all retries are exhausted.
    """
    for attempt in range(_MAX_RETRIES + 1):
        _rate_limit()
        resp = _requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 429 or attempt == _MAX_RETRIES:
            resp.raise_for_status()
            return resp
        wait = 2**attempt  # 1s, 2s, 4s
        logger.warning("Yahoo 429 rate-limited — retry %d/%d in %ds", attempt + 1, _MAX_RETRIES, wait)
        time.sleep(wait)
    return resp  # unreachable but satisfies type checker


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
    import urllib.parse

    base = "https://api.login.yahoo.com/oauth2/request_auth"
    params = urllib.parse.urlencode(
        {
            "client_id": consumer_key,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "language": "en-us",
        }
    )
    return f"{base}?{params}"


def get_league_id_from_env() -> str | None:
    """Read YAHOO_LEAGUE_ID from environment.

    Returns:
        The numeric league ID string, or None if not set.
    """
    import os

    lid = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
    return lid if lid else None


def exchange_code_for_token(
    consumer_key: str,
    consumer_secret: str,
    code: str,
) -> dict | None:
    """Exchange a Yahoo OAuth verification code for an access token.

    Uses Yahoo's token endpoint with the oob redirect_uri. This is the
    second step of Yahoo's out-of-band OAuth 2.0 flow: the user authorizes
    the app on Yahoo's site, receives a verification code, and pastes it
    back into the Streamlit app.

    Args:
        consumer_key: Yahoo app consumer key.
        consumer_secret: Yahoo app consumer secret.
        code: The verification code from Yahoo's authorization page.

    Returns:
        Token dict with access_token, refresh_token, token_type, etc.
        Returns None if the exchange fails.
    """
    import base64
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    token_url = "https://api.login.yahoo.com/oauth2/get_token"
    credentials = base64.b64encode(f"{consumer_key}:{consumer_secret}".encode()).decode("utf-8")
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "redirect_uri": "oob",
            "code": code.strip(),
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(token_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
        logger.info("Yahoo OAuth token exchange successful. Keys: %s", list(token_data.keys()))
        return token_data
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        logger.error("Yahoo token exchange HTTP %s: %s", exc.code, body)
        return None
    except Exception as exc:
        logger.error("Yahoo token exchange failed: %s", exc)
        return None


def create_streamlit_oauth_component(consumer_key: str, consumer_secret: str):
    """Create a streamlit-oauth OAuth2Component for Yahoo Fantasy.

    Returns the component if streamlit-oauth is available, else None.

    .. deprecated::
        Use :func:`build_oauth_url` + :func:`exchange_code_for_token` instead.
        The streamlit-oauth popup flow requires a redirect-based OAuth callback,
        but Yahoo Fantasy uses out-of-band (oob) OAuth which shows a verification
        code instead of redirecting. This function is kept for backward compatibility
        but the oob flow is recommended.
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

        When *token_data* is provided (from :func:`exchange_code_for_token`),
        the token is written to a JSON file so yfpy can load it natively
        without triggering its terminal-based OAuth consent prompt.

        Args:
            consumer_key: Yahoo app consumer key.
            consumer_secret: Yahoo app consumer secret.
            token_data: Optional token dict from ``exchange_code_for_token()``.
                Must contain ``access_token``, ``refresh_token``, and
                ``token_type``.

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
            import json

            _AUTH_DIR.mkdir(parents=True, exist_ok=True)

            # If the caller supplies a fresh OAuth token, build the dict
            # in the format yfpy expects. yfpy's yahoo_access_token_json
            # param accepts a dict or JSON *string* (NOT a file path).
            # Required fields: access_token, consumer_key, consumer_secret,
            # guid, refresh_token, token_time, token_type.
            token_dict: dict | None = None
            if token_data is not None:
                # Preserve original token_time from saved tokens so that
                # yfpy/yahoo-oauth can detect expiry and auto-refresh.
                # Only use time.time() for genuinely fresh tokens (those
                # that don't already have a token_time).
                saved_token_time = token_data.get("token_time")
                token_dict = {
                    "access_token": token_data.get("access_token"),
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "expires_in": token_data.get("expires_in", 3600),
                    "guid": token_data.get("guid", token_data.get("xoauth_yahoo_guid", "")),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_time": saved_token_time if saved_token_time else time.time(),
                    "token_type": token_data.get("token_type", "bearer"),
                }
                # Also persist to disk for future sessions
                token_file = _AUTH_DIR / "yahoo_token.json"
                token_file.write_text(json.dumps(token_dict, indent=2))
                logger.info("Wrote Yahoo token to %s", token_file)

            # Build constructor kwargs for yfpy v17+
            # When a token is provided, it already contains consumer_key/secret
            # inside the JSON -- passing them separately triggers a yfpy warning.
            query_kwargs: dict = {
                "league_id": self.league_id,
                "game_code": self.game_code,
                "browser_callback": False,
            }

            if token_dict is not None:
                query_kwargs["yahoo_access_token_json"] = token_dict
                # Disable env var fallback to suppress yfpy warning about
                # token overriding consumer_key/secret (we don't pass those)
                query_kwargs["env_var_fallback"] = False
            else:
                query_kwargs["yahoo_consumer_key"] = consumer_key
                query_kwargs["yahoo_consumer_secret"] = consumer_secret

            self._query = YahooFantasySportsQuery(**query_kwargs)

            # --- Resolve the correct game_key for our season ---
            # yfpy needs EITHER game_id in the constructor OR league_key
            # cached.  Without either, it falls back to "current game"
            # which may be the WRONG season (e.g. 2025 MLB when we
            # need 2026).  The multi-strategy resolver handles pre-season
            # edge cases where Yahoo hasn't registered the new game_key.
            game_key = self._resolve_game_key()
            if game_key:
                self._validate_game_key(game_key)
                self._query.game_id = int(game_key)
                self._query.league_key = f"{game_key}.l.{self.league_id}"
                logger.info(
                    "Resolved Yahoo game_key=%s -> league_key=%s (season %d)",
                    game_key,
                    self._query.league_key,
                    self.season,
                )
            else:
                logger.error(
                    "Could not resolve game_key for season %d. Yahoo sync will likely return empty data.",
                    self.season,
                )

            # Force a lightweight call to confirm the token is valid.
            # Method name varies across yfpy versions — try several.
            try:
                _rate_limit()
                for method_name in (
                    "get_league_metadata",
                    "get_league_info",
                    "get_league_settings",
                ):
                    fn = getattr(self._query, method_name, None)
                    if fn is not None:
                        result = fn()
                        logger.info("Auth validation via %s succeeded", method_name)
                        break
                else:
                    logger.warning("No known yfpy metadata method found; skipping validation call.")
            except Exception as exc:
                logger.warning(
                    "Auth validation call failed: %s (sync may still work)",
                    exc,
                )
            # Write back the (possibly refreshed) token to disk so that
            # subsequent restarts use the latest access_token/token_time.
            try:
                refreshed = getattr(self._query, "_yahoo_access_token_dict", None)
                if refreshed and refreshed.get("access_token"):
                    updated_dict = {
                        "access_token": refreshed.get("access_token"),
                        "consumer_key": consumer_key,
                        "consumer_secret": consumer_secret,
                        "expires_in": refreshed.get("expires_in", 3600),
                        "guid": refreshed.get("guid", ""),
                        "refresh_token": refreshed.get("refresh_token"),
                        "token_time": refreshed.get("token_time", time.time()),
                        "token_type": refreshed.get("token_type", "bearer"),
                    }
                    token_file = _AUTH_DIR / "yahoo_token.json"
                    token_file.write_text(json.dumps(updated_dict, indent=2))
                    logger.info("Wrote refreshed Yahoo token to %s", token_file)
            except Exception:
                logger.debug("Could not write refreshed token.", exc_info=True)

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
        """Refresh an expired OAuth token and persist to disk.

        Returns:
            True on success, False if refresh fails or client not authenticated.
        """
        if not YFPY_AVAILABLE or self._query is None:
            logger.warning("Cannot refresh token --- client not authenticated.")
            return False
        try:
            # yfpy automatically refreshes on the next API call, but we
            # trigger it explicitly with a lightweight metadata request.
            # Use the same fallback loop as authenticate() since method
            # names vary across yfpy versions.
            _rate_limit()
            for method_name in ("get_league_metadata", "get_league_info", "get_league_settings"):
                fn = getattr(self._query, method_name, None)
                if fn is not None:
                    fn()
                    break
            else:
                logger.warning("No known yfpy metadata method found; skipping refresh call.")

            # Persist refreshed token to disk so future restarts use it
            try:
                refreshed = getattr(self._query, "_yahoo_access_token_dict", None)
                if refreshed and refreshed.get("access_token"):
                    token_file = _AUTH_DIR / "yahoo_token.json"
                    if token_file.exists():
                        import json

                        existing = json.loads(token_file.read_text(encoding="utf-8"))
                        existing["access_token"] = refreshed["access_token"]
                        existing["token_time"] = refreshed.get("token_time", time.time())
                        if refreshed.get("refresh_token"):
                            existing["refresh_token"] = refreshed["refresh_token"]
                        token_file.write_text(json.dumps(existing, indent=2))
                        logger.info("Persisted refreshed token to %s", token_file)
            except Exception:
                logger.debug("Could not persist refreshed token.", exc_info=True)

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

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        """Convert a value to str, decoding bytes if needed (Python 3.14 + yfpy)."""
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _get_bearer_token(self) -> str:
        """Extract OAuth bearer token from yfpy internal state or disk.

        Returns:
            The access token string, or empty string if unavailable.
        """
        # Try yfpy's in-memory token first
        refreshed = getattr(self._query, "_yahoo_access_token_dict", None)
        if refreshed and refreshed.get("access_token"):
            return refreshed["access_token"]
        # Fallback: read from disk
        token_file = _AUTH_DIR / "yahoo_token.json"
        if token_file.exists():
            import json

            try:
                saved = json.loads(token_file.read_text(encoding="utf-8"))
                return saved.get("access_token", "")
            except Exception:
                pass
        return ""

    @staticmethod
    def _extract_position(pos_obj) -> str:
        """Extract a clean position abbreviation from a yfpy position object.

        yfpy eligible_positions entries may be model objects with a
        ``position`` attribute, plain dicts with a ``position`` key,
        or plain strings.  This helper handles all three formats.
        """
        # If it's already a plain string (or bytes), decode and return
        if isinstance(pos_obj, (str, bytes)):
            return pos_obj.decode("utf-8", errors="replace") if isinstance(pos_obj, bytes) else pos_obj
        # Plain dicts from _extracted_data (e.g. {"position": "SP"})
        if isinstance(pos_obj, dict):
            raw = pos_obj.get("position")
            if raw is not None:
                if isinstance(raw, bytes):
                    return raw.decode("utf-8", errors="replace")
                return str(raw)
        # yfpy EligiblePosition model objects have a .position attribute
        raw = getattr(pos_obj, "position", None)
        if raw is not None:
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        # Last resort: try .display_name, then str()
        raw = getattr(pos_obj, "display_name", None)
        if raw is not None:
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        return str(pos_obj)

    def _resolve_game_key(self) -> str | None:
        """Resolve Yahoo game_key for ``self.season`` using 3 fallback strategies.

        Strategy 1: ``get_game_key_by_season(year)`` — direct lookup.
        Strategy 2: ``get_all_yahoo_fantasy_game_keys()`` — enumerate all, filter.
        Strategy 3: ``get_current_game_metadata()`` — use "current" if season matches
                     (or as last resort if nothing else works).

        Returns:
            Game key string (e.g. ``"468"``) or ``None`` if all strategies fail.
        """
        # Strategy 1 — direct season lookup
        try:
            _rate_limit()
            gk = self._query.get_game_key_by_season(self.season)
            logger.info("Strategy 1 succeeded: game_key=%s for season %d", gk, self.season)
            return str(gk)
        except Exception as exc:
            logger.warning(
                "Strategy 1 (get_game_key_by_season(%d)) failed: %s",
                self.season,
                exc,
            )

        # Strategy 2 — enumerate all game keys and filter by season
        try:
            _rate_limit()
            all_games = self._query.get_all_yahoo_fantasy_game_keys()
            for game_obj in all_games:
                game = game_obj.get("game") if isinstance(game_obj, dict) else game_obj
                season_val = getattr(game, "season", None)
                if str(season_val) == str(self.season):
                    gk = getattr(game, "game_key", None)
                    if gk:
                        logger.info(
                            "Strategy 2 succeeded: game_key=%s for season %d",
                            gk,
                            self.season,
                        )
                        return str(gk)
            logger.warning(
                "Strategy 2: no game found for season %d among %d games",
                self.season,
                len(all_games),
            )
        except Exception as exc:
            logger.warning("Strategy 2 (get_all_yahoo_fantasy_game_keys) failed: %s", exc)

        # Strategy 3 — use current game metadata, accept if season matches
        # or use as last resort even if it doesn't
        try:
            _rate_limit()
            current = self._query.get_current_game_metadata()
            current_season = getattr(current, "season", None)
            gk = getattr(current, "game_key", None)
            logger.info(
                "Strategy 3: current game is season=%s, game_key=%s",
                current_season,
                gk,
            )
            if str(current_season) == str(self.season) and gk:
                return str(gk)
            # Even if seasons don't match, this may be the game hosting the
            # user's league.  Accept it as the last resort.
            if gk:
                logger.warning(
                    "Strategy 3: current season %s != requested %d, but using game_key=%s as last resort",
                    current_season,
                    self.season,
                    gk,
                )
                return str(gk)
        except Exception as exc:
            logger.warning("Strategy 3 (get_current_game_metadata) failed: %s", exc)

        return None

    def _validate_game_key(self, game_key: str) -> None:
        """Log a warning if the resolved game_key doesn't match expected MLB 2026 key."""
        if self.season == 2026 and str(game_key) != "469":
            logger.warning(
                "Resolved game_key=%s but expected 469 for MLB 2026. Yahoo data may be from wrong season.",
                game_key,
            )

    def _get_user_team_key(self) -> str | None:
        """Discover the authenticated user's team key.

        Iterates league teams and looks for the ``is_owned_by_current_login``
        flag that yfpy sets on the team owned by the OAuth token holder.

        Returns:
            The team_key string for the user's team, or None if not found.
        """
        if not self._ensure_auth():
            return None
        try:
            _rate_limit()
            teams = self._query.get_league_teams()
            for entry in teams or []:
                team = getattr(entry, "team", entry)
                if self._safe_attr(team, "is_owned_by_current_login"):
                    return str(self._safe_attr(team, "team_key", ""))
        except Exception:
            logger.debug("Could not determine user team key.", exc_info=True)
        return None

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
        """Get current standings for all teams.

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

                # Decode bytes team name (Python 3.14 + yfpy edge case)
                raw_name = self._safe_attr(team, "name", "")
                team_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)

                # Rank may live directly on team OR inside team_standings
                rank_val = self._safe_attr(team, "rank")
                if rank_val is None:
                    ts = self._safe_attr(team, "team_standings")
                    if ts is not None:
                        rank_val = self._safe_attr(ts, "rank")
                rank_val = int(rank_val or 0)

                row: dict = {
                    "team_name": team_name,
                    "team_key": str(self._safe_attr(team, "team_key", "")),
                    "rank": rank_val,
                }

                # Extract H2H record from team_standings.outcome_totals
                team_standings = self._safe_attr(team, "team_standings")
                if team_standings:
                    outcome = self._safe_attr(team_standings, "outcome_totals")
                    if outcome:
                        row["wins"] = float(self._safe_attr(outcome, "wins", 0) or 0)
                        row["losses"] = float(self._safe_attr(outcome, "losses", 0) or 0)
                        row["ties"] = float(self._safe_attr(outcome, "ties", 0) or 0)
                        row["percentage"] = float(self._safe_attr(outcome, "percentage", 0) or 0)
                    row["points_for"] = float(self._safe_attr(team_standings, "points_for", 0) or 0)
                    row["points_against"] = float(self._safe_attr(team_standings, "points_against", 0) or 0)

                    streak = self._safe_attr(team_standings, "streak")
                    if streak:
                        s_type = self._safe_str(self._safe_attr(streak, "type", ""))
                        s_val = self._safe_attr(streak, "value", 0)
                        row["streak"] = f"{s_type}{s_val}" if s_type else ""

                # Extract per-category stat values from team_stats
                # (may be None early in the season when no games have been played)
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

            if not rows:
                return pd.DataFrame()

            # Check if any stat category columns were found.
            # yfpy's get_league_standings() often returns team_stats=None,
            # so we need to supplement with direct Yahoo API calls.
            stat_cols_present = any(
                col in rows[0] for col in ("r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip")
            )

            if not stat_cols_present:
                logger.info(
                    "No stat category data in standings response. "
                    "Fetching season stats via direct Yahoo API for %d teams.",
                    len(rows),
                )
                self._supplement_standings_with_season_stats(rows)

            return pd.DataFrame(rows)
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

                # Decode bytes team name (Python 3.14 + yfpy edge case)
                raw_name = self._safe_attr(team, "name", "")
                team_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                team_key = str(self._safe_attr(team, "team_key", ""))

                # Extract team logo URL (BUG-016 fix)
                team_logos = self._safe_attr(team, "team_logos", [])
                logo_url = ""
                if team_logos:
                    first_logo = team_logos[0] if isinstance(team_logos, list) else team_logos
                    logo_obj = getattr(first_logo, "team_logo", first_logo)
                    logo_url = self._safe_str(self._safe_attr(logo_obj, "url", ""))

                # Extract manager name
                managers = self._safe_attr(team, "managers", [])
                manager_name = ""
                if managers:
                    first_mgr = managers[0] if isinstance(managers, list) else managers
                    mgr_obj = getattr(first_mgr, "manager", first_mgr)
                    manager_name = self._safe_str(self._safe_attr(mgr_obj, "nickname", ""))

                # Extract team details: FAAB, waiver priority, activity counts
                faab_balance = self._safe_attr(team, "faab_balance", None)
                waiver_priority = self._safe_attr(team, "waiver_priority", None)
                number_of_moves = self._safe_attr(team, "number_of_moves", None)
                number_of_trades = self._safe_attr(team, "number_of_trades", None)

                # yfpy's get_team_roster_by_week() internally builds
                # {league_key}.t.{team_id}, so we must pass only the
                # numeric team ID (the part after the last ".t."), not
                # the full team_key — otherwise the key gets doubled.
                team_id = team_key.rsplit(".t.", 1)[-1] if ".t." in team_key else team_key

                _rate_limit()
                roster = self._query.get_team_roster_by_week(team_id)

                # roster is a yfpy Roster object — iterate .players,
                # NOT the Roster itself (which yields attribute names).
                player_list = getattr(roster, "players", None) or []
                for player_entry in player_list:
                    player = getattr(player_entry, "player", player_entry)

                    # Filter out traded-away players: get_team_roster_by_week
                    # returns ALL players who appeared on the roster at any
                    # point during the week, including those traded away.
                    # Traded players have selected_position.position = None.
                    selected_pos_obj = self._safe_attr(player, "selected_position", None)
                    selected_position = ""
                    if selected_pos_obj:
                        selected_position = self._safe_str(self._safe_attr(selected_pos_obj, "position", ""))
                    if not selected_position or selected_position in ("None", "null"):
                        continue  # Skip traded-away player ghost

                    name_obj = self._safe_attr(player, "name")
                    full_name = ""
                    if name_obj:
                        raw_full = self._safe_attr(name_obj, "full", name_obj)
                        full_name = (
                            raw_full.decode("utf-8", errors="replace") if isinstance(raw_full, bytes) else str(raw_full)
                        )

                    # yfpy's eligible_positions attribute drops multi-position data
                    # (library bug: list parsing only keeps first element).
                    # Workaround: read raw _extracted_data which preserves the full list,
                    # then fall back to the model attribute for non-yfpy objects.
                    _raw_elig = getattr(player, "_extracted_data", {})
                    _raw_elig = _raw_elig.get("eligible_positions", []) if isinstance(_raw_elig, dict) else []
                    if isinstance(_raw_elig, dict):
                        _raw_elig = [_raw_elig]
                    if not _raw_elig:
                        # Fallback: use yfpy model attribute (works for mocks/non-yfpy)
                        _raw_elig = self._safe_attr(player, "eligible_positions", [])
                    _pos_parts = []
                    for _ep in _raw_elig if isinstance(_raw_elig, list) else []:
                        _p = self._extract_position(_ep)
                        if _p and _p not in _pos_parts:
                            _pos_parts.append(_p)
                    pos_str = ",".join(_pos_parts) if _pos_parts else ""

                    # Extract injury/ownership fields for news + ownership tables
                    injury_note = self._safe_str(self._safe_attr(player, "injury_note", ""))
                    status_full = self._safe_str(self._safe_attr(player, "status_full", ""))
                    percent_owned_raw = self._safe_attr(player, "percent_owned", None)
                    if hasattr(percent_owned_raw, "value"):
                        percent_owned = float(percent_owned_raw.value)
                    elif percent_owned_raw is not None:
                        try:
                            percent_owned = float(percent_owned_raw)
                        except (TypeError, ValueError):
                            percent_owned = None
                    else:
                        percent_owned = None

                    # Extract additional player context fields
                    editorial_team_abbr = self._safe_str(self._safe_attr(player, "editorial_team_abbr", ""))
                    has_player_notes = bool(self._safe_attr(player, "has_player_notes", False))
                    has_recent_player_notes = bool(self._safe_attr(player, "has_recent_player_notes", False))

                    all_rows.append(
                        {
                            "team_name": team_name,
                            "team_key": team_key,
                            "player_name": full_name,
                            "player_id": str(self._safe_attr(player, "player_id", "")),
                            "position": pos_str,
                            "status": self._safe_str(self._safe_attr(player, "status", "active")),
                            "injury_note": injury_note,
                            "status_full": status_full,
                            "percent_owned": percent_owned,
                            "editorial_team_abbr": editorial_team_abbr,
                            "selected_position": selected_position,
                            "has_player_notes": has_player_notes,
                            "has_recent_player_notes": has_recent_player_notes,
                            "team_logo_url": logo_url,
                            "manager_name": manager_name,
                            "faab_balance": faab_balance,
                            "waiver_priority": waiver_priority,
                            "number_of_moves": number_of_moves,
                            "number_of_trades": number_of_trades,
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
            # Extract numeric team ID — yfpy internally prepends league key
            team_id = team_key.rsplit(".t.", 1)[-1] if ".t." in team_key else team_key
            _rate_limit()
            roster = self._query.get_team_roster_by_week(team_id)

            # roster is a yfpy Roster object — iterate .players,
            # NOT the Roster itself (which yields attribute names).
            player_list = getattr(roster, "players", None) or []
            rows: list[dict] = []
            for player_entry in player_list:
                player = getattr(player_entry, "player", player_entry)

                # Filter out traded-away players: get_team_roster_by_week
                # returns ALL players who appeared on the roster at any
                # point during the week. Traded players have position = None.
                selected_pos_obj = self._safe_attr(player, "selected_position", None)
                sel_pos_str = ""
                if selected_pos_obj:
                    sel_pos_str = self._safe_str(self._safe_attr(selected_pos_obj, "position", ""))
                if not sel_pos_str or sel_pos_str in ("None", "null"):
                    continue  # Skip traded-away player ghost

                name_obj = self._safe_attr(player, "name")
                full_name = ""
                if name_obj:
                    raw_full = self._safe_attr(name_obj, "full", name_obj)
                    full_name = (
                        raw_full.decode("utf-8", errors="replace") if isinstance(raw_full, bytes) else str(raw_full)
                    )

                # yfpy eligible_positions drops multi-position data — use raw _extracted_data
                _raw_elig2 = getattr(player, "_extracted_data", {})
                _raw_elig2 = _raw_elig2.get("eligible_positions", []) if isinstance(_raw_elig2, dict) else []
                if isinstance(_raw_elig2, dict):
                    _raw_elig2 = [_raw_elig2]
                if not _raw_elig2:
                    _raw_elig2 = self._safe_attr(player, "eligible_positions", [])
                _pos_parts2 = []
                for _ep2 in _raw_elig2 if isinstance(_raw_elig2, list) else []:
                    _p2 = self._extract_position(_ep2)
                    if _p2 and _p2 not in _pos_parts2:
                        _pos_parts2.append(_p2)
                pos_str = ",".join(_pos_parts2) if _pos_parts2 else ""

                # Extract injury/ownership fields
                injury_note = self._safe_str(self._safe_attr(player, "injury_note", ""))
                status_full = self._safe_str(self._safe_attr(player, "status_full", ""))
                percent_owned_raw = self._safe_attr(player, "percent_owned", None)
                if hasattr(percent_owned_raw, "value"):
                    percent_owned = float(percent_owned_raw.value)
                elif percent_owned_raw is not None:
                    try:
                        percent_owned = float(percent_owned_raw)
                    except (TypeError, ValueError):
                        percent_owned = None
                else:
                    percent_owned = None

                rows.append(
                    {
                        "player_name": full_name,
                        "player_id": str(self._safe_attr(player, "player_id", "")),
                        "position": pos_str,
                        "status": self._safe_str(self._safe_attr(player, "status", "active")),
                        "injury_note": injury_note,
                        "status_full": status_full,
                        "percent_owned": percent_owned,
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
        start: int = 0,
    ) -> pd.DataFrame:
        """Get available free agents via direct Yahoo API call.

        Bypasses yfpy's ``get_league_players()`` which doesn't support
        ``status=FA`` server-side filtering. Calls the Yahoo Fantasy API
        directly with OAuth Bearer token.

        Args:
            position: Filter by eligible position (e.g. ``"SS"``). ``None``
                returns all positions.
            count: Maximum number of players to return per batch.
            start: Pagination offset (0-based).

        Returns:
            DataFrame with columns: ``player_name``, ``player_key``,
            ``positions``, ``team``, ``percent_owned``.
        """
        if not self._ensure_auth():
            return pd.DataFrame()

        # Get fresh access token from yfpy's internal state
        token_dict = getattr(self._query, "_yahoo_access_token_dict", None)
        if not token_dict:
            # Fallback: read from token file
            try:
                import json as _json

                token_dict = _json.loads((_AUTH_DIR / "yahoo_token.json").read_text())
            except Exception:
                logger.warning("Could not get access token for direct API call.")
                return pd.DataFrame()

        access_token = token_dict.get("access_token", "")
        if not access_token:
            logger.warning("No access token available for direct API call.")
            return pd.DataFrame()

        league_key = f"{self._query.game_id}.l.{self.league_id}"
        headers = {"Authorization": f"Bearer {access_token}"}

        # Aggregate position sets for "B" (batter) / "P" (pitcher) filters
        _BATTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH", "Util"}
        _PITCHER_POSITIONS = {"SP", "RP", "P"}

        try:
            url = (
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/"
                f"{league_key}/players;status=FA;count={count};start={start}"
                f";sort=OR;out=percent_owned?format=json"
            )
            resp = _request_with_backoff(url, headers=headers, timeout=15)
            data = resp.json()

            league = data.get("fantasy_content", {}).get("league", [])
            if not isinstance(league, list) or len(league) < 2:
                return pd.DataFrame()

            players_obj = league[1].get("players", {})
            returned = players_obj.get("count", 0)

            rows: list[dict] = []
            for i in range(returned):
                p = players_obj.get(str(i), {}).get("player", [])
                if not p:
                    continue
                # p[0] = list of player info dicts
                # p[1..N] = subresources (starting_status, percent_owned)
                info = p[0] if isinstance(p[0], list) else []
                name = pos = team_abbr = player_key = ""
                for item in info:
                    if isinstance(item, dict):
                        if "name" in item:
                            name = item["name"].get("full", "")
                        if "editorial_team_abbr" in item:
                            team_abbr = item["editorial_team_abbr"]
                        if "display_position" in item:
                            pos = item["display_position"]
                        if "player_key" in item:
                            player_key = item["player_key"]

                if not name:
                    continue

                # Parse percent_owned from subresources (p[1], p[2], ...)
                pct_owned = 0.0
                for sub in p[1:]:
                    if isinstance(sub, dict) and "percent_owned" in sub:
                        po_data = sub["percent_owned"]
                        if isinstance(po_data, list):
                            for po_item in po_data:
                                if isinstance(po_item, dict) and "value" in po_item:
                                    pct_owned = float(po_item["value"])
                        elif isinstance(po_data, dict):
                            pct_owned = float(po_data.get("value", 0))

                # Position filter — supports exact codes ("SS") and
                # aggregate filters ("B" for all batters, "P" for all pitchers)
                if position:
                    pos_set = {p.strip() for p in pos.split(",")}
                    pos_upper = position.upper()
                    if pos_upper == "B":
                        if not pos_set & _BATTER_POSITIONS:
                            continue
                    elif pos_upper == "P":
                        if not pos_set & _PITCHER_POSITIONS:
                            continue
                    elif position not in pos_set:
                        continue

                rows.append(
                    {
                        "player_name": name,
                        "player_key": player_key,
                        "positions": pos,
                        "team": team_abbr,
                        "percent_owned": pct_owned,
                    }
                )

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch free agents via direct API.")
            return pd.DataFrame()

    def get_all_free_agents(self, max_players: int = 500) -> pd.DataFrame:
        """Fetch all available free agents by paginating the Yahoo API.

        Args:
            max_players: Maximum total FAs to fetch.

        Returns:
            Concatenated DataFrame of all FA batches.
        """
        batch_size = 25
        all_dfs: list[pd.DataFrame] = []
        for start in range(0, max_players, batch_size):
            df = self.get_free_agents(count=batch_size, start=start)
            if df.empty:
                break
            all_dfs.append(df)
            if len(df) < batch_size:
                break
        if not all_dfs:
            return pd.DataFrame()
        result = pd.concat(all_dfs, ignore_index=True)
        logger.info("Fetched %d total free agents from Yahoo API.", len(result))
        return result

    def get_league_transactions(self) -> pd.DataFrame:
        """Get recent adds/drops/trades.

        Returns:
            DataFrame with columns: ``transaction_id``, ``type``,
            ``player_name``, ``team_from``, ``team_to``, ``timestamp``.

        Note:
            Yahoo's API returns all recent transactions without a date
            filter parameter. Client-side filtering by date can be applied
            on the ``timestamp`` column after calling this method.
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
                        raw_full = self._safe_attr(name_obj, "full", name_obj)
                        full_name = self._safe_str(raw_full)

                    tx_data = self._safe_attr(player, "transaction_data")
                    team_from = ""
                    team_to = ""
                    if tx_data:
                        team_from = self._safe_str(self._safe_attr(tx_data, "source_team_name", ""))
                        team_to = self._safe_str(self._safe_attr(tx_data, "destination_team_name", ""))

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
                        raw_full = self._safe_attr(name_obj, "full", name_obj)
                        player_name = self._safe_str(raw_full)
                    else:
                        player_name = self._safe_str(player)

                player_key = str(self._safe_attr(pick_data, "player_key", ""))
                if not player_name:
                    player_name = f"Player {player_key}"

                rows.append(
                    {
                        "pick_number": int(self._safe_attr(pick_data, "pick", 0)),
                        "round": int(self._safe_attr(pick_data, "round", 0)),
                        "team_name": self._safe_str(self._safe_attr(pick_data, "team_name", "")),
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
    # ADP data
    # ------------------------------------------------------------------

    def fetch_yahoo_adp(self) -> list[dict]:
        """Fetch Yahoo Average Draft Position data.

        Uses draft results from the league to compute per-player ADP.
        Returns a list of dicts: ``[{"name": str, "yahoo_adp": float}, ...]``.

        When not authenticated or on failure, returns an empty list
        (graceful degradation).
        """
        if not self._ensure_auth():
            return []
        try:
            draft_df = self.get_draft_results()
            if draft_df.empty:
                return []

            # Compute average pick position per player (ADP)
            adp_df = draft_df.groupby("player_name")["pick_number"].mean().reset_index()
            adp_df.columns = ["name", "yahoo_adp"]
            adp_df = adp_df.sort_values("yahoo_adp").reset_index(drop=True)
            return adp_df.to_dict("records")
        except Exception:
            logger.exception("Failed to fetch Yahoo ADP.")
            return []

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
            get_connection,
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

                # Identify the authenticated user's team so we can flag it
                user_team_key = self._get_user_team_key()
                logger.debug("User team key: %s", user_team_key)

                # Store team metadata (logos, manager names) — BUG-016 fix
                if "team_logo_url" in rosters_df.columns:
                    team_meta = rosters_df.drop_duplicates(subset=["team_key"])[
                        ["team_key", "team_name", "team_logo_url", "manager_name"]
                    ]
                    conn_tm = get_connection()
                    try:
                        for _, tm_row in team_meta.iterrows():
                            tk = tm_row.get("team_key", "")
                            is_user = user_team_key is not None and tk == user_team_key
                            conn_tm.execute(
                                "INSERT OR REPLACE INTO league_teams "
                                "(team_key, team_name, logo_url, manager_name, is_user_team) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (
                                    tk,
                                    tm_row.get("team_name", ""),
                                    tm_row.get("team_logo_url", ""),
                                    tm_row.get("manager_name", ""),
                                    int(is_user),
                                ),
                            )
                        conn_tm.commit()
                        logger.info("Stored %d team metadata entries", len(team_meta))

                        # Persist team details (FAAB, waiver priority, activity counts)
                        detail_cols = ["faab_balance", "waiver_priority", "number_of_moves", "number_of_trades"]
                        if any(c in rosters_df.columns for c in detail_cols):
                            team_details = rosters_df.drop_duplicates(subset=["team_key"])
                            for _, td_row in team_details.iterrows():
                                try:
                                    tk = td_row.get("team_key", "")
                                    faab = td_row.get("faab_balance")
                                    wpri = td_row.get("waiver_priority")
                                    nmov = td_row.get("number_of_moves")
                                    ntrd = td_row.get("number_of_trades")
                                    if any(
                                        v is not None and not (isinstance(v, float) and v != v)
                                        for v in [faab, wpri, nmov, ntrd]
                                    ):
                                        conn_tm.execute(
                                            """UPDATE league_teams SET
                                               faab_balance = COALESCE(?, faab_balance),
                                               waiver_priority = COALESCE(?, waiver_priority),
                                               number_of_moves = COALESCE(?, number_of_moves),
                                               number_of_trades = COALESCE(?, number_of_trades)
                                               WHERE team_key = ?""",
                                            (
                                                float(faab)
                                                if faab is not None and not (isinstance(faab, float) and faab != faab)
                                                else None,
                                                int(wpri)
                                                if wpri is not None and not (isinstance(wpri, float) and wpri != wpri)
                                                else None,
                                                int(nmov)
                                                if nmov is not None and not (isinstance(nmov, float) and nmov != nmov)
                                                else None,
                                                int(ntrd)
                                                if ntrd is not None and not (isinstance(ntrd, float) and ntrd != ntrd)
                                                else None,
                                                tk,
                                            ),
                                        )
                                except Exception:
                                    logger.debug("Could not extract team details for %s", tk)
                            conn_tm.commit()
                    finally:
                        conn_tm.close()

                # Resolve Yahoo player names to local player_ids before storing
                from src.live_stats import match_player_id

                team_indices: dict[str, int] = {}
                idx = 0
                skipped = 0
                for _, row in rosters_df.iterrows():
                    team_name = row.get("team_name", "")
                    if team_name not in team_indices:
                        team_indices[team_name] = idx
                        idx += 1

                    # Resolve player name → local player_id (not Yahoo's ID)
                    # Yahoo appends "(Pitcher)"/"(Batter)" to dual-eligible players
                    # like Ohtani — strip these suffixes before matching.
                    import re as _re

                    raw_player_name = row.get("player_name", "")
                    player_name = _re.sub(r"\s*\((?:Pitcher|Batter|P|B)\)\s*$", "", raw_player_name).strip()
                    team_abbr = row.get("editorial_team_abbr", "")
                    player_id = match_player_id(player_name, team_abbr)
                    if player_id is None:
                        # Fallback: try fuzzy match via DB query
                        conn = get_connection()
                        try:
                            # Use cleaned name (no parenthetical suffix)
                            clean = _re.sub(r"\s*\([^)]*\)\s*$", "", player_name).strip()
                            parts = clean.split()
                            if len(parts) >= 2:
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                                    (f"%{parts[0]}%", f"%{parts[-1]}%"),
                                )
                                result = cursor.fetchone()
                                if result:
                                    player_id = result[0]
                        finally:
                            conn.close()

                    if player_id is None:
                        skipped += 1
                        logger.debug(
                            "Could not match Yahoo player '%s' to local DB, skipping.",
                            player_name,
                        )
                        continue

                    team_key = row.get("team_key", "")
                    is_user = user_team_key is not None and team_key == user_team_key

                    upsert_league_roster_entry(
                        team_name=team_name,
                        team_index=team_indices[team_name],
                        player_id=player_id,
                        roster_slot=row.get("position", ""),
                        is_user_team=is_user,
                        status=row.get("status", "active"),
                        selected_position=row.get("selected_position", ""),
                        editorial_team_abbr=row.get("editorial_team_abbr", ""),
                    )
                    counts["rosters"] += 1

                    # --- Store ownership trends if percent_owned available ---
                    pct_owned = row.get("percent_owned")
                    if pct_owned is not None:
                        try:
                            from datetime import UTC, datetime

                            today = datetime.now(UTC).strftime("%Y-%m-%d")
                            conn_ot = get_connection()
                            try:
                                conn_ot.execute(
                                    "INSERT OR REPLACE INTO ownership_trends "
                                    "(player_id, date, percent_owned) VALUES (?, ?, ?)",
                                    (player_id, today, float(pct_owned)),
                                )
                                conn_ot.commit()
                            finally:
                                conn_ot.close()
                        except Exception:
                            logger.debug("Failed to store ownership for player %s", player_id)

                    # --- Store injury news if injury_note present ---
                    inj_note = row.get("injury_note", "")
                    status_full = row.get("status_full", "")
                    if inj_note or (status_full and status_full.lower() not in ("", "active")):
                        try:
                            from datetime import UTC, datetime

                            body_part = inj_note or ""
                            status_text = status_full or "unknown"
                            # BUG-013 fix: create meaningful headline instead of just body part
                            if body_part and body_part.lower() not in (
                                "injured",
                                "day-to-day",
                                "10-day il",
                                "15-day il",
                                "60-day il",
                            ):
                                headline = f"Placed on IL — {body_part} issue"
                            else:
                                headline = status_text
                            # BUG-007 fix: include published_at to satisfy UNIQUE constraint
                            pub_at = datetime.now(UTC).isoformat()
                            conn_inj = get_connection()
                            try:
                                conn_inj.execute(
                                    "INSERT OR IGNORE INTO player_news "
                                    "(player_id, source, headline, news_type, il_status, "
                                    "injury_body_part, published_at) "
                                    "VALUES (?, 'yahoo', ?, 'injury', ?, ?, ?)",
                                    (player_id, headline, status_text, body_part, pub_at),
                                )
                                conn_inj.commit()
                            finally:
                                conn_inj.close()
                        except Exception:
                            logger.debug("Failed to store injury news for player %s", player_id)

                if skipped:
                    logger.info(
                        "Skipped %d players not found in local DB (sync %d).",
                        skipped,
                        counts["rosters"],
                    )

                update_refresh_log("yahoo_rosters", "success")
                logger.info("Synced %d roster entries to DB.", counts["rosters"])

                # Compute ownership deltas after storing ownership trends
                try:
                    from src.database import compute_ownership_deltas

                    compute_ownership_deltas()
                except Exception:
                    logger.debug("Ownership delta computation failed")
        except Exception:
            logger.exception("Failed to sync rosters to DB.")
            update_refresh_log("yahoo_rosters", "error")

        return counts

    # ------------------------------------------------------------------
    # Supplementary season stats (direct Yahoo REST API)
    # ------------------------------------------------------------------

    def _supplement_standings_with_season_stats(self, rows: list[dict]) -> None:
        """Fetch season-long stat totals for each team via direct Yahoo REST API.

        yfpy's ``get_league_standings()`` returns ``team_stats=None`` in H2H
        leagues, so standings only contain W-L-T records.  This method calls
        the Yahoo REST API directly (same pattern as
        ``_get_team_week_stats_raw``) to fetch season stats for each team
        and merges them into the ``rows`` list in-place.

        Args:
            rows: List of row dicts from ``get_league_standings()``.
                  Modified in-place to add stat category columns.
        """
        token = self._get_bearer_token()
        if not token:
            logger.warning("No bearer token available for supplementary season stats.")
            return

        headers = {"Authorization": f"Bearer {token}"}

        for row in rows:
            team_key = row.get("team_key", "")
            if not team_key:
                continue

            url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/stats;type=season?format=json"
            try:
                _rate_limit()
                resp = _requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    logger.warning(
                        "Yahoo API returned %d for season stats of %s",
                        resp.status_code,
                        row.get("team_name", team_key),
                    )
                    continue

                team_data = resp.json().get("fantasy_content", {}).get("team", [])
                if isinstance(team_data, list):
                    for item in team_data:
                        if not isinstance(item, dict):
                            continue
                        if "team_stats" in item:
                            for entry in item["team_stats"].get("stats", []):
                                stat = entry.get("stat", {})
                                sid = str(stat.get("stat_id", ""))
                                val = str(stat.get("value", ""))
                                if sid in self._STAT_ID_MAP:
                                    cat_name = self._STAT_ID_MAP[sid].lower()
                                    try:
                                        row[cat_name] = float(val)
                                    except (ValueError, TypeError):
                                        row[cat_name] = val
            except Exception:
                logger.exception(
                    "Failed to fetch season stats for %s",
                    row.get("team_name", team_key),
                )

    # ------------------------------------------------------------------
    # Matchup scoreboard
    # ------------------------------------------------------------------

    # Yahoo stat_id → category abbreviation (from league settings)
    _STAT_ID_MAP: dict[str, str] = {
        "7": "R",
        "12": "HR",
        "13": "RBI",
        "16": "SB",
        "3": "AVG",
        "4": "OBP",
        "28": "W",
        "29": "L",
        "32": "SV",
        "42": "K",
        "26": "ERA",
        "27": "WHIP",
    }
    _INVERSE_CATS: set[str] = {"L", "ERA", "WHIP"}
    _ALL_CATS: list[str] = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]

    def _get_team_week_stats_raw(self, team_key: str, week: int) -> tuple[dict[str, str], float]:
        """Fetch per-category stats for a team/week via Yahoo REST API.

        yfpy's ``get_team_stats_by_week()`` crashes on completed weeks because
        the response lacks ``team_projected_points``.  This method calls the
        Yahoo REST API directly with the OAuth bearer token to bypass the
        yfpy parsing bug.

        Args:
            team_key: Full Yahoo team key (e.g. ``"469.l.109662.t.9"``).
            week: Scoring week number.

        Returns:
            Tuple of (stats_dict, category_points) where stats_dict maps
            category abbreviations like ``"R"`` to string values, and
            category_points is the total H2H category wins for the week.
        """
        stats: dict[str, str] = {}
        points: float = 0.0

        token = self._get_bearer_token()
        if not token:
            logger.warning("No bearer token available for raw API call.")
            return stats, points

        url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/stats;type=week;week={week}?format=json"
        try:
            _rate_limit()
            resp = _requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            if resp.status_code != 200:
                logger.warning("Yahoo raw API returned %d for %s week %d", resp.status_code, team_key, week)
                return stats, points

            team_data = resp.json().get("fantasy_content", {}).get("team", [])
            if isinstance(team_data, list):
                for item in team_data:
                    if not isinstance(item, dict):
                        continue
                    if "team_stats" in item:
                        for entry in item["team_stats"].get("stats", []):
                            stat = entry.get("stat", {})
                            sid = str(stat.get("stat_id", ""))
                            val = str(stat.get("value", ""))
                            if sid in self._STAT_ID_MAP:
                                stats[self._STAT_ID_MAP[sid]] = val
                    if "team_points" in item:
                        points = float(item["team_points"].get("total", 0))
        except Exception:
            logger.exception("Raw Yahoo API call failed for %s week %d", team_key, week)

        return stats, points

    def get_current_matchup(self) -> dict | None:
        """Fetch the user's current (or most recent) weekly matchup with scores.

        Returns a dict with keys:
            week, status, user_name, opp_name, wins, losses, ties,
            categories (list of dicts), user_points, opp_points.
        Returns None on error or if not authenticated.
        """
        if not self._ensure_auth():
            return None

        try:
            # Determine current week
            from datetime import UTC, datetime

            today_dt = datetime.now(UTC)
            today = today_dt.strftime("%Y-%m-%d")
            # Fallback: compute week from MLB 2026 fantasy season start (Mar 23).
            # Yahoo's get_game_weeks API sometimes returns empty/stale data,
            # which would strand us on Week 1 forever without this fallback.
            _SEASON_START = datetime(2026, 3, 23, tzinfo=UTC)
            _days_in = (today_dt - _SEASON_START).days
            current_week = max(1, min(24, (_days_in // 7) + 1)) if _days_in >= 0 else 1
            try:
                _rate_limit()
                weeks = self._query.get_game_weeks_by_game_id(game_id=int(self._query.game_id))
                for w in weeks or []:
                    start = str(getattr(w, "start", ""))
                    end = str(getattr(w, "end", ""))
                    wk = getattr(w, "week", None)
                    if start and end and start <= today <= end:
                        current_week = int(wk) if wk else current_week
                        break
            except Exception:
                logger.debug(
                    "Could not determine current week from Yahoo API; using date-based fallback (week %d).",
                    current_week,
                    exc_info=True,
                )

            # Get scoreboard for the current week
            _rate_limit()
            scoreboard = self._query.get_league_scoreboard_by_week(chosen_week=current_week)
            matchups = getattr(scoreboard, "matchups", [])

            user_team_key = self._get_user_team_key()
            if not user_team_key:
                logger.warning("Could not determine user team key for matchup.")
                return None

            # Find user's matchup
            for m in matchups:
                teams = getattr(m, "teams", None)
                if not teams:
                    continue

                team_info = []
                for t in teams:
                    team_obj = getattr(t, "team", t)
                    tk = self._safe_str(getattr(team_obj, "team_key", ""))
                    tn = self._safe_str(getattr(team_obj, "name", ""))
                    team_info.append((tk, tn))

                user_keys = [ti[0] for ti in team_info]
                if str(user_team_key) not in user_keys:
                    continue

                # Found user's matchup
                user_name = ""
                opp_key = ""
                opp_name = ""
                for tk, tn in team_info:
                    if tk == str(user_team_key):
                        user_name = tn
                    else:
                        opp_key = tk
                        opp_name = tn

                status = self._safe_str(getattr(m, "status", ""))

                # Fetch per-category stats for both teams
                user_stats, user_points = self._get_team_week_stats_raw(str(user_team_key), current_week)
                opp_stats, opp_points = self._get_team_week_stats_raw(opp_key, current_week)

                # Compute category wins/losses/ties
                wins = losses = ties = 0
                categories = []
                for cat in self._ALL_CATS:
                    yv_str = user_stats.get(cat, "-")
                    ov_str = opp_stats.get(cat, "-")
                    result = "-"
                    try:
                        yv = float(yv_str) if yv_str not in ("-", "") else None
                        ov = float(ov_str) if ov_str not in ("-", "") else None
                        if yv is not None and ov is not None:
                            if cat in self._INVERSE_CATS:
                                if yv < ov:
                                    result = "WIN"
                                    wins += 1
                                elif yv > ov:
                                    result = "LOSS"
                                    losses += 1
                                else:
                                    result = "TIE"
                                    ties += 1
                            else:
                                if yv > ov:
                                    result = "WIN"
                                    wins += 1
                                elif yv < ov:
                                    result = "LOSS"
                                    losses += 1
                                else:
                                    result = "TIE"
                                    ties += 1
                    except (ValueError, TypeError):
                        pass
                    categories.append({"cat": cat, "you": yv_str, "opp": ov_str, "result": result})

                return {
                    "week": current_week,
                    "status": status,
                    "user_name": user_name,
                    "opp_name": opp_name,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "categories": categories,
                    "user_points": user_points,
                    "opp_points": opp_points,
                }

            logger.warning("User matchup not found in week %d scoreboard.", current_week)
            return None

        except Exception:
            logger.exception("Failed to get current matchup.")
            return None
