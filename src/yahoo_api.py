"""Yahoo Fantasy Sports API integration for league settings and live draft tracking.

Uses the yfpy library for OAuth 2.0 authentication and API access.
Install: pip install yfpy

Setup:
1. Register an app at https://developer.yahoo.com/apps/
2. Get your Consumer Key and Consumer Secret
3. Set game_key to the current MLB season (e.g., "mlb" or the numeric game key)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

YAHOO_CREDS_PATH = Path(__file__).parent.parent / "data" / "yahoo_creds.json"


def save_credentials(consumer_key: str, consumer_secret: str, league_id: str, game_key: str = "mlb"):
    """Save Yahoo API credentials to disk."""
    YAHOO_CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    creds = {
        "consumer_key": consumer_key,
        "consumer_secret": consumer_secret,
        "league_id": league_id,
        "game_key": game_key,
    }
    with open(YAHOO_CREDS_PATH, "w") as f:
        json.dump(creds, f, indent=2)


def load_credentials() -> dict | None:
    """Load saved Yahoo API credentials."""
    if not YAHOO_CREDS_PATH.exists():
        return None
    with open(YAHOO_CREDS_PATH) as f:
        return json.load(f)


def has_credentials() -> bool:
    return YAHOO_CREDS_PATH.exists()


class YahooFantasyClient:
    """Wrapper around yfpy for Yahoo Fantasy API access."""

    def __init__(self, consumer_key: str, consumer_secret: str, league_id: str, game_key: str = "mlb"):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.league_id = league_id
        self.game_key = game_key
        self._query = None

    def connect(self) -> bool:
        """Initialize the Yahoo Fantasy API connection.

        Returns True if successful, False if yfpy is not installed or auth fails.
        """
        try:
            from yfpy.query import YahooFantasySportsQuery
        except ImportError:
            logger.warning("yfpy not installed. Run: pip install yfpy")
            return False

        try:
            auth_dir = Path(__file__).parent.parent / "data"
            self._query = YahooFantasySportsQuery(
                auth_dir=str(auth_dir),
                league_id=self.league_id,
                game_code="mlb",
                game_id=self.game_key if self.game_key.isdigit() else None,
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret,
            )
            return True
        except Exception as e:
            logger.error(f"Yahoo API connection failed: {e}")
            return False

    def get_league_settings(self) -> dict | None:
        """Fetch league settings (categories, roster positions, etc.)."""
        if not self._query:
            return None
        try:
            settings = self._query.get_league_settings()
            result = {
                "name": getattr(settings, "name", "Unknown"),
                "num_teams": getattr(settings, "num_teams", 12),
                "scoring_type": getattr(settings, "scoring_type", "roto"),
            }

            # Extract roster positions
            roster_positions = getattr(settings, "roster_positions", [])
            if roster_positions:
                pos_dict = {}
                for rp in roster_positions:
                    pos = getattr(rp, "position", None) or getattr(rp, "position_type", "")
                    count = getattr(rp, "count", 1)
                    if pos:
                        pos_dict[str(pos)] = int(count)
                result["roster_positions"] = pos_dict

            # Extract stat categories
            stat_categories = getattr(settings, "stat_categories", None)
            if stat_categories:
                cats = []
                for sc in getattr(stat_categories, "stats", []):
                    stat = getattr(sc, "stat", sc)
                    name = getattr(stat, "display_name", getattr(stat, "name", ""))
                    if name:
                        cats.append(str(name))
                result["categories"] = cats

            return result
        except Exception as e:
            logger.error(f"Failed to get league settings: {e}")
            return None

    def get_draft_results(self) -> list | None:
        """Fetch draft results (works during an active draft).

        Returns list of dicts with: pick, round, team_key, player_key, player_name.
        Returns empty list if draft hasn't started yet.
        """
        if not self._query:
            return None
        try:
            draft_results = self._query.get_league_draft_results()
            if not draft_results:
                return []

            picks = []
            for dr in draft_results:
                pick_data = getattr(dr, "draft_result", dr)
                pick = {
                    "pick": int(getattr(pick_data, "pick", 0)),
                    "round": int(getattr(pick_data, "round", 0)),
                    "team_key": str(getattr(pick_data, "team_key", "")),
                    "player_key": str(getattr(pick_data, "player_key", "")),
                }

                # Try to get player name
                player = getattr(pick_data, "player", None)
                if player:
                    name = getattr(player, "name", None)
                    if name:
                        pick["player_name"] = str(getattr(name, "full", name))
                    else:
                        pick["player_name"] = str(player)
                else:
                    pick["player_name"] = f"Player {pick['player_key']}"

                picks.append(pick)

            return sorted(picks, key=lambda x: x["pick"])
        except Exception as e:
            logger.error(f"Failed to get draft results: {e}")
            return None

    def get_teams(self) -> list | None:
        """Fetch list of teams in the league."""
        if not self._query:
            return None
        try:
            teams = self._query.get_league_teams()
            result = []
            for t in teams:
                team = getattr(t, "team", t)
                result.append(
                    {
                        "team_key": str(getattr(team, "team_key", "")),
                        "name": str(getattr(team, "name", "")),
                        "manager": str(getattr(team, "manager", {}).get("nickname", ""))
                        if isinstance(getattr(team, "manager", None), dict)
                        else str(getattr(getattr(team, "manager", None), "nickname", "")),
                    }
                )
            return result
        except Exception as e:
            logger.error(f"Failed to get teams: {e}")
            return None


def apply_league_settings(settings: dict, config) -> list:
    """Apply Yahoo league settings to a LeagueConfig object.

    Returns a list of changes made (for display to the user).
    """
    changes = []

    if "num_teams" in settings:
        old = config.num_teams
        config.num_teams = int(settings["num_teams"])
        if old != config.num_teams:
            changes.append(f"Teams: {old} -> {config.num_teams}")

    if "roster_positions" in settings:
        pos_map = settings["roster_positions"]
        for pos, count in pos_map.items():
            pos_key = str(pos).strip()
            if pos_key in config.roster_slots:
                old = config.roster_slots[pos_key]
                config.roster_slots[pos_key] = int(count)
                if old != int(count):
                    changes.append(f"{pos_key}: {old} -> {count}")

    if "categories" in settings:
        cat_name_map = {
            "R": "R",
            "HR": "HR",
            "RBI": "RBI",
            "SB": "SB",
            "AVG": "AVG",
            "W": "W",
            "SV": "SV",
            "K": "K",
            "ERA": "ERA",
            "WHIP": "WHIP",
            "OBP": "OBP",
            "SLG": "SLG",
            "OPS": "OPS",
            "QS": "QS",
            "HLD": "HLD",
            "HD": "HLD",
            "BB": "BB",
        }
        hit_cats = []
        pitch_cats = []
        hit_set = {"R", "HR", "RBI", "SB", "AVG", "OBP", "SLG", "OPS", "H", "BB", "TB"}
        for raw_cat in settings["categories"]:
            cat = cat_name_map.get(raw_cat.upper(), raw_cat.upper())
            if cat in hit_set:
                hit_cats.append(cat)
            else:
                pitch_cats.append(cat)
        if hit_cats and hit_cats != config.hitting_categories:
            config.hitting_categories = hit_cats
            changes.append(f"Hitting cats: {', '.join(hit_cats)}")
        if pitch_cats and pitch_cats != config.pitching_categories:
            config.pitching_categories = pitch_cats
            changes.append(f"Pitching cats: {', '.join(pitch_cats)}")

    return changes


def sync_draft_picks(client: YahooFantasyClient, draft_state, player_pool) -> int:
    """Sync draft picks from Yahoo API into the local draft state.

    Returns number of new picks synced.
    """
    api_picks = client.get_draft_results()
    if api_picks is None:
        return -1  # API error

    new_picks = 0
    for api_pick in api_picks:
        pick_num = api_pick["pick"] - 1  # Convert to 0-indexed
        if pick_num < draft_state.current_pick:
            continue  # Already recorded

        # Skip picks that are out of order (shouldn't happen but be safe)
        if pick_num != draft_state.current_pick:
            continue

        player_name = api_pick.get("player_name", "Unknown")

        # Try to match player in pool
        matches = player_pool[player_pool["name"].str.lower() == player_name.lower()]
        if matches.empty:
            # Fuzzy match
            parts = player_name.split()
            if len(parts) >= 2:
                matches = player_pool[
                    player_pool["name"].str.contains(parts[-1], case=False, na=False)
                    & player_pool["name"].str.contains(parts[0], case=False, na=False)
                ]

        if not matches.empty:
            p = matches.iloc[0]
            draft_state.make_pick(
                int(p["player_id"]),
                p["name"],
                p["positions"],
            )
            new_picks += 1
        else:
            # Player not in pool — create a placeholder pick
            draft_state.make_pick(
                player_id=-1 * (pick_num + 1),  # negative ID for unknown players
                player_name=player_name,
                positions="Util",
            )
            new_picks += 1

    if new_picks > 0:
        draft_state.save()

    return new_picks
