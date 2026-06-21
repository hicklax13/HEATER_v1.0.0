"""Shared, Muncy-DNA-safe resolver for a player's MLB Stats API ``mlb_id``.

Resolves a player's numeric MLB id from (name, team) via
``statsapi.lookup_player``. The frontend renders headshots (mlb_id) and team
logos (team_id) from this id, so a missing id shows a broken image.

SAFETY (the "Muncy-DNA" rule): only a CONFIDENT, UNIQUE match is ever returned.
Two different MLB players can share a name (two "Max Muncy", two "Will Smith"),
so a shared name is disambiguated by the player's MLB team, and anything still
ambiguous returns ``None`` — never a guess. Assigning the WRONG id is strictly
worse than leaving it NULL. Callers must still apply their own write guards
(never overwrite a non-null id; never duplicate an id onto two player rows).

This lives in ``src/`` so both ``scripts/backfill_player_mlb_ids.py`` (the
operator tool) and ``src/data_bootstrap.py`` (the auto-enrichment phase) share
one implementation instead of duplicating it.

statsapi notes: ``lookup_player`` needs an explicit ``season=`` (None and the
current season often return 0 matches mid-season); it is accent-insensitive, so
correctly-accented names resolve directly.
"""

from __future__ import annotations

import re
import sys
import unicodedata

import statsapi

from src.depth_charts import _STATSAPI_TEAM_IDS
from src.valuation import canonicalize_team

# Non-team placeholder values seen in the ``team`` column — never a real club.
_NON_TEAMS = {"", "MLB", "NAN", "NONE", "FA", "N/A"}


def _safe(text: object) -> str:
    """cp1252-safe rendering for the Windows console (accents -> '?')."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _norm(text: str) -> str:
    """ASCII-fold, drop non-ASCII, collapse whitespace, lowercase."""
    folded = _strip_accents(text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", folded).strip().lower()


def _statsapi_team_id(team_abbr: str | None) -> int | None:
    """Map a (possibly fantasy-style) team abbreviation to its MLB Stats API id."""
    if not team_abbr:
        return None
    ab = team_abbr.strip().upper()
    if ab in _NON_TEAMS:
        return None
    # canonicalize_team folds fantasy/legacy variants (AZ->ARI, SFG->SF, OAK->ATH,
    # WAS->WSH, ...) to the keys used by _STATSAPI_TEAM_IDS (the single-source map).
    return _STATSAPI_TEAM_IDS.get(canonicalize_team(ab))


def _surname_present(our_name: str, candidate_full_name: str) -> bool:
    """True when our name's last token appears in the candidate's full name.

    A light sanity check on a unique ``lookup_player`` result so a loose/fuzzy
    single match can't be accepted as confident.
    """
    tokens = _norm(our_name).split()
    if not tokens:
        return False
    return tokens[-1] in _norm(candidate_full_name).split()


def _coerce_id(raw: object) -> int | None:
    """Coerce a raw statsapi id to a positive int, or None for junk/missing."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_mlb_id(name: str, team_abbr: str | None) -> tuple[int | None, str]:
    """Resolve an mlb_id for (name, team). Returns (mlb_id | None, reason).

    Returns None whenever the match is not confident — never guesses. A
    lookup_player ERROR is distinguished from a genuine absence so an API outage
    cannot masquerade as "no such player", and a non-numeric id is skipped rather
    than crashing the caller.
    """
    name = (name or "").strip()
    if not name:
        return None, "blank name"

    matches: list[dict] = []
    season_used = None
    lookup_errored = False
    for season in (2026, 2025, 2024):
        try:
            matches = statsapi.lookup_player(name, sportId=1, season=season) or []
        except Exception as exc:  # noqa: BLE001 - statsapi wraps many net/parse errors
            lookup_errored = True
            print(
                f"  WARN  lookup_player error for '{_safe(name)}' (season={season}): {type(exc).__name__}",
                file=sys.stderr,
            )
            matches = []
        if matches:
            season_used = season
            break
    if not matches:
        if lookup_errored:
            return None, "lookup_player ERRORED (API outage?) - not a confirmed absence"
        return None, "no lookup_player match (seasons 2026/2025/2024)"

    if len(matches) == 1:
        only = matches[0]
        mlb_id = _coerce_id(only.get("id"))
        if mlb_id is None:
            return None, f"single match '{_safe(only.get('fullName'))}' had non-numeric id"
        if not _surname_present(name, only.get("fullName", "")):
            return None, f"single match '{_safe(only.get('fullName'))}' failed surname check"
        return mlb_id, f"unique name match (season={season_used})"

    # Multiple same-name players -> require team disambiguation (Muncy-DNA gate).
    team_id = _statsapi_team_id(team_abbr)
    if team_id is None:
        return None, f"{len(matches)} matches, no team id for '{_safe(team_abbr)}'"
    on_team = [m for m in matches if (m.get("currentTeam") or {}).get("id") == team_id]
    if len(on_team) == 1:
        mlb_id = _coerce_id(on_team[0].get("id"))
        if mlb_id is None:
            return None, f"name+team match for '{_safe(name)}' had non-numeric id"
        return mlb_id, f"name+team match ({len(matches)} same-name)"
    return None, f"ambiguous: {len(matches)} matches, {len(on_team)} on team {team_abbr}"
