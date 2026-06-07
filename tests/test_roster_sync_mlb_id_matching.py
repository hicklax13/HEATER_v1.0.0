"""PR22 guard (2026-05-21): roster sync must prefer (mlb_id) and then
(name + editorial_team_abbr) before falling back to name-only matching,
and the name-only fallback must log a WARNING when there are multiple
candidates with a team mismatch.

Background — the Muncy DNA collision:
  The DB had exactly ONE "Max Muncy" row (player_id=71, team='ATH',
  mlb_id=691777 — the Oakland Athletics rookie). The user actually
  rosters LAD Max Muncy (mlb_id=571970 — the veteran 3B). Yahoo's
  roster sync passed editorial_team_abbr='LAD' but match_player_id's
  (name + team) match found nothing (no LAD Muncy in DB) so it fell
  through to the name-only match, silently picked the ATH Muncy, and
  the league_rosters row got the wrong player_id. Every downstream
  query (FA recommender, optimizer, lineup, war room, etc.) then
  reasoned about the ATH rookie's stats instead of the LAD veteran's.

  Root cause: name-only fallback was silent. No log entry, no warning.

This guard pins:
  1. When (name + editorial_team_abbr) matches a unique row, use it.
  2. When (mlb_id) is provided by the caller and matches a row, prefer
     it over name+team (mlb_id is more precise than any name match).
  3. When name+team yields nothing but name-only yields a single row
     with a DIFFERENT team than editorial_team_abbr, the match still
     resolves (back-compat) but a WARNING is logged identifying it as
     a DNA-collision risk so operators can investigate.

PR23 (companion) is the one-time migration that fixes the existing
Muncy row in the local DB.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _seed_player(db_path: str, player_id: int, name: str, team: str, mlb_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?, ?)",
            (player_id, name, team, "3B", 1, mlb_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenario 1: BOTH Muncys in the DB. Yahoo says editorial_team_abbr='LAD'.
# The sync must prefer the LAD row, NOT the ATH row.
# ---------------------------------------------------------------------------


def test_two_muncys_yahoo_lad_picks_lad(temp_db):
    """When both ATH-Muncy (rookie) and LAD-Muncy (veteran) exist in
    players, a Yahoo roster row with editorial_team_abbr='LAD' must
    resolve to the LAD player_id — not the ATH one — because the
    (name + team) precise match wins."""
    _seed_player(temp_db, player_id=71, name="Max Muncy", team="ATH", mlb_id=691777)
    _seed_player(temp_db, player_id=999, name="Max Muncy", team="LAD", mlb_id=571970)

    from src.live_stats import match_player_id

    # Yahoo says 'LAD' — must resolve to player_id=999, NOT 71.
    pid = match_player_id("Max Muncy", "LAD")
    assert pid == 999, (
        f"Roster sync DNA bug — Yahoo team='LAD' should pick LAD-Muncy "
        f"(player_id=999), not ATH-Muncy (player_id=71). Got {pid}."
    )


# ---------------------------------------------------------------------------
# Scenario 2: ONLY ATH Muncy in DB. Yahoo says editorial_team_abbr='LAD'.
# Current behavior: silently resolves to ATH Muncy. PR22 requirement:
# a WARNING must be logged so the DNA collision is visible.
# ---------------------------------------------------------------------------


def test_name_only_fallback_with_team_mismatch_logs_warning(temp_db, caplog):
    """When the (name + team) match yields nothing and the name-only
    fallback resolves to a player whose stored team DIFFERS from the
    Yahoo editorial_team_abbr, the function must log a WARNING that
    surfaces the collision. The match still returns (back-compat),
    but the warning makes silent DNA failures visible."""
    _seed_player(temp_db, player_id=71, name="Max Muncy", team="ATH", mlb_id=691777)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Max Muncy", "LAD")

    # Match still resolves (back-compat — the fallback path is preserved).
    assert pid == 71

    # But the operator must see a WARNING with the team mismatch.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    msgs = " ".join(r.getMessage() for r in warnings)
    assert any("Max Muncy" in m for m in (r.getMessage() for r in warnings)), (
        f"Expected WARNING mentioning 'Max Muncy' for DNA collision; got: {msgs!r}"
    )
    assert "LAD" in msgs and "ATH" in msgs, (
        f"Expected WARNING to surface both Yahoo team (LAD) and DB team (ATH); got: {msgs!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: (name + team) precise match — no warning, returns the right id.
# Regression guard: PR22's new warning branch must NOT fire on the happy path.
# ---------------------------------------------------------------------------


def test_team_match_does_not_warn(temp_db, caplog):
    """The happy-path (name + team) match must not log any warning —
    that path is the expected, correct behavior."""
    _seed_player(temp_db, player_id=42, name="Aaron Judge", team="NYY", mlb_id=592450)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Aaron Judge", "NYY")

    assert pid == 42
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, f"Happy-path match should not warn, got: {[r.getMessage() for r in warnings]}"


# ---------------------------------------------------------------------------
# Scenario 4: empty editorial_team_abbr — caller didn't pass a team.
# Falls through to name-only, no DNA collision check applies.
# ---------------------------------------------------------------------------


def test_empty_team_no_collision_warning(temp_db, caplog):
    """If the caller doesn't pass an editorial_team_abbr (legacy callers
    sometimes don't), the collision check can't run. The function must
    still resolve via name-only without spurious warnings."""
    _seed_player(temp_db, player_id=71, name="Max Muncy", team="ATH", mlb_id=691777)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Max Muncy", "")

    assert pid == 71
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    # No team to compare against → no collision check → no warning.
    assert not any("DNA" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# Scenario 5: name-only fallback resolves to a player whose team MATCHES
# the Yahoo team (case-insensitive). No warning.
# ---------------------------------------------------------------------------


def test_team_case_mismatch_does_not_warn(temp_db, caplog):
    """COLLATE NOCASE — if the (name + team) match would have succeeded
    case-insensitively, no warning fires. This is just the happy path
    with case noise."""
    _seed_player(temp_db, player_id=42, name="Aaron Judge", team="NYY", mlb_id=592450)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Aaron Judge", "nyy")  # lowercase Yahoo

    assert pid == 42
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, f"Case-insensitive team match should not warn: {[r.getMessage() for r in warnings]}"


# ---------------------------------------------------------------------------
# DB-C5 (2026-06-07): the trailing team-less last-name fallback
# (`LIKE '% lastname'`, no team filter) silently returned a match. When the
# caller DID pass team_abbr and the resolved player's team differs, it must
# warn too — that branch is the most collision-prone of all.
# ---------------------------------------------------------------------------


def test_lastname_only_fallback_team_mismatch_logs_warning(temp_db, caplog):
    """A caller passes team_abbr='LAD' and only a last name ('Trout'); the
    full-name and team-aware branches all miss, so the team-LESS
    `LIKE '% lastname'` branch resolves to LAA-Trout. Because the caller
    supplied a team that differs from the match, a WARNING must fire."""
    _seed_player(temp_db, player_id=27, name="Mike Trout", team="LAA", mlb_id=545361)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Trout", "LAD")

    # Match still resolves via the last-name fallback (back-compat).
    assert pid == 27
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    msgs = " ".join(r.getMessage() for r in warnings)
    assert "Trout" in msgs, f"Expected WARNING mentioning 'Trout'; got: {msgs!r}"
    assert "LAD" in msgs and "LAA" in msgs, (
        f"Expected WARNING to surface both caller team (LAD) and DB team (LAA); got: {msgs!r}"
    )


def test_lastname_only_fallback_no_team_does_not_warn(temp_db, caplog):
    """When the caller did NOT pass a team, the team-less last-name fallback
    can't detect a mismatch — it must resolve quietly (no spurious warning)."""
    _seed_player(temp_db, player_id=27, name="Mike Trout", team="LAA", mlb_id=545361)

    from src.live_stats import match_player_id

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        pid = match_player_id("Trout", "")

    assert pid == 27
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("DNA" in r.getMessage() for r in warnings), (
        f"No team passed → no collision check → no warning; got: {[r.getMessage() for r in warnings]}"
    )
