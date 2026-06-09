"""Tests for game-log opponent/result enrichment (Combustion dossier game log).

The player-dossier dialog's Game Log (``build_game_log_html`` + the row
assembly in ``_dossier_game_log_rows``) must show, per the approved mockup
``docs/design/mockup-player-popup.html`` (the ``.glog`` table), each game's
opponent (with team logo + "vs/@ ABBR") and a W/L result badge.

That requires the ``game_logs`` table to carry opponent + home/away + result
columns, ``_parse_game_log_row`` to extract them from the statsapi gameLog
split, and the renderer to surface them when present (graceful "—" when not).

These tests lock:
  - ``_parse_game_log_row`` extracts opponent id/abbr, home flag, and W/L from a
    representative statsapi gameLog split dict (both hitting + pitching groups);
  - missing opponent/result fields degrade gracefully (no crash, NULL/None);
  - the schema migration adds the new columns idempotently (``init_db`` twice);
  - ``build_game_log_html`` renders the opponent logo + W/L badge when a row has
    ``opponent_abbr``-derived ``opp`` + ``result``, and "—" when absent.
"""

from __future__ import annotations

from src.player_databank import _parse_game_log_row
from src.ui_shared import build_game_log_html, team_logo_url

# ── Representative statsapi gameLog split ``stat`` blocks ──────────────────────
# Real shape confirmed against /people/{id}?hydrate=stats(type=gameLog):
# each split = {date, isHome, isWin, team{id,name}, opponent{id,name}, stat{...}}.
# ``_parse_game_log_row`` is passed the whole split dict (``raw``); the stat line
# lives under split["stat"].

_HITTING_SPLIT = {
    "date": "2026-06-08",
    "isHome": True,
    "isWin": True,
    "team": {"id": 117, "name": "Houston Astros"},
    "opponent": {"id": 108, "name": "Los Angeles Angels"},
    "stat": {
        "plateAppearances": 5,
        "atBats": 4,
        "hits": 3,
        "runs": 2,
        "homeRuns": 1,
        "rbi": 4,
        "stolenBases": 0,
        "baseOnBalls": 1,
        "hitByPitch": 0,
        "sacFlies": 0,
    },
}

_PITCHING_SPLIT = {
    "date": "2026-06-07",
    "isHome": False,
    "isWin": False,
    "team": {"id": 143, "name": "Philadelphia Phillies"},
    "opponent": {"id": 136, "name": "Seattle Mariners"},
    "stat": {
        "inningsPitched": "6.1",
        "wins": 0,
        "losses": 1,
        "saves": 0,
        "strikeOuts": 7,
        "earnedRuns": 3,
        "baseOnBalls": 2,
        "hits": 6,
    },
}


def _call_parse(split: dict, group: str) -> dict:
    """Invoke _parse_game_log_row the way fetch_game_logs_from_api does."""
    return _parse_game_log_row(
        player_id=1,
        game_date=split["date"],
        season=2026,
        group=group,
        raw=split["stat"],
        split=split,
    )


# ── Parser: opponent / home / result extraction ───────────────────────────────


class TestParseGameLogOpponentResult:
    def test_hitting_extracts_opponent_id_and_abbr(self):
        row = _call_parse(_HITTING_SPLIT, "hitting")
        assert row["opponent_id"] == 108
        assert row["opponent_abbr"] == "LAA"

    def test_hitting_extracts_home_flag(self):
        row = _call_parse(_HITTING_SPLIT, "hitting")
        assert row["is_home"] == 1

    def test_hitting_derives_win_result(self):
        row = _call_parse(_HITTING_SPLIT, "hitting")
        assert row["result"] == "W"

    def test_hitting_preserves_stat_line(self):
        # New fields are additive — the existing stat parse must be untouched.
        row = _call_parse(_HITTING_SPLIT, "hitting")
        assert row["ab"] == 4
        assert row["h"] == 3
        assert row["hr"] == 1
        assert row["rbi"] == 4

    def test_pitching_extracts_opponent_abbr(self):
        row = _call_parse(_PITCHING_SPLIT, "pitching")
        assert row["opponent_id"] == 136
        assert row["opponent_abbr"] == "SEA"

    def test_pitching_away_flag(self):
        row = _call_parse(_PITCHING_SPLIT, "pitching")
        assert row["is_home"] == 0

    def test_pitching_derives_loss_result(self):
        row = _call_parse(_PITCHING_SPLIT, "pitching")
        assert row["result"] == "L"

    def test_pitching_preserves_ip_outs_notation(self):
        row = _call_parse(_PITCHING_SPLIT, "pitching")
        # 6.1 IP = 6 + 1/3 (outs notation), per _ip_outs_to_decimal.
        assert abs(row["ip"] - (6 + 1 / 3)) < 1e-6
        assert row["k"] == 7

    def test_scores_default_none_when_absent(self):
        # The statsapi gameLog split does NOT carry team/opp final scores, so
        # these columns are present (forward-compat) but None.
        row = _call_parse(_HITTING_SPLIT, "hitting")
        assert row["team_score"] is None
        assert row["opp_score"] is None


class TestParseGameLogGraceful:
    def test_no_split_kwarg_omits_enrichment(self):
        # Back-compat: legacy callers that don't pass ``split`` still work; the
        # enrichment fields default to None (rendered as "—").
        row = _parse_game_log_row(
            player_id=1,
            game_date="2026-06-08",
            season=2026,
            group="hitting",
            raw=_HITTING_SPLIT["stat"],
        )
        assert row["opponent_id"] is None
        assert row["opponent_abbr"] is None
        assert row["is_home"] is None
        assert row["result"] is None
        # Stat line still parsed.
        assert row["h"] == 3

    def test_missing_opponent_dict(self):
        split = dict(_HITTING_SPLIT)
        split.pop("opponent")
        row = _call_parse(split, "hitting")
        assert row["opponent_id"] is None
        assert row["opponent_abbr"] is None

    def test_unknown_opponent_id_yields_no_abbr(self):
        split = {**_HITTING_SPLIT, "opponent": {"id": 999999, "name": "Nobody"}}
        row = _call_parse(split, "hitting")
        # id is preserved but abbr can't be resolved → None (renders "—").
        assert row["opponent_id"] == 999999
        assert row["opponent_abbr"] is None

    def test_missing_iswin_yields_none_result(self):
        split = dict(_HITTING_SPLIT)
        split.pop("isWin")
        row = _call_parse(split, "hitting")
        assert row["result"] is None

    def test_missing_ishome_yields_none(self):
        split = dict(_HITTING_SPLIT)
        split.pop("isHome")
        row = _call_parse(split, "hitting")
        assert row["is_home"] is None


# ── Schema migration idempotency ──────────────────────────────────────────────


class TestSchemaMigration:
    def test_init_db_twice_is_idempotent(self):
        from src.database import init_db

        # Must not raise on the second call (duplicate-column ALTERs swallowed).
        init_db()
        init_db()

    def test_new_columns_present_after_init(self):
        from src.database import get_connection, init_db

        init_db()
        conn = get_connection()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(game_logs)").fetchall()}
        finally:
            conn.close()
        for col in ("opponent_id", "opponent_abbr", "is_home", "result", "team_score", "opp_score"):
            assert col in cols, f"game_logs missing enrichment column {col!r}"


# ── Renderer: opponent logo + W/L badge from populated columns ─────────────────


class TestRenderFromColumns:
    def test_renders_opponent_logo_and_win_badge(self):
        rows = [
            {
                "date": "Jun 08",
                "opp": "LAA",
                "home": True,
                "result": "W",
                "score": "6-3",
                "ab": 4,
                "h": 3,
                "hr": 1,
                "rbi": 4,
                "avg": 0.750,
                "form_pct": 95,
            }
        ]
        html = build_game_log_html(rows, is_hitter=True)
        assert team_logo_url("LAA") in html
        assert "vs LAA" in html
        assert "res w" in html
        assert "W 6-3" in html

    def test_renders_loss_badge_away(self):
        rows = [
            {
                "date": "Jun 07",
                "opp": "SEA",
                "home": False,
                "result": "L",
                "score": "2-5",
                "ip": 6.0,
                "h_allowed": 6,
                "er": 3,
                "k": 7,
                "era": 4.50,
                "form_pct": 40,
            }
        ]
        html = build_game_log_html(rows, is_hitter=False)
        assert team_logo_url("SEA") in html
        assert "@ SEA" in html
        assert "res l" in html

    def test_null_columns_render_dash_no_crash(self):
        # Old rows (not yet re-fetched) have no opp/result → graceful "—".
        rows = [
            {
                "date": "Jun 08",
                "opp": None,
                "result": None,
                "ab": 4,
                "h": 2,
                "hr": 0,
                "rbi": 1,
                "avg": 0.5,
                "form_pct": 50,
            }
        ]
        html = build_game_log_html(rows, is_hitter=True)
        assert "tlogo" not in html
        assert "res w" not in html and "res l" not in html
        assert "—" in html
