"""Tests for the Combustion player-dossier pure HTML builders (src/ui_shared.py).

The dossier dialog (``show_player_card_dialog``) is rebuilt to match the
gold-standard mockup ``docs/design/mockup-player-popup.html``. The visual
sections are composed by three PURE builders so they are unit-testable
without Streamlit:

  - ``build_dossier_header_html`` — team-color-tinted header band
  - ``build_game_log_html``       — "Game Log — Last 10" table
  - ``build_upcoming_cards_html`` — "Upcoming · Projections" cards

These tests lock the load-bearing visual contracts: correct columns per
player type, W/L badge classes, opponent-logo URLs, the team-color header
background, graceful empty states, and the no-emoji rule.
"""

from __future__ import annotations

import re

from src.ui_shared import (
    build_dossier_header_html,
    build_game_log_html,
    build_upcoming_cards_html,
    team_color,
    team_logo_url,
    text_on,
)

# ── Fixtures (plain dicts — the builders take pre-shaped data) ─────────────────

_HITTER_PROFILE = {
    "name": "Yordan Alvarez",
    "team": "HOU",
    "positions": "DH",
    "bats": "L",
    "jersey": 44,
    "headshot_url": "https://midfield.mlbstatic.com/v1/people/670541/spots/120",
}

_PITCHER_PROFILE = {
    "name": "Zack Wheeler",
    "team": "PHI",
    "positions": "SP",
    "throws": "R",
    "headshot_url": "",
}

_HITTER_GLOG = [
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
        "hot": True,
    },
    {
        "date": "Jun 06",
        "opp": "SEA",
        "home": False,
        "result": "L",
        "score": "2-5",
        "ab": 4,
        "h": 1,
        "hr": 0,
        "rbi": 1,
        "avg": 0.250,
        "form_pct": 38,
        "hot": False,
    },
]

_PITCHER_GLOG = [
    {
        "date": "Jun 07",
        "opp": "ATL",
        "home": True,
        "result": "W",
        "score": "5-2",
        "ip": 6.0,
        "h_allowed": 4,
        "er": 2,
        "k": 8,
        "era": 3.00,
        "form_pct": 80,
        "hot": True,
    }
]

_HITTER_UPCOMING = [
    {
        "date": "Jun 10",
        "opp": "OAK",
        "home": False,
        "pitcher": "vs L. Severino (R)",
        "proj": {"h": 1.3, "hr": 0.5, "rbi": 1.4, "r": 0.9},
        "conf_pct": 78,
        "conf_note": "favorable park",
    }
]

_PITCHER_UPCOMING = [
    {
        "date": "Jun 12",
        "opp": "NYM",
        "home": True,
        "pitcher": "",
        "proj": {"k": 7.5, "era": 3.10, "ip": 6.1, "w": 0.6},
        "conf_pct": 65,
        "conf_note": "home",
    }
]

# Disallow emoji / pictographs in rendered HTML.
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f0ff\U00002190-\U000021ff\U00002b00-\U00002bff]"
)


def _assert_no_emoji(html: str) -> None:
    found = _EMOJI_RE.findall(html)
    assert not found, f"emoji found in HTML: {found!r}"


# ── Header band ───────────────────────────────────────────────────────────────


class TestDossierHeader:
    def test_header_bg_uses_team_color(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        assert team_color("HOU") in html

    def test_header_uses_readable_ink(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        # text_on returns the readable text color for the team bg — it must appear.
        assert text_on(team_color("HOU")) in html

    def test_header_shows_name(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        assert "Yordan Alvarez" in html

    def test_header_eyebrow_has_roster_pos_jersey(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        assert "ROSTER" in html
        assert "DH" in html
        assert "#44" in html

    def test_header_includes_team_logo(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        assert team_logo_url("HOU") in html

    def test_header_renders_season_chips(self):
        chips = [
            {"label": "AVG", "value": ".317", "accent": True},
            {"label": "HR", "value": "28"},
        ]
        html = build_dossier_header_html(_HITTER_PROFILE, chips)
        assert "AVG" in html and ".317" in html
        assert "HR" in html and "28" in html

    def test_header_escapes_player_name(self):
        profile = dict(_HITTER_PROFILE, name="<script>x</script>")
        html = build_dossier_header_html(profile, [])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_header_empty_chips_does_not_raise(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [])
        assert "mhead" in html

    def test_header_unknown_team_falls_back(self):
        profile = dict(_HITTER_PROFILE, team="???")
        # Orange fallback color; must not raise.
        html = build_dossier_header_html(profile, [])
        assert "mhead" in html

    def test_header_no_emoji(self):
        html = build_dossier_header_html(_HITTER_PROFILE, [{"label": "AVG", "value": ".317"}])
        _assert_no_emoji(html)


# ── Game log table ────────────────────────────────────────────────────────────


class TestGameLog:
    def test_hitter_columns(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        for col in ("Date", "Opp", "Res", "AB", "H", "HR", "RBI", "AVG", "Form"):
            assert f">{col}<" in html, f"missing hitter column header {col}"

    def test_pitcher_columns(self):
        html = build_game_log_html(_PITCHER_GLOG, is_hitter=False)
        for col in ("Date", "Opp", "Res", "IP", "ER", "K", "ERA", "Form"):
            assert f">{col}<" in html, f"missing pitcher column header {col}"

    def test_hitter_has_no_pitcher_columns(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert ">IP<" not in html
        assert ">ERA<" not in html

    def test_win_badge_class(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert "res w" in html

    def test_loss_badge_class(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert "res l" in html

    def test_result_score_rendered(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert "W 6-3" in html
        assert "L 2-5" in html

    def test_opponent_logo_url_present(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert team_logo_url("LAA") in html
        assert team_logo_url("SEA") in html

    def test_home_away_prefix(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert "vs LAA" in html  # home game
        assert "@ SEA" in html  # away game

    def test_missing_opponent_renders_dash_not_logo(self):
        rows = [{"date": "Jun 08", "ab": 4, "h": 2, "hr": 0, "rbi": 1, "avg": 0.5, "form_pct": 50}]
        html = build_game_log_html(rows, is_hitter=True)
        # No opp key → no team-logo <img> in the opp cell, graceful dash.
        assert "tlogo" not in html
        assert "—" in html

    def test_missing_result_renders_dash(self):
        rows = [{"date": "Jun 08", "ab": 4, "h": 2, "hr": 0, "rbi": 1, "avg": 0.5, "form_pct": 50}]
        html = build_game_log_html(rows, is_hitter=True)
        assert "res w" not in html and "res l" not in html

    def test_hot_row_class(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert 'class="hot"' in html

    def test_avg_formatted_three_dp(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert "0.750" in html

    def test_era_formatted_two_dp(self):
        html = build_game_log_html(_PITCHER_GLOG, is_hitter=False)
        assert "3.00" in html

    def test_form_bar_present(self):
        html = build_game_log_html(_HITTER_GLOG, is_hitter=True)
        assert 'class="bar"' in html

    def test_empty_rows_do_not_raise(self):
        html = build_game_log_html([], is_hitter=True)
        assert "No game log" in html

    def test_empty_pitcher_rows_do_not_raise(self):
        html = build_game_log_html([], is_hitter=False)
        assert "No game log" in html

    def test_no_emoji(self):
        _assert_no_emoji(build_game_log_html(_HITTER_GLOG, is_hitter=True))
        _assert_no_emoji(build_game_log_html(_PITCHER_GLOG, is_hitter=False))

    def test_bad_form_pct_does_not_raise(self):
        rows = [{"date": "x", "ab": 1, "h": 0, "hr": 0, "rbi": 0, "avg": 0.0, "form_pct": "bad"}]
        html = build_game_log_html(rows, is_hitter=True)
        assert "glog" in html


# ── Upcoming / projection cards ───────────────────────────────────────────────


class TestUpcomingCards:
    def test_hitter_projection_fields(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        for field in ("H", "HR", "RBI", "R"):
            assert f">{field}<" in html, f"missing hitter proj field {field}"

    def test_pitcher_projection_fields(self):
        html = build_upcoming_cards_html(_PITCHER_UPCOMING, is_hitter=False)
        for field in ("K", "ERA", "IP", "W"):
            assert f">{field}<" in html, f"missing pitcher proj field {field}"

    def test_opponent_logo_url_present(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        assert team_logo_url("OAK") in html

    def test_home_away_prefix(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        assert "@ OAK" in html

    def test_confidence_label_rendered(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        assert "78%" in html
        assert "favorable park" in html

    def test_pitcher_label_rendered(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        assert "L. Severino" in html

    def test_missing_opponent_renders_no_logo(self):
        games = [{"date": "Jun 10", "proj": {"h": 1.0}, "conf_pct": 50}]
        html = build_upcoming_cards_html(games, is_hitter=True)
        assert "tlogo" not in html

    def test_projection_value_formatting(self):
        html = build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True)
        assert "1.3" in html  # H projection (1dp)

    def test_pitcher_era_two_dp(self):
        html = build_upcoming_cards_html(_PITCHER_UPCOMING, is_hitter=False)
        assert "3.10" in html

    def test_empty_games_do_not_raise(self):
        html = build_upcoming_cards_html([], is_hitter=True)
        assert "No upcoming games" in html

    def test_empty_pitcher_games_do_not_raise(self):
        html = build_upcoming_cards_html([], is_hitter=False)
        assert "No upcoming games" in html

    def test_no_emoji(self):
        _assert_no_emoji(build_upcoming_cards_html(_HITTER_UPCOMING, is_hitter=True))
        _assert_no_emoji(build_upcoming_cards_html(_PITCHER_UPCOMING, is_hitter=False))

    def test_bad_conf_pct_does_not_raise(self):
        games = [{"date": "Jun 10", "opp": "OAK", "proj": {"h": 1.0}, "conf_pct": None}]
        html = build_upcoming_cards_html(games, is_hitter=True)
        assert "upc" in html
