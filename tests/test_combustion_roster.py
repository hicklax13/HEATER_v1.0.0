"""Tests for the Combustion ``build_roster_table_html`` roster renderer (Phase C1).

Guards the branded ``.rtbl`` Active Roster table the My Team page renders:
hitter vs pitcher column sets, headshot + team logo presence, the per-row
``--tc`` team-color custom property, status dots, no-emoji, and empty-df safety.
"""

import re

import pandas as pd

from src.ui_shared import build_roster_table_html

# Emoji / pictograph ranges — the whole app is SVG-only, no emoji allowed.
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff←-⇿⌀-⏿]")


def _hitter_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Yordan Alvarez",
                "positions": "DH",
                "team": "HOU",
                "mlb_id": 670541,
                "status": "active",
                "ab": 312,
                "r": 58,
                "h": 99,
                "hr": 28,
                "rbi": 82,
                "sb": 1,
                "avg": 0.317,
                "obp": 0.402,
            },
            {
                "name": "Aaron Judge",
                "positions": "RF",
                "team": "NYY",
                "mlb_id": 592450,
                "status": "active",
                "ab": 340,
                "r": 72,
                "h": 110,
                "hr": 34,
                "rbi": 79,
                "sb": 6,
                "avg": 0.324,
                "obp": 0.448,
            },
        ]
    )


def _pitcher_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Tarik Skubal",
                "positions": "SP",
                "team": "DET",
                "mlb_id": 669373,
                "status": "active",
                "ip": 120.1,
                "w": 11,
                "l": 3,
                "sv": 0,
                "k": 158,
                "era": 2.41,
                "whip": 0.92,
            },
            {
                "name": "Mason Miller",
                "positions": "RP",
                "team": "ATH",
                "mlb_id": 682243,
                "status": "DTD",
                "ip": 38.0,
                "w": 2,
                "l": 4,
                "sv": 18,
                "k": 64,
                "era": 3.10,
                "whip": 1.05,
            },
        ]
    )


def test_hitter_columns_present_and_pitcher_absent():
    html = build_roster_table_html(_hitter_df(), is_hitter=True)
    # Hitter header set
    for col in ("Player", "Pos", "AB", "R", "H", "HR", "RBI", "SB", "AVG", "OBP"):
        assert f">{col}<" in html, f"missing hitter column header {col!r}"
    # Pitcher-only headers must NOT appear as standalone column headers
    assert ">IP<" not in html
    assert ">SV<" not in html
    assert ">WHIP<" not in html


def test_pitcher_columns_present_and_hitter_absent():
    html = build_roster_table_html(_pitcher_df(), is_hitter=False)
    for col in ("Player", "Pos", "IP", "W", "L", "SV", "K", "ERA", "WHIP"):
        assert f">{col}<" in html, f"missing pitcher column header {col!r}"
    # Hitter-only headers must NOT appear
    assert ">AB<" not in html
    assert ">RBI<" not in html
    assert ">OBP<" not in html


def test_headshot_and_team_logo_present():
    html = build_roster_table_html(_hitter_df(), is_hitter=True)
    # Headshot <img> from _headshot_img_html (MLB midfield headshot URL)
    assert "670541" in html
    assert "midfield.mlbstatic.com" in html or "people/670541" in html
    # Team logo for HOU (id 117) and NYY (id 147)
    assert "team-logos/117.svg" in html
    assert "team-logos/147.svg" in html


def test_per_row_team_color_custom_property():
    html = build_roster_table_html(_hitter_df(), is_hitter=True)
    # HOU primary #002D62, NYY primary #0C2340 — emitted as --tc on each row.
    assert "--tc:#002D62" in html
    assert "--tc:#0C2340" in html
    # One <tr ...> per data row.
    assert html.count("<tr") >= 3  # header + 2 data rows


def test_status_dot_classes():
    html = build_roster_table_html(_pitcher_df(), is_hitter=False)
    # active → ok, DTD → dtd
    assert "sdot ok" in html
    assert "sdot dtd" in html


def test_il_status_dot():
    df = _hitter_df()
    df.loc[0, "status"] = "IL15"
    html = build_roster_table_html(df, is_hitter=True)
    assert "sdot il" in html


def test_no_emoji():
    html = build_roster_table_html(_hitter_df(), is_hitter=True)
    found = _EMOJI_RE.findall(html)
    assert not found, f"emoji/pictograph chars found in roster HTML: {found!r}"


def test_player_name_html_escaped():
    df = _hitter_df()
    df.loc[0, "name"] = "O'Neill <Test> & Co"
    html = build_roster_table_html(df, is_hitter=True)
    assert "&lt;Test&gt;" in html
    assert "&amp;" in html
    # raw unescaped angle bracket from the name must not leak
    assert "<Test>" not in html


def test_empty_df_safe():
    html = build_roster_table_html(pd.DataFrame(), is_hitter=True)
    assert isinstance(html, str)
    # Still renders a table shell with the hitter headers, no crash.
    assert "Player" in html


def test_player_ids_wire_click_links():
    df = _hitter_df()
    html = build_roster_table_html(df, is_hitter=True, player_ids=[101, 202])
    # When player_ids supplied, each row links to ?player=<id> so a click opens
    # the existing player-card dialog.
    assert "player=101" in html
    assert "player=202" in html


def test_avg_obp_formatting_no_leading_zero():
    html = build_roster_table_html(_hitter_df(), is_hitter=True)
    # AVG/OBP rendered via format_stat → leading zero stripped (.317, .402)
    assert ".317" in html
    assert ".402" in html
    assert "0.317" not in html


def test_unknown_team_falls_back_to_orange():
    df = _hitter_df()
    df.loc[0, "team"] = None
    df.loc[1, "team"] = "ZZZ"  # not a real abbr
    html = build_roster_table_html(df, is_hitter=True)
    # Fallback team color is orange #ff6d00 for unresolved teams.
    assert "--tc:#ff6d00" in html
