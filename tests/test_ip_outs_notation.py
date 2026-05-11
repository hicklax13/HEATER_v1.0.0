"""Test BUG-004 fix: IP outs notation correctly converted to decimal IP."""

import pytest


def test_ip_outs_notation_conversion():
    """MLB API IP format: '52.2' means 52 + 2/3 IP (NOT 52.2)."""
    from src.live_stats import _ip_outs_to_decimal

    assert _ip_outs_to_decimal("52.0") == pytest.approx(52.0, abs=1e-6)
    assert _ip_outs_to_decimal("52.1") == pytest.approx(52 + 1 / 3, abs=1e-6)
    assert _ip_outs_to_decimal("52.2") == pytest.approx(52 + 2 / 3, abs=1e-6)
    assert _ip_outs_to_decimal("0.0") == 0.0
    assert _ip_outs_to_decimal("0.1") == pytest.approx(1 / 3, abs=1e-6)
    assert _ip_outs_to_decimal(6.0) == pytest.approx(6.0, abs=1e-6)
    assert _ip_outs_to_decimal("") == 0.0
    assert _ip_outs_to_decimal(None) == 0.0
    assert _ip_outs_to_decimal("abc") == 0.0


def test_parse_pitching_stat_uses_outs_notation():
    """_parse_pitching_stat must use _ip_outs_to_decimal for IP."""
    from src.live_stats import _parse_pitching_stat

    player_info = {"fullName": "Test", "team_abbr": "ATL", "mlb_id": 1}
    stat = {
        "inningsPitched": "10.2",
        "earnedRuns": 4,
        "wins": 1,
        "losses": 0,
        "saves": 0,
        "strikeOuts": 12,
        "era": "3.38",
        "whip": "1.18",
        "baseOnBalls": 3,
        "hits": 9,
        "gamesPlayed": 2,
    }
    row = _parse_pitching_stat(player_info, stat)
    assert row["ip"] == pytest.approx(10 + 2 / 3, abs=1e-6), f"BUG-004: expected ip = 10.667, got {row['ip']}"


def test_player_databank_parse_game_log_uses_outs_notation():
    """player_databank._parse_game_log_row should convert IP outs notation."""
    from src.player_databank import _parse_game_log_row

    raw = {
        "inningsPitched": "6.1",
        "wins": 0,
        "losses": 0,
        "saves": 0,
        "strikeOuts": 7,
        "earnedRuns": 2,
        "baseOnBalls": 1,
        "hits": 4,
    }
    row = _parse_game_log_row(
        player_id=1,
        game_date="2026-05-10",
        season=2026,
        group="pitching",
        raw=raw,
    )
    assert row["ip"] == pytest.approx(6 + 1 / 3, abs=1e-6), f"BUG-004: expected ip = 6.333, got {row['ip']}"
