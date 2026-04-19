"""Tests for team code canonicalization."""

import pytest

from src.valuation import TEAM_CODE_CANONICAL, canonicalize_team


def test_ari_az_equivalence():
    assert canonicalize_team("ARI") == "ARI"
    assert canonicalize_team("AZ") == "ARI"


def test_ath_oak_equivalence():
    assert canonicalize_team("ATH") == "ATH"
    assert canonicalize_team("OAK") == "ATH"


def test_wsh_variants():
    assert canonicalize_team("WSH") == "WSH"
    assert canonicalize_team("WSN") == "WSH"
    assert canonicalize_team("WAS") == "WSH"


def test_sf_sfg():
    assert canonicalize_team("SF") == "SF"
    assert canonicalize_team("SFG") == "SF"


def test_sd_sdp():
    assert canonicalize_team("SD") == "SD"
    assert canonicalize_team("SDP") == "SD"


def test_tb_tbr():
    assert canonicalize_team("TB") == "TB"
    assert canonicalize_team("TBR") == "TB"


def test_kc_kcr():
    assert canonicalize_team("KC") == "KC"
    assert canonicalize_team("KCR") == "KC"


def test_cws_chw():
    assert canonicalize_team("CWS") == "CWS"
    assert canonicalize_team("CHW") == "CWS"


def test_all_30_canonical_codes_present():
    canonical_codes = {
        "ARI",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CWS",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "ATH",
        "PHI",
        "PIT",
        "SD",
        "SF",
        "SEA",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WSH",
    }
    for code in canonical_codes:
        assert canonicalize_team(code) == code, f"{code} should map to itself"


def test_case_insensitive():
    assert canonicalize_team("ari") == "ARI"
    assert canonicalize_team("Nyy") == "NYY"


def test_whitespace_stripped():
    assert canonicalize_team("  LAD  ") == "LAD"


def test_unknown_code_passthrough():
    assert canonicalize_team("XYZ") == "XYZ"


def test_league_config_roster_slots():
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    assert config.roster_slots["BN"] == 6
    assert config.roster_slots["IL"] == 4
    total = sum(config.roster_slots.values())
    assert total == 28


def test_league_config_weekly_transaction_limit():
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    assert config.weekly_transaction_limit == 10
