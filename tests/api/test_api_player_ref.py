"""Unit tests for the shared PlayerRef builder + team-id mapping (M0 slice 1)."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.services.player_ref import make_player_ref, team_id_for


def test_team_id_for_known_abbr():
    assert team_id_for("NYY") == 147
    assert team_id_for("LAD") == 119
    assert team_id_for("ATH") == 133  # codebase-canonical (not "OAK")


def test_team_id_for_is_case_and_whitespace_insensitive():
    assert team_id_for(" nyy ") == 147


def test_team_id_for_unknown_or_blank_returns_none():
    assert team_id_for("ZZZ") is None
    assert team_id_for("") is None
    assert team_id_for(None) is None


def test_make_player_ref_populates_enrichment():
    ref = make_player_ref(id=42, name="Aaron Judge", positions="OF", mlb_id=592450, team_abbr="NYY")
    assert isinstance(ref, PlayerRef)
    assert ref.id == 42
    assert ref.mlb_id == 592450
    assert ref.team_abbr == "NYY"
    assert ref.team_id == 147
    assert ref.positions == "OF"


def test_make_player_ref_normalizes_missing_data():
    ref = make_player_ref(id=0, name="x", positions="", mlb_id=0, team_abbr="  ")
    assert ref.mlb_id is None  # 0 -> None (not a real MLB id)
    assert ref.team_abbr is None  # blank -> None
    assert ref.team_id is None


def test_make_player_ref_handles_float_and_nan_mlb_id():
    assert make_player_ref(id=1, name="x", positions="", mlb_id=592450.0).mlb_id == 592450
    assert make_player_ref(id=1, name="x", positions="", mlb_id=float("nan")).mlb_id is None


def test_make_player_ref_unknown_team_keeps_abbr_but_null_id():
    ref = make_player_ref(id=1, name="x", positions="", team_abbr="ZZZ")
    assert ref.team_abbr == "ZZZ"
    assert ref.team_id is None


def test_make_player_ref_defaults_all_enrichment_to_none():
    ref = make_player_ref(id=3, name="y", positions="SP")
    assert ref.mlb_id is None
    assert ref.team_abbr is None
    assert ref.team_id is None
    assert ref.yahoo_player_key is None


def test_make_player_ref_nan_team_abbr_is_none():
    import pandas as pd

    assert make_player_ref(id=1, name="x", positions="", team_abbr=float("nan")).team_abbr is None
    assert make_player_ref(id=1, name="x", positions="", team_abbr=float("nan")).team_id is None
    assert make_player_ref(id=2, name="y", positions="", team_abbr=pd.NA).team_abbr is None
    assert make_player_ref(id=2, name="y", positions="", team_abbr=pd.NA).team_id is None
