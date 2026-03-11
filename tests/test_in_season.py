"""Test in-season analysis: trade analyzer, player comparison, FA ranker."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    players = [
        (1, "Aaron Judge", "NYY", "OF", 1),
        (2, "Shohei Ohtani", "LAD", "DH", 1),
        (3, "Trea Turner", "PHI", "SS", 1),
        (4, "Gerrit Cole", "NYY", "SP", 0),
        (5, "Corbin Burnes", "BAL", "SP", 0),
        (6, "Bobby Witt Jr", "KC", "SS", 1),
    ]
    for p in players:
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (?,?,?,?,?)",
            p,
        )
    hitter_projs = [
        (1, "blended", 600, 550, 160, 100, 42, 110, 8, 0.291),
        (2, "blended", 650, 590, 175, 110, 45, 115, 15, 0.297),
        (3, "blended", 580, 530, 155, 85, 18, 70, 30, 0.292),
        (6, "blended", 620, 570, 170, 95, 28, 90, 35, 0.298),
    ]
    pitcher_projs = [
        (4, "blended", 200, 15, 0, 220, 2.80, 1.05, 62, 50, 160),
        (5, "blended", 190, 13, 0, 200, 3.10, 1.10, 65, 55, 155),
    ]
    for p in hitter_projs:
        conn.execute(
            "INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg) VALUES (?,?,?,?,?,?,?,?,?,?)",
            p,
        )
    for p in pitcher_projs:
        conn.execute(
            "INSERT INTO projections "
            "(player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            p,
        )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _make_player_pool():
    from src.database import load_player_pool

    pool = load_player_pool()
    pool = pool.rename(columns={"name": "player_name"})
    return pool


def test_analyze_trade_returns_result(temp_db):
    from src.in_season import analyze_trade

    config = LeagueConfig()
    pool = _make_player_pool()

    result = analyze_trade(
        giving_ids=[1],
        receiving_ids=[4],
        user_roster_ids=[1, 2, 3],
        player_pool=pool,
        config=config,
    )
    assert result is not None
    assert "verdict" in result
    assert "category_impact" in result
    assert isinstance(result["category_impact"], dict)


def test_analyze_trade_category_impact(temp_db):
    from src.in_season import analyze_trade

    config = LeagueConfig()
    pool = _make_player_pool()

    result = analyze_trade(
        giving_ids=[1],
        receiving_ids=[4],
        user_roster_ids=[1, 2, 3],
        player_pool=pool,
        config=config,
    )
    # Trading away Judge (42 HR hitter) should hurt HR
    assert result["category_impact"]["HR"] < 0


def test_compare_players(temp_db):
    from src.in_season import compare_players

    config = LeagueConfig()
    pool = _make_player_pool()

    result = compare_players(1, 2, pool, config)
    assert "player_a" in result
    assert "player_b" in result
    assert "z_scores_a" in result
    assert "z_scores_b" in result
    assert "composite_a" in result
    assert "composite_b" in result


def test_rank_free_agents(temp_db):
    from src.in_season import rank_free_agents

    config = LeagueConfig()
    pool = _make_player_pool()

    user_roster_ids = [1, 2, 3]
    fa_pool = pool[pool["player_id"].isin([6])]

    ranked = rank_free_agents(user_roster_ids, fa_pool, pool, config)
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) >= 1
    assert "marginal_value" in ranked.columns
