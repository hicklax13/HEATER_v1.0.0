"""Test BUG-012 fix: Bayesian batch_update produces self-consistent rate/counting stats."""

import pandas as pd
import pytest


@pytest.fixture
def bayesian_updater():
    from src.bayesian import BayesianUpdater

    return BayesianUpdater()


def test_hitter_h_div_ab_equals_avg(bayesian_updater):
    """After batch_update, updated["h"] / updated["ab"] must equal updated["avg"]
    (within rounding tolerance). The prior bug computed h from formula that
    mixed observed h with blended avg × remaining_ab, producing inconsistency.

    Note: batch_update_projections merges preseason+season_stats with
    suffixes=('_pre', '_obs'). Both fixtures use raw column names; the
    merge applies the suffixes automatically.
    """
    preseason = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Test Hitter",
                "is_hitter": True,
                "system": "blended",
                "pa": 600,
                "ab": 550,
                "h": 165,
                "avg": 0.300,
                "obp": 0.370,
                "r": 90,
                "hr": 25,
                "rbi": 80,
                "sb": 10,
                "bb": 45,
                "hbp": 5,
                "sf": 5,
            }
        ]
    )
    season_stats = pd.DataFrame(
        [
            {
                "player_id": 1,
                "pa": 220,
                "ab": 200,
                "h": 70,
                "avg": 0.350,
                "obp": 0.420,
                "r": 35,
                "hr": 12,
                "rbi": 35,
                "sb": 4,
            }
        ]
    )
    result = bayesian_updater.batch_update_projections(season_stats, preseason)
    assert len(result) == 1
    row = result.iloc[0]
    derived_h = row["avg"] * row["ab"]
    assert abs(row["h"] - derived_h) <= 1.5, (
        f"BUG-012: h/ab inconsistent with avg. "
        f"avg={row['avg']:.4f}, ab={row['ab']}, h={row['h']}, derived_h={derived_h:.2f}"
    )


def test_pitcher_bb_h_split_uses_observed_ratio(bayesian_updater):
    """For a control pitcher (low BB rate), Bayesian update must NOT impose
    the hardcoded 35/65 BB/H split."""
    preseason = pd.DataFrame(
        [
            {
                "player_id": 100,
                "name": "Control SP",
                "is_hitter": False,
                "system": "blended",
                "ip": 200.0,
                "era": 3.50,
                "whip": 1.10,
                "w": 14,
                "l": 8,
                "sv": 0,
                "k": 180,
                "er": 78,
                "bb_allowed": 40,
                "h_allowed": 180,
            }
        ]
    )
    season_stats = pd.DataFrame(
        [
            {
                "player_id": 100,
                "ip": 100.0,
                "era": 3.0,
                "whip": 1.0,
                "w": 8,
                "l": 3,
                "sv": 0,
                "k": 95,
                "er": 33,
                "bb_allowed": 15,
                "h_allowed": 85,
            }
        ]
    )
    result = bayesian_updater.batch_update_projections(season_stats, preseason)
    row = result.iloc[0]
    total_br = row["bb_allowed"] + row["h_allowed"]
    assert total_br > 0
    bb_share = row["bb_allowed"] / total_br
    assert bb_share < 0.30, (
        f"BUG-012: pitcher bb_share = {bb_share:.3f} appears to use hardcoded "
        f"0.35 ratio, ignoring observed bb:h = 15:85 = 0.15"
    )


def test_pitcher_er_consistent_with_era_and_ip(bayesian_updater):
    """updated["er"] should equal int(era * ip / 9)."""
    preseason = pd.DataFrame(
        [
            {
                "player_id": 200,
                "name": "Test SP",
                "is_hitter": False,
                "system": "blended",
                "ip": 150.0,
                "era": 4.00,
                "whip": 1.20,
                "w": 10,
                "l": 8,
                "sv": 0,
                "k": 130,
                "er": 67,
                "bb_allowed": 50,
                "h_allowed": 130,
            }
        ]
    )
    season_stats = pd.DataFrame(
        [
            {
                "player_id": 200,
                "ip": 80.0,
                "era": 3.50,
                "whip": 1.15,
                "w": 6,
                "l": 4,
                "sv": 0,
                "k": 75,
                "er": 31,
                "bb_allowed": 24,
                "h_allowed": 70,
            }
        ]
    )
    result = bayesian_updater.batch_update_projections(season_stats, preseason)
    row = result.iloc[0]
    expected_er = int(row["era"] * row["ip"] / 9)
    assert abs(row["er"] - expected_er) <= 1, f"BUG-012: er = {row['er']} inconsistent with era*ip/9 = {expected_er}"
