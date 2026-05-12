# tests/test_wave8a_pool_mask.py
"""Wave 8a / Group 5: pool / mask / shape bugs.

Audit IDs covered:
  - D6A-002, D6A-003 + D6B-011..013 (`src/draft_engine.py:~740,779,
    811,820,954`) — `pool.get("is_hitter", True)` defaults to scalar
    ``True`` when the column is missing. Combined with ``== False``,
    this turns the boolean *mask* into a scalar ``False`` that
    broadcasts to every row, silently SKIPPING the pitcher-targeted
    enhancement stages (fip_era_adj, streaming_penalty, spring training
    K-signal) and CRASHING the lineup-protection bonus stage on the
    ``drafted_pool.loc[True]`` call. The row-level ``row.get(
    "is_hitter", True)`` paths inside ``.apply`` default each pitcher's
    is_hitter to True → ALL players get the lineup-protection bonus.

  - D6B-017 (`src/draft_state.py:248-264`) — Per-pick aggregation uses
    ``int(p.get("ab", 0) or 0)`` which CRASHES on NaN (which is truthy
    so the ``or 0`` fallback never fires, then ``int(NaN)`` raises
    ``ValueError``).  Fix: skip players whose OBP-denom inputs are NaN
    using ``pd.isna`` checks rather than fillna(0)-ing them into a fake
    "0 plate appearances" record.

  - D3A-003 (`src/optimizer/projections.py:154-164`) — K3 consistency
    block CRASHES with ``AttributeError: 'int' object has no attribute
    'fillna'`` when only one of {xwoba_delta, babip_delta} is present
    in the DataFrame. ``df.get(missing_col, 0)`` returns the scalar
    default ``0`` (not a Series), and ``pd.to_numeric(0, ...)`` returns
    a scalar int with no ``.fillna``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# D6A-002, D6A-003 + D6B-011..013 — draft_engine.py is_hitter mask
# ---------------------------------------------------------------------------


def _make_minimal_pool_without_is_hitter() -> pd.DataFrame:
    """A pool that lacks the is_hitter column — simulates the data-bug surface."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "name": ["Hitter1", "Hitter2", "Pitcher1", "Pitcher2"],
            "team": ["NYY", "BOS", "NYY", "LAD"],
            "positions": ["OF", "1B", "SP", "RP"],
            "fip": [np.nan, np.nan, 3.50, 2.80],
            "era": [np.nan, np.nan, 4.00, 3.20],
            "ip": [0.0, 0.0, 200.0, 70.0],
            "k": [0, 0, 200, 80],
            "sv": [0, 0, 0, 30],
            "r": [80, 70, 0, 0],
            "hr": [25, 20, 0, 0],
            "rbi": [85, 75, 0, 0],
            "sb": [10, 5, 0, 0],
            "w": [0, 0, 12, 4],
            "l": [0, 0, 8, 2],
            "streaming_penalty": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_d6a002_fip_correction_with_missing_is_hitter_col(caplog):
    """Without is_hitter column, _apply_fip_correction must NOT silently
    skip pitcher correction. Either:
      (a) it logs a warning AND infers from positions, or
      (b) it raises (acceptable — data bug worth surfacing).

    Prior bug: `pool.get("is_hitter", True) == False` is scalar False
    when the column is missing → mask broadcasts to False → no row matches.
    """
    from src.draft_engine import DraftRecommendationEngine
    from src.valuation import LeagueConfig

    engine = DraftRecommendationEngine(config=LeagueConfig(), mode="quick")
    pool = _make_minimal_pool_without_is_hitter()
    pool["fip_era_adj"] = 0.0  # column may not exist

    with caplog.at_level(logging.WARNING, logger="src.draft_engine"):
        result = engine._apply_fip_correction(pool)

    # The bug is silent: pitchers exist with valid FIP, but the mask is False
    # so nothing happens. After fix, we expect EITHER a logged warning
    # (and possibly the era adjusted from FIP) OR an explicit raise. The
    # minimum check: the code path must log something so operators can see
    # the data is malformed.
    log_messages = " | ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "is_hitter" in log_messages.lower(), (
        f"D6A-002: expected WARNING mentioning is_hitter when col missing, got: {log_messages!r}"
    )

    # The returned pool must still be a DataFrame with at least the original rows
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4


def test_d6a003_streaming_penalty_with_missing_is_hitter_col(caplog):
    """Without is_hitter column, _apply_contextual_factors must NOT silently
    skip the pitcher streaming-penalty stage.

    Prior bug: same scalar-False mask trap as D6A-002.
    """
    from src.draft_engine import DraftRecommendationEngine
    from src.draft_state import DraftState
    from src.valuation import LeagueConfig

    engine = DraftRecommendationEngine(config=LeagueConfig(), mode="quick")
    draft_state = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
    pool = _make_minimal_pool_without_is_hitter()

    with caplog.at_level(logging.WARNING, logger="src.draft_engine"):
        result = engine._apply_contextual_factors(pool, draft_state)

    log_messages = " | ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "is_hitter" in log_messages.lower(), (
        f"D6A-003: expected WARNING about missing is_hitter column, got: {log_messages!r}"
    )

    # Returned pool must include the streaming_penalty column with sane values
    assert "streaming_penalty" in result.columns
    # After fix, both pitchers (ip=70 < 80) Pitcher2 should get -0.3, Pitcher1 (ip=200) gets 0
    # but since is_hitter is missing, we can only validate that the code didn't crash
    # and a warning was logged.


def test_d6b011_lineup_protection_does_not_crash_without_is_hitter(caplog):
    """Without is_hitter column AND with user picks drafted, the
    lineup-protection-bonus stage previously CRASHED with KeyError because
    `drafted_pool.loc[True]` raises 'boolean label can not be used without
    a boolean index'. The try/except only catches AttributeError/TypeError/
    IndexError — not KeyError — so the crash bubbles up.

    After fix: stage either logs warning and skips, or correctly handles
    missing column without crash.
    """
    from src.draft_engine import DraftRecommendationEngine
    from src.draft_state import DraftState
    from src.valuation import LeagueConfig

    engine = DraftRecommendationEngine(config=LeagueConfig(), mode="quick")
    draft_state = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
    # Simulate user has drafted 2 players
    draft_state.user_team.picks = [(0, 1, "Hitter1"), (12, 2, "Hitter2")]
    draft_state.drafted_player_ids = {1, 2}

    pool = _make_minimal_pool_without_is_hitter()

    # Must not crash. lineup_protection_bonus must end up as a column
    # with reasonable values (after fix: warn + treat as no-bonus, or
    # raise an explicit data error).
    with caplog.at_level(logging.WARNING, logger="src.draft_engine"):
        result = engine._apply_contextual_factors(pool, draft_state)

    # The key assertion: no crash, and lineup_protection_bonus exists.
    assert "lineup_protection_bonus" in result.columns


def test_d6b013_spring_training_with_missing_is_hitter_col(caplog):
    """Without is_hitter column, _apply_spring_training_signal must NOT
    silently skip pitchers due to scalar-False mask broadcast."""
    from src.draft_engine import DraftRecommendationEngine
    from src.valuation import LeagueConfig

    engine = DraftRecommendationEngine(config=LeagueConfig(), mode="standard")
    pool = _make_minimal_pool_without_is_hitter()
    pool["spring_training_k_rate"] = [np.nan, np.nan, 0.30, 0.28]  # only pitchers have ST K-rate

    with caplog.at_level(logging.WARNING, logger="src.draft_engine"):
        result = engine._apply_spring_training_signal(pool)

    log_messages = " | ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "is_hitter" in log_messages.lower(), (
        f"D6B-013: expected WARNING about missing is_hitter column, got: {log_messages!r}"
    )

    assert "st_signal" in result.columns


# ---------------------------------------------------------------------------
# D6B-017 — draft_state.get_user_roster_totals NaN handling for OBP denom
# ---------------------------------------------------------------------------


def test_d6b017_obp_denom_excludes_nan():
    """Per-pick aggregation must NOT crash on NaN OBP-denom inputs.

    Before fix: `int(p.get("ab", 0) or 0)` raises ValueError on NaN
    (since NaN is truthy, the `or 0` fallback never fires).

    After fix: the function should skip NaN values for OBP-denom inputs,
    NOT coerce them to 0 (which would fake a "0 PA" record and bias the
    denominator), and should still return a valid totals dict.
    """
    from src.draft_state import DraftState

    state = DraftState(num_teams=12, num_rounds=23, user_team_index=0)

    # Add 3 hitters to user's team
    state.make_pick(1, "Hitter A", "OF", team_index=0)
    state.current_pick += 11  # snake to opponents
    state.make_pick(2, "Hitter B", "1B", team_index=0)
    state.current_pick += 11
    state.make_pick(3, "Pitcher C", "SP", team_index=0)

    # Pool: 2 valid hitters + 1 pitcher whose hitting stats are NaN
    pool = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Hitter A", "Hitter B", "Pitcher C"],
            "ab": [400, 350, np.nan],
            "h": [120, 100, np.nan],
            "bb": [50, 40, np.nan],
            "hbp": [3, 2, np.nan],
            "sf": [4, 3, np.nan],
            "r": [80, 70, np.nan],
            "hr": [25, 20, np.nan],
            "rbi": [85, 75, np.nan],
            "sb": [10, 5, np.nan],
            "w": [np.nan, np.nan, 12],
            "l": [np.nan, np.nan, 8],
            "sv": [np.nan, np.nan, 0],
            "k": [np.nan, np.nan, 200],
            "ip": [np.nan, np.nan, 200.0],
            "er": [np.nan, np.nan, 80.0],
            "bb_allowed": [np.nan, np.nan, 60.0],
            "h_allowed": [np.nan, np.nan, 180.0],
        }
    )

    # Must not crash on the pitcher's NaN hitting stats
    totals = state.get_user_roster_totals(pool)

    # Hitter contributions only — pitcher's NaN AB/BB/HBP/SF must NOT
    # be coerced to 0 and summed into the denom (the bias case).
    # Expected: ab=750, bb=90, hbp=5, sf=7 → obp_denom=852
    expected_ab = 400 + 350
    expected_obp_denom = (400 + 50 + 3 + 4) + (350 + 40 + 2 + 3)

    assert totals["ab"] == expected_ab, f"AB sum: expected {expected_ab}, got {totals['ab']}"
    actual_obp_denom = totals["ab"] + totals["bb"] + totals["hbp"] + totals["sf"]
    assert actual_obp_denom == expected_obp_denom, f"OBP denom: expected {expected_obp_denom}, got {actual_obp_denom}"

    # And the pitcher's pitching stats DO get counted
    assert totals["K"] == 200
    assert totals["ip"] == 200.0


def test_d6b017_obp_denom_handles_partial_nan_row():
    """A hitter with some NaN columns (e.g. missing SF) should not crash
    the aggregator — the non-NaN columns contribute, the NaN columns
    are skipped for that player."""
    from src.draft_state import DraftState

    state = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
    state.make_pick(1, "Hitter A", "OF", team_index=0)

    pool = pd.DataFrame(
        {
            "player_id": [1],
            "name": ["Hitter A"],
            "ab": [400],
            "h": [120],
            "bb": [50],
            "hbp": [3],
            "sf": [np.nan],  # NaN here
            "r": [80],
            "hr": [25],
            "rbi": [85],
            "sb": [10],
            "w": [0],
            "l": [0],
            "sv": [0],
            "k": [0],
            "ip": [0.0],
            "er": [0.0],
            "bb_allowed": [0.0],
            "h_allowed": [0.0],
        }
    )

    totals = state.get_user_roster_totals(pool)
    assert totals["ab"] == 400
    assert totals["sf"] == 0  # NaN skipped → stays at initialized 0


# ---------------------------------------------------------------------------
# D3A-003 — optimizer/projections.py K3 block crash on partial delta column
# ---------------------------------------------------------------------------


def test_d3a003_k3_block_handles_xwoba_delta_only():
    """K3 consistency block must not crash when only xwoba_delta is in
    the DataFrame (babip_delta missing entirely).

    Before fix: `pd.to_numeric(df.get('babip_delta', 0), errors='coerce')`
    returns a scalar int 0, then `.fillna(0).abs()` raises
    AttributeError because int has no fillna.
    """
    from src.optimizer.projections import build_enhanced_projections

    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "positions": ["OF", "1B", "SS"],
            "is_hitter": [True, True, True],
            "r": [80, 70, 60],
            "hr": [25, 20, 15],
            "rbi": [85, 75, 65],
            "sb": [10, 5, 20],
            "k": [0, 0, 0],
            "w": [0, 0, 0],
            "l": [0, 0, 0],
            "sv": [0, 0, 0],
            "avg": [0.275, 0.260, 0.290],
            "obp": [0.340, 0.330, 0.360],
            "era": [4.50, 4.50, 4.50],
            "whip": [1.30, 1.30, 1.30],
            "xwoba_delta": [0.02, -0.01, 0.005],
            # babip_delta MISSING
        }
    )

    # Must not crash
    result = build_enhanced_projections(
        roster,
        config=None,
        enable_bayesian=False,
        enable_kalman=False,
        enable_statcast=False,
        enable_injury=False,
        enable_playing_time=False,
    )
    assert len(result) == 3


def test_d3a003_k3_block_handles_babip_delta_only():
    """Symmetric: only babip_delta present."""
    from src.optimizer.projections import build_enhanced_projections

    roster = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "positions": ["OF", "1B"],
            "is_hitter": [True, True],
            "r": [80, 70],
            "hr": [25, 20],
            "rbi": [85, 75],
            "sb": [10, 5],
            "k": [0, 0],
            "w": [0, 0],
            "l": [0, 0],
            "sv": [0, 0],
            "avg": [0.275, 0.260],
            "obp": [0.340, 0.330],
            "era": [4.50, 4.50],
            "whip": [1.30, 1.30],
            "babip_delta": [0.015, -0.020],
            # xwoba_delta MISSING
        }
    )

    result = build_enhanced_projections(
        roster,
        config=None,
        enable_bayesian=False,
        enable_kalman=False,
        enable_statcast=False,
        enable_injury=False,
        enable_playing_time=False,
    )
    assert len(result) == 2


def test_d3a003_k3_block_skips_when_both_missing():
    """When neither delta col is present, the K3 block should be a no-op,
    not crash."""
    from src.optimizer.projections import build_enhanced_projections

    roster = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["A"],
            "positions": ["OF"],
            "is_hitter": [True],
            "r": [80],
            "hr": [25],
            "rbi": [85],
            "sb": [10],
            "k": [0],
            "w": [0],
            "l": [0],
            "sv": [0],
            "avg": [0.275],
            "obp": [0.340],
            "era": [4.50],
            "whip": [1.30],
            # Both xwoba_delta and babip_delta MISSING
        }
    )

    result = build_enhanced_projections(
        roster,
        config=None,
        enable_bayesian=False,
        enable_kalman=False,
        enable_statcast=False,
        enable_injury=False,
        enable_playing_time=False,
    )
    assert len(result) == 1
    # Counting stats should be unchanged when K3 is skipped
    assert result.iloc[0]["r"] == 80
    assert result.iloc[0]["hr"] == 25
