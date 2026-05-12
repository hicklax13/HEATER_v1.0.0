"""Wave 8a Group 3: pitcher / rate-stat bug fixes.

Covers 5 audit IDs:
  - D4B-002: trade_simulator odd-n_sims zero-pollution
  - D4B-003: trade_simulator missing-WHIP fallback
  - D6B-001: game_day ERA/WHIP no-IP "ELITE 0.00" display
  - D6B-002,003: game_day FIP=team_ERA silent proxy
  - D6B-022: simulation np.maximum sign-flip
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── D4B-002: trade_simulator odd n_sims zero-pollution ───────────────


def test_d4b002_odd_nsims_no_trailing_zero():
    """run_paired_monte_carlo with odd n_sims must not leave a fake 0 in surpluses.

    When n_sims is odd, half_sims = n_sims // 2 fills 2*half_sims slots,
    leaving the last slot at np.zeros default. That trailing 0 biases
    mean/std/percentiles toward 0 (especially noticeable for small n_sims
    where a "better" after roster has lots of positive surplus and a
    single fake 0 drags the mean down).
    """
    from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

    # A strictly better after-roster: every surplus must be > 0.
    before = {
        "p1": {
            "hr": 10,
            "r": 40,
            "rbi": 40,
            "sb": 5,
            "avg": 0.240,
            "obp": 0.300,
            "w": 5,
            "k": 80,
            "sv": 0,
            "era": 5.00,
            "whip": 1.50,
        },
    }
    after = {
        "p1": {
            "hr": 40,
            "r": 110,
            "rbi": 120,
            "sb": 15,
            "avg": 0.310,
            "obp": 0.390,
            "w": 18,
            "k": 280,
            "sv": 0,
            "era": 2.50,
            "whip": 0.95,
        },
    }
    # Note: n_sims < MIN_SIMS (100) gets clamped, so use 101 (odd, above floor).
    result = run_paired_monte_carlo(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=101,  # odd!
        seed=42,
    )
    surpluses = np.asarray(result["surplus_distribution"])

    # Every surplus must be strictly positive (after is strictly better).
    # If the bug exists, the last entry stays at 0.0.
    n_zeros = int(np.sum(surpluses == 0.0))
    assert n_zeros == 0, f"odd n_sims=101 left {n_zeros} zero-filled entries; mean polluted toward 0"

    # And the reported length matches the number of *actually filled* slots.
    # (Either the array is sliced to 98, or n_sims is reduced to 98 — either way,
    # there must be no unfilled trailing zero.)
    assert len(surpluses) > 0
    # The reported n_sims should reflect what was actually computed.
    assert int(result["n_sims"]) == len(surpluses)


# ── D4B-003: trade_simulator missing-WHIP fallback ───────────────────


def test_d4b003_missing_whip_does_not_inflate_team_whip():
    """When a pitcher's projection lacks WHIP but has ERA, fallback must
    not contribute 150 IP with 0 baserunners.

    Old behavior: p_ip = 150 (synthetic), p_bb_h_allowed = 0 (whip missing).
    That synthetic pitcher pollutes team-aggregate WHIP downward.

    New behavior: when WHIP is missing, the fallback uses league-average
    WHIP (1.30) × the synthetic IP (so the player contributes 195
    baserunners, not 0), OR omits the player from the WHIP aggregation
    entirely. We accept either fix as long as team WHIP is no longer
    suspiciously near 0.
    """
    from src.engine.monte_carlo.trade_simulator import _simulate_roster_sgp

    # Roster: 1 pitcher with ERA only, no WHIP.
    roster = {
        "ace": {
            "w": 15,
            "k": 230,
            "sv": 0,
            "era": 3.20,
            # whip intentionally missing/zero
        }
    }
    denoms = {
        "R": 50,
        "HR": 10,
        "RBI": 50,
        "SB": 10,
        "AVG": 0.005,
        "OBP": 0.005,
        "W": 5,
        "L": 5,
        "SV": 5,
        "K": 50,
        "ERA": 0.20,
        "WHIP": 0.05,
    }

    # We can't directly read team WHIP from this fn (returns SGP), but we
    # can compare two scenarios:
    #   A) ERA only — buggy code aggregates 150 IP with 0 baserunners → WHIP ≈ 0
    #   B) ERA + reasonable WHIP — yields realistic WHIP ≈ 1.20
    # Both should yield similar SGP because the WHIP contribution should
    # default to league average, not 0.
    sgp_no_whip = _simulate_roster_sgp(
        roster_stats=roster,
        marginals=None,
        copula=None,
        sgp_denominators=denoms,
        seed=42,
        weeks_remaining=16,
        negate_noise=False,
    )

    roster_with_whip = {
        "ace": {
            "w": 15,
            "k": 230,
            "sv": 0,
            "era": 3.20,
            "whip": 1.30,  # league average
        }
    }
    sgp_with_whip = _simulate_roster_sgp(
        roster_stats=roster_with_whip,
        marginals=None,
        copula=None,
        sgp_denominators=denoms,
        seed=42,
        weeks_remaining=16,
        negate_noise=False,
    )

    # The SGPs should be close (within 2 SGP) — the no-whip path should
    # use league-average WHIP, not phantom 0 baserunners. Pre-fix the
    # delta blows up because WHIP=0 vastly inflates SGP for inverse stats.
    delta = abs(sgp_no_whip - sgp_with_whip)
    assert delta < 2.0, (
        f"Missing WHIP should default to league average, not 0 baserunners "
        f"(sgp_no_whip={sgp_no_whip:.2f}, sgp_with_whip={sgp_with_whip:.2f}, "
        f"delta={delta:.2f})"
    )


# ── D6B-001: game_day ERA/WHIP no-IP returns None ────────────────────


def test_d6b001_era_whip_return_none_when_no_ip():
    """_aggregate_pitching_games must NOT return 0.0 for ERA/WHIP when IP=0.

    Old behavior: era=0.0, whip=0.0 when total_ip == 0 — displayed in UI
    as elite (0.00 ERA, 0.00 WHIP), which is wrong: no innings pitched.

    New behavior: return None (or NaN), letting consumers render as "—"
    via format_stat.
    """
    from src.game_day import _aggregate_pitching_games

    # Pitcher with appearances but no innings (e.g. faced 0 batters, was
    # a position-player pitching, or some statsapi corner case).
    games = [
        {
            "inningsPitched": "0",
            "strikeOuts": 0,
            "wins": 0,
            "earnedRuns": 0,
            "baseOnBalls": 0,
            "hits": 0,
        }
    ]
    result = _aggregate_pitching_games(games)
    assert result["era"] is None, f"ERA must be None on no-IP, got {result['era']}"
    assert result["whip"] is None, f"WHIP must be None on no-IP, got {result['whip']}"


def test_d6b001_aggregate_pitching_with_ip_returns_floats():
    """Sanity: with IP > 0, ERA/WHIP must still be floats (not None).

    Don't regress the happy path.
    """
    from src.game_day import _aggregate_pitching_games

    games = [
        {
            "inningsPitched": "6.0",
            "strikeOuts": 8,
            "wins": 1,
            "earnedRuns": 2,
            "baseOnBalls": 1,
            "hits": 5,
        }
    ]
    result = _aggregate_pitching_games(games)
    assert isinstance(result["era"], float), f"ERA must be float on positive IP, got {type(result['era'])}"
    assert isinstance(result["whip"], float), f"WHIP must be float on positive IP, got {type(result['whip'])}"
    assert result["era"] == 3.0  # (2 * 9) / 6 = 3.00
    assert result["whip"] == 1.0  # (1 + 5) / 6 = 1.00


# ── D6B-002,003: game_day FIP=team_ERA proxy flag ────────────────────


def test_d6b002_statsapi_fallback_flags_fip_as_proxy(caplog):
    """When statsapi fallback substitutes team_ERA for FIP, it must:
    (1) log a warning so the substitution is visible in bootstrap logs;
    (2) flag the row with fip_is_proxy=True so consumers know FIP is a proxy.
    """
    from unittest.mock import patch

    import src.database as db_mod
    import src.game_day as gd
    from src.database import init_db

    # Use a temp DB so upsert_team_strength has a table to write into.
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original_db = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()

    try:
        # Mock statsapi.get to return a controllable shape.
        teams_response = {
            "teams": [
                {"id": 1, "abbreviation": "NYY", "name": "New York Yankees"},
            ]
        }
        batting_response = {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "ops": "0.750",
                                "plateAppearances": 6000,
                                "strikeOuts": 1200,
                                "baseOnBalls": 600,
                            }
                        }
                    ]
                }
            ]
        }
        pitching_response = {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "era": "3.85",
                                "whip": "1.20",
                            }
                        }
                    ]
                }
            ]
        }

        def mock_get(endpoint, params=None, **kwargs):
            if endpoint == "teams":
                return teams_response
            if endpoint == "team_stats":
                if params and params.get("group") == "hitting":
                    return batting_response
                return pitching_response
            return {}

        mock_statsapi = type("MockStatsapi", (), {"get": staticmethod(mock_get)})

        with patch.object(gd, "_statsapi", mock_statsapi):
            with caplog.at_level(logging.WARNING, logger=gd.logger.name):
                df = gd._fetch_team_strength_statsapi(2026)
    finally:
        db_mod.DB_PATH = original_db
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Row exists, FIP equals team_era (still the proxy, behavior unchanged),
    # but it MUST be flagged.
    assert not df.empty, "Expected at least one row"
    row = df.iloc[0]
    # Use == True so np.bool_(True) compares equal; identity check fails
    # on numpy booleans.
    assert bool(row.get("fip_is_proxy")) is True, (
        f"Expected fip_is_proxy=True on statsapi-fallback row, got {row.get('fip_is_proxy')!r}"
    )

    # And a logger.warning must have been emitted about the proxy.
    proxy_logs = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "fip" in rec.getMessage().lower() and "proxy" in rec.getMessage().lower()
    ]
    assert proxy_logs, (
        "Expected at least one WARNING log mentioning FIP proxy when statsapi fallback substitutes team_ERA for FIP"
    )


# ── D6B-022: simulation magnitude-clip preserves SGP sign ────────────


def test_d6b022_signed_magnitude_clip_preserves_sign():
    """The helper that replaces `np.maximum(sgp_values, 0.01)` must preserve
    sign for SGP values that are negative (inverse-stat-heavy pitchers,
    bad picks).

    For input [-0.5, 0.005, -0.001, 0.5]:
      -0.5    → -0.5    (|x| >= 0.01, unchanged)
       0.005  → +0.01   (|x| < 0.01, clipped to +sign)
      -0.001  → -0.01   (|x| < 0.01, clipped to -sign) ← THE BUG
      +0.5    → +0.5    (|x| >= 0.01, unchanged)
    """
    from src.simulation import _signed_magnitude_clip

    sgp_values = np.array([-0.5, 0.005, -0.001, 0.5])
    out = _signed_magnitude_clip(sgp_values, min_mag=0.01)

    assert out[0] == pytest.approx(-0.5)
    assert out[1] == pytest.approx(0.01)
    # The critical one: -0.001 must NOT become +0.01.
    assert out[2] < 0, f"Expected negative output for -0.001 input, got {out[2]}"
    assert out[2] == pytest.approx(-0.01)
    assert out[3] == pytest.approx(0.5)


def test_d6b022_zero_input_uses_default_positive_magnitude():
    """For exactly zero input (no SGP signal), the helper must produce a
    non-zero output so downstream weight normalization doesn't divide by 0.
    Sign convention for exactly-zero is positive (np.sign(0) == 0, so we
    treat 0 specially)."""
    from src.simulation import _signed_magnitude_clip

    out = _signed_magnitude_clip(np.array([0.0]), min_mag=0.01)
    assert out[0] != 0
    # Default-positive: 0 isn't negative-SGP, so positive is sensible.
    assert out[0] == pytest.approx(0.01)


def test_d6b022_simulator_uses_signed_clip(monkeypatch):
    """End-to-end: the draft simulator's AI-pick step must invoke the
    signed-magnitude clip (so negative SGP players don't masquerade as
    marginal-positive picks).

    Spot-check by ensuring _signed_magnitude_clip is called from
    simulate_draft when sgp_values has any negative entries.
    """
    import src.simulation as sim

    call_records: list = []
    real_clip = sim._signed_magnitude_clip

    def spy_clip(values, min_mag=0.01):
        call_records.append((np.asarray(values).copy(), min_mag))
        return real_clip(values, min_mag=min_mag)

    monkeypatch.setattr(sim, "_signed_magnitude_clip", spy_clip)

    n_players = 6
    available_ids = np.arange(n_players)
    adp_values = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    # Mix of positive and negative SGP — guarantees the negative-SGP branch
    # is meaningful.
    sgp_values = np.array([2.0, 1.0, -0.5, 0.5, -1.0, 0.1])
    positions = ["1B", "2B", "OF", "SP", "RP", "SS"]

    from src.valuation import LeagueConfig

    simulator = sim.DraftSimulator(config=LeagueConfig())
    result = simulator.simulate_draft(
        available_ids=available_ids,
        adp_values=adp_values,
        sgp_values=sgp_values,
        positions=positions,
        user_team_index=0,
        current_pick=1,
        total_picks=24,
        num_teams=12,
        user_roster_needs=set(),
        candidate_id=0,
        n_simulations=3,
        team_positions=None,
    )
    # Must produce some finite output.
    assert "mean_sgp" in result
    assert np.isfinite(result["mean_sgp"])
    # And the spy must have been called at least once.
    assert call_records, "Expected _signed_magnitude_clip to be invoked"
