"""Tests for dynamic SGP denominators and configurable rate-stat baselines.

Covers:
- SGPCalculator accepting custom denominators via __init__
- Custom denominators producing different SGP values than defaults
- Rate-stat baselines (AB, PA, IP) being configurable on LeagueConfig
- Baselines flowing through total_sgp_batch and _rate_stat_sgp
"""

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator

# ── Helpers ─────────────────────────────────────────────────────────


def _hitter(pid=1, name="Hitter", pa=600, ab=550, h=150, r=80, hr=25,
            rbi=80, sb=10, avg=None, obp=None, positions="OF"):
    if avg is None:
        avg = h / ab if ab > 0 else 0.0
    if obp is None:
        obp = (h + 50) / pa if pa > 0 else 0.0
    return {
        "player_id": pid, "name": name, "team": "TST",
        "positions": positions, "is_hitter": 1, "is_injured": 0,
        "pa": pa, "ab": ab, "h": h, "r": r, "hr": hr, "rbi": rbi,
        "sb": sb, "avg": avg, "obp": obp,
        "bb": 50, "hbp": 5, "sf": 3,
        "ip": 0, "w": 0, "l": 0, "sv": 0, "k": 0,
        "era": 0, "whip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
        "adp": pid,
    }


def _pitcher(pid=100, name="Pitcher", ip=180, w=12, l=8, sv=0, k=180,
             era=3.50, whip=1.20, er=None, bb_allowed=None, h_allowed=None):
    if er is None:
        er = era * ip / 9
    if bb_allowed is None:
        bb_allowed = int(whip * ip * 0.35)
    if h_allowed is None:
        h_allowed = int(whip * ip * 0.65)
    return {
        "player_id": pid, "name": name, "team": "TST",
        "positions": "SP", "is_hitter": 0, "is_injured": 0,
        "pa": 0, "ab": 0, "h": 0, "r": 0, "hr": 0, "rbi": 0,
        "sb": 0, "avg": 0, "obp": 0, "bb": 0, "hbp": 0, "sf": 0,
        "ip": ip, "w": w, "l": l, "sv": sv, "k": k,
        "era": era, "whip": whip, "er": er,
        "bb_allowed": bb_allowed, "h_allowed": h_allowed,
        "adp": pid,
    }


def _pool(players):
    return pd.DataFrame(players)


# ── Custom Denominators Tests ───────────────────────────────────────


class TestCustomDenominators:
    """SGPCalculator accepts and uses custom denominators."""

    def test_default_denominators_match_config(self):
        """Without custom denoms, SGPCalculator uses config defaults."""
        config = LeagueConfig()
        calc = SGPCalculator(config)
        assert calc._denominators is config.sgp_denominators

    def test_custom_denominators_stored(self):
        """Custom denoms passed to __init__ are stored on the instance."""
        config = LeagueConfig()
        custom = {"R": 50.0, "HR": 20.0, "RBI": 50.0, "SB": 20.0,
                  "AVG": 0.006, "OBP": 0.007,
                  "W": 5.0, "L": 4.0, "SV": 12.0, "K": 60.0,
                  "ERA": 0.30, "WHIP": 0.030}
        calc = SGPCalculator(config, denominators=custom)
        assert calc._denominators is custom
        assert calc._denominators["R"] == 50.0

    def test_none_denominators_falls_back_to_config(self):
        """Passing None explicitly still uses config defaults."""
        config = LeagueConfig()
        calc = SGPCalculator(config, denominators=None)
        assert calc._denominators is config.sgp_denominators

    def test_custom_denominators_change_player_sgp(self):
        """Custom denoms produce different SGP values than defaults."""
        config = LeagueConfig()
        player = pd.Series(_hitter(r=80, hr=25, rbi=80, sb=10))

        default_calc = SGPCalculator(config)
        default_sgp = default_calc.player_sgp(player)

        # Double all counting denominators -> halve SGP values
        big_denoms = dict(config.sgp_denominators)
        big_denoms["R"] = 64.0   # 2x default 32
        big_denoms["HR"] = 26.0  # 2x default 13
        big_denoms["RBI"] = 64.0
        big_denoms["SB"] = 28.0
        custom_calc = SGPCalculator(config, denominators=big_denoms)
        custom_sgp = custom_calc.player_sgp(player)

        # With doubled counting denoms, counting SGP should be roughly halved
        assert abs(custom_sgp["R"] - default_sgp["R"] / 2) < 0.01
        assert abs(custom_sgp["HR"] - default_sgp["HR"] / 2) < 0.01

    def test_custom_denominators_change_total_sgp(self):
        """total_sgp uses custom denominators."""
        config = LeagueConfig()
        player = pd.Series(_hitter())

        calc_default = SGPCalculator(config)
        calc_custom = SGPCalculator(config, denominators={
            **config.sgp_denominators, "R": 100.0, "HR": 100.0
        })

        assert calc_default.total_sgp(player) != calc_custom.total_sgp(player)

    def test_custom_denominators_change_batch_sgp(self):
        """total_sgp_batch uses custom denominators."""
        config = LeagueConfig()
        pool = _pool([_hitter(pid=1, r=80), _hitter(pid=2, r=40)])

        calc_default = SGPCalculator(config)
        calc_custom = SGPCalculator(config, denominators={
            **config.sgp_denominators, "R": 100.0
        })

        batch_default = calc_default.total_sgp_batch(pool)
        batch_custom = calc_custom.total_sgp_batch(pool)

        # Different denominators must produce different results
        assert not np.allclose(batch_default, batch_custom)

    def test_custom_denominators_change_marginal_sgp(self):
        """marginal_sgp uses custom denominators."""
        config = LeagueConfig()
        player = pd.Series(_hitter())
        roster = {"R": 800, "HR": 250, "RBI": 800, "SB": 100,
                  "ab": 5500, "h": 1400, "ip": 0, "er": 0,
                  "bb_allowed": 0, "h_allowed": 0,
                  "bb": 500, "hbp": 30, "sf": 40}

        calc_default = SGPCalculator(config)
        calc_custom = SGPCalculator(config, denominators={
            **config.sgp_denominators, "RBI": 100.0
        })

        marg_default = calc_default.marginal_sgp(player, roster)
        marg_custom = calc_custom.marginal_sgp(player, roster)

        assert marg_default["RBI"] != marg_custom["RBI"]

    def test_empty_dict_denominators_falls_back_to_config(self):
        """Empty dict is falsy, so falls back to config defaults."""
        config = LeagueConfig()
        calc = SGPCalculator(config, denominators={})
        # Empty dict is falsy -> falls back to config.sgp_denominators
        assert calc._denominators is config.sgp_denominators


# ── Rate-Stat Baselines Tests ──────────────────────────────────────


class TestRateStatBaselines:
    """LeagueConfig rate-stat baselines are configurable and flow through SGP."""

    def test_default_baselines(self):
        """Default baselines match historical values."""
        config = LeagueConfig()
        assert config.roster_ab_baseline == 5500.0
        assert config.roster_pa_baseline == 6100.0
        assert config.roster_ip_baseline == 1300.0

    def test_custom_baselines(self):
        """Baselines can be overridden at construction."""
        config = LeagueConfig(
            roster_ab_baseline=6000.0,
            roster_pa_baseline=6600.0,
            roster_ip_baseline=1400.0,
        )
        assert config.roster_ab_baseline == 6000.0
        assert config.roster_pa_baseline == 6600.0
        assert config.roster_ip_baseline == 1400.0

    def test_baselines_affect_batch_avg_sgp(self):
        """Different AB baseline changes AVG SGP in total_sgp_batch."""
        pool = _pool([_hitter(pid=1, ab=550, h=165)])  # .300 hitter

        config_default = LeagueConfig()
        config_big_ab = LeagueConfig(roster_ab_baseline=8000.0)

        sgp_default = SGPCalculator(config_default).total_sgp_batch(pool)[0]
        sgp_big_ab = SGPCalculator(config_big_ab).total_sgp_batch(pool)[0]

        # Larger AB baseline dilutes individual contribution -> different SGP
        assert sgp_default != sgp_big_ab

    def test_baselines_affect_batch_era_sgp(self):
        """Different IP baseline changes ERA SGP in total_sgp_batch."""
        pool = _pool([_pitcher(pid=1, ip=200, era=3.00)])

        config_default = LeagueConfig()
        config_big_ip = LeagueConfig(roster_ip_baseline=1600.0)

        sgp_default = SGPCalculator(config_default).total_sgp_batch(pool)[0]
        sgp_big_ip = SGPCalculator(config_big_ip).total_sgp_batch(pool)[0]

        assert sgp_default != sgp_big_ip

    def test_baselines_affect_single_player_avg_sgp(self):
        """Different AB baseline changes AVG in _rate_stat_sgp."""
        config_default = LeagueConfig()
        config_small = LeagueConfig(roster_ab_baseline=3000.0)

        player = pd.Series(_hitter(ab=550, h=165))

        sgp_default = SGPCalculator(config_default).player_sgp(player)["AVG"]
        sgp_small = SGPCalculator(config_small).player_sgp(player)["AVG"]

        # Smaller baseline = player has more impact on rate stat
        assert abs(sgp_small) > abs(sgp_default)

    def test_baselines_affect_single_player_obp_sgp(self):
        """Different PA baseline changes OBP in _rate_stat_sgp."""
        config_default = LeagueConfig()
        config_small = LeagueConfig(roster_pa_baseline=4000.0)

        player = pd.Series(_hitter(pa=600, h=150, ab=550))

        sgp_default = SGPCalculator(config_default).player_sgp(player)["OBP"]
        sgp_small = SGPCalculator(config_small).player_sgp(player)["OBP"]

        assert sgp_default != sgp_small

    def test_baselines_affect_single_player_whip_sgp(self):
        """Different IP baseline changes WHIP in _rate_stat_sgp."""
        config_default = LeagueConfig()
        config_small = LeagueConfig(roster_ip_baseline=800.0)

        player = pd.Series(_pitcher(ip=200, whip=1.10))

        sgp_default = SGPCalculator(config_default).player_sgp(player)["WHIP"]
        sgp_small = SGPCalculator(config_small).player_sgp(player)["WHIP"]

        assert sgp_default != sgp_small

    def test_batch_and_single_consistent_with_custom_baselines(self):
        """total_sgp_batch and total_sgp agree for counting stats with custom baselines."""
        config = LeagueConfig(roster_ab_baseline=6000.0, roster_ip_baseline=1400.0)
        calc = SGPCalculator(config)

        # Use a hitter with only counting stats (no rate complexity)
        players = [_hitter(pid=1, r=80, hr=25, rbi=80, sb=10, ab=0, h=0, pa=0)]
        pool = _pool(players)

        batch_val = calc.total_sgp_batch(pool)[0]
        single_val = calc.total_sgp(pd.Series(players[0]))

        # Should be very close (both skip rate stats when ab/pa=0)
        assert abs(batch_val - single_val) < 0.01
