"""Tests for volume-aware SGP computation in trade finder.

Verifies that _player_sgp_volume_aware() properly accounts for AB/IP volume
in rate stats, unlike _totals_sgp() which treats rates as raw values.
"""

import pandas as pd
import pytest

from src.in_season import _roster_category_totals
from src.trade_finder import _player_sgp_volume_aware, _totals_sgp
from src.valuation import LeagueConfig, SGPCalculator


@pytest.fixture
def config():
    return LeagueConfig()


@pytest.fixture
def sgp_calc(config):
    return SGPCalculator(config)


def _make_hitter(player_id, ab, avg, hr=10, r=30, rbi=30, sb=5, obp=0.350):
    """Create a hitter row with specified AB and AVG."""
    h = int(ab * avg)
    bb = int(ab * (obp - avg) / (1 - obp + 0.01)) if obp > avg else 0
    pa = ab + bb
    return {
        "player_id": player_id,
        "name": f"Hitter_{player_id}",
        "team": "NYY",
        "positions": "OF",
        "is_hitter": 1,
        "pa": pa,
        "ab": ab,
        "h": h,
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "avg": avg,
        "obp": obp,
        "bb": bb,
        "hbp": 0,
        "sf": 0,
        "ip": 0,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    }


def _make_pitcher(player_id, ip, era, w=5, l=3, sv=0, k=60, whip=1.20):
    """Create a pitcher row with specified IP and ERA."""
    er = ip * era / 9.0
    h_allowed = int(ip * whip * 0.6)
    bb_allowed = int(ip * whip * 0.4)
    return {
        "player_id": player_id,
        "name": f"Pitcher_{player_id}",
        "team": "LAD",
        "positions": "SP",
        "is_hitter": 0,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "ip": ip,
        "w": w,
        "l": l,
        "sv": sv,
        "k": k,
        "era": era,
        "whip": whip,
        "er": er,
        "bb_allowed": bb_allowed,
        "h_allowed": h_allowed,
    }


def _make_pool(players):
    """Build a DataFrame from a list of player dicts."""
    return pd.DataFrame(players)


class TestVolumeAwareHitterSGP:
    """Test that high-AB hitters get more AVG/OBP SGP than low-AB hitters."""

    def test_600ab_hitter_more_avg_sgp_than_200ab(self, sgp_calc):
        """A 600 AB .300 hitter should have MORE AVG SGP than a 200 AB .300 hitter."""
        high_vol = pd.Series(_make_hitter(1, ab=600, avg=0.300))
        low_vol = pd.Series(_make_hitter(2, ab=200, avg=0.300))

        high_sgp = sgp_calc.player_sgp(high_vol)
        low_sgp = sgp_calc.player_sgp(low_vol)

        assert high_sgp["AVG"] > low_sgp["AVG"], (
            f"600 AB .300 hitter AVG SGP ({high_sgp['AVG']:.4f}) should exceed "
            f"200 AB .300 hitter AVG SGP ({low_sgp['AVG']:.4f})"
        )

    def test_600ab_hitter_roughly_3x_avg_impact(self, sgp_calc):
        """A 600 AB .300 hitter should move team AVG roughly 3x more than 200 AB."""
        high_vol = pd.Series(_make_hitter(1, ab=600, avg=0.300))
        low_vol = pd.Series(_make_hitter(2, ab=200, avg=0.300))

        high_sgp = sgp_calc.player_sgp(high_vol)
        low_sgp = sgp_calc.player_sgp(low_vol)

        # Not exactly 3x due to non-linearity, but should be in 2.5-3.5x range
        ratio = high_sgp["AVG"] / low_sgp["AVG"] if low_sgp["AVG"] != 0 else float("inf")
        assert 2.0 < ratio < 4.0, f"AVG SGP ratio should be ~3x, got {ratio:.2f}x"

    def test_player_sgp_volume_aware_uses_sgpcalculator(self, config):
        """_player_sgp_volume_aware should match SGPCalculator.total_sgp for same player."""
        player = _make_hitter(1, ab=500, avg=0.290, hr=25, r=80, rbi=90, sb=10)
        pool = _make_pool([player])

        vol_sgp = _player_sgp_volume_aware(1, pool, config)

        sgp_calc = SGPCalculator(config)
        expected = sgp_calc.total_sgp(pool.iloc[0])

        assert abs(vol_sgp - expected) < 1e-9, (
            f"_player_sgp_volume_aware ({vol_sgp:.4f}) should match SGPCalculator.total_sgp ({expected:.4f})"
        )

    def test_volume_aware_differentiates_same_avg(self, config):
        """Two hitters with same AVG but different AB should have different SGP."""
        high_vol = _make_hitter(1, ab=600, avg=0.300, hr=10, r=30, rbi=30, sb=5)
        low_vol = _make_hitter(2, ab=200, avg=0.300, hr=10, r=30, rbi=30, sb=5)
        pool = _make_pool([high_vol, low_vol])

        sgp_high = _player_sgp_volume_aware(1, pool, config)
        sgp_low = _player_sgp_volume_aware(2, pool, config)

        # High-volume player should have higher total SGP due to AVG/OBP contribution
        assert sgp_high > sgp_low, f"600 AB player SGP ({sgp_high:.4f}) should exceed 200 AB player SGP ({sgp_low:.4f})"


class TestVolumeAwarePitcherSGP:
    """Test that high-IP pitchers get more ERA/WHIP SGP than low-IP pitchers."""

    def test_200ip_pitcher_more_era_impact_than_50ip(self, sgp_calc):
        """A 200 IP 3.00 ERA pitcher should have MORE ERA impact than a 50 IP 3.00 ERA."""
        high_vol = pd.Series(_make_pitcher(1, ip=200, era=3.00))
        low_vol = pd.Series(_make_pitcher(2, ip=50, era=3.00))

        high_sgp = sgp_calc.player_sgp(high_vol)
        low_sgp = sgp_calc.player_sgp(low_vol)

        # Both have below-average ERA (3.00 < 3.80 baseline), so both positive
        # But the 200 IP pitcher should have a larger positive ERA SGP
        assert high_sgp["ERA"] > low_sgp["ERA"], (
            f"200 IP ERA SGP ({high_sgp['ERA']:.4f}) should exceed 50 IP ERA SGP ({low_sgp['ERA']:.4f})"
        )

    def test_volume_aware_pitcher_differentiation(self, config):
        """Two pitchers with same ERA but different IP should have different SGP."""
        high_vol = _make_pitcher(1, ip=200, era=3.00, w=12, l=6, k=180)
        low_vol = _make_pitcher(2, ip=50, era=3.00, w=3, l=2, k=45)
        pool = _make_pool([high_vol, low_vol])

        sgp_high = _player_sgp_volume_aware(1, pool, config)
        sgp_low = _player_sgp_volume_aware(2, pool, config)

        # High-IP pitcher should have higher total SGP overall
        assert sgp_high > sgp_low, (
            f"200 IP pitcher SGP ({sgp_high:.4f}) should exceed 50 IP pitcher SGP ({sgp_low:.4f})"
        )


class TestTotalsSGPRosterTotals:
    """Regression tests: _totals_sgp on full roster totals still works correctly."""

    def test_totals_sgp_basic(self, config):
        """_totals_sgp correctly computes SGP from roster totals dict."""
        totals = {
            "R": 700,
            "HR": 200,
            "RBI": 700,
            "SB": 100,
            "AVG": 0.265,
            "OBP": 0.330,
            "W": 80,
            "L": 60,
            "SV": 50,
            "K": 1200,
            "ERA": 3.80,
            "WHIP": 1.25,
        }
        sgp = _totals_sgp(totals, config)
        # Should be a reasonable positive number
        assert sgp > 0, f"Roster-level SGP should be positive, got {sgp:.2f}"

    def test_totals_sgp_inverse_stats_subtract(self, config):
        """Inverse stats (L, ERA, WHIP) should reduce total SGP."""
        low_era = {
            "R": 0,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0,
            "OBP": 0,
            "W": 0,
            "L": 0,
            "SV": 0,
            "K": 0,
            "ERA": 3.00,
            "WHIP": 0,
        }
        high_era = {
            "R": 0,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0,
            "OBP": 0,
            "W": 0,
            "L": 0,
            "SV": 0,
            "K": 0,
            "ERA": 5.00,
            "WHIP": 0,
        }

        sgp_low = _totals_sgp(low_era, config)
        sgp_high = _totals_sgp(high_era, config)

        # Lower ERA = better = higher (less negative) SGP
        assert sgp_low > sgp_high, "Lower ERA should yield higher SGP"

    def test_totals_sgp_still_used_for_roster_deltas(self, config):
        """Verify that roster-level before/after deltas using _totals_sgp are unaffected."""
        roster_before = {
            "R": 700,
            "HR": 200,
            "RBI": 700,
            "SB": 100,
            "AVG": 0.265,
            "OBP": 0.330,
            "W": 80,
            "L": 60,
            "SV": 50,
            "K": 1200,
            "ERA": 3.80,
            "WHIP": 1.25,
        }
        roster_after = {
            "R": 710,
            "HR": 205,
            "RBI": 710,
            "SB": 102,
            "AVG": 0.267,
            "OBP": 0.332,
            "W": 82,
            "L": 59,
            "SV": 52,
            "K": 1210,
            "ERA": 3.75,
            "WHIP": 1.23,
        }

        delta = _totals_sgp(roster_after, config) - _totals_sgp(roster_before, config)
        # Adding better players should improve total SGP
        assert delta > 0, f"Improvement in all cats should yield positive delta, got {delta:.4f}"


class TestPlayerSGPVolumeAwareEdgeCases:
    """Edge cases for _player_sgp_volume_aware."""

    def test_missing_player_returns_zero(self, config):
        """Non-existent player ID should return 0.0."""
        pool = _make_pool([_make_hitter(1, ab=500, avg=0.280)])
        assert _player_sgp_volume_aware(999, pool, config) == 0.0

    def test_zero_ab_hitter(self, config):
        """A hitter with 0 AB should have zero rate-stat SGP."""
        player = _make_hitter(1, ab=0, avg=0.000, hr=0, r=0, rbi=0, sb=0)
        pool = _make_pool([player])

        sgp = _player_sgp_volume_aware(1, pool, config)
        # Should be 0 or very close since all stats are 0
        assert abs(sgp) < 0.1, f"Zero-stat player SGP should be ~0, got {sgp:.4f}"

    def test_zero_ip_pitcher(self, config):
        """A pitcher with 0 IP should have zero rate-stat SGP."""
        player = _make_pitcher(1, ip=0, era=0.00, w=0, l=0, sv=0, k=0)
        pool = _make_pool([player])

        sgp = _player_sgp_volume_aware(1, pool, config)
        assert abs(sgp) < 0.1, f"Zero-IP pitcher SGP should be ~0, got {sgp:.4f}"


class TestOldVsNewComparison:
    """Demonstrate the bug fix: old method vs new method for individual players."""

    def test_old_method_ignores_volume(self, config):
        """Show that _totals_sgp on single-player totals ignores volume (the bug)."""
        high_vol = _make_hitter(1, ab=600, avg=0.300, hr=10, r=30, rbi=30, sb=5)
        low_vol = _make_hitter(2, ab=200, avg=0.300, hr=10, r=30, rbi=30, sb=5)
        pool = _make_pool([high_vol, low_vol])

        # Old method: _totals_sgp on single-player totals
        old_high = _totals_sgp(_roster_category_totals([1], pool), config)
        old_low = _totals_sgp(_roster_category_totals([2], pool), config)

        # Old method gives same AVG SGP because .300/.004 = 75.0 for both
        # The only difference is from counting stats (same HR/R/RBI/SB)
        # So old_high should equal old_low (the bug)
        assert abs(old_high - old_low) < 0.01, (
            "Old _totals_sgp should give nearly identical SGP for same-rate, "
            f"same-counting players regardless of volume: {old_high:.4f} vs {old_low:.4f}"
        )

        # New method: _player_sgp_volume_aware properly differentiates
        new_high = _player_sgp_volume_aware(1, pool, config)
        new_low = _player_sgp_volume_aware(2, pool, config)

        assert new_high > new_low, f"Volume-aware SGP should differentiate: 600AB={new_high:.4f} > 200AB={new_low:.4f}"
