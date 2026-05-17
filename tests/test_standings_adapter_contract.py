"""D5: Verify pages/6_League_Standings.py adapter contract holds end-to-end.

PR #8 (Wave 2-G) replaced a custom 70-line ``_compute_projected_team_totals``
with a thin adapter wrapping ``standings_utils.get_all_team_totals``. The
adapter:

  * Pulls **season-to-date** category totals from Yahoo (or projection
    fallback) via ``get_all_team_totals``.
  * Divides counting stats (R, HR, RBI, SB, W, L, SV, K) by 26 to produce a
    per-matchup-week pace.
  * Passes rate stats (AVG, OBP, ERA, WHIP) through unchanged.
  * Backfills league-average rate defaults if any rate stat is missing.

Two downstream consumers depend on the resulting ``team_weekly_totals`` dict:

  1. ``src.standings_engine.simulate_season_enhanced`` — uses the values as
     **weekly means** for a multivariate-Normal MC simulation against
     ``CALIBRATED_WEEKLY_TAU`` (which is calibrated to weekly variance).
  2. ``src.standings_engine.compute_team_strength_profiles`` — ranks teams
     per category to derive roster_quality / category_balance for the power
     rating.

This test reproduces the adapter's scaling step, then drives both consumers
with the produced dict to assert:

  * The dict shape is what the consumers expect (``{team: {cat: float}}``
    with all 12 LeagueConfig categories present).
  * Counting stats survive the divide-by-26 within the magnitude
    ``CALIBRATED_WEEKLY_TAU`` was calibrated for.
  * Rate stats round-trip unchanged.
  * Inverse-category semantics hold: a team with a lower ERA/WHIP wins more
    sims and ranks higher in the strength profile than a team with higher
    ERA/WHIP, even after the adapter passes the values through.
  * Both consumers run without KeyError/ValueError on the adapter's output
    and return their documented schema.
"""

from __future__ import annotations

import pytest

from src.standings_engine import (
    ALL_CATEGORIES,
    CALIBRATED_WEEKLY_TAU,
    compute_team_strength_profiles,
    simulate_season_enhanced,
)
from src.valuation import LeagueConfig

# ── Adapter scaling contract (mirrors pages/6_League_Standings.py:147-163) ──
_WEEKS = 26.0
_COUNTING = {"R", "HR", "RBI", "SB", "W", "L", "SV", "K"}
_RATE = {"AVG", "OBP", "ERA", "WHIP"}
_RATE_DEFAULTS = {"AVG": 0.250, "OBP": 0.330, "ERA": 4.00, "WHIP": 1.25}


def _adapter_scale(season_totals: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Reproduce the adapter scaling step from pages/6_League_Standings.py.

    Kept in sync with ``_compute_projected_team_totals`` (lines 147-163). If
    that scaling logic ever changes (e.g. switching to elapsed-week pacing),
    update this helper AND assert the new contract here.
    """
    weekly: dict[str, dict[str, float]] = {}
    for team, cat_map in season_totals.items():
        per_week: dict[str, float] = {}
        for cat, val in cat_map.items():
            if cat in _COUNTING:
                per_week[cat] = float(val) / _WEEKS
            elif cat in _RATE:
                per_week[cat] = float(val)
        for cat, default in _RATE_DEFAULTS.items():
            per_week.setdefault(cat, default)
        weekly[str(team)] = per_week
    return weekly


def _make_season_totals(strong_team: str = "A") -> dict[str, dict[str, float]]:
    """Build a synthetic 4-team league with one clearly-strong team.

    Counting stats are season totals (full 26 weeks). Rate stats are raw rates.
    Strong team has higher counting stats AND better rate stats (lower ERA/WHIP).
    """
    teams = ["A", "B", "C", "D"]
    base = {
        # Counting (season totals over 26 weeks)
        "R": 800.0,
        "HR": 200.0,
        "RBI": 780.0,
        "SB": 110.0,
        "W": 75.0,
        "L": 70.0,
        "SV": 50.0,
        "K": 1300.0,
        # Rate stats (raw)
        "AVG": 0.260,
        "OBP": 0.330,
        "ERA": 3.90,
        "WHIP": 1.25,
    }
    season: dict[str, dict[str, float]] = {}
    for t in teams:
        # The strong team gets +20% counting and -10% on inverse rate stats.
        if t == strong_team:
            season[t] = {
                **{c: v * 1.20 for c, v in base.items() if c in _COUNTING},
                "AVG": 0.290,
                "OBP": 0.365,
                "ERA": 3.30,
                "WHIP": 1.10,
            }
        else:
            season[t] = dict(base)
    return season


# ── Tests ─────────────────────────────────────────────────────────────


class TestAdapterShapeContract:
    """The adapter output must match the dict shape both consumers ingest."""

    def test_all_12_categories_present_per_team(self):
        season = _make_season_totals()
        weekly = _adapter_scale(season)
        for team, cat_map in weekly.items():
            for cat in ALL_CATEGORIES:
                assert cat in cat_map, f"{team} missing category {cat} after adapter"

    def test_categories_match_league_config(self):
        """Adapter coverage must match LeagueConfig (single source of truth)."""
        cfg = LeagueConfig()
        expected = set(cfg.all_categories)
        weekly = _adapter_scale(_make_season_totals())
        for cat_map in weekly.values():
            assert set(cat_map.keys()) >= expected, "adapter dropped a configured category"

    def test_counting_stats_scaled_by_26(self):
        season = _make_season_totals()
        weekly = _adapter_scale(season)
        for team in season:
            for cat in _COUNTING:
                assert weekly[team][cat] == pytest.approx(season[team][cat] / 26.0)

    def test_rate_stats_pass_through_unchanged(self):
        season = _make_season_totals()
        weekly = _adapter_scale(season)
        for team in season:
            for cat in _RATE:
                assert weekly[team][cat] == pytest.approx(season[team][cat])

    def test_rate_stat_defaults_backfilled(self):
        """If a rate stat is missing from upstream, adapter fills league-avg."""
        partial_season = {
            "A": {
                "R": 780.0,
                "HR": 195.0,
                "RBI": 760.0,
                "SB": 100.0,
                "W": 70.0,
                "L": 70.0,
                "SV": 45.0,
                "K": 1280.0,
                # No rate stats provided
            }
        }
        weekly = _adapter_scale(partial_season)
        assert weekly["A"]["AVG"] == pytest.approx(0.250)
        assert weekly["A"]["OBP"] == pytest.approx(0.330)
        assert weekly["A"]["ERA"] == pytest.approx(4.00)
        assert weekly["A"]["WHIP"] == pytest.approx(1.25)


class TestWeeklyMagnitudeMatchesTau:
    """Counting stats divided by 26 should be in the magnitude ballpark of
    ``CALIBRATED_WEEKLY_TAU`` (the calibrated weekly SDs).

    Catches a class of bugs where someone "fixes" the scaling to ``/24`` or
    drops it entirely (counting stats become 100x the tau, producing
    nonsensical near-deterministic sim outcomes).
    """

    def test_counting_stat_means_within_order_of_magnitude_of_tau(self):
        weekly = _adapter_scale(_make_season_totals())
        a = weekly["A"]
        for cat in _COUNTING:
            tau = CALIBRATED_WEEKLY_TAU[cat]
            # Mean must be at least the same order of magnitude as tau (i.e.
            # the noise term doesn't completely swamp the signal).
            assert a[cat] / tau > 0.05, (
                f"{cat} weekly mean {a[cat]:.2f} is tiny vs tau {tau:.2f} (adapter probably dropped scaling)"
            )
            # And the mean shouldn't be 1000x the tau (would happen if the
            # adapter forgot to divide season totals by weeks).
            assert a[cat] / tau < 100.0, (
                f"{cat} weekly mean {a[cat]:.2f} dwarfs tau {tau:.2f} (adapter probably missing the /26 scaling)"
            )


class TestSimulateSeasonEnhancedConsumesAdapterOutput:
    """``simulate_season_enhanced`` must accept the adapter's dict and return
    the documented schema without raising KeyError on any category."""

    def _run(self, weekly_totals, n_sims=200, seed=42, current_week=10, playoff_spots=2):
        teams = list(weekly_totals.keys())
        current = {t: {"W": 5, "L": 5, "T": 0} for t in teams}
        # Round-robin schedule over 5 remaining weeks
        schedule = {w: [(teams[0], teams[1]), (teams[2], teams[3])] for w in range(current_week, current_week + 5)}
        return simulate_season_enhanced(
            current_standings=current,
            team_weekly_totals=weekly_totals,
            full_schedule=schedule,
            current_week=current_week,
            n_sims=n_sims,
            seed=seed,
            playoff_spots=playoff_spots,
        )

    def test_consumer_returns_documented_schema(self):
        weekly = _adapter_scale(_make_season_totals())
        result = self._run(weekly)
        assert set(result.keys()) == {
            "projected_records",
            "playoff_probability",
            "confidence_intervals",
            "strength_of_schedule",
        }
        for team in weekly:
            assert team in result["projected_records"]
            rec = result["projected_records"][team]
            assert {"W", "L", "T", "win_pct"}.issubset(rec.keys())

    def test_consumer_runs_without_keyerror(self):
        """Smoke test: every category must be readable from the dict."""
        weekly = _adapter_scale(_make_season_totals())
        # Should not raise KeyError, ValueError, or numpy errors.
        result = self._run(weekly, n_sims=50)
        assert result is not None

    def test_strong_team_wins_more_simulations(self):
        """Sanity check: the synthetic strong team (better counting + better
        rate stats) should beat the others more often than not. Confirms the
        consumer is actually using the adapter values rather than ignoring
        them."""
        weekly = _adapter_scale(_make_season_totals(strong_team="A"))
        result = self._run(weekly, n_sims=500)
        a_wins = result["projected_records"]["A"]["W"]
        for other in ("B", "C", "D"):
            other_wins = result["projected_records"][other]["W"]
            assert a_wins >= other_wins, (
                f"Strong team A only won {a_wins} vs {other}'s {other_wins}; "
                "adapter values may not be reaching the simulator correctly."
            )

    def test_inverse_category_semantics_preserved(self):
        """ERA/WHIP are inverse: lower is better. Adapter passes them through
        unchanged, so the consumer's inverse-mask logic must still see the
        strong team's lower ERA/WHIP as an advantage. If the adapter ever
        accidentally inverted/scaled the rate stats, the strong team's edge
        in pitching would vanish — caught here by the playoff_probability
        gap. Use playoff_spots=1 so the strong team isn't auto-clinching."""
        weekly = _adapter_scale(_make_season_totals(strong_team="A"))
        result = self._run(weekly, n_sims=500, playoff_spots=1)
        # Strong team should have noticeably higher playoff odds.
        a_prob = result["playoff_probability"]["A"]
        avg_other = sum(result["playoff_probability"][t] for t in "BCD") / 3.0
        assert a_prob > avg_other, (
            f"Strong team playoff prob {a_prob:.3f} not greater than average other prob {avg_other:.3f}"
        )


class TestComputeTeamStrengthProfilesConsumesAdapterOutput:
    """``compute_team_strength_profiles`` must accept the same dict shape and
    rank teams correctly per category."""

    def test_consumer_returns_one_profile_per_team(self):
        weekly = _adapter_scale(_make_season_totals())
        schedule = {1: [("A", "B"), ("C", "D")]}
        profiles = compute_team_strength_profiles(weekly, schedule, current_week=1)
        assert len(profiles) == len(weekly)
        for p in profiles:
            assert "team_name" in p
            assert "power_rating" in p
            assert "roster_quality" in p
            assert "category_balance" in p
            assert 0.0 <= p["power_rating"] <= 100.0

    def test_strong_team_ranks_first(self):
        """The synthetic strong team should top the power ranking. Validates
        that the per-category z-scoring inside the consumer is reading
        adapter values, not zeros."""
        weekly = _adapter_scale(_make_season_totals(strong_team="A"))
        schedule = {1: [("A", "B"), ("C", "D")]}
        profiles = compute_team_strength_profiles(weekly, schedule, current_week=1)
        ranked = sorted(profiles, key=lambda p: -p["power_rating"])
        assert ranked[0]["team_name"] == "A", f"Expected A on top, got {[p['team_name'] for p in ranked]}"

    def test_consumer_runs_without_schedule(self):
        """Schedule is optional in the signature (default None). Adapter
        callers sometimes pass None when full schedule isn't loaded yet."""
        weekly = _adapter_scale(_make_season_totals())
        profiles = compute_team_strength_profiles(weekly, full_schedule=None)
        assert len(profiles) == len(weekly)
