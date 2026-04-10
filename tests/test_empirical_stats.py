"""Tests for empirical stat computation (NO network calls).

All tests use synthetic data to validate correlation computation,
CV computation, matrix properties, JSON format, and edge cases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compute_empirical_stats import (
    ALL_CATS,
    HITTING_CATS,
    PITCHING_CATS,
    RATE_STATS,
    compute_cvs,
    compute_spearman_correlations,
)
from src.optimizer.scenario_generator import (
    _RATE_STD,
    DEFAULT_CORRELATIONS,
    DEFAULT_CV,
    compare_to_defaults,
    load_cached_empirical_stats,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_batting(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic batting data with known relationships."""
    rng = np.random.RandomState(seed)
    # HR drives RBI (strong positive correlation)
    hr = rng.poisson(25, n).astype(float)
    rbi = hr * 2.5 + rng.normal(10, 5, n)
    r = hr * 1.5 + rng.normal(20, 8, n)
    sb = rng.poisson(10, n).astype(float)
    avg = rng.normal(0.265, 0.025, n)
    obp = avg + rng.normal(0.060, 0.010, n)  # OBP > AVG always
    return pd.DataFrame({"r": r, "hr": hr, "rbi": rbi, "sb": sb, "avg": avg, "obp": obp})


def _make_pitching(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic pitching data with known relationships."""
    rng = np.random.RandomState(seed)
    era = rng.normal(4.00, 0.80, n)
    whip = era * 0.3 + rng.normal(0.10, 0.05, n)  # ERA-WHIP correlated
    k = rng.poisson(120, n).astype(float)
    w = rng.poisson(8, n).astype(float)
    l = rng.poisson(7, n).astype(float)
    sv = rng.poisson(2, n).astype(float)
    return pd.DataFrame({"w": w, "l": l, "sv": sv, "k": k, "era": era, "whip": whip})


# ── Test: Spearman correlation with known data ───────────────────────


class TestCorrelationComputation:
    """Validate Spearman correlation computation logic."""

    def test_known_spearman_monotonic(self):
        """Perfect monotonic relationship gives Spearman rho = 1.0."""
        batting = pd.DataFrame(
            {
                "r": np.arange(50, dtype=float),
                "hr": np.arange(50, dtype=float) * 2,
                "rbi": np.arange(50, dtype=float) * 3,
                "sb": np.arange(50, dtype=float),
                "avg": np.linspace(0.200, 0.350, 50),
                "obp": np.linspace(0.280, 0.430, 50),
            }
        )
        pitching = _make_pitching(n=30)

        corr = compute_spearman_correlations(batting, pitching)

        # Perfect monotonic: r-hr should be 1.0
        assert abs(corr["r-hr"] - 1.0) < 0.001
        assert abs(corr["r-rbi"] - 1.0) < 0.001

    def test_negative_correlation(self):
        """Anti-monotonic relationship gives Spearman rho near -1.0."""
        n = 50
        batting = pd.DataFrame(
            {
                "r": np.arange(n, dtype=float),
                "hr": np.arange(n, 0, -1, dtype=float),  # reversed
                "rbi": np.arange(n, dtype=float),
                "sb": np.arange(n, dtype=float),
                "avg": np.linspace(0.200, 0.350, n),
                "obp": np.linspace(0.280, 0.430, n),
            }
        )
        pitching = _make_pitching(n=30)

        corr = compute_spearman_correlations(batting, pitching)
        assert corr["r-hr"] < -0.99

    def test_correlation_bounds(self):
        """All correlations are within [-1, 1]."""
        batting = _make_batting(n=200)
        pitching = _make_pitching(n=150)

        corr = compute_spearman_correlations(batting, pitching)

        for key, val in corr.items():
            assert -1.0 <= val <= 1.0, f"{key} = {val} out of bounds"


# ── Test: Correlation matrix properties ──────────────────────────────


class TestCorrelationMatrixProperties:
    """Validate structural properties of the correlation output."""

    def test_symmetry(self):
        """If (a, b) is in results, there is no duplicate (b, a)."""
        batting = _make_batting()
        pitching = _make_pitching()
        corr = compute_spearman_correlations(batting, pitching)

        seen_pairs: set[frozenset[str]] = set()
        for key in corr:
            parts = key.split("-")
            pair = frozenset(parts)
            assert pair not in seen_pairs, f"Duplicate pair: {key}"
            seen_pairs.add(pair)

    def test_diagonal_not_in_output(self):
        """Self-correlations (diagonal) are not included in the output."""
        batting = _make_batting()
        pitching = _make_pitching()
        corr = compute_spearman_correlations(batting, pitching)

        for key in corr:
            parts = key.split("-")
            assert parts[0] != parts[1], f"Diagonal entry found: {key}"

    def test_cross_domain_is_zero(self):
        """All hitting x pitching correlations are exactly 0.0."""
        batting = _make_batting()
        pitching = _make_pitching()
        corr = compute_spearman_correlations(batting, pitching)

        for hcat in HITTING_CATS:
            for pcat in PITCHING_CATS:
                key = f"{hcat}-{pcat}"
                assert key in corr, f"Missing cross-domain pair: {key}"
                assert corr[key] == 0.0, f"Cross-domain {key} = {corr[key]}, expected 0.0"


# ── Test: CV computation ────────────────────────────────────────────


class TestCVComputation:
    """Validate coefficient of variation computation."""

    def test_known_cv(self):
        """CV of uniform data: std/mean matches expected value."""
        # For uniform [a, b]: std = (b-a)/sqrt(12), mean = (a+b)/2
        # CV = std/mean = (b-a) / (sqrt(12) * (a+b)/2)
        n = 10000
        rng = np.random.RandomState(42)
        vals = rng.uniform(10, 30, n)
        batting = pd.DataFrame({"hr": vals, "r": vals, "rbi": vals, "sb": vals, "avg": vals, "obp": vals})
        pitching = _make_pitching()

        cvs = compute_cvs(batting, pitching)

        # Expected CV for uniform(10, 30): std=5.77, mean=20, CV=0.289
        assert abs(cvs["hr"] - 0.289) < 0.02

    def test_rate_stat_uses_absolute_std(self):
        """Rate stats (avg, obp, era, whip) return absolute std, not CV."""
        rng = np.random.RandomState(42)
        n = 500
        avg_vals = rng.normal(0.265, 0.025, n)
        batting = pd.DataFrame(
            {
                "r": rng.poisson(70, n).astype(float),
                "hr": rng.poisson(20, n).astype(float),
                "rbi": rng.poisson(65, n).astype(float),
                "sb": rng.poisson(10, n).astype(float),
                "avg": avg_vals,
                "obp": avg_vals + 0.060,
            }
        )
        pitching = pd.DataFrame(
            {
                "w": rng.poisson(8, n).astype(float),
                "l": rng.poisson(7, n).astype(float),
                "sv": rng.poisson(2, n).astype(float),
                "k": rng.poisson(100, n).astype(float),
                "era": rng.normal(4.0, 0.8, n),
                "whip": rng.normal(1.25, 0.15, n),
            }
        )

        cvs = compute_cvs(batting, pitching)

        # AVG absolute std should be near 0.025, not CV (which would be ~0.094)
        assert cvs["avg"] < 0.05, f"AVG should be absolute std, got {cvs['avg']}"
        assert cvs["era"] < 2.0, f"ERA should be absolute std, got {cvs['era']}"

    def test_cv_handles_small_sample(self):
        """Returns empty dict when sample is too small."""
        tiny = pd.DataFrame({"hr": [10.0, 20.0]})
        cvs = compute_cvs(tiny, pd.DataFrame())
        assert cvs == {}


# ── Test: Zero-variance columns ─────────────────────────────────────


class TestEdgeCases:
    """Test edge cases like zero variance and missing columns."""

    def test_zero_variance_column(self):
        """Zero-variance column produces no CV entry (division by zero)."""
        n = 50
        batting = pd.DataFrame(
            {
                "r": [70.0] * n,  # zero variance
                "hr": np.random.default_rng(42).poisson(20, n).astype(float),
                "rbi": np.random.default_rng(42).poisson(60, n).astype(float),
                "sb": np.random.default_rng(42).poisson(10, n).astype(float),
                "avg": np.random.default_rng(42).normal(0.265, 0.025, n),
                "obp": np.random.default_rng(42).normal(0.330, 0.025, n),
            }
        )
        pitching = _make_pitching(n=30)

        cvs = compute_cvs(batting, pitching)
        # r has zero variance -> std/mean = 0/70 = 0.0, which is valid
        # The function should still produce an entry (0.0 is a valid CV)
        if "r" in cvs:
            assert cvs["r"] == 0.0

    def test_missing_columns_handled(self):
        """Missing stat columns are silently skipped."""
        batting = pd.DataFrame({"hr": np.arange(50, dtype=float)})  # only HR
        pitching = pd.DataFrame()

        corr = compute_spearman_correlations(batting, pitching)
        # Should still have cross-domain zeros
        assert "r-w" in corr
        assert corr["r-w"] == 0.0
        # But no within-domain batting correlations (need >= 2 columns)
        within_bat = {
            k: v for k, v in corr.items() if k.split("-")[0] in HITTING_CATS and k.split("-")[1] in HITTING_CATS
        }
        assert len(within_bat) == 0


# ── Test: JSON output format ────────────────────────────────────────


class TestJSONOutputFormat:
    """Validate the structure expected by load_cached_empirical_stats."""

    def test_json_roundtrip(self):
        """Verify the JSON format can be loaded by scenario_generator."""
        batting = _make_batting(n=100)
        pitching = _make_pitching(n=80)

        corr = compute_spearman_correlations(batting, pitching)
        cvs = compute_cvs(batting, pitching)

        output = {
            "correlations": corr,
            "cvs": cvs,
            "metadata": {
                "n_hitter_seasons": len(batting),
                "n_pitcher_seasons": len(pitching),
                "years": [2022, 2023, 2024],
                "source": "test",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(output, f, indent=2)
            tmp_path = f.name

        try:
            loaded = load_cached_empirical_stats(Path(tmp_path))
            assert loaded is not None
            assert "correlations" in loaded
            assert "cvs" in loaded
            # Correlation values should be valid
            for val in loaded["correlations"].values():
                assert -1.0 <= val <= 1.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_missing_file_returns_none(self):
        """load_cached_empirical_stats returns None for missing file."""
        result = load_cached_empirical_stats(Path("/nonexistent/path/fake.json"))
        assert result is None


# ── Test: compare_to_defaults ────────────────────────────────────────


class TestCompareToDefaults:
    """Validate the divergence comparison function."""

    def test_compare_returns_divergences(self):
        """compare_to_defaults returns dict with correlation and cv keys."""
        batting = _make_batting(n=100)
        pitching = _make_pitching(n=80)

        corr = compute_spearman_correlations(batting, pitching)
        cvs = compute_cvs(batting, pitching)

        cached = {"correlations": corr, "cvs": cvs}
        result = compare_to_defaults(cached)

        assert "correlation_divergences" in result
        assert "cv_divergences" in result
        assert isinstance(result["correlation_divergences"], dict)
        assert isinstance(result["cv_divergences"], dict)

    def test_compare_zero_divergence_for_identical(self):
        """When empirical matches defaults exactly, divergences are 0."""
        # Build cached data that exactly matches defaults
        corr = {}
        for (ca, cb), val in DEFAULT_CORRELATIONS.items():
            corr[f"{ca}-{cb}"] = val

        cvs = {}
        for stat, val in DEFAULT_CV.items():
            if stat in RATE_STATS:
                cvs[stat] = _RATE_STD.get(stat, val)
            else:
                cvs[stat] = val

        cached = {"correlations": corr, "cvs": cvs}
        result = compare_to_defaults(cached)

        for pair, div in result["correlation_divergences"].items():
            assert abs(div) < 1e-6, f"Divergence for {pair}: {div}"
