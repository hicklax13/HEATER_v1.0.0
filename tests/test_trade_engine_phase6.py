"""Tests for Trade Engine Phase 6 — Full Production.

Tests cover:
  - Convergence diagnostics: ESS, split-R̂, running mean stability
  - Cache: TTL, get/set, invalidation, get_or_compute, singleton
  - Simulation config: adaptive n_sims, mode selection, time budgets
"""

import time
import unittest

import numpy as np


class TestConvergenceDiagnostics(unittest.TestCase):
    """Test MC convergence diagnostics."""

    def test_ess_independent_samples(self):
        """Independent samples should have ESS near N."""
        from src.engine.production.convergence import effective_sample_size

        rng = np.random.RandomState(42)
        samples = rng.normal(0, 1, 10000)
        ess = effective_sample_size(samples)
        # ESS should be close to N for independent samples
        assert ess > 2000  # Geyer pair-sum is conservative; well above correlated ESS

    def test_ess_correlated_samples_lower(self):
        """Correlated samples should have lower ESS."""
        from src.engine.production.convergence import effective_sample_size

        # Create correlated samples (random walk)
        rng = np.random.RandomState(42)
        samples = np.cumsum(rng.normal(0, 0.1, 1000))
        ess = effective_sample_size(samples)
        # ESS should be much less than N for correlated data
        assert ess < 500

    def test_ess_constant_series(self):
        """Constant series should have ESS = N."""
        from src.engine.production.convergence import effective_sample_size

        samples = np.ones(100)
        ess = effective_sample_size(samples)
        assert ess == 100.0

    def test_ess_short_series(self):
        """Very short series should return len."""
        from src.engine.production.convergence import effective_sample_size

        samples = np.array([1.0, 2.0, 3.0])
        ess = effective_sample_size(samples)
        assert ess == 3.0

    def test_split_rhat_converged(self):
        """IID Normal samples should have R̂ ≈ 1.0."""
        from src.engine.production.convergence import split_rhat

        rng = np.random.RandomState(42)
        samples = rng.normal(0, 1, 10000)
        rhat = split_rhat(samples)
        assert abs(rhat - 1.0) < 0.05

    def test_split_rhat_diverged(self):
        """Drifting chain should have R̂ > 1.0."""
        from src.engine.production.convergence import split_rhat

        # First half from N(0,1), second half from N(5,1) — clear divergence
        rng = np.random.RandomState(42)
        chain1 = rng.normal(0, 1, 500)
        chain2 = rng.normal(5, 1, 500)
        samples = np.concatenate([chain1, chain2])
        rhat = split_rhat(samples)
        assert rhat > 1.05

    def test_split_rhat_short_series(self):
        """Very short series should return 1.0 (not enough data)."""
        from src.engine.production.convergence import split_rhat

        samples = np.array([1.0, 2.0, 3.0])
        assert split_rhat(samples) == 1.0

    def test_running_mean_stable(self):
        """Converged chain should have stable running mean."""
        from src.engine.production.convergence import running_mean_stability

        rng = np.random.RandomState(42)
        samples = rng.normal(1.0, 0.5, 5000)
        stability = running_mean_stability(samples)
        assert stability < 0.05  # Very stable

    def test_running_mean_unstable(self):
        """Drifting chain should have unstable running mean."""
        from src.engine.production.convergence import running_mean_stability

        # Linear trend — running mean never stabilizes
        samples = np.linspace(0, 10, 1000)
        stability = running_mean_stability(samples)
        assert stability > 0.01  # Unstable

    def test_check_convergence_has_all_keys(self):
        """check_convergence should return complete report."""
        from src.engine.production.convergence import check_convergence

        rng = np.random.RandomState(42)
        samples = rng.normal(0, 1, 5000)
        report = check_convergence(samples)

        assert "ess" in report
        assert "ess_ok" in report
        assert "rhat" in report
        assert "rhat_ok" in report
        assert "stability" in report
        assert "stability_ok" in report
        assert "n_samples" in report
        assert "converged" in report
        assert "quality" in report
        assert report["quality"] in ("excellent", "good", "marginal", "poor")

    def test_check_convergence_good_samples(self):
        """Well-behaved IID samples should be 'good' or 'excellent'."""
        from src.engine.production.convergence import check_convergence

        rng = np.random.RandomState(42)
        samples = rng.normal(0, 1, 10000)
        report = check_convergence(samples)
        assert report["converged"] is True
        assert report["quality"] in ("excellent", "good")

    def test_recommend_n_sims_scales_up(self):
        """Low ESS should recommend more sims."""
        from src.engine.production.convergence import recommend_n_sims

        recommended = recommend_n_sims(current_ess=50, target_ess=500, current_n=10000)
        assert recommended > 10000
        assert recommended <= 100000

    def test_recommend_n_sims_caps_at_100k(self):
        """Recommendation should never exceed 100K."""
        from src.engine.production.convergence import recommend_n_sims

        recommended = recommend_n_sims(current_ess=1, target_ess=10000, current_n=50000)
        assert recommended == 100000


class TestCache(unittest.TestCase):
    """Test precomputation cache."""

    def test_set_and_get(self):
        """Basic set/get should work."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        cache.set("test_key", {"value": 42}, ttl=60)
        assert cache.get("test_key") == {"value": 42}

    def test_missing_key_returns_none(self):
        """Missing key should return None."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        assert cache.get("nonexistent") is None

    def test_stale_entry_returns_none(self):
        """Expired entry should return None."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        cache.set("test_key", "value", ttl=0)  # TTL=0 → immediately stale
        # Allow a tiny bit of time to pass
        time.sleep(0.01)
        assert cache.get("test_key") is None

    def test_invalidate(self):
        """Invalidation should remove the entry."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        cache.set("key1", "val1", ttl=3600)
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.invalidate("key1") is False  # Already gone

    def test_clear(self):
        """Clear should remove all entries."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        cache.set("a", 1, ttl=3600)
        cache.set("b", 2, ttl=3600)
        assert cache.clear() == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_get_or_compute(self):
        """get_or_compute should compute on miss, cache on hit."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        call_count = 0

        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        # First call: computes
        v1 = cache.get_or_compute("key", expensive_fn, ttl=3600)
        assert v1 == "computed_value"
        assert call_count == 1

        # Second call: cached
        v2 = cache.get_or_compute("key", expensive_fn, ttl=3600)
        assert v2 == "computed_value"
        assert call_count == 1  # Not called again

    def test_stats(self):
        """Stats should track hits and misses."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        cache.set("key", "val", ttl=3600)
        cache.get("key")  # Hit
        cache.get("key")  # Hit
        cache.get("miss")  # Miss

        stats = cache.stats()
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] > 60.0

    def test_global_cache_singleton(self):
        """Global cache should be a singleton."""
        from src.engine.production.cache import get_trade_cache, reset_trade_cache

        reset_trade_cache()
        c1 = get_trade_cache()
        c2 = get_trade_cache()
        assert c1 is c2
        reset_trade_cache()

    def test_reset_trade_cache(self):
        """Reset should create a new cache instance."""
        from src.engine.production.cache import get_trade_cache, reset_trade_cache

        c1 = get_trade_cache()
        c1.set("test", "val", ttl=3600)
        reset_trade_cache()
        c2 = get_trade_cache()
        assert c2.get("test") is None  # New instance, old data gone


class TestSimConfig(unittest.TestCase):
    """Test simulation scaling configuration."""

    def test_quick_mode(self):
        """Quick mode should return 1K sims."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        assert compute_adaptive_n_sims(mode="quick") == 1000

    def test_full_mode(self):
        """Full mode should return 100K sims."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        assert compute_adaptive_n_sims(mode="full") == 100000

    def test_standard_baseline(self):
        """Standard 1-for-1 trade should use 10K sims."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        n = compute_adaptive_n_sims(n_giving=1, n_receiving=1, mode="standard")
        assert n == 10000

    def test_complex_trade_more_sims(self):
        """Complex trade (3-for-2) should use more sims than simple."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        simple = compute_adaptive_n_sims(n_giving=1, n_receiving=1, mode="standard")
        complex_trade = compute_adaptive_n_sims(n_giving=3, n_receiving=2, mode="standard")
        assert complex_trade > simple

    def test_time_budget_caps_sims(self):
        """Time budget should cap simulation count."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        uncapped = compute_adaptive_n_sims(n_giving=3, n_receiving=3, mode="production")
        capped = compute_adaptive_n_sims(n_giving=3, n_receiving=3, mode="production", time_budget_s=1.0)
        assert capped <= uncapped

    def test_never_below_minimum(self):
        """Should never go below 1K sims even with tiny time budget."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        n = compute_adaptive_n_sims(mode="standard", time_budget_s=0.001)
        assert n >= 1000

    def test_never_above_100k(self):
        """Should never exceed 100K sims."""
        from src.engine.production.sim_config import compute_adaptive_n_sims

        n = compute_adaptive_n_sims(n_giving=10, n_receiving=10, mode="production")
        assert n <= 100000

    def test_get_sim_mode(self):
        """Mode selection based on context."""
        from src.engine.production.sim_config import get_sim_mode

        assert get_sim_mode(interactive=True) == "standard"
        assert get_sim_mode(interactive=False) == "production"

    def test_estimate_runtime(self):
        """Runtime estimate should be positive and proportional."""
        from src.engine.production.sim_config import estimate_runtime_seconds

        t10k = estimate_runtime_seconds(10000)
        t100k = estimate_runtime_seconds(100000)
        assert t10k > 0
        assert t100k > t10k
        assert abs(t100k / t10k - 10.0) < 0.1  # Linear scaling

    def test_sim_config_summary(self):
        """Summary should include all config keys."""
        from src.engine.production.sim_config import sim_config_summary

        summary = sim_config_summary(n_giving=2, n_receiving=1, mode="standard")
        assert "n_sims" in summary
        assert "mode" in summary
        assert "estimated_runtime_s" in summary
        assert "complexity" in summary
        assert "capped_by_time" in summary
        assert summary["mode"] == "standard"


if __name__ == "__main__":
    unittest.main()
