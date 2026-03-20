"""Monte Carlo convergence diagnostics.

Spec reference: Section 17 Phase 6 item 26

Detects whether a Monte Carlo simulation has converged (enough sims
to trust the output). Uses two industry-standard diagnostics:

1. Effective Sample Size (ESS): accounts for autocorrelation in the
   surplus samples. ESS < 100 suggests unreliable estimates.

2. Running Mean Stability: checks if the running mean of surplus has
   stabilized (std of last 1000 running means < threshold).

3. Split-R̂ (Gelman-Rubin inspired): splits the chain in half and
   checks if both halves agree. R̂ > 1.05 suggests non-convergence.

Wires into:
  - src/engine/monte_carlo/trade_simulator.py: post-simulation diagnostics
  - src/engine/output/trade_evaluator.py: convergence quality flag
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Convergence thresholds
MIN_EFFECTIVE_SAMPLE_SIZE: int = 100
MAX_RHAT: float = 1.05
RUNNING_MEAN_WINDOW: int = 500
RUNNING_MEAN_STABILITY_THRESHOLD: float = 0.01


def effective_sample_size(samples: np.ndarray) -> float:
    """Compute effective sample size accounting for autocorrelation.

    ESS = N / (1 + 2 * sum of autocorrelations at each lag).
    When samples are independent, ESS ≈ N. When highly correlated,
    ESS << N, meaning fewer "effective" data points than actual samples.

    Args:
        samples: 1-D array of MC surplus samples.

    Returns:
        Effective sample size (float). Higher = better.
    """
    n = len(samples)
    if n < 10:
        return float(n)

    # Compute autocorrelation via FFT (fast for large N)
    mean = np.mean(samples)
    var = np.var(samples)

    if var < 1e-12:
        return float(n)  # Constant series — no autocorrelation

    centered = samples - mean
    # Use FFT for autocorrelation
    fft_result = np.fft.fft(centered, n=2 * n)
    acf_full = np.fft.ifft(fft_result * np.conj(fft_result)).real[:n]
    acf = acf_full / acf_full[0]  # Normalize

    # Geyer's initial positive sequence: sum ACF in consecutive pairs
    max_lag = n // 2
    tau = 0.0
    k = 0
    while 2 * k + 1 < max_lag:
        pair_sum = acf[2 * k] + acf[2 * k + 1]
        if pair_sum < 0:
            break
        tau += pair_sum
        k += 1
    ess = n / (1.0 + 2.0 * tau)
    return max(1.0, ess)


def split_rhat(samples: np.ndarray) -> float:
    """Compute split-R̂ diagnostic for convergence.

    Splits the sample chain in half and compares the variance within
    each half to the variance between halves. R̂ near 1.0 indicates
    convergence; R̂ > 1.05 suggests the chain hasn't stabilized.

    This is a simplified version of the Gelman-Rubin diagnostic,
    applied to a single chain by splitting it.

    Args:
        samples: 1-D array of MC surplus samples.

    Returns:
        R̂ value. Values near 1.0 indicate convergence.
    """
    n = len(samples)
    if n < 20:
        return 1.0  # Not enough data to assess

    mid = n // 2
    n1 = mid
    n2 = n - mid
    chain1 = samples[:n1]
    chain2 = samples[n1:]

    # Within-chain variance
    w = (np.var(chain1, ddof=1) + np.var(chain2, ddof=1)) / 2.0

    # Between-chain variance (weighted average for unequal chain lengths)
    mean1 = np.mean(chain1)
    mean2 = np.mean(chain2)
    overall_mean = (mean1 * n1 + mean2 * n2) / (n1 + n2)
    b = n1 * (mean1 - overall_mean) ** 2 + n2 * (mean2 - overall_mean) ** 2

    if w < 1e-12:
        return 1.0  # No variance — trivially converged

    # Pooled variance estimate
    n_avg = (n1 + n2) / 2.0
    var_hat = ((n_avg - 1) / n_avg) * w + (1.0 / n_avg) * b

    rhat = np.sqrt(var_hat / w)
    return float(rhat)


def running_mean_stability(
    samples: np.ndarray,
    window: int = RUNNING_MEAN_WINDOW,
) -> float:
    """Measure stability of the running mean over the last `window` samples.

    If the running mean is still drifting, the simulation hasn't converged.
    Returns the drift of the running mean relative to the sample std dev.
    Normalizing by sample std (not mean) avoids numerical issues when
    the true mean is near zero.

    Args:
        samples: 1-D array of MC surplus samples.
        window: Number of tail samples to evaluate.

    Returns:
        Stability metric (lower = more stable). < 0.01 is excellent.
    """
    n = len(samples)
    if n < window:
        window = n

    tail = samples[-window:]
    running_means = np.cumsum(tail) / np.arange(1, len(tail) + 1)

    if len(running_means) < 10:
        return 1.0

    # Use the last half of running means (initial values are noisy)
    stable_portion = running_means[len(running_means) // 2 :]

    # Normalize by the overall sample std dev (not the mean)
    # This avoids division-by-near-zero for zero-mean distributions
    sample_std = np.std(samples)
    if sample_std < 1e-12:
        return 0.0  # Constant series — perfectly stable

    return float(np.std(stable_portion) / sample_std)


def check_convergence(
    samples: np.ndarray,
) -> dict[str, float | bool | str]:
    """Run all convergence diagnostics on MC surplus samples.

    Produces a convergence report with pass/fail for each metric
    and an overall quality assessment.

    Args:
        samples: 1-D array of MC surplus samples.

    Returns:
        Dict with:
          - ess: effective sample size
          - ess_ok: True if ESS >= MIN_EFFECTIVE_SAMPLE_SIZE
          - rhat: split R̂ value
          - rhat_ok: True if R̂ <= MAX_RHAT
          - stability: running mean CV
          - stability_ok: True if CV < threshold
          - n_samples: actual sample count
          - converged: overall convergence (all checks pass)
          - quality: "excellent", "good", "marginal", or "poor"
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)

    ess = effective_sample_size(samples)
    rhat = split_rhat(samples)
    stability = running_mean_stability(samples)

    ess_ok = ess >= MIN_EFFECTIVE_SAMPLE_SIZE
    rhat_ok = rhat <= MAX_RHAT
    stability_ok = stability < RUNNING_MEAN_STABILITY_THRESHOLD

    converged = ess_ok and rhat_ok and stability_ok

    # Quality classification
    if converged and ess > 1000:
        quality = "excellent"
    elif converged:
        quality = "good"
    elif ess_ok or rhat_ok:
        quality = "marginal"
    else:
        quality = "poor"

    return {
        "ess": round(ess, 1),
        "ess_ok": ess_ok,
        "rhat": round(rhat, 4),
        "rhat_ok": rhat_ok,
        "stability": round(stability, 6),
        "stability_ok": stability_ok,
        "n_samples": n,
        "converged": converged,
        "quality": quality,
    }


def recommend_n_sims(
    current_ess: float,
    target_ess: int = 500,
    current_n: int = 10_000,
) -> int:
    """Recommend number of simulations to achieve target ESS.

    If current ESS is 200 from 10K sims, we need roughly
    (target / current) * n_current = 25K sims.

    Args:
        current_ess: Current effective sample size.
        target_ess: Desired ESS.
        current_n: Current number of simulations.

    Returns:
        Recommended number of simulations. Capped at 100K.
    """
    if current_ess <= 0:
        return 100_000

    ratio = target_ess / current_ess
    recommended = int(current_n * ratio)

    return min(max(recommended, current_n), 100_000)
