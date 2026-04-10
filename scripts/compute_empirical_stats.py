"""Compute empirical stat correlations and CVs from real MLB data.

Fetches 2022-2024 batting/pitching stats via pybaseball and computes
Spearman correlations and coefficients of variation to validate or
replace the hardcoded defaults in scenario_generator.py.

Usage:
    python scripts/compute_empirical_stats.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer.scenario_generator import (  # noqa: E402
    _RATE_STD,
    DEFAULT_CORRELATIONS,
    DEFAULT_CV,
)

# ── Constants ───────────────────────────────────────────────────────

YEARS = [2022, 2023, 2024]
HITTING_CATS: list[str] = ["r", "hr", "rbi", "sb", "avg", "obp"]
PITCHING_CATS: list[str] = ["w", "l", "sv", "k", "era", "whip"]
ALL_CATS: list[str] = HITTING_CATS + PITCHING_CATS
RATE_STATS: set[str] = {"avg", "obp", "era", "whip"}
DIVERGENCE_THRESHOLD = 0.20  # 20% relative divergence threshold

# pybaseball column name mapping
_BATTING_COL_MAP: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "OBP": "obp",
}
_PITCHING_COL_MAP: dict[str, str] = {
    "W": "w",
    "L": "l",
    "SV": "sv",
    "SO": "k",
    "ERA": "era",
    "WHIP": "whip",
}


# ── Data fetching ───────────────────────────────────────────────────


def _fetch_batting(qual: int = 50) -> pd.DataFrame:
    """Fetch 2022-2024 batting stats via pybaseball.

    Args:
        qual: Minimum plate appearances to qualify.

    Returns:
        DataFrame with columns: r, hr, rbi, sb, avg, obp, year.
    """
    try:
        from pybaseball import batting_stats
    except ImportError:
        print("ERROR: pybaseball not installed. Run: pip install pybaseball")
        return pd.DataFrame()

    all_frames: list[pd.DataFrame] = []
    for year in YEARS:
        try:
            print(f"  Fetching batting {year} (qual={qual})...")
            df = batting_stats(year, qual=qual)
            # Rename columns to lowercase fantasy names
            rename = {}
            for src, dst in _BATTING_COL_MAP.items():
                if src in df.columns:
                    rename[src] = dst
            df = df.rename(columns=rename)
            df["year"] = year
            keep = [c for c in ["r", "hr", "rbi", "sb", "avg", "obp", "year"] if c in df.columns]
            all_frames.append(df[keep])
            print(f"    {len(df)} hitters fetched")
            time.sleep(1.0)  # Be polite to FanGraphs
        except Exception as e:
            if "403" in str(e):
                print(f"    Batting {year}: HTTP 403 Forbidden (rate limited)")
            else:
                print(f"    Batting {year}: FAILED ({e})")

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _fetch_pitching(qual: int = 20) -> pd.DataFrame:
    """Fetch 2022-2024 pitching stats via pybaseball.

    Args:
        qual: Minimum innings pitched to qualify.

    Returns:
        DataFrame with columns: w, l, sv, k, era, whip, year.
    """
    try:
        from pybaseball import pitching_stats
    except ImportError:
        print("ERROR: pybaseball not installed. Run: pip install pybaseball")
        return pd.DataFrame()

    all_frames: list[pd.DataFrame] = []
    for year in YEARS:
        try:
            print(f"  Fetching pitching {year} (qual={qual})...")
            df = pitching_stats(year, qual=qual)
            # Rename columns to lowercase fantasy names
            rename = {}
            for src, dst in _PITCHING_COL_MAP.items():
                if src in df.columns:
                    rename[src] = dst
            df = df.rename(columns=rename)
            df["year"] = year
            keep = [c for c in ["w", "l", "sv", "k", "era", "whip", "year"] if c in df.columns]
            all_frames.append(df[keep])
            print(f"    {len(df)} pitchers fetched")
            time.sleep(1.0)
        except Exception as e:
            if "403" in str(e):
                print(f"    Pitching {year}: HTTP 403 Forbidden (rate limited)")
            else:
                print(f"    Pitching {year}: FAILED ({e})")

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def fetch_data(
    bat_qual: int = 50,
    pitch_qual: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 2022-2024 batting and pitching stats via pybaseball.

    Args:
        bat_qual: Minimum PA for hitters (default 50).
        pitch_qual: Minimum IP for pitchers (default 20).

    Returns:
        Tuple of (batting_df, pitching_df).
    """
    print("Fetching data from FanGraphs via pybaseball...")
    batting = _fetch_batting(qual=bat_qual)
    pitching = _fetch_pitching(qual=pitch_qual)
    return batting, pitching


# ── Computation ─────────────────────────────────────────────────────


def compute_spearman_correlations(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
) -> dict[str, float]:
    """Compute pairwise Spearman correlations for the 12 H2H categories.

    Within-domain correlations (hitting x hitting, pitching x pitching) are
    computed from real data. Cross-domain correlations (hitting x pitching)
    are set to 0.0 since hitters and pitchers are different player pools.

    Args:
        batting: DataFrame with columns r, hr, rbi, sb, avg, obp.
        pitching: DataFrame with columns w, l, sv, k, era, whip.

    Returns:
        Dict mapping "cat_a-cat_b" to Spearman rho. Includes all unique
        pairs from the 12 categories (upper triangle only).
    """
    results: dict[str, float] = {}

    # Hitting correlations (within-domain)
    bat_available = [c for c in HITTING_CATS if c in batting.columns]
    if len(bat_available) >= 2 and len(batting) >= 20:
        bat_df = batting[bat_available].apply(pd.to_numeric, errors="coerce").dropna()
        if len(bat_df) >= 20:
            for i, ca in enumerate(bat_available):
                for cb in bat_available[i + 1 :]:
                    rho, _ = spearmanr(bat_df[ca].values, bat_df[cb].values)
                    if np.isfinite(rho):
                        results[f"{ca}-{cb}"] = round(float(rho), 4)

    # Pitching correlations (within-domain)
    pit_available = [c for c in PITCHING_CATS if c in pitching.columns]
    if len(pit_available) >= 2 and len(pitching) >= 20:
        pit_df = pitching[pit_available].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pit_df) >= 20:
            for i, ca in enumerate(pit_available):
                for cb in pit_available[i + 1 :]:
                    rho, _ = spearmanr(pit_df[ca].values, pit_df[cb].values)
                    if np.isfinite(rho):
                        results[f"{ca}-{cb}"] = round(float(rho), 4)

    # Cross-domain correlations are 0.0 (different player pools)
    for hcat in HITTING_CATS:
        for pcat in PITCHING_CATS:
            results[f"{hcat}-{pcat}"] = 0.0

    return results


def compute_cvs(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
) -> dict[str, float]:
    """Compute CV for counting stats, absolute std for rate stats.

    Args:
        batting: DataFrame with hitting stat columns.
        pitching: DataFrame with pitching stat columns.

    Returns:
        Dict mapping stat name to CV (counting) or absolute std (rate).
    """
    results: dict[str, float] = {}

    # Batting CVs
    for col in HITTING_CATS:
        if col not in batting.columns:
            continue
        vals = pd.to_numeric(batting[col], errors="coerce").dropna()
        if len(vals) < 20:
            continue
        if col in RATE_STATS:
            results[col] = round(float(vals.std()), 4)
        else:
            mean_val = vals.mean()
            if abs(mean_val) > 1e-6:
                results[col] = round(float(vals.std() / mean_val), 4)

    # Pitching CVs
    for col in PITCHING_CATS:
        if col not in pitching.columns:
            continue
        vals = pd.to_numeric(pitching[col], errors="coerce").dropna()
        if len(vals) < 20:
            continue
        if col in RATE_STATS:
            results[col] = round(float(vals.std()), 4)
        else:
            mean_val = vals.mean()
            if abs(mean_val) > 1e-6:
                results[col] = round(float(vals.std() / mean_val), 4)

    return results


# ── Comparison ──────────────────────────────────────────────────────


def _pct_divergence(hardcoded: float, empirical: float) -> float:
    """Compute percentage divergence between two values.

    Uses the absolute value of the hardcoded as reference.
    Returns 0.0 if the hardcoded value is near zero.
    """
    if abs(hardcoded) < 1e-6:
        return abs(empirical) * 100.0 if abs(empirical) > 1e-6 else 0.0
    return abs(empirical - hardcoded) / abs(hardcoded)


def compare_correlations(empirical: dict[str, float]) -> list[dict]:
    """Print comparison table: hardcoded vs empirical correlations.

    Flags any value diverging >20% from the current default.
    """
    print("\n" + "=" * 80)
    print("CORRELATION COMPARISON: Hardcoded vs Empirical Spearman (2022-2024 MLB)")
    print("=" * 80)
    print(f"{'Stat Pair':<16} {'Hardcoded':>10} {'Empirical':>10} {'Delta':>8} {'% Div':>8} {'Flag':>8}")
    print("-" * 80)

    changes: list[dict] = []
    for (cat_a, cat_b), hardcoded in sorted(DEFAULT_CORRELATIONS.items()):
        key_fwd = f"{cat_a}-{cat_b}"
        key_rev = f"{cat_b}-{cat_a}"
        emp = empirical.get(key_fwd, empirical.get(key_rev))

        if emp is not None:
            delta = emp - hardcoded
            pct = _pct_divergence(hardcoded, emp)
            flagged = pct > DIVERGENCE_THRESHOLD
            flag = ">20%" if flagged else "ok"
            print(f"  {cat_a:>4}-{cat_b:<8} {hardcoded:>10.4f} {emp:>10.4f} {delta:>+8.4f} {pct:>7.1%} {flag:>8}")
            changes.append(
                {
                    "pair": f"{cat_a}-{cat_b}",
                    "hardcoded": hardcoded,
                    "empirical": emp,
                    "delta": round(delta, 4),
                    "pct_divergence": round(pct, 4),
                    "flagged": flagged,
                }
            )
        else:
            print(f"  {cat_a:>4}-{cat_b:<8} {hardcoded:>10.4f} {'N/A':>10}")

    return changes


def compare_cvs(empirical: dict[str, float]) -> list[dict]:
    """Print comparison table: hardcoded vs empirical CVs.

    Flags any value diverging >20% from the current default.
    """
    print("\n" + "=" * 80)
    print("CV COMPARISON: Hardcoded vs Empirical (2022-2024 MLB)")
    print("=" * 80)
    print(f"{'Stat':<8} {'Hardcoded':>10} {'Empirical':>10} {'Delta':>8} {'% Div':>8} {'Flag':>8} {'Type':>12}")
    print("-" * 80)

    changes: list[dict] = []
    for stat in ALL_CATS:
        emp = empirical.get(stat)
        stat_type = "abs_std" if stat in RATE_STATS else "CV"

        if stat in RATE_STATS:
            hardcoded = _RATE_STD.get(stat, DEFAULT_CV.get(stat, 0.0))
        else:
            hardcoded = DEFAULT_CV.get(stat, 0.0)

        if emp is not None:
            delta = emp - hardcoded
            pct = _pct_divergence(hardcoded, emp)
            flagged = pct > DIVERGENCE_THRESHOLD
            flag = ">20%" if flagged else "ok"
            print(f"  {stat:<6} {hardcoded:>10.4f} {emp:>10.4f} {delta:>+8.4f} {pct:>7.1%} {flag:>8} {stat_type:>12}")
            changes.append(
                {
                    "stat": stat,
                    "hardcoded": hardcoded,
                    "empirical": emp,
                    "delta": round(delta, 4),
                    "pct_divergence": round(pct, 4),
                    "flagged": flagged,
                    "type": stat_type,
                }
            )
        else:
            print(f"  {stat:<6} {hardcoded:>10.4f} {'N/A':>10} {'':>8} {'':>8} {'':>8} {stat_type:>12}")

    return changes


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Fetch data, compute stats, compare, and save results."""
    batting, pitching = fetch_data()

    if batting.empty and pitching.empty:
        print("\nERROR: No data fetched. Cannot compute empirical stats.")
        sys.exit(1)

    n_bat = len(batting)
    n_pit = len(pitching)
    print(f"\nTotal: {n_bat} hitter-seasons, {n_pit} pitcher-seasons")

    # Compute empirical values
    print("\nComputing Spearman correlations...")
    empirical_corr = compute_spearman_correlations(batting, pitching)
    within_domain = {k: v for k, v in empirical_corr.items() if v != 0.0}
    print(f"  {len(within_domain)} within-domain + {len(empirical_corr) - len(within_domain)} cross-domain (=0.0)")

    print("Computing CVs...")
    empirical_cvs = compute_cvs(batting, pitching)
    print(f"  {len(empirical_cvs)} CVs computed")

    # Compare to hardcoded values
    corr_changes = compare_correlations(empirical_corr)
    cv_changes = compare_cvs(empirical_cvs)

    # Summary
    corr_flagged = [c for c in corr_changes if c["flagged"]]
    cv_flagged = [c for c in cv_changes if c["flagged"]]
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Correlations: {len(corr_flagged)} of {len(corr_changes)} diverge >20% from defaults")
    print(f"  CVs: {len(cv_flagged)} of {len(cv_changes)} diverge >20% from defaults")

    if corr_flagged:
        print("\n  Flagged correlations:")
        for c in corr_flagged:
            print(f"    {c['pair']}: {c['hardcoded']:.4f} -> {c['empirical']:.4f} ({c['pct_divergence']:.1%})")

    if cv_flagged:
        print("\n  Flagged CVs:")
        for c in cv_flagged:
            print(f"    {c['stat']}: {c['hardcoded']:.4f} -> {c['empirical']:.4f} ({c['pct_divergence']:.1%})")

    # Save combined results to single JSON
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "empirical_stats.json"
    output = {
        "correlations": empirical_corr,
        "cvs": empirical_cvs,
        "correlation_divergences": [{k: v for k, v in c.items()} for c in corr_changes],
        "cv_divergences": [{k: v for k, v in c.items()} for c in cv_changes],
        "metadata": {
            "n_hitter_seasons": n_bat,
            "n_pitcher_seasons": n_pit,
            "years": YEARS,
            "batting_qual": "50 PA",
            "pitching_qual": "20 IP",
            "source": "FanGraphs via pybaseball",
            "method": "Spearman rank correlation",
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "note": "CV = std/mean for counting stats; absolute std for rate stats. "
            "Cross-domain correlations (hitting x pitching) are 0.0.",
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved combined results to {output_path}")

    # Also save legacy split files for backwards compatibility
    corr_path = data_dir / "empirical_correlations.json"
    with open(corr_path, "w") as f:
        json.dump(
            {
                "correlations": empirical_corr,
                "n_hitter_seasons": n_bat,
                "n_pitcher_seasons": n_pit,
                "years": YEARS,
                "batting_qual": "50 PA",
                "pitching_qual": "20 IP",
                "source": "FanGraphs via pybaseball",
            },
            f,
            indent=2,
        )

    cv_path = data_dir / "empirical_cvs.json"
    with open(cv_path, "w") as f:
        json.dump(
            {
                "cvs": empirical_cvs,
                "n_hitter_seasons": n_bat,
                "n_pitcher_seasons": n_pit,
                "years": YEARS,
                "batting_qual": "50 PA",
                "pitching_qual": "20 IP",
                "source": "FanGraphs via pybaseball",
                "note": "CV = std/mean for counting stats; absolute std for rate stats",
            },
            f,
            indent=2,
        )

    print(f"Saved legacy files to {corr_path} and {cv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
