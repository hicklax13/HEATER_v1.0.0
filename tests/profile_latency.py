"""Performance profiling script for the recommendation pipeline.

Measures end-to-end recommendation latency and verifies it stays under
the 15-second threshold required for live draft usage.

Usage:
    python tests/profile_latency.py
"""

import sys
import os
import time
import statistics

# Ensure the project root is on the path so src.* imports resolve.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.database import init_db, load_player_pool
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    value_all_players,
    compute_replacement_levels,
    compute_category_weights,
)
from src.draft_state import DraftState
from src.simulation import DraftSimulator

LATENCY_THRESHOLD_SECONDS = 15.0
NUM_RUNS = 10
URGENCY_CANDIDATES = 20


def build_mid_draft_scenario(player_pool):
    """Create a DraftState and simulate picks to reach a mid-draft scenario.

    Simulates roughly 4 complete rounds (48 picks in a 12-team league) so
    that positional scarcity, roster totals, and opponent behavior are all
    non-trivial when the profiling run begins.
    """
    draft = DraftState(num_teams=12, num_rounds=23, user_team_index=0)

    # Use ADP ordering to simulate realistic early-round picks.
    sorted_pool = player_pool.sort_values("adp").reset_index(drop=True)
    picks_to_simulate = min(48, len(sorted_pool))  # ~4 rounds

    for i in range(picks_to_simulate):
        if i >= len(sorted_pool):
            break
        row = sorted_pool.iloc[i]
        draft.make_pick(
            player_id=int(row["player_id"]),
            player_name=str(row["name"]),
            positions=str(row["positions"]),
        )

    return draft


def run_recommendation_pipeline(player_pool, draft, config):
    """Execute the full recommendation pipeline once and return elapsed time."""
    start = time.perf_counter()

    # 1. Filter to available players
    available = draft.available_players(player_pool)

    # 2. Compute roster totals for the user's team
    roster_totals = draft.get_user_roster_totals(player_pool)

    # 3. Compute category weights
    cat_weights = compute_category_weights(roster_totals, config)

    # 4. Compute replacement levels
    sgp_calc = SGPCalculator(config)
    replacement_levels = compute_replacement_levels(available, config, sgp_calc)

    # 5. Value all available players
    valued = value_all_players(
        available, config, roster_totals, cat_weights, replacement_levels
    )

    # 6. Compute urgency for the top N candidates
    simulator = DraftSimulator(config, sigma=10.0)
    current_pick = draft.current_pick
    next_user_pick = draft.next_user_pick()
    if next_user_pick is None or next_user_pick <= current_pick:
        next_user_pick = current_pick + draft.num_teams

    top_candidates = valued.head(URGENCY_CANDIDATES)
    for _, candidate in top_candidates.iterrows():
        simulator.compute_urgency(candidate, valued, current_pick, next_user_pick)

    elapsed = time.perf_counter() - start
    return elapsed


def main():
    print("=" * 60)
    print("  Recommendation Pipeline Latency Profiler")
    print("=" * 60)

    # --- Setup -----------------------------------------------------------
    print("\n[setup] Initializing database...")
    init_db()

    print("[setup] Loading player pool...")
    player_pool = load_player_pool()

    if player_pool.empty:
        print(
            "\nERROR: Player pool is empty. Import projection data before "
            "running the profiler.\n"
            "  Example:\n"
            "    from src.database import import_hitter_csv, import_pitcher_csv, "
            "create_blended_projections\n"
            "    import_hitter_csv('data/steamer_hitters.csv', 'steamer')\n"
            "    import_pitcher_csv('data/steamer_pitchers.csv', 'steamer')\n"
            "    create_blended_projections()"
        )
        sys.exit(1)

    print(f"[setup] Loaded {len(player_pool)} players.")

    config = LeagueConfig()

    print("[setup] Building mid-draft scenario (~48 picks)...")
    draft = build_mid_draft_scenario(player_pool)
    print(
        f"[setup] Draft at pick {draft.current_pick}, "
        f"round {draft.current_round}. "
        f"User roster has {len(draft.user_team.picks)} players."
    )

    available_count = len(draft.available_players(player_pool))
    print(f"[setup] {available_count} players still available.\n")

    # --- Profiling -------------------------------------------------------
    print(f"Running {NUM_RUNS} pipeline iterations...\n")
    latencies = []

    for i in range(NUM_RUNS):
        elapsed = run_recommendation_pipeline(player_pool, draft, config)
        latencies.append(elapsed)
        status = "OK" if elapsed < LATENCY_THRESHOLD_SECONDS else "SLOW"
        print(f"  Run {i + 1:>2}/{NUM_RUNS}:  {elapsed:7.3f}s  [{status}]")

    # --- Report ----------------------------------------------------------
    lat_min = min(latencies)
    lat_max = max(latencies)
    lat_avg = statistics.mean(latencies)
    lat_med = statistics.median(latencies)
    lat_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    print("\n" + "-" * 60)
    print("  Latency Summary")
    print("-" * 60)
    print(f"  Min:      {lat_min:.3f}s")
    print(f"  Avg:      {lat_avg:.3f}s")
    print(f"  Median:   {lat_med:.3f}s")
    print(f"  Max:      {lat_max:.3f}s")
    print(f"  Std Dev:  {lat_std:.3f}s")
    print(f"  Threshold: {LATENCY_THRESHOLD_SECONDS:.1f}s")
    print("-" * 60)

    passed = lat_max < LATENCY_THRESHOLD_SECONDS
    if passed:
        print(f"\n  PASS  -- All {NUM_RUNS} runs completed under "
              f"{LATENCY_THRESHOLD_SECONDS:.0f}s (worst: {lat_max:.3f}s)")
    else:
        slow_count = sum(1 for t in latencies if t >= LATENCY_THRESHOLD_SECONDS)
        print(f"\n  FAIL  -- {slow_count}/{NUM_RUNS} runs exceeded "
              f"{LATENCY_THRESHOLD_SECONDS:.0f}s (worst: {lat_max:.3f}s)")

    print()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
