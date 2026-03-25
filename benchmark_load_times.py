"""HEATER Benchmark Script -- measures wall-clock time for key operations.

Imports each module, runs its primary function multiple times, and reports the
median execution time in milliseconds, sorted slowest-to-fastest.

Usage:
    python benchmark_load_times.py
"""

import os
import statistics
import sys
import time
import traceback

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Number of repetitions per benchmark (default)
N_RUNS = 3


def time_fn(fn, *args, n_runs=N_RUNS, **kwargs):
    """Run fn n_runs times, return list of elapsed times in ms."""
    times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
    return times, result


def classify(ms):
    """Classify timing into speed bucket."""
    if ms < 100:
        return "Fast"
    elif ms < 500:
        return "OK"
    elif ms < 2000:
        return "Slow"
    else:
        return "Very Slow"


def main():
    results = []  # list of (name, median_ms, classification)
    errors = []  # list of (name, error_message)

    # ── 0. Initialize the database ───────────────────────────────────
    print("Initializing database...")
    try:
        from src.database import init_db

        init_db()
        print("  Database initialized.\n")
    except Exception as e:
        print(f"  FATAL: Could not init DB: {e}")
        sys.exit(1)

    # ── 1. load_player_pool() ────────────────────────────────────────
    pool = None
    print("Benchmarking: load_player_pool()")
    try:
        from src.database import load_player_pool

        times, pool = time_fn(load_player_pool)
        med = statistics.median(times)
        results.append(("load_player_pool()", med, classify(med)))
        print(f"  -> {med:.1f} ms (pool size: {len(pool)} rows)\n")
    except Exception as e:
        errors.append(("load_player_pool()", str(e)))
        traceback.print_exc()
        print()

    if pool is None or pool.empty:
        print("FATAL: No player pool loaded. Cannot continue benchmarks.")
        sys.exit(1)

    # ── 2. value_all_players() ───────────────────────────────────────
    print("Benchmarking: value_all_players()")
    try:
        from src.valuation import LeagueConfig, value_all_players

        config = LeagueConfig()
        times, valued_pool = time_fn(value_all_players, pool, config)
        med = statistics.median(times)
        results.append(("value_all_players()", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("value_all_players()", str(e)))
        traceback.print_exc()
        print()

    # ── 3. compute_replacement_levels() ──────────────────────────────
    print("Benchmarking: compute_replacement_levels()")
    try:
        from src.valuation import LeagueConfig, SGPCalculator, compute_replacement_levels

        config = LeagueConfig()
        sgp_calc = SGPCalculator(config)
        times, rep_levels = time_fn(compute_replacement_levels, pool, config, sgp_calc)
        med = statistics.median(times)
        results.append(("compute_replacement_levels()", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("compute_replacement_levels()", str(e)))
        traceback.print_exc()
        print()

    # ── 4. compute_sgp_denominators() ────────────────────────────────
    print("Benchmarking: compute_sgp_denominators()")
    try:
        from src.valuation import LeagueConfig, compute_sgp_denominators

        config = LeagueConfig()
        times, _ = time_fn(compute_sgp_denominators, pool, config)
        med = statistics.median(times)
        results.append(("compute_sgp_denominators()", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("compute_sgp_denominators()", str(e)))
        traceback.print_exc()
        print()

    # ── 5. DraftRecommendationEngine (quick mode) ────────────────────
    # Uses n_runs=1 because MC simulations on 9k players are very expensive
    print("Benchmarking: DraftRecommendationEngine.recommend() [quick, 50 sims]")
    try:
        from src.draft_engine import DraftRecommendationEngine
        from src.draft_state import DraftState
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        engine_quick = DraftRecommendationEngine(config, mode="quick")
        ds = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
        times, recs_quick = time_fn(
            engine_quick.recommend,
            pool,
            ds,
            top_n=8,
            n_simulations=50,
            n_runs=1,
        )
        med = statistics.median(times)
        results.append(("DraftEngine.recommend() [quick]", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(recs_quick)} recommendations)\n")
    except Exception as e:
        errors.append(("DraftEngine.recommend() [quick]", str(e)))
        traceback.print_exc()
        print()

    # ── 6. DraftRecommendationEngine (standard mode) ─────────────────
    print("Benchmarking: DraftRecommendationEngine.recommend() [standard, 100 sims]")
    try:
        from src.draft_engine import DraftRecommendationEngine
        from src.draft_state import DraftState
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        engine_std = DraftRecommendationEngine(config, mode="standard")
        ds = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
        times, recs_std = time_fn(
            engine_std.recommend,
            pool,
            ds,
            top_n=8,
            n_simulations=100,
            n_runs=1,
        )
        med = statistics.median(times)
        results.append(("DraftEngine.recommend() [standard]", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(recs_std)} recommendations)\n")
    except Exception as e:
        errors.append(("DraftEngine.recommend() [standard]", str(e)))
        traceback.print_exc()
        print()

    # ── 7. evaluate_trade() ──────────────────────────────────────────
    print("Benchmarking: evaluate_trade() [Phase 1 deterministic]")
    try:
        from src.engine.output.trade_evaluator import evaluate_trade
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Pick realistic player IDs from the pool
        hitters = pool[pool["is_hitter"] == 1].head(20)
        pitchers = pool[pool["is_hitter"] == 0].head(20)
        if len(hitters) >= 5 and len(pitchers) >= 5:
            # Build a fake roster of 23 players
            user_roster = list(hitters.head(12)["player_id"]) + list(pitchers.head(11)["player_id"])
            giving = [user_roster[0]]  # give 1 player
            receiving = [int(hitters.iloc[15]["player_id"])]  # receive 1 player not on roster

            times, trade_result = time_fn(
                evaluate_trade,
                giving,
                receiving,
                user_roster,
                pool,
                config=config,
                enable_mc=False,
                enable_context=False,
                enable_game_theory=False,
            )
            med = statistics.median(times)
            results.append(("evaluate_trade() [Phase 1]", med, classify(med)))
            grade = trade_result.get("grade", "N/A") if isinstance(trade_result, dict) else "N/A"
            print(f"  -> {med:.1f} ms (grade: {grade})\n")
        else:
            errors.append(("evaluate_trade() [Phase 1]", "Not enough players in pool"))
            print("  -> Skipped: not enough players\n")
    except Exception as e:
        errors.append(("evaluate_trade() [Phase 1]", str(e)))
        traceback.print_exc()
        print()

    # ── 8. compare_players() ─────────────────────────────────────────
    print("Benchmarking: compare_players()")
    try:
        from src.in_season import compare_players
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        # Pick two hitters to compare
        hitters = pool[pool["is_hitter"] == 1].head(5)
        if len(hitters) >= 2:
            pid_a = int(hitters.iloc[0]["player_id"])
            pid_b = int(hitters.iloc[1]["player_id"])
            times, cmp_result = time_fn(compare_players, pid_a, pid_b, pool, config)
            med = statistics.median(times)
            results.append(("compare_players()", med, classify(med)))
            print(f"  -> {med:.1f} ms\n")
        else:
            errors.append(("compare_players()", "Not enough hitters"))
            print("  -> Skipped\n")
    except Exception as e:
        errors.append(("compare_players()", str(e)))
        traceback.print_exc()
        print()

    # ── 9. compute_category_leaders() ────────────────────────────────
    print("Benchmarking: compute_category_leaders()")
    try:
        import pandas as pd

        from src.database import get_connection
        from src.leaders import compute_category_leaders

        conn = get_connection()
        try:
            season_stats = pd.read_sql_query("SELECT * FROM season_stats", conn)
        finally:
            conn.close()

        if not season_stats.empty:
            # Add name column if missing (leaders needs it for display)
            if "name" not in season_stats.columns and "player_id" in season_stats.columns:
                season_stats = season_stats.merge(
                    pool[["player_id", "name"]].drop_duplicates(), on="player_id", how="left"
                )
            times, leaders = time_fn(compute_category_leaders, season_stats)
            med = statistics.median(times)
            results.append(("compute_category_leaders()", med, classify(med)))
            print(f"  -> {med:.1f} ms ({len(leaders)} categories)\n")
        else:
            errors.append(("compute_category_leaders()", "No season_stats data"))
            print("  -> Skipped: no season_stats data\n")
    except Exception as e:
        errors.append(("compute_category_leaders()", str(e)))
        traceback.print_exc()
        print()

    # ── 10. rank_free_agents() ───────────────────────────────────────
    print("Benchmarking: rank_free_agents()")
    try:
        from src.in_season import rank_free_agents
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        hitters = pool[pool["is_hitter"] == 1]
        pitchers = pool[pool["is_hitter"] == 0]
        # Build a fake roster
        user_roster = list(hitters.head(12)["player_id"]) + list(pitchers.head(11)["player_id"])
        # FA pool = players NOT on roster
        fa_pool = pool[~pool["player_id"].isin(user_roster)].head(100)

        if not fa_pool.empty:
            times, fa_ranked = time_fn(rank_free_agents, user_roster, fa_pool, pool, config)
            med = statistics.median(times)
            results.append(("rank_free_agents()", med, classify(med)))
            print(f"  -> {med:.1f} ms ({len(fa_ranked)} agents ranked)\n")
        else:
            errors.append(("rank_free_agents()", "No FA pool"))
            print("  -> Skipped\n")
    except Exception as e:
        errors.append(("rank_free_agents()", str(e)))
        traceback.print_exc()
        print()

    # ── 11. compute_weekly_matchup_ratings() ─────────────────────────
    print("Benchmarking: compute_weekly_matchup_ratings()")
    try:
        from src.matchup_planner import compute_weekly_matchup_ratings
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Build a minimal roster DF
        roster = pool.head(23).copy()
        times, ratings = time_fn(
            compute_weekly_matchup_ratings,
            roster,
            weekly_schedule=None,
            park_factors=None,
            team_batting_stats=None,
            config=config,
        )
        med = statistics.median(times)
        results.append(("compute_weekly_matchup_ratings()", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(ratings)} ratings)\n")
    except Exception as e:
        errors.append(("compute_weekly_matchup_ratings()", str(e)))
        traceback.print_exc()
        print()

    # ── 12. start_sit_recommendation() ───────────────────────────────
    print("Benchmarking: start_sit_recommendation()")
    try:
        from src.start_sit import start_sit_recommendation
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Pick 3 hitters to compare
        hitters = pool[pool["is_hitter"] == 1].head(10)
        if len(hitters) >= 3:
            pids = list(hitters.head(3)["player_id"].astype(int))
            times, ss_result = time_fn(
                start_sit_recommendation,
                pids,
                pool,
                config=config,
            )
            med = statistics.median(times)
            results.append(("start_sit_recommendation()", med, classify(med)))
            print(f"  -> {med:.1f} ms\n")
        else:
            errors.append(("start_sit_recommendation()", "Not enough hitters"))
            print("  -> Skipped\n")
    except Exception as e:
        errors.append(("start_sit_recommendation()", str(e)))
        traceback.print_exc()
        print()

    # ── 13. compute_add_drop_recommendations() ───────────────────────
    print("Benchmarking: compute_add_drop_recommendations()")
    try:
        from src.valuation import LeagueConfig
        from src.waiver_wire import compute_add_drop_recommendations

        config = LeagueConfig()

        hitters = pool[pool["is_hitter"] == 1]
        pitchers = pool[pool["is_hitter"] == 0]
        user_roster = list(hitters.head(12)["player_id"]) + list(pitchers.head(11)["player_id"])

        times, waiver_recs = time_fn(
            compute_add_drop_recommendations,
            user_roster,
            pool,
            config=config,
            max_moves=2,
            max_fa_candidates=15,
            max_drop_candidates=3,
        )
        med = statistics.median(times)
        results.append(("compute_add_drop_recommendations()", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(waiver_recs)} recommendations)\n")
    except Exception as e:
        errors.append(("compute_add_drop_recommendations()", str(e)))
        traceback.print_exc()
        print()

    # ── 14. get_prospect_rankings() ──────────────────────────────────
    print("Benchmarking: get_prospect_rankings()")
    try:
        from src.prospect_engine import get_prospect_rankings

        times, prospects = time_fn(get_prospect_rankings, top_n=50)
        med = statistics.median(times)
        results.append(("get_prospect_rankings()", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(prospects)} prospects)\n")
    except Exception as e:
        errors.append(("get_prospect_rankings()", str(e)))
        traceback.print_exc()
        print()

    # ── 15. compute_trade_values() ───────────────────────────────────
    print("Benchmarking: compute_trade_values()")
    try:
        from src.trade_value import compute_trade_values
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        times, tv = time_fn(compute_trade_values, pool, config=config)
        med = statistics.median(times)
        results.append(("compute_trade_values()", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(tv)} players valued)\n")
    except Exception as e:
        errors.append(("compute_trade_values()", str(e)))
        traceback.print_exc()
        print()

    # ── 16. LineupOptimizerPipeline.optimize() ───────────────────────
    print("Benchmarking: LineupOptimizerPipeline.optimize() [quick]")
    try:
        from src.optimizer.pipeline import LineupOptimizerPipeline
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Build roster DF with required columns
        hitters = pool[pool["is_hitter"] == 1].head(14)
        pitchers = pool[pool["is_hitter"] == 0].head(9)
        roster = pd.concat([hitters, pitchers], ignore_index=True)

        pipeline = LineupOptimizerPipeline(roster, mode="quick", alpha=0.5, weeks_remaining=16, config=config)
        times, opt_result = time_fn(pipeline.optimize)
        med = statistics.median(times)
        results.append(("LineupOptimizerPipeline.optimize() [quick]", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("LineupOptimizerPipeline.optimize() [quick]", str(e)))
        traceback.print_exc()
        print()

    # ── 17. simulate_season() ────────────────────────────────────────
    print("Benchmarking: simulate_season() [100 sims]")
    try:
        # Build fake team totals for 12 teams
        import numpy as np

        from src.standings_projection import simulate_season

        np.random.seed(42)
        team_totals = {}
        for i in range(12):
            team_totals[f"Team_{i + 1}"] = {
                "R": 35 + np.random.normal(0, 5),
                "HR": 10 + np.random.normal(0, 2),
                "RBI": 33 + np.random.normal(0, 5),
                "SB": 5 + np.random.normal(0, 2),
                "AVG": 0.260 + np.random.normal(0, 0.01),
                "OBP": 0.330 + np.random.normal(0, 0.01),
                "W": 3 + np.random.normal(0, 1),
                "L": 3 + np.random.normal(0, 1),
                "SV": 3 + np.random.normal(0, 1),
                "K": 50 + np.random.normal(0, 10),
                "ERA": 4.0 + np.random.normal(0, 0.5),
                "WHIP": 1.25 + np.random.normal(0, 0.05),
            }
        times, season_result = time_fn(simulate_season, team_totals, n_sims=100)
        med = statistics.median(times)
        results.append(("simulate_season() [100 sims]", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("simulate_season() [100 sims]", str(e)))
        traceback.print_exc()
        print()

    # ── 18. compute_power_rankings() ─────────────────────────────────
    print("Benchmarking: compute_power_rankings()")
    try:
        from src.power_rankings import compute_power_rankings

        team_data = []
        for i in range(12):
            team_data.append(
                {
                    "team_name": f"Team_{i + 1}",
                    "roster_quality": 50 + i * 3,
                    "category_balance": 0.6 + i * 0.02,
                    "schedule_strength": 0.5 + i * 0.01,
                    "injury_exposure": 0.1 + i * 0.02,
                    "momentum": 0.5 + i * 0.03,
                }
            )
        times, pr = time_fn(compute_power_rankings, team_data)
        med = statistics.median(times)
        results.append(("compute_power_rankings()", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("compute_power_rankings()", str(e)))
        traceback.print_exc()
        print()

    # ── 19. bootstrap_all_data() [no-force, staleness check only] ────
    # Uses n_runs=1 because this involves network I/O
    print("Benchmarking: bootstrap_all_data() [staleness check, no force]")
    try:
        from src.data_bootstrap import bootstrap_all_data

        times, bs_result = time_fn(
            bootstrap_all_data,
            yahoo_client=None,
            force=False,
            n_runs=1,
        )
        med = statistics.median(times)
        results.append(("bootstrap_all_data() [no force]", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("bootstrap_all_data() [no force]", str(e)))
        traceback.print_exc()
        print()

    # ── 20. load_ecr_consensus() ─────────────────────────────────────
    print("Benchmarking: load_ecr_consensus()")
    try:
        from src.ecr import load_ecr_consensus

        times, ecr = time_fn(load_ecr_consensus)
        med = statistics.median(times)
        results.append(("load_ecr_consensus()", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(ecr)} rows)\n")
    except Exception as e:
        errors.append(("load_ecr_consensus()", str(e)))
        traceback.print_exc()
        print()

    # ── 21. DraftEngine.enhance_player_pool() ────────────────────────
    print("Benchmarking: DraftEngine.enhance_player_pool()")
    try:
        from src.draft_engine import DraftRecommendationEngine
        from src.draft_state import DraftState
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        engine = DraftRecommendationEngine(config, mode="quick")
        ds = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
        times, enhanced = time_fn(engine.enhance_player_pool, pool, ds)
        med = statistics.median(times)
        results.append(("DraftEngine.enhance_player_pool()", med, classify(med)))
        print(f"  -> {med:.1f} ms\n")
    except Exception as e:
        errors.append(("DraftEngine.enhance_player_pool()", str(e)))
        traceback.print_exc()
        print()

    # ── 22. evaluate_trade() with context enabled ────────────────────
    print("Benchmarking: evaluate_trade() [Phase 1 + context]")
    try:
        from src.engine.output.trade_evaluator import evaluate_trade
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        hitters = pool[pool["is_hitter"] == 1].head(20)
        pitchers = pool[pool["is_hitter"] == 0].head(20)
        if len(hitters) >= 5 and len(pitchers) >= 5:
            user_roster = list(hitters.head(12)["player_id"]) + list(pitchers.head(11)["player_id"])
            giving = [user_roster[0]]
            receiving = [int(hitters.iloc[15]["player_id"])]

            times, trade_ctx = time_fn(
                evaluate_trade,
                giving,
                receiving,
                user_roster,
                pool,
                config=config,
                enable_mc=False,
                enable_context=True,
                enable_game_theory=False,
            )
            med = statistics.median(times)
            results.append(("evaluate_trade() [Phase 1+context]", med, classify(med)))
            grade = trade_ctx.get("grade", "N/A") if isinstance(trade_ctx, dict) else "N/A"
            print(f"  -> {med:.1f} ms (grade: {grade})\n")
        else:
            errors.append(("evaluate_trade() [Phase 1+context]", "Not enough players"))
            print("  -> Skipped\n")
    except Exception as e:
        errors.append(("evaluate_trade() [Phase 1+context]", str(e)))
        traceback.print_exc()
        print()

    # ── 23. Database: load season_stats query ────────────────────────
    print("Benchmarking: DB query - season_stats")
    try:
        import pandas as pd

        from src.database import get_connection

        def load_season_stats():
            conn = get_connection()
            try:
                return pd.read_sql_query("SELECT * FROM season_stats", conn)
            finally:
                conn.close()

        times, ss = time_fn(load_season_stats)
        med = statistics.median(times)
        results.append(("DB: load season_stats", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(ss)} rows)\n")
    except Exception as e:
        errors.append(("DB: load season_stats", str(e)))
        traceback.print_exc()
        print()

    # ── 24. Database: load standings ─────────────────────────────────
    print("Benchmarking: DB query - league_standings")
    try:
        import pandas as pd

        from src.database import get_connection

        def load_standings():
            conn = get_connection()
            try:
                return pd.read_sql_query("SELECT * FROM league_standings", conn)
            finally:
                conn.close()

        times, standings = time_fn(load_standings)
        med = statistics.median(times)
        results.append(("DB: load league_standings", med, classify(med)))
        print(f"  -> {med:.1f} ms ({len(standings)} rows)\n")
    except Exception as e:
        errors.append(("DB: load league_standings", str(e)))
        traceback.print_exc()
        print()

    # ══════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  HEATER BENCHMARK REPORT")
    print("=" * 78)
    print()

    # Sort by median time, slowest first
    results.sort(key=lambda x: x[1], reverse=True)

    # Column widths
    name_width = max(len(r[0]) for r in results) if results else 30
    print(f"  {'Function':<{name_width}}  {'Median (ms)':>12}  {'Classification':<12}")
    print(f"  {'-' * name_width}  {'-' * 12}  {'-' * 12}")

    for name, med, cls in results:
        if cls == "Fast":
            marker = "[OK]"
        elif cls == "OK":
            marker = "[OK]"
        elif cls == "Slow":
            marker = "[!!]"
        else:
            marker = "[XX]"
        print(f"  {name:<{name_width}}  {med:>10.1f}ms  {cls:<12} {marker}")

    # Summary statistics
    if results:
        total = sum(r[1] for r in results)
        fast_count = sum(1 for r in results if r[2] == "Fast")
        ok_count = sum(1 for r in results if r[2] == "OK")
        slow_count = sum(1 for r in results if r[2] == "Slow")
        vslow_count = sum(1 for r in results if r[2] == "Very Slow")

        print()
        print(f"  Total benchmarked: {len(results)} functions")
        print(f"  Combined time:     {total:.0f} ms ({total / 1000:.1f} s)")
        print(f"  Fast (<100ms):     {fast_count}")
        print(f"  OK (100-500ms):    {ok_count}")
        print(f"  Slow (500-2000ms): {slow_count}")
        print(f"  Very Slow (>2s):   {vslow_count}")

    if errors:
        print()
        print(f"  ERRORS ({len(errors)}):")
        for name, err in errors:
            print(f"    {name}: {err[:120]}")

    print()
    print("=" * 78)


if __name__ == "__main__":
    # Need pandas imported for some inline benchmarks
    main()
