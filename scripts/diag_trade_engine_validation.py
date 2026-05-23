"""End-to-end validation of the trade engine on a real Team Hickey scenario.

Created 2026-05-23 as the empirical validation step before deciding
whether to invest in the report's missing pieces (weekly H2H matrix,
playoff/championship probability, IP-floor penalty, G-score, etc.).

Pipeline:
  1. Load the player pool + league rosters + standings from SQLite.
  2. Run find_trade_opportunities() to get Trade Finder's top candidates.
  3. Print the top 5 by composite_score, plus alternate rankings
     (top by user_sgp_gain, top by acceptance_probability).
  4. Take the top-by-composite trade and run evaluate_trade() with
     ALL flags on (enable_mc=True, enable_context=True,
     enable_game_theory=True, apply_ytd_blend=True).
  5. Print the full evaluator output, then analyze gaps vs the report:
     - Does it produce a playoff probability? No (expected).
     - Does it produce a weekly matrix? No (expected).
     - Does it produce an IP-floor flag? No (expected).
     - What does it actually produce as the decision signal?

Re-runnable with `python scripts/diag_trade_engine_validation.py`.
"""

from __future__ import annotations

import io
import sys
from typing import Any

# Force UTF-8 stdout on Windows (team names contain emojis like 🏆)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from src.database import (  # noqa: E402
    get_connection,
    load_league_standings,
    load_player_pool,
)
from src.engine.output.trade_evaluator import evaluate_trade  # noqa: E402
from src.engine.portfolio.category_analysis import build_standings_totals  # noqa: E402
from src.trade_finder import find_trade_opportunities  # noqa: E402
from src.valuation import LeagueConfig  # noqa: E402

USER_TEAM = "🏆 Team Hickey"


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_subheader(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def _load_league_rosters_dict() -> dict[str, list[int]]:
    """Build {team_name: [player_id, ...]} from league_rosters table."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT team_name, player_id FROM league_rosters WHERE player_id IS NOT NULL",
            conn,
        )
    finally:
        conn.close()

    rosters: dict[str, list[int]] = {}
    for team_name, grp in df.groupby("team_name"):
        rosters[str(team_name)] = grp["player_id"].astype(int).tolist()
    return rosters


def _summarize_trade(trade: dict[str, Any], idx: int) -> None:
    give = ", ".join(trade.get("giving_names", []))
    recv = ", ".join(trade.get("receiving_names", []))
    print(
        f"  [{idx}] {trade.get('trade_type', '?'):>7s} | "
        f"composite={trade.get('composite_score', 0):.3f} | "
        f"user_sgp={trade.get('user_sgp_gain', 0):+.2f} | "
        f"opp_sgp={trade.get('opponent_sgp_gain', 0):+.2f} | "
        f"p_accept={trade.get('acceptance_probability', 0):.2f} "
        f"({trade.get('acceptance_label', '?')})"
    )
    print(f"      GIVE: {give}")
    print(f"      RECV: {recv}")
    if "adp_fairness" in trade:
        print(
            f"      ADP fair={trade.get('adp_fairness', 0):.2f}  "
            f"ECR fair={trade.get('ecr_fairness', 0):.2f}  "
            f"cat_fit={trade.get('category_fit', 0):.2f}  "
            f"opp_need={trade.get('opp_need_match', 0):.2f}"
        )


def main() -> None:
    config = LeagueConfig()

    _print_header(f"STEP 1: Load context for {USER_TEAM}")
    pool = load_player_pool()
    pool = pool.rename(columns={"name": "player_name"})
    print(f"  player_pool rows: {len(pool)}")

    standings = load_league_standings()
    all_team_totals = build_standings_totals(standings)
    print(f"  standings teams with category totals: {len(all_team_totals)}")

    rosters = _load_league_rosters_dict()
    print(f"  league_rosters teams: {len(rosters)}")

    if USER_TEAM not in rosters:
        print(f"  ERROR: {USER_TEAM!r} not found in rosters!")
        print(f"  Available teams: {sorted(rosters.keys())}")
        sys.exit(1)

    user_roster_ids = rosters[USER_TEAM]
    print(f"  {USER_TEAM} roster size: {len(user_roster_ids)}")

    # User's standings position
    user_rank = None
    user_record = None
    if not standings.empty:
        wins_row = standings[(standings["team_name"] == USER_TEAM) & (standings["category"] == "WINS")]
        losses_row = standings[(standings["team_name"] == USER_TEAM) & (standings["category"] == "LOSSES")]
        ties_row = standings[(standings["team_name"] == USER_TEAM) & (standings["category"] == "TIES")]
        if not wins_row.empty:
            wins = int(wins_row.iloc[0]["total"])
            losses = int(losses_row.iloc[0]["total"]) if not losses_row.empty else 0
            ties = int(ties_row.iloc[0]["total"]) if not ties_row.empty else 0
            user_record = (wins, losses, ties)
            user_rank = int(wins_row.iloc[0]["rank"])
            print(f"  {USER_TEAM} rank={user_rank}  record={wins}-{losses}-{ties}")

    _print_header("STEP 2: Run Trade Finder")
    trades = find_trade_opportunities(
        user_roster_ids=user_roster_ids,
        player_pool=pool,
        config=config,
        all_team_totals=all_team_totals,
        user_team_name=USER_TEAM,
        league_rosters=rosters,
        max_results=20,
        top_partners=5,
        standings_rank=user_rank,
        team_record=user_record,
    )
    print(f"  Trade Finder returned {len(trades)} candidates")

    if not trades:
        print("  No trades surfaced. Engine validation cannot proceed on a real scenario.")
        print("  (This itself is a signal — Trade Finder thinks there's no good trade.)")
        return

    _print_subheader("Top 5 by composite_score (UI default)")
    for i, t in enumerate(sorted(trades, key=lambda x: x["composite_score"], reverse=True)[:5]):
        _summarize_trade(t, i + 1)

    _print_subheader("Top 5 by user_sgp_gain (pure SGP advantage)")
    for i, t in enumerate(sorted(trades, key=lambda x: x["user_sgp_gain"], reverse=True)[:5]):
        _summarize_trade(t, i + 1)

    _print_subheader("Top 5 by acceptance_probability (most-acceptable)")
    for i, t in enumerate(sorted(trades, key=lambda x: x["acceptance_probability"], reverse=True)[:5]):
        _summarize_trade(t, i + 1)

    # Pick top by composite (UI default)
    top_trade = max(trades, key=lambda x: x["composite_score"])
    _print_header("STEP 3: Deep-evaluate top trade with evaluate_trade() — ALL flags on")
    print(f"  GIVE: {', '.join(top_trade.get('giving_names', []))}")
    print(f"  RECV: {', '.join(top_trade.get('receiving_names', []))}")
    print()

    result = evaluate_trade(
        giving_ids=top_trade["giving_ids"],
        receiving_ids=top_trade["receiving_ids"],
        user_roster_ids=user_roster_ids,
        player_pool=pool,
        config=config,
        user_team_name=USER_TEAM,
        enable_mc=True,
        n_sims=10_000,
        enable_context=True,
        enable_game_theory=True,
        apply_ytd_blend=True,
    )

    _print_subheader("Headline decision signals (Phase 1 weighted SGP — AUTHORITY)")
    print(f"  grade          : {result.get('grade')}")
    print(f"  verdict        : {result.get('verdict')}")
    print(f"  surplus_sgp    : {result.get('surplus_sgp'):+.3f}")
    print(f"  confidence_pct : {result.get('confidence_pct')}%")
    grade_range = result.get("grade_range")
    if grade_range:
        print(
            f"  grade range    : {grade_range.get('grade_low')} → "
            f"{grade_range.get('grade')} → {grade_range.get('grade_high')} "
            f"({grade_range.get('confidence')} conf)"
        )

    # Bug A (2026-05-23): MC's own grader output is exposed as mc_grade/
    # mc_verdict/mc_confidence_pct so we can compare against Phase 1's authority.
    if "mc_grade" in result:
        _print_subheader("MC grader (diagnostic only — does NOT override Phase 1)")
        print(f"  mc_grade            : {result.get('mc_grade')}")
        print(f"  mc_verdict          : {result.get('mc_verdict')}")
        print(f"  mc_confidence_pct   : {result.get('mc_confidence_pct')}%")
        if result.get("mc_grade") != result.get("grade"):
            print(f"  ⚠ MC and Phase 1 DISAGREE on grade: Phase 1={result.get('grade')}  MC={result.get('mc_grade')}")
            print("    (Phase 1 weighted SGP wins — see tests/test_phase1_is_grade_authority.py)")
        else:
            print(f"  ✓ MC and Phase 1 agree on grade: {result.get('grade')}")

    _print_subheader("Per-category impact (weighted SGP)")
    cat_impact = result.get("category_impact", {})
    for cat in [
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "OBP",
        "W",
        "L",
        "SV",
        "K",
        "ERA",
        "WHIP",
    ]:
        v = cat_impact.get(cat, 0.0)
        marker = "✓" if v > 0 else ("✗" if v < 0 else "·")
        print(f"  {marker} {cat:>4s}: {v:+.3f}")

    _print_subheader("Penalties applied")
    print(f"  replacement_penalty : {result.get('replacement_penalty', 0):+.3f}")
    print(f"  flexibility_penalty : {result.get('flexibility_penalty', 0):+.3f}")
    print(f"  concentration_pen   : {result.get('concentration_penalty', 0):+.3f}")
    print(f"  bench_cost          : {result.get('bench_cost', 0):+.3f}")
    print(f"  ip_floor_penalty    : {result.get('ip_floor_penalty', 0):+.3f}")
    ipd = result.get("ip_floor_detail", {}) or {}
    if ipd:
        print(
            f"    └─ weekly IP: {ipd.get('before_weekly_ip', '?')} → "
            f"{ipd.get('after_weekly_ip', '?')}  (floor={ipd.get('threshold_ip_per_week', '?')}, "
            f"below_floor={ipd.get('below_floor', '?')})"
        )

    _print_subheader("Roster cap enforcement")
    print(f"  lineup_constrained : {result.get('lineup_constrained')}")
    print(f"  drop_candidate     : {result.get('drop_candidate')}")
    print(f"  fa_pickup          : {result.get('fa_pickup')}")
    reshuffle = result.get("reshuffle", {})
    if reshuffle:
        print(f"  reshuffle_sgp      : {reshuffle.get('reshuffle_sgp', 0):+.3f}")
        print(f"  reshuffle_pct      : {reshuffle.get('reshuffle_pct', 0)}%")
        if reshuffle.get("promoted"):
            promoted_names = ", ".join(p["name"] for p in reshuffle["promoted"])
            print(f"  promoted           : {promoted_names}")
        if reshuffle.get("demoted"):
            demoted_names = ", ".join(d["name"] for d in reshuffle["demoted"])
            print(f"  demoted            : {demoted_names}")

    _print_subheader("Monte Carlo distribution (Phase 2)")
    print(f"  mc_mean   : {result.get('mc_mean'):+.3f}")
    print(f"  mc_std    : {result.get('mc_std'):.3f}")
    print(f"  mc_median : {result.get('mc_median'):+.3f}")
    print(
        f"  percentiles: p5={result.get('p5'):+.3f} p25={result.get('p25'):+.3f} "
        f"p75={result.get('p75'):+.3f} p95={result.get('p95'):+.3f}"
    )
    print(f"  prob_positive : {result.get('prob_positive')}")
    print(f"  var5 (5% VaR) : {result.get('var5'):+.3f}")
    print(f"  cvar5         : {result.get('cvar5'):+.3f}")
    print(f"  sharpe        : {result.get('sharpe'):.3f}")
    ci = result.get("confidence_interval")
    if ci:
        print(f"  95% CI for mean : ({ci[0]:+.3f}, {ci[1]:+.3f})")
    print(f"  convergence quality : {result.get('convergence_quality')}")
    print(f"  ESS={result.get('convergence_ess'):.0f}  R-hat={result.get('convergence_rhat'):.3f}")

    _print_subheader("Game theory (Phase 5)")
    market_values = result.get("market_values", {})
    if market_values:
        for pid_str, mv in market_values.items():
            print(
                f"  player_id={pid_str}: market_price={mv.get('market_price'):+.2f}  "
                f"max_bidder={mv.get('max_bidder')}  max_bid={mv.get('max_bid'):+.2f}  "
                f"demand={mv.get('demand')}"
            )
    sensitivity = result.get("sensitivity_report", {})
    if sensitivity:
        print(f"  vulnerability     : {sensitivity.get('vulnerability')}")
        print(f"  breakeven_gap     : {sensitivity.get('breakeven_gap')}")
        bd = sensitivity.get("biggest_driver")
        if bd:
            print(
                f"  biggest_driver    : {bd.get('category')} {bd.get('impact'):+.2f} "
                f"(weight={bd.get('weight')}, {bd.get('direction')})"
            )
        bg = sensitivity.get("biggest_drag")
        if bg:
            print(
                f"  biggest_drag      : {bg.get('category')} {bg.get('impact'):+.2f} "
                f"(weight={bg.get('weight')}, {bg.get('direction')})"
            )

    _print_subheader("Risk flags")
    for flag in result.get("risk_flags", []):
        print(f"  ⚠ {flag}")
    if not result.get("risk_flags"):
        print("  (none)")

    _print_subheader("Punt categories")
    print(f"  {result.get('punt_categories', [])}")

    _print_header("STEP 4: Gap analysis vs Enhanced Trade Engine spec")
    print()
    print("  Outputs the current engine PRODUCED:")
    print("    ✓ Letter grade (A+ to F) — from SGP-surplus threshold")
    print("    ✓ surplus_sgp scalar (weighted SGP delta)")
    print("    ✓ Per-category SGP impact (12 cats)")
    print("    ✓ MC distribution (mean/std/percentiles/CVaR5/Sharpe)")
    print("    ✓ Confidence interval + convergence diagnostics (ESS, R-hat)")
    print("    ✓ Per-player opponent market values + clearing price")
    print("    ✓ Sensitivity report (biggest driver/drag, breakeven)")
    print("    ✓ Roster reshuffle transparency (LP promote/demote)")
    print("    ✓ Punt-category awareness")
    print()
    print("  Outputs the report's HCV-Hybrid would ALSO produce — and current doesn't:")
    print("    ✗ ΔΠ_playoff (probability of making top-4) — NOT computed")
    print("    ✗ ΔΠ_champ (probability of winning championship) — NOT computed")
    print("    ✗ Weekly 26×12 win-probability matrix p_{w,c} — NOT computed")
    print("    ✗ IP-floor penalty (κ × (20 - IP_w)²) — NOT applied to this trade")
    print("    ✗ G-score (Rosenof) — current uses pure z-scores with σ*=0")
    print("    ✗ Dynamic Markov FA replacement — current uses static best-current-FA")
    print("    ✗ Bracket simulation for top-4 playoff — not modeled")
    print("    ✗ Three-horizon split (pre-deadline / regular / playoff window)")
    print()
    print("  Key question for the user: which of these missing outputs would have")
    print("  CHANGED YOUR DECISION on this specific trade?")
    print()
    print("  If grade is A+ or F, the missing pieces are unlikely to flip the call.")
    print("  If grade is B/B-/C+, they might — that's where to invest build effort.")


if __name__ == "__main__":
    main()
