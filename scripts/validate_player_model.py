"""Real-data Phase-1 gate run for the Layer-0 player model.

Assembles historical weekly game-log records (predicted posterior vs realized outcome) and runs
src.player_model.validation.validate_layer0 to print the gate verdict: does the model's MAE match
the pool baseline AND are its posterior intervals calibrated?

Network + DB (MLB-StatsAPI via src/optimizer/backtest_runner.py) -> run MANUALLY, not in CI:
    python -m scripts.validate_player_model --weeks 6 --category HR

`build_records` is a pure, DB-free join (unit-tested); the data assembly around it needs the live DB.
"""

from __future__ import annotations

import argparse

import pandas as pd


def build_records(predictions: dict[int, dict], actuals: dict[int, float]) -> pd.DataFrame:
    """Join per-player predicted posterior {mean, sigma, baseline} with realized outcomes into the
    validate_layer0 records frame. Players without a realized outcome are skipped. Pure / DB-free."""
    rows = []
    for pid, pred in predictions.items():
        if pid not in actuals:
            continue
        rows.append(
            {
                "player_id": int(pid),
                "model_mean": float(pred.get("mean", 0.0)),
                "model_sigma": float(pred.get("sigma", 0.0)),
                "baseline_pred": float(pred.get("baseline", pred.get("mean", 0.0))),
                "realized": float(actuals[pid]),
            }
        )
    return pd.DataFrame(rows, columns=["player_id", "model_mean", "model_sigma", "baseline_pred", "realized"])


def run_gate(weeks: int, category: str) -> dict:  # pragma: no cover - needs live DB + network
    """Assemble real game-log records for `category` over the last `weeks` and run the gate.
    Imports the heavy deps lazily so the module + build_records stay import-light for tests."""
    import math

    from src.database import load_player_pool
    from src.optimizer.backtest_runner import run_backtest  # noqa: F401  (assembly hook)
    from src.player_model import build_player_models
    from src.player_model.validation import validate_layer0
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()
    pool = load_player_pool()
    models = build_player_models(pool, cfg)

    # Predicted per-week posterior for the category: mean + sigma=sqrt(sigma2 + tau2);
    # baseline = the pool's own per-week projection (the incumbent we must not degrade).
    predictions: dict[int, dict] = {}
    for pid, pm in models.items():
        post = pm.posteriors.get(category)
        if post is None:
            continue
        predictions[pid] = {
            "mean": post.mean,
            "sigma": math.sqrt(max(post.sigma2 + post.tau2, 0.0)),
            "baseline": post.mean,  # reuse-the-mean: baseline == model mean this phase
        }

    # NOTE: assembling `actuals` (realized weekly category totals per player) from game logs uses
    # src.optimizer.backtest_runner.run_backtest / statsapi over the chosen weeks. Operator wires the
    # specific weeks here; left as the integration point (needs the live DB + network).
    actuals: dict[int, float] = {}
    records = build_records(predictions, actuals)
    return validate_layer0(
        records, mean_col="model_mean", sigma_col="model_sigma", baseline_col="baseline_pred", realized_col="realized"
    )


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Layer-0 player-model Phase-1 gate (real data).")
    parser.add_argument("--weeks", type=int, default=6, help="historical weeks to backtest")
    parser.add_argument("--category", type=str, default="HR", help="scoring category to validate")
    args = parser.parse_args()
    verdict = run_gate(args.weeks, args.category)
    print("Layer-0 gate verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    if verdict.get("n", 0) == 0:
        # Distinguish "no data assembled" from a real model failure (the actuals game-log
        # assembly is the operator's integration point — an empty frame is NOT a gate failure).
        print("NO DATA — wire the actuals game-log assembly in run_gate before reading the verdict")
    else:
        print("PASS" if verdict.get("passes_gate") else "FAIL (see coverage / sigma_scale_hint)")


if __name__ == "__main__":  # pragma: no cover
    main()
