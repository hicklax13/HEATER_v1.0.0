"""BUG-005 architectural guard: engine/signals/* must not be imported by
the core trade-engine evaluation pipeline.

These modules are designed for the Trade-Readiness UI helper
(src/trade_signals.py), not for trade_evaluator. The architecture doc
promised "Kalman feeds Bayesian blend" but the wiring was never
completed. If a future engineer adds an import from engine.signals
into the trade-evaluation pipeline without first wiring it through a
proper adapter in engine.projections.bayesian_blend, this guard will
fail — surfacing the architectural choice.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules that MUST NOT import from engine.signals
FORBIDDEN_IMPORTERS = [
    "src/engine/output/trade_evaluator.py",
    "src/engine/monte_carlo/trade_simulator.py",
    "src/engine/projections/bayesian_blend.py",
    "src/engine/portfolio/category_analysis.py",
    "src/engine/game_theory/opponent_valuation.py",
]


def test_core_trade_engine_does_not_import_engine_signals():
    """Each FORBIDDEN_IMPORTERS file must NOT import from src.engine.signals.

    Allowed imports of engine.signals: src/trade_signals.py (UI helper)
    and the signals submodules themselves.

    If you intentionally wire signals into the trade engine, REMOVE the
    offending file from FORBIDDEN_IMPORTERS in this test AND update
    src/engine/signals/__init__.py docstring to reflect the new
    architecture.
    """
    pat = re.compile(r"(?:from|import)\s+src\.engine\.signals(?:\b|\.)")
    offenders: list[tuple[str, int, str]] = []
    for rel in FORBIDDEN_IMPORTERS:
        p = REPO_ROOT / rel
        if not p.exists():
            continue  # Module may have moved or been renamed
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pat.search(line):
                offenders.append((rel, lineno, stripped))
    assert not offenders, (
        "BUG-005 architectural boundary violation: core trade-engine module "
        "imported from src.engine.signals.* — these modules are for the "
        "UI helper, not the evaluation pipeline. To wire signals into the "
        "trade engine intentionally, route through engine.projections.bayesian_blend "
        "and update src/engine/signals/__init__.py docstring. Offenders:\n"
        + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offenders)
    )


def test_signals_modules_still_exist():
    """Sanity: the signals modules should still be on disk (consumed by
    trade_signals.py UI helper). This guard fails if someone deletes the
    modules outright."""
    required = ["decay.py", "kalman.py", "regime.py", "statcast.py"]
    signals_dir = REPO_ROOT / "src" / "engine" / "signals"
    missing = [m for m in required if not (signals_dir / m).exists()]
    assert not missing, f"signals modules disappeared: {missing}. trade_signals.py UI helper relies on them."
