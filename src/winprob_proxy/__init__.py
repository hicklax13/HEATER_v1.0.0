"""HEATER Layer-1 cheap win-probability proxy (Advanced Value Engine, Phase 2).

Analytic per-category P(win/tie/loss) over Layer-0 player-model team totals -> Δ category-wins ->
Δ playoff-odds. The instant/interactive tier; the deep policy-aware Monte-Carlo is Layer 2 (Phase 3).
Both tiers share the SAME margin model (spec Tier-1 #2): the counting Skellam / rate Welch here is
what Layer 2 samples from.
"""
