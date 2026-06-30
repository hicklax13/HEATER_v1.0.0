# HEATER Advanced Value Engine — Master Design

**Date:** 2026-06-30 · **Status:** design (brainstormed + research-validated; pre-plan)
**Companion:** [`2026-06-30-heater-value-engine-research-validation.md`](./2026-06-30-heater-value-engine-research-validation.md) — the citation-backed justification for every decision below (12-agent web sweep, 254 searches, 30+ primary sources re-verified).

---

## 1. North Star

Build **the most advanced fantasy-baseball trade/value engine possible** for HEATER (FourzynBurn: 12-team Yahoo Head-to-Head **Categories**, 12 scoring cats — R/HR/RBI/SB/AVG/OBP, W/L/SV/K/ERA/WHIP; L/ERA/WHIP inverse), and make it the **single source of truth** for *every* HEATER surface.

**One shared stochastic player model** feeds: the 3 Trades tabs (Finder / Build / Compare), the Optimizer, Matchup, Standings, Punt, Free Agents, Leaders, Streaming, Start/Sit, and Draft. **No surface may compute or display a value that contradicts the engine.**

### Honest non-goal (the load-bearing truth)
This is **not** a crystal ball, and any tool that claims to be is lying. For a *marginal* trade the true championship-probability signal (~0.5–2%) is genuinely **below the schedule-luck + projection noise floor** — no estimator can manufacture signal that isn't there. The achievable bar is **calibrated, decision-useful, and honest about its own uncertainty.** **"No detectable edge"** is a valid, common, and *correct* output. That honesty is the engine's actual edge over every overconfident competitor.

---

## 2. The objective (what the engine maximizes)

**Trade value to YOU = Δ E[U(final placement)]** — the change in your *expected end-of-season placement utility*, under optimal continuation play, estimated on the **post-trade** league.

- **U(f)** = graded placement utility: champion ≫ make-playoffs ≫ better seed (owner-chosen 2026-06-30; tunable). **Risk posture emerges automatically** from the convexity of `U ∘ standing` evaluated from your *current* state — a long-shot becomes correctly variance-*seeking*, a front-runner variance-*averse*; there is **no separate risk knob**.
- **Equivalent benchmark-relative form** (Haugh–Singal, *Management Science* 2021): maximize *P(beat the field)* up a ladder — Δ P(win week) → Δ P(playoff/seed) → Δ P(champ). The inner weekly lineup LP must optimize **Δ P(win the matchup), not Δ E[points]** — that is what makes bold-play-when-behind emerge.
- **Estimated** by **paired-seed (common-random-number) policy-aware Monte-Carlo** of rest-of-season + playoffs.
- **Graded** via a **backtest-tuned shrinkage blend** up the ladder (raw Δ-champ alone is below noise for marginal trades), reported **with an uncertainty band**; **"no detectable edge"** when the band straddles zero.
- **Finder ≠ Evaluator** (distinct objectives): Evaluator (you bring a trade) = one-sided, your Δ-value. Finder (proposes deals) = **P(accept) × max(0, your Δ-value)** + a mutual-benefit pre-filter.

---

## 3. Architecture — tiered, over one shared player model

### Layer 0 — Shared player model (one per player; feeds EVERY surface)
- **Mean projection:** per-category **double-shrunk** ridge-stacking of the 7 systems (shrink learned weights toward equal-weights via WLS **and** toward zero via ridge/LASSO), with a hard **equal-weight out-of-sample fallback** (forecast-combination puzzle). Marcel reliability-regression + aging as **transparent (non-learned)** layers; regress toward the *projection prior*, not the raw league mean. **Do NOT** replace the core with a neural ensemble (ML wins single-stat only; outlier-blind).
- **Skill regression — ASYMMETRIC, per-category:** hitters → capped barrel% / EV-on-air / xwOBA nudges (power YoY ≈ 0.80–0.85). Pitchers → **SIERA/xFIP anchor with LARGE variance**, K via CSW% / K-BB%, **NO pitcher xERA / batted-ball overrides** (xwOBA "no better than FIP" for pitchers), **de-confounded / down-weighted Stuff+** (switched-team corr ≈ 0.14). **Recency (L7/L14/L30) is a low-weight update to fast-stabilizing skill inputs only** — three studies show recent *outcomes* add ≈ 0 predictive value once a real projection exists.
- **Per-player posterior (distribution, not a point):** **heteroscedastic** true-talent posterior per category via **SURE / variance-dependent shrinkage** (Xie–Kou–Brown) over variance-stabilized rates; supplies **σ² (player-to-player)** and **τ² (week-to-week)** separately. **NB** margins for counting cats, **beta-binomial** for rate cats — *variance unequal across players*.
- **Availability survival distribution (NEW — frontier gap, no academic prior):** per-player games-/IP-remaining + current-injury hazard (age + position + injury-history + status), sampled per replicate. Built in-house from `injury_model.py` / `il_manager.py` / return-date curves.

### Layer 1 — Cheap analytic proxy (instant grades + ranks the wide candidate set)
- Per-category **P(win / tie / loss)** via **analytic Skellam** (counting cats, **explicit tie mass** — 6-6 weeks are decisive) + Welch/Normal on volume-weighted ratios (rate cats), Rao-Blackwellized.
- Threshold-aware **Δ category-wins → Δ playoff-odds** ladder. **Category-need fit is subsumed here** (it values *winnable* weak categories, not locked ones) — surfaced as an explicit "this trade fixes: SB, AVG" readout.
- **G-score = DISPLAY-ONLY** interpretable per-player scalar — **never** the probability engine (its equal-variance + normal assumptions contradict NB/Skellam).
- **MUST use the SAME margins + dependence model as Layer 2** (mandatory — a proxy whose math disagrees with the deep sim silently mis-ranks the candidate set the deep tier never re-examines).

### Layer 2 — Deep policy-aware paired-MC (finalists only, async)
- **Hierarchical two-level sampling:** per replicate, draw each player's true-talent rate from its posterior, **then** draw weekly outcomes; sample availability from the survival curve. (Without this, championship bands are **dishonestly narrow** and the honesty mechanism is fake.)
- **Cross-category dependence:** Gaussian copula (Cholesky → Φ → inverse-NB / beta-binomial), latent correlation **shrunk (Ledoit-Wolf) and calibrated JOINTLY with the margins via rank-correlation matching** (discrete-margin copulas are non-identifiable if you bolt a continuous correlation onto discrete margins).
- **Policy-aware play of ALL 12 teams:** each simulated week, re-solve the weekly LP (optimizing Δ P(win matchup)) **and** run a bounded, budget-aware **streaming/FA replacement step** (the 10-add/week budget) — **for you AND opponents** (frozen opponents bias every grade toward "your trade helps").
- **Season → top-4 → bracket** (Bradley-Terry / log5); **grade = Δ E[placement utility]**.
- **Variance-reduction stack:** paired **CRN** (synchronize streams; perturb only swapped players) ∘ **antithetic** ∘ **proxy-as-control-variate**.
- **Budget across candidates:** **OCBA** allocation (Nᵢ ∝ (σᵢ/δ₁ᵢ)²) + sequential stopping — ~10× fewer sims for the same selection confidence. Do **NOT** use importance sampling / splitting (wrong regime — this is a *difference of moderate probabilities*, not a rare event).

### Layer 3 — Finder vs Evaluator
- **Evaluator** (user brings a trade): one-sided, your Δ-value + band.
- **Finder:** `P(accept) × max(0, your Δ-value)` + mutual-benefit pre-filter. Opponent Δ-value = the **same engine** run on their roster. **P(accept)** = a calibrated, **league-history-updated classifier** (features: opponent surplus + apparent/headline fairness on raw player value + manager activity + contender status + roster-hole urgency + #pieces; Platt → isotonic calibration; Bayesian cold-start prior). Present **fairness as a range**, acceptance as a **probability + band** (Myerson-Satterthwaite: no objective fair price exists).

### Backtest harness (gates EVERY layer — nothing ships unless it beats the prior version OOS)
- **CPCV** (combinatorial purged cross-validation, purge + embargo) to tune the shrinkage-blend weight — season/ROS labels overlap, so plain k-fold leaks.
- **Deflated Sharpe / PBO + Minimum-Backtest-Length** whenever selecting any best-of-many config (else "backtest-tuned" = "backtest-overfit" on a near-unpowered 1-bit champ label).
- **Diebold-Mariano + Hansen Model Confidence Set** to certify the single-source-of-truth variant.
- **Adaptive / weighted Conformal** intervals (baseball data are non-exchangeable); rank by **proper score** (Brier / log-loss); ECE = diagnostic only.
- **Calibrate every per-category variance + the correlation matrix on the backtest** (replace the hand-set `scenario_generator.py` DEFAULT_CV ~50% guesses); validate by **reliability / coverage on realized weekly category outcomes**. *This is the highest-value open item — if σ/ρ are wrong, no variance reduction helps; you precisely estimate a biased number.*

---

## 4. Phasing — build order (each phase = its own plan, each backtest-gated)

| Phase | Builds | Gate |
|---|---|---|
| **0 · Backtest harness + baseline** | The measuring stick (CPCV + DSR/PBO + DM/MCS + conformal + rank-IC/regret); freeze today's YTD value as the baseline; **calibrate the variance/correlation terms** | exists + reproduces the baseline's accuracy |
| **1 · Layer 0 shared player model** | mean (double-shrunk stacking) + asymmetric skill + **heteroscedastic NB/beta-binomial posterior** + **availability survival** | projection MAE + coverage beat the current pool |
| **2 · Layer 1 cheap proxy** | analytic Skellam/Welch win-prob ladder (tie mass, need-fit) → **wired to all 3 Trades tabs** | OOS rank-IC > the YTD baseline |
| **3 · Layer 2 deep sim** | hierarchical + copula + **policy-aware opponents + streaming** + CRN/CV/OCBA, async | beats the proxy on held-out decision-regret; latency within budget |
| **4 · Layer 3 acceptance model** | calibrated P(accept) classifier | calibration (reliability/Brier) on league accept/reject history |
| **5 · App-wide rollout + consistency lock** | re-point every surface to the engine; a test that locks consistency | no surface computes a divergent value |

We build in this order so a **validated** improvement ships by **Phase 2** (not after the whole thing), and the deep tier only deploys once it's *proven* to beat the proxy.

---

## 5. App-wide consistency (single source of truth)

Every surface consumes **Layer 0/1 (instant)** and **Layer 2 (deep, where relevant)** — never its own valuation. A **consistency test** asserts no surface re-derives a divergent value. The **cheap tier powers interactive surfaces**; the **deep tier is async**. Coupling risk (a miscalibration or slow path propagating app-wide) is mitigated by: interactive surfaces always on the cheap tier; deep tier async; and **DM/MCS certification required before any variant becomes canonical**.

---

## 6. Module boundaries (units — each testable in isolation)

- **`player_model`** (Layer 0): `(player_id, league_ctx) → {per-cat posterior (mean, σ², τ²), availability_survival, display_g_score}`.
- **`winprob_proxy`** (Layer 1): `(roster_before, roster_after, matchup, standings) → {per-cat P(win/tie/loss), Δcat_wins, Δplayoff_odds, band}` — pure given Layer 0.
- **`season_sim`** (Layer 2): `(post_trade_league, policy) → {placement_distribution, ΔE[U], band}` — async.
- **`acceptance_model`** (Layer 3): `(trade, opponent) → {opponent_Δvalue, P_accept ± band, fairness_range}`.
- **`backtest`**: `(valuation_fn, history) → {rank_IC, calibration, regret, DM/MCS verdict}`.

Same margins + copula instance shared by `winprob_proxy` and `season_sim` (single config object) — enforced, not duplicated.

---

## 7. Reuse map (existing HEATER machinery to compose, not reinvent)

`marcel.py` · `projection_stacking.py` · `bayesian.py` (BMA + variance) · `engine/projections/{marginals,bayesian_blend}.py` · `engine/signals/{kalman,decay,regime,statcast}.py` · `engine/output/weekly_matrix.py` (Gaussian + Skellam, `build_team_kalman_variances`) · `engine/output/playoff_sim.py` · CARA utility + λ-sweep · `engine/monte_carlo/trade_simulator.py` (paired/antithetic) · `engine/output/backtest_calibration.py` + `optimizer/backtest_runner.py` · `lineup_optimizer.py` (LP) · `optimizer/stream_analyzer.py` (`build_stream_board` / `recommend_streaming_moves`) · `injury_model.py` / `il_manager.py` (availability signals) · `optimizer/scenario_generator.py` (variance — to be **calibrated**, not hand-set).

---

## 8. Non-goals / explicit rejections (research-backed)

- Not a crystal ball; "no detectable edge" is a first-class output.
- No neural/ML ensemble as the primary projector (single-stat only; outlier-blind).
- No importance sampling / splitting (wrong regime); no Kelly / minimax-regret as the objective (one-shot season; over-conservative given probabilities).
- **G-score is display-only**, never the probability engine.
- No frozen-opponent / lineup-only policy (biases every grade).

---

## 9. Risks + mitigations

1. **Model misspecification > MC noise** (the #1 risk) → backtest-calibrate σ/ρ; validate by reliability/coverage.
2. **Proxy ↔ deep inconsistency** → identical margins/copula both tiers; periodically deep-MC a sample of proxy-*rejected* trades to measure rank-agreement + recall.
3. **Unmodeled availability** → the survival layer (Phase 1).
4. **Policy incompleteness / frozen opponents** → policy-aware opponents + in-loop streaming (Phase 3).
5. **1-bit champ label near-unpowered** → lean on the surrogate ladder (higher-information rungs) + shrink toward equal-weight when the deflated test is inconclusive.
6. **Single-source coupling** → cheap-tier interactive, deep-tier async, DM/MCS gate before canonical.
7. **Compute/latency at candidate-set scale** → prototype the finalist tier + measure wall-time before wiring behind interactive surfaces; OCBA + CRN + control-variate keep it to ~1e3–5e3 paired seasons/finalist.

---

## 10. Verdict (honest)

- **Architecture: provably best-in-class** — every published method (Rosenof G/H-score, RotoGraphs roto-sim, the lone GA fantasy-trade optimizer, commercial sum-and-compare tools) is a *degraded special case*; none simulate the weekly category matchup, re-optimize lineups in-sim, or grade by a policy-aware Δ-championship. The corrected design would be a genuine, defensibly-publishable first.
- **"Provably best numbers" is the wrong bar** — and the design correctly admits it. The win is **calibration + decision-usefulness + honest uncertainty**, delivered *only after the Phase-0/1 corrections* (especially hierarchical parameter draws — without them the bands are dishonestly narrow).
- **Residual uncertainty** (and how Phase 0 closes it): variance/correlation calibration (highest-value open item), availability modeling (in-house, no prior), proxy↔deep rank-agreement (recall audit), candidate-scale latency (prototype + measure), 1-bit-label power (surrogate ladder).

Full per-subject SOTA verdicts, the prioritized 17-item upgrade list, and all citations are in the companion validation report.
