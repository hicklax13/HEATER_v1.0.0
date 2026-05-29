# HEATER v2 — Line-up Optimizer Design Spec

**Date:** 2026-05-22
**Status:** Draft for user review
**Author:** Brainstorming session output, anchored to 11-agent research swarm synthesis
**Brainstorm session:** `.superpowers/brainstorm/36109-1779488367/`
**Research reports:** 11 files in `.superpowers/brainstorm/36109-1779488367/research/`
**Synthesis source:** `.superpowers/brainstorm/36109-1779488367/synthesis-final.md`

---

## 1. Executive Summary

HEATER's current Line-up Optimizer is a 21-module pipeline that runs through a deterministic PuLP integer LP, blends 7 industry projection systems via ridge regression, applies multiplicative matchup adjustments (platoon/park/weather/Log5), and produces single-day or single-week lineup recommendations. Live audit found ~12 distinct issues spanning correctness (standings rendered with 13th place in a 12-team league, ERA/AVG ranks inverted), UX (mode and risk-aversion settings silently ignored in "Today" scope, 159-second compute time on click, 6 of 8 pitcher slots left empty), and missing features (no date selection, no per-day customization, no save/share, no advanced algorithms beyond mean-variance + scenario CVaR).

**HEATER v2 is a six-layer probabilistic optimizer** that solves these issues by replacing the deterministic-LP-with-bolt-on-multipliers model with an ensemble-of-Bayesian-and-foundation-model projection stack, causally-corrected matchup layer, robust-and-variance-aware decision engine, learned re-ranker, and local-LLM explanation surface. The user's requirements (Hybrid weekly engine, per-day date drill-down, save/share to 6 endpoints, more accuracy, more customization, more advanced algorithms, more factors, PhD-grade rigor, local LLM/AI focus) all map onto this architecture.

**Scope:** All 20 Tier-1 quick wins + all 11 Tier-2 foundational upgrades are in for v2 release. The 4 Tier-3 moonshots are documented in Appendix A as v2.5+ roadmap. Save/Share spans six endpoints: PNG, Markdown, iMessage, Gmail, Outlook, SMS.

**Build estimate:** Phase 1 (Tier 1, 20 items) ≈ 3-5 weeks. Phase 2 (Tier 2, 11 items) ≈ 8-12 weeks. Total v2 release ≈ 3-4 months.

---

## 2. Goals & Non-Goals

### Goals (from user, explicit)
- **G1.** Get more accurate answers than the current optimizer (top-line, drives everything else)
- **G2.** Select the exact date to optimize for (any date in the 26-week 2026 season, today through end)
- **G3.** More customization (per-factor toggles, per-day overrides, per-category risk preferences)
- **G4.** More advanced algorithms (PhD-grade math/physics rigor — Bayesian, causal, foundation-model, robust optimization)
- **G5.** More factors considered (Statcast bat tracking, Vegas implied totals, causal confounder adjustment, opposing-bullpen quality, lineup-spot PA modeling)
- **G6.** Hybrid weekly engine — single compute → results for each day of the selected week; drill into any single day; per-day setting overrides
- **G7.** Save optimization results inside HEATER
- **G8.** Share results to: PNG, Markdown, iMessage, Gmail, Outlook, SMS

### Non-Goals (explicit deferrals)
- **NG1.** Custom Baseball-BERT pretraining (12-week project, $50-100 A100 rental). Deferred to v2.5+ roadmap. v2 uses Heaton & Mitra checkpoint instead.
- **NG2.** Wasserstein-DRO via Mosek ($2,250 perpetual license). Deferred to v2.5+. v2 uses Bertsimas-Sim Γ-budget robust mode as the 80-of-value substitute.
- **NG3.** Sports Game Odds player props ($99/mo). Deferred until player-prop use case clearly demanded.
- **NG4.** Full Pyomo + Gurobi migration. v2 ships HiGHS as the open-source 4-10× speedup; Gurobi reserved for v2.5 DRO phase if needed.
- **NG5.** Cloud LLM commitments (OpenAI/Anthropic APIs in the production path). v2 is local-LLM-first via Ollama. Cloud reserved for explicit user-opt-in features.
- **NG6.** Backwards-incompatible page deletion. v2 ships as enhanced `pages/2_Line-up_Optimizer.py`; the audit-identified bugs (159s compute, standings render) are also fixed inline.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER (real-time + cached)                                          │
│   MLB statsapi (5-min poll) · The Odds API · Statcast bat tracking      │
│   Ollama LLM news extractor · RAG corpus (ChromaDB)                     │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PROJECTION STACK — Ensemble                                              │
│   PyMC Bayesian MARCEL · TabPFN-v2.5 · Temporal Fusion Transformer      │
│   Ornstein-Uhlenbeck SDE · 7-system ridge anchor                        │
│   → blended + MAPIE Conformal Quantile Regression                       │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MATCHUP LAYER — Causally Corrected                                       │
│   Hierarchical Bayesian Log5 (PyMC) · DoWhy back-door adjustment        │
│   causal-learn DAG audit · Gaussian copula same-team correlation        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ DECISION ENGINE — Robust + Variance-Aware                                │
│   HiGHS solver (replaces CBC) · Bertsimas-Sim Γ-budget robust           │
│   Variance-Aware QP (sign-flipped λ on win_prob)                        │
│   k=20 candidate lineups via epsilon-constraint, gapRel=0.005           │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ SELECTION — Learned Re-Ranking + Game Theory                             │
│   LambdaMART (XGBoost rank:ndcg) · Vowpal Wabbit contextual bandits     │
│   Dirichlet-Multinomial FA market · LLM-as-judge NL constraints         │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ EXPLANATION LAYER — Local LLM + RAG                                      │
│   Mistral Small 3.2 24B explainer · LlamaIndex + BGE-M3 RAG             │
│   LangGraph "Ask HEATER" tool-using agent · Ollama daemon (local)       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Cross-cutting concerns** (applied across all layers):
- **Hybrid weekly engine**: single compute produces 7-day results; date picker drills into any single day
- **Save/Share subsystem**: snapshot inputs+outputs; export to 6 endpoints
- **Per-day override layer**: user can adjust settings for any single day; rest of week unchanged

---

## 4. Data Layer Design

### 4.1 New sources (5)

| Source | Purpose | Library / Endpoint | TTL | Cost |
|---|---|---|---|---|
| **MLB statsapi 5-min polling** | Late scratches, lineup mutations, in-game probable-pitcher swaps | `statsapi.schedule?hydrate=lineups,probablePitcher` | 5 min (1 min T-4h to first pitch) | $0 |
| **The Odds API free tier** | Vegas implied team totals, money lines | `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds` | 1 day | $0 (500 credits/mo) |
| **Statcast bat tracking** | bat_speed, squared_up%, blast%, ideal_attack_angle% | pybaseball + direct Savant scrape | 24h | $0 |
| **Ollama llama3.1:8b news extractor** | Structured news events (late scratches, return dates, suspensions) via FSM-constrained JSON | Local Ollama daemon at `localhost:11434/v1`, model `llama3.1:8b-instruct-q4_K_M` (~5GB) | per news fetch | $0 (local) |
| **RAG sabermetric corpus** | FanGraphs Library, BPro, Tom Tango blog, Carleton "Sample Size" series for grounded explanations | ChromaDB embedded + LlamaIndex + BGE-M3 embeddings | rebuilt monthly | $0 (local) |

### 4.2 Module-level changes

**New files:**
- `src/optimizer/lineup_poll.py` — 5-min MLB statsapi polling for lineup mutations
- `src/data_sources/odds_api.py` — The Odds API client + nightly refresh
- `src/data_sources/bat_tracking.py` — Statcast bat-tracking ingest (joins existing `statcast_archive` schema)
- `src/llm/news_extractor.py` — Ollama-based structured news extraction (replaces regex `parse_suspension_days`)
- `src/rag/corpus_builder.py` — One-time + scheduled monthly RAG corpus build
- `src/rag/retrieval.py` — Query interface for `RAG over corpus → top-K snippets`

**Modified files:**
- `src/data_bootstrap.py` — Add 5 new bootstrap phases: `_bootstrap_odds_api`, `_bootstrap_bat_tracking`, `_bootstrap_lineup_poll`, `_bootstrap_rag_corpus`, `_bootstrap_news_extractor_warmup`
- `src/database.py` — Add `bat_tracking` table (bat_speed, squared_up_pct, blast_pct, ideal_attack_angle_pct per player); add `news_events` table (structured `{event_type, player_id, return_date, severity, source_url, parsed_at}`); add `odds_archive` table (date, team, implied_total, money_line)
- `src/news_sentiment.py` — Replace regex `parse_suspension_days` with Ollama-based call to `news_extractor.extract(text)` returning Pydantic-validated event

### 4.3 Refresh strategy

```
       Phase           Cadence       Source                  TTL
       ─────           ───────       ──────                  ───
   1.  bootstrap       on-start      all 33 phases           per-source
   2.  lineup_poll     5 min         MLB statsapi            5 min
       (1 min from T-4h to first pitch)
   3.  odds_api        24h           The Odds API            24h
   4.  bat_tracking    24h           Statcast                24h
   5.  news_extractor  on-news       Ollama (local)          immediate
   6.  rag_corpus      monthly       FG/BPro/Tango scrape    30 days
```

---

## 5. Projection Stack Design

### 5.1 Architecture

The new projection stack runs **5 parallel paths** + ridge-regression blender + calibrated uncertainty:

```
[Inputs: roster, league context, recent form, projection-system snapshots]
     │
     ├──→ Path 1: PyMC Bayesian MARCEL  (replaces hardcoded 5/4/3 blender)
     │       output: posterior mean + variance per cat per player
     │
     ├──→ Path 2: TabPFN-v2.5 zero-shot (cold-start: minor leaguers, call-ups)
     │       output: posterior + native interval per cat per player
     │
     ├──→ Path 3: Temporal Fusion Transformer (weekly forecasts)
     │       output: P10/P50/P90 quantiles per cat per week
     │
     ├──→ Path 4: Ornstein-Uhlenbeck SDE (continuous-time recency)
     │       output: latent skill mean + variance per player
     │
     └──→ Path 5: Existing 7-system ridge (Steamer/ZiPS/DC/ATC/THE BAT/THE BAT X/Marcel)
             output: point estimates (current behavior)
     │
     ▼
[Ridge stacker] — blends 5 paths via fit-on-historical weights
     │
     ▼
[MAPIE Conformalized Quantile Regression] — distribution-free intervals
     │
     ▼
[Output: per-player per-game per-cat: (mean, P10, P50, P90, variance)]
     → fed into Matchup Layer
```

### 5.2 Per-path detail

**Path 1: PyMC Bayesian MARCEL** (Agent 01, T1.11)
- Replaces `src/optimizer/projections.py`'s hardcoded `recent_form_weight = 0.25` and quadratic aging
- PyMC 5 + NumPyro NUTS backend
- Dirichlet-distributed season weights (vs hardcoded 5/4/3)
- Beta-binomial hierarchical pooling on rate stats
- Parameter-fit aging peak (no longer assumes age 27)
- R-hat target ≤ 1.01
- Citation: Fonnesbeck (PyMC-Labs, Aug 2024)

**Path 2: TabPFN-v2.5** (Agent 03, T1.3)
- New module `src/optimizer/tabpfn_oracle.py`
- `pip install tabpfn` (HuggingFace `Prior-Labs/TabPFN-v2`)
- Features: 60-80 cols already in `_build_player_pool` (xwOBA, barrel%, hard_hit%, sprint_speed, age, BB%, K%, GP, park, weather)
- Target: per-cat counting stats
- One forward pass; no training loop
- Best for cold-start regimes (<200 PA)
- Output: prediction + native CRPS-competitive interval
- License: TabPFN Academic — personal use OK

**Path 3: Temporal Fusion Transformer** (Agent 03, T2.2)
- New module `src/optimizer/tft_forecast.py`
- `pytorch-forecasting` 1.2+ on PyTorch Lightning 2.6+
- Static covariates: handedness, age, park
- Known-future covariates: opponent wRC+, weather, probable status
- Observed history: last N weeks of cat stats
- Output: P10/P50/P90 quantiles per cat per week
- Training: ~30 min CPU; 1,200 players × 26 weeks × 3 seasons
- Citation: Lim 2021 + Pitcher Performance Prediction MLB by TFT (CMC 2025)

**Path 4: Ornstein-Uhlenbeck SDE** (Agent 01, T2.3)
- Replaces `recent_form_weight` linear interpolation in `projections.py`
- `dynamax` library
- Continuous-time mean-reverting Bayesian filter
- Reversion rate θ fit per player
- Captures: latent skill drift over the season, regression to player-specific mean
- Citation: Wieland-Mews-Langrock 2021, arXiv:2011.04384

**Path 5: Existing 7-system ridge** (preserved)
- `src/data_pipeline.py::create_blended_projections`
- Stays as anchor / fallback when other paths fail

**Blender: Ridge stacker** (extended)
- `src/optimizer/projections.py::stack_projection_paths`
- Fit weights on historical week-on-week outcomes
- Outputs to MAPIE for calibration

**Calibrator: MAPIE Conformal Quantile Regression** (Agent 01, T1.7)
- New module `src/optimizer/calibration.py`
- `pip install mapie`
- Distribution-free finite-sample coverage
- Feeds calibrated P10/P50/P90 directly into `scenario_generator.py`'s CVaR

### 5.3 Module-level changes

**New files:**
- `src/optimizer/marcel_bayesian.py` — PyMC Bayesian MARCEL
- `src/optimizer/tabpfn_oracle.py` — TabPFN-v2.5 zero-shot
- `src/optimizer/tft_forecast.py` — Temporal Fusion Transformer
- `src/optimizer/ou_recency.py` — Ornstein-Uhlenbeck SDE recency
- `src/optimizer/calibration.py` — MAPIE CQR wrapper

**Modified files:**
- `src/optimizer/projections.py` — Add `build_enhanced_projections_v2()` that runs 5 paths in parallel via `concurrent.futures.ThreadPoolExecutor`, stacks via ridge regression, calibrates via MAPIE, returns `(mean, P10, P50, P90, variance)` per player per cat
- `src/optimizer/constants_registry.py` — Add new entries for ridge stacker weights, MAPIE alpha, TabPFN max sample size
- `src/optimizer/pipeline.py` — Stage 1 (Enhanced Projections) calls v2 path; preserves v1 as fallback

---

## 6. Matchup Layer Design

### 6.1 Architecture

```
[per-player projection from Stack] + [per-game context: opponent, park, ump, catcher, weather]
                                          │
                                          ▼
                  ┌─────────────────────────────────────┐
                  │ Hierarchical Bayesian Log5 (PyMC)  │  T2.1
                  │   beta priors + handedness offsets  │
                  │   replaces multiplicative Log5      │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ DoWhy back-door adjustment          │  T1.15
                  │   confounders: ump ↔ catcher ↔ H/A  │
                  │   produces causal_multiplier()      │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ Per-cat adjusted projection         │
                  └────────────────┬────────────────────┘
                                   │
                            (in scenarios:)
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ Gaussian copula same-team corr.     │  T1.19
                  │   in scenario_generator.py          │
                  └─────────────────────────────────────┘
```

### 6.2 Hierarchical Bayesian Log5 (T2.1)

- **Replaces:** Multiplicative `_HIT_PROB_LOG5` and `_PITCHER_QUALITY_FACTOR` in `matchup_adjustments.py`
- **Library:** PyMC 5 + NumPyro
- **Model:** `pitcher_rate ~ Beta(α_p, β_p)`, `batter_rate ~ Beta(α_b, β_b)`, handedness offset γ_h. Equation 3 of Adams et al. 2025: combine in logit space with constraints `1 ≤ P + B ≤ 2`.
- **Expected gain:** ~1 additional win per 162-game season (Adams 2025)
- **Compute:** ~5-10s per matchup (cached per week)
- **New file:** `src/optimizer/hier_log5.py`

### 6.3 DoWhy back-door causal layer (T1.15)

- **Library:** `dowhy` 0.14 + `econml`
- **Confounders identified:** umpire ↔ catcher ↔ home/away (the empirically established confounded triple)
- **Pipeline:** Pearl's 4 steps — model → identify → estimate → refute
- **Refutation:** placebo / random-cause becomes a structural-invariant test
- **New file:** `src/causal_layer.py`
- **Integration:** wraps the Hierarchical Bayesian Log5 output

### 6.4 Causal-DAG audit (T1.10)

- **Library:** `causal-learn` 0.1.3.6 (PC + FCI algorithms)
- **Data:** 2024-2025 Statcast (already in HEATER)
- **Output:** `data/causal_dag.json` — calibration artifact telling us which multipliers double-count
- **Frequency:** annual at spring-training
- **One-time analysis** (not production code)

### 6.5 Gaussian copula same-team correlation (T1.19)

- **Replaces:** Current `scenario_generator.py` treats co-rostered Yankees as independent
- **Math:** Sample marginals (Gamma hitters, Normal pitchers) through team-pair correlation matrices
- **Expected gain:** +5% RMSE on weekly sims; +1.79 pts spillover (MLB DFS research)
- **Modified file:** `src/optimizer/scenario_generator.py`

---

## 7. Decision Engine Design

### 7.1 Architecture

```
[adjusted projections from Matchup Layer with (mean, P10, P50, P90, variance) per player per cat]
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ HiGHS solver  (replaces CBC)        │  T1.1
                    │   4-10× faster, MIT, gapRel=0.005   │  T1.2
                    └──────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ Three decision modes — pick one or   │
                    │ run all and surface to user:         │
                    │                                      │
                    │ (a) Deterministic LP (current)      │
                    │ (b) Bertsimas-Sim Γ-budget robust   │  T2.5
                    │ (c) Variance-Aware QP (cvxpy)       │  T2.6
                    └──────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ k=20 candidate lineups via           │
                    │ epsilon-constraint                   │
                    │   passes to Selection Layer          │
                    └──────────────────────────────────────┘
```

### 7.2 HiGHS migration (T1.1, T1.2)

- **One-line change:** `PULP_CBC_CMD` → `HiGHS` in `src/lineup_optimizer.py`
- **MIT licensed**
- **Performance:** SGM 743 vs SCIP's 853 (Mittelmann May 2026 benchmark) → 4-10× faster on mid-sized MIPs
- **gapRel=0.005:** 0.5% MIP gap is sufficient for fantasy; bounds solve time once CVaR scenarios push the deterministic-equivalent MIP into thousands of binaries

### 7.3 Bertsimas-Sim Γ-budget robust mode (T2.5)

- **New file:** `src/optimizer/robust_lp.py` (sibling to `advanced_lp.py`)
- **Problem size:** 28 binaries + 29 continuous + 28 linear constraints (nearly identical to current LP)
- **Half-widths:** from existing `DEFAULT_CV` constants
- **Guarantee:** Constraint-violation bound `exp(-Γ²/2n)`
- **Effort:** medium, highest accuracy-gain per unit effort

### 7.4 Variance-Aware QP (T2.6)

- **Migration:** PuLP → cvxpy for the variance-aware path
- **Objective:** `maximize μᵀx + λ(win_prob) · Var(x)`
  - λ > 0 when trailing in matchup (variance-seeking — go for ceiling)
  - λ < 0 when winning (variance-minimizing — protect the lead)
  - λ continuous between -0.3 and +0.3 based on `win_probability` from H2H engine
- **Expected gain:** +3-5% win rate in close matchups (Mlčoch 2024)
- **Modified file:** `src/lineup_optimizer.py` adds `solve_variance_aware()` alongside `solve_deterministic()`

### 7.5 k=20 candidate lineups via epsilon-constraint

- **Existing infrastructure:** `src/optimizer/advanced_lp.py` already supports epsilon-constraint
- **Configure for k=20** in the call site
- **Output:** list of 20 distinct lineups, each within 5% of optimal objective
- **Passes to Selection Layer for re-ranking**

---

## 8. Selection Layer Design

### 8.1 LambdaMART re-ranker (T1.13)

- **Library:** XGBoost 2.x/3.x with `rank:ndcg` objective
- **Input:** k=20 LP-optimal lineups from Decision Engine
- **Training data:** historical `lineup_history` table (saved runs + actual outcomes)
- **Output:** Re-ranked top-1 lineup based on observed week-on-week category-wins
- **Closes the RMSE-optimal vs category-win-optimal gap that pure LP cannot**
- **New file:** `src/optimizer/lineup_ranker.py`

### 8.2 Vowpal Wabbit contextual bandits for streamers (T1.12)

- **Library:** `vowpalwabbit` Python bindings
- **Algorithm:** Thompson sampling / LinUCB
- **Domain:** streamer pickup decisions (only HEATER subproblem with enough trial volume — ~250-400/season)
- **Features:** 50+ per-FA-pitcher feature vectors already in `fa_recommender.py`
- **Online learning:** updates with each saved-run outcome
- **New file:** `src/optimizer/streamer_bandit.py`

### 8.3 Dirichlet-Multinomial FA Ownership Model (T1.14)

- **Math:** Haugh-Singal (2021 *Management Science*) opponent-modeling ported to FA market
- **Score:** `marginal_sgp × (1 - E[other_team_picks_up])`
- **12-team posterior converges fast** (small N)
- **Expected gain:** +10-15% pickup quality
- **Modified file:** `src/optimizer/fa_recommender.py` adds `_dirichlet_ownership_factor()`

### 8.4 LLM-as-judge for NL strategy constraints (T1.18)

- **Library:** Ollama-based local Mistral Small 3.2 24B
- **Pattern:** DIAMOND (ACL REALM'25; F1 lift 42.9 → 84.8)
- **Use case:** User types "punt SV this week" or "max K" — LLM-judge re-ranks the LambdaMART top-5 lineups against the natural-language strategy
- **Cost:** $0 (local)
- **Integration:** wraps `lineup_ranker.py` output before final selection

---

## 9. Explanation Layer Design

### 9.1 Mistral Small 3.2 24B lineup explainer (T1.16)

- **Model:** `ollama pull mistral-small3.2:24b-instruct-q4_K_M` (~14GB, 81% MMLU)
- **Throughput:** 3× faster than Llama 3.3 70B per Mistral AI benchmarks
- **Input:** LP slacks, DCV scores, forced-start flags, matchup multipliers from the v2 pipeline
- **Output:** 2-3 sentence "why X over Y" rationales grounded in HEATER's actual numbers (no hallucination — schema-constrained)
- **New file:** `src/explanation.py`
- **UI integration:** "Why this lineup?" expander on each player row

### 9.2 RAG over sabermetric corpus (T1.17)

- **Vector DB:** ChromaDB embedded
- **Embedding model:** BGE-M3 (multilingual, strong) OR `nomic-embed-text` for smaller
- **Framework:** LlamaIndex
- **Corpus:** FanGraphs Library (1000s of articles), Baseball Prospectus, Tom Tango blog, Russell Carleton's "Sample Size" series, MLBAM publications
- **Pipeline:** Chunk → embed → retrieve top-K → grounded LLM output (MEGA-RAG showed 40% hallucination reduction vs ungrounded)
- **New files:** `src/rag/corpus_builder.py`, `src/rag/retrieval.py`

### 9.3 LangGraph "Ask HEATER" tool-using agent (T2.8, T2.9)

- **Library:** LangGraph + Ollama backend
- **Tools (existing HEATER Python functions wrapped):**
  - `get_projection(player_id, date)`
  - `simulate_swap(player_a, player_b, date_range)`
  - `evaluate_trade(give_ids, get_ids)`
  - `recommend_fa_moves()`
  - `get_yahoo_data_service().get_standings()`
  - `optimize_lineup(date, scope, overrides)`
- **Two-tier architecture:**
  - **Llama 3.1 8B** routes user query → tool calls (fast, native tool-calling)
  - **Mistral Small 3.2 24B** composes natural-language reply from tool outputs
- **UI:** New "Ask HEATER" tab on Line-up Optimizer page; chat interface
- **New file:** `src/ask_heater.py`

---

## 10. Hybrid Weekly Engine Design

### 10.1 User flow

```
1. User opens Line-up Optimizer page
2. Date Picker prominently displayed at top: "Optimize for week of:" [calendar]
   - Default: current week
   - Allowed range: today through season end
3. User clicks "Optimize Week" button
4. Single compute runs: produces 7-day deployment plan
   - Each day's lineup (today through 7 days out)
   - Pitcher start-day assignments (each SP placed on their start day)
   - Util slot deployments (each Util player placed on play days)
   - Streaming pickup recommendations
   - Yahoo lineup mismatch warnings per day
5. Output: 7-day strip at top showing each day's planned lineup
6. User can click any day to drill into that day's detail
7. On the drill-down view, user can ADJUST SETTINGS FOR JUST THAT DAY:
   - Force-start / force-bench a player
   - Override matchup factor toggles (e.g., "ignore weather for this day")
   - Different risk lambda for this day vs rest of week
   - Save these as per-day overrides
8. Per-day overrides are persisted; re-optimize re-computes only the overridden day(s)
9. User clicks "Apply to Yahoo" — generates Yahoo lineup commands per day
10. User clicks "Save Result" — snapshots the full week to lineup_history
11. User clicks "Share" — exports to chosen endpoint (PNG/Markdown/iMessage/Gmail/Outlook/SMS)
```

### 10.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Date picker → week_start, week_end                          │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Weekly Optimizer (NEW: src/optimizer/weekly_engine.py)      │
│                                                             │
│   For each day in [week_start, week_end]:                  │
│     1. Build day-specific data context                      │
│     2. Get probable pitchers, game times, weather, etc.     │
│   Then run cross-day optimization:                         │
│     - Pitcher deployment LP (multi-day stochastic MIP)      │
│     - Per-day batter lineup LP                              │
│     - Util slot allocation across week                      │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: WeeklyDeploymentPlan dataclass                      │
│   .daily_lineups: dict[date, Lineup]                        │
│   .pitcher_schedule: dict[player_id, start_dates]           │
│   .streaming_recs: list[FAPickup]                           │
│   .ip_pace: dict[date, cumulative_ip]                       │
│   .yahoo_mismatches: dict[date, list[Mismatch]]             │
│   .confidence: dict[date, float]                            │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ UI renders:                                                 │
│   - Day strip (7 days, color-coded by lineup quality)       │
│   - Click any day → drill view with overrides allowed       │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 Per-day override system

- **Storage:** New SQLite table `daily_overrides(date, setting_key, value, created_at)`
- **Setting examples:**
  - `force_start: [player_id_1, player_id_2]`
  - `force_bench: [player_id_3]`
  - `ignore_factor: [weather, ump]`
  - `risk_lambda: 0.30` (this day only, vs week default)
- **Re-optimize semantics:** when user changes overrides, only the affected day(s) re-compute
- **UI:** sidebar panel on drill-down view with toggles, sliders, multi-selects

---

## 11. Save/Share Subsystem Design

### 11.1 Save (T2.11)

**New table:** `lineup_history`

```sql
CREATE TABLE lineup_history (
  id INTEGER PRIMARY KEY,
  saved_at TEXT NOT NULL,            -- ISO 8601 UTC
  week_start DATE NOT NULL,          -- the week being optimized
  user_team_id INTEGER,
  optimizer_version TEXT,            -- e.g. "v2.0.0"
  settings_json TEXT,                -- mode, risk, factor toggles
  inputs_snapshot_json TEXT,         -- projections, matchup factors at save time (for reproducibility)
  outputs_json TEXT,                 -- WeeklyDeploymentPlan serialized
  actual_outcomes_json TEXT,         -- filled in retroactively after the week
  notes TEXT                         -- user notes
);
```

**Why snapshot inputs:** without it, next week's data refresh invalidates last week's stored recommendation. Reproducibility requires the inputs.

**New module:** `src/save_share.py`
- `save_result(plan: WeeklyDeploymentPlan, settings: dict) -> int`
- `load_result(id: int) -> tuple[WeeklyDeploymentPlan, dict]`
- `list_saved_results(user_team_id: int) -> pd.DataFrame`
- `retroactive_outcome_fill(id: int)` — runs after the week ends to add actual_outcomes for backtest

### 11.2 Share — 6 endpoints

| Endpoint | Mechanism | Implementation cost | Notes |
|---|---|---|---|
| **PNG export** | Render the day-strip UI to PNG via Playwright or `streamlit-image-coordinates` | Low (1-2 days) | Save to disk, open in default viewer |
| **Markdown export** | Render plan as formatted markdown | Low (0.5 day) | Copy to clipboard or save to file |
| **Gmail** | `mailto:` link with body pre-filled (no auth needed) OR Gmail API with OAuth | Low (`mailto`) / High (API) | Start with `mailto:`; defer API |
| **Outlook** | Same as Gmail — `mailto:` link, or Microsoft Graph API | Low / High | Start with `mailto:` |
| **SMS / Text** | Twilio API ($0.0075/message US) OR Windows Share API for native handoff | Medium (Twilio) | Defer Twilio to v2.1; ship Windows Share API in v2.0 |
| **iMessage** | Genuinely hard from Windows. Workarounds: Beeper Cloud (web-based), BlueBubbles (self-hosted Mac relay). User has stated need so we ship Windows Share API + clipboard-copy + "paste into iMessage" instructions as v2.0 stopgap. | High (real iMessage) / Low (workaround) | Document the workaround in UI tooltip |

**v2.0 ship list:**
- PNG export (Playwright headless screenshot)
- Markdown export (clipboard + file)
- Gmail / Outlook via `mailto:` (opens user's default email client with pre-filled subject + body)
- Windows Share API (native intent — opens Windows share menu which includes installed iMessage relays, Messages app, etc.)

**v2.1 follow-up:**
- Twilio SMS (paid)
- Gmail API direct send (OAuth)

**New file:** `src/share/exporters.py`

### 11.3 UI

- "Save" button — saves current week's plan with optional notes
- "Share" dropdown — pick endpoint, opens system handler
- "History" tab — list of saved results, with backtested accuracy once actual outcomes come in

---

## 12. UI Redesign

### 12.1 Page layout

```
┌────────────────────────────────────────────────────────────────────────┐
│ HEATER  [logo]                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ LINE-UP OPTIMIZER  v2                                                  │
├─────────────────┬──────────────────────────────────────────────────────┤
│ CONTEXT SIDEBAR │ MAIN AREA                                            │
│ ─────────────── │ ─────────────                                        │
│ Team Hickey     │ ┌─ DATE PICKER ─────────────────────────────────────┐│
│ 22 active + 5IL │ │ Optimize for week of: [📅 May 22, 2026     ▼]   ││
│                 │ │   ◀ This Week  Next Week ▶                        ││
│ Matchup state   │ └────────────────────────────────────────────────────┘│
│ 3-7-2 LOSING    │                                                      │
│                 │ ┌─ TABS ────────────────────────────────────────────┐│
│ IP budget       │ │ Optimize · Start/Sit · Cat Analysis · Streaming  ││
│ 25.8 / 54       │ │ Ask HEATER  · History                            ││
│                 │ └────────────────────────────────────────────────────┘│
│ Win prob 29.9%  │                                                      │
│                 │ ┌─ DAY STRIP (7 days) ──────────────────────────────┐│
│ Settings        │ │ FRI│SAT│SUN│MON│TUE│WED│THU                      ││
│ - Mode: Full    │ │ ███│██▌│███│██░│░░░│██▌│███   (DCV quality bar) ││
│ - Risk: auto    │ │ Click any day to drill in                         ││
│ - Factors: all  │ └────────────────────────────────────────────────────┘│
│                 │                                                      │
│ Data freshness  │ ┌─ ACTIONS ──────────────────────────────────────────┐│
│ ✓ All sources   │ │ [Optimize Week]  [Apply to Yahoo]  [Save]  [Share]││
│                 │ └────────────────────────────────────────────────────┘│
│                 │                                                      │
│                 │ ┌─ SELECTED DAY DETAIL (when drilled in) ───────────┐│
│                 │ │ Selected: Saturday, May 23                       ││
│                 │ │   Batters table (with DCV + matchup + reason)    ││
│                 │ │   Pitchers table (with start probability)        ││
│                 │ │   Per-day override panel (force-start, etc.)     ││
│                 │ │   Why-this-lineup explanation (Mistral)          ││
│                 │ │   Yahoo mismatch warnings                        ││
│                 │ └────────────────────────────────────────────────────┘│
└─────────────────┴──────────────────────────────────────────────────────┘
```

### 12.2 New tabs

| Tab | Purpose |
|---|---|
| **Optimize** | Current main; redesigned with date picker + day strip + drill |
| **Start/Sit** | Existing; enhanced with confidence bands from MAPIE |
| **Cat Analysis** | Existing; **bug fixes from audit** (13th-place rank in 12-team league, AVG .169 misranked, etc.) |
| **Streaming** | Existing; enhanced with contextual bandit recommendations |
| **Ask HEATER** | **NEW** — LangGraph tool-using agent, chat interface |
| **History** | **NEW** — saved results with backtest accuracy |

### 12.3 Settings overhaul

The current sidebar has Mode (Quick/Standard/Full), Risk Aversion slider, plus implicit Scope on the main panel. **All three are unified into a single Settings panel** with:

- **Date range** (the date picker)
- **Risk preference:** auto (per-matchup-state, default) | manual slider | per-category
- **Factor toggles:** weather · platoon · park · umpire · catcher framing · recent form · bat tracking · Vegas totals · causal correction (each on/off, default all on)
- **Algorithm depth:** Quick · Standard · Full · PhD-grade (new — runs all 5 projection paths, hierarchical Log5, causal layer)
- **Override settings for this day only:** force start/bench, ignore specific factors

---

## 13. Module-Level Changes — Master File Map

### 13.1 New files (28 modules)

```
src/optimizer/
  marcel_bayesian.py             # PyMC Bayesian MARCEL                 (T1.11)
  tabpfn_oracle.py               # TabPFN-v2.5 zero-shot                (T1.3)
  tft_forecast.py                # Temporal Fusion Transformer          (T2.2)
  ou_recency.py                  # Ornstein-Uhlenbeck SDE              (T2.3)
  calibration.py                 # MAPIE CQR                             (T1.7)
  hier_log5.py                   # Hierarchical Bayesian Log5           (T2.1)
  robust_lp.py                   # Bertsimas-Sim Γ-budget               (T2.5)
  lineup_ranker.py               # LambdaMART re-ranker                 (T1.13)
  streamer_bandit.py             # Vowpal Wabbit contextual bandits     (T1.12)
  weekly_engine.py               # Hybrid weekly engine                 (T2.11)
  offline_rl.py                  # Offline IQL via d3rlpy               (T2.10)

src/data_sources/
  odds_api.py                    # The Odds API client                   (T1.6)
  bat_tracking.py                # Statcast bat-tracking ingest         (T1.20)

src/llm/
  news_extractor.py              # Ollama-based news extraction         (T1.4)
  lineup_explainer.py            # Mistral Small 24B explainer          (T1.16)

src/rag/
  corpus_builder.py              # Sabermetric corpus build              (T1.17)
  retrieval.py                   # RAG query interface                   (T1.17)

src/causal/
  causal_layer.py                # DoWhy back-door adjustment           (T1.15)

src/save_share/
  save_share.py                  # Save/load runs to SQLite             (T2.11)
  exporters.py                   # PNG/MD/share endpoints                (T2.11)

src/agent/
  ask_heater.py                  # LangGraph tool-using agent           (T2.8)
  conversational.py              # Two-tier Llama+Mistral chat          (T2.9)
```

### 13.2 Modified files

```
src/optimizer/projections.py        # Add build_enhanced_projections_v2 (5-path)
src/optimizer/constants_registry.py # Update ERA/WHIP stab + new entries (T1.8, T1.9)
src/optimizer/pipeline.py           # Stage 1-5 wire v2 paths; preserve v1 fallback
src/optimizer/matchup_adjustments.py # Plug in hier_log5 + causal_layer
src/optimizer/scenario_generator.py # Gaussian copula same-team corr   (T1.19)
src/optimizer/advanced_lp.py        # Wire robust_lp; epsilon-constraint k=20
src/optimizer/fa_recommender.py     # Dirichlet-Multinomial ownership   (T1.14)
src/lineup_optimizer.py             # HiGHS backend + gapRel=0.005     (T1.1, T1.2)
                                     # + Variance-Aware QP (cvxpy)      (T2.6)
src/data_bootstrap.py               # Add 5 new bootstrap phases
src/database.py                     # New tables: lineup_history, bat_tracking,
                                     #             news_events, odds_archive,
                                     #             daily_overrides
src/news_sentiment.py               # Replace regex with Ollama call    (T1.4)

pages/2_Line-up_Optimizer.py        # Major redesign: date picker, day strip,
                                     # per-day drill, new tabs, override panel,
                                     # save/share UI
```

### 13.3 Removed (none — backward compatibility)

No file deletion. v1 code paths preserved as fallbacks. Once v2 is validated for 2 weeks, dead code can be pruned in a follow-up commit.

---

## 14. Library Additions — requirements.txt diff

```
# Bayesian / Probabilistic
pymc>=5.13.0                # already in HEATER but version bump
numpyro>=0.15.0             # NEW
dynamax>=0.1.5              # NEW (Ornstein-Uhlenbeck SDE)
tinygp>=0.3.0               # NEW (Gaussian process kernels)
mapie>=0.8.5                # NEW (Conformal Quantile Regression)

# Foundation models
tabpfn>=2.5.0               # NEW (TabPFN-v2.5)
chronos-forecasting>=2.0.0  # NEW (Chronos-Bolt fallback)

# Forecasting / ML
pytorch-forecasting>=1.2.0  # NEW (TFT)
pytorch-lightning>=2.6.0    # NEW
torch>=2.5.0                # version bump (already in HEATER)
xgboost>=2.0.0              # already pinned; version bump

# Optimization
highspy>=1.7.0              # NEW (HiGHS open-source solver)
cvxpy>=1.5.0                # NEW (Variance-Aware QP)

# Causal inference
dowhy>=0.14                 # NEW (back-door adjustment)
econml>=0.15.0              # NEW (treatment effect estimation)
causal-learn>=0.1.3.6       # NEW (PC + FCI algorithms — one-time audit)

# Reinforcement learning
vowpalwabbit>=9.10.0        # NEW (contextual bandits)
d3rlpy>=2.5.0               # NEW (Offline IQL)

# RAG / Embeddings / Vector DB
chromadb>=0.5.0             # NEW (embedded vector DB)
llama-index>=0.11.0         # NEW (RAG orchestration)
sentence-transformers>=3.0  # NEW (BGE-M3 embeddings)

# Agent framework
langgraph>=0.2.0            # NEW (tool-using agent)
instructor>=1.5.0           # NEW (Pydantic+LLM retry-on-validation)

# Data sources
# the-odds-api          (no SDK; direct HTTP via requests)
# pybaseball-bat-tracking (pybaseball 2.2.7 covers; no new dep)

# Export / Share
playwright>=1.45.0          # NEW (PNG headless screenshot)
markdown-it-py>=3.0.0       # NEW (Markdown serializer)
# twilio                (defer to v2.1)
```

**Estimated total disk impact:** +~3-4 GB Python packages, +~20 GB Ollama models (5GB Llama 3.1 8B + 14GB Mistral Small 3.2 24B + ~1GB embedding models).

---

## 15. Phasing & Build Sequence

### Phase 1 — Quick Wins (Tier 1, 20 items, ~3-5 weeks)

Items grouped by dependency-free batches. Each batch can be built in parallel by separate development streams.

**Batch A — Solver + constants (1-2 days):**
- T1.1: PuLP → HiGHS backend
- T1.2: gapRel=0.005
- T1.8: ERA/WHIP stabilization PA update
- T1.9: Stuff+ + pitcher barrel downweight

**Batch B — Data ingest (4-6 days):**
- T1.5: MLB statsapi 5-min lineup polling
- T1.6: The Odds API free tier
- T1.20: Statcast bat tracking

**Batch C — Local LLM stack (3-5 days):**
- T1.4: Ollama llama3.1:8b structured news extractor
- T1.16: Mistral Small 3.2 24B lineup explainer
- T1.17: RAG corpus + retrieval
- T1.18: LLM-as-judge re-rank wrapper

**Batch D — Projection paths (5-8 days):**
- T1.3: TabPFN-v2.5 oracle
- T1.7: MAPIE Conformal Quantile Regression
- T1.11: PyMC Bayesian MARCEL

**Batch E — Selection layer (3-5 days):**
- T1.12: Vowpal Wabbit contextual bandits
- T1.13: LambdaMART re-ranker
- T1.14: Dirichlet-Multinomial FA ownership

**Batch F — Causal + correlation (3-5 days):**
- T1.10: Causal-DAG audit (one-time)
- T1.15: DoWhy back-door adjustment
- T1.19: Gaussian copula same-team correlation

**Batch G — OAuth decoupling (3-4 days):**
- T1.21: Decouple `yds.get_rosters(force_refresh=True)` from optimize click path. Use cached data unless explicit "Refresh Yahoo" pressed. Eliminates the 159s hang documented in the audit.

### Phase 2 — Foundational v2 (Tier 2, 11 items, ~8-12 weeks)

Order chosen to surface load-bearing pieces first:

1. **T2.11 Save/Share subsystem** (foundational — other features depend on saved runs for backtest/training data)
2. **T2.1 Hierarchical Bayesian Log5** (matchup quality core)
3. **T2.5 Bertsimas-Sim Γ-budget robust mode** (decision-engine alternative)
4. **T2.2 Temporal Fusion Transformer** (replaces linear recency)
5. **T2.6 Variance-Aware QP** (PuLP→cvxpy migration)
6. **T2.3 Ornstein-Uhlenbeck SDE recency** (replaces simple Kalman)
7. **T2.7 Hierarchical options framework** (architectural refactor)
8. **T2.10 Offline IQL** (requires saved-runs from T2.11)
9. **T2.8 LangGraph Ask HEATER** (depends on Mistral explainer T1.16)
10. **T2.9 Two-tier conversational agent** (depends on T2.8)
11. **T2.4 Pyomo + Gurobi 12 evaluation** (optional — only commit if HiGHS hits a wall on DRO scale)

### Phase 3 — v2.5+ Roadmap (Tier 3, deferred)

Documented in Appendix A.

---

## 16. Migration Path

**Backward compatibility:** All v1 code paths preserved. Feature flags gate v2 paths:

```python
# In CONSTANTS_REGISTRY
"use_v2_projection_stack": True,    # default ON in v2.0
"use_v2_causal_matchup": True,
"use_v2_robust_decision": True,
"use_v2_selection_ranker": True,
"use_v2_local_llm_explain": True,
"use_v2_hybrid_weekly": True,
```

**Rollout plan:**
1. Phase 1 ships behind feature flags defaulted OFF, available via CLI env var override
2. Phase 1 validated against 2 weeks of live league use
3. Flags default ON for Phase 1 release
4. Phase 2 ships behind same flag pattern, repeats
5. After 4 weeks of v2 stability, v1 fallbacks pruned

**Database migrations:**
- All new tables added via additive `init_db()` calls (no drops)
- Existing tables (`players`, `season_stats`, `league_rosters`, etc.) unchanged
- `lineup_history` and `daily_overrides` are pure additions

---

## 17. Testing Strategy

### 17.1 Structural-invariant tests (new)

Adding to existing 90+ `tests/test_no_*.py` and `tests/test_sf*.py` suite:

```
tests/test_v2_projection_stack_paths.py     # All 5 paths produce valid outputs
tests/test_v2_hier_log5_handedness.py        # Log5 handles all 4 batter/pitcher combos
tests/test_v2_causal_layer_refutation.py     # DoWhy refutation step passes
tests/test_v2_highs_solver_speedup.py        # HiGHS solves k=20 in <5s
tests/test_v2_save_share_roundtrip.py        # Save → load → identical output
tests/test_v2_per_day_overrides.py           # Override layer correctly limits re-compute
tests/test_v2_weekly_engine_no_empty_slots.py # Weekly engine fills all roster slots
tests/test_v2_bat_tracking_ingest.py         # Bat tracking phase produces non-zero rows
tests/test_v2_news_extractor_json.py         # Ollama news extractor returns valid Pydantic
tests/test_v2_dirichlet_ownership_convergence.py # 12-team posterior converges
```

### 17.2 Backtest framework (extended)

- Existing `src/optimizer/backtest_runner.py` extended to evaluate v2 vs v1 on historical FourzynBurn weeks
- Saved-run snapshots feed continuous backtesting
- Metrics: RMSE per cat, category wins per week, "would-you-have-won" simulator
- Output: `data/backtests/v2_vs_v1_$(date).json`

### 17.3 Sample-bias guards

- Per Agent 03's distribution-shift caveat: train/val/test splits MUST be temporal (e.g. 2018-2023 train / 2024 val / 2025-2026 test), never random
- Per Agent 03's data-leakage caveat: industry projections (Steamer/ZiPS) snapshotted with `as_of_date` for backtests

---

## 18. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **TabPFN-v2.5 license requires commercial license for paid deployment** | High | Med | HEATER is personal-use only; OK. Document so future commercial use triggers re-evaluation. |
| **PyMC/NumPyro NUTS compute time blocks page render** | Med | High | Cache MCMC posteriors per week; recompute only on data change. Async generation with progress bar. |
| **Ollama local LLM hangs Windows** | Med | Med | Ollama daemon health-check before each call; timeout 30s; fallback to regex on timeout. |
| **HiGHS doesn't actually beat CBC on HEATER's specific problem** | Low | Low | A/B benchmark in Phase 1 Batch A; revert if no speedup. |
| **Causal DAG audit reveals current code has wrong multiplier directions** | Med | High | Treat the audit output as a bug-discovery step; fix before v2 release. |
| **159s compute time recurs** | High (current bug) | High | Root-cause: Yahoo `force_refresh=True` in optimize click path. Decouple in T3.4 (now T2.x in v2 plan). |
| **Statcast bat tracking has gaps for early-season call-ups** | High | Med | Use Bayesian shrinkage; cold-start regime handled by TabPFN-v2.5. |
| **Distribution shift across 2023→2026 rule changes** | High | Med | Rule-era categorical feature in projection stack; annual retrain. |
| **Save/Share `mailto:` doesn't work in some browsers** | Med | Low | Fallback: copy-to-clipboard + manual paste. |
| **iMessage sharing genuinely hard from Windows** | High | Low | Ship Windows Share API + workaround instructions; defer real iMessage to BlueBubbles/Beeper exploration in v2.1. |

---

## 19. Open Questions

These are non-blocking but worth noting before implementation:

1. **Backtest training data:** We need ~3 seasons of saved-run history for LambdaMART training. Until that data exists, the re-ranker bootstraps from a simpler baseline (e.g. raw LP objective ranking). How aggressive should this bootstrap-training be?
2. **Daily Yahoo lineup save automation:** Currently the user manually saves lineups in Yahoo. Should v2 surface a "one-click apply" via Yahoo API (write permission)? — out of v2.0 scope but flagging.
3. **PyMC vs NumPyro vs JAX for hierarchical Log5:** Agent 02 recommended PyMC 5; Agent 01 mentioned NumPyro NUTS backend as fastest. Likely both work; pick PyMC for ergonomics unless benchmark shows PyMC>5s per matchup.
4. **TFT input window:** How many weeks of history? Default 13 weeks (one half-season); evaluate during Phase 2.
5. **RAG corpus license:** FanGraphs/BPro/Tango blog content — are we storing snippets locally legally? Need to verify Fair Use bounds. Mitigation: store only embeddings, fetch full text at retrieval-time. (Defer to legal-conservative interpretation.)
6. **Mistral Small 3.2 24B context length limit:** 32K context. Will lineup explanations exceed? Unlikely but worth a safety guardrail.
7. **Variance-Aware QP λ tuning:** What's the optimal sigmoid mapping `win_prob → λ`? Calibrate empirically via Phase 2 backtest.

---

## 20. Estimated Effort Summary

| Phase | Items | Effort estimate | Calendar weeks (1 dev) |
|---|---|---|---|
| Phase 1 (Tier 1) | 21 | 84-124 person-days | 3-5 |
| Phase 2 (Tier 2) | 11 | 200-300 person-days | 8-12 |
| **v2.0 total** | **32** | **284-424 person-days** | **3-4 months** |
| Phase 3 (Tier 3, v2.5+) | 3 | 100-200+ person-days | 4-8 weeks later |

---

## Appendix A: v2.5+ Roadmap (Tier 3 Moonshots)

Items intentionally deferred from v2.0:

### A.1 Wasserstein-DRO + CVaR via Mosek
- **Citation:** Esfahani & Kuhn 2018 + Agent 04 deep dive
- **What:** Replaces empirical CV approximation in scenario_generator.py with distributionally-robust optimization. Finite-sample performance guarantees vs current implicit Gaussian copula assumption.
- **Effort:** ~6-8 weeks
- **Cost:** Mosek 10 license ($2,250 perpetual + $562/yr maintenance)
- **Trigger to ship:** when v2.0 backtest data shows CVaR-based decisions consistently fail in real outcomes

### A.2 Custom Baseball-BERT Foundation Model
- **Citation:** Ahn et al. 2026 *Neural Sabermetrics* + Agent 03 deep dive
- **What:** Pretrain BERT-style encoder on Retrosheet (1916-2025) + Statcast (2015-2025) pitch sequences. Mean-pool to get player/season embeddings used as features in projection stack.
- **Effort:** ~12 weeks
- **Cost:** ~$50-100 in A100 rental for 48h pretraining
- **Trigger to ship:** when HEATER commits to multi-season productization and the embedding-as-feature gain justifies the investment

### A.3 Sports Game Odds Player Props ($99/mo)
- **What:** Player-level prop bets (batter Ks O/U, pitcher Ks O/U) — Vegas crowd-of-sharps market for individual player outcomes
- **Effort:** ~1 week to integrate
- **Cost:** $99/mo
- **Trigger to ship:** when user demand for player-prop-aware optimization is clear

### A.4 ~~OAuth-bypass refactor~~ — **Promoted to T1.21 in v2.0 Phase 1 Batch G** (user-approved 2026-05-22)

---

## Appendix B: Full Citation List

See individual agent reports in `.superpowers/brainstorm/36109-1779488367/research/` for complete citations. Master list available in `synthesis-final.md`. Key foundational references:

- **Bayesian:** Hollmann (TabPFN-v2.5, arXiv:2511.08667), Fonnesbeck (Bayesian MARCEL, PyMC-Labs 2024), Wieland-Mews-Langrock (Ornstein-Uhlenbeck, arXiv:2011.04384)
- **Causal:** Pearl (do-calculus), Adams et al. 2025 (Hierarchical Bayesian Log5 for MLB, arXiv:2511.17733), Sharma et al. (DoWhy library)
- **Forecasting:** Lim 2021 (TFT), Hollmann et al. 2025 (TabPFN-v2.5)
- **Optimization:** Bertsimas & Sim (Γ-budget robust), Esfahani & Kuhn 2018 (Wasserstein-DRO)
- **DFS:** Haugh & Singal (Management Science 2021, opponent modeling), Mlčoch 2024 (variance-aware QP)
- **RL:** Kostrikov 2021 (IQL, arXiv:2110.06169)
- **Sabermetrics:** Tango et al. *The Book*, Carleton "Sample Size" series (BPro), MLBAM Statcast bat-tracking methodology
- **LLM:** Mistral AI (Small 3.2 24B), Meta (Llama 3.1 8B), Mistral docs for tool calling

---

*End of design spec — 2026-05-22-heater-v2-optimizer-design.md*
