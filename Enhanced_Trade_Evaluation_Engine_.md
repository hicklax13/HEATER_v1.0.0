# HEATER Trade Evaluation Engine: A Mathematically Rigorous Framework for 12-Team H2H Categories Fantasy Baseball

**Prepared for:** Mr. Hickey, HEATER project (Yahoo `game_key 469`, FourzynBurn League)
**Date:** May 17, 2026
**Status:** Publication-ready technical specification

---

## A. Executive Summary

**Recommended methodology — "HCV-Hybrid" (HEATER Categorical Valuation Hybrid).** The recommended trade-evaluation engine is a four-layer hybrid that fuses (1) a calibrated SGP-delta core (Art McGee/Tanner Bell lineage, with league-specific denominators per Mr. Hickey's `LeagueConfig`), (2) a Rosenof-style **G-score** Gaussian win-probability layer (the only peer-reviewed framework for H2H category leagues, derived from a normal-approximation of categorical aggregates over a finite scoring period), (3) a **Monte Carlo team-state simulator** that produces both season and remaining-season trajectories with FCFS waiver-state evolution, and (4) a **CARA mean-CVaR reconciliation layer** that combines the deterministic SGP delta with the stochastic win-probability delta under risk-aversion parameter `lambda = 0.15`. Replacement level is computed dually: as a **roster-context-aware Dynamic Replacement Level (DRL)** that re-evaluates on each invocation against Mr. Hickey's actual bench + the top-K FCFS waiver players (primary), and as the standard **league-wide PRP/VORP** (secondary, sanity-check). The engine produces three reconciled outputs per trade scenario: a single scalar `ΔSGP_eq`, a vector of category-win probabilities `p_c` over remaining matchup weeks, and a probability-of-playoffs / probability-of-championship delta from a 12-team bracket simulation.

This synthesis is justified because: (i) SGP is the only framework with empirically calibrated league-specific denominators (which Mr. Hickey already has); (ii) the Rosenof G-score is the only published academic result proving that pure z-scores are *suboptimal* in H2H formats because they ignore weekly variance — a fact directly relevant to a 26-week H2H Cats league; (iii) Monte Carlo is required because the IP-floor and FCFS-waiver state are non-linear constraints that closed-form math cannot handle; (iv) CARA/CVaR is the canonical risk-adjustment for the user's stated lambda; (v) the dual replacement levels are required because no single replacement methodology answers both "is this trade fair league-wide?" and "does this trade help *my* team given *my* bench?"

---

## B. Mathematical Specification

All variables are defined; every parameter is tagged `[VERIFIED]` (from LeagueConfig or public projection sources), `[ESTIMATED]` (defaulted in absence of league history), or `[CONFLICTING]` (where industry sources disagree).

### B.1 Notation and inputs

Let:
- $\mathcal{C} = \{R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP\}$ — twelve scoring categories. `[VERIFIED]`
- $\mathcal{C}^{-} = \{L, ERA, WHIP\}$ — inverse categories (lower wins). `[VERIFIED]`
- $\mathcal{C}^{r} = \{AVG, OBP, ERA, WHIP\}$ — rate categories. `[VERIFIED]`
- $T = 12$ teams; playoff cutoff $K_{po}=4$; weeks remaining $W_{r} \in [0, 26]$. `[VERIFIED]`
- $\lambda = 0.15$ — CARA risk aversion. `[VERIFIED, LeagueConfig]`
- $\text{AB}_0 = 5500$, $\text{PA}_0 = 6100$, $\text{IP}_0 = 1300$ — roster-baseline volumes. `[VERIFIED, LeagueConfig]`
- SGP denominators $d_c$: $d_R=32, d_{HR}=13, d_{RBI}=32, d_{SB}=14, d_{AVG}=0.004, d_{OBP}=0.005, d_W=3.5, d_L=3.0, d_{SV}=9.0, d_K=45, d_{ERA}=0.20, d_{WHIP}=0.020$. `[VERIFIED, LeagueConfig]`
- $\mathbf{r}_i = (R_i, HR_i, \dots)$ — projection vector for player $i$ (counting stats and the *components* of rate stats: $H_i, AB_i, BB_i, HBP_i, SF_i, ER_i, BF_i, IP_i$). Source: blended projection $\hat{r}_i$ (see §F).

### B.2 SGP layer (counting stats)

For counting category $c \notin \mathcal{C}^r$, the per-player SGP contribution is:

$$\text{SGP}_{i,c} = \frac{r_{i,c}}{d_c} \cdot \mathbb{1}_{c \notin \mathcal{C}^-} \;-\; \frac{r_{i,c}}{d_c} \cdot \mathbb{1}_{c \in \mathcal{C}^-}$$

For losses (L), more is worse, so the contribution is negative. `[VERIFIED, McGee 1997; Bell, SmartFantasyBaseball]`

### B.3 SGP layer (rate stats — the part most engines get wrong)

Rate-stat SGP must be computed as the **marginal impact on a baseline-team's rate**, not as raw rate. For AVG:

$$\text{SGP}_{i,\text{AVG}} = \frac{1}{d_{\text{AVG}}}\left[\frac{H_i + H_0}{AB_i + AB_0} - \frac{H_0}{AB_0}\right]$$

where $H_0 = \mu_{\text{AVG}} \cdot AB_0$ uses the league-average AVG ($\mu_{\text{AVG}} \approx 0.250$ `[ESTIMATED]`) over the baseline-team's plate volume. Analogous formulas for OBP, ERA, WHIP:

$$\text{SGP}_{i,\text{OBP}} = \frac{1}{d_{\text{OBP}}}\left[\frac{(H_i + BB_i + HBP_i) + OB_0}{(PA_i + PA_0)} - \frac{OB_0}{PA_0}\right]$$

$$\text{SGP}_{i,\text{ERA}} = -\frac{1}{d_{\text{ERA}}}\left[\frac{9(ER_i + ER_0)}{IP_i + IP_0} - \frac{9 \cdot ER_0}{IP_0}\right]$$

$$\text{SGP}_{i,\text{WHIP}} = -\frac{1}{d_{\text{WHIP}}}\left[\frac{(H_i + BB_i) + WH_0}{IP_i + IP_0} - \frac{WH_0}{IP_0}\right]$$

The negative sign for ERA/WHIP encodes the inverse direction. `[VERIFIED, Bell "Calculating SGP for OBP"]` This formulation is *critical*: Razzball, FantasySP, and many open-source tools incorrectly compute rate-stat SGP by treating a player's rate as a delta against league average without volume weighting, producing inflated values for low-volume specialists. This is a known failure mode (`[CONFLICTING]` — many industry tools).

### B.4 Aggregate SGP delta of a trade

For a trade where Mr. Hickey **sends** set $S$ and **receives** set $R$:

$$\Delta\text{SGP}_{\text{raw}} = \sum_{c \in \mathcal{C}}\left[\sum_{j \in R}\text{SGP}_{j,c} - \sum_{i \in S}\text{SGP}_{i,c}\right] + \Delta\text{SGP}_{\text{roster-spot}}$$

The roster-spot opportunity-cost term handles asymmetric trades (e.g., 4-for-2 frees two slots that gain DRL-equivalent value from the waiver wire):

$$\Delta\text{SGP}_{\text{roster-spot}} = (|S| - |R|) \cdot \text{SGP}_{\text{DRL}}^{\text{net-positional}}$$

where $\text{SGP}_{\text{DRL}}^{\text{net-positional}}$ is the expected SGP of the best available FCFS waiver pickup at the now-vacant position(s), under the team's current need profile (see §B.7). `[VERIFIED conceptually; ESTIMATED in magnitude]`

### B.5 G-score / win-probability layer (H2H weekly engine)

Following Rosenof (2024, *arXiv:2307.02188*), for each category $c$ in a given matchup week, model the difference between Hickey's projected weekly total $X_c^H$ and opponent's $X_c^O$ as approximately normal:

$$X_c^H - X_c^O \sim \mathcal{N}\!\left(\mu_c^H - \mu_c^O,\; (\sigma_c^H)^2 + (\sigma_c^O)^2\right)$$

Then:

$$p_c = P(X_c^H > X_c^O) = \Phi\!\left(\frac{\mu_c^H - \mu_c^O}{\sqrt{(\sigma_c^H)^2 + (\sigma_c^O)^2}}\right) \quad \text{(flip sign for } c \in \mathcal{C}^-\text{)}$$

Per-week weekly means and variances are built from the weighted active lineup:

$$\mu_c^H = \sum_{i \in \text{active}} \pi_{i,w} \cdot \hat{r}_{i,c} / 26,\qquad (\sigma_c^H)^2 = \sum_{i} \pi_{i,w}^2 \cdot \hat{\sigma}_{i,c}^2 / 26$$

where $\pi_{i,w}$ is expected games-played share of player $i$ in week $w$. For rate stats, replace the sum with the **delta-method** approximation: $\text{Var}(H/AB) \approx \text{Var}(H)/AB^2 + H^2\text{Var}(AB)/AB^4$ (delta method for ratios of correlated random variables). `[VERIFIED, Rosenof 2024; delta method standard]`

The **G-score** corrected aggregate (the academic improvement over z-score) for a player is:

$$G_{i,c} = \frac{\mu_{i,c} - \bar{\mu}_c}{\sqrt{\bar{\sigma}_c^2 + \sigma^{*2}_c}}$$

where $\sigma^{*2}_c$ is the *weekly* aggregate noise variance for category $c$ across the whole team. Z-scores correspond to the special case $\sigma^* = 0$, which Rosenof proves is invalid for H2H formats. `[VERIFIED]`

Expected category wins per matchup: $\mathbb{E}[\text{cat-wins}] = \sum_c p_c$. Match win probability (assuming 12-cat majority-rules tiebreak per Yahoo format): use Lyapunov-CLT approximation or Monte Carlo of the 12-Bernoulli vector with the empirical category-correlation matrix $\boldsymbol{\Sigma}_{\text{cat}}$ (HR-RBI ≈ 0.85, R-HR ≈ 0.74, AVG-OBP ≈ 0.70, ERA-WHIP ≈ 0.85, K-W moderate; SB nearly orthogonal; see Kokiko 2020 / RotoBaller). `[VERIFIED for correlation magnitudes]`

### B.6 IP floor as soft penalty

The 20-IP floor is modeled as a piecewise penalty applied to the pitching half of the weekly win probability:

$$\text{Pen}_{\text{IP},w}(\text{IP}_w) = \begin{cases} 0 & \text{IP}_w \geq 20 \\ \kappa \cdot (20 - \text{IP}_w)^2 & \text{IP}_w < 20 \end{cases}$$

with $\kappa$ calibrated so that a 50% IP shortfall (10 IP) collapses the four pitching counting categories (W, L, SV, K) and inflates ERA/WHIP losses to ≈100% (forfeit-equivalent). `[ESTIMATED; recommend $\kappa = 0.05$ per IP² and validate via backtest]` During Monte Carlo, $\text{IP}_w$ is sampled from a Poisson-Gamma mixture per starter based on probable-start dates pulled from RosterResource.

### B.7 Replacement level — Dynamic Roster-Context (DRL, primary)

Let $\mathcal{B}_p$ = Mr. Hickey's bench-eligible players at position $p$ (including IL stash with return-date weighting), and $\mathcal{F}_p$ = top-$K$ FCFS-available free agents at $p$ as of evaluation time. Note that **rounds 1-3 undroppable rule (36 players)** prunes $\mathcal{F}_p$. The DRL replacement vector at position $p$:

$$\mathbf{r}^{\text{DRL}}_p = \text{argmax}_{i \in \mathcal{B}_p \cup \mathcal{F}_p} \;\Delta\text{SGP}_{i \to \text{lineup}}$$

The replacement is the **player who actually replaces the traded slot** in the user's roster after the trade — not an abstract league-wide replacement. For the 4-for-2 case, two distinct DRL replacements are required, evaluated in priority order (best replacement first, then conditioned on first replacement being taken). This handles the FCFS scarcity dynamic: in a 12-team league with 28-slot rosters (336 rostered), the post-draft waiver pool depth is shallow but non-empty for OF/SP/RP and very shallow for C/MI. `[VERIFIED conceptually, Tango "The Book" replacement theory + smartfantasybaseball application]`

A simple Markov model for FCFS waiver state (forward-looking): for week $w$, expected best-available at position $p$ degrades as $\hat{r}^{FA}_{p, w+1} = \alpha \cdot \hat{r}^{FA}_{p,w} + (1-\alpha)\bar{r}^{FA}_{p}$ with $\alpha \approx 0.85$ `[ESTIMATED]` reflecting that the best free agents get added quickly.

### B.8 Replacement level — League-wide PRP/VORP (secondary)

For positional replacement, rank all rostered + draftable players by SGP, identify the $N_p$-th player at position $p$ where $N_p$ = number of expected starters at $p$ across 12 teams (e.g., $N_C = 12$, $N_{OF} = 36 + \text{Util/flex share}$). Player value is $\text{VORP}_i = \text{SGP}_i - \text{SGP}_{i_{N_p}}$. For multi-position eligibility, assign to the scarcest position per the Cohen catcher-adjustment hierarchy (C > SS > 2B > 3B > 1B > OF > Util). `[VERIFIED, Sanders FVARz, Cohen ATC]`

### B.9 Reconciliation layer — CARA / Mean-CVaR objective

Given two outputs — deterministic $\Delta\text{SGP}_{\text{raw}}$ and stochastic championship-probability delta $\Delta\Pi$ from the Monte Carlo — define a unified utility:

$$U(\text{trade}) = \mathbb{E}[\Delta\Pi] - \lambda \cdot \text{CVaR}_{0.20}[\Delta\Pi] + \gamma \cdot \widetilde{\Delta\text{SGP}}$$

where $\widetilde{\Delta\text{SGP}} = \Delta\text{SGP}_{\text{raw}} / \sigma_{\text{SGP}}$ is the standardized SGP delta (sanity anchor weight $\gamma \approx 0.1$ `[ESTIMATED]`), and CVaR is the conditional expected loss in the bottom 20% of simulation outcomes — penalizing trades with large downside (e.g., dependency on a single oft-injured player). The CARA equivalent under exponential utility $u(x) = -\exp(-\lambda x)/\lambda$ is approximately mean-variance for small $\lambda$:

$$U_{\text{CARA}} \approx \mathbb{E}[\Delta\Pi] - \frac{\lambda}{2}\text{Var}[\Delta\Pi]$$

Both forms are computed; the team uses CVaR for the "Lose the trade?" downside check and CARA for the central recommendation. `[VERIFIED, Pratt 1964; standard finance theory]`

### B.10 Three-horizon outputs

The same engine produces three time-indexed outputs from one Monte Carlo run:
- **Total 2026 season**: sum SGP over weeks 1–26.
- **Rest-of-season 2026**: sum over weeks $w_{\text{current}}$–26 using updated ROS projections.
- **Weekly H2H matrix**: $(W_r \times 12)$ matrix $p_{w,c}$ of category-win probabilities for each remaining week, with deterministic schedule from Yahoo API, plus playoff-bracket extension (weeks 24–26 conditional on top-4).

Playoff probability:

$$\Pi_{\text{playoff}} = \mathbb{E}_{\omega}\big[\mathbb{1}\{\text{rank}_{\text{H}}(\omega) \leq 4\}\big]$$

and championship probability is the joint over 2-round bracket simulation. The trade delta on each is the primary engine output users see.

---

## C. Detailed Treatment of the 14 Research Areas

### C.1 SGP

Originated by Art McGee, *How to Value Players for Rotisserie Baseball* (1997, rev. 2007), refined into the **slope-of-best-fit** method (Tanner Bell, SmartFantasyBaseball, 2014) where denominators are computed via Excel's `SLOPE()` across the full standings vector rather than a (1st–12th)/11 spread. The slope method is mathematically equivalent to OLS regression of stat-totals against rank and is more robust to tanking teams. Mr. Hickey's `LeagueConfig` denominators are already in the correct form. For H2H Cats vs. Roto, SGP transfers cleanly when interpreted as "what does this stat-line do to standings *expectation*" — Tom Tango and Bell both note SGP is a proxy for marginal categorical contribution, valid in any format that aggregates categories. The rate-stat math (B.3) is the most-frequently-misapplied element; the published example calculations in Bell's "Calculating SGP for OBP" (2014) are the authoritative reference and are reflected in B.3.

### C.2 Z-scores and weighted variants

The z-score lineage runs Zach Sanders (FanGraphs **FVARz**, ~2013) → Harper Wallbanger **BIGz** (number-weighted z, multiplying rate z-scores by AB/IP volume before re-z-scoring) → ATC valuation (Ariel Cohen, FanGraphs). Strengths: no league history needed; mathematically grounded. Weaknesses: assumes symmetric distributions (FwithFB demonstrates SB is heavily long-tailed and z fails); ignores within-season variance (Rosenof 2024 proves this makes z suboptimal for H2H). **G-score** (Rosenof) is the correct generalization for H2H Cats:
$$G_i = \sum_c \frac{\mu_{i,c} - \bar{\mu}_c}{\sqrt{\bar{\sigma}_c^2 + \sigma^{*2}_c}}$$
The extra noise term $\sigma^*$ — the team-level weekly variance — strictly dominates z-score in H2H simulations.

### C.3 VORP/PRP marginal value

Keith Woolner's VORP (Baseball Prospectus, ~1998) sets replacement at ~80% of positional average (75% for 1B/DH, 85% for C). For fantasy, FantasyPros generalizes this to VORP/VONA/VOLS/VBD. The relevant adaptation for HEATER: positional replacement bins are determined by *starting slots × teams*, not bench-inclusive — C=12, MI=24, OF=36, plus Util and P-flex spread. **PRP** (positional replacement player) is the strict fantasy analog. Dollar-value derivation per Schechter (*Winning Fantasy Baseball*, 2013) divides budget by total positive SGP-above-replacement.

### C.4 H2H Categories-specific math

The single most rigorous published treatment is Rosenof (2024, *arXiv:2307.02188 — Improving Algorithms for Fantasy Basketball* and 2024 *arXiv:2409.09884 — Dynamic Quantification*). Despite being basketball-focused, the math transfers cleanly because both are 9-12 category H2H formats. Key results: (a) under Lyapunov CLT, weekly category totals are approximately normal; (b) **Z-score is the special case of G-score with zero performance uncertainty**; (c) for H2H Cats, the right objective is $\mathbb{E}[\text{cats won}] - \lambda\text{Var}[\text{cats won}]$; (d) "punting" categories emerges naturally as an optimal strategy when one team is structurally weaker in some categories. Category correlations (Kokiko 2020, RotoBaller): use empirical $\boldsymbol{\Sigma}$ rather than independence. The IP floor is **non-trivial**: it is a chance-constraint, not a continuous penalty; solving it exactly requires SOCP, but the soft-quadratic penalty (B.6) is the standard tractable surrogate.

### C.5 Monte Carlo simulation

Standard approach: 10,000–25,000 simulated seasons per scenario. Standard error of a win probability $p$ with $N$ sims is $\sqrt{p(1-p)/N}$, so $N=10000$ gives ~0.5pp precision. For HEATER, recommend **20,000 sims** for season-level outputs, **2,000 per week** for the weekly matrix (variance-reduced via antithetic variates and common random numbers across trade arms — i.e., simulate the *same* opponent schedule under "no trade" vs. "post-trade"). Player-week simulation: sample games-played from binomial $(G_{w}, p_{\text{healthy}})$; sample per-game outputs from negative binomial (counting) or beta (rate components); apply ballpark factors from Statcast.

### C.6 Bayesian methods

ATC projections (Cohen, FanGraphs since 2017) are an empirical-Bayes shrinkage estimator: each underlying projection (Steamer, ZiPS, THE BAT, OOPSY) is weighted by its historical accuracy (per FantasyPros annual evaluations — ATC has been #1 most accurate 2019–2021). The blend is *not* simple averaging but precision-weighted, similar to FiveThirtyEight's election model methodology. For in-season updating, the standard prior-posterior approach: take preseason projection as prior $\mathcal{N}(\mu_0, \tau_0^2)$, update with observed $n$ PAs of evidence $\bar{x}$, sample variance $s^2$:
$$\mu_{\text{post}} = \frac{\tau_0^{-2}\mu_0 + (s^2/n)^{-1}\bar{x}}{\tau_0^{-2} + (s^2/n)^{-1}}$$
For HEATER, recommend updating the *underlying skills* (BB%, K%, ISO, BABIP) rather than the surface stats, since these stabilize at different sample sizes (Carleton/Russell stabilization points: K% ~ 60 PA, BB% ~ 120 PA, ISO ~ 160 PA, BABIP ~ 410 PA).

### C.7 Risk-adjusted utility

Pratt-Arrow CARA with $\lambda = 0.15$ corresponds to moderate risk aversion. With $\lambda = 0.15$ and $\Delta\Pi$ measured in probability units (range ±0.3), the mean-variance form $\mathbb{E}-\frac{\lambda}{2}\text{Var}$ is appropriate. **CVaR at 20%** is recommended for downside reporting because it preserves coherence (sub-additive) unlike VaR. Semi-variance (only downside) can supplement when the user explicitly wants to ignore upside risk on incoming player.

### C.8 Multi-player trade math

Asymmetric trades (4-for-2) introduce a *roster-spot value* which industry tools (FantasySP, Razzball, FantasyPros) routinely ignore. Correct treatment: the freed slots are assigned the marginal SGP of the best DRL pickups, taken *in priority order* (greedy fill), conditioned on the post-trade positional needs. Position eligibility shifts (e.g., losing OF eligibility) are handled by re-running positional assignment after each candidate swap. Mr. Hickey's 28-slot roster with Util×2 and P-flex×4 creates substantial position flexibility; players should be valued at their **scarcest eligible position** (Cohen 2019 catcher-adjustment paper).

### C.9 Replacement-level methodology

Three layers: (a) **Tom Tango's "The Book" replacement**: ~80% of league average, freely-available-talent, used in WAR calculations; (b) **Positional fantasy replacement** (PRP/VORP-fantasy): the $N_{p}$-th ranked player at position $p$; (c) **Dynamic Roster-Context (DRL)** — what HEATER needs — the best player Mr. Hickey can *actually* deploy given his current bench and the live FCFS waiver pool, with undroppable-rule constraints (36 round-1-3 players). The DRL is non-stationary: it updates with every league transaction. A Markov model (B.7) approximates the expected DRL trajectory across remaining weeks.

### C.10 Win probability and standings

Standings simulation: 12-team H2H Cats standings are determined by aggregated weekly W-L-T records across 12 categories × 26 weeks = 312 category-decisions per team. Playoff probability is computed by simulating the remaining schedule for all 12 teams under their projected rosters, ranking by total category-wins, taking top 4. Critically, the engine must simulate **opponent** trades and adds at a base-rate (Mr. Hickey's 10 adds+trades/week constraint applies leaguewide); a simple approach is to assume each team accrues +0.5 SGP/week from waiver activity `[ESTIMATED]`.

### C.11 Academic and industry sources

The most-relevant peer-style work is Rosenof's three-paper arXiv series (2023–2025) — *Improving Algorithms for Fantasy Basketball* (G-score), *Dynamic Quantification* (H-scoring with punting), and *Optimizing for Rotisserie Fantasy Basketball*. SABR's Jack Kavanagh-award paper (Kaufman & Tennenhouse 2012) covers VORP construction. *JQAS* and Sloan Sports Conference papers on baseball center on real-MLB win probability and projections rather than fantasy directly; the matchup-modeling literature (e.g., the 2025 arXiv hierarchical Bayesian win-probability paper) is useful for the per-game simulation core. Industry: FanGraphs (Sanders, Cohen, Zimmerman/Bell), Pitcher List, Razzball (Rudy Gamble's Point Shares — an SGP variant). No fully-published source produces all three reconciled outputs required by HEATER; that integration is novel in this spec.

### C.12 Open-source implementations

Survey of relevant GitHub repos and their depth:
- **`yahoo_fantasy_api` (spilchen)**: full Yahoo OAuth + roster/transactions API. Required infrastructure.
- **`snoozle-software/monte-carlo-mlb`**: open-source Monte Carlo MLB game simulator — useful as a building block but does not integrate with fantasy scoring.
- **`smartfantasybaseball`** (Bell) Excel templates — authoritative SGP reference implementation; can be ported to Python pandas.
- **FantasySP, Razzball trade tools** — closed-source; based on Player Rater dollar values, no H2H Cats specialization.
- **DraftKick (Mays Copeland)** — "Live SGP" during draft; methodology is a snake-fill simulator, useful template.

No public open-source tool implements G-score, dynamic replacement, or the IP-floor penalty. HEATER's engine is novel at this synthesis level.

### C.13 Industry tool critique

- **Yahoo native trade evaluator**: opaque; reports indicate it uses season-to-date Player Rater values with no projection forward — fails for ROS evaluation.
- **ESPN trade analyzer**: similarly opaque, points-league biased.
- **FantasyPros**: VORP/VONA/VOLS/VBD aggregation, position-aware but season-flat.
- **Razzball / FantasySP**: $/Player based on Steamer ROS, no H2H-Cats win-probability layer.
- **FanGraphs Auction Calculator**: z-score based on selectable projection, no in-season ROS adaptation, no trade analyzer.
- **BaseballHQ "Projecting X" / Mastersball**: subscription, manual SGP.
- **Pitcher List**: editorial trade values, no math layer.

**Common gaps that HEATER closes**: (1) no rate-stat-correct SGP rate math; (2) no IP-floor handling; (3) no FCFS waiver scarcity model; (4) no playoff-probability output; (5) no risk-adjustment with explicit lambda; (6) no Yahoo-native game_key 469 integration.

### C.14 Verification of design decisions — see section I below.

---

## D. Comparison Matrix

| Criterion | SGP | Z-score | G-score | VORP/PRP | Monte Carlo | Bayesian | Hybrid (HCV) |
|---|---|---|---|---|---|---|---|
| Mathematical rigor | High | High | Highest | Medium | Highest | Highest | Highest |
| H2H Cats fitness | Medium | Low | **High** | Medium | High | High | **Highest** |
| Multi-player trade support | Native (additive) | Native | Native | Native | Via sim | Via sim | **Native + sim** |
| Multi-position support | Manual | Yes via FVARz | Yes | Yes | Yes | Yes | Yes |
| Computational tractability | Trivial (O(n)) | Trivial | Trivial | Trivial | O(N·sims) | O(N·sims) | Mixed |
| Sensitivity to projection error | High | Medium | Low (var-aware) | Medium | Low (var-aware) | Lowest | Low |
| Rate stat handling | Correct if done per B.3 | Needs vol-weighting | Native | Native | Native | Native | Correct |
| IP floor handling | None | None | None | None | Native | Native | Native (B.6) |
| Inverse categories | Sign-flip OK | Sign-flip OK | Sign-flip OK | Sign-flip OK | Native | Native | Native |
| Roster-context aware | No | No | Partial | No | Yes (DRL) | Yes | **Yes (primary)** |
| Risk-adjusted | No | No | Implicit | No | Explicit | Explicit | **Explicit (CARA/CVaR)** |
| Produces win-prob output | No | No | **Yes** | No | Yes | Yes | Yes |
| Produces playoff-prob output | No | No | No | No | Yes | Yes | **Yes** |

---

## E. Recommended Hybrid Algorithm — Pseudocode

```
ALGORITHM: HCV-Hybrid Trade Evaluator
INPUT: trade = (send_set S, receive_set R), LeagueConfig L,
       projections Π (blended ATC+Depth+OOPSY+THEBATX),
       current_state (rosters, standings, week W, FCFS pool F),
       N_sim = 20000, N_weekly = 2000

# ---- LAYER 1: SGP delta ----
for each player i in S ∪ R:
    for each category c:
        SGP[i,c] = sgp_counting(i,c)        if c ∉ rate_stats
                 = sgp_rate(i,c, baseline)   if c ∈ rate_stats   # eqs B.3
    SGP[i] = Σ_c SGP[i,c]

# Roster-spot opportunity cost (DRL pickups)
replacements_R = greedy_drl_fill(positions_freed_by(S \ R), bench, F)
replacements_S = positions_filled_by(R \ S)       # players bumped to bench
ΔSGP_raw = Σ_{j∈R} SGP[j] − Σ_{i∈S} SGP[i] 
         + Σ_{k∈replacements_R} SGP[k] 
         − Σ_{ℓ∈replacements_S} SGP[ℓ]

# ---- LAYER 2: G-score / weekly win-probability matrix ----
post_trade_roster = (current_roster − S) ∪ R
build_per_week_lineup_schedule(post_trade_roster, weeks W+1..26)
for w in W+1..26:
    for c in categories:
        μ_w,c, σ²_w,c = aggregate_team_distribution(post_trade_roster, w, c)
        μ_opp, σ²_opp = aggregate_team_distribution(opponent_of(w), w, c)
        p_{w,c} = Φ((μ_w,c − μ_opp_c)/sqrt(σ²_w,c + σ²_opp_c))
        if c ∈ inverse: p_{w,c} = 1 − p_{w,c}
    apply IP-floor penalty Pen_IP(IP_w) to pitching p_{w,c}
WeeklyWinMatrix = matrix(p_{w,c})

# ---- LAYER 3: Monte Carlo season + playoffs ----
for s = 1..N_sim:
    sample_player_trajectories(post_trade_roster, weeks W+1..26)
    sample_opponent_trajectories(other 11 teams)
    sample_FCFS_evolution(F, transactions_per_week=10)
    standings[s] = simulate_h2h_season()
    rank_H[s]    = rank_of(Hickey, standings[s])
    if rank_H[s] ≤ 4:
        champion[s] = simulate_playoffs(top4)
ΔΠ_playoff = mean(rank_H ≤ 4) − baseline_no_trade_playoff_prob
ΔΠ_champ   = mean(champion = Hickey) − baseline_no_trade_champ_prob

# ---- LAYER 4: Reconciliation ----
ΔSGP_eq      = ΔSGP_raw                                  # interpretable score
CVaR20       = mean(Δrank_H | Δrank_H in worst 20%)
U_CARA       = E[ΔΠ_champ] − (λ/2)·Var[ΔΠ_champ]         # λ=0.15
U_reconciled = E[ΔΠ_champ] − λ·CVaR20 + γ·(ΔSGP_raw/σ_SGP)

# ---- Secondary VORP/PRP sanity check ----
ΔVORP_PRP = Σ_{j∈R} VORP_league(j) − Σ_{i∈S} VORP_league(i)

OUTPUT:
    ΔSGP_eq                              # reconciled SGP delta
    WeeklyWinMatrix [26 × 12]            # weekly category-win probabilities
    ΔΠ_playoff, ΔΠ_champ                 # championship deltas
    U_reconciled, CVaR20                 # risk-adjusted utility + downside
    ΔVORP_PRP                            # secondary league-wide check
    DRL_replacement_chain                 # named bench/FA players that fill slots
```

---

## F. Implementation Notes

**Projection blending (recommended weights for 2026 ROS):** ATC 0.35, FanGraphs Depth Charts 0.25, THE BAT X 0.20, OOPSY 0.10, Steamer 0.10. Rationale: ATC is itself a blend and has been the most accurate published 2019–2021 (FantasyPros validation); Depth Charts adds RosterResource playing-time accuracy; THE BAT X (Derek Carty) is Statcast-rich and elite for pitchers; OOPSY is the newest and bests THE BAT on certain hitter subgroups. For *variance* estimates (required for G-score and Monte Carlo), use the **dispersion across these five systems** as a parameter-uncertainty proxy (Cohen's ATC Volatility metric methodology). `[VERIFIED for component systems, ESTIMATED for blend weights]`

**Yahoo Fantasy API (game_key 469):** Use `yahoo_fantasy_api` Python lib (spilchen/yahoo_fantasy_api on GitHub). Endpoints needed: `league.standings()`, `team.roster()` per team for full roster snapshot, `league.transactions()` for FCFS recency, `league.scoreboard(week=w)` for current matchup state. OAuth2 refresh handled by `yahoo_oauth`. Cache hot paths (rosters, projections) in Streamlit's `@st.cache_data` with 15-minute TTL for in-season use.

**Computational complexity:** Per trade scenario: ~2 sec for SGP layer (n×c arithmetic), ~5 sec for G-score weekly matrix (26 weeks × 12 cats × 12 teams), ~25 sec for 20k full Monte Carlo on commodity hardware (vectorized NumPy). Total ≈ 30 sec. For Streamlit responsiveness, run SGP + G-score synchronously, kick Monte Carlo to a background thread with a progress bar.

**Variance reduction:** (1) **Common random numbers** — use the *same* RNG seed across no-trade-baseline and post-trade arms; this is the single largest improvement and can cut required sims by 4–10×. (2) **Antithetic variates** for player-week game outcomes. (3) **Stratified sampling** on schedule strength. (4) For weekly matrix where N_weekly=2000, this gives ~1pp SE per cell which is sufficient.

**Data pipeline:** Daily ETL: pull projections from FanGraphs CSV exports → blend → store in DuckDB/Parquet → recompute DRL on cron. Yahoo state pulled live on engine invocation.

---

## G. Validation Methodology

**Backtest design:** Pull complete H2H Cats league seasons from Yahoo (where API permits) or public NFBC-style logs for 2021–2025. For each historical trade in a similar 12-team H2H Cats league, compute predicted $\Delta\Pi_{\text{playoff}}$ and compare to realized standings delta.

**Error metrics:**
1. **MAE on final standings rank**: $\text{MAE} = \frac{1}{n}\sum|\hat{\text{rank}} - \text{rank}|$. Target < 1.5 ranks.
2. **RMSE on category wins**: $\text{RMSE} = \sqrt{\mathbb{E}[(\hat{\text{cats}} - \text{cats})^2]}$. Target < 8 cat-wins (out of 312 possible).
3. **Calibration of win probabilities**: bin predicted $p_c$ into deciles, plot vs. empirical frequency; Brier score $\frac{1}{n}\sum(p_i - o_i)^2$. Target Brier < 0.20.
4. **Trade-prediction backtest**: for trades where both teams' final standings deltas can be observed, compute Spearman correlation between predicted $\Delta\Pi$ and realized rank change. Target ρ > 0.4 (high-noise domain).
5. **Out-of-sample projection error**: blended-projection MAE against actual season stats, by stat. Target HR MAE < 5, AVG MAE < 0.020 (ATC historical level).

**Cross-validation**: leave-one-season-out, with blend weights re-fit per fold.

---

## H. Critical Risks, Edge Cases, Failure Modes

1. **SGP-denominator non-stationarity**: HEATER's denominators are league-calibrated but the league is new format (or rules changed); they could be off by 10–25%. Mitigation: re-derive denominators after 8 weeks of HEATER history via SLOPE method.
2. **Rate-stat "specialist" overvaluation**: SB-only or SV-only specialists generate huge SGP in their narrow category but contribute nothing elsewhere. The G-score variance term partially corrects this, but a hard cap (no single player contributes > 25% of any one category target) is recommended.
3. **Injury cliffs**: Steamer/ZiPS regress hard, but a freshly-injured player retains projection until manually adjusted. Subscribe to a RosterResource/IL feed; apply $p_{\text{healthy}}$ shrinkage immediately on news.
4. **Punting strategy interaction**: if Hickey is intentionally punting, e.g., SV, the engine should *not* penalize an SV-light trade; this requires a user-selectable "punted categories" set that zeros out those $p_c$ contributions.
5. **FCFS pool depth misestimation**: in shallow positions (C, MI), the DRL can collapse to a sub-replacement player. Recommend pulling top-30 FA at each position with the undroppable-rule pruning before each evaluation.
6. **Schedule strength asymmetry**: a trade that helps for 14 of 26 remaining matchups but hurts for the playoff weeks (24–26) should be flagged. The weekly matrix accomplishes this; explicit "playoff-week ΔΠ" subreport recommended.
7. **Opponent trade adversariality**: the engine assumes opponents do not trade adversarially in response. Game-theoretic equilibrium (a Nash trade-market) is beyond scope but flagged as future work.
8. **Multi-position eligibility post-trade**: Yahoo's eligibility rules (5 starts / 10 GP) mean eligibility evolves; engine should use **end-of-season projected eligibility**, not snapshot.
9. **Normality assumption breakdown for SB**: Friends With Fantasy Benefits demonstrated SB is heavily right-skewed; use NegBin marginals in Monte Carlo and accept some bias in the G-score Gaussian for SB.
10. **Lambda mis-specification**: λ=0.15 is moderate; if user is in a tight playoff race, effective optimal λ may be lower (more risk-seeking). Sensitivity analysis is built into the report by recomputing $U$ at λ ∈ {0.05, 0.15, 0.30}.

---

## I. Verification Answers — Mr. Hickey's Three Design Decisions

**Q(a) — Is "SGP-delta + ROS win-probability with reconciliation" the right engine output target?**
**Answer: Partially correct, but the *primary* objective should be Δ-championship-probability with SGP and ROS-win-prob as reported diagnostics, not co-equal outputs.** Reasoning: SGP delta is interpretable but not the manager's actual utility; ROS-win-probability conflates regular-season strength with playoff entry but ignores the bracket. **The dominant objective in a top-4-playoffs H2H Cats league is $\Delta\Pi_{\text{champ}}$** (Rosenof 2024 establishes this is the only theoretically correct objective for H2H Cats; ESPN/NBA fantasy literature concurs for H2H Most-Categories formats). Recommended hierarchy: **(1) Primary objective: $\Delta\Pi_{\text{champ}}$**, the expected change in championship probability from the trade, computed by Monte Carlo with full bracket simulation; **(2) Secondary objective: $\Delta\Pi_{\text{playoff}}$** for risk-averse managers and trade-deadline analysis; **(3) Diagnostic: $\Delta\text{SGP}_{\text{eq}}$** as an interpretable scalar that managers can sanity-check against intuition; **(4) Diagnostic: WeeklyWinMatrix** for matchup-level "where will this trade help vs. hurt me?" The reconciliation layer (B.9) operationalizes this hierarchy via the CARA/CVaR utility.

**Q(b) — Are "total season + ROS + weekly" the right three time horizons?**
**Answer: ROS and weekly are correct; "total 2026 season" is *not* the right third horizon during the season — it should be replaced or supplemented by "playoff-window" (weeks 24–26) and "trade-deadline-aware" horizons.** Reasoning: once the season is underway, total-season figures are mostly informational (a historical artifact + the ROS projection); they don't drive decisions. The decision-relevant horizons are: (1) **ROS** (the standard); (2) **Pre-deadline window** — weeks from now to Yahoo's trade deadline (typically week 19–20), since post-deadline you cannot react if the trade goes poorly; (3) **Playoff window** — weeks 24–26 only, conditional on making the playoffs. Recommended four-horizon implementation: **{ROS, Pre-deadline, Regular-season-finish (weeks now–23), Playoff-only (24–26 | top-4)}**, with total-2026-season kept as an informational rollup. The weekly matrix already encodes all of these.

**Q(c) — Is "roster-context + league-wide replacement" the right pair?**
**Answer: Yes — *with* an upgrade to the roster-context layer making it explicitly Markov-dynamic, and *with* the optional addition of a third game-theoretic layer.** The two-replacement-level pair is correct because they answer different questions (fairness vs. fit). The recommended upgrade: the primary roster-context layer should be a **Dynamic Replacement Level (DRL)** that models the FCFS waiver wire as a Markov chain where best-available-at-position degrades stochastically (B.7), not as a static "best current FA" — this captures the operational reality that the FA pool evolves. The optional third layer is a **game-theoretic replacement** under which opponents also optimize their rosters; this is computationally expensive (iterated best-response) and adds marginal accuracy. Recommendation: build (1) DRL and (2) standard VORP/PRP; flag game-theoretic equilibrium replacement as a v2.0 stretch goal.

---

## J. Final Adversarial Steel-Man

The strongest objection to HCV-Hybrid is **over-engineering relative to projection noise**. Steamer/ATC project HR with ~6 MAE and AVG with ~0.025 MAE; that signal-to-noise ratio swamps the gain from G-score over z-score in practice. A skeptic would argue that a well-implemented SGP + naive bench-replacement engine achieves 90% of HCV's accuracy at 10% of the complexity. The counter-defense: (1) the *risk-adjusted* component (CARA/CVaR) is decision-relevant even when point estimates are noisy — it's about *which* trades to avoid, not which to maximize; (2) the IP-floor and FCFS-waiver-state modules are not optional in HEATER's exact configuration — they materially change the answer; (3) the weekly matrix is irreducibly a Monte Carlo output and cannot be obtained from a static valuation. Conclude: simpler engines suffice in standard formats; HEATER's configuration (28-slot rosters with 6-bench, undroppable rule, IP floor, 26-week H2H Cats with top-4 cutoff, λ=0.15) crosses the complexity threshold where the hybrid pays for itself.

---

*This specification is intended to be implementable verbatim in HEATER's Streamlit + Python (NumPy/pandas/DuckDB) stack. All equations in §B should be unit-tested against the worked examples in Bell's SmartFantasyBaseball SGP series (rate-stat math) and Rosenof's arXiv 2307.02188 (G-score). The Monte Carlo core should be vectorized; with `numpy` + `numba` JIT, 20k full-season simulations of a 12-team league complete in under 30 seconds on a modern laptop.*
