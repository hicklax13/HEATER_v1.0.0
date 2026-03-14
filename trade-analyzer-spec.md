# Trade Analyzer Algorithm Specification
# Medallion-Grade Fantasy Baseball Trade Engine
# Version: 3.0 | 12 Layers | 147 Signal Dimensions
#
# LOCATION: /docs/trade-analyzer-spec.md
# This is the AUTHORITATIVE source of truth for all trade analyzer math.
# Claude Code: implement EXACTLY what is specified here. Do NOT simplify,
# approximate, or substitute alternative methods unless a section explicitly
# offers a fallback.

---

## TABLE OF CONTENTS

1. System Overview
2. Data Sources and Schema
3. L0: Signal Harvesting (147 dimensions)
4. L1: Signal Processing
5. L2: Regime Detection
6. L3: Bayesian Projection Engine
7. L4: Dependency Tensor
8. L5: Injury and Availability Engine
9. L6: Granular Matchup Engine
10. L7: Roster Portfolio Optimizer (YOUR team)
11. L8: Game Theory Layer
12. L9: Dynamic Programming / Real Options
13. L10: Copula Monte Carlo (100K sims)
14. L11: Decision Output
15. League Configuration
16. Test Cases
17. Implementation Priority (6 Phases)

---

## 1. SYSTEM OVERVIEW

### Core Principle
The roster is a portfolio. The trade is a rebalancing action. The question is NOT
"is Player A better than Player B." The question is: "does this rebalancing move
MY portfolio closer to the efficient frontier across all 10 category dimensions,
given the current state of the league?"

### Architecture
```
Input: Trade proposal (players given, players received)
       + My current roster (from Yahoo API)
       + All 12 league rosters and standings (from Yahoo API)
       + External data feeds (Statcast, projections, weather, injury)

Pipeline: L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7 -> L8 -> L9 -> L10 -> L11

Output: Trade grade (A+ to F) with confidence interval
        Championship equity delta
        Category impact tensor (10-dim)
        VaR / CVaR risk metrics
        Sensitivity report
        Counter-offer suggestions
```

---

## 2. DATA SOURCES AND SCHEMA

### Yahoo Fantasy API (already integrated in this project)
```
Required data:
  - league/{league_id}/standings          -> category totals per team, W-L records
  - league/{league_id}/scoreboard         -> weekly matchup results
  - league/{league_id}/transactions       -> trade/add/drop history
  - team/{team_id}/roster/players         -> roster with positions, status
  - league/{league_id}/players;status=FA  -> free agent pool
  - league/{league_id}/settings           -> scoring categories, roster slots, trade deadline

Yahoo stat_id mapping (5x5 H2H):
  Hitting: 7=R, 12=HR, 13=RBI, 16=SB, 60=H/AB (compute AVG)
  Pitching: 28=W, 32=SV, 42=K, 26=ERA, 27=WHIP
```

### Baseball Savant (Statcast) via pybaseball
```python
from pybaseball import statcast, statcast_batter, statcast_pitcher, playerid_lookup

Per batter: launch_speed, launch_angle, barrel, hard_hit,
            xba, xslg, xwoba, xiso, sprint_speed, hp_to_1b

Per pitcher: release_speed, release_spin_rate, release_extension,
             pfx_x, pfx_z, stuff_plus, p_throws

Granularity: pitch-level, aggregated to rolling windows
```

### FanGraphs (via pybaseball)
```python
from pybaseball import batting_stats, pitching_stats

Required: Steamer ROS, ZiPS ROS, ATC, THE BAT X projections
          Park factors (by stat: HR, R, H, 2B, 3B, BB, SO)
          Historical league standings for SGP calibration
```

### Weather: Open-Meteo (free, no auth)
### MLB Schedule: statsapi.mlb.com (free, no auth)

---

## 3. L0: SIGNAL HARVESTING (147 Dimensions)

### Feature Groups

#### Group 1: Statcast Batted Ball (18 features per batter)
```python
BATTED_BALL_FEATURES = [
    'ev_mean',            # mean exit velocity
    'ev_p90',             # 90th percentile exit velocity
    'ev_std',             # exit velocity standard deviation
    'la_mean',            # mean launch angle
    'la_sweet_spot_pct',  # launch angle 8-32 degrees %
    'barrel_pct',         # barrel rate
    'hard_hit_pct',       # EV >= 95 mph %
    'pull_pct', 'center_pct', 'oppo_pct',  # spray distribution
    'gb_pct', 'fb_pct', 'ld_pct',          # batted ball type
    'xba', 'xslg', 'xwoba', 'xiso',       # Statcast expected stats
    'avg_hr_distance',                      # average HR distance (ft)
]
```

#### Group 2: Plate Discipline (12 features per batter)
```python
PLATE_DISCIPLINE_FEATURES = [
    'o_swing_pct',       # chase rate
    'z_swing_pct',       # zone swing rate
    'o_contact_pct',     # contact on pitches outside zone
    'z_contact_pct',     # contact on pitches in zone
    'swstr_pct',         # swinging strike %
    'csw_pct',           # called strike + whiff %
    'f_strike_pct',      # first pitch strike %
    'whiff_by_fb',       # whiff rate vs fastballs
    'whiff_by_brk',      # whiff rate vs breaking balls
    'whiff_by_offspeed', # whiff rate vs offspeed
    'zone_pct_seen',     # % of pitches seen in zone
    'chase_rate_2strike', # chase rate with 2 strikes
]
```

#### Group 3: Baserunning / Speed (6 features)
```python
SPEED_FEATURES = [
    'sprint_speed',             # ft/s
    'sprint_speed_delta',       # vs 30-day-ago (LEADING injury indicator)
    'hp_to_1b',                 # home to first time
    'sb_attempt_rate',          # SB attempts per time on 1B
    'sb_success_rate',          # SB success %
    'baserunning_runs',         # BsR from FanGraphs
]
```

#### Group 4: Statcast Pitching (20 features per pitcher)
```python
PITCHING_FEATURES = [
    'ff_avg_speed', 'ff_spin_rate', 'ff_ivb', 'ff_hb',  # 4-seam
    'sl_avg_speed', 'sl_spin_rate',                       # slider
    'cu_avg_speed', 'ch_avg_speed',                       # curve, change
    'extension',                                           # release extension
    'release_x_delta', 'release_z_delta',                 # release point drift
    'stuff_plus', 'location_plus', 'pitching_plus',       # composites
    'k_pct', 'bb_pct', 'hr_per_fb',                      # outcomes
    'gb_pct_pitcher', 'whiff_pct', 'csw_pct_pitcher',    # batted ball + whiff
]
```

#### Group 5: Context / Environment (15 features per player-game)
```python
CONTEXT_FEATURES = [
    'park_factor_hr', 'park_factor_r', 'park_factor_h',
    'temp_f', 'wind_speed_mph', 'wind_direction_relative',
    'humidity_pct', 'elevation_ft', 'day_night', 'turf_grass',
    'timezone_crossings', 'rest_days', 'travel_miles_3day',
    'lineup_position', 'protection_woba',  # wOBA of on-deck hitter
]
```

#### Group 6: Pitcher-Batter Matchup (10 features)
```python
MATCHUP_FEATURES = [
    'bvp_pa', 'bvp_woba',                        # career head-to-head
    'platoon_advantage',                           # +1/-1/0
    'batter_vs_fb_woba', 'batter_vs_brk_woba', 'batter_vs_off_woba',
    'pitcher_fb_usage', 'pitcher_brk_usage', 'pitcher_off_usage',
    'matchup_xwoba',                               # log5 prediction
]
```

#### Group 7: Roster / Lineup Context (14 features)
```python
ROSTER_CONTEXT_FEATURES = [
    'games_remaining', 'probable_starts_remaining',
    'innings_limit', 'closer_role_security', 'high_leverage_ip_share',
    'lineup_protection_delta',                     # wOBA with/without adjacent protection
    'catcher_framing_runs',                        # catcher-pitcher pair effect
    'team_win_pct', 'team_runs_per_game', 'bullpen_era',
    'callup_probability',                          # minor league prospect
    'il_10_probability', 'il_60_probability',      # injury risk
    'age',
]
```

#### Group 8: League / Market (12 features)
```python
LEAGUE_MARKET_FEATURES = [
    'yahoo_pct_owned', 'yahoo_adds_7day', 'yahoo_trend',
    'position_scarcity_index', 'waiver_replacement_value',
    'your_rank_by_cat',      # 10-element vector
    'your_total_by_cat',     # 10-element vector
    'marginal_sgp_by_cat',   # 10-element vector
    'gap_to_next_rank',      # 10-element vector
    'weeks_remaining', 'games_remaining_in_week',
    'trade_deadline_days_remaining',
]
```

---

## 4. L1: SIGNAL PROCESSING

### 1A. Exponential Decay Weighting

```python
def decay_weight(observation_date: date, today: date, lambda_param: float) -> float:
    days_ago = (today - observation_date).days
    return np.exp(-lambda_param * days_ago)

DECAY_LAMBDAS = {
    'statcast_ev':       0.020,  # half-life ~35 days
    'statcast_spin':     0.015,  # half-life ~46 days
    'plate_discipline':  0.012,  # half-life ~58 days
    'traditional_rate':  0.008,  # half-life ~87 days
    'sprint_speed':      0.005,  # half-life ~139 days
    'injury_history':    0.003,  # half-life ~231 days
    'season_counting':   0.000,  # no decay (cumulative stats)
}
```

### 1B. Wavelet Decomposition
```python
import pywt

def decompose_signal(time_series: np.ndarray, wavelet: str = 'db4', level: int = 3):
    """
    Separate: trend (true talent) + seasonal (schedule cycles) + noise (BABIP luck).
    Input: rolling 14-day stat rate (min 60 data points).
    """
    coeffs = pywt.wavedec(time_series, wavelet, level=level)
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet)[:len(time_series)]
    seasonal_coeffs = [np.zeros_like(coeffs[0])] + [coeffs[1]] + \
                      [np.zeros_like(c) for c in coeffs[2:]]
    seasonal = pywt.waverec(seasonal_coeffs, wavelet)[:len(time_series)]
    noise = time_series - trend - seasonal
    return trend, seasonal, noise
```

### 1C. Mutual Information Feature Selection
```python
from sklearn.feature_selection import mutual_info_regression

def select_features(X: np.ndarray, y: np.ndarray, threshold: float = 0.01):
    """Filter 147 features to those with non-zero predictive power."""
    mi_scores = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
    selected = np.where(mi_scores > threshold)[0]
    return selected, mi_scores

def greedy_conditional_mi(X, y, max_features=50, threshold=0.005):
    """Forward selection maximizing I(X_j; Y | X_selected)."""
    selected = []
    remaining = list(range(X.shape[1]))
    for _ in range(max_features):
        best_mi, best_j = -1, -1
        for j in remaining:
            if not selected:
                mi = mutual_info_regression(X[:, [j]], y, n_neighbors=5)[0]
            else:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(X[:, selected], y)
                residual = y - lr.predict(X[:, selected])
                mi = mutual_info_regression(X[:, [j]], residual, n_neighbors=5)[0]
            if mi > best_mi:
                best_mi, best_j = mi, j
        if best_mi < threshold:
            break
        selected.append(best_j)
        remaining.remove(best_j)
    return selected
```

### 1D. Kalman Filter for True Talent

```python
def kalman_true_talent(
    observations: np.ndarray,
    obs_variance: np.ndarray,
    process_variance: float,
    prior_mean: float,
    prior_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    State-space model:
      True_talent(t) = True_talent(t-1) + process_noise    [state]
      Observed(t) = True_talent(t) + obs_noise              [measurement]
    """
    n = len(observations)
    filtered_mean = np.zeros(n)
    filtered_var = np.zeros(n)
    pred_mean, pred_var = prior_mean, prior_variance

    for t in range(n):
        pred_var += process_variance
        K = pred_var / (pred_var + obs_variance[t])
        filtered_mean[t] = pred_mean + K * (observations[t] - pred_mean)
        filtered_var[t] = (1 - K) * pred_var
        pred_mean, pred_var = filtered_mean[t], filtered_var[t]

    return filtered_mean, filtered_var

def observation_variance(stat_type: str, sample_size: float) -> float:
    """Variance of observed rate given sample size."""
    BASE = {
        'ba': lambda n: 0.25 / max(n, 1),
        'hr_rate': lambda n: 0.03 / max(n, 1),
        'sb_rate': lambda n: 0.05 / max(n, 1),
        'era': lambda n: 15.0 / max(n, 1),
        'whip': lambda n: 0.5 / max(n, 1),
        'k_rate': lambda n: 0.20 / max(n, 1),
    }
    return BASE.get(stat_type, lambda n: 1.0 / max(n, 1))(sample_size)
```

---

## 5. L2: REGIME DETECTION

### 2A. Bayesian Online Changepoint Detection (BOCPD)

```python
class BOCPD:
    """
    Adams & MacKay (2007).
    Detects discrete changepoints: mechanical change, injury return, role change.
    Run on rolling 14-day xwOBA per player.
    When cp_prob > 0.7, post-changepoint data ONLY feeds projections.
    """
    def __init__(self, hazard_lambda=200, mu0=0, kappa0=1, alpha0=1, beta0=1):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.run_length_probs = np.array([1.0])
        self.suff_stats = [(mu0, kappa0, alpha0, beta0)]

    def update(self, x: float) -> tuple[float, np.ndarray]:
        n = len(self.run_length_probs)
        pred_probs = np.zeros(n)
        for r in range(n):
            mu, kappa, alpha, beta = self.suff_stats[r]
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_probs[r] = norm.pdf(x, loc=mu, scale=np.sqrt(pred_var))

        growth = self.run_length_probs * pred_probs * (1 - self.hazard)
        cp = np.sum(self.run_length_probs * pred_probs * self.hazard)
        new_probs = np.append(cp, growth)
        new_probs /= new_probs.sum()
        self.run_length_probs = new_probs

        new_stats = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]
        for r in range(n):
            mu, kappa, alpha, beta = self.suff_stats[r]
            kn = kappa + 1
            mn = (kappa * mu + x) / kn
            an = alpha + 0.5
            bn = beta + kappa * (x - mu)**2 / (2 * kn)
            new_stats.append((mn, kn, an, bn))
        self.suff_stats = new_stats

        return new_probs[0], new_probs  # cp_prob, full posterior
```

### 2B. Hidden Markov Model (4 States)

```python
from hmmlearn import hmm

def fit_player_hmm(obs_matrix: np.ndarray, n_states: int = 4):
    """
    obs_matrix: (n_timepoints, 3) = [xwOBA_14d, ev_p90_14d, barrel_pct_14d]
    States: 0=Elite, 1=Above-avg, 2=Below-avg, 3=Replacement
    """
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='full',
                            n_iter=200, random_state=42)
    model.startprob_ = np.array([0.1, 0.5, 0.3, 0.1])
    model.fit(obs_matrix)
    state_probs = model.predict_proba(obs_matrix)
    return model, state_probs[-1]  # model + current state probs

def regime_conditional_projection(current_probs, projections_by_state):
    """Mixture of ROS projections weighted by current state probability."""
    return sum(current_probs[s] * projections_by_state[s]
               for s in range(len(current_probs)))
```

---

## 6. L3: BAYESIAN PROJECTION ENGINE

### 3A. Bayesian Model Averaging

```python
SYSTEM_FORECAST_SIGMA = {
    'steamer':   {'hr':5.2, 'rbi':14.1, 'r':12.8, 'sb':4.5, 'avg':0.022,
                  'w':2.8, 'k':22.5, 'sv':6.1, 'era':0.65, 'whip':0.09},
    'zips':      {'hr':5.5, 'rbi':14.8, 'r':13.2, 'sb':4.8, 'avg':0.024,
                  'w':3.0, 'k':23.0, 'sv':6.5, 'era':0.70, 'whip':0.10},
    'atc':       {'hr':5.0, 'rbi':13.5, 'r':12.5, 'sb':4.3, 'avg':0.021,
                  'w':2.7, 'k':21.0, 'sv':5.8, 'era':0.62, 'whip':0.085},
    'the_bat_x': {'hr':5.3, 'rbi':14.0, 'r':12.9, 'sb':4.6, 'avg':0.023,
                  'w':2.9, 'k':22.0, 'sv':6.0, 'era':0.68, 'whip':0.092},
    'pecota':    {'hr':5.4, 'rbi':14.5, 'r':13.0, 'sb':4.7, 'avg':0.023,
                  'w':2.9, 'k':22.8, 'sv':6.3, 'era':0.67, 'whip':0.095},
}

def bayesian_model_average(ytd_stats, projections, prior_weights=None):
    """
    Posterior weights: P(Model_i | YTD) = P(YTD | Model_i) * P(Model_i) / Z
    Returns: posterior_weights, blended_projection, blended_variance
    """
    systems = list(projections.keys())
    if prior_weights is None:
        prior_weights = {s: 1.0/len(systems) for s in systems}

    log_liks = {}
    for sys in systems:
        ll = sum(norm.logpdf(ytd_stats.get(st,0), loc=projections[sys].get(st,0),
                             scale=SYSTEM_FORECAST_SIGMA[sys].get(st,10))
                 for st in ytd_stats if st in projections[sys])
        log_liks[sys] = ll

    mx = max(log_liks.values())
    posts = {s: np.exp(log_liks[s]-mx) * prior_weights[s] for s in systems}
    Z = sum(posts.values())
    pw = {s: posts[s]/Z for s in systems}

    blended, blended_var = {}, {}
    for st in ytd_stats:
        blended[st] = sum(projections[s].get(st,0)*pw[s] for s in systems)
        v_within = sum(pw[s]*SYSTEM_FORECAST_SIGMA[s].get(st,10)**2 for s in systems)
        v_between = sum(pw[s]*(projections[s].get(st,0)-blended[st])**2 for s in systems)
        blended_var[st] = v_within + v_between

    return pw, blended, blended_var
```

### 3B. Hierarchical Shrinkage
```python
def hierarchical_shrinkage(ytd_rate, pa, pop_mean, pop_var):
    """Bayesian shrinkage toward positional population mean."""
    obs_var = ytd_rate * (1 - ytd_rate) / max(pa, 1)
    shrink = pop_var / (pop_var + obs_var)
    post_mean = shrink * ytd_rate + (1 - shrink) * pop_mean
    post_var = 1.0 / (1.0/pop_var + pa/obs_var) if obs_var > 0 else pop_var
    return post_mean, post_var
```

### 3C. Aging Curves (Delta Method, Survivorship-Corrected)
```python
def compute_aging_curve(player_seasons_df, stat):
    """
    Paired delta: for each player at age A and A+1, compute change.
    Weight by harmonic mean of PA. Correct for survivorship at age 30+.
    """
    df = player_seasons_df.copy()
    df_next = df.copy()
    df_next['age'] -= 1
    df_next = df_next.rename(columns={stat: f'{stat}_next', 'pa': 'pa_next'})
    paired = df.merge(df_next[['player_id','age',f'{stat}_next','pa_next']],
                      on=['player_id','age'])
    paired['w'] = 2*paired['pa']*paired['pa_next']/(paired['pa']+paired['pa_next'])
    paired['delta'] = paired[f'{stat}_next'] - paired[stat]

    curve = {}
    for age in range(20, 42):
        ad = paired[paired['age']==age]
        curve[age] = np.average(ad['delta'], weights=ad['w']) if len(ad)>=20 else 0.0

    # Survivorship correction at 30+
    for age in range(30, 42):
        dropout = _compute_dropout_rate(player_seasons_df, age)
        if dropout > 0.05:
            curve[age] -= dropout * 0.5
    return curve
```

---

## 7. L4: DEPENDENCY TENSOR

### 4A. Vine Copula
```python
def fit_category_copula(historical_seasons_df):
    """
    Fit C-vine copula to 10 fantasy categories.
    Input: rows = player-seasons (min 500), cols = [R,HR,RBI,SB,AVG,W,K,SV,ERA,WHIP]
    Invert ERA/WHIP so higher=better before fitting.
    """
    from copulas.multivariate import VineCopula
    data = historical_seasons_df.copy()
    data['ERA'] = -data['ERA']
    data['WHIP'] = -data['WHIP']
    copula = VineCopula('center')
    copula.fit(data[CATEGORIES])
    return copula

def sample_correlated_stats(copula, player_marginals, n=1):
    """Draw n stat lines via copula + player-specific KDE marginals."""
    u = copula.sample(n)
    stats = np.zeros((n, 10))
    for i, cat in enumerate(CATEGORIES):
        if cat in INVERSE_CATEGORIES:
            stats[:,i] = -player_marginals[cat.lower()].ppf(1-u[:,i])
        else:
            stats[:,i] = player_marginals[cat.lower()].ppf(u[:,i])
    return stats
```

### 4B. Roster Concentration Risk
```python
def roster_concentration_hhi(roster_players):
    """Herfindahl index of MLB team exposure. High = correlated downside."""
    team_pa = {}
    for p in roster_players:
        team_pa[p.mlb_team] = team_pa.get(p.mlb_team, 0) + p.projected_pa
    total = sum(team_pa.values())
    return sum((pa/total)**2 for pa in team_pa.values())
```

---

## 8. L5: INJURY AND AVAILABILITY ENGINE

### 5A. Cox Proportional Hazards
```python
from lifelines import CoxPHFitter

def fit_injury_model(historical_il_data):
    """Covariates: age, prior_il_days_3yr, cumulative_pa, sprint_speed_delta, position."""
    cox = CoxPHFitter(penalizer=0.01)
    cox.fit(historical_il_data, duration_col='duration', event_col='event',
            formula='age + prior_il_days_3yr + cumulative_pa + sprint_speed_delta + C(position)')
    return cox
```

### 5B. Injury Duration (Semi-Markov)
```python
from scipy.stats import weibull_min

INJURY_DURATION = {
    'hamstring': {'shape':1.8, 'scale':18, 'loc':10},
    'oblique':   {'shape':2.0, 'scale':25, 'loc':15},
    'ucl':       {'shape':3.0, 'scale':365,'loc':300},
    'shoulder':  {'shape':1.5, 'scale':45, 'loc':15},
    'back':      {'shape':1.3, 'scale':30, 'loc':10},
    'other':     {'shape':1.5, 'scale':20, 'loc':10},
}

def sample_injury_duration(injury_type, frailty=1.0):
    p = INJURY_DURATION.get(injury_type, INJURY_DURATION['other'])
    return max(10, int(weibull_min.rvs(c=p['shape'], scale=p['scale']*frailty, loc=p['loc'])))
```

---

## 9. L6: GRANULAR MATCHUP ENGINE

```python
def log5_matchup(batter_rate, pitcher_rate, league_avg):
    """Odds ratio method for specific matchup."""
    bo = batter_rate / (1 - batter_rate)
    po = pitcher_rate / (1 - pitcher_rate)
    lo = league_avg / (1 - league_avg)
    mo = (bo * po) / lo
    return mo / (1 + mo)

LINEUP_SLOT_PA = {1:4.65, 2:4.55, 3:4.45, 4:4.35, 5:4.25,
                  6:4.15, 7:4.05, 8:3.95, 9:3.85}

def game_projection(batter, pitcher, park, weather, slot):
    """Full game-level stat projection for one batter."""
    base = log5_matchup(batter.xwoba, pitcher.xwoba_against, 0.315)
    platoon = 1.05 if batter.bats != pitcher.throws else 0.95
    temp_adj = 1 + 0.002 * (weather.temp_f - 72)
    wind_adj = 1 + 0.001 * weather.wind_out
    pa = LINEUP_SLOT_PA.get(slot, 4.0)
    xwoba = base * platoon * park.hr_factor * temp_adj * wind_adj
    return {
        'hr': estimate_hr_rate(xwoba)*pa, 'r': estimate_r_rate(xwoba,slot)*pa,
        'rbi': estimate_rbi_rate(xwoba,slot)*pa, 'sb': batter.sb_rate*pa*batter.obp,
        'h': estimate_h_rate(xwoba)*pa, 'ab': pa*(1-batter.bb_rate),
    }
```

---

## 10. L7: ROSTER PORTFOLIO OPTIMIZER (YOUR TEAM)

This is the layer that makes the analyzer personal to YOUR roster.

### 7A. Marginal Category Elasticity

```python
def compute_marginal_sgp(your_totals, all_team_totals, categories):
    """
    dE[standings_points] / d(your_stat) at YOUR current total.
    Non-linear: value depends on exact proximity to other teams.
    """
    marginal = {}
    for cat in categories:
        totals_sorted = sorted([all_team_totals[t][cat] for t in all_team_totals])
        your_val = your_totals[cat]

        if cat in ['era', 'whip']:  # lower is better
            better = sorted([t for t in totals_sorted if t < your_val], reverse=True)
        else:
            better = sorted([t for t in totals_sorted if t > your_val])

        if not better:
            marginal[cat] = 0.01  # already 1st
        else:
            gap = max(abs(better[0] - your_val), 0.001)
            marginal[cat] = 1.0 / gap  # 1 standings point per gap
    return marginal
```

### 7B. Category Gap Analysis + Punt Detection

```python
def category_gap_analysis(your_totals, all_team_totals, your_team_id, weeks_remaining):
    """
    Per category: rank, gap to each position, achievability, punt flag.
    PUNT = cannot gain any standings position in remaining weeks.
    """
    analysis = {}
    for cat in CATEGORIES:
        ranked = sorted(all_team_totals.items(),
                        key=lambda x: x[1][cat],
                        reverse=(cat not in INVERSE_CATEGORIES))
        your_rank = next(i+1 for i,(tid,_) in enumerate(ranked) if tid==your_team_id)

        weekly_rate = estimate_weekly_production(your_totals, cat)  # from your roster projection
        gainable = 0
        for target_rank in range(1, your_rank):
            gap = abs(ranked[target_rank-1][1][cat] - your_totals[cat])
            weeks_needed = gap / max(weekly_rate, 0.001)
            if weeks_needed <= weeks_remaining * 1.2:
                gainable += 1

        is_punt = (gainable == 0 and your_rank >= 10)
        analysis[cat] = {
            'rank': your_rank,
            'is_punt': is_punt,
            'marginal_value': 0.0 if is_punt else compute_marginal_sgp(
                your_totals, all_team_totals, [cat])[cat]
        }
    return analysis
```

### 7C. Integer LP Lineup Optimizer

```python
from scipy.optimize import linprog

def optimize_lineup(roster, position_slots, active_slots):
    """
    Max value subject to: positional eligibility, slot counts, active limit.
    Handles multi-position eligibility from Yahoo.
    """
    n_p = len(roster)
    positions = list(position_slots.keys())
    n_pos = len(positions)

    # Objective: maximize sum of player values (minimize negative)
    c = []
    for player in roster:
        for pos in positions:
            c.append(-player.value if (pos in player.eligible_positions or pos in ['UTIL','BN']) else 0)

    # Each player <= 1 position
    A_ub, b_ub = [], []
    for i in range(n_p):
        row = [0]*(n_p*n_pos)
        for j in range(n_pos):
            row[i*n_pos+j] = 1
        A_ub.append(row); b_ub.append(1)

    # Each position <= slot count
    for j, pos in enumerate(positions):
        row = [0]*(n_p*n_pos)
        for i in range(n_p):
            row[i*n_pos+j] = 1
        A_ub.append(row); b_ub.append(position_slots[pos])

    # Bounds: 0 or forced-0 for ineligible
    bounds = []
    for i, player in enumerate(roster):
        for j, pos in enumerate(positions):
            if pos in player.eligible_positions or pos in ['UTIL','BN']:
                bounds.append((0,1))
            else:
                bounds.append((0,0))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return -res.fun, res.x.reshape(n_p, n_pos)
```

### 7D. Bench Option Value
```python
def bench_option_value(weeks_remaining, streaming_sgp_per_week=0.15):
    """Value of an empty roster slot (streaming + hot FA potential)."""
    stream = streaming_sgp_per_week * weeks_remaining
    hot_pickup_prob = 0.15 * weeks_remaining
    hot_pickup_val = 0.5  # rough expected value
    return stream + hot_pickup_prob * hot_pickup_val * 0.5
```

### 7E. Roster Fragility
```python
def roster_fragility(roster, injury_model):
    """Total expected value loss from injuries across the roster."""
    total_fragility = 0
    for player in roster:
        p_il = predict_injury_probability(injury_model, player.features, horizon=30)
        backup_value = get_best_replacement(player.primary_position).value
        value_at_risk = (player.value - backup_value) * p_il
        total_fragility += value_at_risk
    return total_fragility
```

---

## 11. L8: GAME THEORY LAYER

```python
def estimate_opponent_valuations(all_teams, player):
    """Each opponent's willingness-to-pay based on THEIR category needs."""
    vals = {}
    for tid, team in all_teams.items():
        if tid == YOUR_TEAM_ID: continue
        their_marginals = compute_marginal_sgp(team.totals, all_teams, CATEGORIES)
        vals[tid] = sum(player.proj[c]*their_marginals[c] for c in CATEGORIES)
    return vals

def market_clearing_price(valuations):
    """Nash equilibrium = 2nd highest bidder."""
    s = sorted(valuations.values(), reverse=True)
    return s[1] if len(s)>=2 else (s[0] if s else 0)

def adverse_selection_discount(offering_manager_history):
    """Bayesian: P(flaw | offered). Calibrated from manager trade history."""
    p_flaw = 0.15
    if len(offering_manager_history) >= 3:
        under = sum(1 for t in offering_manager_history
                    if t['actual'] < t['projected'] * 0.8)
        p_flaw = under / len(offering_manager_history)

    p_off_flaw, p_off_ok = 0.60, 0.20
    p_flaw_given_off = (p_off_flaw*p_flaw) / (p_off_flaw*p_flaw + p_off_ok*(1-p_flaw))
    return 1 - (p_flaw_given_off * 0.25)
```

---

## 12. L9: DYNAMIC PROGRAMMING

```python
def bellman_rollout(roster, trade, weeks_rem, n_lookahead=2, n_sims=1000):
    """Approximate future trade option value via rollout."""
    immediate = compute_trade_surplus(roster, trade)
    post_roster = apply_trade(roster, trade)
    gamma = get_gamma(playoff_probability(post_roster))

    future_val = 0
    for _ in range(n_sims):
        r, cum = post_roster.copy(), 0
        for w in range(min(n_lookahead, weeks_rem)):
            ft = sample_plausible_trade(r)
            if ft and compute_trade_surplus(r, ft) > 0:
                cum += compute_trade_surplus(r, ft) * (gamma**(w+1))
                r = apply_trade(r, ft)
        future_val += cum
    return immediate + future_val / n_sims

def get_gamma(playoff_prob):
    if playoff_prob > 0.7: return 0.98
    if playoff_prob > 0.3: return 0.95
    return 0.85
```

---

## 13. L10: COPULA MONTE CARLO (100K SIMS)

```python
def run_monte_carlo_trade_eval(
    roster, trade, all_teams, copula, injury_model, schedule,
    n_sims=100_000, seed=42
):
    """
    Paired comparison: identical seeds for pre/post.
    Isolates causal trade effect.
    """
    rng = np.random.RandomState(seed)
    results_pre, results_post = [], []
    post_roster = apply_trade(roster, trade)

    for sim in range(n_sims):
        s = rng.randint(0, 2**31)
        results_pre.append(simulate_season(roster, all_teams, copula, injury_model, schedule, s))
        results_post.append(simulate_season(post_roster, all_teams, copula, injury_model, schedule, s))

    return compute_evaluation_metrics(results_pre, results_post)

def simulate_season(roster, all_teams, copula, injury_model, schedule, seed):
    """Single season sim: correlated draws, injuries, LP lineup per week, H2H."""
    rng = np.random.RandomState(seed)
    player_stats = {}
    for p in get_all_players(all_teams):
        player_stats[p.id] = sample_correlated_stats(copula, p.marginals, 1)[0]
        if rng.random() < predict_injury_probability(injury_model, p.features):
            dur = sample_injury_duration(sample_injury_type(p), p.frailty)
            player_stats[p.id] *= max(0, 1 - dur/p.games_remaining)

    wins, losses = 0, 0
    for week in schedule.remaining:
        avail = [p for p in roster if player_stats[p.id] is not None]
        my_totals = compute_week(avail, player_stats, schedule.games[week])
        opp_totals = compute_week(all_teams[schedule.opp[week]].roster, player_stats, schedule.games[week])
        w, l = count_cat_wins(my_totals, opp_totals)
        wins += w; losses += l

    made_playoffs = wins >= PLAYOFF_THRESHOLD
    won_chip = simulate_bracket(roster, player_stats, schedule, rng) if made_playoffs else False
    return SimResult(wins=wins, losses=losses, playoffs=made_playoffs, championship=won_chip)
```

---

## 14. L11: DECISION OUTPUT

```python
def compute_evaluation_metrics(pre, post):
    n = len(pre)
    ce_pre = np.mean([r.championship for r in pre])
    ce_post = np.mean([r.championship for r in post])
    ce_delta = ce_post - ce_pre
    ce_ci = 1.96 * np.std([b.championship-a.championship for a,b in zip(pre,post)]) / np.sqrt(n)

    deltas = [b.wins-a.wins for a,b in zip(pre,post)]
    var5 = np.percentile(deltas, 5)
    cvar5 = np.mean([d for d in deltas if d <= var5])

    p_helps = np.mean([b.championship>=a.championship for a,b in zip(pre,post)])
    gain = np.mean([b.championship-a.championship for a,b in zip(pre,post) if b.championship>=a.championship]) or 0.001
    loss = abs(np.mean([b.championship-a.championship for a,b in zip(pre,post) if b.championship<a.championship]) or 0.001)
    b_ratio = gain/loss
    kelly = max(0, min(1, (p_helps*b_ratio-(1-p_helps))/b_ratio))

    sharpe = np.mean(deltas)/max(np.std(deltas), 0.001)
    grade = _grade(ce_delta, sharpe, kelly)

    return {'grade':grade, 'ce_delta':ce_delta, 'ce_ci':ce_ci,
            'var5':var5, 'cvar5':cvar5, 'kelly':kelly, 'sharpe':sharpe}

def _grade(ce, sharpe, kelly):
    s = (ce*100)*0.4 + sharpe*0.3 + kelly*0.3
    for threshold, g in [(2.0,'A+'),(1.5,'A'),(1.0,'A-'),(0.7,'B+'),(0.4,'B'),
                         (0.2,'B-'),(0.0,'C+'),(-0.2,'C'),(-0.5,'C-'),(-1.0,'D')]:
        if s > threshold: return g
    return 'F'
```

---

## 15. LEAGUE CONFIGURATION

```python
LEAGUE_CONFIG = {
    'league_id': 'FILL_FROM_YAHOO',
    'num_teams': 12,
    'format': 'H2H',
    'team_name': 'My Dad Jonny Gomes',
    'categories': {
        'hitting': ['R','HR','RBI','SB','AVG'],
        'pitching': ['W','K','SV','ERA','WHIP'],
    },
    'roster_slots': {
        'C':1,'1B':1,'2B':1,'SS':1,'3B':1,'OF':3,'UTIL':1,
        'SP':6,'RP':3,'BN':5,'IL':3,'IL+':1,'NA':1,
    },
    'rounds': 23,
}
CATEGORIES = ['R','HR','RBI','SB','AVG','W','K','SV','ERA','WHIP']
INVERSE_CATEGORIES = ['ERA','WHIP']
YAHOO_STAT_IDS = {7:'R',12:'HR',13:'RBI',16:'SB',3:'AVG',28:'W',32:'SV',42:'K',26:'ERA',27:'WHIP'}
```

---

## 16. TEST CASES

### Test 1: Marginal SGP sanity
```
Your HR=150. Teams above: [152,158,170].
Expected: marginal_sgp(HR) ~ 0.5 (high, you're 2 HR from gaining a spot)
Your HR=200. Next team: 160.
Expected: marginal_sgp(HR) ~ 0.0 (dominant, no gain possible)
```

### Test 2: Punt detection
```
11th in SB, 30 SB total. 10th has 65 SB. 8 weeks left. No roster player >2 SB/week.
Expected: is_punt(SB) = True. SB marginal value = 0.
```

### Test 3: Adverse selection
```
Manager X: 5 trades, 3 underperformed by >20%.
Expected: discount < 0.90
```

### Test 4: Positional scarcity overrides raw stats
```
Trade: your 20 HR player for their 18 HR player.
But theirs fills SS (you have replacement-level). Yours is redundant OF.
Expected: ce_delta > 0 (positional value outweighs HR loss)
```

### Test 5: Bench option value
```
2-for-1 trade loses a bench spot.
Expected: surplus subtracts 0.5-1.5 SGP for lost streaming/flexibility value.
```

---

## 17. IMPLEMENTATION PRIORITY

### Phase 1: MVP (produces useful trade grades NOW)
1. Wire into existing Yahoo API client (already in this project)
2. FanGraphs Steamer ROS ingestion
3. Z-score + SGP valuation (L2 simplified)
4. YOUR roster's marginal category elasticity (L7A)
5. Category gap analysis + punt detection (L7C)
6. Roster LP optimizer (L7B)
7. Trade surplus + grade (L11 deterministic)
8. Streamlit page integration (Trade Analyzer page)

### Phase 2: Statistical Sophistication
9. Bayesian Model Averaging (L3A)
10. KDE marginals (replace Normal)
11. Vine copula (L4A)
12. Basic Monte Carlo 10K sims (L10)

### Phase 3: Signal Intelligence
13. Statcast harvesting (L0)
14. Signal decay (L1A)
15. Kalman filter (L1D)
16. BOCPD + HMM (L2)

### Phase 4: Context Engine
17. Game-level matchup (L6)
18. Injury stochastic process (L5)
19. Bench/flexibility value (L7D)
20. Concentration risk (L4B)

### Phase 5: Game Theory + Optimization
21. Opponent valuations (L8A)
22. Adverse selection (L8B)
23. Dynamic programming rollout (L9)
24. Sensitivity + counter-offer (L11)

### Phase 6: Full Production
25. Scale to 100K sims
26. Convergence diagnostics
27. Caching layer (precompute copula, SGP, hazards daily)
28. Full Streamlit UI polish
