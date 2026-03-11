"""Valuation engine: SGP, replacement level, VORP, marginal category value."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── League Configuration ─────────────────────────────────────────────


@dataclass
class LeagueConfig:
    """League settings that drive the valuation engine."""

    num_teams: int = 12
    roster_slots: dict = field(
        default_factory=lambda: {
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "OF": 3,
            "Util": 2,
            "SP": 2,
            "RP": 2,
            "P": 4,
            "BN": 5,
        }
    )
    hitting_categories: list = field(default_factory=lambda: ["R", "HR", "RBI", "SB", "AVG"])
    pitching_categories: list = field(default_factory=lambda: ["W", "SV", "K", "ERA", "WHIP"])
    # SGP denominators — defaults for 12-team 5x5 roto
    sgp_denominators: dict = field(
        default_factory=lambda: {
            "R": 32.0,
            "HR": 13.0,
            "RBI": 32.0,
            "SB": 14.0,
            "AVG": 0.004,
            "W": 3.5,
            "SV": 9.0,
            "K": 45.0,
            "ERA": 0.20,
            "WHIP": 0.020,
        }
    )
    risk_aversion: float = 0.15  # lambda for variance penalty

    @property
    def all_categories(self):
        return self.hitting_categories + self.pitching_categories

    @property
    def rate_stats(self):
        return {"AVG", "ERA", "WHIP"}

    @property
    def inverse_stats(self):
        """Stats where lower is better."""
        return {"ERA", "WHIP"}

    def hitter_starters_at(self, pos: str) -> int:
        """Number of league-wide starters at a hitting position."""
        if pos == "Util":
            return self.num_teams * self.roster_slots.get("Util", 0)
        return self.num_teams * self.roster_slots.get(pos, 0)

    def pitcher_starters(self) -> dict:
        """Estimated number of SP and RP started league-wide."""
        sp_dedicated = self.roster_slots.get("SP", 0)
        rp_dedicated = self.roster_slots.get("RP", 0)
        p_flex = self.roster_slots.get("P", 0)
        # Assume flex P slots split ~60/40 SP/RP in competitive leagues
        sp_total = sp_dedicated + int(p_flex * 0.6)
        rp_total = rp_dedicated + int(p_flex * 0.4)
        return {
            "SP": self.num_teams * sp_total,
            "RP": self.num_teams * rp_total,
        }


# ── SGP Calculator ───────────────────────────────────────────────────


class SGPCalculator:
    """Compute Standings Gain Points for players."""

    def __init__(self, config: LeagueConfig):
        self.config = config

    def player_sgp(self, player: pd.Series) -> dict:
        """Compute raw SGP per category for a single player."""
        sgp = {}
        for cat in self.config.all_categories:
            denom = self.config.sgp_denominators.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            stat_val = self._get_stat(player, cat)

            if cat in self.config.rate_stats:
                # Rate stats: need volume adjustment — handled in marginal SGP
                # For raw SGP, use volume-weighted contribution
                sgp[cat] = self._rate_stat_sgp(player, cat, denom)
            elif cat in self.config.inverse_stats:
                sgp[cat] = -stat_val / denom  # lower is better
            else:
                sgp[cat] = stat_val / denom
        return sgp

    def total_sgp(self, player: pd.Series) -> float:
        """Total SGP across all categories."""
        return sum(self.player_sgp(player).values())

    def marginal_sgp(self, player: pd.Series, roster_totals: dict, category_weights: dict = None) -> dict:
        """Compute marginal SGP contribution given current roster totals.

        roster_totals: dict with keys like 'R', 'HR', ..., 'ab', 'ip', 'h', 'er', etc.
        category_weights: optional dict of weights per category (from category balance).
        """
        sgp = {}
        for cat in self.config.all_categories:
            denom = self.config.sgp_denominators.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            weight = (category_weights or {}).get(cat, 1.0)

            if cat == "AVG":
                sgp[cat] = self._marginal_avg_sgp(player, roster_totals, denom) * weight
            elif cat == "ERA":
                sgp[cat] = self._marginal_era_sgp(player, roster_totals, denom) * weight
            elif cat == "WHIP":
                sgp[cat] = self._marginal_whip_sgp(player, roster_totals, denom) * weight
            elif cat in self.config.inverse_stats:
                sgp[cat] = -self._get_stat(player, cat) / denom * weight
            else:
                sgp[cat] = self._get_stat(player, cat) / denom * weight

        return sgp

    def _rate_stat_sgp(self, player: pd.Series, cat: str, denom: float) -> float:
        """Volume-weighted SGP for rate stats using a baseline roster assumption."""
        if cat == "AVG":
            ab = player.get("ab", 0) or 0
            h = player.get("h", 0) or 0
            if ab == 0:
                return 0
            # Approximate: assume league-average roster has ~5500 AB, .265 AVG
            roster_ab, roster_h = 5500, int(5500 * 0.265)
            new_avg = (roster_h + h) / (roster_ab + ab)
            old_avg = roster_h / roster_ab
            return (new_avg - old_avg) / denom
        elif cat == "ERA":
            ip = player.get("ip", 0) or 0
            er = player.get("er", 0) or 0
            if ip == 0:
                return 0
            roster_ip, roster_er = 1300, int(1300 * 3.80 / 9)
            new_era = (roster_er + er) * 9 / (roster_ip + ip)
            old_era = roster_er * 9 / roster_ip
            return -(new_era - old_era) / denom  # lower ERA = positive value
        elif cat == "WHIP":
            ip = player.get("ip", 0) or 0
            bb = player.get("bb_allowed", 0) or 0
            ha = player.get("h_allowed", 0) or 0
            if ip == 0:
                return 0
            roster_ip = 1300
            roster_whip_total = 1300 * 1.25  # ~1.25 WHIP baseline
            new_whip = (roster_whip_total + bb + ha) / (roster_ip + ip)
            old_whip = roster_whip_total / roster_ip
            return -(new_whip - old_whip) / denom
        return 0

    def _marginal_avg_sgp(self, player: pd.Series, roster: dict, denom: float) -> float:
        ab = player.get("ab", 0) or 0
        h = player.get("h", 0) or 0
        if ab == 0:
            return 0
        if abs(denom) < 1e-9:
            denom = 1.0
        r_ab = roster.get("ab", 0)
        r_h = roster.get("h", 0)
        if r_ab == 0:
            return (h / ab - 0.265) / denom if ab > 0 else 0
        old_avg = r_h / r_ab
        new_avg = (r_h + h) / (r_ab + ab)
        return (new_avg - old_avg) / denom

    def _marginal_era_sgp(self, player: pd.Series, roster: dict, denom: float) -> float:
        ip = player.get("ip", 0) or 0
        er = player.get("er", 0) or 0
        if ip == 0:
            return 0
        if abs(denom) < 1e-9:
            denom = 1.0
        r_ip = roster.get("ip", 0)
        r_er = roster.get("er", 0)
        if r_ip == 0:
            return 0
        old_era = r_er * 9 / r_ip
        new_era = (r_er + er) * 9 / (r_ip + ip)
        return -(new_era - old_era) / denom  # negative change in ERA = positive value

    def _marginal_whip_sgp(self, player: pd.Series, roster: dict, denom: float) -> float:
        ip = player.get("ip", 0) or 0
        bb = player.get("bb_allowed", 0) or 0
        ha = player.get("h_allowed", 0) or 0
        if ip == 0:
            return 0
        if abs(denom) < 1e-9:
            denom = 1.0
        r_ip = roster.get("ip", 0)
        r_bb = roster.get("bb_allowed", 0)
        r_ha = roster.get("h_allowed", 0)
        if r_ip == 0:
            return 0
        old_whip = (r_bb + r_ha) / r_ip
        new_whip = (r_bb + bb + r_ha + ha) / (r_ip + ip)
        return -(new_whip - old_whip) / denom

    @staticmethod
    def _get_stat(player: pd.Series, cat: str) -> float:
        mapping = {
            "R": "r",
            "HR": "hr",
            "RBI": "rbi",
            "SB": "sb",
            "AVG": "avg",
            "W": "w",
            "SV": "sv",
            "K": "k",
            "ERA": "era",
            "WHIP": "whip",
        }
        col = mapping.get(cat, cat.lower())
        val = player.get(col, 0)
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0


# ── Replacement Level ────────────────────────────────────────────────


def compute_replacement_levels(player_pool: pd.DataFrame, config: LeagueConfig, sgp_calc: SGPCalculator) -> dict:
    """Compute replacement-level SGP for each position.

    Returns dict: position -> replacement_sgp (float).
    """
    replacement = {}

    # Hitting positions
    hitting_positions = ["C", "1B", "2B", "3B", "SS", "OF"]
    for pos in hitting_positions:
        eligible = player_pool[
            player_pool["positions"].str.contains(pos, na=False) & (player_pool["is_hitter"] == 1)
        ].copy()
        if eligible.empty:
            replacement[pos] = 0
            continue

        eligible["total_sgp"] = eligible.apply(sgp_calc.total_sgp, axis=1)
        eligible = eligible.sort_values("total_sgp", ascending=False)

        n_starters = config.hitter_starters_at(pos)
        if len(eligible) > n_starters:
            replacement[pos] = eligible.iloc[n_starters]["total_sgp"]
        else:
            replacement[pos] = eligible.iloc[-1]["total_sgp"] * 0.5

    # Pitching positions
    pitcher_counts = config.pitcher_starters()
    for pos in ["SP", "RP"]:
        eligible = player_pool[
            player_pool["positions"].str.contains(pos, na=False) & (player_pool["is_hitter"] == 0)
        ].copy()
        if eligible.empty:
            replacement[pos] = 0
            continue

        eligible["total_sgp"] = eligible.apply(sgp_calc.total_sgp, axis=1)
        eligible = eligible.sort_values("total_sgp", ascending=False)

        n_starters = pitcher_counts[pos]
        if len(eligible) > n_starters:
            replacement[pos] = eligible.iloc[n_starters]["total_sgp"]
        else:
            replacement[pos] = eligible.iloc[-1]["total_sgp"] * 0.5

    # Util — use the deepest hitting replacement as baseline
    replacement["Util"] = min(replacement.get(p, 0) for p in hitting_positions)

    return replacement


def compute_vorp(player: pd.Series, sgp_calc: SGPCalculator, replacement_levels: dict) -> float:
    """Compute Value Over Replacement Player.

    Includes a multi-position flexibility premium: players eligible at
    multiple scarce positions are more valuable because they provide
    roster construction flexibility.
    """
    total_sgp = sgp_calc.total_sgp(player)
    positions = [p.strip() for p in str(player.get("positions", "Util")).split(",")]

    # Use the best (highest replacement level) position for VORP
    valid_positions = [p for p in positions if p in replacement_levels]
    if not valid_positions:
        best_repl = 0
    else:
        best_repl = max(replacement_levels.get(p, 0) for p in valid_positions)

    vorp = total_sgp - best_repl

    # Multi-position flexibility bonus: each additional eligible position
    # adds value since the player can fill multiple roster needs.
    # Scale bonus by position scarcity (shallow positions like C, SS get more bonus).
    scarce_positions = {"C", "SS", "2B"}  # typically shallow positions
    num_eligible = len(valid_positions)
    if num_eligible > 1:
        scarce_count = sum(1 for p in valid_positions if p in scarce_positions)
        vorp += 0.12 * (num_eligible - 1) + 0.08 * scarce_count

    return vorp


# ── Category Balance ─────────────────────────────────────────────────


def compute_category_weights(
    roster_totals: dict,
    config: LeagueConfig,
    target_rank: float = 5.0,
    draft_progress: float = 0.0,
    league_totals: list = None,
) -> dict:
    """Compute per-category weights based on current roster strength.

    Categories where the roster is weak get higher weight.
    Categories where the roster is already strong get lower weight.

    Args:
        roster_totals: User's current roster stat totals.
        config: League configuration.
        target_rank: Target standings rank (lower = more aggressive).
        draft_progress: 0.0 = draft start, 1.0 = draft end. Controls how
            aggressively weak categories are boosted — early draft keeps
            weights flat, late draft amplifies imbalances.
        league_totals: Optional list of dicts with all teams' stat totals.
            When provided, computes projected standings rank per category
            instead of comparing to static benchmarks.
    """
    # League average stat totals per team (approximate for 12-team 5x5)
    league_avg = {
        "R": 780,
        "HR": 200,
        "RBI": 760,
        "SB": 110,
        "AVG": 0.262,
        "W": 70,
        "SV": 60,
        "K": 1100,
        "ERA": 3.90,
        "WHIP": 1.25,
    }

    weights = {}
    for cat in config.all_categories:
        current = roster_totals.get(cat, 0)

        if league_totals and len(league_totals) > 1:
            # Standings-based: estimate rank by comparing to other teams
            other_vals = [t.get(cat, 0) for t in league_totals]
            if cat in config.inverse_stats:
                # Lower is better: count how many teams we beat
                rank = sum(1 for v in other_vals if v > 0 and (current == 0 or v < current)) + 1
            else:
                rank = sum(1 for v in other_vals if v > current) + 1
            # rank 1 = best, num_teams = worst
            # Convert rank to weight: bad rank = high weight
            rank_ratio = rank / config.num_teams  # 0 = best, 1 = worst
            weight = min(2.0, max(0.2, 0.5 + 1.5 * rank_ratio))
        else:
            avg = league_avg.get(cat, 1)

            if cat in config.inverse_stats:
                if current == 0:
                    ratio = 0.5
                else:
                    ratio = current / avg
            else:
                if avg == 0:
                    ratio = 1.0
                else:
                    ratio = current / avg

            if cat in config.inverse_stats:
                weight = min(2.0, max(0.2, ratio * 1.2))
            else:
                weight = min(2.0, max(0.2, 2.0 - ratio))

        weights[cat] = weight

    # Draft progress scaling: early = flatten toward 1.0, late = amplify
    # At progress=0: scale differences by 0.4 (nearly flat)
    # At progress=1: scale differences by 1.5 (strongly directional)
    progress_scale = 0.4 + 1.1 * draft_progress
    for cat in weights:
        weights[cat] = 1.0 + (weights[cat] - 1.0) * progress_scale

    return weights


# ── Full Player Valuation ────────────────────────────────────────────


def value_all_players(
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    roster_totals: dict = None,
    category_weights: dict = None,
    replacement_levels: dict = None,
    current_round: int = None,
    num_rounds: int = 23,
) -> pd.DataFrame:
    """Compute full valuations for all players in the pool.

    Returns the player_pool DataFrame with added columns:
    - total_sgp, vorp, marginal_sgp, pick_score

    Args:
        current_round: Current draft round (1-indexed). Used for bench
            optimization and projection confidence weighting.
        num_rounds: Total rounds in the draft.
    """
    sgp_calc = SGPCalculator(config)

    # Compute replacement levels if not provided
    if replacement_levels is None:
        replacement_levels = compute_replacement_levels(player_pool, config, sgp_calc)

    pool = player_pool.copy()

    # Compute SGP and VORP for each player
    pool["total_sgp"] = pool.apply(sgp_calc.total_sgp, axis=1)
    pool["vorp"] = pool.apply(lambda p: compute_vorp(p, sgp_calc, replacement_levels), axis=1)

    # Compute marginal SGP if roster totals are available
    if roster_totals:
        if category_weights is None:
            category_weights = compute_category_weights(roster_totals, config)

        def calc_marginal(player):
            m = sgp_calc.marginal_sgp(player, roster_totals, category_weights)
            return sum(m.values())

        pool["marginal_sgp"] = pool.apply(calc_marginal, axis=1)
    else:
        pool["marginal_sgp"] = pool["vorp"]

    # Pick score = marginal SGP (urgency added later by simulation/draft engine)
    pool["pick_score"] = pool["marginal_sgp"]

    # Apply injury risk discount if the column is present
    if "injury_risk" in pool.columns:
        pool["injury_discount"] = 1.0 - (config.risk_aversion * pool["injury_risk"].fillna(0))
        pool["pick_score"] = pool["pick_score"] * pool["injury_discount"]

    # Projection confidence: players with more projected playing time have
    # more reliable projections. Lightly discount low-volume players.
    pa_col = "pa" if "pa" in pool.columns else ("ab" if "ab" in pool.columns else None)
    if pa_col:
        pa = pool[pa_col].fillna(0)
        ip = pool["ip"].fillna(0) if "ip" in pool.columns else pd.Series(0, index=pool.index)
        volume = np.where(pool["is_hitter"] == 1, pa / 650.0, ip / 200.0)
        confidence = np.clip(0.8 + 0.2 * volume, 0.8, 1.0)
        pool["pick_score"] = pool["pick_score"] * confidence

    # Late-draft bench optimization: prioritize roster flexibility
    if current_round is not None:
        draft_progress = current_round / num_rounds
        if draft_progress > 0.7:

            def _pos_count(pos_str):
                return len([p for p in str(pos_str).split(",") if p.strip()])

            flex_bonus = pool["positions"].apply(_pos_count) * 0.08 * (draft_progress - 0.7) / 0.3
            pool["pick_score"] = pool["pick_score"] + flex_bonus

    return pool.sort_values("pick_score", ascending=False)


# ── SGP Denominator Auto-Computation ────────────────────────────────


def compute_sgp_denominators(player_pool: pd.DataFrame, config: LeagueConfig) -> dict:
    """Auto-compute SGP denominators from player pool statistics.

    Simulates team construction by distributing top players across teams
    and measuring the standard deviation of team category totals.
    SGP denominator = stddev of team totals (one sigma = one standing point).
    """
    n = config.num_teams
    hitters = player_pool[player_pool["is_hitter"] == 1].copy()
    pitchers = player_pool[player_pool["is_hitter"] == 0].copy()
    denoms = {}

    # Counting stats for hitters
    top_n_h = n * 10
    for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
        if col not in hitters.columns:
            denoms[cat] = config.sgp_denominators.get(cat, 1.0)
            continue
        vals = hitters[col].fillna(0).sort_values(ascending=False).head(top_n_h).values
        if len(vals) >= n:
            team_totals = [vals[i::n].sum() for i in range(n)]
            denoms[cat] = max(1.0, float(np.std(team_totals)))
        else:
            denoms[cat] = config.sgp_denominators.get(cat, 1.0)

    # AVG: compute team-level batting averages
    if "ab" in hitters.columns and "h" in hitters.columns:
        top_h = hitters.nlargest(top_n_h, "ab")
        if len(top_h) >= n:
            team_avgs = []
            for i in range(n):
                team = top_h.iloc[i::n]
                total_ab = team["ab"].fillna(0).sum()
                total_h = team["h"].fillna(0).sum()
                if total_ab > 0:
                    team_avgs.append(total_h / total_ab)
            if len(team_avgs) >= 2:
                denoms["AVG"] = max(0.001, float(np.std(team_avgs)))
            else:
                denoms["AVG"] = config.sgp_denominators.get("AVG", 0.004)
        else:
            denoms["AVG"] = config.sgp_denominators.get("AVG", 0.004)
    else:
        denoms["AVG"] = config.sgp_denominators.get("AVG", 0.004)

    # Counting stats for pitchers
    top_n_p = n * 8
    for cat, col in [("W", "w"), ("SV", "sv"), ("K", "k")]:
        if col not in pitchers.columns:
            denoms[cat] = config.sgp_denominators.get(cat, 1.0)
            continue
        vals = pitchers[col].fillna(0).sort_values(ascending=False).head(top_n_p).values
        if len(vals) >= n:
            team_totals = [vals[i::n].sum() for i in range(n)]
            denoms[cat] = max(0.5, float(np.std(team_totals)))
        else:
            denoms[cat] = config.sgp_denominators.get(cat, 1.0)

    # ERA and WHIP: team-level rate stats
    if "ip" in pitchers.columns:
        top_pit = pitchers.nlargest(top_n_p, "ip")
        if len(top_pit) >= n:
            team_eras, team_whips = [], []
            for i in range(n):
                team = top_pit.iloc[i::n]
                total_ip = team["ip"].fillna(0).sum()
                total_er = team["er"].fillna(0).sum() if "er" in team.columns else 0
                total_bb = team["bb_allowed"].fillna(0).sum() if "bb_allowed" in team.columns else 0
                total_ha = team["h_allowed"].fillna(0).sum() if "h_allowed" in team.columns else 0
                if total_ip > 0:
                    team_eras.append(total_er * 9 / total_ip)
                    team_whips.append((total_bb + total_ha) / total_ip)
            if len(team_eras) >= 2:
                denoms["ERA"] = max(0.05, float(np.std(team_eras)))
            if len(team_whips) >= 2:
                denoms["WHIP"] = max(0.005, float(np.std(team_whips)))

    # Fill any missing with defaults
    for cat in config.all_categories:
        if cat not in denoms:
            denoms[cat] = config.sgp_denominators.get(cat, 1.0)

    return denoms


# ── Tier Assignment ─────────────────────────────────────────────────


def assign_tiers(valued_pool: pd.DataFrame, score_col: str = "pick_score", n_tiers: int = 8) -> pd.DataFrame:
    """Assign tier labels to players based on score distribution gaps.

    Uses natural breaks in the score distribution to identify tiers.
    Players within the same tier are roughly interchangeable.
    """
    if valued_pool.empty:
        return valued_pool
    pool = valued_pool.sort_values(score_col, ascending=False).copy()
    scores = pool[score_col].values
    n = len(scores)

    if n <= n_tiers:
        pool["tier"] = range(1, n + 1)
        return pool

    # Compute score gaps between adjacent players
    gaps = np.diff(scores)  # negative values (descending)
    abs_gaps = np.abs(gaps)

    # Find the largest gaps to use as tier boundaries
    # Use a combination of quantile-based and gap-based tiers
    if n > n_tiers * 2:
        # Find top (n_tiers - 1) largest gaps for natural breaks
        gap_threshold = np.percentile(abs_gaps, 100 * (1 - (n_tiers - 1) / n))
        break_indices = np.where(abs_gaps >= gap_threshold)[0]
        # Take only the top n_tiers - 1 breaks
        if len(break_indices) > n_tiers - 1:
            top_gap_indices = np.argsort(abs_gaps[break_indices])[::-1][: n_tiers - 1]
            break_indices = sorted(break_indices[top_gap_indices])
        else:
            break_indices = sorted(break_indices)
    else:
        break_indices = []

    # Assign tiers
    tiers = np.ones(n, dtype=int)
    current_tier = 1
    for i in range(n):
        tiers[i] = current_tier
        if i in break_indices and current_tier < n_tiers:
            current_tier += 1

    pool["tier"] = tiers
    return pool
