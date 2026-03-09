"""Monte Carlo draft simulation engine with opponent modeling."""

import numpy as np
import pandas as pd
from typing import Optional
from src.valuation import SGPCalculator, LeagueConfig, compute_category_weights


class DraftSimulator:
    """Monte Carlo draft simulation for evaluating pick candidates."""

    def __init__(self, config: LeagueConfig, sigma: float = 10.0):
        """
        Args:
            config: League configuration.
            sigma: ADP noise parameter. Lower = opponents follow ADP more tightly.
                   For skilled leagues, use 8-12. For casual leagues, 15-25.
        """
        self.config = config
        self.sigma = sigma
        self.sgp_calc = SGPCalculator(config)

    def survival_probability(self, player_adp: float, current_pick: int,
                              next_user_pick: int,
                              positions_needed_league: dict = None,
                              player_positions: str = None) -> float:
        """Estimate the probability a player survives to the user's next pick.

        Uses a normal CDF approximation based on ADP distance, with optional
        adjustment for league-wide positional scarcity.

        Args:
            player_adp: Player's average draft position.
            current_pick: Current overall pick number.
            next_user_pick: User's next pick number.
            positions_needed_league: Dict {pos: num_teams_needing} for scarcity.
            player_positions: Comma-separated position string for the player.
        """
        if next_user_pick is None or next_user_pick <= current_pick:
            return 0.0

        picks_between = next_user_pick - current_pick
        if player_adp <= 0:
            player_adp = 999

        picks_until_adp = player_adp - current_pick

        if picks_until_adp <= 0:
            return max(0.0, 0.1 * (1 + picks_until_adp / 10))

        if picks_until_adp > picks_between * 3:
            return min(1.0, 0.95)

        from scipy.stats import norm
        z = (player_adp - next_user_pick) / (self.sigma * max(1, picks_between ** 0.3))
        base_prob = float(np.clip(norm.cdf(z), 0.01, 0.99))

        # Positional scarcity adjustment: if many teams need this position,
        # the player is more likely to be taken before our pick
        if positions_needed_league and player_positions:
            pos_list = [p.strip() for p in player_positions.split(",")]
            max_demand = 0
            for pos in pos_list:
                demand = positions_needed_league.get(pos, 0)
                max_demand = max(max_demand, demand)
            if max_demand > 0:
                # More teams needing this position = lower survival probability
                # Scale: if 6+ teams need the position, reduce survival by up to 20%
                scarcity_factor = min(0.2, max_demand / 50)
                base_prob *= (1.0 - scarcity_factor)

        return float(np.clip(base_prob, 0.01, 0.99))

    def compute_urgency(self, player: pd.Series, available_pool: pd.DataFrame,
                         current_pick: int, next_user_pick: int) -> float:
        """Compute pick urgency: how much value is lost by waiting.

        urgency = (1 - P_survive) * positional_dropoff
        """
        p_survive = self.survival_probability(
            player.get("adp", 999), current_pick, next_user_pick)

        # Positional drop-off: difference between this player and next-best at position
        positions = str(player.get("positions", "")).split(",")
        best_pos = positions[0].strip() if positions else "Util"

        eligible = available_pool[
            available_pool["positions"].str.contains(best_pos, na=False) &
            (available_pool["player_id"] != player.get("player_id"))
        ]

        if eligible.empty or "pick_score" not in eligible.columns:
            dropoff = player.get("pick_score", 0) * 0.3
        else:
            next_best = eligible["pick_score"].max()
            dropoff = max(0, player.get("pick_score", 0) - next_best)

        urgency = (1 - p_survive) * dropoff
        return urgency

    def opponent_pick_probability(self, available: pd.DataFrame, pick_num: int,
                                   team_positions: list = None,
                                   roster_slots: dict = None) -> np.ndarray:
        """Compute pick probabilities for an opponent, factoring in positional need.

        Args:
            available: DataFrame of available players (must have 'adp' column).
            pick_num: The overall pick number for this opponent pick.
            team_positions: List of position strings the opponent has already drafted.
            roster_slots: Dict of {position: count} from league config for slot limits.

        Returns:
            Normalized probability array over available players.
        """
        adp_vals = available["adp"].fillna(999).values
        # ADP-based weight
        weights = np.exp(-np.abs(adp_vals - pick_num) / self.sigma)

        # Positional need boost using actual roster data
        if team_positions is not None:
            # Count how many of each position the team has filled
            filled_counts = {}
            for p in team_positions:
                filled_counts[p] = filled_counts.get(p, 0) + 1

            # Determine slots per position
            slots = roster_slots or {
                "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
                "OF": 3, "SP": 2, "RP": 2,
            }

            for i, (_, player) in enumerate(available.iterrows()):
                pos_list = [p.strip() for p in str(player.get("positions", "")).split(",")]
                max_boost = 1.0
                for pos in pos_list:
                    filled = filled_counts.get(pos, 0)
                    limit = slots.get(pos, 0)
                    if limit > 0 and filled < limit:
                        # More urgently needed positions get bigger boost
                        remaining_need = (limit - filled) / limit
                        boost = 1.0 + remaining_need * 0.8  # up to 1.8x boost
                        max_boost = max(max_boost, boost)
                    elif limit > 0 and filled >= limit:
                        # Position already filled — slight penalty
                        max_boost = max(max_boost, 0.6)
                weights[i] *= max_boost

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.ones(len(available)) / len(available)
        return weights / weight_sum

    def greedy_rollout_value(self, candidate: pd.Series, available: pd.DataFrame,
                              draft_state, config) -> float:
        """Estimate total team value if we pick this candidate, then draft greedily.

        For each candidate being evaluated, simulate the user's remaining picks
        by greedily picking the highest-marginal-SGP player available at each
        future user pick. This gives a better estimate of total team value.

        Args:
            candidate: Series for the candidate player being considered now.
            available: DataFrame of currently available players.
            draft_state: Current DraftState instance.
            config: LeagueConfig instance.

        Returns:
            Estimated total SGP value from this candidate plus greedy future picks.
        """
        from src.valuation import SGPCalculator, value_all_players
        sgp_calc = SGPCalculator(config)

        # Start with candidate's value
        total_value = sgp_calc.total_sgp(candidate)

        # Simulate greedy future picks (just estimate 3-4 future rounds)
        remaining = available[available["player_id"] != candidate.get("player_id", -1)].copy()
        future_picks = min(4, draft_state.num_rounds - draft_state.current_round)

        for _ in range(future_picks):
            if remaining.empty:
                break
            remaining["_sgp"] = remaining.apply(sgp_calc.total_sgp, axis=1)
            best_idx = remaining["_sgp"].idxmax()
            total_value += remaining.loc[best_idx, "_sgp"]
            remaining = remaining.drop(best_idx)

        return total_value

    def simulate_draft(self, available_ids: np.ndarray, adp_values: np.ndarray,
                        sgp_values: np.ndarray, positions: list,
                        user_team_index: int, current_pick: int,
                        total_picks: int, num_teams: int,
                        user_roster_needs: set,
                        candidate_id: int,
                        n_simulations: int = 300,
                        team_positions: dict = None) -> dict:
        """Run Monte Carlo draft simulations for a candidate pick.

        Args:
            available_ids: Array of available player IDs.
            adp_values: Array of ADP values for available players (aligned).
            sgp_values: Array of total SGP values for available players (aligned).
            positions: List of position strings for available players.
            user_team_index: Index of user's team (0-based).
            current_pick: Current overall pick number.
            total_picks: Total picks in the draft.
            num_teams: Number of teams.
            user_roster_needs: Set of positions the user still needs.
            candidate_id: Player ID the user is considering drafting now.
            n_simulations: Number of simulations to run.
            team_positions: Dict {team_index: [drafted_positions]} for opponent modeling.

        Returns:
            dict with 'mean_sgp', 'std_sgp', 'p25_sgp' for expected roster outcome.
        """
        n_players = len(available_ids)
        if n_players == 0:
            return {"mean_sgp": 0, "std_sgp": 0, "p25_sgp": 0}

        results = np.zeros(n_simulations)
        rng = np.random.default_rng()

        # Pre-parse positions for each player into lists for fast lookup
        pos_lists = [str(p).split(",") for p in positions]

        for sim in range(n_simulations):
            is_available = np.ones(n_players, dtype=bool)
            # Track simulated opponent rosters (copy from actual state)
            sim_team_pos = {}
            if team_positions:
                for k, v in team_positions.items():
                    sim_team_pos[k] = list(v)
            else:
                for k in range(num_teams):
                    sim_team_pos[k] = []

            # Draft the candidate for the user
            candidate_idx = np.where(available_ids == candidate_id)[0]
            if len(candidate_idx) > 0:
                is_available[candidate_idx[0]] = False

            user_sgp_total = sgp_values[candidate_idx[0]] if len(candidate_idx) > 0 else 0

            # Limit simulation horizon to ~6 rounds ahead for performance
            sim_horizon = min(total_picks, current_pick + 1 + num_teams * 6)
            for pick in range(current_pick + 1, sim_horizon):
                n_avail = np.sum(is_available)
                if n_avail == 0:
                    break

                round_num = pick // num_teams
                pos_in_round = pick % num_teams
                if round_num % 2 == 0:
                    team_idx = pos_in_round
                else:
                    team_idx = num_teams - 1 - pos_in_round

                if team_idx == user_team_index:
                    avail_sgp = np.where(is_available, sgp_values, -999)
                    best_idx = np.argmax(avail_sgp)
                    is_available[best_idx] = False
                    user_sgp_total += sgp_values[best_idx]
                else:
                    avail_indices = np.where(is_available)[0]
                    if len(avail_indices) == 0:
                        break

                    adp_diffs = np.abs(adp_values[avail_indices] - pick)
                    weights = np.exp(-adp_diffs / self.sigma)

                    # Apply positional need boost for this opponent
                    opp_positions = sim_team_pos.get(team_idx, [])
                    if opp_positions:
                        filled = {}
                        for p in opp_positions:
                            filled[p] = filled.get(p, 0) + 1
                        for j, ai in enumerate(avail_indices):
                            for pos in pos_lists[ai]:
                                pos = pos.strip()
                                if pos and filled.get(pos, 0) == 0:
                                    weights[j] *= 1.4
                                    break

                    weight_sum = weights.sum()
                    if weight_sum <= 0:
                        weights = np.ones(len(avail_indices))
                        weight_sum = len(avail_indices)
                    probs = weights / weight_sum

                    chosen = rng.choice(avail_indices, p=probs)
                    is_available[chosen] = False

                    # Track the simulated pick for this opponent
                    for pos in pos_lists[chosen]:
                        pos = pos.strip()
                        if pos:
                            sim_team_pos.setdefault(team_idx, []).append(pos)
                            break

            results[sim] = user_sgp_total

        return {
            "mean_sgp": float(np.mean(results)),
            "std_sgp": float(np.std(results)),
            "p25_sgp": float(np.percentile(results, 25)),
        }

    def evaluate_candidates(self, player_pool: pd.DataFrame,
                             draft_state, top_n: int = 8,
                             n_simulations: int = 300) -> pd.DataFrame:
        """Evaluate the top N candidate picks using Monte Carlo simulation.

        Args:
            player_pool: Full player pool DataFrame (with pick_score column).
            draft_state: DraftState instance.
            top_n: Number of top candidates to simulate.
            n_simulations: Simulations per candidate.

        Returns:
            DataFrame with simulation results for each candidate.
        """
        available = draft_state.available_players(player_pool)
        if available.empty:
            return pd.DataFrame()

        # Get top candidates by greedy pick_score
        candidates = available.nlargest(top_n, "pick_score")

        # Prepare arrays for simulation
        available_ids = available["player_id"].values
        adp_values = available["adp"].values.astype(float)
        sgp_values = available["total_sgp"].values.astype(float) if "total_sgp" in available.columns else available["pick_score"].values.astype(float)
        positions = available["positions"].tolist()

        user_needs = set(draft_state.user_team.open_positions())
        current_pick = draft_state.current_pick
        next_pick = draft_state.next_user_pick()

        # Get actual opponent rosters for enhanced simulation
        team_positions = draft_state.get_all_teams_positions()

        results = []
        for _, candidate in candidates.iterrows():
            # Compute urgency
            urgency = self.compute_urgency(
                candidate, available, current_pick,
                next_pick if next_pick and next_pick > current_pick else current_pick + draft_state.num_teams
            )

            # Run simulation with opponent roster tracking
            sim_result = self.simulate_draft(
                available_ids=available_ids,
                adp_values=adp_values,
                sgp_values=sgp_values,
                positions=positions,
                user_team_index=draft_state.user_team_index,
                current_pick=current_pick,
                total_picks=draft_state.total_picks,
                num_teams=draft_state.num_teams,
                user_roster_needs=user_needs,
                candidate_id=candidate["player_id"],
                n_simulations=n_simulations,
                team_positions=team_positions,
            )

            p_survive = self.survival_probability(
                candidate.get("adp", 999), current_pick,
                next_pick if next_pick else current_pick + draft_state.num_teams
            )

            results.append({
                "player_id": candidate["player_id"],
                "name": candidate["name"],
                "team": candidate.get("team", ""),
                "positions": candidate["positions"],
                "adp": candidate.get("adp", 999),
                "pick_score": candidate.get("pick_score", 0),
                "urgency": urgency,
                "p_survive": p_survive,
                "mc_mean_sgp": sim_result["mean_sgp"],
                "mc_std_sgp": sim_result["std_sgp"],
                "mc_p25_sgp": sim_result["p25_sgp"],
                "combined_score": sim_result["mean_sgp"] + urgency * 0.4,
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("combined_score", ascending=False)
        return results_df


def detect_position_run(pick_log: list, lookback: int = 8, threshold: int = 3) -> list:
    """Detect position runs in recent picks.

    Returns list of positions where a run is detected.
    """
    if len(pick_log) < threshold:
        return []

    recent = pick_log[-lookback:]
    position_counts = {}
    for entry in recent:
        for pos in entry.get("positions", "").split(","):
            pos = pos.strip()
            if pos:
                position_counts[pos] = position_counts.get(pos, 0) + 1

    return [pos for pos, count in position_counts.items() if count >= threshold]
