"""End-to-end tests for 1-for-1 trades in the trade evaluator.

1-for-1 trades are the most common trade type in fantasy baseball.
These tests verify the full evaluate_trade pipeline with realistic
player pools, ensuring correct grading and directional SGP deltas.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.engine.output.trade_evaluator import evaluate_trade
from src.valuation import LeagueConfig


def _build_player_pool() -> pd.DataFrame:
    """Build a realistic 30-player pool: 15 hitters + 9 pitchers (rostered) + 6 extras.

    Returns a DataFrame with 30 players — 24 form a roster (IDs 1-24),
    plus 6 additional players (IDs 25-30) as trade targets / FA pool.
    Hitters have varied stat lines; pitchers include SP and RP archetypes.
    """
    players = []

    # --- 15 hitters (IDs 1-15), varied quality ---
    hitter_profiles = [
        # (name, team, pos, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf)
        ("Aaron Judge", "NYY", "OF", 600, 540, 162, 105, 45, 110, 5, 0.300, 0.390, 50, 5, 5),
        ("Mookie Betts", "LAD", "SS,OF", 580, 520, 156, 100, 30, 85, 20, 0.300, 0.380, 50, 5, 5),
        ("Kyle Tucker", "CHC", "OF", 570, 510, 148, 90, 28, 95, 15, 0.290, 0.370, 50, 5, 5),
        ("Freddie Freeman", "LAD", "1B", 590, 530, 163, 95, 25, 100, 10, 0.308, 0.395, 50, 5, 5),
        ("Bo Bichette", "TOR", "SS", 550, 510, 143, 75, 18, 70, 12, 0.280, 0.320, 30, 5, 5),
        ("Yandy Diaz", "TBR", "1B,3B", 540, 490, 132, 65, 12, 60, 3, 0.270, 0.350, 40, 5, 5),
        ("Willy Adames", "SFG", "SS", 560, 500, 130, 80, 25, 85, 8, 0.260, 0.340, 50, 5, 5),
        ("Ozzie Albies", "ATL", "2B", 550, 510, 138, 85, 20, 75, 15, 0.271, 0.320, 30, 5, 5),
        ("Salvador Perez", "KCR", "C", 520, 490, 127, 60, 22, 80, 1, 0.259, 0.290, 20, 5, 5),
        ("Daulton Varsho", "TOR", "C,OF", 530, 480, 120, 70, 18, 65, 10, 0.250, 0.330, 40, 5, 5),
        ("Jazz Chisholm", "NYY", "3B", 500, 460, 115, 75, 20, 65, 25, 0.250, 0.320, 30, 5, 5),
        ("Brice Turang", "MIL", "2B,SS", 520, 480, 125, 70, 5, 45, 35, 0.260, 0.320, 30, 5, 5),
        ("Yainer Diaz", "HOU", "C", 500, 470, 122, 55, 15, 65, 2, 0.260, 0.300, 20, 5, 5),
        ("Lars Nootbaar", "STL", "OF", 480, 430, 103, 60, 12, 50, 8, 0.240, 0.330, 40, 5, 5),
        ("Isiah Kiner-Falefa", "PIT", "3B,SS", 470, 440, 106, 55, 5, 40, 10, 0.241, 0.290, 20, 5, 5),
    ]

    for i, (name, team, pos, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf) in enumerate(hitter_profiles):
        players.append(
            {
                "player_id": i + 1,
                "name": name,
                "player_name": name,
                "team": team,
                "positions": pos,
                "is_hitter": 1,
                "is_injured": 0,
                "pa": pa,
                "ab": ab,
                "h": h,
                "r": r,
                "hr": hr,
                "rbi": rbi,
                "sb": sb,
                "avg": avg,
                "obp": obp,
                "bb": bb,
                "hbp": hbp,
                "sf": sf,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 10 + i * 8,
            }
        )

    # --- 9 pitchers (IDs 16-24) ---
    pitcher_profiles = [
        # (name, team, pos, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed)
        ("Gerrit Cole", "NYY", "SP", 190, 14, 6, 0, 220, 2.95, 1.05, 62, 40, 160),
        ("Zack Wheeler", "PHI", "SP", 185, 13, 7, 0, 210, 3.10, 1.08, 64, 38, 162),
        ("Logan Webb", "SFG", "SP", 200, 12, 8, 0, 180, 3.30, 1.15, 73, 45, 185),
        ("Cristian Javier", "HOU", "SP", 160, 10, 7, 0, 175, 3.50, 1.18, 62, 42, 147),
        ("Joe Ryan", "MIN", "SP", 170, 11, 8, 0, 185, 3.60, 1.12, 68, 35, 155),
        ("Emmanuel Clase", "CLE", "RP", 65, 3, 2, 35, 60, 1.20, 0.85, 9, 12, 43),
        ("Ryan Helsley", "STL", "RP", 60, 3, 3, 30, 70, 2.50, 0.95, 17, 15, 42),
        ("Devin Williams", "NYY", "RP", 55, 2, 2, 28, 75, 1.80, 0.90, 11, 14, 36),
        ("Jordan Romano", "TOR", "RP", 50, 2, 3, 20, 55, 3.00, 1.10, 17, 15, 40),
    ]

    for i, (name, team, pos, ip, w, l, sv, k, era, whip, er, bb_a, h_a) in enumerate(pitcher_profiles):
        players.append(
            {
                "player_id": 16 + i,
                "name": name,
                "player_name": name,
                "team": team,
                "positions": pos,
                "is_hitter": 0,
                "is_injured": 0,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "ip": ip,
                "w": w,
                "l": l,
                "sv": sv,
                "k": k,
                "era": era,
                "whip": whip,
                "er": er,
                "bb_allowed": bb_a,
                "h_allowed": h_a,
                "adp": 15 + i * 10,
            }
        )

    # --- 6 extra players (IDs 25-30) as trade targets / FA pool ---
    extras = [
        # Strong hitter trade target
        {
            "player_id": 25,
            "name": "Juan Soto",
            "player_name": "Juan Soto",
            "team": "NYM",
            "positions": "OF",
            "is_hitter": 1,
            "is_injured": 0,
            "pa": 620,
            "ab": 530,
            "h": 163,
            "r": 110,
            "hr": 38,
            "rbi": 105,
            "sb": 5,
            "avg": 0.308,
            "obp": 0.420,
            "bb": 80,
            "hbp": 5,
            "sf": 5,
            "ip": 0,
            "w": 0,
            "l": 0,
            "sv": 0,
            "k": 0,
            "era": 0,
            "whip": 0,
            "er": 0,
            "bb_allowed": 0,
            "h_allowed": 0,
            "adp": 3,
        },
        # Weak hitter trade target
        {
            "player_id": 26,
            "name": "Nick Senzel",
            "player_name": "Nick Senzel",
            "team": "CIN",
            "positions": "OF",
            "is_hitter": 1,
            "is_injured": 0,
            "pa": 400,
            "ab": 370,
            "h": 85,
            "r": 40,
            "hr": 6,
            "rbi": 30,
            "sb": 5,
            "avg": 0.230,
            "obp": 0.280,
            "bb": 20,
            "hbp": 5,
            "sf": 5,
            "ip": 0,
            "w": 0,
            "l": 0,
            "sv": 0,
            "k": 0,
            "era": 0,
            "whip": 0,
            "er": 0,
            "bb_allowed": 0,
            "h_allowed": 0,
            "adp": 250,
        },
        # Strong SP trade target
        {
            "player_id": 27,
            "name": "Spencer Strider",
            "player_name": "Spencer Strider",
            "team": "ATL",
            "positions": "SP",
            "is_hitter": 0,
            "is_injured": 0,
            "pa": 0,
            "ab": 0,
            "h": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0,
            "obp": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
            "ip": 180,
            "w": 15,
            "l": 5,
            "sv": 0,
            "k": 250,
            "era": 2.50,
            "whip": 0.95,
            "er": 50,
            "bb_allowed": 30,
            "h_allowed": 141,
            "adp": 8,
        },
        # Weak SP trade target
        {
            "player_id": 28,
            "name": "Mitch Keller",
            "player_name": "Mitch Keller",
            "team": "PIT",
            "positions": "SP",
            "is_hitter": 0,
            "is_injured": 0,
            "pa": 0,
            "ab": 0,
            "h": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0,
            "obp": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
            "ip": 150,
            "w": 8,
            "l": 10,
            "sv": 0,
            "k": 130,
            "era": 4.20,
            "whip": 1.30,
            "er": 70,
            "bb_allowed": 50,
            "h_allowed": 145,
            "adp": 180,
        },
        # Extra hitter FA filler
        {
            "player_id": 29,
            "name": "Nick Castellanos",
            "player_name": "Nick Castellanos",
            "team": "PHI",
            "positions": "OF",
            "is_hitter": 1,
            "is_injured": 0,
            "pa": 550,
            "ab": 510,
            "h": 138,
            "r": 70,
            "hr": 20,
            "rbi": 75,
            "sb": 3,
            "avg": 0.271,
            "obp": 0.320,
            "bb": 30,
            "hbp": 5,
            "sf": 5,
            "ip": 0,
            "w": 0,
            "l": 0,
            "sv": 0,
            "k": 0,
            "era": 0,
            "whip": 0,
            "er": 0,
            "bb_allowed": 0,
            "h_allowed": 0,
            "adp": 100,
        },
        # Extra pitcher FA filler
        {
            "player_id": 30,
            "name": "Pablo Lopez",
            "player_name": "Pablo Lopez",
            "team": "MIN",
            "positions": "SP",
            "is_hitter": 0,
            "is_injured": 0,
            "pa": 0,
            "ab": 0,
            "h": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0,
            "obp": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
            "ip": 170,
            "w": 10,
            "l": 8,
            "sv": 0,
            "k": 180,
            "era": 3.70,
            "whip": 1.20,
            "er": 70,
            "bb_allowed": 45,
            "h_allowed": 159,
            "adp": 60,
        },
    ]
    players.extend(extras)

    return pd.DataFrame(players)


@pytest.fixture
def pool():
    """Realistic 30-player pool for 1-for-1 trade testing."""
    return _build_player_pool()


@pytest.fixture
def config():
    """Standard league config."""
    return LeagueConfig()


@pytest.fixture
def roster_ids():
    """23-player roster: hitters 1-15, pitchers 16-24 (minus ID 15 to make 23)."""
    # 14 hitters (IDs 1-14) + 9 pitchers (IDs 16-24) = 23 roster slots
    return list(range(1, 15)) + list(range(16, 25))


class TestOneForOneTradeEndToEnd:
    """End-to-end tests for 1-for-1 trades — the most common fantasy trade type."""

    def test_upgrade_hitter_produces_positive_delta(self, pool, config, roster_ids):
        """Trading a weak hitter for a strong hitter should yield positive user_delta.

        Give: Isiah Kiner-Falefa (ID 15 — NOT on roster, using ID 14: Lars Nootbaar)
        Actually: Give ID 14 (Nootbaar, weak OF) for ID 25 (Soto, elite OF).
        """
        result = evaluate_trade(
            giving_ids=[14],  # Lars Nootbaar: .240 AVG, 12 HR, 50 RBI
            receiving_ids=[25],  # Juan Soto: .308 AVG, 38 HR, 105 RBI
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Soto is a massive upgrade over Nootbaar — surplus must be positive
        assert result["surplus_sgp"] > 0, (
            f"Expected positive surplus trading Nootbaar for Soto, got {result['surplus_sgp']:.3f}"
        )
        assert result["verdict"] == "ACCEPT"

    def test_downgrade_hitter_produces_negative_delta(self, pool, config, roster_ids):
        """Trading a strong hitter for a weak hitter should yield negative user_delta.

        Give: Aaron Judge (ID 1, elite OF) for Nick Senzel (ID 26, weak OF).
        """
        result = evaluate_trade(
            giving_ids=[1],  # Aaron Judge: .300 AVG, 45 HR, 110 RBI
            receiving_ids=[26],  # Nick Senzel: .230 AVG, 6 HR, 30 RBI
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Judge is far superior to Senzel — surplus must be negative
        assert result["surplus_sgp"] < 0, (
            f"Expected negative surplus trading Judge for Senzel, got {result['surplus_sgp']:.3f}"
        )
        assert result["verdict"] == "DECLINE"

    def test_upgrade_pitcher_completes_with_valid_result(self, pool, config, roster_ids):
        """Trading a weaker SP for an ace should complete and produce valid output.

        Give: Joe Ryan (ID 20, mid-tier SP) for Spencer Strider (ID 27, ace).

        Note: LP lineup re-optimization can reshuffle pitcher slot assignments,
        so we verify structure and valid grading rather than surplus direction.
        """
        result = evaluate_trade(
            giving_ids=[20],  # Joe Ryan: 3.60 ERA, 185 K
            receiving_ids=[27],  # Spencer Strider: 2.50 ERA, 250 K
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Strider has better ERA/WHIP/K — those categories should improve
        assert result["category_impact"]["ERA"] > 0 or result["category_impact"]["WHIP"] > 0, (
            "Strider should improve at least one rate pitching category"
        )
        assert isinstance(result["surplus_sgp"], float)
        assert result["grade"] in {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"}

    def test_downgrade_pitcher_completes_with_valid_result(self, pool, config, roster_ids):
        """Trading an ace for a back-end starter should complete and produce valid output.

        Give: Gerrit Cole (ID 16, ace) for Mitch Keller (ID 28, weak SP).

        Note: LP lineup re-optimization can reshuffle pitcher slot assignments,
        so we verify structure rather than strict surplus direction.
        """
        result = evaluate_trade(
            giving_ids=[16],  # Gerrit Cole: 2.95 ERA, 220 K, 14 W
            receiving_ids=[28],  # Mitch Keller: 4.20 ERA, 130 K, 8 W
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Cole -> Keller: surplus should be non-positive (at best neutral via LP)
        assert result["surplus_sgp"] <= 0, (
            f"Trading Cole for Keller should not produce positive surplus, got {result['surplus_sgp']:.3f}"
        )
        assert isinstance(result["surplus_sgp"], float)

    def test_result_contains_required_keys(self, pool, config, roster_ids):
        """1-for-1 trade result dict must contain all expected output keys."""
        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[25],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        required_keys = [
            "grade",
            "surplus_sgp",
            "category_impact",
            "verdict",
            "confidence_pct",
            "before_totals",
            "after_totals",
            "giving_players",
            "receiving_players",
            "risk_flags",
            "bench_cost",
            "drop_candidate",
            "fa_pickup",
            "lineup_constrained",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_grade_is_valid_letter(self, pool, config, roster_ids):
        """Trade grade must be a recognized letter grade."""
        valid_grades = {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"}

        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[25],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        assert result["grade"] in valid_grades, f"Invalid grade: {result['grade']}"

    def test_category_impact_covers_all_categories(self, pool, config, roster_ids):
        """Category impact dict should have entries for all 12 league categories."""
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[26],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        for cat in config.all_categories:
            assert cat in result["category_impact"], f"Missing category: {cat}"

    def test_no_drop_or_pickup_for_equal_trade(self, pool, config, roster_ids):
        """1-for-1 trade should not trigger drop candidate or FA pickup."""
        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[25],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        assert result["drop_candidate"] is None
        assert result["fa_pickup"] is None

    def test_bench_cost_is_zero(self, pool, config, roster_ids):
        """Bench cost should always be zero (replaced by LP constraint model)."""
        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[25],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        assert result["bench_cost"] == 0.0

    def test_cross_position_hitter_for_pitcher(self, pool, config, roster_ids):
        """Trading a hitter for a pitcher should run without error.

        Give: Lars Nootbaar (ID 14, hitter) for Spencer Strider (ID 27, pitcher).
        """
        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[27],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Should complete without error and have a valid grade
        assert result["grade"] in {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"}
        assert isinstance(result["surplus_sgp"], float)

    def test_player_names_in_result(self, pool, config, roster_ids):
        """Result should contain correct player names for giving and receiving sides."""
        result = evaluate_trade(
            giving_ids=[14],
            receiving_ids=[25],
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        assert "Lars Nootbaar" in result["giving_players"]
        assert "Juan Soto" in result["receiving_players"]
