"""Tests for Phase 4 features: N1 category fit, O3 call-up signals,
M6 ratio lock alert, P4 per-category replacement levels."""

import pandas as pd

# ── N1: Category Fit Indicator ──────────────────────────────────────


class TestCategoryFit:
    """Tests for compute_category_fit (src/in_season.py)."""

    def test_player_helps_weak_categories(self):
        """Player contributing to 3 weak categories gets fit_score > 0."""
        from src.in_season import compute_category_fit

        profile = {
            "R": "weak",
            "HR": "weak",
            "RBI": "weak",
            "SB": "strong",
            "AVG": "strong",
            "OBP": "punt",
        }
        player_sgp = {"R": 0.5, "HR": 0.8, "RBI": 0.6, "SB": 0.1, "AVG": 0.2, "OBP": 0.0}
        result = compute_category_fit(player_sgp, profile)

        assert len(result["helps"]) == 3
        assert "R" in result["helps"]
        assert "HR" in result["helps"]
        assert "RBI" in result["helps"]
        assert result["fit_score"] > 0

    def test_player_wastes_on_strong_categories(self):
        """Player contributing only to strong/punt categories gets low fit."""
        from src.in_season import compute_category_fit

        profile = {
            "R": "strong",
            "HR": "strong",
            "RBI": "punt",
            "SB": "weak",
            "AVG": "weak",
            "OBP": "weak",
        }
        # Player has big SGP in strong/punt cats, nothing in weak cats
        player_sgp = {"R": 1.2, "HR": 0.9, "RBI": 0.7, "SB": 0.0, "AVG": 0.05, "OBP": 0.0}
        result = compute_category_fit(player_sgp, profile)

        assert result["fit_score"] == 0
        assert len(result["helps"]) == 0
        assert len(result["wastes"]) >= 2  # R and HR are strong with big SGP

    def test_empty_profile(self):
        """Empty profile returns zero fit."""
        from src.in_season import compute_category_fit

        result = compute_category_fit({}, {})
        assert result["fit_score"] == 0
        assert result["helps"] == []
        assert result["wastes"] == []

    def test_mixed_fit(self):
        """Player helping some weak cats and wasting on some strong cats."""
        from src.in_season import compute_category_fit

        profile = {"R": "weak", "HR": "strong", "SB": "weak", "AVG": "punt"}
        player_sgp = {"R": 0.5, "HR": 0.8, "SB": 0.3, "AVG": 0.6}
        result = compute_category_fit(player_sgp, profile)

        assert "R" in result["helps"]
        assert "SB" in result["helps"]
        assert "HR" in result["wastes"]
        assert "AVG" in result["wastes"]
        assert result["fit_score"] > 0


# ── O3: Call-Up Signals ─────────────────────────────────────────────


class TestCallUpSignals:
    """Tests for compute_call_up_signals (src/prospect_engine.py)."""

    def test_40_man_with_2026_eta_high_score(self):
        """Prospect on 40-man with 2026 ETA gets high score."""
        from src.prospect_engine import compute_call_up_signals

        prospect = {"on_40_man": True, "fg_eta": "2026", "position": "SS", "team": "BOS"}
        result = compute_call_up_signals(prospect)

        assert result["on_40_man"] is True
        assert result["call_up_score"] >= 70
        assert result["signal"] == "IMMINENT"

    def test_not_on_40_man_lower_score(self):
        """Prospect not on 40-man gets lower score."""
        from src.prospect_engine import compute_call_up_signals

        prospect = {"on_40_man": False, "fg_eta": "2027", "position": "OF", "team": "NYY"}
        result = compute_call_up_signals(prospect)

        assert result["on_40_man"] is False
        assert result["call_up_score"] <= 40
        assert result["signal"] != "IMMINENT"

    def test_position_opening_boost(self):
        """IL opening at same team/position adds 20 points."""
        from src.prospect_engine import compute_call_up_signals

        prospect = {"on_40_man": True, "fg_eta": "2026", "position": "SS", "team": "BOS"}
        il_players = [{"team": "BOS", "positions": "SS,2B"}]
        result = compute_call_up_signals(prospect, mlb_team_il_players=il_players)

        assert result["call_up_score"] >= 70 + 20  # 40 + 30 + 20 = 90
        assert result["signal"] == "IMMINENT"

    def test_no_il_data(self):
        """Without IL data, still computes based on 40-man and ETA."""
        from src.prospect_engine import compute_call_up_signals

        prospect = {"on_40_man": True, "fg_eta": "2028", "position": "SP", "team": "LAD"}
        result = compute_call_up_signals(prospect)

        assert result["call_up_score"] == 40  # Only 40-man points
        assert result["signal"] == "WATCH"

    def test_score_capped_at_100(self):
        """Score never exceeds 100."""
        from src.prospect_engine import compute_call_up_signals

        prospect = {"on_40_man": True, "fg_eta": "2026", "position": "SS", "team": "BOS"}
        il_players = [{"team": "BOS", "positions": "SS"}]
        result = compute_call_up_signals(prospect, mlb_team_il_players=il_players)

        assert result["call_up_score"] <= 100


# ── M6: Ratio Lock Alert ────────────────────────────────────────────


class TestRatioLockAlert:
    """Tests for generate_ratio_lock_alert (src/alerts.py)."""

    def test_winning_era_whip_generates_alert(self):
        """Winning ERA by 0.80 with 40 IP banked generates alert."""
        from src.alerts import generate_ratio_lock_alert

        result = generate_ratio_lock_alert(
            current_era=2.80,
            opp_era=3.60,
            current_whip=1.05,
            opp_whip=1.20,
            banked_ip=40,
            remaining_starts=3,
        )

        assert result is not None
        assert result["type"] == "ratio_lock"
        assert result["era_lead"] == 0.80
        assert result["whip_lead"] == 0.15
        assert "Ratio Lock" in result["title"]

    def test_losing_era_no_alert(self):
        """Losing ERA returns None."""
        from src.alerts import generate_ratio_lock_alert

        result = generate_ratio_lock_alert(
            current_era=4.50,
            opp_era=3.20,
            current_whip=1.30,
            opp_whip=1.10,
            banked_ip=50,
            remaining_starts=2,
        )

        assert result is None

    def test_small_lead_no_alert(self):
        """ERA lead < 0.50 does not trigger alert."""
        from src.alerts import generate_ratio_lock_alert

        result = generate_ratio_lock_alert(
            current_era=3.40,
            opp_era=3.80,
            current_whip=1.15,
            opp_whip=1.30,
            banked_ip=40,
        )

        assert result is None  # ERA lead is only 0.40

    def test_insufficient_ip_no_alert(self):
        """Less than 30 IP banked does not trigger alert."""
        from src.alerts import generate_ratio_lock_alert

        result = generate_ratio_lock_alert(
            current_era=2.50,
            opp_era=4.00,
            current_whip=1.00,
            opp_whip=1.30,
            banked_ip=20,
        )

        assert result is None


# ── P4: Per-Category Replacement Level ───────────────────────────────


class TestPerCategoryReplacement:
    """Tests for compute_per_category_replacement (src/valuation.py)."""

    def _make_pool(self) -> pd.DataFrame:
        """Create a synthetic player pool for testing."""
        rows = []
        # 15 catchers with varied stats
        for i in range(15):
            rows.append(
                {
                    "player_id": 1000 + i,
                    "name": f"Catcher_{i}",
                    "positions": "C",
                    "is_hitter": True,
                    "r": 40 + i * 2,
                    "hr": 10 + i,
                    "rbi": 35 + i * 2,
                    "sb": 2 + i,
                    "avg": 0.230 + i * 0.005,
                    "obp": 0.300 + i * 0.005,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                }
            )
        # 70 outfielders with higher stats
        for i in range(70):
            rows.append(
                {
                    "player_id": 2000 + i,
                    "name": f"OF_{i}",
                    "positions": "OF",
                    "is_hitter": True,
                    "r": 60 + i,
                    "hr": 15 + i,
                    "rbi": 50 + i,
                    "sb": 5 + i,
                    "avg": 0.250 + i * 0.002,
                    "obp": 0.320 + i * 0.002,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_replacement_levels_per_position(self):
        """Returns replacement levels keyed by position and category."""
        from src.valuation import compute_per_category_replacement

        pool = self._make_pool()
        result = compute_per_category_replacement(pool)

        assert "C" in result
        assert "OF" in result
        assert "R" in result["C"]
        assert "HR" in result["C"]

    def test_c_replacement_differs_from_of(self):
        """Catcher replacement level differs from OF replacement level."""
        from src.valuation import compute_per_category_replacement

        pool = self._make_pool()
        result = compute_per_category_replacement(pool)

        # Catchers are weaker pool, so replacement AVG should be lower
        assert result["C"]["AVG"] != result["OF"]["AVG"]
        assert result["C"]["AVG"] < result["OF"]["AVG"]

    def test_custom_position_counts(self):
        """Custom position counts change replacement cutoff."""
        from src.valuation import compute_per_category_replacement

        pool = self._make_pool()
        result_small = compute_per_category_replacement(pool, position_counts={"C": 5, "OF": 30})
        result_large = compute_per_category_replacement(pool, position_counts={"C": 12, "OF": 62})

        # Smaller count = higher replacement level (top 5 vs top 12)
        assert result_small["C"]["HR"] >= result_large["C"]["HR"]

    def test_empty_pool(self):
        """Empty pool returns empty dict."""
        from src.valuation import compute_per_category_replacement

        result = compute_per_category_replacement(pd.DataFrame())
        assert result == {}
