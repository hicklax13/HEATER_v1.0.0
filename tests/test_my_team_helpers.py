"""Unit tests for src/my_team_helpers.py — verifies the helpers work standalone."""

import pandas as pd
import pytest

from src.my_team_helpers import (
    compute_category_totals,
    news_type_label,
    ownership_arrow,
    rank_priority_losing_cats,
    sentiment_indicator,
    source_badge,
)

# ── sentiment_indicator ───────────────────────────────────────────────────────


class TestSentimentIndicator:
    def test_positive_score_returns_positive_label(self):
        html = sentiment_indicator(0.5)
        assert "Positive" in html
        assert "<span" in html

    def test_zero_score_returns_neutral(self):
        html = sentiment_indicator(0.0)
        assert "Neutral" in html

    def test_negative_score_returns_negative(self):
        html = sentiment_indicator(-0.5)
        assert "Negative" in html

    def test_boundary_exactly_02_is_positive(self):
        # threshold is >= 0.2
        assert "Positive" in sentiment_indicator(0.2)

    def test_boundary_just_below_02_is_neutral(self):
        assert "Neutral" in sentiment_indicator(0.19)

    def test_boundary_exactly_neg_02_is_neutral(self):
        assert "Neutral" in sentiment_indicator(-0.2)

    def test_boundary_just_below_neg_02_is_negative(self):
        assert "Negative" in sentiment_indicator(-0.21)


# ── ownership_arrow ───────────────────────────────────────────────────────────


class TestOwnershipArrow:
    def test_up_direction_contains_up_arrow(self):
        html = ownership_arrow("up", 5.0)
        assert "&#9650;" in html
        assert "+5.0%" in html

    def test_down_direction_contains_down_arrow(self):
        html = ownership_arrow("down", -3.0)
        assert "&#9660;" in html
        assert "-3.0%" in html

    def test_flat_direction_contains_dash(self):
        html = ownership_arrow("flat", 0.0)
        assert "&#8212;" in html
        assert "0.0%" in html

    def test_zero_delta_shows_zero(self):
        html = ownership_arrow("up", 0.0)
        assert "0.0%" in html

    def test_returns_html_string(self):
        result = ownership_arrow("up", 1.5)
        assert isinstance(result, str)
        assert "<span" in result


# ── source_badge ──────────────────────────────────────────────────────────────


class TestSourceBadge:
    def test_known_source_espn(self):
        html = source_badge("espn")
        assert "ESPN" in html
        assert "#c41230" in html  # ESPN brand color (allowlisted)

    def test_known_source_yahoo(self):
        html = source_badge("yahoo")
        assert "Yahoo" in html

    def test_unknown_source_uppercased(self):
        html = source_badge("rotogrinders")
        assert "ROTOGRINDERS" in html

    def test_returns_span(self):
        assert "<span" in source_badge("mlb")


# ── news_type_label ───────────────────────────────────────────────────────────


class TestNewsTypeLabel:
    def test_injury_label(self):
        assert "Injury" in news_type_label("injury")

    def test_transaction_label(self):
        assert "Transaction" in news_type_label("transaction")

    def test_callup_label(self):
        assert "Call-Up" in news_type_label("callup")

    def test_unknown_defaults_to_general(self):
        assert "General" in news_type_label("unknown_type")

    def test_returns_span(self):
        assert "<span" in news_type_label("lineup")


# ── compute_category_totals ───────────────────────────────────────────────────


class TestComputeCategoryTotals:
    def _make_hitter_row(self, r=10, hr=2, rbi=8, sb=1, ab=40, h=12, bb=4, hbp=0, sf=0):
        return {
            "is_hitter": 1,
            "r": r,
            "hr": hr,
            "rbi": rbi,
            "sb": sb,
            "ab": ab,
            "h": h,
            "bb": bb,
            "hbp": hbp,
            "sf": sf,
        }

    def _make_pitcher_row(self, w=1, l=0, sv=0, k=8, ip=6.0, er=2, bb_allowed=2, h_allowed=5):
        return {
            "is_hitter": 0,
            "w": w,
            "l": l,
            "sv": sv,
            "k": k,
            "ip": ip,
            "er": er,
            "bb_allowed": bb_allowed,
            "h_allowed": h_allowed,
        }

    def test_no_rows_returns_empty_dicts(self):
        # The function requires an is_hitter column; an empty-rows df with columns works.
        df = pd.DataFrame(
            columns=[
                "is_hitter",
                "r",
                "hr",
                "rbi",
                "sb",
                "ab",
                "h",
                "bb",
                "hbp",
                "sf",
                "w",
                "l",
                "sv",
                "k",
                "ip",
                "er",
                "bb_allowed",
                "h_allowed",
            ]
        )
        hit, pitch = compute_category_totals(df)
        assert hit == {}
        assert pitch == {}

    def test_hitter_counting_stats(self):
        df = pd.DataFrame([self._make_hitter_row(r=10, hr=3, rbi=9, sb=2)])
        hit, pitch = compute_category_totals(df)
        assert hit["R"] == 10
        assert hit["HR"] == 3
        assert hit["RBI"] == 9
        assert hit["SB"] == 2

    def test_hitter_avg_calculation(self):
        # 12 hits / 40 AB = .300
        df = pd.DataFrame([self._make_hitter_row(h=12, ab=40)])
        hit, _ = compute_category_totals(df)
        assert hit["AVG"] == ".300"

    def test_hitter_obp_calculation(self):
        # (12 + 4) / (40 + 4) = 16/44 ≈ .364
        df = pd.DataFrame([self._make_hitter_row(h=12, ab=40, bb=4, hbp=0, sf=0)])
        hit, _ = compute_category_totals(df)
        assert "OBP" in hit

    def test_hitter_zero_ab_avg_fallback(self):
        df = pd.DataFrame([self._make_hitter_row(ab=0, h=0)])
        hit, _ = compute_category_totals(df)
        assert hit["AVG"] == ".000"

    def test_pitcher_counting_stats(self):
        df = pd.DataFrame([self._make_pitcher_row(w=2, l=1, sv=3, k=10)])
        _, pitch = compute_category_totals(df)
        assert pitch["W"] == 2
        assert pitch["L"] == 1
        assert pitch["SV"] == 3
        assert pitch["K"] == 10

    def test_pitcher_era_calculation(self):
        # 2 ER in 9 IP → ERA 2.00
        df = pd.DataFrame([self._make_pitcher_row(er=2, ip=9.0)])
        _, pitch = compute_category_totals(df)
        assert pitch["ERA"] == "2.00"

    def test_pitcher_whip_calculation(self):
        # (2 BB + 5 H) / 6 IP = 1.17
        df = pd.DataFrame([self._make_pitcher_row(bb_allowed=2, h_allowed=5, ip=6.0)])
        _, pitch = compute_category_totals(df)
        assert "WHIP" in pitch

    def test_pitcher_zero_ip_fallback(self):
        df = pd.DataFrame([self._make_pitcher_row(ip=0)])
        _, pitch = compute_category_totals(df)
        assert pitch["ERA"] == "0.00"
        assert pitch["WHIP"] == "0.00"

    def test_mixed_roster(self):
        df = pd.DataFrame([self._make_hitter_row(), self._make_pitcher_row()])
        hit, pitch = compute_category_totals(df)
        assert "R" in hit
        assert "W" in pitch


# ── rank_priority_losing_cats ─────────────────────────────────────────────────


class TestRankPriorityLosingCats:
    def _row(self, cat, diff, above=False, tied=False):
        return {"cat": cat, "diff": diff, "above": above, "tied": tied}

    def test_returns_empty_when_no_losing_cats(self):
        rows = [self._row("HR", 5.0, above=True), self._row("AVG", 0.01, above=True)]
        result = rank_priority_losing_cats(rows, {"HR": 3.0, "AVG": 0.02})
        assert result == []

    def test_returns_at_most_top_n(self):
        rows = [
            self._row("HR", -2.0),
            self._row("R", -5.0),
            self._row("SB", -1.0),
        ]
        sigmas = {"HR": 2.0, "R": 8.0, "SB": 2.0}
        result = rank_priority_losing_cats(rows, sigmas, top_n=2)
        assert len(result) == 2

    def test_closest_to_flip_first(self):
        # HR gap=-2 with sigma=2 → z=1.0
        # R gap=-10 with sigma=8 → z=1.25
        # HR should rank first (smaller z-gap = closer to flipping)
        rows = [self._row("HR", -2.0), self._row("R", -10.0)]
        sigmas = {"HR": 2.0, "R": 8.0}
        result = rank_priority_losing_cats(rows, sigmas, top_n=2)
        assert result[0]["cat"] == "HR"
        assert result[1]["cat"] == "R"

    def test_tied_cats_excluded(self):
        rows = [self._row("SB", 0.0, tied=True), self._row("HR", -3.0)]
        result = rank_priority_losing_cats(rows, {"HR": 2.0, "SB": 1.0})
        assert all(r["cat"] != "SB" for r in result)

    def test_winning_cats_excluded(self):
        rows = [self._row("HR", 3.0, above=True), self._row("R", -5.0)]
        result = rank_priority_losing_cats(rows, {"HR": 2.0, "R": 8.0})
        assert all(r["cat"] != "HR" for r in result)

    def test_missing_sigma_sorted_last(self):
        rows = [self._row("K", -1.0), self._row("R", -2.0)]
        # Only R has a sigma; K has none → K goes last
        sigmas = {"R": 1.0}  # K missing
        result = rank_priority_losing_cats(rows, sigmas, top_n=2)
        assert result[0]["cat"] == "R"
        assert result[1]["cat"] == "K"

    def test_top_n_default_is_2(self):
        rows = [self._row(f"CAT{i}", -float(i)) for i in range(1, 6)]
        sigmas = {f"CAT{i}": float(i) for i in range(1, 6)}
        result = rank_priority_losing_cats(rows, sigmas)
        assert len(result) == 2
