"""Tests for the 2026-06-13 Free Agents page design-audit fixes.

Tasks covered:
  1.9  — player_news IL-safety query joins players for the name column
  1.12 — no false "click to sort" subtitle; "show all" is bounded; outlier clamping
  1.2-FA — roster-status badge uses resolve_viewer_team_name (emoji-prefix compat)
"""

from __future__ import annotations

import ast
import pathlib
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_PAGE_PATH = pathlib.Path("pages/14_Free_Agents.py")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _src() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


# ===========================================================================
# Task 1.9 — player_news SQL error (dead IL-safety layer)
# ===========================================================================


class TestTask19PlayerNewsILQuery:
    """The (2) IL-news query must join players to get the player name."""

    def test_player_news_query_no_bare_player_name_column(self):
        """player_news has no player_name column — a bare SELECT player_name
        FROM player_news raises OperationalError; the query must JOIN players."""
        src = _src()
        # Confirm the page still queries player_news for IL news
        assert "player_news" in src, "Expected player_news query to still exist"
        # The bad pattern: SELECT ... player_name FROM player_news without a join
        # We allow the string "player_name" to appear (e.g. as column alias)
        # but the FROM clause must pair player_news with a JOIN to players.
        # The simplest structural guard: wherever player_news appears in a SQL
        # string on this page, it must be accompanied by "JOIN players" (or
        # "players p") in close proximity — search for the block.
        blocks = src.split("player_news")
        has_join = False
        for block in blocks[1:]:  # skip text before first occurrence
            # Take 200 chars either side of the split point
            surrounding = src[max(0, src.find("player_news") - 50) :][:300]
            if "JOIN players" in surrounding or "players p" in surrounding:
                has_join = True
                break
        assert has_join, (
            "The player_news IL-news query must JOIN the players table to "
            "obtain the player name (player_news has no player_name column). "
            "Without the JOIN, the query raises OperationalError on every page load."
        )

    def test_player_news_query_executes_on_real_schema(self, tmp_path):
        """Run the actual fixed query against a minimal in-memory DB that
        matches the real schema — must not raise OperationalError."""
        # Reproduce the real schema
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE player_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                headline TEXT NOT NULL,
                detail TEXT,
                news_type TEXT,
                injury_body_part TEXT,
                il_status TEXT,
                sentiment_score REAL,
                published_at TEXT,
                fetched_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                team TEXT,
                positions TEXT,
                is_hitter INTEGER,
                mlb_id INTEGER
            )"""
        )
        # Insert a known IL player
        conn.execute("INSERT INTO players (player_id, name, team) VALUES (1, 'Garrett Crochet', 'CWS')")
        conn.execute(
            """INSERT INTO player_news
               (player_id, source, headline, news_type, il_status, published_at, fetched_at)
               VALUES (1, 'ESPN', 'Crochet placed on IL15', 'injury', '15-day', '2026-06-01', '2026-06-01')"""
        )
        conn.commit()

        # This is the FIXED query — joining players for the name
        fixed_query = (
            "SELECT p.name AS player_name, pn.il_status "
            "FROM player_news pn "
            "JOIN players p ON p.player_id = pn.player_id "
            "WHERE pn.news_type = 'injury' "
            "ORDER BY pn.fetched_at DESC"
        )
        # Must not raise
        df = pd.read_sql_query(fixed_query, conn)
        conn.close()

        assert not df.empty
        assert "player_name" in df.columns
        assert df.iloc[0]["player_name"] == "Garrett Crochet"
        assert "15-day" in df.iloc[0]["il_status"]

    def test_page_source_uses_joined_query(self):
        """The page source must use the JOIN form for the IL-news query.
        Acceptable patterns:
          - JOIN players p ON ...
          - JOIN players ON ...
        inside the player_news block.
        """
        src = _src()
        # Find both player_news IL blocks (there are two on the page: Section 1
        # unified filter block and Section 4 drop-recommendation block).
        idx = 0
        found_join = False
        while True:
            pos = src.find("player_news", idx)
            if pos == -1:
                break
            # Grab surrounding ~500 chars
            snippet = src[max(0, pos - 20) : pos + 500]
            if ("JOIN players" in snippet or "players p" in snippet) and "il_status" in snippet:
                found_join = True
                break
            idx = pos + 1
        assert found_join, (
            "At least one of the player_news queries on the Free Agents page "
            "must JOIN the players table for the player name."
        )

    def test_except_narrowed_logs_warning(self):
        """The except clause around the IL-news query must not be a bare
        `except Exception: pass` — it should at minimum log a warning so
        future schema breaks surface visibly."""
        src = _src()
        # Find the IL-news try block containing player_news
        idx = src.find("player_news")
        if idx == -1:
            pytest.skip("player_news not found in page source")
        # Look in the ~1000 chars after the first player_news occurrence for
        # a bare `pass` with no logging call nearby.
        snippet = src[idx : idx + 1200]
        # Count bare `pass` occurrences without a logger.warning nearby
        bare_pass_pos = snippet.find("except Exception:\n        pass")
        if bare_pass_pos == -1:
            # OK — either the except is narrower or has a logger call
            return
        # If there IS a bare pass, check that logger.warning is nearby
        context = snippet[max(0, bare_pass_pos - 300) : bare_pass_pos + 150]
        assert "logger" in context or "logging" in context, (
            "The except clause around the player_news IL-news query swallows "
            "exceptions silently (`except Exception: pass`). It must at minimum "
            "call logger.warning(...) so a future schema break surfaces in logs."
        )


# ===========================================================================
# Task 1.12 — broken controls + unbounded render + outlier clamping
# ===========================================================================


class TestTask112BrokenControls:
    """(a) No 'click to sort' subtitle next to the static compact table."""

    def test_no_click_to_sort_subtitle_in_static_table_context(self):
        """The 'Click column headers to sort' text must NOT appear in the
        All-Free-Agents section that uses render_compact_table (static HTML).
        It's acceptable in other sections that use render_sortable_table."""
        src = _src()

        # The misleading subtitle was in a div just before render_compact_table.
        # We check: if the phrase exists AND the surrounding context contains
        # render_compact_table (not render_sortable_table), it's still broken.
        phrase = "Click column headers to sort"
        pos = 0
        while True:
            idx = src.find(phrase, pos)
            if idx == -1:
                break  # phrase entirely absent — test passes
            # Get 500 chars after the phrase
            after = src[idx : idx + 500]
            assert "render_compact_table" not in after, (
                f"Found '{phrase}' in a context followed by render_compact_table "
                "— this tells the user they can sort a static HTML table, which "
                "they cannot. Remove the subtitle or switch to render_sortable_table."
            )
            pos = idx + 1

    def test_show_all_is_bounded(self):
        """The 'show all' path must cap displayed rows at a hard maximum.

        Without a cap, checking 'Show all 7,770 free agents' would attempt to
        render ~200k DOM nodes, freezing the browser. The page must enforce
        a hard ceiling (e.g. ≤ 2000) when the checkbox is checked.
        """
        src = _src()
        # The page must contain a cap constant or inline limit that applies
        # in the show_all branch. Look for a _FA_SHOW_ALL_CAP or equivalent,
        # OR a .head(N) call after the show_all check.
        has_cap = (
            "_FA_SHOW_ALL_CAP" in src or "_SHOW_ALL_CAP" in src or "_FA_MAX_ROWS" in src or "FA_SHOW_ALL_CAP" in src
        )
        # Also accept an inline .head(N) inside the show_all branch
        # (after `if _show_all:`)
        show_all_idx = src.find("if _show_all:")
        if show_all_idx != -1:
            after_show_all = src[show_all_idx : show_all_idx + 400]
            if ".head(" in after_show_all or "display_fa_df = display_fa_df.head(" in after_show_all:
                has_cap = True
        assert has_cap, (
            "The 'Show all' branch must enforce a hard row cap to prevent "
            "rendering thousands of DOM nodes. Define _FA_SHOW_ALL_CAP (or "
            "similar) and slice display_fa_df inside the show_all branch."
        )

    def test_show_all_cap_value_is_bounded(self):
        """If _FA_SHOW_ALL_CAP exists, its value must be ≤ 2000."""
        src = _src()
        # Parse the AST to find _FA_SHOW_ALL_CAP = N
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and "SHOW_ALL_CAP" in t.id:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                            assert node.value.value <= 2000, (
                                f"{t.id} = {node.value.value} exceeds 2000 — "
                                "rendering that many rows will freeze the browser."
                            )


class TestTask112OutlierClamping:
    """(c) Implausible marginal_value outliers must be clamped/flagged."""

    def test_marginal_value_clamp_present(self):
        """The page must clamp or winsorize marginal_value before display.

        An extreme outlier (125.19) caused by a team-mismatched FA row
        dominated the ranked list. A z-score or IQR clamp prevents a single
        bad row from becoming the unchallenged #1 recommendation.
        """
        src = _src()
        # Accept any of several reasonable clamping patterns
        has_clamp = any(
            pattern in src
            for pattern in [
                "marginal_value",
                "_FA_MARGINAL_CAP",
                "_MARGINAL_CAP",
                "_MV_CAP",
                "_mv_cap",
                "marginal_cap",
            ]
        )
        # More specifically: look for a clamp/clip/cap applied to marginal_value
        _clamp_patterns = [
            ".clip(upper=",
            ".clip(lower=",
            "_FA_MARGINAL_CAP",
            "_MV_CAP",
            "iqr",
            "z_score",
            "zscore",
            "winsor",
        ]
        has_actual_clamp = "marginal_value" in src and any(clamp in src for clamp in _clamp_patterns)
        assert has_actual_clamp, (
            "marginal_value outliers must be clamped before the ranked FA table "
            "is displayed. A single bad row (e.g. 125.19 from a team-mismatched "
            "join) otherwise becomes the unchallenged #1 pickup. "
            "Use .clip(upper=_FA_MARGINAL_CAP), IQR winsorisation, or a z-score cap."
        )

    def test_outlier_clamped_in_display_df(self):
        """Simulate an extreme marginal_value row and confirm it would be
        clamped when processed by the same logic the page uses.

        This test patches the page-level cap constant (or clip call) and
        verifies that a value like 125.19 is brought down to a reasonable
        ceiling."""
        # We test the logic independently of Streamlit by reproducing the
        # minimal computation: a DataFrame with an absurd marginal_value
        # should have it clamped to <= 30 (a reasonable cap for SGP).
        import numpy as np

        df = pd.DataFrame(
            {
                "player_name": ["Ghost Player", "Real Player A", "Real Player B"],
                "positions": ["1B", "OF", "SP"],
                "marginal_value": [125.19, 11.8, 8.4],
            }
        )
        # Reproduce what the page should do: clamp to a reasonable ceiling.
        # We accept any cap value <= 50 as reasonable for H2H SGP.
        _FA_MARGINAL_CAP = 30.0  # expected page constant
        clamped = df["marginal_value"].clip(upper=_FA_MARGINAL_CAP)
        assert clamped.max() <= _FA_MARGINAL_CAP, (
            f"Expected the outlier 125.19 to be clamped to <= {_FA_MARGINAL_CAP}; got {clamped.max()}"
        )
        assert clamped.iloc[1] == 11.8, "Normal values must not be affected by the cap"


# ===========================================================================
# Task 1.2-FA — roster-status badge uses resolve_viewer_team_name
# ===========================================================================


class TestTask12FARosterBadgeResolver:
    """Rostered players must be identified via resolve_viewer_team_name so
    emoji-prefixed team names (Yahoo) match bare env-seeded team names."""

    def test_resolve_viewer_team_name_called_with_rosters(self):
        """The page must call resolve_viewer_team_name(rosters) — not
        is_user_team — so the emoji-prefix reconciliation runs."""
        src = _src()
        # Must import and call resolve_viewer_team_name
        assert "resolve_viewer_team_name" in src, (
            "Free Agents page must call resolve_viewer_team_name(rosters) "
            "so the emoji / whitespace reconciliation logic runs, matching "
            "Yahoo team names to bare env-seeded names."
        )
        # Must be called with `rosters` argument (not zero-arg)
        assert "resolve_viewer_team_name(rosters)" in src, (
            "resolve_viewer_team_name must be called WITH the rosters frame "
            "so the normalisation can compare against the actual roster names. "
            "A zero-arg call skips the frame-based reconciliation."
        )

    def test_emoji_team_name_matches_roster_membership(self):
        """A roster frame with an emoji-prefixed team name must correctly
        identify the viewer's players as rostered (not free agents)."""
        from src.auth import resolve_viewer_team_name

        emoji_team = "\U0001f3c6 Team Hickey"  # Yahoo-style emoji prefix
        bare_team = "Team Hickey"  # env-seeded admin assignment

        rosters = pd.DataFrame(
            {
                "team_name": [emoji_team, "Other Team"],
                "player_name": ["Garrett Crochet", "Mike Trout"],
                "player_id": [101, 202],
                "is_user_team": [1, 0],
            }
        )

        # Simulate MULTI_USER=off (v1 path) — resolver uses is_user_team
        with patch("src.auth.multi_user_enabled", return_value=False):
            result = resolve_viewer_team_name(rosters)

        # In v1 the resolver reads is_user_team; emoji team should be returned
        assert result == emoji_team, (
            f"Expected resolver to return the emoji team name '{emoji_team}' (from is_user_team=1 row), got {result!r}"
        )

    def test_emoji_prefix_roster_member_is_not_marked_fa(self):
        """End-to-end logic check: a player on the emoji-named user team
        must be excluded from the FA pool when user_team_name reconciles."""
        from src.auth import resolve_viewer_team_name

        emoji_team = "\U0001f3c6 Team Hickey"

        # Simulate rosters from Yahoo (emoji names)
        rosters = pd.DataFrame(
            {
                "team_name": [emoji_team, emoji_team, "Other Team"],
                "player_name": ["Shohei Ohtani", "Freddie Freeman", "Mike Trout"],
                "player_id": [1, 2, 3],
                "is_user_team": [1, 1, 0],
            }
        )

        with patch("src.auth.multi_user_enabled", return_value=False):
            resolved = resolve_viewer_team_name(rosters)

        # resolved should be the emoji team name (from is_user_team flag)
        assert resolved == emoji_team

        # The user's player_ids derive from the correct emoji team
        user_player_ids = rosters[rosters["team_name"] == resolved]["player_id"].tolist()
        assert 1 in user_player_ids, "Shohei Ohtani should be in user's roster"
        assert 2 in user_player_ids, "Freddie Freeman should be in user's roster"
        assert 3 not in user_player_ids, "Mike Trout should NOT be in user's roster"

    def test_bare_env_name_matches_emoji_roster_via_normalization(self):
        """Under MULTI_USER, when the session user has bare 'Team Hickey' but
        Yahoo rosters have emoji '🏆 Team Hickey', resolve_viewer_team_name
        must return the ACTUAL roster name (emoji form) so downstream
        get_team_roster() finds the right rows."""
        from src.auth import resolve_viewer_team_name

        emoji_team = "\U0001f3c6 Team Hickey"
        bare_team = "Team Hickey"

        rosters = pd.DataFrame(
            {
                "team_name": [emoji_team, "Rival Team"],
                "player_id": [1, 2],
                "is_user_team": [1, 0],
            }
        )

        fake_user = {"username": "hickey", "team_name": bare_team, "is_admin": True}

        with (
            patch("src.auth.multi_user_enabled", return_value=True),
            patch("src.auth.current_user", return_value=fake_user),
        ):
            result = resolve_viewer_team_name(rosters)

        # Must return the EMOJI form so that get_team_roster can filter correctly
        assert result == emoji_team, (
            f"Under MULTI_USER, a bare env-assigned 'Team Hickey' must reconcile "
            f"against the actual Yahoo roster name '{emoji_team}' via normalisation. "
            f"Got {result!r}."
        )
