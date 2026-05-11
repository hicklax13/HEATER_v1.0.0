"""pages/1_My_Team.py must consume the enriched player pool, not raw SQL."""

import re
from pathlib import Path


def test_no_direct_season_stats_sql():
    """Direct SELECT from season_stats bypasses pool enrichments (Statcast, regression flags)."""
    text = Path("pages/1_My_Team.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    text_no_strings = re.sub(r'"""[\s\S]*?"""', "", text_no_comments)
    # Match SELECT ... FROM season_stats but allow it inside read_sql wrappers that
    # are part of canonical pool builders (we don't have any in pages/, but be safe)
    bad = re.findall(
        r'SELECT[^"\']*FROM\s+season_stats',
        text_no_strings,
        re.IGNORECASE | re.DOTALL,
    )
    assert bad == [], (
        f"Direct season_stats SQL in pages/1_My_Team.py: {bad}\nUse load_player_pool() / _build_player_pool() instead."
    )


def test_uses_enriched_player_pool():
    """The page should consume load_player_pool() so Statcast/regression-flag
    enrichments and bats/throws flow through to the user's roster view."""
    text = Path("pages/1_My_Team.py").read_text(encoding="utf-8")
    assert "load_player_pool" in text, (
        "pages/1_My_Team.py must import + call load_player_pool() to receive "
        "the unified pool's Statcast columns and regression flags."
    )


def test_historical_seasons_route_through_loader():
    """Historical season fetches should go through load_season_stats(), the
    canonical helper in src.database — not raw read_sql_query against season_stats."""
    text = Path("pages/1_My_Team.py").read_text(encoding="utf-8")
    assert "load_season_stats" in text, (
        "pages/1_My_Team.py must use load_season_stats() (in src.database) for "
        "historical-year fetches; the player pool only carries 2026 ytd_* columns."
    )
