"""pages/7_Player_Compare.py must use player pool (ECR + Statcast already in pool)."""

import re
from pathlib import Path


def test_no_direct_ecr_or_stats_sql():
    text = Path("pages/7_Player_Compare.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    text_no_strings = re.sub(r'"""[\s\S]*?"""', "", text_no_comments)
    bad = []
    for tbl in ("ecr_rankings", "season_stats", "statcast_archive"):
        if re.search(rf"SELECT[^\"']*FROM\s+{tbl}", text_no_strings, re.IGNORECASE | re.DOTALL):
            bad.append(tbl)
    assert bad == [], (
        f"Direct SQL on pool-enriched tables: {bad}\nUse load_player_pool() — these tables already join into the pool."
    )
