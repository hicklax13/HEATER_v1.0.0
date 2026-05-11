"""pages/4_Trade_Analyzer.py must use player pool, not raw ECR/stats SQL."""

import re
from pathlib import Path


def test_no_direct_pool_table_sql():
    text = Path("pages/4_Trade_Analyzer.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    text_no_strings = re.sub(r'"""[\s\S]*?"""', "", text_no_comments)
    bad = []
    for tbl in ("ecr_rankings", "season_stats", "statcast_archive"):
        if re.search(rf"SELECT[^\"']*FROM\s+{tbl}", text_no_strings, re.IGNORECASE | re.DOTALL):
            bad.append(tbl)
    assert bad == [], f"Direct SQL on pool-enriched tables: {bad}"
