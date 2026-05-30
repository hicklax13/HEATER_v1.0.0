"""Scripts and app.py must not open SQLite directly — use get_connection.

A raw sqlite3 connection skips get_connection()'s WAL + busy_timeout=60000 +
synchronous=NORMAL pragmas, so a bare connect during the bootstrap phase can
hit lock contention without the busy_timeout cushion (SF-4's root cause).
"""

import ast
import re
from pathlib import Path

SCRIPTS_TO_CHECK = [
    "scripts/draft_vs_current.py",
    "scripts/extract_trade_data.py",
    "scripts/optimal_roster_sim.py",
]


def test_no_direct_sqlite_connect_in_scripts():
    """Bypassing get_connection() drops WAL+busy_timeout protections.
    SF-4's root cause was concurrent writers without busy_timeout."""
    bad = []
    for s in SCRIPTS_TO_CHECK:
        p = Path(s)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # Strip comments to avoid false positives
        text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        # Strip strings
        text_no_strings = re.sub(r'"[^"]*"', "", text_no_comments)
        text_no_strings = re.sub(r"'[^']*'", "", text_no_strings)
        if re.search(r"\bsqlite3\.connect\s*\(", text_no_strings):
            bad.append(s)
    assert bad == [], (
        f"These scripts still call sqlite3.connect directly (bypasses WAL+busy_timeout): {bad}\n"
        f"Use `from src.database import get_connection; conn = get_connection()` instead."
    )


def test_no_direct_sqlite_connect_in_app():
    """app.py must reach SQLite via get_connection(), never a raw sqlite3
    connection. Detected via AST so an aliased ``import sqlite3 as _sql``
    can't slip past a name-based regex."""
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3" or alias.name.startswith("sqlite3."):
                    suffix = f" as {alias.asname}" if alias.asname else ""
                    offenders.append(f"line {node.lineno}: import {alias.name}{suffix}")
        elif isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
            offenders.append(f"line {node.lineno}: from sqlite3 import ...")
    assert offenders == [], (
        "app.py imports sqlite3 directly (bypasses get_connection's "
        "WAL+busy_timeout):\n  " + "\n  ".join(offenders) + "\n"
        "Use `from src.database import get_connection` instead."
    )
