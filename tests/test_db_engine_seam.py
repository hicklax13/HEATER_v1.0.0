"""B2.0 engine-seam guards. SQLite stays the default + byte-identical; the
SQLAlchemy engine is the future Postgres swap point (selected by DATABASE_URL)."""

import sqlite3

import pytest


def test_sqlalchemy_is_available():
    import sqlalchemy  # noqa: F401
