"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-19

NOTE: Real `alembic upgrade head` against a live Postgres is DEFERRED until
a PG instance exists (B2.2). This migration is verified via SQLite upgrade
and PG-dialect compilation only.
"""

from __future__ import annotations

from alembic import op
from src.db.schema import metadata

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    metadata.create_all(bind)


def downgrade() -> None:
    metadata.drop_all(op.get_bind())
