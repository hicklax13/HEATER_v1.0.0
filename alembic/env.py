from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from src.db.engine import engine_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema is the single source for autogenerate/metadata; imported here so
# `alembic` commands see it. (Created in B2.1 Task 2.)
try:
    from src.db.schema import metadata as target_metadata
except Exception:  # during Task 1, schema.py may not exist yet
    target_metadata = None


def _url() -> str:
    return engine_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
