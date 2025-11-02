"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import your models' Base for autogenerate support
from service.database import Base

# Import cache models to register them with Base.metadata
try:
    from github_client.enrichment_cache import (
        ActorProfileCache,
        CommitVerificationCache,
        RepositoryContextCache,
        WorkflowStatusCache,
    )
except ImportError:
    pass  # Enrichment cache tables not available

target_metadata = Base.metadata

# Get database URL from environment variable
database_url = os.getenv("SERVICE_DATABASE_URL")
if not database_url:
    raise ValueError(
        "SERVICE_DATABASE_URL environment variable is not set. "
        "Please set it to your PostgreSQL connection string."
    )

# Convert to asyncpg driver if using postgresql://
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # asyncpg doesn't support sslmode or channel_binding query params, use ssl=require instead
    if "sslmode=" in database_url:
        import re
        # Remove sslmode and channel_binding params, add ssl=require
        database_url = re.sub(r'[&?]sslmode=[^&]*', '', database_url)
        database_url = re.sub(r'[&?]channel_binding=[^&]*', '', database_url)
        # Add ssl=require
        separator = '&' if '?' in database_url else '?'
        database_url = database_url + separator + 'ssl=require'
elif database_url.startswith("sqlite"):
    database_url = database_url.replace("sqlite", "sqlite+aiosqlite", 1)

# Set the sqlalchemy.url in config for Alembic to use
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Helper function to run migrations with a connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (async-aware)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
