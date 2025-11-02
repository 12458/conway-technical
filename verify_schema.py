import os
import asyncio
import re
from sqlalchemy import text, inspect
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

async def verify_schema() -> None:
    # Get DATABASE_URL and convert to asyncpg
    database_url = os.getenv('SERVICE_DATABASE_URL')

    # Convert postgresql:// to postgresql+asyncpg://
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if "sslmode=" in database_url:
            database_url = re.sub(r'[&?]sslmode=[^&]*', '', database_url)
            database_url = re.sub(r'[&?]channel_binding=[^&]*', '', database_url)
            separator = '&' if '?' in database_url else '?'
            database_url = database_url + separator + 'ssl=require'

    engine = create_async_engine(database_url)

    async with engine.connect() as conn:
        # Check tables
        result = await conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = result.fetchall()
        print("Tables in database:")
        for table in tables:
            print(f"  - {table[0]}")

        # Check github_events table columns
        print("\ngithub_events table columns:")
        result = await conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'github_events'
            ORDER BY ordinal_position
        """))
        for row in result.fetchall():
            print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")

        # Check anomaly_summaries table columns
        print("\nanomaly_summaries table columns:")
        result = await conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'anomaly_summaries'
            ORDER BY ordinal_position
        """))
        for row in result.fetchall():
            print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")

        # Check for enrichment cache tables
        print("\nEnrichment cache tables:")
        for table_name in ['actor_profile_cache', 'repository_context_cache', 'workflow_status_cache', 'commit_verification_cache']:
            result = await conn.execute(text(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = '{table_name}'
            """))
            exists = result.scalar()
            status = "✓ exists" if exists > 0 else "✗ missing"
            print(f"  {table_name}: {status}")

    await engine.dispose()

asyncio.run(verify_schema())
