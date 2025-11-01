#!/usr/bin/env python3
"""Reset the database by deleting the SQLite file and recreating tables.

This script will:
1. Delete the existing github_events.db file
2. Recreate all tables (github_events, anomaly_summary, enrichment cache tables)
"""

import asyncio
import os
import sys

from service.database import Base, engine


async def reset_database():
    """Delete and recreate the database."""
    db_path = "github_events.db"

    # Check if database exists
    if os.path.exists(db_path):
        print(f"📦 Found existing database: {db_path}")
        file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        print(f"   Size: {file_size:.2f} MB")

        # Delete the database file
        try:
            os.remove(db_path)
            print(f"✅ Deleted {db_path}")
        except Exception as e:
            print(f"❌ Error deleting database: {e}")
            sys.exit(1)
    else:
        print(f"ℹ️  No existing database found at {db_path}")

    # Recreate tables
    print("\n🔧 Creating fresh database schema...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database schema created successfully")

        # Show created tables
        from sqlalchemy import inspect
        async with engine.connect() as conn:
            tables = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )

        print(f"\n📋 Created {len(tables)} tables:")
        for table in tables:
            print(f"   • {table}")

    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        sys.exit(1)

    print("\n✨ Database reset complete!")
    print("\n⚠️  IMPORTANT: You should also reset the RRCF anomaly detector state")
    print("   due to feature dimension changes (299 features, includes entropy)")


if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE RESET")
    print("=" * 60)
    print("\n⚠️  WARNING: This will delete ALL data!")
    print("   - All GitHub events")
    print("   - All anomaly summaries")
    print("   - All enrichment cache data")

    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Aborted")
        sys.exit(0)

    print()
    asyncio.run(reset_database())
