import os
import asyncio
import re
from sqlalchemy import text
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

async def async_main() -> None:
    # Get DATABASE_URL and convert to asyncpg
    database_url = os.getenv('SERVICE_DATABASE_URL')

    # Convert postgresql:// to postgresql+asyncpg://
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg doesn't support sslmode, use ssl=require
        if "sslmode=" in database_url:
            database_url = re.sub(r'[&?]sslmode=[^&]*', '', database_url)
            database_url = re.sub(r'[&?]channel_binding=[^&]*', '', database_url)
            separator = '&' if '?' in database_url else '?'
            database_url = database_url + separator + 'ssl=require'

    print(f"Connecting to: {database_url[:50]}...")

    engine = create_async_engine(database_url, echo=True)
    async with engine.connect() as conn:
        result = await conn.execute(text("select 'hello world'"))
        print("Query result:", result.fetchall())
    await engine.dispose()
    print("Connection successful!")

asyncio.run(async_main())
